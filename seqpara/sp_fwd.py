import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

class AttentionSP(nn.Module):

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        from timm.models.vision_transformer import use_fused_attn
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # no norm for q,k in the original implementation
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor of shape [B, N_local, C], where N_local is the local sequence length.
        """
        
        original_shape = x.shape
        B, N_local, C = x.shape  # N_local is the local sequence length

        # Compute local Q, K, V
        qkv = self.qkv(x)  # Shape: [B, N_local, 3 * num_heads * head_dim]
        qkv = qkv.reshape(B, N_local, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # Shape: [3, B, num_heads, N_local, head_dim]
        q, k, v = qkv.unbind(0)  # Each has shape: [B, num_heads, N_local, head_dim]
        q, k = self.q_norm(q), self.k_norm(k)
        
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        # Verify shapes of q, k, v
        assert q.shape == (B, self.num_heads, N_local, self.head_dim), f"q shape mismatch: {q.shape}"
        assert k.shape == (B, self.num_heads, N_local, self.head_dim), f"k shape mismatch: {k.shape}"
        assert v.shape == (B, self.num_heads, N_local, self.head_dim), f"v shape mismatch: {v.shape}"

        # All-gather K and V from all processes
        world_size = dist.get_world_size()

        # Prepare empty tensors to gather K and V
        k_list = [x.new_empty(k.shape).contiguous() for _ in range(world_size)]
        v_list = [x.new_empty(v.shape).contiguous() for _ in range(world_size)]

        # All-gather K and V tensors
        dist.all_gather(k_list, k)
        dist.all_gather(v_list, v)

        # Concatenate K and V along the sequence dimension
        k_all = torch.cat(k_list, dim=2)  # Shape: [B, num_heads, N_total, head_dim]
        v_all = torch.cat(v_list, dim=2)

        # Calculate total sequence length
        N_total = N_local * world_size

        # Verify shapes of k_all and v_all
        assert k_all.shape == (B, self.num_heads, N_total, self.head_dim), f"k_all shape mismatch: {k_all.shape}"
        assert v_all.shape == (B, self.num_heads, N_total, self.head_dim), f"v_all shape mismatch: {v_all.shape}"

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k_all, v_all,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k_all.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v_all

        # Reshape and project
        x = x.transpose(1, 2).reshape(B, N_local, C)
        # Verify shape after transpose and reshape
        assert x.shape == (B, N_local, C), f"x shape mismatch after transpose and reshape: {x.shape}"
        x = self.proj(x)
        x = self.proj_drop(x)
        # Verify final output shape matches the original input shape
        assert x.shape == original_shape, f"Final output shape mismatch: {x.shape}"
        return x

def sp_broadcast(x):
    """
    Broadcast the input tensor to all ranks.
    """
    x=x.contiguous()
    # Broadcast the tensor
    dist.broadcast(tensor=x, src=0)

    return x

def sp_scatter(x):
    """
    Dispatch tokens to different ranks using dist.scatter for Sequence Parallelism.
    Each rank will get a slice of the sequence tokens.
    """
    x=x.contiguous()
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    if x.dim() == 2:
        x = x.unsqueeze(1)
    N, T, D = x.shape
    T_local = T // world_size
    if rank == 0:
        assert T % world_size == 0, "The sequence length must be divisible by the world size."
        
        # Split x into chunks for each rank
        x_chunks = list(x.chunk(world_size, dim=1))  # List of [N, T_local, D]
        x_chunks = [x_chunk.contiguous() for x_chunk in x_chunks]
    else:
        x_chunks = None  # Placeholder for other ranks

    # Create tensor for local rank to receive its chunk
    x_local = x.new_empty((N, T_local, D)).contiguous()
    
    # Scatter the tokens
    dist.scatter(tensor=x_local, scatter_list=x_chunks, src=0)

    # Assert that the local chunk has the correct shape
    assert x_local.shape == (N, T_local, D), "Input shape mismatch after scattering tokens, {x_local.shape}!=({N}, {T_local}, {D})"
    return x_local

def sp_allgather(x_local):
    """
    AllGather tokens from all ranks after processing local slices.
    This is used after local attention processing in sequence parallelism.
    """
    N, T_local, D = x_local.shape
    world_size = dist.get_world_size()
    T = T_local * world_size
    x_local = x_local.contiguous()  # Ensure the tensor is contiguous
    

    # Prepare a list to gather the local tensors from all ranks
    x_gather_list = [torch.zeros_like(x_local).contiguous() for _ in range(world_size)]

    # Perform all_gather to gather from all ranks
    dist.all_gather(tensor_list=x_gather_list, tensor=x_local)

    # Concatenate gathered slices to form the full sequence
    x_full = torch.cat(x_gather_list, dim=1)  # Shape: [N, T, D]
    
    assert x_full.shape == (N, T, D), f"Output shape mismatch after all-gathering tokens: {x_full.shape}"
    return x_full
