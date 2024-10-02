import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

class FlashSelfMHAModifiedSP(nn.Module):
    """
    self-attention with flashattention
    """
    def __init__(self,
                 dim,
                 num_heads,
                 qkv_bias=True,
                 qk_norm=False,
                 attn_drop=0.0,
                 proj_drop=0.0,
                 device=None,
                 dtype=None,
                 norm_layer=nn.LayerNorm,
                 layer_idx=None,
                 async_op=None,
                 ):
        from flash_attn.modules.mha import FlashCrossAttention
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        assert self.dim % num_heads == 0, "self.kdim must be divisible by num_heads"
        self.head_dim = self.dim // num_heads
        assert self.head_dim % 8 == 0 and self.head_dim <= 128, "Only support head_dim <= 128 and divisible by 8"

        self.Wqkv = nn.Linear(dim, 3 * dim, bias=qkv_bias, **factory_kwargs)
        # TODO: eps should be 1 / 65530 if using fp16
        self.q_norm = norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.inner_attn = FlashCrossAttention(attention_dropout=attn_drop)
        self.out_proj = nn.Linear(dim, dim, bias=qkv_bias, **factory_kwargs)
        self.proj_drop = nn.Dropout(proj_drop)
        self.cache_key = layer_idx
        self.async_op = async_op
        assert async_op is not None, "async_op must be specified"
    
    def forward(self, x,):
        """
        Parameters
        ----------
        x: torch.Tensor
            (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim)
        """
        b, s, d = x.shape

        qkv = self.Wqkv(x)
        qkv = qkv.view(b, s, 3, self.num_heads, self.head_dim)  # [b, s, 3, h, d]
        q, k, v = qkv.unbind(dim=2) # [b, s, h, d]
        q = self.q_norm(q).half()   # [b, s, h, d]
        k = self.k_norm(k).half()
        
        k = k.contiguous()
        v = v.contiguous()
        assert q.shape == (b, s, self.num_heads, self.head_dim), f"q shape mismatch: {q.shape}"
        assert k.shape == (b, s, self.num_heads, self.head_dim), f"k shape mismatch: {k.shape}"
        assert v.shape == (b, s, self.num_heads, self.head_dim), f"v shape mismatch: {v.shape}"
        world_size = dist.get_world_size()
        """
        NOTE:
        Sequence Parallelism: All-gather K and V from all processes.
        """
        if self.async_op:
            from .df import sp_all_gather_async
            k_all, v_all = sp_all_gather_async(k, v, key=self.cache_key, concat_dim=1)
        else:
            k_all = sp_all_gather(k, concat_dim=1)
            v_all = sp_all_gather(v, concat_dim=1)
        assert k_all.shape == (b, s*world_size, self.num_heads, self.head_dim), f"k_all shape mismatch: {k_all.shape}!=({b}, {s*world_size}, {self.num_heads}, {self.head_dim})"
        assert v_all.shape == (b, s*world_size, self.num_heads, self.head_dim), f"v_all shape mismatch: {v_all.shape}!=({b}, {s*world_size}, {self.num_heads}, {self.head_dim})"

        kv = torch.stack([k_all, v_all], dim=2)     # [b, s, 2, h, d]
        context = self.inner_attn(q,kv)
        out = self.out_proj(context.view(b, s, d))
        out = self.proj_drop(out)

        return out

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
            layer_idx = None,
            async_op = None,
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
        self.async_op = async_op
        self.cache_key = layer_idx
        

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
        
        k = k.contiguous()
        v = v.contiguous()

        # Verify shapes of q, k, v
        assert q.shape == (B, self.num_heads, N_local, self.head_dim), f"q shape mismatch: {q.shape}"
        assert k.shape == (B, self.num_heads, N_local, self.head_dim), f"k shape mismatch: {k.shape}"
        assert v.shape == (B, self.num_heads, N_local, self.head_dim), f"v shape mismatch: {v.shape}"

        world_size = dist.get_world_size()
        """
        NOTE:
        Sequence Parallelism: All-gather K and V from all processes.
        """
        if self.async_op:
            from .df import sp_all_gather_async
            k_all, v_all = sp_all_gather_async(k, v, key=self.cache_key, concat_dim=2)
        else:
            k_all = sp_all_gather(k, concat_dim=2)
            v_all = sp_all_gather(v, concat_dim=2)

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

def sp_all_gather(x_local, concat_dim, async_op=False):
    """
    AllGather tokens from all ranks after processing local slices.
    This is used after local attention processing in sequence parallelism.
    """
    world_size = dist.get_world_size()
    x_local = x_local.contiguous()  # Ensure the tensor is contiguous
    

    # Prepare a list to gather the local tensors from all ranks
    x_gather_list = [torch.zeros_like(x_local).contiguous() for _ in range(world_size)]

    # Perform all_gather to gather from all ranks
    handle = dist.all_gather(tensor_list=x_gather_list, tensor=x_local, async_op=async_op)
    
    if async_op:
        assert concat_dim is None, "Cannot concatenate tensors in async mode."
        return x_gather_list, handle
    
    if concat_dim is None:
        return x_gather_list

    # Concatenate gathered slices to form the full sequence
    x_full = torch.cat(x_gather_list, dim=concat_dim)  # Shape: [N, T, D]
    
    expected_shape = list(x_local.shape)
    expected_shape[concat_dim] *= world_size
    expected_shape = tuple(expected_shape)
    
    assert x_full.shape == expected_shape, f"Output shape mismatch after all-gathering tokens: {x_full.shape}"
    return x_full
