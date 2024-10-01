import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import random
import time
import torch.nn.functional as F
import pytest
from seqpara.sp_fwd import AttentionSP
from .utils import find_free_port, set_seed
from seqpara.sp_fwd import FlashSelfMHAModifiedSP, AttentionSP

class AttnSim(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            attn: AttentionSP,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        from timm.models.vision_transformer import use_fused_attn
        self.fused_attn = use_fused_attn()
        
        self.qkv = attn.qkv
        self.q_norm = attn.q_norm
        self.k_norm = attn.k_norm
        self.attn_drop = attn.attn_drop
        self.proj = attn.proj
        self.proj_drop = attn.proj_drop

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class FlashAttnSim(nn.Module):
    """
    self-attention with flashattention
    """
    def __init__(self,
                 dim,
                 num_heads,
                 attn: FlashSelfMHAModifiedSP,
                 attn_drop = 0.0,
                 ):
        from flash_attn.modules.mha import FlashSelfAttention 
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        assert self.dim % num_heads == 0, "self.kdim must be divisible by num_heads"
        self.head_dim = self.dim // num_heads
        assert self.head_dim % 8 == 0 and self.head_dim <= 128, "Only support head_dim <= 128 and divisible by 8"
        self.Wqkv = attn.Wqkv
        self.q_norm = attn.q_norm
        self.k_norm = attn.k_norm
        self.inner_attn = FlashSelfAttention(attention_dropout=attn_drop)
        self.out_proj = attn.out_proj
        self.proj_drop = attn.proj_drop

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

        qkv = torch.stack([q, k, v], dim=2)     # [b, s, 3, h, d]
        context = self.inner_attn(qkv)
        out = self.out_proj(context.view(b, s, d))
        out = self.proj_drop(out)

        return out


def run_attention_test(rank, world_size, port, seed, flash_attn=False):
    """Runs the attention test on a given process."""
    # Initialize the process group
    dist.init_process_group(
        backend='nccl',  # Use 'gloo' if 'nccl' is not available
        init_method=f'tcp://127.0.0.1:{port}',
        rank=rank,
        world_size=world_size
    )
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)

    # Set the random seed for reproducibility
    set_seed(seed)
    # Define sequence length, embedding dimension, and batch size
    N = 8  # Batch size
    T_total = 16  # Total sequence length (must be divisible by world_size)
    D = 64  # Embedding dimension
    num_heads = 8
    # Create input tensor on rank 0 and scatter to all ranks
    x = torch.randn(N, T_total, D, device=device)
    if flash_attn:
        x=x.half()
        attn_sp = FlashSelfMHAModifiedSP(dim=D, num_heads=num_heads, async_op=False).to(device).half()
    else:
        attn_sp = AttentionSP(dim=D, num_heads=num_heads, async_op=False).to(device)

    # Ensure T_total is divisible by world_size
    assert T_total % world_size == 0, "Sequence length must be divisible by world size for SP."

    
    # Scatter the input tensor across processes
    from seqpara.sp_fwd import sp_scatter, sp_all_gather
    
    original_x = x.clone()
    x_temp = sp_scatter(x)
    x = sp_all_gather(x_temp,concat_dim=1)
    if rank == 0:
        assert torch.equal(x, original_x), f"Scatter and gather did not work as expected, relative error: {((x - original_x).norm() / original_x.norm()):.2e}"
    x_local = sp_scatter(x)
    assert torch.equal(sp_all_gather(x_local,concat_dim=1), x), "Scatter and gather did not work as expected."
    

    # Run AttentionSP on local slice
    output_sp_local = attn_sp(x_local)  # Shape: [N, T_local, D]

    output_sp = sp_all_gather(output_sp_local,concat_dim=1)  # Shape: [N, T_total, D]

    # Run original Attention on the full input (only on rank 0)
    if rank == 0:
        # Create the original Attention module
        if flash_attn:
            attn = FlashAttnSim(dim=D, num_heads=num_heads, attn=attn_sp).to(device).half()
        else:
            attn = AttnSim(dim=D, num_heads=num_heads, attn=attn_sp).to(device)

        output = attn(x)  # Shape: [N, T_total, D]
        # Compare outputs
        assert torch.allclose(output_sp, output, atol=1e-6), f"Outputs do not match, relative error: {((output_sp - output).norm() / output.norm()):.2e}"
    # Cleanup
    dist.barrier()
    dist.destroy_process_group()

@pytest.mark.parametrize('seed,flash_attn', [(42, False), (42, True), (43, False), (43, True)])
def test_attention_sp_sync(seed, flash_attn):
    """Main test function to be called by pytest."""
    world_size = 2  # Number of processes
    port = find_free_port()
    mp.set_start_method('spawn', force=True)
    mp.spawn(
        run_attention_test,
        args=(world_size, port, seed, flash_attn),
        nprocs=world_size,
        join=True
    )
