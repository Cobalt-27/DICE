import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import random
import socket
import time
import torch.nn.functional as F
import pytest
from seqpara.sp_fwd import AttentionSP


class AttnTest(nn.Module):
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

def find_free_port():
    """Finds a free port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

def run_attention_test(rank, world_size, port, seed):
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
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Define sequence length, embedding dimension, and batch size
    N = 2  # Batch size
    T_total = 16  # Total sequence length (must be divisible by world_size)
    D = 64  # Embedding dimension
    num_heads = 8
    from seqpara.sp_fwd import AttentionSP
    attn_sp = AttentionSP(dim=D, num_heads=num_heads).to(device)

    # Ensure T_total is divisible by world_size
    assert T_total % world_size == 0, "Sequence length must be divisible by world size for SP."

    # Create input tensor on rank 0 and scatter to all ranks
    x = torch.randn(N, T_total, D, device=device)
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
        attn = AttnTest(dim=D, num_heads=num_heads, attn=attn_sp).to(device)

        output = attn(x)  # Shape: [N, T_total, D]
        # Compare outputs
        assert torch.allclose(output_sp, output, atol=1e-6), f"Outputs do not match, relative error: {((output_sp - output).norm() / output.norm()):.2e}"
    # Cleanup
    dist.barrier()
    dist.destroy_process_group()

@pytest.mark.parametrize('seed', [42, 43])
def test_attention_sp(seed):
    """Main test function to be called by pytest."""
    world_size = 2  # Number of processes
    port = find_free_port()
    mp.set_start_method('spawn', force=True)
    mp.spawn(
        run_attention_test,
        args=(world_size, port, seed),
        nprocs=world_size,
        join=True
    )
