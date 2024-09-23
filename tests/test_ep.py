import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from expertpara.ep_fwd import moe_infer_ep
import random
import socket
import time

"""
NOTE

To run tests:

python3 -m tests.test_ep

If the test shows message like:

[W socket.cpp:601] [c10d] The client socket has failed to connect to [localhost]:33683 (errno: 99 - Cannot assign requested address).

But the test is passing, then let other rank except rank 0 to sleep for a while before initializing the process group. 
This is because rank 0 needs to initialize the process group first before other ranks can join. This is a workaround for the issue.
"""

@torch.no_grad()
def moe_infer_single_node(experts, num_experts_per_tok, x, flat_expert_indices, flat_expert_weights):
    """
    Single worker inference, taken from original dit-moe code.
    """
    expert_cache = torch.zeros_like(x) 
    idxs = flat_expert_indices.argsort()
    tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
    token_idxs = idxs // num_experts_per_tok 
    for i, end_idx in enumerate(tokens_per_expert):
        start_idx = 0 if i == 0 else tokens_per_expert[i-1]
        if start_idx == end_idx:
            continue
        expert = experts[i]
        exp_token_idx = token_idxs[start_idx:end_idx]
        expert_tokens = x[exp_token_idx]
        expert_out = expert(expert_tokens)
        expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]]) 
        
        # for fp16 and other dtype
        expert_cache = expert_cache.to(expert_out.dtype)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning) # scatter_reduce_ is in beta, ignore the warning
            expert_cache.scatter_reduce_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out, reduce='sum')
    return expert_cache

def _find_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        port = s.getsockname()[1]
    return port

N_TOKENS = 128
N_ITER = 4
HIDDEN_SIZE = 36
WORLD_SIZE = 4
NUM_TOTAL_EXPERTS = 8
NUM_EXPERTS_PER_TOK = 2


def run_test(rank, world_size, num_total_experts, num_experts_per_tok, n_tokens, n_iter, hidden_size, port, seed, async_op):
    try:
        # Initialize the process group
        if rank != 0:
            time.sleep(.5)  # Wait for the rank 0 to initialize the process group
        dist.init_process_group(backend='gloo', init_method=f'tcp://localhost:{port}/',
                                world_size=world_size, rank=rank)
        # Set the device to CPU
        device = torch.device('cpu')
        # Set the random seed for reproducibility
        torch.manual_seed(seed) # all proc shall use the same seed for expert initialization
        experts = []
        for _ in range(num_total_experts):
            linear = nn.Linear(hidden_size, hidden_size)
            nn.init.uniform_(linear.weight, a=0.0, b=1.0)  # Initialize weights with random values
            experts.append(linear)
        
        torch.manual_seed(seed + rank) # each proc shall use a different seed for input
        
        # experts = [nn.Identity() for _ in range(num_total_experts)]
        
        # Run the _infer_grouped function
        test_passed = True
        prev_out = []
        for i in range(n_iter):
            inp = torch.randn(n_tokens, hidden_size).to(device)
            flat_expert_indices = torch.randint(0, num_total_experts, (n_tokens * num_experts_per_tok,)).to(device)
            flat_expert_weights = torch.rand(n_tokens * num_experts_per_tok, 1).to(device)
            expert_out = moe_infer_ep(
                inp=inp,
                experts=experts,
                flat_expert_indices=flat_expert_indices,
                flat_expert_weights=flat_expert_weights,
                num_experts_per_tok=num_experts_per_tok,
                async_op=async_op,
                cache_key="test_cache_key",
            )
            
            single_node_out = moe_infer_single_node(
                experts=experts,
                num_experts_per_tok=num_experts_per_tok,
                x=inp,
                flat_expert_indices=flat_expert_indices,
                flat_expert_weights=flat_expert_weights,
            )
            
            """
            If async all2all, the output shall be the same as the previous two steps.
            
            """
            if async_op:
                if len(prev_out) == 0:
                    ans = single_node_out
                elif len(prev_out) == 1:
                    ans = prev_out[0]
                else:
                    ans = prev_out[-2]
            else:
                ans = single_node_out
            prev_out.append(single_node_out)
            # Verify the output
            if not torch.allclose(expert_out, ans, atol=1e-6):
                test_passed = False
                print(f"Rank {rank}: Test failed at iter {i}, relative error: {torch.norm(expert_out - single_node_out) / torch.norm(single_node_out)}")
            
        if test_passed:
            print(f"Rank {rank}: Test passed.")
    finally:
        # Clean up
        dist.destroy_process_group()

def test_infer(repeats=3):
    seed = 42
    for i in range(repeats):
        print(f"\nRunning test iteration {i+1} with seed {seed + i}...")
        print("Running sync test...")
        mp.spawn(
            run_test,
            args=(WORLD_SIZE, NUM_TOTAL_EXPERTS, NUM_EXPERTS_PER_TOK, N_TOKENS, N_ITER, HIDDEN_SIZE, _find_port(), seed + i, False),
            nprocs=WORLD_SIZE,
            join=True
        )
        print(f"Running async test...")
        mp.spawn(
            run_test,
            args=(WORLD_SIZE, NUM_TOTAL_EXPERTS, NUM_EXPERTS_PER_TOK, N_TOKENS, N_ITER, HIDDEN_SIZE, _find_port(), seed + i, True),
            nprocs=WORLD_SIZE,
            join=True
        )

if __name__ == '__main__':
    test_infer(repeats=3)
