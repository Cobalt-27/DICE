import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from expertpara.etrim import trim_module_list, trim_state_dict, DummyExpert, _needed_expert_indices
from .test_ep import find_port
import time

def run_trim_test(rank, world_size, num_total_experts, port):
    # Initialize the process group
    if rank != 0:
        time.sleep(.5)
    dist.init_process_group("gloo", rank=rank, world_size=world_size, init_method=f'tcp://localhost:{port}/')
    torch.manual_seed(42+rank)  # Ensure reproducibility

    # Create a SparseMoeBlock and replace experts
    original_module_list = nn.ModuleList([nn.Linear(16, 16) for _ in range(num_total_experts)])
    trimmed_module_list = trim_module_list(original_module_list, num_total_experts)

    # Check if experts are correctly replaced
    needed_experts = _needed_expert_indices(num_total_experts)
    assert len(trimmed_module_list) == num_total_experts, "Length mismatch after trimming"

    for idx, expert in enumerate(trimmed_module_list):
        if idx in needed_experts:
            assert not isinstance(expert, DummyExpert), f"Expert {idx} should be retained but replaced with DummyExpert"
        else:
            assert isinstance(expert, DummyExpert), f"Expert {idx} should be DummyExpert but wasn't replaced"

    # Simulate loading a state dict for trimming
    state_dict = {
        f"blocks.0.moe.experts.{i}.weight": torch.rand(16, 16)
        for i in range(num_total_experts)
    }
    trimmed_state_dict = trim_state_dict(state_dict, num_total_experts)

    # Check if the state dict contains only needed experts
    for key in trimmed_state_dict.keys():
        expert_idx = int(key.split('.')[4])  # Extract the expert index from the key
        assert expert_idx in needed_experts, f"Expert {expert_idx} is in state_dict but shouldn't be"

    print(f"Rank {rank}: Trimming test passed.")

    dist.destroy_process_group()


def test_expert_trimming(num_total_experts, world_size):
    mp.spawn(run_trim_test, args=(world_size, num_total_experts, find_port()), nprocs=world_size, join=True)


if __name__ == "__main__":
    print("Running expert trimming tests...")
    print("Test 1: Trim 8 experts with 4 workers")
    test_expert_trimming(num_total_experts=8, world_size=4)
    print("Test 2: Trim 4 experts with 2 workers")
    test_expert_trimming(num_total_experts=4, world_size=2)
