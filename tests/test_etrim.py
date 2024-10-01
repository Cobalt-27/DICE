import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from expertpara.etrim import trim_module_list, trim_state_dict, DummyExpert, _needed_expert_indices
from .utils import find_free_port, set_seed
import time
import pytest

def run_trim_test(rank, world_size, num_total_experts, port, seed):
    # Initialize the process group
    if rank != 0:
        time.sleep(.5)
    dist.init_process_group("gloo", rank=rank, world_size=world_size, init_method=f'tcp://localhost:{port}/')
    set_seed(seed)

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

    dist.destroy_process_group()
    

@pytest.mark.parametrize("num_total_experts,world_size,seed", [(8, 4, 42), (4, 2, 43), (16, 4, 44)])
def test_trimming(num_total_experts, world_size, seed):
    mp.set_start_method('spawn', force=True) # must set, otherwise mp.Queue fails
    mp.spawn(run_trim_test, args=(world_size, num_total_experts, find_free_port(), seed), nprocs=world_size, join=True)