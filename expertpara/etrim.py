import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.distributed as dist

"""
During expert parallelism, only a subset of experts are needed in each worker.

Trim (filter) the unused experts from the expert list.
"""

class DummyExpert(nn.Module):
    def __init__(self):
        super().__init__()
        # No parameters or layers

    def forward(self, x):
        raise RuntimeError("DummyExpert should not be called")

_etrim_enable = True # enabled by default

def _needed_expert_indices(total_experts):
    return [ i for i in range(dist.get_rank(), total_experts, dist.get_world_size()) ]

def trim_module_list(module_list: nn.ModuleList, total_experts):
    """
    Replace unused experts with a dummy expert.
    """
    assert _etrim_enable, "Expert trimming is disabled"
    trimmed = []
    needed= _needed_expert_indices(total_experts)
    for i, module in enumerate(module_list):
        if i in needed:
            trimmed.append(module)
        else:
            trimmed.append(DummyExpert())
    return nn.ModuleList(trimmed)
    
def trim_state_dict(state_dict, total_experts):
    """
    Remove the parameters of unused experts from the state_dict.
    """
    # key format: blocks.11.moe.experts.7.gate_proj.weight
    assert _etrim_enable, "Expert trimming is disabled"
    needed = _needed_expert_indices(total_experts)
    trimmed_state_dict = {}
    kept = 0
    removed = 0
    for key, value in state_dict.items():
        # Check if the key corresponds to an expert parameter
        if '.experts.' in key:
            # The key format is 'blocks.{block_idx}.moe.experts.{expert_idx}.<layer_name>.weight'
            
            parts = key.split('.')
            # Find the index of 'experts' in the key
            experts_idx = parts.index('experts') # no need to try&catch, if it fails, it should fail loudly
            expert_idx = int(parts[experts_idx + 1])
            if expert_idx not in needed:
                # Skip the parameters of unused experts
                removed += 1
                continue
            else:
                kept += 1
        trimmed_state_dict[key] = value
    assert removed == kept * (dist.get_world_size() - 1), "Removed experts should be equal to kept experts times (world_size - 1)"
    return trimmed_state_dict