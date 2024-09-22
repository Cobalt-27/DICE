import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.distributed as dist

# ref: https://github.com/laekov/fastmoe

@torch.no_grad()
# parallel version of SparseMoeBlock.moe_infer()
def moe_infer_ep(inp: torch.Tensor, experts: nn.ModuleList, flat_expert_indices, flat_expert_weights,num_total_experts, num_experts_per_tok):
    """
    Perform inference using a mixture of experts (MoE) model in an expert parallel (EP) setting.
    Args:
        inp (Tensor): Input tensor to be processed by the experts, 
            [#input tokens, h].
        experts (nn.ModuleList): List of expert modules, with length equal to the total number of experts.
            Only index = global_rank + i * world_size is valid for each worker, other indices are None.
        flat_expert_indices (Tensor): Flattened tensor containing indices of experts assigned to each token, 
            [#input tokens * num_experts_per_tok].
        flat_expert_weights (Tensor): Flattened tensor containing weights of experts assigned to each token,
            [#input tokens * num_experts_per_tok].
        num_total_experts (int): Total number of experts in the model.
        num_experts_per_tok (int): Number of experts assigned to each token.
    Returns:
        Tensor: Output tensor after processing by the experts.
    """
    
    """
    NOTE: each worker has num_local_experts, with index = global_rank + i * world_size
    """
    world_size = dist.get_world_size()
    global_rank = dist.get_rank()
    num_local_experts = num_total_experts // world_size
    assert num_total_experts % world_size == 0, "Number of experts must be divisible by world size"
    
    # NOTE: Tensor of size [num_total_experts], counts of tokens routed to each expert on the local worker.
    token_counts_local = flat_expert_indices.bincount()
    # NOTE: refer to exchange_token_counts for details, counts of tokens routed to this worker's local experts for all workers.
    # used for dispatch recv and combine send
    token_counts_global = exchange_token_counts(token_counts_local)
    assert token_counts_local.size(0) == num_total_experts and token_counts_global.size() == (world_size,num_local_experts)
    
    # Local Dispatch
    """
    NOTE: grouped_idx_dup
    mapping from duplicated grouped index  (tokens destined for same expert are contiguous) to original duplicated index
    [#input tokens * num_experts_per_tok] -> [#input tokens * num_experts_per_tok]
    Here DUPLICATED indicates that one input token is replicated num_experts_per_tok times
    """
    grouped_idx_dup = flat_expert_indices.argsort()
    """
    NOTE: grouped_idx
    mapping from duplicated grouped index  (tokens destined for same expert are contiguous) to original index
    [#input tokens * num_experts_per_tok] -> [#input tokens]
    Here the mapped result represents real input token index
    """
    grouped_idx = grouped_idx_dup // num_experts_per_tok 
    # mapping: [#input tokens, h] -> [#input tokens * num_experts_per_tok, h]
    grouped_dup_inp = _local_dispatch(inp=inp, pos=grouped_idx) 
    
    # NOTE: Global Dispatch
    # mapping: [#input tokens * num_experts_per_tok, h] -> [#combined tokens * num_experts_per_tok, h]
    
    mlp_inp = _global_dispatch(grouped_dup_inp, token_counts_local, token_counts_global)
    
    # MLP
    mlp_outp = proc_experts(inp=mlp_inp, experts=experts, token_counts_global=token_counts_global, num_local_experts=num_local_experts)
    
    # NOTE: Global Combine
    # mapping: [#combined tokens * num_experts_per_tok, h] -> [#input tokens * num_experts_per_tok, h]
    
    grouped_dup_outp = _global_combine(mlp_outp, token_counts_local, token_counts_global)
    
    # Local combine
    # mapping: [#input tokens * num_experts_per_tok, h] -> [#input tokens * num_experts_per_tok, h]
    outp_dup = _local_combine(inp=grouped_dup_outp, pos=grouped_idx_dup, out_size=flat_expert_indices.size(0))
    outp_dup.mul_(flat_expert_weights)
    # [#input tokens * num_experts_per_tok, h] -> [#input tokens, h]
    expert_out = outp_dup.view(inp.size(0), num_experts_per_tok, inp.size(1)).sum(dim=1)
    assert expert_out.shape == inp.shape, f"{expert_out.shape} != {inp.shape}"
    return expert_out

@torch.no_grad()
def proc_experts(inp, experts, token_counts_global, num_local_experts):
    world_size = dist.get_world_size()
    global_rank = dist.get_rank()
    mlp_token_counts = token_counts_global.sum(dim=0)
    outp = torch.empty_like(inp)
    start = 0
    for i in range(num_local_experts):
        e = global_rank + i * world_size
        outp[start:start+mlp_token_counts[i]] = experts[e](inp[start:start+mlp_token_counts[i]])
        start += mlp_token_counts[i]
    assert start == inp.size(0)
    return outp
    

@torch.no_grad()
def exchange_token_counts(token_counts_local):
    """
    combines expert token counts from all workers and computes the global expert token counts.

    Args:
        expert_local_tokens_count (torch.Tensor): Tensor of size [num_total_experts], counts of tokens
            routed to each expert on the local worker.

    Returns:
        expert_global_tokens_count (torch.Tensor): Tensor of size [world_size, num_local_experts], counts
            of tokens routed to CURRENT worker's local experts from ALL workers.
            Shape: [world_size, num_local_experts]
    """
    world_size = dist.get_world_size()
    global_rank = dist.get_rank()
    num_total_experts = token_counts_local.size(0)
    num_local_experts = num_total_experts // world_size
    assert num_total_experts % world_size == 0, "Number of experts must be divisible by world size"

    # gather token counts from all workers
    token_counts_local_list = [torch.zeros_like(token_counts_local) for _ in range(world_size)]
    dist.all_gather(token_counts_local_list, token_counts_local)

    # Stack the counts into a tensor of shape [world_size, num_total_experts]
    # Each row corresponds to a worker's expert_local_tokens_count
    token_counts_all_workers = torch.stack(token_counts_local_list, dim=0)  # Shape: [world_size, num_total_experts]

    token_counts_global = token_counts_all_workers[:,global_rank::world_size]
    assert token_counts_global.shape == (world_size,num_local_experts)
    return token_counts_global

@torch.no_grad()
def _global_dispatch(grouped_dup_inp, token_counts_local, token_counts_global):
    """
    dispatch the grouped input tensor to the experts on all workers.
    Args:
        grouped_dup_inp (Tensor): Grouped input tensor to be processed by the experts, 
            [#input tokens * num_experts_per_tok, h].
        token_counts_local (Tensor): Tensor of size [num_total_experts], counts of tokens
            routed to each expert on the local worker.
        token_counts_global (Tensor): Tensor of size [world_size, num_local_experts], counts
            of tokens routed to CURRENT worker's local experts from ALL workers.
    Returns:
        Tensor: all2all output
    """
    world_size = dist.get_world_size()
    global_rank = dist.get_rank()
    num_total_experts = token_counts_local.size(0)
    num_local_experts = num_total_experts // world_size
    assert num_total_experts % world_size == 0, "Number of experts must be divisible by world size"
    assert token_counts_global.size() == (world_size,num_local_experts) and token_counts_local.size(0) == num_total_experts
    assert grouped_dup_inp.size(0) == token_counts_local.sum()
    
    send_size = token_counts_local.view(num_local_experts, world_size).sum(dim=1)
    recv_size = token_counts_global.sum(dim=0)
    
    buf = grouped_dup_inp.new_empty((token_counts_global.sum(), grouped_dup_inp.size(1)))
    
    send_start = 0
    recv_start = 0
    for i in range(num_local_experts):
        dist.all_to_all_single(
            output=buf[recv_start:recv_start+recv_size[i]],
            input=grouped_dup_inp[send_start:send_start+send_size[i]],
            output_split_sizes=token_counts_global[:,i].flatten().tolist(),
            input_split_sizes=token_counts_local[i*world_size:(i+1)*world_size].flatten().tolist(),
            async_op=False,
        )
        recv_start += recv_size[i]
        send_start += send_size[i]
    assert send_start == grouped_dup_inp.size(0) and recv_start == buf.size(0)
    return buf

@torch.no_grad()
def _global_combine(grouped_dup_outp, token_counts_local, token_counts_global):
    """
    combine the grouped output tensor from all workers.
    Args:
        grouped_dup_outp (Tensor): Grouped output tensor after processing by the experts, 
            [#input tokens * num_experts_per_tok, h].
        token_counts_local (Tensor): Tensor of size [num_total_experts], counts of tokens
            routed to each expert on the local worker.
        token_counts_global (Tensor): Tensor of size [world_size, num_local_experts], counts
            of tokens routed to CURRENT worker's local experts from ALL workers.
    Returns:
        Tensor: all2all output
    """
    world_size = dist.get_world_size()
    global_rank = dist.get_rank()
    num_total_experts = token_counts_local.size(0)
    num_local_experts = num_total_experts // world_size
    assert num_total_experts % world_size == 0, "Number of experts must be divisible by world size"
    assert token_counts_global.size() == (world_size,num_local_experts) and token_counts_local.size(0) == num_total_experts
    assert grouped_dup_outp.size(0) == token_counts_global.sum()
    
    send_size = token_counts_global.sum(dim=0)
    recv_size = token_counts_local.view(num_local_experts, world_size).sum(dim=1)
    
    buf = grouped_dup_outp.new_empty((token_counts_local.sum(), grouped_dup_outp.size(1)))
    
    send_start = 0
    recv_start = 0
    for i in range(num_local_experts):
        dist.all_to_all_single(
            output=buf[recv_start:recv_start+recv_size[i]],
            input=grouped_dup_outp[send_start:send_start+send_size[i]],
            output_split_sizes=token_counts_local[i*world_size:(i+1)*world_size].flatten().tolist(),
            input_split_sizes=token_counts_global[:,i].flatten().tolist(),
            async_op=False,
        )
        recv_start += recv_size[i]
        send_start += send_size[i]
    assert send_start == grouped_dup_outp.size(0) and recv_start == buf.size(0)
    return buf

@torch.no_grad()
def _local_dispatch(inp, pos):
    buf = torch.index_select(inp, 0, pos)
    return buf

@torch.no_grad()
def _local_combine(inp, pos, out_size):
    buf = torch.zeros(out_size, inp.shape[-1],
            dtype=inp.dtype, device=inp.device)
    buf.index_copy_(0, pos, inp) # no overlap in forward
    return buf