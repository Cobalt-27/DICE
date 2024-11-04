import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.distributed as dist
from .diep import global_combine_async,global_dispatch_async, get_result_to_skip, put_result_for_skip, ep_skip_this_step, ep_skip_enabled
from cudaprof.prof import CudaProfiler
# ref: https://github.com/laekov/fastmoe

_use_latest_expert_weights = None

def use_latest_expert_weights(val):
    assert val in [True, False], "val must be a boolean"
    global _use_latest_expert_weights
    _use_latest_expert_weights = val

@torch.no_grad()
# parallel version of SparseMoeBlock.moe_infer()
def moe_infer_ep(inp: torch.Tensor, experts: nn.ModuleList, flat_expert_indices, flat_expert_weights, num_experts_per_tok, async_op, cache_key=None):
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
        async_op (bool): Whether to use async all2all for dispatch and combine.
        cache_key (int): Cache key to store and retrieve intermediate results.
        skip_if_possible (bool): Whether to skip the current step if possible.
    Returns:
        Tensor: Output tensor after processing by the experts.
    """
    """
    NOTE: each worker has num_local_experts, with index = global_rank + i * world_size
    """
    world_size = dist.get_world_size()
    global_rank = dist.get_rank()
    num_total_experts = len(experts)
    num_local_experts = num_total_experts // world_size
    assert num_total_experts % world_size == 0, "Number of experts must be divisible by world size"
    
    """
    Prepare to skip(drop) commu and comp of unimportant tokens
    Drop tokens with lower router scores
    """
    if async_op and ep_skip_this_step(cache_key):
        assert ep_skip_enabled(), "skip_if_possible is True but skip is not enabled"
        assert num_experts_per_tok % 2 == 0, "num_experts_per_tok must be even"
        original_num_experts_per_tok = num_experts_per_tok
        num_experts_per_tok //= 2
        flat_expert_indices = flat_expert_indices[::2]
        # not dropping the weights, since they are only used in scaling the output
        assert flat_expert_indices.size(0) == inp.size(0) * num_experts_per_tok
    else:
        original_num_experts_per_tok = num_experts_per_tok
        
    
    # NOTE: Tensor of size [num_total_experts], counts of tokens routed to each expert on the local worker.
    token_counts_local = flat_expert_indices.bincount()
    
    # XXX: XL model have extreme cases, some experts receivs no input
    if token_counts_local.size(0) != num_total_experts:
        expanded_counts = torch.zeros(num_total_experts, dtype=token_counts_local.dtype, device=token_counts_local.device)
        expanded_counts[:token_counts_local.size(0)] = token_counts_local
        token_counts_local = expanded_counts
    
    if async_op:
        with CudaProfiler.scope('moe.wait'):
            from .diep import ep_wait
            """
            NOTE: wait for the previous async all2all to finish
            Call wait here to precisely measure the time, otherwise handles are waited in the following all_gather
            """
            ep_wait()
    
    # NOTE: refer to exchange_token_counts for details, counts of tokens routed to this worker's local experts for all workers.
    with CudaProfiler.scope('exchange_token_counts'):
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
    with CudaProfiler.scope('local_dispatch'):
        # mapping: [#input tokens, h] -> [#input tokens * num_experts_per_tok, h]
        grouped_dup_inp = _local_dispatch(inp=inp, pos=grouped_idx) 
        grouped_idx = None # no longer needed
    
    """
    NOTE
    Have to update meta data like grouped_idx_dup
    
    meta data to be stored in cache:
    - token counts, local & global
    - grouped_idx_dup, since async all2all receives tokens in a different order
    - grouped_idx
    - flat_expert_weights, using WRONG WEIGHTS will lead to CORRUPTED RESULTS
    
    """
    latest_flat_expert_weights = flat_expert_weights.clone()
    # NOTE: Global Dispatch
    # mapping: [#input tokens * num_experts_per_tok, h] -> [#combined tokens * num_experts_per_tok, h]
    with CudaProfiler.scope('global_dispatch'):
        if not async_op:
            mlp_inp, _ = global_dispatch(grouped_dup_inp, token_counts_local, token_counts_global)
        else:
            (
                mlp_inp, 
                token_counts_local, 
                token_counts_global, 
                grouped_idx_dup, 
                flat_expert_weights
            ) = global_dispatch_async(grouped_dup_inp=grouped_dup_inp,
                                        token_counts_local=token_counts_local,
                                        token_counts_global=token_counts_global,
                                        grouped_idx_dup=grouped_idx_dup,
                                        flat_expert_weights=flat_expert_weights,
                                        cache_key=cache_key)
            # grouped_idx = grouped_idx_dup // num_experts_per_tok # update grouped_idx in case it's used
    # MLP
    with CudaProfiler.scope('moe.mlp'):
        mlp_outp = proc_experts(inp=mlp_inp, experts=experts, token_counts_global=token_counts_global, num_local_experts=num_local_experts)
    # NOTE: Global Combine
    # mapping: [#combined tokens * num_experts_per_tok, h] -> [#input tokens * num_experts_per_tok, h]
    with CudaProfiler.scope('global_combine'):
        if not async_op:
            grouped_dup_outp, _ = global_combine(mlp_outp, token_counts_local, token_counts_global)
        else:
            (
                grouped_dup_outp, 
                grouped_idx_dup, 
                flat_expert_weights
            ) = global_combine_async(grouped_dup_outp=mlp_outp,
                                        token_counts_local=token_counts_local,
                                        token_counts_global=token_counts_global,
                                        grouped_idx_dup=grouped_idx_dup,
                                        flat_expert_weights=flat_expert_weights,
                                        cache_key=cache_key)
            # grouped_idx = grouped_idx_dup // num_experts_per_tok # update grouped_idx in case it's used
    # Local combine
    with CudaProfiler.scope('local_combine'):
        # mapping: [#input tokens * num_experts_per_tok, h] -> [#input tokens * num_experts_per_tok, h]
        outp_dup = _local_combine(inp=grouped_dup_outp, pos=grouped_idx_dup, out_size=grouped_dup_outp.size(0))
    
    if async_op and ep_skip_enabled():
        # cache the unimportant tokens, as they can be used in the next step
        if outp_dup.size(0) != inp.size(0) * original_num_experts_per_tok:
            assert outp_dup.size(0) == inp.size(0) * original_num_experts_per_tok // 2
            # unimporant tokens are dropped
            dropped = get_result_to_skip(cache_key) # use the cached result for all dropped positions
            final_outp = outp_dup.new_empty((outp_dup.size(0) * 2, outp_dup.size(1)))
            final_outp[::2] = outp_dup
            final_outp[1::2] = dropped
        else:
            put_result_for_skip(cache_key, outp_dup[1::2]) # cache the dropped positions
            final_outp = outp_dup
    else:
        final_outp = outp_dup
        
    if _use_latest_expert_weights:
        final_outp.mul_(latest_flat_expert_weights) # flat_expert_weights is updated in global_combine_async
    else:
        final_outp.mul_(flat_expert_weights) # flat_expert_weights is NOT updated in async all2all
    # [#input tokens * num_experts_per_tok, h] -> [#input tokens, h]
    expert_out = final_outp.view(inp.size(0), original_num_experts_per_tok, inp.size(1)).sum(dim=1)
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
    with CudaProfiler.scope('exchange_token_counts.all_gather'):
        dist.all_gather(tensor_list=token_counts_local_list, tensor=token_counts_local)

    # Stack the counts into a tensor of shape [world_size, num_total_experts]
    # Each row corresponds to a worker's expert_local_tokens_count
    token_counts_all_workers = torch.stack(token_counts_local_list, dim=0)  # Shape: [world_size, num_total_experts]

    token_counts_global = token_counts_all_workers[:,global_rank::world_size]
    assert token_counts_global.shape == (world_size,num_local_experts)
    return token_counts_global

@torch.no_grad()
def global_dispatch(grouped_dup_inp, token_counts_local, token_counts_global, async_op=False):
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
    handle_list = []
    for i in range(num_local_experts):
        with CudaProfiler.scope('global_dispatch.call_dist'):
            handle=dist.all_to_all_single(
                output=buf[recv_start:recv_start+recv_size[i]],
                input=grouped_dup_inp[send_start:send_start+send_size[i]],
                output_split_sizes=token_counts_global[:,i].flatten().tolist(),
                input_split_sizes=token_counts_local[i*world_size:(i+1)*world_size].flatten().tolist(),
                async_op=True,
            )
        with CudaProfiler.scope('moe.wait'):
            if not async_op:
                handle.wait()
        handle_list.append(handle)
        recv_start += recv_size[i]
        send_start += send_size[i]
    assert send_start == grouped_dup_inp.size(0) and recv_start == buf.size(0)
    return buf, handle_list

@torch.no_grad()
def global_combine(grouped_dup_outp, token_counts_local, token_counts_global, async_op=False):
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
        list: list of handles for async all2all
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
    handle_list = []
    for i in range(num_local_experts):
        with CudaProfiler.scope('global_combine.call_dist'):
            handle=dist.all_to_all_single(
                output=buf[recv_start:recv_start+recv_size[i]],
                input=grouped_dup_outp[send_start:send_start+send_size[i]],
                output_split_sizes=token_counts_local[i*world_size:(i+1)*world_size].flatten().tolist(),
                input_split_sizes=token_counts_global[:,i].flatten().tolist(),
                async_op=True,
            )
        with CudaProfiler.scope('moe.wait'):
            if not async_op:
                handle.wait()
        handle_list.append(handle)
        recv_start += recv_size[i]
        send_start += send_size[i]
    assert send_start == grouped_dup_outp.size(0) and recv_start == buf.size(0)
    return buf, handle_list

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