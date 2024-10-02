import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.distributed as dist
from cudaprof.prof import CudaProfiler
from .ep_cache import All2AllCache
"""
DiEP: Diffusion model with async Expert Parallelism

- Adjacent diffusion steps have similar activations.
- Expert parallelism: dispatch and combine tokens in one step (sync).
- DiEP: async expert parallelism, all2all communication starts at one step and ends at next step.

Basically async all2all + cache 
"""

"""
format:
(handles, buf, token_counts_local, token_counts_global, grouped_idx_dup, flat_expert_weights, grouped_dup_inp)
"""
_diep_cache_dispatch = None

"""
format:
(handles, buf, grouped_idx_dup, flat_expert_weights, grouped_dup_outp)
"""
_diep_cache_combine = None

_CACHE_DISPATCH_VAL_LEN = 7
_CACHE_COMBINE_VAL_LEN = 5

def ep_cached_tensors_size():
    """
    Measures the size of all the tensors in the cache in bytes.
    """
    return _diep_cache_dispatch.tensors_size() + _diep_cache_combine.tensors_size()


def ep_cache_init(cache_capacity, auto_gc=False, offload=False, prefetch_size=None, offload_mask=None):
    if not offload:
        assert prefetch_size == None
    global _diep_cache_dispatch, _diep_cache_combine
    _diep_cache_dispatch = All2AllCache(
        capacity=cache_capacity,
        auto_gc=auto_gc,
        prefetch_size=prefetch_size,
        offload=offload,
        val_len=_CACHE_DISPATCH_VAL_LEN,
        offload_mask=offload_mask,
    )
    _diep_cache_combine = All2AllCache(
        capacity=cache_capacity,
        auto_gc=auto_gc,
        prefetch_size=prefetch_size,
        offload=offload,
        val_len=_CACHE_COMBINE_VAL_LEN,
        offload_mask=offload_mask,
    )


def ep_cache_clear():
    _diep_cache_dispatch.clear()
    _diep_cache_combine.clear()

@torch.no_grad()
def _wait(handles):
    if handles is not None:
        for handle in handles:
            handle.wait()

@torch.no_grad()
def global_dispatch_async(grouped_dup_inp, token_counts_local, token_counts_global, grouped_idx_dup, flat_expert_weights, cache_key):
    """
    Dispatch tokens to experts, async all2all.
    
    Also returns token counts since they are needed in the following all2all combine in this step.
    """
    from .ep_fwd import global_dispatch
    assert cache_key is not None
    if not _diep_cache_dispatch.contains(cache_key):
        # to be warm up
        buf, _ = global_dispatch(
            grouped_dup_inp=grouped_dup_inp,
            token_counts_local=token_counts_local,
            token_counts_global=token_counts_global,
            async_op=False,
        )
        _diep_cache_dispatch.put(cache_key, (None, buf, token_counts_local, token_counts_global, grouped_idx_dup, flat_expert_weights, None))
        return buf, token_counts_local, token_counts_global, grouped_idx_dup, flat_expert_weights
    # reads from cache, wait for the handle, get the results for current step, then start a new async all2all
    (
        prev_handles, 
        prev_buf, 
        prev_token_counts_local,
        prev_token_counts_global,
        prev_grouped_idx_dup,
        prev_flat_expert_weights,
        prev_send_buf
        ) = _diep_cache_dispatch.get(cache_key)
    CudaProfiler.prof().start('global_dispatch.wait')
    _wait(prev_handles)
    CudaProfiler.prof().stop('global_dispatch.wait')
    # async all2all, to be received in next step
    buf, handles = global_dispatch(
        grouped_dup_inp=grouped_dup_inp,
        token_counts_local=token_counts_local,
        token_counts_global=token_counts_global,
        async_op=True,
    )
    _diep_cache_dispatch.put(cache_key, (handles, buf, token_counts_local, token_counts_global, grouped_idx_dup, flat_expert_weights, grouped_dup_inp))
    return prev_buf, prev_token_counts_local, prev_token_counts_global, prev_grouped_idx_dup, prev_flat_expert_weights

@torch.no_grad()
def global_combine_async(grouped_dup_outp, token_counts_local, token_counts_global, grouped_idx_dup, flat_expert_weights, cache_key):
    """
    Combine tokens from experts, async all2all
    
    NOTE: have to cache grouped_idx_dup, since it is needed in the following local_combine in this step.
    """
    
    from .ep_fwd import global_combine
    assert cache_key is not None
    if not _diep_cache_combine.contains(cache_key):
        # to be warm up
        buf, _ = global_combine(
            grouped_dup_outp=grouped_dup_outp,
            token_counts_local=token_counts_local,
            token_counts_global=token_counts_global,
            async_op=False,
        )
        _diep_cache_combine.put(cache_key, (None, buf, grouped_idx_dup, flat_expert_weights, None))
        return buf, grouped_idx_dup, flat_expert_weights
    # reads from cache, wait for the handle, get the results for current step, then start a new async all2all
    (
        prev_handles,
        prev_buf,
        prev_grouped_idx_dup,
        prev_flat_expert_weights,
        prev_send_buf
        ) = _diep_cache_combine.get(cache_key)
    CudaProfiler.prof().start('global_combine.wait')
    _wait(prev_handles)
    CudaProfiler.prof().stop('global_combine.wait')
    # async all2all, to be received in next step
    buf, handles = global_combine(
        grouped_dup_outp=grouped_dup_outp,
        token_counts_local=token_counts_local,
        token_counts_global=token_counts_global,
        async_op=True,
    )
    _diep_cache_combine.put(cache_key, (handles, buf, grouped_idx_dup, flat_expert_weights, grouped_dup_outp)) # no need to store token counts
    return prev_buf, prev_grouped_idx_dup, prev_flat_expert_weights
