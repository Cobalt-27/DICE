import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.distributed as dist
from .prof import CudaProfiler
"""
DiEP: Diffusion model with async Expert Parallelism

- Adjacent diffusion steps have similar activations.
- Expert parallelism: dispatch and combine tokens in one step (sync).
- DiEP: async expert parallelism, all2all communication starts at one step and ends at next step.

Basically async all2all + cache 
"""


"""
Cache structure:

DiEP enables async all2all (token dispatch and combine).

In step T, forward func calls async dispatch_async:

- recv the tokens from previous step (get this from cache)
- starts a new async all2all, to be received in next step,
    this async all2all shall be stored in this cache.

To store a all2all operation, we need to store the following information:
- a list of async op handles, since each dispatch/combine needs multiple dist.all_to_all
- recv_buf
- token_counts_local
- token_counts_global


"""
_diep_cache_dispatch = {}
_diep_cache_combine = {}

def cache_clear():
    _diep_cache_dispatch.clear()
    _diep_cache_combine.clear()

def _cache_put(cache, key, val):
    # must hold the send buf to prevent it from being released
    cache[key] = val

def _cache_get(cache, key):
    return cache[key]


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
    if not cache_key in _diep_cache_dispatch:
        # to be warm up
        buf, _ = global_dispatch(
            grouped_dup_inp=grouped_dup_inp,
            token_counts_local=token_counts_local,
            token_counts_global=token_counts_global,
            async_op=False,
        )
        _cache_put(_diep_cache_dispatch, cache_key, (None, buf, token_counts_local, token_counts_global, grouped_idx_dup, flat_expert_weights, None))
        return buf, token_counts_local, token_counts_global, grouped_idx_dup, flat_expert_weights
    # reads from cache, wait for the handle, get the results for current step, then start a new async all2all
    prev_handles, prev_buf, prev_token_counts_local, prev_token_counts_global, prev_grouped_idx_dup, prev_flat_expert_weights, prev_send_buf = _cache_get(_diep_cache_dispatch, cache_key)
    CudaProfiler.prof().start('dispatch_wait')
    _wait(prev_handles)
    CudaProfiler.prof().stop('dispatch_wait')
    # async all2all, to be received in next step
    buf, handles = global_dispatch(
        grouped_dup_inp=grouped_dup_inp,
        token_counts_local=token_counts_local,
        token_counts_global=token_counts_global,
        async_op=True,
    )
    _cache_put(_diep_cache_dispatch, cache_key, (handles, buf, token_counts_local, token_counts_global, grouped_idx_dup, flat_expert_weights, grouped_dup_inp))
    return prev_buf, prev_token_counts_local, prev_token_counts_global, prev_grouped_idx_dup, prev_flat_expert_weights

@torch.no_grad()
def global_combine_async(grouped_dup_outp, token_counts_local, token_counts_global, grouped_idx_dup, flat_expert_weights, cache_key):
    """
    Combine tokens from experts, async all2all
    
    NOTE: have to cache grouped_idx_dup, since it is needed in the following local_combine in this step.
    """
    
    from .ep_fwd import global_combine
    assert cache_key is not None
    if not cache_key in _diep_cache_combine:
        # to be warm up
        buf, _ = global_combine(
            grouped_dup_outp=grouped_dup_outp,
            token_counts_local=token_counts_local,
            token_counts_global=token_counts_global,
            async_op=False,
        )
        _cache_put(_diep_cache_combine, cache_key, (None, buf, grouped_idx_dup, flat_expert_weights, None))
        return buf, grouped_idx_dup, flat_expert_weights
    # reads from cache, wait for the handle, get the results for current step, then start a new async all2all
    prev_handles, prev_buf, prev_grouped_idx_dup, prev_flat_expert_weights, prev_send_buf = _cache_get(_diep_cache_combine, cache_key)
    CudaProfiler.prof().start('combine_wait')
    _wait(prev_handles)
    CudaProfiler.prof().stop('combine_wait')
    # async all2all, to be received in next step
    buf, handles = global_combine(
        grouped_dup_outp=grouped_dup_outp,
        token_counts_local=token_counts_local,
        token_counts_global=token_counts_global,
        async_op=True,
    )
    _cache_put(_diep_cache_combine, cache_key, (handles, buf, grouped_idx_dup, flat_expert_weights, grouped_dup_outp)) # no need to store token counts
    return prev_buf, prev_grouped_idx_dup, prev_flat_expert_weights
