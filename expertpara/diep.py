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
_cache_capacity = None # number of entries in the cache, should equals to the layer count

_cache_auto_gc = False # whether to automatically garbage collect the cache

def cache_size():
    """
    Measures the size of all the tensors in the cache in bytes.
    """
    return sum(
        sum(
            [sum([x.element_size()*x.numel() for x in v if isinstance(x, torch.Tensor)]) for v in cache if v is not None]
            ) for cache in [_diep_cache_dispatch, _diep_cache_combine]
        )
    
def cache_init(cache_capacity, auto_gc = False):
    global _diep_cache_dispatch, _diep_cache_combine, _cache_capacity, _cache_auto_gc
    _cache_capacity = cache_capacity
    _cache_auto_gc = auto_gc
    _diep_cache_dispatch = [None for _ in range(cache_capacity)]
    _diep_cache_combine = [None for _ in range(cache_capacity)]

def cache_clear():
    assert _cache_capacity is not None
    global _diep_cache_dispatch, _diep_cache_combine
    _diep_cache_dispatch = [None for _ in range(_cache_capacity)]
    _diep_cache_combine = [None for _ in range(_cache_capacity)]

def _cache_put(cache, key:int, val):
    assert _cache_capacity is not None and cache is not None
    # must hold the send buf to prevent it from being released
    assert isinstance(key, int), "Key must be an integer"
    
    if _cache_auto_gc: # garbage collect before putting
        _cache_gc(cache)
    
    cache[key] = val

def _cache_get(cache, key:int):
    assert _cache_capacity is not None and cache is not None
    assert isinstance(key, int), "Key must be an integer"
    return cache[key]

def _cache_contain(cache, key:int):
    assert _cache_capacity is not None and cache is not None
    assert isinstance(key, int), "Key must be an integer"
    return cache[key] is not None

def _cache_gc(cache):
    """
    Cache entries are async all2all handles and their bufs(send/recv).
    If the all2all has completed, we can release the send buf, recv buf is needed when cache is read.
    """
    assert len(cache) == _cache_capacity and cache is not None
    for i in range(_cache_capacity):
        if cache[i] is not None:
            item = list(cache[i])
            prev_handles = item[0]
            if prev_handles is not None:
                can_clear = True
                for handle in prev_handles:
                    if not handle.is_completed():
                        can_clear = False
                        break
                if can_clear:# only remove the handle and send buf
                    item[0] = None # NOTE: make sure the first element is the handles
                    item[-1] = None # NOTE: make sure the last element is the send buf
                    cache[i] = tuple(item)

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
    if not _cache_contain(_diep_cache_dispatch, cache_key):
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
    if not _cache_contain(_diep_cache_combine, cache_key):
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
