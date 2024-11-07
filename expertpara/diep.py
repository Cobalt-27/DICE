import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.distributed as dist
from cudaprof.prof import CudaProfiler
from .ep_cache import All2AllCache, EPSkipCache
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

"""
stores all2all combine results, for each layer, used to skip all2all and mlps in some steps
"""
_diep_cache_skip = None

_CACHE_DISPATCH_VAL_LEN = 8
_CACHE_COMBINE_VAL_LEN = 5

max_mem = -1
def ep_cached_tensors_size():
    """
    Measures the size of all the tensors in the cache in bytes.
    """
    return ep_all2all_cached_tensors_size() + ep_skip_cached_tensors_size()
                    
def ep_all2all_cached_tensors_size():
    """
    Measures the size of all the tensors in the cache in bytes.
    """
    return _diep_cache_dispatch.tensors_size() + _diep_cache_combine.tensors_size()
    
def ep_skip_cached_tensors_size():
    """
    Measures the size of all the tensors in the cache in bytes.
    """
    return _diep_cache_skip.tensors_size()

def ep_set_step(step):
    global _step
    _step = step

def ep_skip_now(idx):
    """
    Decide whether to skip some tokens in this step.
    """
    if not _enable_skip or not _diep_cache_skip.contains(idx):
        # disabled, or no cache due to first step
        return False
    return _step % _noskip_interval != 0

_rand_mask = None

def ep_skip_mask(len):
    mask = torch.zeros(len, dtype=torch.bool)
    if _skip_mode == 'high':
        mask[::2] = True # skip the first token of each pair, the higher score one
        return mask
    elif _skip_mode == 'low':
        mask[1::2] = True # skip the second token of each pair
        return mask
    elif _skip_mode == 'rand':
        global _rand_mask
        if _rand_mask is None:
            n = len//2
            bool_tensor = torch.zeros((n, 2), dtype=torch.bool)
            random_indices = torch.randint(0, 2, (n,))
            bool_tensor[torch.arange(n), random_indices] = True
            _rand_mask = bool_tensor.view(-1)
        return _rand_mask
    else:
        raise ValueError(f"Unknown skip mode: {_skip_mode}")

def ep_wait():
    _diep_cache_dispatch.wait()
    _diep_cache_combine.wait()

_enable_skip = False
def ep_skip_enabled():
    return _enable_skip

_forced_sync = False
def diep_force_sync():
    global _forced_sync
    assert not _forced_sync
    _forced_sync = True

def diep_cancel_sync():
    global _forced_sync
    assert _forced_sync
    _forced_sync = False
   

def ep_cache_init(cache_capacity, auto_gc=False, noskip_step = 1, skip_mode = None, async_pipeline=False, selective_async_strategy=None):
    
    assert selective_async_strategy in [None, 'shallow', 'deep', 'interleaved', 'all']
    global _selective_async_strategy
    _selective_async_strategy = selective_async_strategy
    
    global _async_pipeline
    _async_pipeline = async_pipeline
    
    global _noskip_interval
    """
    perform commu for part of the tokens every _noskip_interval steps
    """
    _noskip_interval = noskip_step
    global _enable_skip
    _enable_skip = noskip_step != 1
    """
    skip mode
    high: skip tokens with high router scores
    low: skip tokens with low router scores
    random: skip tokens randomly
    None: no skip
    """
    if _enable_skip:
        assert skip_mode is not None
    global _skip_mode
    _skip_mode = skip_mode
    
    global _diep_cache_dispatch, _diep_cache_combine
    _diep_cache_dispatch = All2AllCache(
        capacity=cache_capacity,
        auto_gc=auto_gc,
        val_len=_CACHE_DISPATCH_VAL_LEN,
    )
    _diep_cache_combine = All2AllCache(
        capacity=cache_capacity,
        auto_gc=auto_gc,
        val_len=_CACHE_COMBINE_VAL_LEN,
    )
    
    global _diep_cache_skip
    _diep_cache_skip = EPSkipCache(
        capacity=cache_capacity if _enable_skip else 0,
    )

def ep_cache_clear():
    _diep_cache_dispatch.clear()
    _diep_cache_combine.clear()
    _diep_cache_skip.clear()

def ep_skip_get(key):
    """
    Get the previous result to skip the current step.
    """
    return _diep_cache_skip.get(key)

def ep_skip_put(key, tensor):
    """
    Put the result for the next step to skip the current step.
    """
    assert isinstance(tensor, torch.Tensor)
    _diep_cache_skip.put(key, tensor)

@torch.no_grad()
def _wait(handles):
    if handles is not None:
        for handle in handles:
            handle.wait()

def _past_layer_key(key):
    total_layers = _diep_cache_dispatch.capacity
    if _selective_async_strategy == 'all':
        return (key - 1) % total_layers
    if _selective_async_strategy == 'shallow':
        return (key - 1) % (total_layers // 2)
    if _selective_async_strategy == 'deep':
        return (key - 1) % (total_layers // 2) + total_layers // 2
    if _selective_async_strategy == 'interleaved':
        return (key - 2) % total_layers
    raise ValueError(f"Unknown selective async strategy: {_selective_async_strategy}")

@torch.no_grad()
def global_dispatch_async(grouped_dup_inp, token_counts_local, token_counts_global, grouped_idx_dup, flat_expert_weights, experts, cache_key):
    """
    Dispatch tokens to experts, async all2all.
    
    Also returns token counts since they are needed in the following all2all combine in this step.
    """
    from .ep_fwd import global_dispatch
    assert cache_key is not None
    # if not _diep_cache_dispatch.contains(cache_key):
    if _forced_sync:
        # to be warm up
        # if dist.get_rank() == 0:
        #     print("no caches")
        buf, _ = global_dispatch(
            grouped_dup_inp=grouped_dup_inp,
            token_counts_local=token_counts_local,
            token_counts_global=token_counts_global,
            async_op=False,
        )
        assert not _diep_cache_dispatch.contains(cache_key)
        _diep_cache_dispatch.put(cache_key, (None, buf, token_counts_local, token_counts_global, grouped_idx_dup, flat_expert_weights, experts, None))
        return buf, token_counts_local, token_counts_global, grouped_idx_dup, flat_expert_weights, experts
    
    if _async_pipeline:
        past_layer_cache_key = _past_layer_key(cache_key)
        assert _diep_cache_dispatch.contains(past_layer_cache_key)
        key_to_get = past_layer_cache_key
    else:
        key_to_get = cache_key
    # reads from cache, wait for the handle, get the results for current step, then start a new async all2all
    (
        prev_handles, 
        prev_buf, 
        prev_token_counts_local,
        prev_token_counts_global,
        prev_grouped_idx_dup,
        prev_flat_expert_weights,
        prev_experts,
        prev_send_buf
        ) = _diep_cache_dispatch.get(key_to_get)
    # if dist.get_rank() == 0:
    #     print(is_vc)
        
    # CudaProfiler.prof().start('global_dispatch.wait')
    _wait(prev_handles)
    # CudaProfiler.prof().stop('global_dispatch.wait')
    # async all2all, to be received in next step
    buf, handles = global_dispatch(
        grouped_dup_inp=grouped_dup_inp,
        token_counts_local=token_counts_local,
        token_counts_global=token_counts_global,
        async_op=True,
    )
    _diep_cache_dispatch.put(cache_key, (handles, buf, token_counts_local, token_counts_global, grouped_idx_dup, flat_expert_weights, experts, grouped_dup_inp))
    
    mem_use = ep_cached_tensors_size()
    global max_mem
    if mem_use > max_mem:
        max_mem = mem_use

    return prev_buf, prev_token_counts_local, prev_token_counts_global, prev_grouped_idx_dup, prev_flat_expert_weights, prev_experts

@torch.no_grad()
def global_combine_async(grouped_dup_outp, token_counts_local, token_counts_global, grouped_idx_dup, flat_expert_weights, cache_key):
    """
    Combine tokens from experts, async all2all
    
    NOTE: have to cache grouped_idx_dup, since it is needed in the following local_combine in this step.
    """
    
    from .ep_fwd import global_combine
    assert cache_key is not None
    # if not _diep_cache_combine.contains(cache_key):
    if _forced_sync:
        # to be warm up
        buf, _ = global_combine(
            grouped_dup_outp=grouped_dup_outp,
            token_counts_local=token_counts_local,
            token_counts_global=token_counts_global,
            async_op=False,
        )
        assert not _diep_cache_combine.contains(cache_key)
        _diep_cache_combine.put(cache_key, (None, buf, grouped_idx_dup, flat_expert_weights, None))
        return buf, grouped_idx_dup, flat_expert_weights

    if _async_pipeline:
        past_layer_cache_key = _past_layer_key(cache_key)
        assert _diep_cache_combine.contains(cache_key)
        key_to_put = past_layer_cache_key
    else:
        key_to_put = cache_key
    # reads from cache, wait for the handle, get the results for current step, then start a new async all2all
    (
        prev_handles,
        prev_buf,
        prev_grouped_idx_dup,
        prev_flat_expert_weights,
        prev_send_buf
        ) = _diep_cache_combine.get(cache_key)
    # CudaProfiler.prof().start('global_combine.wait')
    _wait(prev_handles)
    # CudaProfiler.prof().stop('global_combine.wait')
    # async all2all, to be received in next step
    buf, handles = global_combine(
        grouped_dup_outp=grouped_dup_outp,
        token_counts_local=token_counts_local,
        token_counts_global=token_counts_global,
        async_op=True,
    )
    _diep_cache_combine.put(key_to_put, (handles, buf, grouped_idx_dup, flat_expert_weights, grouped_dup_outp)) # no need to store token counts
    return prev_buf, prev_grouped_idx_dup, prev_flat_expert_weights

def ep_get_max_mem():
    return max_mem