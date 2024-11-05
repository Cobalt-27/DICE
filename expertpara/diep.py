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

_CACHE_DISPATCH_VAL_LEN = 7
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
    if _use_separate_cache:
        return _diep_cache_dispatch_vc.tensors_size()+ _diep_cache_combine_vc.tensors_size()+\
                    _diep_cache_dispatch_vu.tensors_size()+_diep_cache_combine_vu.tensors_size()
    else:
        return _diep_cache_dispatch.tensors_size() + _diep_cache_combine.tensors_size()
    
def ep_skip_cached_tensors_size():
    """
    Measures the size of all the tensors in the cache in bytes.
    """
    if _use_separate_cache:
        return _diep_cache_skip_vc.tensors_size()+_diep_cache_skip_vu.tensors_size()
    else:
        return _diep_cache_skip.tensors_size()

def ep_set_step(step):
    global _step
    _step = step

def ep_separate_cache():
    return _use_separate_cache

def ep_skip_this_step(idx):
    """
    Decide whether to skip unimportant tokens in this step.
    """
    if not _enable_skip or not _diep_cache_skip.contains(idx):
        # disabled, or no cache due to first step
        return False
    return _step % _comm_step != 0

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

def ep_skip_enabled():
    return _enable_skip
   

def ep_cache_init(cache_capacity, auto_gc=False, separate_cache =True, comm_step = 1, skip_mode = None):
    
    global _comm_step
    """
    perform all2all every comm_step steps, for other steps, skip all2all and mlp via cache_skip
    """
    _comm_step = comm_step
    global _enable_skip
    _enable_skip = comm_step != 1
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
    global _use_separate_cache
    
    _use_separate_cache =separate_cache

    if separate_cache:
        global _diep_cache_dispatch_vc, _diep_cache_combine_vc,_diep_cache_dispatch_vu,_diep_cache_combine_vu,is_vc
        is_vc =True
        _diep_cache_dispatch_vc,_diep_cache_combine_vc = _diep_cache_dispatch,_diep_cache_combine
        _diep_cache_dispatch_vu = All2AllCache(
            capacity=cache_capacity,
            auto_gc=auto_gc,
            val_len=_CACHE_DISPATCH_VAL_LEN,
            cl_name= '_diep_cache_dispatch_vu'
        )
        _diep_cache_combine_vu = All2AllCache(
            capacity=cache_capacity,
            auto_gc=auto_gc,
            val_len=_CACHE_COMBINE_VAL_LEN,
            cl_name = '_diep_cache_combine_vu' 
        )
        _diep_cache_dispatch_vc.cl_name = '_diep_cache_dispatch_vc'
        _diep_cache_combine_vc.cl_name = '_diep_cache_combine_vc'
        
        global _diep_cache_skip_vc, _diep_cache_skip_vu
        _diep_cache_skip_vu = _diep_cache_skip
        _diep_cache_skip_vc = EPSkipCache(
            capacity=cache_capacity if _enable_skip else 0,
        )

def ep_to_vc():
    assert _use_separate_cache, "Only the RectifiedFlow can make ep cache space change"
    global _diep_cache_dispatch, _diep_cache_combine
    _diep_cache_dispatch, _diep_cache_combine = _diep_cache_dispatch_vc,_diep_cache_combine_vc
    
    global _diep_cache_skip
    _diep_cache_skip = _diep_cache_skip_vc

    global is_vc 
    is_vc = True

def ep_to_vu():
    assert _use_separate_cache, "Only the RectifiedFlow can make ep cache space change"
    global _diep_cache_dispatch, _diep_cache_combine
    _diep_cache_dispatch, _diep_cache_combine = _diep_cache_dispatch_vu,_diep_cache_combine_vu

    global _diep_cache_skip
    _diep_cache_skip = _diep_cache_skip_vu
    
    global is_vc
    is_vc = False

def ep_get_rf_state():
    return is_vc

def ep_cache_clear():
    if _use_separate_cache:
        _diep_cache_dispatch_vu.clear()
        _diep_cache_combine_vu.clear()
        _diep_cache_dispatch_vc.clear()
        _diep_cache_combine_vc.clear()
        _diep_cache_skip_vu.clear()
        _diep_cache_skip_vc.clear()
    else:
        _diep_cache_dispatch.clear()
        _diep_cache_combine.clear()
        _diep_cache_skip.clear()

def get_result_to_skip(key):
    """
    Get the previous result to skip the current step.
    """
    return _diep_cache_skip.get(key)

def put_result_for_skip(key, tensor):
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
        # if dist.get_rank() == 0:
        #     print("no caches")
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
    _diep_cache_dispatch.put(cache_key, (handles, buf, token_counts_local, token_counts_global, grouped_idx_dup, flat_expert_weights, grouped_dup_inp))
    
    mem_use = ep_cached_tensors_size()
    global max_mem
    if mem_use > max_mem:
        max_mem = mem_use

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
    _diep_cache_combine.put(cache_key, (handles, buf, grouped_idx_dup, flat_expert_weights, grouped_dup_outp)) # no need to store token counts
    return prev_buf, prev_grouped_idx_dup, prev_flat_expert_weights

def ep_get_max_mem():
    return max_mem