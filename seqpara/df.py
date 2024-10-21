import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.distributed as dist
from .sp_cache import AllGatherCache
from .sp_fwd import sp_all_gather
from cudaprof.prof import CudaProfiler

_kcache = None
_vcache = None

max_mem = -1

def sp_cache_init(auto_gc,is_rf = False):
    global _kcache, _vcache
    _kcache = AllGatherCache(auto_gc)
    _vcache = AllGatherCache(auto_gc)

    global is_rf_g
    is_rf_g = is_rf
    if is_rf:
        global _kcache_vc, _vcache_vc,_kcache_vu, _vcache_vu,is_vc
        is_vc = True
        _kcache_vc, _vcache_vc = _kcache,_vcache
        _kcache_vu = AllGatherCache(auto_gc)
        _vcache_vu = AllGatherCache(auto_gc)

def sp_to_vc():
    assert is_rf_g
    global _kcache,_vcache
    _kcache,_vcache = _kcache_vc, _vcache_vc

    global is_vc
    is_vc =True

def sp_to_vu():
    assert is_rf_g
    global _kcache,_vcache
    _kcache,_vcache = _kcache_vu, _vcache_vu

    global is_vc
    is_vc =True

def sp_get_rf_state():
    return is_vc

def sp_cached_tensors_size():
    if is_rf_g:
        return _kcache_vc.tensors_size() + _vcache_vc.tensors_size() + \
            _kcache_vu.tensors_size() + _vcache_vu.tensors_size()
    else: 
        return _kcache.tensors_size() + _vcache.tensors_size()

def sp_cache_clear():
    if is_rf_g:
        _kcache_vc.clear()
        _vcache_vc.clear()
        _kcache_vu.clear()
        _vcache_vu.clear()
    else:
        _kcache.clear()
        _vcache.clear()

@torch.no_grad()
def _all_gather_async(x_local,key, cache, concat_dim):
    if not cache.contains(key):
        x_list = sp_all_gather(x_local=x_local, concat_dim=None, async_op=False)
        x_all = torch.cat(x_list, dim=concat_dim)
        cache.put(key, (None, x_list, x_local))
        return x_all
    
    prev_handle, prev_x_list, prev_x = cache.get(key)
    if prev_handle is not None:
        with CudaProfiler.scope("all_gather.wait"):
            prev_handle.wait()
    """
    NOTE: prev_x_list are kv tensors in previous step, have to replace the current rank's tensor with the new one
    """
    assert prev_x_list[dist.get_rank()] is None
    prev_x_list[dist.get_rank()] = x_local
    x_all = torch.cat(prev_x_list, dim=concat_dim)
    x_list, handle = sp_all_gather(x_local=x_local, concat_dim=None, async_op=True)
    cache.put(key, (handle, x_list, x_local))
    
    mem_use = sp_cached_tensors_size()
    global max_mem
    if mem_use > max_mem:
        max_mem = mem_use

    return x_all

@torch.no_grad()
def sp_all_gather_async(k, v, key, concat_dim):
    k_all = _all_gather_async(k, key, _kcache, concat_dim)
    v_all = _all_gather_async(v, key, _vcache, concat_dim)
    return k_all, v_all
    
def sp_get_max_mem():
    return max_mem