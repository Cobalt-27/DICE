import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.distributed as dist
from .sp_cache import AllGatherCache
from .sp_fwd import sp_all_gather

_kcache = None
_vcache = None

def sp_cache_init(auto_gc):
    global _kcache, _vcache
    _kcache = AllGatherCache(auto_gc)
    _vcache = AllGatherCache(auto_gc)

def sp_cached_tensors_size():
    return _kcache.tensors_size() + _vcache.tensors_size()

def sp_cache_clear():
    _kcache.clear()
    _vcache.clear()

@torch.no_grad()
def _all_gather_async(x_local,key, cache, concat_dim):
    if not cache.contains(key):
        x_list = sp_all_gather(x_local=x_local, concat_dim=None, async_op=False)
        cache.put(key, (None, x_list, x_local))
        x_all = torch.cat(x_list, dim=concat_dim)
        return x_all
    
    prev_handle, prev_x_list, prev_x = cache.get(key)
    if prev_handle is not None:
        prev_handle.wait()
    prev_x_list[dist.get_rank()] = x_local
    x_all = torch.cat(prev_x_list, dim=concat_dim)
    x_list, handle = sp_all_gather(x_local=x_local, concat_dim=None, async_op=True)
    cache.put(key, (handle, x_list, x_local))
    return x_all

@torch.no_grad()
def sp_all_gather_async(k, v, key, concat_dim):
    k_all = _all_gather_async(k, key, _kcache, concat_dim)
    v_all = _all_gather_async(v, key, _vcache, concat_dim)
    return k_all, v_all
    