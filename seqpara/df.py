import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.distributed as dist
from .sp_cache import AllGatherCache
from .sp_fwd import sp_all_gather
from .comm_manager import PatchParallelismCommManager
from cudaprof.prof import CudaProfiler

# current kv cache
_kcache = None
_vcache = None

# current comm manager
comm_manager = None

is_vc = None
"""
if True, use distrifuser's comm manager for async allgather
"""
use_mngr = None

max_mem = -1

def sp_init(sp_use_mngr, capacity = None, comm_checkpoint = None, auto_gc = True, separate_cache = None):
    global sp_separate_cache
    sp_separate_cache = separate_cache
    global use_mngr
    use_mngr = sp_use_mngr
    if sp_use_mngr:
        _sp_mngr_init(capacity, comm_checkpoint, separate_cache)
    else:
        _sp_cache_init(auto_gc, separate_cache)

def _sp_mngr_init(capacity, comm_checkpoint, separate_cache):
    assert comm_checkpoint is not None
    if separate_cache:
        global comm_manager_vu, comm_manager_vc
        comm_manager_vu = PatchParallelismCommManager(capacity, comm_checkpoint)
        comm_manager_vc = PatchParallelismCommManager(capacity, comm_checkpoint)
    else:
        global comm_manager
        comm_manager = PatchParallelismCommManager(capacity, comm_checkpoint)

def _sp_cache_init(auto_gc , separate_cache):
    """
    auto_gc: if True, automatically clear local kv tensors after allgather completes
    separate_cache: if True, use separate cache for cond and null-cond in diffusion CFG
    """
    assert auto_gc is not None and separate_cache is not None

    if separate_cache:
        global _kcache_vc, _vcache_vc,_kcache_vu, _vcache_vu
        _kcache_vc = AllGatherCache(auto_gc)
        _vcache_vc = AllGatherCache(auto_gc)
        _kcache_vu = AllGatherCache(auto_gc)
        _vcache_vu = AllGatherCache(auto_gc)
    else:
        global _kcache, _vcache
        _kcache = AllGatherCache(auto_gc)
        _vcache = AllGatherCache(auto_gc)

def sp_to_vc():
    assert sp_separate_cache
    if use_mngr:
        global comm_manager
        comm_manager = comm_manager_vc
    else:
        global _kcache,_vcache
        _kcache,_vcache = _kcache_vc, _vcache_vc

    global is_vc
    is_vc =True

def sp_to_vu():
    assert sp_separate_cache
    if use_mngr:
        global comm_manager
        comm_manager = comm_manager_vu
    else:
        global _kcache,_vcache
        _kcache,_vcache = _kcache_vu, _vcache_vu

    global is_vc
    is_vc =True

def sp_cached_tensors_size():
    """
    total buffer size in bytes
    """
    if use_mngr:
        if sp_separate_cache:
            return comm_manager_vc.buf_size() + comm_manager_vu.buf_size()
        else:
            return comm_manager.buf_size()
    else:
        if sp_separate_cache:
            return _kcache_vc.tensors_size() + _vcache_vc.tensors_size() + \
                _kcache_vu.tensors_size() + _vcache_vu.tensors_size()
        else: 
            return _kcache.tensors_size() + _vcache.tensors_size()

def sp_cache_clear():
    if use_mngr:
        if sp_separate_cache:
            comm_manager_vc.clear()
            comm_manager_vu.clear()
        else:
            comm_manager.clear()
    else:
        if sp_separate_cache:
            _kcache_vc.clear()
            _vcache_vc.clear()
            _kcache_vu.clear()
            _vcache_vu.clear()
        else:
            _kcache.clear()
            _vcache.clear()

@torch.no_grad()
def _all_gather_async_cache(x_local,key, cache, concat_dim):
    if not cache.contains(key):
        x_list = sp_all_gather(x_local=x_local, concat_dim=None, async_op=False)
        x_all = torch.cat(x_list, dim=concat_dim)
        cache.put(key, (None, x_list, x_local))
        return x_all
    
    prev_handle, prev_x_list, prev_x = cache.get(key)
    if prev_handle is not None:
        with CudaProfiler.scope("all_gather.wait", cpu=True):
            prev_handle.wait()
    """
    NOTE: prev_x_list are kv tensors in previous step, have to replace the current rank's tensor with the new one
    """
    assert prev_x_list[dist.get_rank()] is None
    prev_x_list[dist.get_rank()] = x_local
    x_all = torch.cat(prev_x_list, dim=concat_dim)
    x_list, handle = sp_all_gather(x_local=x_local, concat_dim=None, async_op=True)
    cache.put(key, (handle, x_list, x_local))
    
    global max_mem
    max_mem = max(max_mem, sp_cached_tensors_size())

    return x_all

@torch.no_grad()
def _all_gather_async_mngr(k_local, v_local, idx, concat_dim):
    kv_local = torch.stack([k_local, v_local], dim=0)
    if comm_manager.buffer_list is None:
        comm_manager.register_tensor_in_bulk(shape=kv_local.shape, layer_type='attn', torch_dtype=kv_local.dtype)
        comm_manager.create_buffer()
    
    if comm_manager.handles[idx] is not None:
        with CudaProfiler.scope("all_gather.wait", cpu=True):
            comm_manager.handles[idx].wait()
    else:
        # cache is empty
        dist.all_gather(comm_manager.get_buffer_list(idx), kv_local, async_op=False)
        class DummyHandle:
            def wait(self):
                pass
        comm_manager.handles[idx] = DummyHandle()
    kv_list = [buf for buf in comm_manager.get_buffer_list(idx)]
    kv_list[dist.get_rank()] = kv_local # replace the current rank's tensor with the new one
    assert kv_list[0].shape == kv_local.shape
    k_list = [kv[0] for kv in kv_list]
    v_list = [kv[1] for kv in kv_list]
    k_all = torch.cat(k_list, dim=concat_dim)
    v_all = torch.cat(v_list, dim=concat_dim)
    
    comm_manager.enqueue(idx, kv_local)
    global max_mem
    max_mem = max(max_mem, sp_cached_tensors_size())
    return k_all, v_all

@torch.no_grad()
def sp_all_gather_async(k, v, key, concat_dim):
    if use_mngr:
        return _all_gather_async_mngr(k, v, key, concat_dim)
    else:
        k_all = _all_gather_async_cache(k, key, _kcache, concat_dim)
        v_all = _all_gather_async_cache(v, key, _vcache, concat_dim)
        return k_all, v_all

def sp_get_max_mem():
    return max_mem
