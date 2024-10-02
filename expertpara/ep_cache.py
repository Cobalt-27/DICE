import torch
from .offload import AsyncTensorOffloading
from .prof import CudaProfiler

"""
Cache structure:

DiEP enables async all2all (token dispatch and combine), this cache stores the all2all (results + meta + handler)l

We have a cache entry for each layer.

In step T, forward func calls async dispatch_async:

- recv the tokens from previous step (get this from cache)
- starts a new async all2all, to be received in next step,
    this async all2all shall be stored in this cache.

To store a all2all operation, we need to store the following data:
- a list of async op handles, since each dispatch/combine needs multiple dist.all_to_all
- recv_buf
- token_counts_local
- token_counts_global
- send_buf, incase it's released before op completes

"""

"""

Cache optimization

AUTO GC and OFFLOADING are orthogonal.

AUTO GC:
Free send_buf after all2all completes

OFFLOADING:
Move recv_buf to cpu after all2all completes, move it back when time is ripe

Since the layers are access in a ring (layer 1..L then 1). 
We use a ring-style offloading, keeping only a continuous section of the recv_buf in GPU.
Leaving the rest in CPU.

Entry states:
- BUSY: all2all is in progress, recv_buf is in GPU
- LOADED: all2all is completed, recv_buf is in GPU or is being transferred to GPU
- OFFLOADED: all2all is completed, recv_buf is in CPU

offloading param:
prefetch_size: the size of the recv_buf to keep in GPU, 0 for no prefetch

"""

HANDLES_IDX = 0
RECV_BUF_IDX = 1
SEND_BUF_IDX = -1

# NOTE: idx and key are the same thing here

def is_completed(handles):
    if handles is not None:
        return all(handle.is_completed() for handle in handles)
    return True

class All2AllCache:
    def __init__(self, capacity, val_len, auto_gc, offload, prefetch_size = None, offload_mask = None):
        """
        capacity: the number of entries in the cache
        auto_gc: automatically clear send buffer after all2all completes
        prefetch_size: the size of the recv_buf to keep in GPU, 0 for no prefetch
        val_len: the length of the value tuple
        offload: whether to offload each recv_buf to CPU
        """ 
        self.capacity = capacity
        self.auto_gc = auto_gc
        self.cache = [None] * capacity
        self.val_len = (
            val_len  # len of the value tuple, useful to avoid missing items in tuple
        )
        self.offload = offload
        self.prefetch_size = prefetch_size
        self.offload_mask = offload_mask if offload_mask is not None else [True] * capacity
        if self.offload:
            assert prefetch_size is not None
    """
    NOTE: the second item in the value tuple is the recv_buf
    """

    def _try_release_entry_gpu_mem(self, idx):
        value = list(self.cache[idx])
        recv_buf = value[RECV_BUF_IDX]
        if isinstance(recv_buf, AsyncTensorOffloading):
            recv_buf.release_gpu_mem_if_possible()

    def _try_offload_entry(self, idx):
        value = list(self.cache[idx])
        recv_buf = value[RECV_BUF_IDX]
        if isinstance(recv_buf, AsyncTensorOffloading):
            return
        assert isinstance(recv_buf, torch.Tensor)
        offloaded = AsyncTensorOffloading(recv_buf)
        value[1] = offloaded
        self.cache[idx] = tuple(value)

    def _try_prefetch_entry(self, idx):
        value = list(self.cache[idx])
        recv_buf = value[RECV_BUF_IDX]
        if isinstance(recv_buf, AsyncTensorOffloading):
            recv_buf.async_prefetch()

    def _wait_entry_if_needed(self, idx):
        value = list(self.cache[idx])
        recv_buf = value[RECV_BUF_IDX]
        if isinstance(recv_buf, AsyncTensorOffloading):
            recv_buf = recv_buf.wait()
            
        value[RECV_BUF_IDX] = recv_buf
        self.cache[idx] = tuple(value)

    def _offload_and_prefetch(self, current_idx):
        """
        current_idx: the index being accessed
        
        We iterate across the cache in a ring-style, offloading and prefetching the recv_buf
        - Try to prefetch back the next prefetch_size entries
        - For other entries, try to offload the recv_buf
        - Noted that we can only TRY to prefetch/offload, the all2all of any entry is not guaranteed to be completed
        """
        for i in range(1,self.capacity): # skip the current_idx, since it's being accessed
            idx = (current_idx + i) % self.capacity
            if self.cache[idx] is not None and self.offload_mask[idx]:
                if is_completed(self.cache[idx][HANDLES_IDX]):
                    if i <= self.prefetch_size:
                        self._try_prefetch_entry(idx)
                    else:
                        self._try_offload_entry(idx)
                    self._try_release_entry_gpu_mem(idx)

    def clear(self):
        self.cache = [None] * self.capacity

    # @CudaProfiler.prof_func('cache.put')
    def put(self, key, value):
        assert isinstance(key, int) and len(value) == self.val_len
        if value is not None:
            """
            The first item will always be the handles, the last item will always be the send buffer
            """
            assert value[HANDLES_IDX] is None or isinstance(value[HANDLES_IDX], list)
            assert value[SEND_BUF_IDX] is None or isinstance(value[SEND_BUF_IDX], torch.Tensor)

        if self.auto_gc:
            self._gc()
        """
        NOTE:
        Putting offload checks here because in cache warm-up iteration, we are only putting items.
        Must trigger offloading here to ease the peak memory usage.
        """
        if self.offload and self.offload_mask[key]:
            with CudaProfiler.scope(f"cache.offload&prefetch"):
                self._offload_and_prefetch(key)
        self.cache[key] = value

    # @CudaProfiler.prof_func('cache.get')
    def get(self, key):
        assert isinstance(key, int)
        if self.offload and self.offload_mask[key]:
            with CudaProfiler.scope(f"cache.wait"):
                self._wait_entry_if_needed(key)
            # with CudaProfiler.scope(f"cache.offload&prefetch"):
            #     self._offload_and_prefetch(key)
        assert isinstance(self.cache[key][RECV_BUF_IDX], torch.Tensor)
        return self.cache[key]

    def contains(self, key):
        assert isinstance(key, int)
        return self.cache[key] is not None

    def _gc(self):
        """
        If an async all2all operation is completed, clear the send buffer.
        """
        for i in range(self.capacity):
            if self.cache[i] is not None:
                item = list(self.cache[i])
                prev_handles = item[HANDLES_IDX]
                if is_completed(prev_handles):
                    item[HANDLES_IDX] = None
                    item[SEND_BUF_IDX] = None  # Clear send buffer
                    self.cache[i] = tuple(item)

    def tensors_size(self):
        """
        Returns the size of all the tensors in the cache in bytes.
        """
        return sum(
            sum(
                (
                    x.element_size() * x.numel()
                    if isinstance(x, torch.Tensor)
                    else (
                        x.gpu_mem_size() if isinstance(x, AsyncTensorOffloading) else 0
                    )
                )
                for x in v
            )
            for v in self.cache
        )
