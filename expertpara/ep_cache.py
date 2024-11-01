import torch
from cudaprof.prof import CudaProfiler

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

NO LONGER USED
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

import torch.distributed as dist
class All2AllCache:
    def __init__(self, capacity, val_len, auto_gc, cl_name = None):
        """
        capacity: the number of entries in the cache
        auto_gc: automatically clear send buffer after all2all completes
        val_len: the length of the value tuple
        """ 
        self.capacity = capacity
        self.auto_gc = auto_gc
        self.cache = [None] * capacity
        self.val_len = (
            val_len  # len of the value tuple, useful to avoid missing items in tuple
        )
        self.cl_name = cl_name
    """
    NOTE: the second item in the value tuple is the recv_buf
    """

    def clear(self):
        self.cache = [None] * self.capacity

    # @CudaProfiler.prof_func('cache.put')
    def put(self, key, value):
        # if self.cl_name:
        #     if dist.get_rank() == 0:
        #         print(f'put in {self.cl_name}')

        assert isinstance(key, int) and len(value) == self.val_len
        if value is not None:
            """
            The first item will always be the handles, the last item will always be the send buffer
            """
            assert value[HANDLES_IDX] is None or isinstance(value[HANDLES_IDX], list)
            assert value[SEND_BUF_IDX] is None or isinstance(value[SEND_BUF_IDX], torch.Tensor)

        if self.auto_gc:
            self._gc()
        self.cache[key] = value

    # @CudaProfiler.prof_func('cache.get')
    def get(self, key):
        # if self.name:
        #     if dist.get_rank() == 0:
        #         print(f'get from {self.cl_name}')
        
        assert isinstance(key, int)
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
        # return 0
        
        return sum(
            sum(
                (
                    x.element_size() * x.numel()
                    if isinstance(x, torch.Tensor)
                    else 0
                )
                for x in v
            ) if v is not None else 0
            for 
            v in self.cache
        )

class EPSkipCache:
    """
    cache to help skip non-share experts
    skipping: dispatch + non-share mlp + combine
    stores the result of previous step's combine
    """
    def __init__(self, capacity):
        """
        capacity: the number of entries in the cache
        auto_gc: automatically clear send buffer after all2all completes
        val_len: the length of the value tuple
        """ 
        self.capacity = capacity
        self.cache = [None] * capacity

    """
    NOTE: the second item in the value tuple is the recv_buf
    """

    def clear(self):
        self.cache = [None] * self.capacity

    def put(self, key, tensor):
        assert isinstance(key, int)
        assert isinstance(tensor, torch.Tensor)
        self.cache[key] = tensor

    def get(self, key):
        assert isinstance(key, int)
        assert isinstance(self.cache[key], torch.Tensor)
        return self.cache[key]

    def contains(self, key):
        assert isinstance(key, int)
        return self.cache[key] is not None

    def tensors_size(self):
        """
        Returns the size of all the tensors in the cache in bytes.
        """
        # return 0
        
        return sum(
            x.element_size() * x.numel() if isinstance(x, torch.Tensor) else 0
            for x in self.cache
        )
    