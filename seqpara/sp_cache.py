import torch
import torch.distributed as dist
"""
Cache structure:

This the DiT version of Distrifusion.
Distrifusion enables async sequence parallelism for diffusion models.
- normal SP requires gathering the entire K and V for attn in this step
- async SP aquires K and V from from previous step
- thus, the all_gather is started in the previous step and ends here
- the cache is used to store async all_gather operations

To store a all_gather operation, we need to store the handles, send_buf, and recv_buf

"""
HANDLES_IDX = 0
RECV_BUF_IDX = 1
SEND_BUF_IDX = 2
ENTRY_VAL_LEN = 3

# NOTE: idx and key are the same thing here


class AllGatherCache:
    def __init__(self, auto_gc):
        """
        capacity: the number of entries in the cache
        auto_gc: automatically clear local kv tensors after allgather completes
        """ 
        self.auto_gc = auto_gc
        self.cache = {}
    
    def clear(self):
        self.cache = {}

    def put(self, key, value):
        assert isinstance(key, int) and len(value) == ENTRY_VAL_LEN
        """
        The send buf is tensor, the recv buf is a list of tensors
        we should be keeping the send buf to avoid it's deallocated before the allgather completes
                
        NOTE:
        now the send buf is part of the recv buf list, there is no need to store it, but I am keeping it for minimal changes
        """
        assert value[SEND_BUF_IDX] is None or isinstance(value[SEND_BUF_IDX], torch.Tensor) and value[SEND_BUF_IDX].is_contiguous()
        assert isinstance(value[RECV_BUF_IDX], list) 
        assert all(tensor.is_contiguous() for tensor in value[RECV_BUF_IDX])
        if value[HANDLES_IDX] is None:
            value[RECV_BUF_IDX][dist.get_rank()] = None
        
        if self.auto_gc:
            self._gc()
        self.cache[key] = value

    def get(self, key):
        assert isinstance(key, int)
        assert isinstance(self.cache[key][RECV_BUF_IDX], list)
        return self.cache[key]

    def contains(self, key):
        assert isinstance(key, int)
        return key in self.cache

    def _gc(self):
        """
        If an async all2all operation is completed, clear the send buffer and the recv buffer of the rank.
        """
        for key, value in self.cache.items():
            if value[SEND_BUF_IDX] is None:
                continue
            if value[HANDLES_IDX] is not None and value[HANDLES_IDX].is_completed():
                value = list(value)
                value[SEND_BUF_IDX] = None
                # no need to keep recv buf belonging to this rank, since it will be overwritten in the next step
                value[RECV_BUF_IDX][dist.get_rank()] = None
                self.cache[key] = tuple(value)
    
    def tensors_size(self):
        """
        return the total size of all tensors in bytes
        """
        size = 0
        for key, value in self.cache.items():
            if value[SEND_BUF_IDX] is not None:
                size += value[SEND_BUF_IDX].numel() * value[SEND_BUF_IDX].element_size()
            for tensor in value[RECV_BUF_IDX]:
                if tensor is not None:
                    size += tensor.numel() * tensor.element_size()
        return size

