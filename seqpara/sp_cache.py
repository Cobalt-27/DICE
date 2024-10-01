import torch
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
        auto_gc: automatically clear send buffer after allgather completes
        """ 
        self.auto_gc = auto_gc
        self.cache = {}

    def clear(self):
        self.cache = {}

    def put(self, key, value):
        assert isinstance(key, int) and len(value) == ENTRY_VAL_LEN
        if value is not None:
            """
            The send buf is tensor, the recv buf is a list of tensors
            """
            assert value[SEND_BUF_IDX] is None or isinstance(value[SEND_BUF_IDX], torch.Tensor) and value[SEND_BUF_IDX].is_contiguous()
            assert isinstance(value[RECV_BUF_IDX], list) 
            assert all(tensor.is_contiguous() for tensor in value[RECV_BUF_IDX])
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
        If an async all2all operation is completed, clear the send buffer.
        """
        # TODO: gc
        pass

