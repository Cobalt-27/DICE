import torch

"""
Cache structure:

DiEP enables async all2all (token dispatch and combine).

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

class All2AllCache:
    def __init__(self, capacity, auto_gc, val_len):
        self.capacity = capacity
        self.auto_gc = auto_gc
        self.cache = [None] * capacity
        self.val_len = (
            val_len  # len of the value tuple, useful to avoid missing items in tuple
        )

    def clear(self):
        self.cache = [None] * self.capacity

    def put(self, key, value):
        
        assert isinstance(key, int) and len(value) == self.val_len
        if value is not None:
            """
            The first item will always be the handles, the last item will always be the send buffer
            """
            assert value[0] is None or isinstance(value[0], list)
            # should contains cuda in device name
            assert value[-1] is None or (isinstance(value[-1], torch.Tensor) and value[-1].device.type == "cuda")
        if self.auto_gc:
            self.gc()
        self.cache[key] = value

    def get(self, key):
        assert isinstance(key, int)
        return self.cache[key]

    def contains(self, key):
        assert isinstance(key, int)
        return self.cache[key] is not None

    def gc(self):
        """
        If an async all2all operation is completed, clear the send buffer.
        """
        for i in range(self.capacity):
            if self.cache[i] is not None:
                item = list(self.cache[i])
                prev_handles = item[0]
                if prev_handles is not None and all(
                    handle.is_completed() for handle in prev_handles
                ):
                    item[0] = None
                    item[-1] = None  # Clear send buffer
                    self.cache[i] = tuple(item)

    def tensors_size(self):
        """
        Returns the size of all the tensors in the cache in bytes.
        """
        return sum(
            [
                sum(
                    [
                        x.element_size() * x.numel()
                        for x in v
                        if isinstance(x, torch.Tensor)
                    ]
                )
                for v in self.cache
                if v is not None
            ]
        )
