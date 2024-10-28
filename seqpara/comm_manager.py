import torch
import torch.distributed as dist

DF_VERBOSE = True

class PatchParallelismCommManager:
    def __init__(
        self,
        capacity: int = None,
        comm_checkpoint: int = 4,
                 ):
        self.capacity = capacity
        self.torch_dtype = None
        self.numel = 0
        self.numel_dict = {}

        self.buffer_list = None

        self.starts = []
        self.ends = []
        self.shapes = []

        self.idx_queue = []

        self.handles = None
        self.device = torch.cuda.current_device()
        self.comm_checkpoint = comm_checkpoint

    def register_tensor_in_bulk(
        self,
        shape: tuple,
        torch_dtype: torch.dtype,
        layer_type: str,
    ) -> None:
        for _ in range(self.capacity):
            self._register_tensor(shape, torch_dtype, layer_type)

    def _register_tensor(
        self, shape: tuple, torch_dtype: torch.dtype, layer_type: str = None
    ) -> int:
        if self.torch_dtype is None:
            self.torch_dtype = torch_dtype
        else:
            assert self.torch_dtype == torch_dtype
        self.starts.append(self.numel)
        numel = 1
        for dim in shape:
            numel *= dim
        self.numel += numel
        if layer_type is not None:
            if layer_type not in self.numel_dict:
                self.numel_dict[layer_type] = 0
            self.numel_dict[layer_type] += numel

        self.ends.append(self.numel)
        self.shapes.append(shape)
        return len(self.starts) - 1

    def create_buffer(self):
        # print("clear cache")
        # torch.cuda.empty_cache()
        if dist.get_rank() == 0 and DF_VERBOSE:
            print(
                f"Create buffer with {self.numel / 1e6:.3f}M parameters for {len(self.starts)} tensors on each device."
            )
            for layer_type, numel in self.numel_dict.items():
                print(f"  {layer_type}: {numel / 1e6:.3f}M parameters")
        assert self.buffer_list is None, "Buffer has already been created."
        self.buffer_list = [
            torch.empty(self.numel, dtype=self.torch_dtype, device=self.device)
            for _ in range(dist.get_world_size())
        ]
        self.handles = [None for _ in range(len(self.starts))]

    def get_buffer_list(self, idx: int) -> list:
        buffer_list = [t[self.starts[idx] : self.ends[idx]].view(self.shapes[idx]) for t in self.buffer_list]
        return buffer_list

    def _communicate(self): # communicate func shall be private
        start = self.starts[self.idx_queue[0]]
        end = self.ends[self.idx_queue[-1]]
        tensor = self.buffer_list[dist.get_rank()][start:end]
        buffer_list = [t[start:end] for t in self.buffer_list]
        handle = dist.all_gather(buffer_list, tensor, async_op=True)
        for i in self.idx_queue:
            self.handles[i] = handle
        self.idx_queue = []

    def enqueue(self, idx: int, tensor: torch.Tensor):
        if idx == 0 and len(self.idx_queue) > 0:
            self._communicate()
        assert len(self.idx_queue) == 0 or self.idx_queue[-1] == idx - 1
        self.idx_queue.append(idx)
        self.buffer_list[dist.get_rank()][self.starts[idx] : self.ends[idx]].copy_(tensor.flatten())

        if len(self.idx_queue) == self.comm_checkpoint:
            self._communicate()

    def clear(self):
        if len(self.idx_queue) > 0:
            self._communicate()
        if self.handles is not None:
            for i in range(len(self.handles)):
                if self.handles[i] is not None:
                    self.handles[i].wait()
                    self.handles[i] = None

    def buf_size(self):
        if self.buffer_list is None:
            return 0
        return sum([t.numel() * t.element_size() for t in self.buffer_list])
