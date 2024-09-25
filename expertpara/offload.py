import torch
from .prof import CudaProfiler
class AsyncTensorOffloading:
    """
    Used to offload tensors between CPU and GPU asynchronously.
    1. offload a tensor by creating an instance of this class
    2. async load the tensor back to GPU by calling async_prefetch
    3. wait for the async operation to complete by calling wait
    """
    def __init__(self, tensor):
        self.offload_finish_event = torch.cuda.Event() # to cpu
        self.fetch_finish_event = torch.cuda.Event() # to gpu
        self.stream = torch.cuda.Stream()  # Use a separate stream
        self.device = tensor.device
        """Asynchronously copy the tensor from GPU to CPU."""
        with CudaProfiler.scope('offload.to_cpu', stream=self.stream):
            with torch.cuda.stream(self.stream):
                cpu_tensor = tensor.to(device='cpu', non_blocking=True)  # Async copy to CPU
                self.offload_finish_event.record(self.stream)  # Record event for tracking
                
        self.cpu_tensor = cpu_tensor
        self.gpu_tensor = None # not yet copied back to GPU

    def async_prefetch(self):
        """Asynchronously copy the tensor from CPU to GPU."""
        if self.gpu_tensor is not None: # avoid multiple prefetch
            return
        with CudaProfiler.scope('offload.to_gpu', stream=self.stream):
            with torch.cuda.stream(self.stream):
                gpu_tensor = self.cpu_tensor.to(self.device, non_blocking=True)  # Async copy to GPU
                self.fetch_finish_event.record(self.stream)  # Record event for tracking
        self.gpu_tensor = gpu_tensor


    def wait(self):
        """Wait for the async operation to complete."""
        if self.gpu_tensor is None:
            self.async_prefetch()
        self.fetch_finish_event.synchronize()
        return self.gpu_tensor