import torch

class AsyncTensorOffloading:
    """
    Used to offload tensors between CPU and GPU asynchronously.
    1. offload a tensor by creating an instance of this class
    2. async load the tensor back to GPU by calling async_prefetch
    3. wait for the async operation to complete by calling wait
    """
    def __init__(self, tensor):
        self.stream = torch.cuda.Stream()
        self.copy_event = torch.cuda.Event()
        self.device = tensor.device
        """Asynchronously copy the tensor from GPU to CPU."""
        with torch.cuda.stream(self.stream):
            cpu_tensor = tensor.cpu()  # Async copy to CPU
            self.copy_event.record(self.stream)  # Record event for tracking
        self.cpu_tensor = cpu_tensor
        self.gpu_tensor = None # not yet copied back to GPU

    def async_prefetch(self):
        """Asynchronously copy the tensor from CPU to GPU."""
        with torch.cuda.stream(self.stream):
            gpu_tensor = self.cpu_tensor.to(self.device)  # Async copy to GPU
            self.copy_event.record(self.stream)  # Record event for tracking
        self.gpu_tensor = gpu_tensor

    def is_completed(self):
        """Check if the async operation has completed."""
        return self.copy_event.query()

    def wait(self):
        """Wait for the async operation to complete."""
        if self.gpu_tensor is None:
            self.async_prefetch()
        self.copy_event.synchronize()
        return self.gpu_tensor