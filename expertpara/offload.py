import torch
from cudaprof.prof import CudaProfiler
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
            self.main_stream = torch.cuda.current_stream()
            can_start_offload_event = torch.cuda.Event()
            can_start_offload_event.record(self.main_stream)
            with torch.cuda.stream(self.stream):
                self.stream.wait_event(can_start_offload_event)
                # Allocate pinned CPU memory
                self._cpu_tensor = torch.empty_like(tensor, device='cpu', pin_memory=True)
                # Asynchronously copy data from GPU to pinned CPU memory
                self._cpu_tensor.copy_(tensor, non_blocking=True)
                self.offload_finish_event.record(self.stream)  # Record event for tracking
        """
        NOTE: IMPORTANT
        The input tensor's reference MUST BE HOLD otherwise it will be released I guess
        """
        self._inp_tensor = tensor
        self._gpu_tensor = None # not yet copied back to GPU

    def release_gpu_mem_if_possible(self):
        if self.offload_finish_event.query():
            # release the input tensor otherwise offloading saves no memory
            self._inp_tensor = None
    
    def async_prefetch(self):
        """Asynchronously copy the tensor from CPU to GPU."""
        
        """
        NOTE: 
        MUST wait for the offload to FINISH before fetching
        Otherwise the output will be corrupted
        """
        if self._gpu_tensor is not None: # avoid multiple prefetch
            return False
        with CudaProfiler.scope('offload.to_gpu', stream=self.stream):
            with torch.cuda.stream(self.stream):
                self.stream.wait_event(self.offload_finish_event)
                                # Allocate GPU tensor
                self._gpu_tensor = torch.empty_like(self._cpu_tensor, device=self.device)
                # Asynchronously copy data from pinned CPU memory to GPU
                self._gpu_tensor.copy_(self._cpu_tensor, non_blocking=True)
                self.fetch_finish_event.record(self.stream)  # Record event for tracking
        return True

    def wait(self):
        """Wait for the async operation to complete."""
        if self._gpu_tensor is None:
            self.async_prefetch()
        self.fetch_finish_event.synchronize()
        return self._gpu_tensor
    
    def gpu_mem_size(self):
        size = 0
        if self._gpu_tensor is not None:
            size = self._gpu_tensor.element_size() * self._gpu_tensor.nelement()
        if self._inp_tensor is not None:
            size += self._inp_tensor.element_size() * self._inp_tensor.nelement()
        return size