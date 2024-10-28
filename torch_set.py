import os

alloc_conf = os.environ.get('PYTORCH_CUDA_ALLOC_CONF', None)
if alloc_conf is not None:
    print(f"Current PYTORCH_CUDA_ALLOC_CONF: {alloc_conf}")
else:
    print("PYTORCH_CUDA_ALLOC_CONF is not set.")
