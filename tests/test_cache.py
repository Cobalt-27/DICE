import pytest
import torch
from expertpara.ep_cache import All2AllCache
from expertpara.offload import AsyncTensorOffloading
import random

# Set a global random seed for all tests
@pytest.fixture(scope="session", autouse=True)
def set_seed():
    seed = 42  # Any number you prefer
    random.seed(seed)
    torch.manual_seed(seed)


# DummyHandle class as defined above
class DummyHandle:
    def __init__(self, completed=True):
        self.completed = completed

    def is_completed(self):
        return self.completed

def make_dummy_handle(completed=True):
    return DummyHandle(completed=completed)

def arbitrary_tensor_op():
    # Arbitrary tensor operation
    tensor = torch.randn(10, 10, device='cuda')
    result = torch.matmul(tensor, tensor)

@pytest.fixture
def setup_cache():
    # Fixture to create a cache instance for testing
    capacity = 4
    auto_gc = False
    offload = False
    prefetch_size = 0
    val_len = 3  # At least 3: handles, recv_buf, send_buf
    cache = All2AllCache(capacity=capacity, val_len=val_len, auto_gc=auto_gc, offload=offload, prefetch_size=prefetch_size)
    return cache

def test_cache_size():
    cache = All2AllCache(capacity=10,val_len=3, auto_gc=False, offload=False)
    # Clear caches before starting
    # Test 1: Check cache size when caches are empty
    assert cache.tensors_size() == 0, "Cache size should be 0 when empty."

    # Test 2: Add a tensor to the dispatch cache
    tensor1 = torch.randn(10, 10)
    cache.put(1, (None, tensor1, None))
    expected_size = tensor1.element_size() * tensor1.numel()
    assert cache.tensors_size() == expected_size, f"Expected cache size {expected_size}, got {cache.tensors_size()}."

    # Test 3: Add another tensor to the combine cache
    tensor2 = torch.randn(5, 5)
    cache.put(5, (None, tensor2, None))
    expected_size += tensor2.element_size() * tensor2.numel()
    assert cache.tensors_size() == expected_size, f"Expected cache size {expected_size}, got {cache.tensors_size()}."

    # Test 4: Clear the cache and check size
    cache.clear()
    assert cache.tensors_size() == 0, "Cache size should be 0 after clearing."

    print("All cache size tests completed.")

def test_basic_cache_operations(setup_cache):
    cache = setup_cache
    
    # Create sample tensors
    tensor1 = torch.randn(10, 10, device='cuda')
    tensor2 = torch.randn(20, 20, device='cuda')
    send_buf1 = torch.randn(5, 5, device='cuda')
    send_buf2 = torch.randn(5, 5, device='cuda')

    # Create dummy handles
    handles1 = [make_dummy_handle(completed=True)]
    handles2 = [make_dummy_handle(completed=False)]

    # Put entries into cache
    cache.put(0, (handles1, tensor1.clone(), send_buf1.clone()))
    cache.put(1, (handles2, tensor2.clone(), send_buf2.clone()))

    # Retrieve entries and verify
    entry0 = cache.get(0)
    assert entry0[0] == handles1  # Handles
    assert torch.equal(entry0[1], tensor1)  # recv_buf
    assert torch.equal(entry0[2], send_buf1)  # send_buf

    entry1 = cache.get(1)
    assert entry1[0] == handles2
    assert torch.equal(entry1[1], tensor2)
    assert torch.equal(entry1[2], send_buf2)

    # Test contains method
    assert cache.contains(0) is True
    assert cache.contains(1) is True
    assert cache.contains(2) is False  # Entry not added

def test_gc_behavior():
    # Initialize cache with auto_gc=True
    capacity = 2
    auto_gc = True
    offload = False
    prefetch_size = 0
    val_len = 3
    cache = All2AllCache(capacity=capacity, auto_gc=auto_gc, offload=offload, prefetch_size=prefetch_size, val_len=val_len)

    # Create sample tensors
    tensor = torch.randn(10, 10, device='cuda')
    send_buf = torch.randn(5, 5, device='cuda')

    # Create a dummy handle that is not completed
    handles = [make_dummy_handle(completed=False)]
    cache.put(0, (handles, tensor.clone(), send_buf.clone()))

    # Before GC, send_buf should still be present
    assert cache.get(0)[2] is not None  # send_buf is present

    # Simulate handle completion
    handles[0].completed = True

    # Since auto_gc=True, gc should be called automatically during put
    cache.put(1, (None, None, None))  # Trigger gc

    # After GC, send_buf should be cleared
    assert cache.get(0)[2] is None  # send_buf is cleared

def test_offloading_behavior():
    # Initialize cache with offloading enabled
    capacity = 3
    auto_gc = False
    offload = True
    prefetch_size = 0
    val_len = 3
    cache = All2AllCache(capacity=capacity, auto_gc=auto_gc, offload=offload, prefetch_size=prefetch_size, val_len=val_len)

    # Create sample tensors
    tensor1 = torch.randn(10, 10, device='cuda')
    tensor2 = torch.randn(10, 10, device='cuda')
    send_buf1 = torch.randn(5, 5, device='cuda')
    send_buf2 = torch.randn(5, 5, device='cuda')

    # Create dummy handles that are completed
    handles1 = [make_dummy_handle(completed=True)]
    handles2 = [make_dummy_handle(completed=True)]

    arbitrary_tensor_op()
    cache.put(0, (handles1, tensor1.clone(), send_buf1.clone()))
    arbitrary_tensor_op()
    cache.put(1, (handles2, tensor2.clone(), send_buf2.clone()))
    arbitrary_tensor_op()

    # Access entry 0 to trigger offloading of other entries
    cache.get(0)
    arbitrary_tensor_op()
    # Entry 1 should be offloaded
    entry1 = cache.cache[1]
    recv_buf1 = entry1[1]
    assert isinstance(recv_buf1, AsyncTensorOffloading)

def test_random_operations():
    import random

    capacity = 5
    val_len = 3
    # Test different cache settings
    settings = [
        {'auto_gc': True, 'offload': False, 'prefetch_size': 0},
        {'auto_gc': False, 'offload': True, 'prefetch_size': 1},
        {'auto_gc': True, 'offload': True, 'prefetch_size': 2},
    ]

    for setting in settings:
        cache = All2AllCache(capacity=capacity, auto_gc=setting['auto_gc'], offload=setting['offload'], prefetch_size=setting['prefetch_size'], val_len=val_len)
        sim_cache = [None] * capacity  # Simulated simple cache

        for _ in range(50):  # Perform 50 random operations
            operation = random.choice(['put', 'get'])
            idx = random.randint(0, capacity - 1)
            arbitrary_tensor_op()
            if operation == 'put':
                # Create tensors
                tensor = torch.randn(10, 10, device='cuda')
                send_buf = torch.randn(5, 5, device='cuda')
                handles = [make_dummy_handle(completed=random.choice([True, False]))]

                # Clone tensors to prevent cross-contamination
                cache.put(idx, (handles, tensor.clone(), send_buf.clone()))
                sim_cache[idx] = (handles, tensor.clone(), send_buf.clone())

            elif operation == 'get':
                sim_entry = sim_cache[idx]                
                if sim_entry is not None:
                    cache_entry = cache.get(idx)
                    # Handle offloading
                    recv_buf_cache = cache_entry[1]
                    assert not isinstance(recv_buf_cache, AsyncTensorOffloading)
                    arbitrary_tensor_op()
                    # Compare results
                    assert torch.equal(recv_buf_cache, sim_entry[1])

def test_offloading_async_behavior():
    import random
    import time

    capacity = 5
    auto_gc = False
    offload = True
    prefetch_size = 2
    val_len = 3
    cache = All2AllCache(capacity=capacity, auto_gc=auto_gc, offload=offload, prefetch_size=prefetch_size, val_len=val_len)
    sim_cache = [None] * capacity

    # Initialize cache entries
    for idx in range(capacity):
        tensor = torch.full((10, 10), idx, device='cuda')
        send_buf = torch.full((5, 5), idx, device='cuda')
        handles = [make_dummy_handle(completed=True)]
        cache.put(idx, (handles, tensor.clone(), send_buf.clone()))
        sim_cache[idx] = (handles, tensor.clone(), send_buf.clone())

    # Access entries in random order
    indices = list(range(capacity))
    random.shuffle(indices)
    for idx in indices:
        cache_entry = cache.get(idx)
        sim_entry = sim_cache[idx]

        # Insert arbitrary tensor operations
        time.sleep(0.1)  # Simulate delay
        arbitrary_tensor_op()
        recv_buf_cache = cache_entry[1]
        assert not isinstance(recv_buf_cache, AsyncTensorOffloading)

        arbitrary_tensor_op()

        # Verify correctness
        assert torch.equal(recv_buf_cache, sim_entry[1])
