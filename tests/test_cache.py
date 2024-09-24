import torch
from expertpara.ep_cache import All2AllCache

def test_cache_size():
    cache = All2AllCache(10, False, 3)
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
    assert cache.tensors_size() == expected_size, f"Expected cache size {expected_size}, got {cache_size()}."

    # Test 4: Clear the cache and check size
    cache.clear()
    assert cache.tensors_size() == 0, "Cache size should be 0 after clearing."

    print("All cache size tests completed.")

if __name__ == "__main__":
    test_cache_size()