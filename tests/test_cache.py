import torch
from expertpara.diep import cache_size, cache_clear, _cache_put, _diep_cache_dispatch, _diep_cache_combine

def test_cache_size():
    # Clear caches before starting
    cache_clear()

    # Test 1: Check cache size when caches are empty
    assert cache_size() == 0, "Cache size should be 0 when empty."

    # Test 2: Add a tensor to the dispatch cache
    tensor1 = torch.randn(10, 10)
    _cache_put(_diep_cache_dispatch, 'key1', tensor1)
    expected_size = tensor1.element_size() * tensor1.numel()
    assert cache_size() == expected_size, f"Expected cache size {expected_size}, got {cache_size()}."

    # Test 3: Add another tensor to the combine cache
    tensor2 = torch.randn(5, 5)
    _cache_put(_diep_cache_combine, 'key2', tensor2)
    expected_size += tensor2.element_size() * tensor2.numel()
    assert cache_size() == expected_size, f"Expected cache size {expected_size}, got {cache_size()}."

    # Test 4: Clear the cache and check size
    cache_clear()
    assert cache_size() == 0, "Cache size should be 0 after clearing."

    print("All cache size tests completed.")

if __name__ == "__main__":
    test_cache_size()