import torch
from expertpara.prof import CudaProfiler

def test_single_section():
    profiler = CudaProfiler()
    profiler.start('section_1')
    torch.cuda.synchronize()  # Simulate some GPU work
    profiler.stop('section_1')
    elapsed_time = profiler.elapsed_time('section_1')
    assert elapsed_time > 0, "Elapsed time should be greater than 0"

def test_multiple_sections():
    profiler = CudaProfiler()
    profiler.start('section_1')
    torch.cuda.synchronize()  # Simulate some GPU work
    profiler.stop('section_1')

    profiler.start('section_2')
    torch.cuda.synchronize()  # Simulate some GPU work
    profiler.stop('section_2')

    elapsed_time_1 = profiler.elapsed_time('section_1')
    elapsed_time_2 = profiler.elapsed_time('section_2')

    assert elapsed_time_1 > 0, "Elapsed time for section_1 should be greater than 0"
    assert elapsed_time_2 > 0, "Elapsed time for section_2 should be greater than 0"

def test_accumulated_time():
    profiler = CudaProfiler()
    profiler.start('section_1')
    torch.cuda.synchronize()  # Simulate some GPU work
    profiler.stop('section_1')

    profiler.start('section_1')
    torch.cuda.synchronize()  # Simulate some GPU work
    profiler.stop('section_1')

    elapsed_time = profiler.elapsed_time('section_1')
    assert elapsed_time > 0, "Accumulated elapsed time should be greater than 0"

def test_reset():
    profiler = CudaProfiler()
    profiler.start('section_1')
    torch.cuda.synchronize()  # Simulate some GPU work
    profiler.stop('section_1')

    profiler.reset()
    try:
        profiler.elapsed_time('section_1')
        assert False, "Expected ValueError after reset"
    except ValueError:
        pass

def test_singleton():
    profiler1 = CudaProfiler.prof()
    profiler2 = CudaProfiler.prof()
    assert profiler1 is profiler2, "Both instances should be the same (singleton pattern)"

if __name__ == '__main__':
    print("Running profiling tests...")
    test_single_section()
    test_multiple_sections()
    test_accumulated_time()
    test_reset()
    test_singleton()
    print("Profiling tests passed.")