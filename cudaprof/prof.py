import torch
import time
import functools

class CudaProfiler:
    def __init__(self):
        self.events = {}
        self.is_started = False

    def start(self, name, stream=None, cpu=False):
        """
        Start recording time for a named section. Supports multiple starts.
        """
        if name not in self.events:
            self.events[name] = {'start': [], 'end': [], 'elapsed': 0.0, 'cpu': cpu}
        assert len(self.events[name]['start']) == len(self.events[name]['end']), \
            f"Cannot start '{name}' as there are more starts than stops"
        
        if cpu:
            start_event = time.time()
        else:
            start_event = torch.cuda.Event(enable_timing=True)
            if stream is not None:
                start_event.record(stream)
            else:
                start_event.record()
        
        self.events[name]['start'].append(start_event)

    def stop(self, name, stream=None, cpu=False):
        """
        Stop recording time for a named section. Accumulates total time for multiple stops.
        """
        assert name in self.events, f"No events recorded for '{name}'"
        assert len(self.events[name]['start']) - 1 == len(self.events[name]['end']), \
            f"Cannot stop '{name}' as there are more stops than starts"

        if cpu:
            end_event = time.time()
        else:
            end_event = torch.cuda.Event(enable_timing=True)
            if stream is not None:
                end_event.record(stream)
            else:
                end_event.record()
        
        self.events[name]['end'].append(end_event)

    def elapsed_time(self, name):
        """
        Get the total accumulated time for a specific named section.
        Syncs and stores the result, clearing start and end events after.
        """
        if name not in self.events:
            raise ValueError(f"No events recorded for '{name}'")
        
        # Check if CPU timing or CUDA event timing is used
        cpu = self.events[name]['cpu']
        total_time = self.events[name]['elapsed']
        total_count = len(self.events[name]['start'])
        # Accumulate new times
        if cpu:
            for start, end in zip(self.events[name]['start'], self.events[name]['end']):
                total_time += (end - start) * 1000  # Convert seconds to ms
        else:
            torch.cuda.synchronize()
            for start, end in zip(self.events[name]['start'], self.events[name]['end']):
                total_time += start.elapsed_time(end)

        # Store the accumulated time and clear the events
        self.events[name]['elapsed'] = total_time
        self.events[name]['start'] = []
        self.events[name]['end'] = []
        avg_time = total_time / total_count if total_count > 0 else 0
        return total_time, avg_time

    def get_all_elapsed_times(self):
        """
        Returns a dictionary of the accumulated elapsed times for each recorded event.
        """
        total_times = {}
        avg_times = {}
        for name in self.events:
            total_times[name], avg_times[name] = self.elapsed_time(name)
        return total_times, avg_times

    def sync(self):
        """
        Manually synchronize all recorded events.
        """
        torch.cuda.synchronize()

    def reset(self):
        """
        Reset all stored events.
        """
        self.events = {}

    _instance = None

    @staticmethod
    def prof():
        """
        Singleton pattern to get the instance of the profiler.
        """
        if CudaProfiler._instance is None:
            CudaProfiler._instance = CudaProfiler()
        return CudaProfiler._instance
    
    class ProfileContext:
        def __init__(self, profiler, name, stream=None, cpu=False):
            self.profiler = profiler
            self.name = name
            self.stream = stream
            self.cpu = cpu

        def __enter__(self):
            self.profiler.start(self.name, self.stream, cpu=self.cpu)

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.profiler.stop(self.name, self.stream, cpu=self.cpu)

    def scope(name, stream=None, cpu=False):
        """
        Create a context manager for profiling a block of code.
        Usage:
        with CudaProfiler.scope('name', cpu=True):
            # Code to profile
        """
        prof = CudaProfiler.prof()
        return prof.ProfileContext(prof, name, stream, cpu)

    @staticmethod
    def prof_func(name, cpu=False):
        """
        Decorator to profile a function using the CudaProfiler.
        Logs the total execution time of the function.
        """
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Start profiling
                CudaProfiler.prof().start(name, cpu=cpu)
                result = func(*args, **kwargs)
                # Stop profiling
                CudaProfiler.prof().stop(name, cpu=cpu)
                return result
            return wrapper
        return decorator

"""
# Example usage
profiler = CudaProfiler()

# Start profiling different sections multiple times
profiler.start('diffusion_step_1')
# Perform some GPU-intensive operation
# model.forward()
profiler.stop('diffusion_step_1')

# Do it again for the same step
profiler.start('diffusion_step_1')
# Perform another operation
# model.forward()
profiler.stop('diffusion_step_1')

# Sync manually if needed
profiler.sync()

# Get total elapsed time for this section (accumulated over multiple start/stop)
print(f"Elapsed time for diffusion_step_1: {profiler.elapsed_time('diffusion_step_1')} ms")

# Get all recorded times
print(profiler.get_all_elapsed_times())

# Get total time for all steps
print(f"Total elapsed time: {profiler.total_elapsed_time()} ms")

# Reset when needed
profiler.reset()
"""