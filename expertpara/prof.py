import torch

class CudaProfiler:
    def __init__(self):
        self.events = {}
        self.current_start = None
        self.is_started = False

    def start(self, name):
        """
        Start recording time for a named section. Supports multiple starts.
        """
        if name not in self.events:
            self.events[name] = {'start': [], 'end': [], 'elapsed': 0.0}
        start_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        self.events[name]['start'].append(start_event)
        self.is_started = True

    def stop(self, name):
        """
        Stop recording time for a named section. Accumulates total time for multiple stops.
        """
        if not self.is_started:
            raise RuntimeError("No timer was started.")
        
        assert name in self.events, f"No events recorded for '{name}'"
        end_event = torch.cuda.Event(enable_timing=True)
        end_event.record()
        self.events[name]['end'].append(end_event)
        self.is_started = False

    def elapsed_time(self, name):
        """
        Get the total accumulated time for a specific named section.
        It syncs only when calculating the elapsed time.
        """
        if name not in self.events:
            raise ValueError(f"No events recorded for '{name}'")

        total_time = 0.0
        for start, end in zip(self.events[name]['start'], self.events[name]['end']):
            torch.cuda.synchronize()
            total_time += start.elapsed_time(end)
        
        return total_time

    def get_all_elapsed_times(self):
        """
        Returns a dictionary of the accumulated elapsed times for each recorded event.
        """
        times = {}
        for name in self.events:
            times[name] = self.elapsed_time(name)
        return times

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
        self.is_started = False
        self.current_start = None

    _instance = None
    @staticmethod
    def prof():
        """
        Singleton pattern to get the instance of the profiler.
        """
        if CudaProfiler._instance is None:
            CudaProfiler._instance = CudaProfiler()
        return CudaProfiler._instance

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