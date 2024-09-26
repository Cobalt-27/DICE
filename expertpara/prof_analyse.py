from .prof import CudaProfiler
import os
def analyse_prof(profiler: CudaProfiler):
    """
    Analyse the professional data of the expert.
    """
    times = profiler.get_all_elapsed_times()
    total_time = profiler.elapsed_time('total')
    split = "-" * 20
    output_lines = []
    output_lines.append(split)
    # sort by time
    for key, time in sorted(times.items(), key=lambda item: item[1], reverse=True):
        line = f"[{key}] {time/1000:.2f}s {time/total_time:.2%}"
        output_lines.append(line)
    output_lines.append(split)
    return output_lines