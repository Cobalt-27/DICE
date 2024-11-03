from cudaprof.prof import CudaProfiler
import os
def analyse_prof(profiler: CudaProfiler):
    """
    Analyse the professional data of the expert.
    """
    total_times, avg_times = profiler.get_all_elapsed_times()
    total_time, _ = profiler.elapsed_time('total')
    split = "-" * 20
    output_lines = []
    output_lines.append(split)
    # sort by time
    for key, total_times in sorted(total_times.items(), key=lambda item: item[1], reverse=True):
        line = f"[{key}] {total_times/1000:.2f}s {total_times/total_time:.2%} avg={avg_times[key]:.2f}ms"
        output_lines.append(line)
    output_lines.append(split)
    return output_lines