from .prof import CudaProfiler
import os
def analyse_prof(profiler: CudaProfiler, path: str = None):
    """
    Analyse the professional data of the expert.
    """
    times = profiler.get_all_elapsed_times()
    total_time = profiler.elapsed_time('total')
    split = "-" * 20
    output_lines = []
    output_lines.append(split)
    print(split)
    for key, time in times.items():
        line = f"[{key}] {time/1000:.2f}s {time/total_time:.2%}"
        print(line)
        output_lines.append(line)
    
    output_lines.append(split)
    print(split)
    
    if path is not None:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'a') as file:
            for line in output_lines:
                file.write(line + '\n')