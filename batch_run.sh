#!/bin/bash

# 定义想要变化的参数，例如 per_proc_batch_size 的不同值
batch_sizes=(4 8 16 32 64)
model_names=(1 4)
combinations=("yynnn"  "ynnnn" "nnyyn") # 

for model_name in "${model_names[@]}"; do
    for combination in "${combinations[@]}"; do
        echo "Running with combination: $combination"
        
        for batch_size in "${batch_sizes[@]}"; do
            echo "Running with per-process batch size: $batch_size"

            first_choice="${combination:1:1}"
            if [ "$first_choice" == "y" ]; then
                auto_gc="y"
                echo "using gc"
            else
                auto_gc="n"
            fi
        # 使用命令行参数来调用原始脚本并传递需要的值
        ./run_ddp.sh <<EOF
$model_name
${combination:0:1}
${combination:1:1}
${combination:2:1}
${combination:3:1}
${combination:4:1}
4
$batch_size
$auto_gc
n
n
n
0
0
0
0
n

1.5




EOF
        done
    done
done