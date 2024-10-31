#!/bin/bash

# 定义想要变化的参数，例如 per_proc_batch_size 的不同值
batch_sizes=(1)
model_names=(1 4)
combinations=("yynnn" "nnyyn" "ynnnn") #  "nnyyn" "ynnnn"
img_sizes=(256 512 1024 2048 4096)
sharedCache=('n')
# combinations=("nnnnn" )
for model_name in "${model_names[@]}"; do
    for combination in "${combinations[@]}"; do
        echo "Running with combination: $combination"
        for image_size in "${img_sizes[@]}"; do
            echo "Running with image size: $image_size"
                for shared_cache in "${sharedCache[@]}"; do
                    echo "Running with shared cache: $shared_cache"

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
                        ./latency_run_ddp.sh <<EOF
$model_name
${combination:0:1}
${combination:1:1}
${combination:2:1}
${combination:3:1}
${combination:4:1}
8
$batch_size
$image_size
$auto_gc
$shared_cache
n
n
0
0
0
0
y

1.5


latencyTest

EOF
                done
            done
        done
    done
done