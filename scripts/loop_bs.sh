#!/bin/bash

# to log results:
# screen -L -Logfile my_logfile.log

echo "Select model:"
echo "1) DiT-XL/2"
echo "2) DiT-B/2"
echo "3) DiT-S/2"
read -p "Enter choice [1-3]: " choice

case $choice in
    1)
        model="DiT-XL/2"
        ckpt_path="/mnt/dit_moe_xl_8E2A.pt"
        ;;
    2)
        model="DiT-B/2"
        ckpt_path="/mnt/dit_moe_b_8E2A.pt"
        ;;
    3)
        model="DiT-S/2"
        ckpt_path="/mnt/dit_moe_s_8E2A.pt"
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

# read -p "Enter number of fid samples (default is 256): " fid_samples
# fid_samples=${fid_samples:-256}

read -p "Enter CUDA devices (default is 0,1,2,3): " cuda_devices
cuda_devices=${cuda_devices:-0,1,2,3}
export CUDA_VISIBLE_DEVICES=$cuda_devices

read -p "Enter world size (default is 2): " world_size
world_size=${world_size:-2}

# Loop over batch sizes, doubling each time
read -p "Enter start batch size (default is 4): " start_batch_size
start_batch_size=${start_batch_size:-4}

read -p "Enter end batch size (default is 128): " end_batch_size
end_batch_size=${end_batch_size:-128}

read -p "Enter experiment folder name: " experiment_folder


num_experts=8
vae_path="/mnt/vae"

num_sample_steps=500
image_size=256
cfg_scale=2.0
folder_name="loop-bs-${model//\//-}-range-${start_batch_size}-to-${end_batch_size}-GPUx${world_size}-${experiment_folder}"
cache_prefetch=2
cache_stride=2



batch_size=$start_batch_size

while [ $batch_size -le $end_batch_size ]; do
    echo "Running test with per_proc_batch_size: $batch_size"
    
    fid_samples=$((batch_size * world_size * 2)) # 2 iters

    torchrun --nproc_per_node $world_size sample_ddp.py \
    --per-proc-batch-size $batch_size \
    --model $model \
    --vae-path $vae_path \
    --ckpt $ckpt_path \
    --num-experts $num_experts \
    --image-size $image_size \
    --cfg-scale $cfg_scale \
    --num-sampling-steps $num_sample_steps \
    --num-fid-samples $fid_samples \
    --tf32 \
    --extra-folder-name $folder_name \
    --filter-samples \
    --diep \
    --auto-gc \
    # --offload \
    # --cache-prefetch $cache_prefetch \
    # --cache-stride $cache_stride \

    torchrun --nproc_per_node $world_size sample_ddp.py \
    --per-proc-batch-size $batch_size \
    --model $model \
    --vae-path $vae_path \
    --ckpt $ckpt_path \
    --num-experts $num_experts \
    --image-size $image_size \
    --cfg-scale $cfg_scale \
    --num-sampling-steps $num_sample_steps \
    --num-fid-samples $fid_samples \
    --extra-folder-name $folder_name \
    --tf32 \
    --filter-samples \
    
    echo "Test completed for per_proc_batch_size: $batch_size"
    
    batch_size=$((batch_size * 2))
done
done
