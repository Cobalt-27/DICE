#!/bin/bash

# to log results:
# screen -L -Logfile my_logfile.log

echo "Select model:"
echo "1) DiT-B/2"
echo "2) DiT-S/2"
read -p "Enter choice [1 or 2]: " choice

case $choice in
    1)
        model="DiT-B/2"
        ckpt_path="/mnt/dit_moe_b_8E2A.pt"
        ;;
    2)
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

num_experts="8"
vae_path="/mnt/vae"
num_sample_steps="500"
image_size="256"
cfg_scale="1.5"
folder_name="loop-bs-${model//\//-}-GPUx${world_size}"
cache_prefetch="2"


# Loop over batch sizes from 1 to 64, doubling each time
for per_proc_batch_size in 4 8 16 32 64 128; do
    echo "Running test with per_proc_batch_size: $per_proc_batch_size"
    
    fid_samples=$((per_proc_batch_size * 4))

    torchrun --nproc_per_node $world_size sample_ddp.py \
    --per-proc-batch-size $per_proc_batch_size \
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
    --diep \
    --auto-gc \
    --offload \
    --cache-prefetch $cache_prefetch \

    torchrun --nproc_per_node $world_size sample_ddp.py \
    --per-proc-batch-size $per_proc_batch_size \
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
    
    echo "Test completed for per_proc_batch_size: $per_proc_batch_size"
done
