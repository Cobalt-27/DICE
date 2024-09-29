#!/bin/bash

world_size=${WORLD_SIZE:-2}

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

vae_path="/mnt/vae"
num_experts=8
image_size=256

read -p "Enter number of sampling steps (default 500): " num_sample_steps
num_sample_steps=${num_sample_steps:-500}

read -p "Enter CFG scale (default 1.5): " cfg_scale
cfg_scale=${cfg_scale:-1.5}

read -p "Enter per-process batch size (default 4): " per_proc_batch_size
per_proc_batch_size=${per_proc_batch_size:-4}

read -p "Enter cache prefetch (default 2): " cache_prefetch
cache_prefetch=${cache_prefetch:-2}

read -p "Enter cache stride (default 2): " cache_stride
cache_stride=${cache_stride:-2}

read -p "Enter number of FID samples (default $((per_proc_batch_size * world_size))): " fid_samples
fid_samples=${fid_samples:-$((per_proc_batch_size * world_size))}

read -p "Enter CUDA visible devices (default 0,1): " cuda_visible_devices
cuda_visible_devices=${cuda_visible_devices:-0,1}
export CUDA_VISIBLE_DEVICES=$cuda_visible_devices

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
--diep \
--auto-gc \
--offload \
--cache-prefetch $cache_prefetch \
--cache-stride $cache_stride \
