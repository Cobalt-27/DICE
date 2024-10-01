#!/bin/bash

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

echo "Select para_mode: 1) none    2) ep    3) diep    4) sp    5) df"
read -p "Enter choice [1-5]: " para_choice

para_modes=("none" "ep" "diep" "sp" "df")
if [[ $para_choice -ge 1 && $para_choice -le 5 ]]; then
    para_mode=${para_modes[$((para_choice-1))]}
else
    echo "Invalid choice. Exiting."
    exit 1
fi

vae_path="/mnt/vae"
num_experts=8
image_size=256

read -p "Enter world size (default 2): " world_size
world_size=${world_size:-2}

read -p "Enter per-process batch size (default 4): " per_proc_batch_size
per_proc_batch_size=${per_proc_batch_size:-4}


# Ask the user if they want to include optional arguments
# Initialize extra_args as an empty string
extra_args=""

read -p "Use --auto-gc? (y/n, default n): " use_auto_gc
use_auto_gc=${use_auto_gc:-n}
if [ "$use_auto_gc" = "y" ]; then
    extra_args+=" --auto-gc"
fi

read -p "Use --offload? (y/n, default n): " use_offload
use_offload=${use_offload:-n}
if [ "$use_offload" = "y" ]; then
    extra_args+=" --offload"
fi

read -p "Use --trim-samples? (y/n, default y): " use_trim_samples
use_trim_samples=${use_trim_samples:-y}
if [ "$use_trim_samples" = "y" ]; then
    extra_args+=" --trim-samples"
fi

read -p "Enter cache prefetch (default None): " cache_prefetch

read -p "Enter cache stride (default None): " cache_stride

if [ -n "$cache_prefetch" ]; then
    extra_args+=" --cache-prefetch $cache_prefetch"
fi

if [ -n "$cache_stride" ]; then
    extra_args+=" --cache-stride $cache_stride"
fi

read -p "Enter number of sampling steps (invalid for XL&G, default 500): " num_sample_steps
num_sample_steps=${num_sample_steps:-500}

read -p "Enter CFG scale (default 1.5): " cfg_scale
cfg_scale=${cfg_scale:-1.5}

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
--para-mode $para_mode \
$extra_args