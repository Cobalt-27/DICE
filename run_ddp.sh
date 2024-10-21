#!/bin/bash

output_script="last_run_ddp_param.sh"



echo "Select model:"
echo "1) DiT-XL/2"
echo "2) DiT-B/2"
echo "3) DiT-S/2"
echo "4) DiT-G/2"
read -p "Enter choice [1-4]: " choice

num_experts=8
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
    4)
        model="DiT-G/2"
        ckpt_path="/root/autodl-tmp/dit_moe_g_16E2A.pt"
        num_experts=16
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac
vae_path="/mnt/vae"
image_size=256
extra_args=""

read -p "Use --ep? (y/n, default n): " use_ep
use_ep=${use_ep:-n}
if [ "$use_ep" = "y" ]; then
    extra_args+=" --ep"
fi

read -p "Use --ep-async? (y/n, default n): " use_ep_async
use_ep_async=${use_ep_async:-n}
if [ "$use_ep_async" = "y" ]; then
    extra_args+=" --ep-async"
fi

read -p "Use --sp? (y/n, default n): " use_sp
use_sp=${use_sp:-n}
if [ "$use_sp" = "y" ]; then
    extra_args+=" --sp"
fi

read -p "Use --sp-async? (y/n, default n): " use_sp_async
use_sp_async=${use_sp_async:-n}
if [ "$use_sp_async" = "y" ]; then
    extra_args+=" --sp-async"
fi



read -p "Enter world size (default 2): " world_size
world_size=${world_size:-2}

read -p "Enter per-process batch size (default 4): " per_proc_batch_size
per_proc_batch_size=${per_proc_batch_size:-4}


# Ask the user if they want to include optional arguments
# Initialize extra_args as an empty string




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

read -p "Use --trim-samples? (y/n, default n): " use_trim_samples
use_trim_samples=${use_trim_samples:-n}
if [ "$use_trim_samples" = "y" ]; then
    extra_args+=" --trim-samples"
fi

read -p "Use --ep-async-warm-up? (default 0): " ep_async_warm_up
ep_async_warm_up=${ep_async_warm_up:-0}

read -p "Use --strided-sync? (default 0): " strided_sync
strided_sync=${strided_sync:-0}

read -p "Use --sp-async-warm-up? (default 0): " sp_async_warm_up
sp_async_warm_up=${sp_async_warm_up:-0}

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

read -p "Enter CUDA visible devices (default all): " cuda_visible_devices
cuda_visible_devices=${cuda_visible_devices:-all}
if [ "$cuda_visible_devices" != "all" ]; then
    export CUDA_VISIBLE_DEVICES=$cuda_visible_devices
fi

function save_run_command {
    echo "#!/bin/bash" > "$output_script"
    echo "" >> "$output_script"
    echo "torchrun --nproc_per_node $world_size sample_ddp.py \\" >> "$output_script"
    echo "--per-proc-batch-size $per_proc_batch_size \\" >> "$output_script"
    echo "--model $model \\" >> "$output_script"
    echo "--vae-path $vae_path \\" >> "$output_script"
    echo "--ckpt $ckpt_path \\" >> "$output_script"
    echo "--num-experts $num_experts \\" >> "$output_script"
    echo "--image-size $image_size \\" >> "$output_script"
    echo "--cfg-scale $cfg_scale \\" >> "$output_script"
    echo "--num-sampling-steps $num_sample_steps \\" >> "$output_script"
    echo "--num-fid-samples $fid_samples \\" >> "$output_script"
    echo "--tf32 \\" >> "$output_script"
    echo "--ep-async-warm-up $ep_async_warm_up \\" >> "$output_script"
    echo "--strided-sync $strided_sync \\" >> "$output_script"
    echo "--sp-async-warm-up $sp_async_warm_up \\" >> "$output_script"
    echo "$extra_args" >> "$output_script"

    chmod +x "$output_script"
    echo "Run command saved to $output_script"
}

save_run_command

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
--ep-async-warm-up $ep_async_warm_up \
--strided-sync $strided_sync \
--sp-async-warm-up $sp_async_warm_up \
$extra_args