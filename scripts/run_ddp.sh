#!/bin/bash

output_script="./scripts/last_run_ddp_param.sh"



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
        ckpt_path="/root/autodl-tmp/models/dit_moe_g_16E2A.pt"
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

read -p "Use --sp-legacy-cache? (y/n, default n): " use_sp_legacy_cache
use_sp_legacy_cache=${use_sp_legacy_cache:-n}
if [ "$use_sp_legacy_cache" = "y" ]; then
    extra_args+=" --sp-legacy-cache"
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

# read -p "Use --ep-share-cache? (y/n, default n): " use_ep_share_cache
# use_ep_share_cache=${use_ep_share_cache:-n}
# if [ "$use_ep_share_cache" = "y" ]; then
#     extra_args+=" --ep-share-cache"
# fi

# read -p "Use --offload? (y/n, default n): " use_offload
# use_offload=${use_offload:-n}
# if [ "$use_offload" = "y" ]; then
#     extra_args+=" --offload"
# fi

read -p "Use --trim-samples? (y/n, default n): " use_trim_samples
use_trim_samples=${use_trim_samples:-n}
if [ "$use_trim_samples" = "y" ]; then
    extra_args+=" --trim-samples"
fi

read -p "Use --ep-score-use-latest? (y/n, default n): " ep_score_use_latest
ep_score_use_latest=${ep_score_use_latest:-n}
if [ "$ep_score_use_latest" = "y" ]; then
    extra_args+=" --ep-score-use-latest"
fi

read -p "Use --ep-async-pipeline? (y/n, default n): " use_ep_async_pipeline
use_ep_async_pipeline=${use_ep_async_pipeline:-n}
if [ "$use_ep_async_pipeline" = "y" ]; then
    extra_args+=" --ep-async-pipeline"
fi

read -p "Enter --ep-async-mode (default all): " ep_async_mode
ep_async_mode=${ep_async_mode:-all}
if [ "$ep_async_mode" != "None" ]; then
    extra_args+=" --ep-async-mode $ep_async_mode"
fi

read -p "Use --ep-async-warm-up? (default 0): " ep_async_warm_up
ep_async_warm_up=${ep_async_warm_up:-0}

read -p "Use --strided-sync? (default 0): " strided_sync
strided_sync=${strided_sync:-0}

read -p "Use --sp-async-warm-up? (default 0): " sp_async_warm_up
sp_async_warm_up=${sp_async_warm_up:-0}

read -p "Use --ep-async-cool-down? (default 0): " ep_async_cool_down
ep_async_cool_down=${ep_async_cool_down:-0}

read -p "Use --ep-async-noskip-step? (default 1): " ep_async_noskip_step
ep_async_noskip_step=${ep_async_noskip_step:-1}

read -p "Enter --ep-async-skip-strategy (default None): " ep_async_skip_strategy
ep_async_skip_strategy=${ep_async_skip_strategy:-None}
if [ "$ep_async_skip_strategy" != "None" ]; then
    extra_args+=" --ep-async-skip-strategy $ep_async_skip_strategy"
fi

# read -p "Use --ep-async-intra-step-skip? (y/n, default n): " use_ep_async_intra_step_skip
# use_ep_async_intra_step_skip=${use_ep_async_intra_step_skip:-n}
# if [ "$use_ep_async_intra_step_skip" = "y" ]; then
#     extra_args+=" --ep-async-intra-step-skip"
# fi

# read -p "Use --ep-reordered-cfg? (y/n, default n): " use_ep_reordered_cfg
# use_ep_reordered_cfg=${use_ep_reordered_cfg:-n}
# if [ "$use_ep_reordered_cfg" = "y" ]; then
#     extra_args+=" --ep-reordered-cfg"
# fi

read -p "Use --single-img? (y/n, default n): " use_single_img
use_single_img=${use_single_img:-n}
if [ "$use_single_img" = "y" ]; then
    extra_args+=" --single-img"
fi


# read -p "Enter cache prefetch (default None): " cache_prefetch

# read -p "Enter cache stride (default None): " cache_stride

# if [ -n "$cache_prefetch" ]; then
#     extra_args+=" --cache-prefetch $cache_prefetch"
# fi

# if [ -n "$cache_stride" ]; then
#     extra_args+=" --cache-stride $cache_stride"
# fi

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
read -p "Enter --extra-name (default None): " extra_name
extra_name=${extra_name:-None}
extra_args+=" --extra-name $extra_name"

# load all the parameters and save them to a script
function save_run_command {
    echo "#!/bin/bash" > "$output_script"
    echo "" >> "$output_script"
    if [ "$cuda_visible_devices" != "all" ]; then
        echo "export CUDA_VISIBLE_DEVICES=$cuda_visible_devices" >> "$output_script"
        echo "" >> "$output_script"
    fi
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
    echo "--ep-async-cool-down $ep_async_cool_down \\" >> "$output_script"
    echo "--ep-async-noskip-step $ep_async_noskip_step \\" >> "$output_script"
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
--ep-async-cool-down $ep_async_cool_down \
--ep-async-noskip-step $ep_async_noskip_step \
$extra_args