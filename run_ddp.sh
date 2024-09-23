#!/bin/bash

model="DiT-S/2"
ckpt_path="/mnt/dit_moe_s_8E2A.pt"
vae_path="/mnt/vae"
num_experts="8"
num_sample_steps="500"
image_size="256"
cfg_scale="1.5"
fid_samples="256"
per_proc_batch_size="32"

CUDA_VISIBLE_DEVICES=0,1

torchrun --nproc_per_node 2 sample_ddp.py \
--per-proc-batch-size $per_proc_batch_size \
--model $model \
--vae-path $vae_path \
--ckpt $ckpt_path \
--image-size $image_size \
--cfg-scale $cfg_scale \
--num-sampling-steps $num_sample_steps \
--num-fid-samples $fid_samples \
--tf32 \
--diep \
