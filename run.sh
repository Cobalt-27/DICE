#!/bin/bash

model="DiT-S/2"
ckpt_path="/mnt/dit_moe_s_8E2A.pt"
vae_path="/mnt/vae"
num_experts="8"
num_sample_steps="1000"
image_size="256"
cfg_scale="1.5"

CUDA_VISIBLE_DEVICES=0

python3 sample.py \
--model $model \
--ckpt $ckpt_path \
--vae-path $vae_path \
--image-size $image_size \
--cfg-scale $cfg_scale \
--num_experts $num_experts \
--num-sampling-steps $num_sample_steps \
