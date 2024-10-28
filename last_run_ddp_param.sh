#!/bin/bash

torchrun --nproc_per_node 8 sample_ddp.py \
--per-proc-batch-size 32 \
--model DiT-G/2 \
--vae-path /mnt/vae \
--ckpt /root/autodl-tmp/models/dit_moe_g_16E2A.pt \
--num-experts 16 \
--image-size 256 \
--cfg-scale 1.5 \
--num-sampling-steps 0 \
--num-fid-samples 256 \
--tf32 \
--ep-async-warm-up 0 \
--strided-sync 0 \
--sp-async-warm-up 0 \
--ep-async-cool-down 0 \
 --ep --ep-async --sp --sp-async --sp-legacy-cache --auto-gc --ep-share-cache
