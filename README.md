# DICE: Staleness-Centric Optimizations for Efficient Diffusion MoE Inference

This repository contains the open-source code for our paper on DICE: Staleness-Centric Optimizations for Efficient Diffusion MoE Inference. We plan to provide more comprehensive and well-organized information in the future. For now, you can refer to the code for specific details and usage.

Based on DIT-MoE's codebase, use `run_ddp.sh`  for various parallelism.

arguments for core features, refer to `sample_ddp.py` for details:
- `--ep-async`: displaced expert parallelism
- `--sp-async`: displaced seq parallelism (Distrifusion)
- `--ep-async-pipeline`: interweaved parallelism
- `--ep-async-mode`: set to `shallow` for selective sync
- `--ep-async-skip-strategy`: set to `low` for conditional comm