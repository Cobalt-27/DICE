# DICE: Staleness-Centric Optimizations for Efficient Diffusion MoE Inference

> 📄 This is the official repository for the ICCV 2025 paper  
> **DICE: Staleness-Centric Optimizations for Parallel Diffusion MoE Inference**.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


DICE is a framework for **scalable and efficient inference of MoE-based diffusion models**, designed to maximizing image quality and hardware utilization. It supports multiple parallelism strategies, including:
- **Data Parallelism**
- **Expert Parallelism**
- **Displaced Expert Parallelism** (communication-computation overlap, 2-step staleness)
- **Interweaved Parallelism** (1-step staleness, our core contribution)
- **Sequence(Patch) Parallelism**
- **Displaced Sequence Parallelism** (DistriFusion)


## Key Features
- **Interweaved Parallelism**: Halves the reduced staleness compared to displaced expert parallelism.
- **Selective Synchronization**: Ensures critical layers receive fresh activations while allowing asynchronous execution elsewhere.
- **Conditional Communication**: Dynamically reduces communication volume by prioritizing important tokens.

![DICE Architecture](https://img.picgo.net/2025/01/30/archdb5ed79559af67dc.png)

Interweaved Parallelism achieves a **"free lunch" improvement over Displaced Expert Parallelism**—it reduces staleness (1-step vs. 2-step) and memory usage while preserving image quality. For further acceleration, Selective Synchronization and Conditional Communication introduce tunable quality-efficiency tradeoffs

<p align="center">
    <img src="https://img.picgo.net/2025/03/07/_20250307123749f1e5339224df3e8a.png" alt="Performance Trade-off" width="50%" height="50%">
</p>

## Prerequisites
Set up your environment following the [DiT-MoE](https://github.com/feizc/DiT-MoE/tree/main). DICE requires the same dependencies. 



## Code Structure

This is an overview of the various parallelism techniques implemented in the project.
<u>For detailed explanations, please refer to the comments in the corresponding files.</u>

```
/expertpara # Expert Parallelism
├── diep.py # Core implementation for displaced & interweaved EP
├── ep_cache.py # Activation caching for async expert parallelism
├── ep_fwd.py # Vanilla synchronous expert parallelism
└── prof_analyse.py # Profiling

/seqpara # Sequence Parallelism
├── comm_manager.py # Async communication for displaced SP
├── sp_cache.py # Buffer management
├── sp_fwd.py # Synchronous sequence parallelism
└── df.py # DistriFusion adaptation

/tests # Unit tests
```


## Acknowledgements
This repository builds upon the [DiT-MoE](https://github.com/feizc/DiT-MoE/tree/main) codebase,with displaced sequence parallelism implementations adapted from [xDiT](https://github.com/xdit-project/xDiT) and [DistriFusion](https://github.com/mit-han-lab/distrifuser). We thank the authors of these projects for their foundational work.

## Usage

Run `sample_ddp.py` or `run_ddp.sh` for serving over various parallelism(DP/EP/SP).

Configuration:

`--nproc_per_node <int>` : world size (the number of GPU)

`--per-proc-batch-size <int>` : batch size per GPU

`--model <String>` : model name (currently supports `DiT-XL/2` and `DiT-G/2`)

`--num-experts <int>` : The number of experts (`8` for `DiT-XL/2` and `16` for `DiT-G/2`)

`--image-size <int>` : The resolution.

`--cfg-scale <float>` : CFG(Classifier-free guidence)

`--num-sampling-steps <int>` :The total sampling steps (Suggested value `50` for rectified flow)

`num-fid-samples <int>` : The number of images generated.

`--ckpt <path>` : Path to model weights.

`--vae-path <path>` : Path to the vae model.

`--tf32` : Enables TF32.

`--ep` : Enables expert parallelism.

`--ep-async` : Enables asynchronous expert parallelism (Only available for `--ep`)

`--auto-gc` : Enables automatic garbage collection for asynchronous expert parallelism (Only available for `--ep-async`)

`--ep-score-use-latest` : While using asynchronous expert parallelism, the latest router score while be used. (Only available for `--ep-async`)


`--ep-async-warm-up <int>` : Specifies the number of sync steps to run before asynchronous expert parallelism begins.

`--ep-async-cool-down <int>` : Specifies how many steps *before the end* of sampling asynchronous expert parallelism should stop.

`--strided-sync <int>` : Periodic sync.

`--ep-async-pipeline` : Enables interweaved parallelism.

`--ep-async-noskip-step <int>` : `2` for using conditional communication, set to `1` otherwise.

`--ep-async-mode` : Specifies the mode for partial asynchronous expert parallelism. Options are `shallow`, `deep`, `interleaved`, `all`

`--ep-async-skip-strategy <string>` : Conditional Communication. Specifies the strategy for skipping tokens during asynchronous expert parallelism. Options are `rand`, `low`, and `high`.


`--sp` : Enables sequence parallelism

`--sp-async` : Displaced sequence parallelism (Only available for `--sp`) (DistriFusion)

`--sp-async-warm-up` : Specifies the number of sync steps to run before asynchronous sequence parallelism begins

<!-- `--trim-samples` : When processing the VAE, only a single image is processed. This does not affect the number of inferences performed by the DiT model -->

`--single-img` : Set global batch size to 1. Only available for `--sp`.

