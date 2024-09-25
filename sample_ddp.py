# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Samples a large number of images from a pre-trained DiT model using DDP.
Subsequently saves a .npz file that can be used to compute FID and other
evaluation metrics via the ADM repo: https://github.com/openai/guided-diffusion/tree/main/evaluations

For a simple single-GPU/CPU sampling script, see sample.py.
"""
import torch
import torch.distributed as dist
from models import DiT_models
from download import find_model
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from diffusion.rectified_flow import RectifiedFlow 
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import math
import argparse
from expertpara.prof import CudaProfiler
from expertpara.prof_analyse import analyse_prof
from expertpara.etrim import trim_state_dict
from expertpara.diep import cache_clear, cached_tensors_size, cache_init


def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


def main(args):
    """
    Run sampling.
    """
    from models import set_ep_async
    set_ep_async(args.diep)

    torch.backends.cuda.matmul.allow_tf32 = args.tf32  # True: fast but may lead to some small numerical differences
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    # Setup DDP:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000
    rf=False
    if args.model == "DiT-XL/2" or args.model == "DiT-G/2": 
        pretraining_tp=1
        use_flash_attn=True 
        dtype = torch.float16
        rf=True
    else:
        pretraining_tp=2
        use_flash_attn=False 
        dtype = torch.float32
    # Load model:
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        num_experts=args.num_experts, # NOTE: should be added, otherwise expert num will be set to default 8
        pretraining_tp=pretraining_tp,
        use_flash_attn=use_flash_attn,
    ).to(device)
    
    if dtype == torch.float16:
        model = model.half()
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    """
    NOTE: trim unused experts, for ep only
    """
    trimmed_state_dict = trim_state_dict(state_dict, args.num_experts)
    model.load_state_dict(trimmed_state_dict)
    
    model.eval()  # important!
    if rf:
        diffusion = RectifiedFlow(model)
    else:
        diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(args.vae_path).to(device)
    assert args.cfg_scale >= 1.0, "In almost all cases, cfg_scale be >= 1.0"
    using_cfg = args.cfg_scale > 1.0

    # Create folder to save samples:
    model_string_name = args.model.replace("/", "-")
    # ckpt_string_name = os.path.basename(args.ckpt).replace(".pt", "") if args.ckpt else "pretrained"
    # folder_name = f"{model_string_name}-{ckpt_string_name}-size-{args.image_size}-vae-{args.vae}-" \
    #               f"cfg-{args.cfg_scale}-seed-{args.global_seed}-async-{args.diep}"
    folder_name = f"{model_string_name}-bs-{args.per_proc_batch_size}" \
                  f"-seed-{args.global_seed}-diep-{args.diep}-gc-{args.auto_gc}-offload-{args.offload}-prefetch-{args.cache_prefetch}{'' if args.extra_name is None else f'-{args.extra_name}'}"
    sample_folder_dir = os.path.join(args.sample_dir, folder_name)
    if args.extra_folder_name is not None:
        sample_folder_dir = os.path.join(args.sample_dir, args.extra_folder_name, folder_name)
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    dist.barrier()

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = args.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)
    if rank == 0:
        # print(f"Total number of images that will be sampled: {total_samples}")
        print(f"Sampling {total_samples} images with batch size {n} on {dist.get_world_size()} workers.")
    assert total_samples % dist.get_world_size() == 0, "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar
    total = 0
    prof_path = os.path.join(sample_folder_dir, "prof.txt")
    if rank == 0 and os.path.exists(prof_path):
        os.remove(prof_path)
    # Ensure the profile path exists
    if rank == 0:
        os.makedirs(os.path.dirname(prof_path), exist_ok=True)

    cache_init(
        cache_capacity=model.depth,
        auto_gc=args.auto_gc,
        offload=args.offload,
        prefetch_size=args.cache_prefetch,
    )
    for _ in pbar:
        # Sample images:
        cache_clear()
        prof_lines=[]
        if dtype == torch.float16: 
            # use rf
            with torch.autocast(device_type='cuda'):
                STEPSIZE = 50
                init_noise = torch.randn(n, model.in_channels, latent_size, latent_size, device=device) 
                # conds = torch.tensor(class_labels, device=device)
                conds = torch.randint(0, args.num_classes, (n,), device=device)
                CudaProfiler.prof().start('total')
                images = diffusion.sample_with_xps(init_noise, conds, null_cond = torch.tensor([1000] * n).cuda(), sample_steps = STEPSIZE, cfg = 7.0)
                CudaProfiler.prof().stop('total')
                samples = vae.decode(images[-1] / 0.18215).sample # only the last one is needed
                samples = samples[:samples.shape[0] // n] # keep only one sample per batch
        else:
            z = torch.randn(n, model.in_channels, latent_size, latent_size, device=device)
            y = torch.randint(0, args.num_classes, (n,), device=device)

            # Setup classifier-free guidance:
            if using_cfg:
                z = torch.cat([z, z], 0)
                y_null = torch.tensor([1000] * n, device=device)
                y = torch.cat([y, y_null], 0)
                model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
                sample_fn = model.forward_with_cfg
            else:
                model_kwargs = dict(y=y)
                sample_fn = model.forward
            CudaProfiler.prof().start('total')
            samples = diffusion.p_sample_loop(
                sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device
            )
            CudaProfiler.prof().stop('total')
            mem_usage_line = f"Memory usage: {torch.cuda.memory_allocated() / (1024 * 1024):.2f} MB"
            prof_lines.append(mem_usage_line)
            samples = samples[:samples.shape[0] // n] # keep only one sample per batch
            if using_cfg:
                samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
            # XXX: extremely large mem usage after each iter, need to find out why
            # XXX: 0.18215???
            samples = vae.decode(samples / 0.18215).sample
        samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

        # Save samples to disk as individual .png files
        for i, sample in enumerate(samples):
            index = i * dist.get_world_size() + rank + total
            Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")
        total += global_batch_size

        if rank == 0:
            cache_line = f"Cache size: {cached_tensors_size() / (1024 * 1024):.2f} MB"
            prof_lines.append(cache_line)
            prof_lines+=analyse_prof(CudaProfiler.prof())
            with open(prof_path, "a") as f:
                f.writelines([line + "\n" for line in prof_lines])
            print("\n".join(prof_lines))

    # Make sure all processes have finished saving their samples before attempting to convert to .npz
    dist.barrier()
    if rank == 0:
        # XXX: no npz for now
        # create_npz_from_sample_folder(sample_folder_dir, args.num_fid_samples)
        print("Done.")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae",  type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--vae-path", type=str, default="samples")
    parser.add_argument("--sample-dir", type=str, default="samples")
    parser.add_argument("--per-proc-batch-size", type=int, default=32)
    parser.add_argument("--num-fid-samples", type=int, default=50_000)
    parser.add_argument('--num-experts', default=16, type=int,) 
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale",  type=float, default=1.5)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--tf32", action="store_true",
                        help="By default, use TF32 matmuls. This massively accelerates sampling on Ampere GPUs.")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    parser.add_argument("--extra-folder-name",type=str,default=None)
    parser.add_argument("--extra-name",type=str,default=None)
    
    # DiEP related
    parser.add_argument("--diep", action="store_true", help="Use DiEP for async expert parallelism.")
    parser.add_argument("--auto-gc", action="store_true", help="Automatically garbage collect the cache.")
    parser.add_argument("--offload", action="store_true", help="Offload cache to CPU.")
    parser.add_argument("--cache-prefetch", type=int, default=None, help="prefetch size for cache offloading")
    
    args = parser.parse_args()
    main(args)
