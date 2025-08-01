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
from models import DiT_models, ParaMode
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
from cudaprof.prof import CudaProfiler
from expertpara.prof_analyse import analyse_prof
from expertpara.etrim import trim_state_dict
from expertpara.diep import ep_cache_clear, ep_cached_tensors_size, ep_cache_init,ep_get_max_mem
from seqpara.df import sp_init, sp_cache_clear, sp_cached_tensors_size,sp_get_max_mem
from expertpara.ep_fwd import use_latest_expert_weights
import time


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

import torch.profiler as profiler
from profiler_class import ProfileExp

def main(args):
    """
    Run sampling.
    """
    torch.backends.cuda.matmul.allow_tf32 = args.tf32  # True: fast but may lead to some small numerical differences
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    # Setup dist
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    print(f"args.ckpt:{args.ckpt}")
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
        para_mode=args.para_mode,
    )
    if dtype == torch.float16:
        model = model.half()
    model.to(device)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    # Only for performance testing, we can generate larger images but the quality will be poor
    if args.image_size != 256:
        assert 'pos_embed' in state_dict, "Custom checkpoints must have 'pos_embed' key."
        del state_dict['pos_embed']

    if args.para_mode.ep:
        """
        NOTE: trim unused experts, for ep only
        """
        state_dict = trim_state_dict(state_dict, args.num_experts)
    model.load_state_dict(state_dict, strict=args.image_size == 256)
    
    model.eval()  # important!
    if rf:
        diffusion = RectifiedFlow(model,args.para_mode)
        
    else:
        diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(args.vae_path).to(device)
    assert args.cfg_scale >= 1.0, "In almost all cases, cfg_scale be >= 1.0"
    
    using_cfg = args.cfg_scale > 1.0

    # Create folder to save samples:
    model_string_name = args.model.split("/")[0]
    folder_name = f"{'' if args.extra_name is None else f'{args.extra_name}--'}{model_string_name}-bs-{args.per_proc_batch_size}" \
                f"-seed-{args.global_seed}-mode-{args.para_mode.verbose()}-worldSize-{dist.get_world_size()}-gc-{args.auto_gc}-cfg-{args.cfg_scale}" \
                f"-epWarmUp-{args.ep_async_warm_up}-strideSync-{args.strided_sync}"\
                f"-pipeline-{args.ep_async_pipeline}-epMode-{args.ep_async_mode}"\
                f"-epCoolDown-{args.ep_async_cool_down}-spWarmUp-{args.sp_async_warm_up}-spLegacyCache-{args.sp_legacy_cache}-imgSize-{args.image_size}"\
                f"-noskipStep-{args.ep_async_noskip_step}-skipStrategy-{args.ep_async_skip_strategy}"
                
    if dist.get_rank() == 0:
        print(f"Saving samples to folder: {folder_name}")
        
    sample_folder_dir = os.path.join(args.sample_dir, folder_name)
    if args.extra_folder_name is not None:
        sample_folder_dir = os.path.join(args.sample_dir, args.extra_folder_name, folder_name)
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    dist.barrier()

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    bs = args.per_proc_batch_size
    global_batch_size = bs * dist.get_world_size()
    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)
    if rank == 0:
        # print(f"Total number of images that will be sampled: {total_samples}")
        print(f"Sampling {total_samples} images with batch size {bs} on {dist.get_world_size()} workers.")
    # assert total_samples % dist.get_world_size() == 0, "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    assert samples_needed_this_gpu % bs == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // bs)
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar
    total = 0
    prof_path = os.path.join(sample_folder_dir, "prof.txt")
    mem_summary_path = os.path.join(sample_folder_dir, "mem.txt")
    if rank == 0:
        if os.path.exists(prof_path):
            os.remove(prof_path)
        if os.path.exists(mem_summary_path):
            os.remove(mem_summary_path)
    # Ensure the profile path exists
    if rank == 0:
        os.makedirs(os.path.dirname(prof_path), exist_ok=True)
        os.makedirs(os.path.dirname(mem_summary_path), exist_ok=True)
    
    if rank == 0:
        args_path = os.path.join(sample_folder_dir, "args.txt")
        with open(args_path, "w+") as f:
            import json
            args_dict = vars(args).copy()
            args_dict['para_mode'] = args.para_mode.verbose()
            formatted_args = json.dumps(args_dict, indent=4)
            f.write(formatted_args)

    if args.para_mode.ep and args.para_mode.ep_async:
        
        ep_cache_init(
            cache_capacity=model.depth,
            auto_gc=args.auto_gc,
            noskip_step = args.ep_async_noskip_step,
            skip_mode= args.ep_async_skip_strategy,
            async_pipeline=args.ep_async_pipeline,
            selective_async_strategy=args.ep_async_mode,
        )
        use_latest_expert_weights(not args.ep_score_use_latest)
    if args.para_mode.sp and args.para_mode.sp_async:
        sp_init(sp_use_mngr=not args.sp_legacy_cache, capacity = model.depth, comm_checkpoint = 4, auto_gc = True)
    
    
    # prof= ProfileExp("profile_exp",profile_at= 10 if torch.distributed.get_rank()==0 else -100,
    #                 wait = 0,warmup = 0,active = 1,repeat = 1)
    # if dist.get_rank()==0:
    #     prof._start()
    
    for iter in pbar:
        CudaProfiler.prof().reset()
        # Sample images:
        if args.para_mode.ep_async:
            ep_cache_clear()
        if args.para_mode.sp_async:
            sp_cache_clear()
        prof_lines=[]
        z = torch.randn(bs, model.in_channels, latent_size, latent_size, device=device)
        y = torch.randint(0, args.num_classes, (bs,), device=device)
        
        if args.para_mode.sp and not args.single_img:
            z_list = [torch.zeros_like(z) for _ in range(dist.get_world_size())]
            y_list = [torch.zeros_like(y) for _ in range(dist.get_world_size())]
            dist.all_gather(z_list, z)
            dist.all_gather(y_list, y)
            z = torch.cat(z_list, dim=0)
            y = torch.cat(y_list, dim=0)
            # we gather all z and y to rank0, so that sp is able to produce the same samples as DP and EP
            # latent in rank0 will be scattered to all ranks later
            
        torch.cuda.synchronize()
        time_start = time.time()
        CudaProfiler.prof().start('total')
        if rf: 
            # use rf
            with torch.autocast(device_type='cuda'):
                STEPSIZE = 50
                init_noise = z
                conds = y
                samples = diffusion.sample_with_xps(init_noise, conds, null_cond = torch.tensor([1000] * y.shape[0]).cuda(), 
                            sample_steps = STEPSIZE, cfg = args.cfg_scale,para_mode= args.para_mode)
        else:
            # Setup classifier-free guidance:
            if using_cfg:
                z = torch.cat([z, z], 0)
                y_null = torch.tensor([1000] * y.shape[0], device=device)
                y = torch.cat([y, y_null], 0)
                model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
                sample_fn = model.forward_with_cfg
            else:
                model_kwargs = dict(y=y)
                sample_fn = model.forward
            samples = diffusion.p_sample_loop(
                sample_fn, z.shape, z, clip_denoised=False, 
                model_kwargs=model_kwargs, progress=False, device=device,para_mode=args.para_mode
            )
        CudaProfiler.prof().stop('total')
        torch.cuda.synchronize()
        time_end = time.time()
        
        CudaProfiler.prof().start('vae')
        mem_summary = torch.cuda.memory_summary() + "\n"
        mem_usage_line = f"Memory usage: {torch.cuda.memory_allocated() / (1024 * 1024):.2f} MB"
        measured_total_time_line = f"Measured time: {(time_end - time_start):.2f}s"
        peak_memory_allocated = torch.cuda.max_memory_allocated() / (1024 * 1024)
        peak_memory_allocated_line = f"Max memory usage:{peak_memory_allocated:.2f} MB"
        
        
        if args.trim_samples:
            samples = samples[:samples.shape[0] // bs] # keep only one sample per batch

        
            
        if not rf and using_cfg:
            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
        
        if rank == 0:
            cache_line = ""
            if args.para_mode.ep_async:
                cache_line += f"\nEP Cache size: {ep_cached_tensors_size() / (1024 * 1024):.2f} MB"
                cache_line += f"\nEP Max Cache size: {ep_get_max_mem() / (1024 * 1024):.2f} MB"
            if args.para_mode.sp_async:
                cache_line += f"\nSP Cache size: {sp_cached_tensors_size() / (1024 * 1024):.2f} MB"
                cache_line += f"\nSP Max Cache size: {sp_get_max_mem() / (1024 * 1024):.2f} MB"
            prof_lines.append(cache_line)
            
            

        if args.para_mode.ep_async:
            ep_cache_clear()
        if args.para_mode.sp_async:
            sp_cache_clear()
        
        
        
        if args.para_mode.sp and not args.single_img:
            """
            NOTE:
            use DP for vae if sp is enabled, otherwise all samples will be decoded on rank0
            """
            assert samples.shape[0] % dist.get_world_size() == 0, "samples.shape[0] must be divisible by world_size"
            if rank == 0:
                # Rank 0 splits the tensor into chunks for each rank.
                chunks = list(torch.chunk(samples, dist.get_world_size(), dim=0))  # Split along dim 0
                chunks = [chunk.contiguous() for chunk in chunks]
            else:
                chunks = None  # Other ranks don't have the full tensor
            samples_shape = list(samples.size())
            samples_shape[0] = samples_shape[0] // dist.get_world_size()
            samples_shape = tuple(samples_shape)
            # Create an empty tensor on all ranks to receive the scattered chunk
            local_tensor = samples.new_empty(samples_shape).contiguous()

            # Scatter the chunks from rank 0 to all ranks
            dist.scatter(tensor=local_tensor, scatter_list=chunks, src=0)
            samples = local_tensor
            
        vae.enable_slicing()
        torch.cuda.reset_peak_memory_stats()

        if (not args.single_img or rank == 0) and args.image_size <= 1024:
            samples = vae.decode(samples / 0.18215).sample # magic number due to normalization in the VAE
            samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
            print(f"rank:{rank} samples.shape:{samples.shape}")
            if not args.trim_samples:
                assert len(samples) == bs, f"Expected {bs} samples, got {len(samples)}"
            # Save samples to disk as individual .png files
            for i, sample in enumerate(samples):
                index = i * dist.get_world_size() + rank + total
                Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")
        total += global_batch_size

        peak_memory_allocated_when_vae = torch.cuda.max_memory_allocated() / (1024 * 1024)
        peak_memory_allocated_when_vae_line = f"Max memory usage_when_vae:{peak_memory_allocated_when_vae:.2f} MB"

        prof_lines += [mem_usage_line, measured_total_time_line,
            peak_memory_allocated_line,peak_memory_allocated_when_vae_line]

        CudaProfiler.prof().stop('vae')

        if rank == 0:
            prof_lines+=analyse_prof(CudaProfiler.prof())
            with open(prof_path, "a") as f:
                f.writelines([line + "\n" for line in prof_lines])
            print("\n".join(prof_lines))
            with open(mem_summary_path, "a") as f:
                f.write(mem_summary)

        torch.cuda.empty_cache()

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
    parser.add_argument("--image-size", type=int,  default=256) #choices=[256, 512,1024,4096],
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
    
    # parallelism
    parser.add_argument("--trim-samples", action="store_true", help="Keep only one sample per batch to save mem.")
    
    parser.add_argument("--sp", action="store_true", help="Use SeqPara.")
    parser.add_argument("--ep", action="store_true", help="Use ExpertPara.")
    parser.add_argument("--sp-async", action="store_true", help="Use asynchronous SeqPara.")
    parser.add_argument("--ep-async", action="store_true", help="Use asynchronous ExpertPara.")
    
    parser.add_argument("--auto-gc", action="store_true", help="Automatically garbage collect the cache.")
    
    parser.add_argument("--ep-async-pipeline", action="store_true", help="pipelined EP async")
    parser.add_argument("--ep-async-mode", type=str, default=None, choices=['shallow', 'deep', 'interleaved', 'all'], help="async EP strategy")
    parser.add_argument("--ep-async-warm-up", type=int, default=0, help="Enable ep async warm-up feature (default: 0)")
    parser.add_argument("--ep-async-cool-down", type=int, default=0, help="Enable ep async cool-down feature (default: 0)")
    parser.add_argument("--strided-sync", type=int, default=0, help="Enable stride sync feature (default: 0)")
    parser.add_argument("--sp-async-warm-up", type=int, default=0, help="Enable sp async warm-up feature (default: 0)")
    parser.add_argument("--ep-async-noskip-step", type=int, default=1, help="comm stride")
    parser.add_argument("--ep-async-skip-strategy", type=str, default=None, choices=['low', 'high', 'rand'], help="comm strategy")
    
    parser.add_argument("--sp-legacy-cache", action="store_true", help="Use legacy SP cache implementation")
    parser.add_argument("--ep-score-use-latest", action="store_true", help="Use latest router score in EP")
    parser.add_argument("--single-img",action="store_true",help="Generate single image")
    
    args = parser.parse_args()
    
    # arguments check
    # args.para_mode = ParaMode(sp=args.sp, sp_async=args.sp_async, ep=args.ep, ep_async=args.ep_async)
    args.para_mode = ParaMode(sp=args.sp, sp_async=args.sp_async, ep=args.ep, 
                              ep_async=args.ep_async,num_sampling_steps= args.num_sampling_steps,
                              ep_async_warm_up=args.ep_async_warm_up,strided_sync =args.strided_sync,
                              sp_async_warm_up=args.sp_async_warm_up, ep_async_cool_down=args.ep_async_cool_down,
                              ep_async_mode=args.ep_async_mode,
                              )
    

    
    if not args.para_mode.sp_async and not args.para_mode.ep_async:
        assert not args.auto_gc, "auto_gc is only available when using asynchronous operations."
    if args.ep_async_warm_up > 0:
        assert args.ep_async, "warm up is only available whem using ep async."
    if args.ep_async_cool_down > 0:
        assert args.ep_async, "cool down is only available whem using ep async."
    if args.strided_sync > 0:
        assert args.ep_async, "Strided sync is only available whem using ep async."
    
    if args.model == "DiT-S/2":  
        assert args.num_sampling_steps > args.ep_async_warm_up, "Warm up steps should be smaller than the total steps of denoising."
    
    if args.sp_async_warm_up > 0:
        assert args.sp_async, "warm up is only available whem using sp async."
    # sp_legacy_cache True means using caches written by ourselves, False means using Distrifusion caches
    if args.sp_legacy_cache:
        assert args.sp_async, "Legacy cache is only available when using SP async."
    if args.ep_score_use_latest:
        assert args.ep_async, "Use latest score is only available when using EP async."
    
    if args.ep_async_pipeline:
        assert args.ep_async, "Pipeline is only available when using EP async."
    
    main(args)
