# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Samples a large number of images from a pre-trained UDT model using DDP.
Subsequently saves a .npz file that can be used to compute FID and other
evaluation metrics via the ADM repo: https://github.com/openai/guided-diffusion/tree/main/evaluations

For a simple single-GPU/CPU sampling script, see sample.py.
"""
import torch
import torch.distributed as dist
from diffusers.models import AutoencoderKL
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import math
import argparse
from samplers import euler_sampler, euler_maruyama_sampler
from models.UDT import UDT_models


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("true", "1", "yes", "y"):
        return True
    elif v.lower() in ("false", "0", "no", "n"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
    
def create_npz_from_sample_folder(sample_dir, target_count=50_000, num=50_170):
    """
    Builds a single .npz file from a folder of .png samples.
    - Tries up to `num` samples
    - Skips missing files
    - Stops when `target_count` samples are collected
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        img_path = os.path.join(sample_dir, f"{i:06d}.png")
        if not os.path.exists(img_path):
            continue  

        try:
            sample_pil = Image.open(img_path)
            sample_np = np.asarray(sample_pil).astype(np.uint8)
            samples.append(sample_np)
        except Exception as e:
            print(f"Warning: failed to load {img_path}: {e}")
            continue

        if len(samples) >= target_count:
            print(f"Collected {target_count} samples. Stopping early.")
            break

    samples = np.stack(samples)
    assert samples.shape[0] == target_count, f"Collected {samples.shape[0]} samples, expected {target_count}."

    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")

    return npz_path

def main(args):
    """
    Run sampling.
    """
    torch.backends.cuda.matmul.allow_tf32 = args.tf32  # True: fast but may lead to some small numerical differences
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    # Setup DDP:cd
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Load model:
    latent_size = args.resolution // 8
    block_kwargs = {"use_swiglu": args.use_swiglu, "use_rmsnorm": args.use_rmsnorm,
                    "qk_norm": args.qk_norm, "attn_drop": args.attn_drop,
                    "proj_drop": args.proj_drop, "use_partial_rope" : args.use_partial_rope}

    model = UDT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        num_seg=args.num_seg,               
        r_type=args.r_type,
        **block_kwargs
    )
        
    # Auto-download a pre-trained model or load a custom UDT checkpoint from train.py:
    ckpt_path = args.ckpt
    with torch.serialization.safe_globals([argparse.Namespace]):     
        state_dict = torch.load(ckpt_path, map_location=f'cuda:{device}')['ema']

    # Filter out REPA projectors as they are only used during training, not generation
    filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith("projectors")}
    model.load_state_dict(filtered_state_dict, strict=False)
    model = model.to(device)
    model.eval()   
    
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    assert args.cfg_scale >= 1.0, "In almost all cases, cfg_scale be >= 1.0"

    # Create folder to save samples:
    model_string_name = args.model.replace("/", "-")
    ckpt_string_name = os.path.basename(args.ckpt).replace(".pt", "") if args.ckpt else "pretrained"
    folder_name = (
    f"{args.udt_mode}-"             
    f"{model_string_name}-"
    f"{ckpt_string_name}-"
    f"size-{args.resolution}-"
    f"vae-{args.vae}-"
    f"cfg-{args.cfg_scale}-"
    f"seed-{args.global_seed}-"
    f"{args.mode}-"
    f"ghigh-{args.guidance_high}"  
    )

    sample_folder_dir = f"{args.sample_dir}/{folder_name}"
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
        print(f"Total number of images that will be sampled: {total_samples}")
        print(f"UDT Parameters: {sum(p.numel() for p in model.parameters()):,}")
    assert total_samples % dist.get_world_size() == 0, "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar
    total = 0
    for _ in pbar:
        # Sample inputs:
        z = torch.randn(n, model.in_channels, latent_size, latent_size, device=device)
        y = torch.randint(0, args.num_classes, (n,), device=device)

        ## Use this to generate a specific index
        # indices = torch.tensor([980], device=device)
        # y = indices.repeat(n)

        # Sample images:
        sampling_kwargs = dict(
            model=model, 
            latents=z,
            y=y,
            num_steps=args.num_steps, 
            heun=args.heun,
            cfg_scale=args.cfg_scale,
            guidance_low=args.guidance_low,
            guidance_high=args.guidance_high,
            path_type=args.path_type,
            timestep_shift=args.timestep_shift,
        )
        with torch.no_grad():
            if args.mode == "sde":
                samples = euler_maruyama_sampler(**sampling_kwargs).to(torch.float32)
            elif args.mode == "ode":
                samples = euler_sampler(**sampling_kwargs).to(torch.float32)
            else:
                raise NotImplementedError()

            latents_scale = torch.tensor([0.18215, 0.18215, 0.18215, 0.18215,]).view(1, 4, 1, 1).to(device)
            latents_bias = -torch.tensor([0., 0., 0., 0.,]).view(1, 4, 1, 1).to(device)
            samples = vae.decode((samples -  latents_bias) / latents_scale).sample
            samples = (samples + 1) / 2.
            samples = torch.clamp(
                255. * samples, 0, 255
                ).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

            # Save samples to disk as individual .png files
            for i, sample in enumerate(samples):
                index = i * dist.get_world_size() + rank + total
                Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")
        total += global_batch_size

    # Make sure all processes have finished saving their samples before attempting to convert to .npz
    dist.barrier()
    if rank == 0:
        create_npz_from_sample_folder(sample_folder_dir, args.num_fid_samples)
        print("Done.")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # seed
    parser.add_argument("--global-seed", type=int, default=0)

    # precision
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True,
                        help="By default, use TF32 matmuls. This massively accelerates sampling on Ampere GPUs.")

    # logging/saving:
    parser.add_argument("--ckpt", type=str, default=None, help="Optional path to a SiT checkpoint.")
    parser.add_argument("--sample_dir", type=str, default="samples")

    # model
    # parser.add_argument("--model", type=str, choices=list(SiT_models.keys()), default="SiT-XL/2")
    parser.add_argument("--model", type=str)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--resolution", type=int, choices=[256, 512], default=256)

    # vae
    parser.add_argument("--vae",  type=str, choices=["ema", "mse"], default="ema")

    # number of samples
    parser.add_argument("--per-proc-batch-size", type=int, default=32)
    parser.add_argument("--num-fid-samples", type=int, default=50_000)

    # sampling related hyperparameters
    parser.add_argument("--mode", type=str, default="sde")
    parser.add_argument("--cfg-scale",  type=float, default=1.0)
    parser.add_argument("--path-type", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--num-steps", type=int, default=250)
    parser.add_argument("--heun", action=argparse.BooleanOptionalAction, default=False) # only for ode
    parser.add_argument("--guidance-low", type=float, default=0.0)
    parser.add_argument("--guidance-high", type=float, default=1.0)
    parser.add_argument("--timestep_shift", type=float, default=1.0) # Optional  

    parser.add_argument("--name_add", type=str, default='') 

    # UDT models & Token Merge
    parser.add_argument("--udt_mode", default="udt+", type=str, choices=["udt", "udt+"])
    parser.add_argument("--num_seg", type=int, default=112)
    parser.add_argument("--r_type", default="parall", type=str, choices=["parall", "random"])

    # Architectural Optimization 
    # (Defaults for udt+; if you set "--udt_mode" to "udt", the configurations below will updated automatically)    
    parser.add_argument("--use_swiglu", type=str2bool, default=True)
    parser.add_argument("--use_partial_rope", type=str2bool, default=True)
    parser.add_argument("--use_rmsnorm", type=str2bool, default=False)  # Optional
    parser.add_argument("--qk_norm", type=str2bool, default=False)      # Optional
    parser.add_argument("--attn_drop", type=float, default=0.0)         # Optional
    parser.add_argument("--proj_drop", type=float, default=0.0)         # Optional

    args = parser.parse_args()

    # baseline UDT
    if args.udt_mode == "udt":
        args.use_swiglu = False
        args.use_partial_rope = False
        args.weighting = "uniform"
    
        
    main(args)
