import argparse
import copy
from copy import deepcopy
import logging
import os
from pathlib import Path
from collections import OrderedDict
import json

import torch
import torch.utils.checkpoint
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from loss.loss import SILoss                         # Loss
from dataset import CustomDataset               # Dataset
from diffusers.models import AutoencoderKL      # Model(VAE)
from models.UDT import UDT_models               # Model(UDT)

import wandb
import math
from torchvision.utils import make_grid
import time
import datetime
logger = get_logger(__name__)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("true", "1", "yes", "y"):
        return True
    elif v.lower() in ("false", "0", "no", "n"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
    
def array2grid(x):
    nrow = round(math.sqrt(x.size(0)))
    x = make_grid(x.clamp(0, 1), nrow=nrow, value_range=(0, 1))
    x = x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return x


@torch.no_grad()
def sample_posterior(moments, latents_scale=1., latents_bias=0.):
    device = moments.device
    
    mean, std = torch.chunk(moments, 2, dim=1)
    z = mean + std * torch.randn_like(mean)
    z = (z * latents_scale + latents_bias) 
    return z 

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):    
    # set accelerator
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
        )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    save_dir = os.path.join(args.output_dir, args.exp_name)
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        save_dir = os.path.join(args.output_dir, args.exp_name)
        os.makedirs(save_dir, exist_ok=True)
        args_dict = vars(args)
        # Save to a JSON file
        json_dir = os.path.join(save_dir, "args.json")
        with open(json_dir, 'w') as f:
            json.dump(args_dict, f, indent=4)
        checkpoint_dir = f"{save_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(save_dir)
        logger.info(f"Experiment directory created at {save_dir}")
    device = accelerator.device
    if torch.backends.mps.is_available():
        accelerator.native_amp = False    
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)
    
    # Create model
    assert args.resolution % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.resolution // 8

    
    block_kwargs = {"use_swiglu": args.use_swiglu, "use_rmsnorm": args.use_rmsnorm,
                    "qk_norm": args.qk_norm, "attn_drop": args.attn_drop,
                    "proj_drop": args.proj_drop, "use_partial_rope" : args.use_partial_rope}

    model = UDT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        num_seg=args.num_seg,               
        r_type=args.r_type,
        log_dir=save_dir,
        **block_kwargs
    )
    
    model = model.to(device)
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(device)
    requires_grad(ema, False)

    latents_scale = torch.tensor([0.18215, 0.18215, 0.18215, 0.18215]).view(1, 4, 1, 1).to(device)
    latents_bias = torch.tensor([0., 0., 0., 0.]).view(1, 4, 1, 1).to(device)
    if accelerator.is_main_process:
        logger.info(f"UDT Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss function
    loss_fn = SILoss(
        prediction=args.prediction,
        path_type=args.path_type, 
        weighting=args.weighting,
        P_mean = args.P_mean,
        P_std = args.P_std,
    )

    # Optimizer (Default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4)
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )    
    
    # Data Loading
    train_dataset = CustomDataset(args.data_dir)
    local_batch_size = int(args.batch_size // accelerator.num_processes)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=4
    )
    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(train_dataset):,} images ({args.data_dir})")
    
    # Prepare models for training:
    update_ema(ema, model, decay=0)   
    model.train()   
    ema.eval()   
    
    # Resume, if needed
    global_step = 0
    resume_epoch = 0
    if args.resume_epoch > 0:
        ckpt_name = f'epoch-{args.resume_epoch}.pt'

        with torch.serialization.safe_globals([argparse.Namespace]):
            ckpt = torch.load(
                f'{os.path.join(args.output_dir, args.exp_name)}/checkpoints/{ckpt_name}',
                map_location='cpu',
            )

        model.load_state_dict(ckpt['model'])
        ema.load_state_dict(ckpt['ema'])
        optimizer.load_state_dict(ckpt['opt'])
        global_step = ckpt['steps']
        resume_epoch = ckpt['epoch'] + 1


    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    if accelerator.is_main_process:
        tracker_config = vars(copy.deepcopy(args))
        accelerator.init_trackers(
            project_name="UDT", 
            config=tracker_config,
            init_kwargs={
                "wandb": {"name": f"{args.exp_name}"}
            },
        )
        
    if args.max_train_steps == 0:
        args.max_train_steps = (len(train_dataset) * args.epochs) // (args.batch_size * args.gradient_accumulation_steps)

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    # Visualization 
    sample_batch_size = 64 // accelerator.num_processes
    gt_xs, _ = next(iter(train_dataloader))
    gt_xs = gt_xs[:sample_batch_size]
    gt_xs = sample_posterior(gt_xs.to(device), latents_scale=latents_scale, latents_bias=latents_bias)
    ys = torch.randint(1000, size=(sample_batch_size,), device=device)
    ys = ys.to(device)
    
    n = ys.size(0)  # Create sampling noise
    xT = torch.randn((n, 4, latent_size, latent_size), device=device)

    # Training Loop
    start_time = time.time()    
    for epoch in range(resume_epoch, args.epochs):
        model.train()
        for x, y in train_dataloader:
            x = x.squeeze(dim=1).to(device)
            y = y.to(device)
            labels = y
            with torch.no_grad():
                x = sample_posterior(x, latents_scale=latents_scale, latents_bias=latents_bias)

            with accelerator.accumulate(model):
                model_kwargs = dict(y=labels)                
                loss = loss_fn(model, x, model_kwargs)
                loss = loss.mean()
                    
                # optimization
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = model.parameters()
                    grad_norm = accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                if accelerator.sync_gradients:
                    update_ema(ema, model)  

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1       

            if args.sampling_steps > 0:
                if (global_step == 1 or (global_step % args.sampling_steps == 0 and global_step > 0)):
                    from samplers import euler_sampler
                    with torch.no_grad():
                        samples = euler_sampler(
                            model, xT, ys,
                            num_steps=50, 
                            cfg_scale=4.0,
                            guidance_low=0.,
                            guidance_high=1.,
                            path_type=args.path_type,
                            heun=False,
                        ).to(torch.float32)
                        samples = vae.decode((samples -  latents_bias) / latents_scale).sample
                        gt_samples = vae.decode((gt_xs - latents_bias) / latents_scale).sample
                        samples = (samples + 1) / 2.
                        gt_samples = (gt_samples + 1) / 2.
                    out_samples = accelerator.gather(samples.to(torch.float32))
                    gt_samples = accelerator.gather(gt_samples.to(torch.float32))
                    accelerator.log({"samples": wandb.Image(array2grid(out_samples)),
                                    #  "gt_samples": wandb.Image(array2grid(gt_samples))
                                    })
                    logging.info("Generating EMA samples done.")

            logs = {"loss": accelerator.gather(loss).mean().detach().item(),
                    "global_step" : global_step, "epoch": epoch,
                    }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

        # save checkpoint in terms of epoch
        if (epoch+1) % args.checkpointing_epochs == 0:
            if accelerator.is_main_process:
                checkpoint = {
                    "model": model.module.state_dict(),
                    "ema": ema.state_dict(),
                    "opt": optimizer.state_dict(),
                    "args": args,
                    "epoch": epoch,
                    "steps": global_step,
                }
                checkpoint_path = f"{checkpoint_dir}/epoch-{epoch}.pt"
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")

    model.eval()   
    
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("Done!")
    accelerator.end_training()

    if accelerator.is_main_process:
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info("Training time: %s", total_time_str)

        save_dir = os.path.join(args.output_dir, args.exp_name)
        txt_path = os.path.join(save_dir, "training_time.txt")
        with open(txt_path, "a") as f:
            f.write(f"Training time: {total_time_str}\n")

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Training")

    # logging:
    parser.add_argument("--output-dir", type=str, default="exps")
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--logging-dir", type=str, default="logs")
    parser.add_argument("--report-to", type=str, default="wandb")
    parser.add_argument("--sampling-steps", type=int, default=0)
    parser.add_argument("--resume-epoch", type=int, default=0)

    # model
    parser.add_argument("--model", type=str)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--encoder-depth", type=int, default=8)

    # dataset
    parser.add_argument("--data-dir", type=str, default="../data/imagenet256")
    parser.add_argument("--resolution", type=int, choices=[256, 512], default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--subset", action=argparse.BooleanOptionalAction, default=False)

    # precision
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--mixed-precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])

    # optimization
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--max-train-steps", type=int, default=0)            # Optional
    parser.add_argument("--checkpointing-steps", type=int, default=100_000)  # Optional (checkpoints are saved by epoch in this script)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--checkpointing_epochs", type=int, default=10)
    
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--adam-beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam-beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam-weight-decay", type=float, default=0., help="Weight decay to use.")
    parser.add_argument("--adam-epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")

    # seed
    parser.add_argument("--seed", type=int, default=0)

    # cpu
    parser.add_argument("--num-workers", type=int, default=8)

    # loss
    parser.add_argument("--path-type", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--prediction", type=str, default="v", choices=["v"]) 
    parser.add_argument("--cfg-prob", type=float, default=0.1)
    parser.add_argument("--weighting", default="lognormal", type=str, help="uniform or lognormal")
    parser.add_argument("--P_mean", type=float, default=0.0)
    parser.add_argument("--P_std", type=float, default=1.0)

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


    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
        
    return args

if __name__ == "__main__":
    args = parse_args()
    
    # baseline UDT
    if args.udt_mode == "udt":
        args.use_swiglu = False
        args.use_partial_rope = False
        args.weighting = "uniform"
        
    main(args)
