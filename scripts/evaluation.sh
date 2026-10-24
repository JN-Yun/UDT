
EXP_NAME="UDT+-XL-2-imagenet256"

ckpt=epoch-79
CFG=1.0          # 1.0 (w/o CFG)
GUIDANCE=1.0
torchrun --nnodes=1 --nproc_per_node=4 generate_UDT.py \
  --model "UDT-XL/2" \
  --num-fid-samples 50000 \
  --ckpt ./exps/${EXP_NAME}/checkpoints/${ckpt}.pt \
  --resolution 256 \
  --per-proc-batch-size=64 \
  --mode=sde \
  --num-steps=250 \
  --cfg-scale=${CFG} \
  --guidance-high=${GUIDANCE} \
  --udt_mode udt+ \
  --num_seg 112 \
  --sample_dir ./samples


# .npz written by generate_UDT.py: <sample_dir>/<udt_mode>-<model>-<ckpt>-size-<resolution>-vae-<vae>-cfg-<cfg>-seed-<seed>-<mode>-ghigh-<guidance_high>.npz
sample_npz=./samples/udt+-UDT-XL-2-${ckpt}-size-256-vae-ema-cfg-${CFG}-seed-0-sde-ghigh-${GUIDANCE}.npz
python evaluator/evaluator.py ./evaluator/VIRTUAL_imagenet256_labeled.npz ${sample_npz}
