
EXP_DIR="UDT"
EXP_NAME="UDT+-XL/2-imagenet256"

ckpt=epoch-79
CFG=1.0          # 1.0 (w/o CFG)
GUIDANCE=1.0
torchrun --nnodes=1 --nproc_per_node=4 generate_UDT.py \
  --model "UDT-XL/2" \
  --num-fid-samples 50000 \
  --ckpt ./${EXP_DIR}/exps/${EXP_NAME}/checkpoints/${ckpt}.pt \
  --resolution 256 \
  --per-proc-batch-size=64 \
  --mode=sde \
  --num-steps=250 \
  --cfg-scale=${CFG} \
  --guidance-high=${GUIDANCE} \
  --udt_mode udt+ \
  --num_seg 112 \
  --sample_dir ./${EXP_DIR}/samples 


sample_dir=SAMPLE.npz
python evaluator.py ./evaluator/VIRTUAL_imagenet256_labeled.npz ${sample_dir}

   