export WANDB_PROJECT="UDT"

EXP_DIR="UDT"
EXP_NAME="UDT+-XL-2-imagenet256"

# Use train_UDT_REPA.py instead of train_UDT.py to enable REPA training
accelerate launch --num_processes 4 train_UDT.py \
  --report-to="wandb" \
  --allow-tf32 \
  --mixed-precision="fp16" \
  --model="UDT-XL/2" \
  --gradient-accumulation-steps 1 \
  --batch-size 256 \
  --num_seg 112 \
  --epochs 80 \
  --udt_mode udt+ \
  --output-dir="./${EXP_DIR}/exps" \
  --exp-name=${EXP_NAME} \
  --data-dir=/data_dir
  
