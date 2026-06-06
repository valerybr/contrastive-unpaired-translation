#!/bin/bash
#SBATCH --job-name=fastcut_scheduled_ddp_masks
#SBATCH --output=/home/valeryb/logs/fastcut_ddp.out
#SBATCH --error=/home/valeryb/logs/fastcut_ddp.err
#SBATCH --partition=gpu-rtx
#SBATCH --gres=gpu:rtx6000:6
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00


source /home/valeryb/.bashrc
conda activate mgdetect
cd /home/valeryb/contrastive-unpaired-translation

# wandb auth: set WANDB_API_KEY in ~/.bashrc or export it here.
# Compute nodes may not see ~/.netrc, so prefer the env var.
#export WANDB_API_KEY="${WANDB_API_KEY:?WANDB_API_KEY is not set; export it before sbatch}"

# DDP-specific tunables
export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=4
# Bump NCCL collective timeout from 10 min → 30 min to tolerate slow NFS
# writes (rank 0 saves checkpoints to a shared filesystem while other ranks
# wait at the post-save barrier).
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800

RUN_NAME=vindr_scheduled_ddp_g1_nce5_masks_$(date +%Y%m%d)

# Curriculum: start at 50% random pairing, ramp down to 0% (fully paired)
# over the first 80 epochs. Past the schedule, p holds at the last value
# (0.0), so the remaining epochs train as a paired dataset.
PAIR_SCHEDULE="50:0.7,50:0.5,50:0.3,50:0.0"

# nproc_per_node should match the number of GPUs requested above.
# --batch_size below is PER-RANK; effective global batch = batch_size * nproc_per_node.
# --gpu_ids is ignored under torchrun (each rank binds to its LOCAL_RANK device).
torchrun --standalone --nnodes=1 --nproc_per_node=6 train.py \
  --dataroot /home/management/projects/gilba/valeryb/data/vindr-masks/images \
  --annotations_csv /home/management/projects/gilba/valeryb/data/vindr-masks/finding_annotations.csv \
  --split training \
  --flip_right \
  --name $RUN_NAME \
  --CUT_mode fastcut \
  --dataset_mode scheduled_bilateral \
  --pair_schedule "$PAIR_SCHEDULE" \
  --pair_schedule_seed 0 \
  --batch_size 1 \
  --num_threads 4 \
  --display_id 0 \
  --checkpoints_dir /home/management/projects/gilba/valeryb/fastcut_checkpoints \
  --use_wandb \
  --wandb_project fastcut-vindr \
  --wandb_run_name $RUN_NAME \
  --lambda_GAN 1 \
  --lambda_NCE 5 \
  --flip_equivariance False \
  --masked_loss True


#  --continue_train --epoch 251 --epoch_count 252
