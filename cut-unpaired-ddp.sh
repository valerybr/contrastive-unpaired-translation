#!/bin/bash
#SBATCH --job-name=cut_unpaired_ddp
#SBATCH --output=/home/valeryb/logs/cut_unpaired_ddp.out
#SBATCH --error=/home/valeryb/logs/cut_unpaired_ddp.err
#SBATCH --partition=gpu-rtx
#SBATCH --gres=gpu:rtx6000:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00


source /home/valeryb/.bashrc
conda activate mgdetect
cd /home/valeryb/contrastive-unpaired-translation

# wandb auth: set WANDB_API_KEY in ~/.bashrc or export it here.
# Compute nodes may not see ~/.netrc, so prefer the env var.
export WANDB_API_KEY="${WANDB_API_KEY:?WANDB_API_KEY is not set; export it before sbatch}"

# DDP-specific tunables
export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=4
# Bump NCCL collective timeout from 10 min → 30 min to tolerate slow NFS
# writes (rank 0 saves checkpoints to a shared filesystem while other ranks
# wait at the post-save barrier).
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800

RUN_NAME=vindr_unpaired_bilateral_ddp1

# nproc_per_node should match the number of GPUs requested above.
# --batch_size below is PER-RANK; effective global batch = batch_size * nproc_per_node.
# --gpu_ids is ignored under torchrun (each rank binds to its LOCAL_RANK device).
torchrun --standalone --nnodes=1 --nproc_per_node=4 train.py \
  --dataroot /home/management/projects/gilba/valeryb/data/vindr/images \
  --annotations_csv /home/management/projects/gilba/valeryb/data/vindr/finding_annotations.csv \
  --split training \
  --flip_right \
  --name $RUN_NAME \
  --CUT_mode CUT \
  --dataset_mode unpaired_bilateral \
  --batch_size 2 \
  --num_threads 4 \
  --display_id 0 \
  --checkpoints_dir /home/management/projects/gilba/valeryb/cut_checkpoints \
  --use_wandb \
  --wandb_project cut-vindr \
  --wandb_run_name $RUN_NAME
  #\
  #--continue_train --epoch 60 --epoch_count 61
