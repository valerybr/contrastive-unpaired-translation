#!/bin/bash
#SBATCH --job-name=cut_unpaired
#SBATCH --output=/home/valeryb/logs/cut_unpaired.out
#SBATCH --error=/home/valeryb/logs/cut_unpaired.err
#SBATCH --partition=gpu-rtx
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00


source /home/valeryb/.bashrc
conda activate mgdetect
cd /home/valeryb/contrastive-unpaired-translation

# wandb auth: set WANDB_API_KEY in ~/.bashrc or export it here.
# Compute nodes may not see ~/.netrc, so prefer the env var.
export WANDB_API_KEY="${WANDB_API_KEY:?WANDB_API_KEY is not set; export it before sbatch}"


python train.py \
  --dataroot /home/management/projects/gilba/valeryb/data/vindr/images \
  --annotations_csv /home/management/projects/gilba/valeryb/data/vindr/finding_annotations.csv \
  --split training \
  --flip_right \
  --name vindr_unpaired_bilateral_CUT \
  --CUT_mode CUT \
  --dataset_mode unpaired_bilateral \
  --batch_size 1 \
  --num_threads 2 \
  --display_id 0 \
  --checkpoints_dir /home/management/projects/gilba/valeryb/cut_checkpoints \
  --use_wandb \
  --wandb_project cut-vindr \
  --wandb_run_name vindr_unpaired_bilateral_cut1
  #\
  #--continue_train --epoch 60 --epoch_count 61





