#!/bin/bash
#SBATCH --job-name=cut_unpaired
#SBATCH --output=/home/valeryb/logs/cut_unpaired.out
#SBATCH --error=/home/valeryb/logs/cut_unpaired.err
#SBATCH --partition=gpu-rtx
#SBATCH --gres=gpu:rtx6000:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=24:00:00


source /home/valeryb/.bashrc
conda activate mgdetect
cd /home/valeryb/contrastive-unpaired-translation


RUN_NAME=vindr_unpaired_bilateral_cut1

python train.py \
  --dataroot /home/management/projects/gilba/valeryb/data/vindr/images \
  --annotations_csv /home/management/projects/gilba/valeryb/data/vindr/finding_annotations.csv \
  --split training \
  --flip_right \
  --name $RUN_NAME \
  --CUT_mode CUT \
  --dataset_mode unpaired_bilateral \
  --gpu_ids 0,1 \
  --batch_size 2 \
  --num_threads 4 \
  --display_id 0 \
  --checkpoints_dir /home/management/projects/gilba/valeryb/cut_checkpoints \
  --use_wandb \
  --wandb_project cut-vindr \
  --wandb_run_name  $RUN_NAME
  #\
  #--continue_train --epoch 60 --epoch_count 61





