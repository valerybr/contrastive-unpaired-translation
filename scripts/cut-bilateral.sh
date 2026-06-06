#!/bin/bash
#SBATCH --job-name=cut_bilateral
#SBATCH --output=/home/valeryb/logs/cut_bilateral.out
#SBATCH --error=/home/valeryb/logs/cut_bilateral.err
#SBATCH --partition=gpu-v100
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00


source /home/valeryb/.bashrc
conda activate mgdetect
cd /home/valeryb/contrastive-unpaired-translation

RUN_NAME=vindr_bilateral_cut1_$(date +%Y%m%d)

python train.py \
  --dataroot /home/management/projects/gilba/valeryb/data/vindr/images \
  --annotations_csv /home/management/projects/gilba/valeryb/data/vindr/finding_annotations.csv \
  --split training \
  --flip_right \
  --name $RUN_NAME\
  --CUT_mode CUT \
  --dataset_mode bilateral \
  --gpu_ids 0\
  --batch_size 2 \
  --num_threads 4 \
  --display_id 0 \
  --checkpoints_dir /home/management/projects/gilba/valeryb/cut_checkpoints \
  --use_wandb \
  --wandb_project cut-windr 
#  --continue_train --epoch 60 --epoch_count 61





