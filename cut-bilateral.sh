#!/bin/bash
#SBATCH --job-name=cut_bilateral
#SBATCH --output=/home/valeryb/logs/cut_bilateral.out
#SBATCH --error=/home/valeryb/logs/cut_bilateral.err
#SBATCH --partition=gpu-rtx
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00


source /home/valeryb/.bashrc
conda activate mgdetect
cd /home/valeryb/contrastive-unpaired-translation


python train.py \
  --dataroot /home/management/projects/gilba/valeryb/data/vindr/images \
  --annotations_csv /home/management/projects/gilba/valeryb/data/vindr/finding_annotations.csv \
  --split training \
  --flip_right \
  --name vindr_bilateral_CUT \
  --CUT_mode CUT \
  --dataset_mode bilateral \
  --batch_size 1 \
  --num_threads 2 \
  --display_id 0 \
  --checkpoints_dir /home/management/projects/gilba/valeryb/cut_checkpoints \
  --continue_train --epoch 60 --epoch_count 61





