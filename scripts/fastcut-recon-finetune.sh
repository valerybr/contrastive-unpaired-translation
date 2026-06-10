#!/bin/bash
#SBATCH --job-name=fastcut_recon_ft
#SBATCH --output=/home/valeryb/logs/fastcut_recon_ft.out
#SBATCH --error=/home/valeryb/logs/fastcut_recon_ft.err
#SBATCH --partition=gpu-v100
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00

# Paired-reconstruction finetuning for FastCUT (contralateral L-CC -> R-CC).
#
# Why: stock FastCUT collapses to identity (G(L) ~= L) because, after
# --flip_right, L and R look like one domain. This resumes a collapsed checkpoint
# on the *paired* `bilateral` adapter (B = the true same-study R) and adds an L1
# (and/or L2/MSE) reconstruction term so G learns the specific L->R mapping. GAN
# and NCE stay on so the discriminator re-sharpens what recon blurs.
#
# Override anything via the environment, e.g.:
#   BASE_NAME=vindr_fastcut_20260601 BASE_EPOCH=200 LAMBDA_L1=10 sbatch scripts/fastcut-recon-finetune.sh
#   LAMBDA_L1=0 LAMBDA_L2=10 sbatch ...   # MSE-only A/B run

source /home/valeryb/.bashrc
conda activate mgdetect
cd /home/valeryb/contrastive-unpaired-translation

# --- base run to finetune from -------------------------------------------------
BASE_NAME=${BASE_NAME:?set BASE_NAME to the collapsed FastCUT run to resume}
BASE_EPOCH=${BASE_EPOCH:-latest}     # checkpoint epoch to load (e.g. 200 or latest)
EPOCH_COUNT=${EPOCH_COUNT:-1}        # first epoch index of the finetune run

# --- reconstruction weights (L1 recommended; L2/MSE available) -----------------
LAMBDA_L1=${LAMBDA_L1:-10}
LAMBDA_L2=${LAMBDA_L2:-0}

# --- finetune optimization: low LR, short schedule -----------------------------
LR=${LR:-0.00005}
N_EPOCHS=${N_EPOCHS:-20}
N_EPOCHS_DECAY=${N_EPOCHS_DECAY:-10}

DATA_ROOT=${DATA_ROOT:-/home/management/projects/gilba/valeryb/data/vindr/images}
ANNOTATIONS=${ANNOTATIONS:-/home/management/projects/gilba/valeryb/data/vindr/finding_annotations.csv}
CKPT_DIR=${CKPT_DIR:-/home/management/projects/gilba/valeryb/cut_checkpoints}

RUN_NAME=${RUN_NAME:-${BASE_NAME}_reconft_$(date +%Y%m%d)}

# --continue_train loads from checkpoints/<name>/, so seed the new run's dir with
# the base run's checkpoints (so we finetune a copy and never clobber the base).
if [ "$RUN_NAME" != "$BASE_NAME" ]; then
  mkdir -p "$CKPT_DIR/$RUN_NAME"
  cp "$CKPT_DIR/$BASE_NAME/${BASE_EPOCH}_net_"*.pth "$CKPT_DIR/$RUN_NAME/" 2>/dev/null || \
    cp "$CKPT_DIR/$BASE_NAME/latest_net_"*.pth "$CKPT_DIR/$RUN_NAME/"
fi

python train.py \
  --dataroot "$DATA_ROOT" \
  --annotations_csv "$ANNOTATIONS" \
  --split training \
  --flip_right \
  --crop_width 360 \
  --masked_loss \
  --name "$RUN_NAME" \
  --CUT_mode FastCUT \
  --dataset_mode bilateral \
  --continue_train --epoch "$BASE_EPOCH" --epoch_count "$EPOCH_COUNT" \
  --lr "$LR" --n_epochs "$N_EPOCHS" --n_epochs_decay "$N_EPOCHS_DECAY" \
  --lambda_GAN 1 --lambda_NCE 10 \
  --lambda_L1 "$LAMBDA_L1" --lambda_L2 "$LAMBDA_L2" \
  --gpu_ids 0 \
  --batch_size 2 \
  --num_threads 4 \
  --display_id 0 \
  --checkpoints_dir "$CKPT_DIR" \
  --use_wandb \
  --wandb_project cut-windr

# Watch wandb recon_L1/recon_L2 (should fall) and the web/ samples: fake_B should
# start diverging from the input L toward R. If D_fake collapses, lower LAMBDA_L1.
