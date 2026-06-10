#!/bin/bash
#SBATCH --job-name=fastcut_recon_ft
#SBATCH --output=/home/valeryb/logs/fastcut_ddp.out
#SBATCH --error=/home/valeryb/logs/fastcut_ddp.err
#SBATCH --partition=gpu-rtx
#SBATCH --gres=gpu:rtx6000:6
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
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


# DDP-specific tunables
export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=4
# Bump NCCL collective timeout from 10 min → 30 min to tolerate slow NFS
# writes (rank 0 saves checkpoints to a shared filesystem while other ranks
# wait at the post-save barrier).
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800



# --- base run to finetune from -------------------------------------------------
BASE_NAME=${BASE_NAME:-vindr_scheduled_ddp_g1_nce5_m_bd_20260607}
BASE_EPOCH=${BASE_EPOCH:-latest}     # checkpoint epoch to load (e.g. 200 or latest)
EPOCH_COUNT=${EPOCH_COUNT:-201}        # first epoch index of the finetune run

# --- reconstruction weights (L1 recommended; L2/MSE available) -----------------
LAMBDA_L1=${LAMBDA_L1:-5}
LAMBDA_L2=${LAMBDA_L2:-0}

# --- finetune optimization: low LR, short schedule -----------------------------
# NOTE: train.py loops `range(epoch_count, n_epochs + n_epochs_decay + 1)`, so
# --n_epochs/--n_epochs_decay are ABSOLUTE epoch numbers, not extra epochs. We
# express the finetune length as *additional* epochs (FT_*) and derive the
# absolute values from EPOCH_COUNT, so the loop is never empty when resuming.
LR=${LR:-0.00005}
FT_EPOCHS=${FT_EPOCHS:-50}              # full-LR finetune epochs
FT_EPOCHS_DECAY=${FT_EPOCHS_DECAY:-30}  # LR-decay finetune epochs
N_EPOCHS=$(( EPOCH_COUNT + FT_EPOCHS - 1 ))   # absolute epoch at which decay starts
N_EPOCHS_DECAY=${FT_EPOCHS_DECAY}
echo "[finetune] epochs ${EPOCH_COUNT}..$(( N_EPOCHS + N_EPOCHS_DECAY )) " \
     "(${FT_EPOCHS} full-LR + ${FT_EPOCHS_DECAY} decay); n_epochs=${N_EPOCHS} n_epochs_decay=${N_EPOCHS_DECAY}"

DATA_ROOT=${DATA_ROOT:-/home/management/projects/gilba/valeryb/data/vindr-masks/images}
ANNOTATIONS=${ANNOTATIONS:-/home/management/projects/gilba/valeryb/data/vindr-masks/finding_annotations.csv}
CKPT_DIR=${CKPT_DIR:-/home/management/projects/gilba/valeryb/fastcut_checkpoints}

RUN_NAME=${RUN_NAME:-${BASE_NAME}_reconft_l1${LAMBDA_L1}_l2${LAMBDA_L2}_$(date +%Y%m%d)}

# --continue_train loads from checkpoints/<name>/, so seed the new run's dir with
# the base run's checkpoints (so we finetune a copy and never clobber the base).
if [ "$RUN_NAME" != "$BASE_NAME" ]; then
  mkdir -p "$CKPT_DIR/$RUN_NAME"
  cp "$CKPT_DIR/$BASE_NAME/${BASE_EPOCH}_net_"*.pth "$CKPT_DIR/$RUN_NAME/" 2>/dev/null || \
    cp "$CKPT_DIR/$BASE_NAME/latest_net_"*.pth "$CKPT_DIR/$RUN_NAME/"
fi

torchrun --standalone --nnodes=1 --nproc_per_node=6 train.py \
  --dataroot "$DATA_ROOT" \
  --annotations_csv "$ANNOTATIONS" \
  --split training \
  --flip_right \
  --masked_loss \
  --name "$RUN_NAME" \
  --CUT_mode FastCUT \
  --dataset_mode bilateral \
  --lr "$LR" --n_epochs "$N_EPOCHS" --n_epochs_decay "$N_EPOCHS_DECAY" \
  --lambda_GAN 1 --lambda_NCE 5 \
  --lambda_L1 "$LAMBDA_L1" --lambda_L2 "$LAMBDA_L2" \
  --batch_size 1 \
  --num_threads 4 \
  --display_id 0 \
  --checkpoints_dir "$CKPT_DIR" \
  --use_wandb \
  --wandb_project fastcut-vindr \
  --wandb_run_name $RUN_NAME \
  --bidirectional \
  --continue_train --epoch "$BASE_EPOCH" --epoch_count "$EPOCH_COUNT" 

# Watch wandb recon_L1/recon_L2 (should fall) and the web/ samples: fake_B should
# start diverging from the input L toward R. If D_fake collapses, lower LAMBDA_L1.
