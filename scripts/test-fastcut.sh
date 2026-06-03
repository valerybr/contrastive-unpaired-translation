#!/bin/bash
#SBATCH --job-name=test_fastcut
#SBATCH --output=/home/valeryb/logs/fastcut_test.out
#SBATCH --error=/home/valeryb/logs/fastcut_test.err
#SBATCH --partition=gpu-rtx
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#
# Run test.py for all three trained checkpoints (paired / unpaired / scheduled)
# through the deterministic *paired* `bilateral` adapter, so both breasts are
# fixed and unshuffled and every model sees identical inputs (fair comparison).
# The generator's output depends only on the left (A) image; the paired right
# is the reference shown alongside it.
#
# Submit:   sbatch test_bilateral.sh
# Override any variable via --export, e.g.:
#   sbatch --export=ALL,EPOCH=80,NUM_TEST=500 test_bilateral.sh

set -euo pipefail

source /home/valeryb/.bashrc
conda activate mgdetect
cd /home/valeryb/contrastive-unpaired-translation

DATAROOT="${DATAROOT:-/home/management/projects/gilba/valeryb/data/vindr/images}"
ANNOTATIONS_CSV="${ANNOTATIONS_CSV:-/home/management/projects/gilba/valeryb/data/vindr/finding_annotations.csv}"
NAME1="${NAME1:-vindr_scheduled_ddp_g1}"
NAME2="${NAME2:-vindr_scheduled_ddp_g1_nce5}"
# NAME3="${SCHEDULED_NAME:-heduled_ddp_g5}"
SPLIT="${SPLIT:-test}"
EPOCH="${EPOCH:-latest}"
NUM_TEST="${NUM_TEST:-200}"
RESULTS_DIR="${RESULTS_DIR:-/home/management/projects/gilba/valeryb/fastcut_results}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-/home/management/projects/gilba/valeryb/fastcut_checkpoints}"
BILATERAL_SIZE="${BILATERAL_SIZE:-512 384}"

run_test() {
  local mode="$1" name="$2"
  echo "==> Testing dataset_mode=${mode} name=${name}"
  python test.py \
    --dataroot "${DATAROOT}" \
    --annotations_csv "${ANNOTATIONS_CSV}" \
    --dataset_mode "${mode}" \
    --name "${name}" \
    --model cut --CUT_mode CUT \
    --split "${SPLIT}" \
    --phase test \
    --epoch "${EPOCH}" \
    --num_test "${NUM_TEST}" \
    --results_dir "${RESULTS_DIR}" \
    --checkpoints_dir "${CHECKPOINTS_DIR}" \
    --bilateral_size ${BILATERAL_SIZE}
}

# All three use the paired `bilateral` adapter (deterministic, no shuffling on
# either breast); only the checkpoint --name differs.
run_test bilateral "${NAME1}"
run_test bilateral "${NAME2}"
#run_test bilateral "${NAME3}"

echo "Results written under ${RESULTS_DIR}/{${NAME1},${NAME2}/test_${EPOCH}/"
