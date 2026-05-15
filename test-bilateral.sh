#!/bin/bash
#SBATCH --job-name=cut_test_bilateral
#SBATCH --output=/home/valeryb/logs/cut_test_bilateral.out
#SBATCH --error=/home/valeryb/logs/cut_test_bilateral.err
#SBATCH --partition=gpu-rtx
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#
# Run test.py against both bilateral dataset adapters (paired + unpaired).
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
PAIRED_NAME="${PAIRED_NAME:-vindr_bilateral_ddp2}"
UNPAIRED_NAME="${UNPAIRED_NAME:-vindr_unpaired_bilateral_ddp2}"
SPLIT="${SPLIT:-test}"
EPOCH="${EPOCH:-latest}"
NUM_TEST="${NUM_TEST:-200}"
RESULTS_DIR="${RESULTS_DIR:-/home/management/projects/gilba/valeryb/cut_results}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-/home/management/projects/gilba/valeryb/cut_checkpoints}"
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

run_test bilateral          "${PAIRED_NAME}"
run_test unpaired_bilateral "${UNPAIRED_NAME}"

echo "Results written under ${RESULTS_DIR}/{${PAIRED_NAME},${UNPAIRED_NAME}}/test_${EPOCH}/"
