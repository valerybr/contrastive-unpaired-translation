#!/bin/bash
# Run test.py against both bilateral dataset adapters.
#
# Edit the variables below or override via env, e.g.:
#   DATAROOT=/path/to/pngs ANNOTATIONS_CSV=/path/to/annotations.csv \
#   PAIRED_NAME=bilateral_CUT UNPAIRED_NAME=unpaired_bilateral_CUT \
#   ./test_bilateral.sh

set -euo pipefail

DATAROOT="${DATAROOT:-./datasets/bilateral_pngs}"
ANNOTATIONS_CSV="${ANNOTATIONS_CSV:-./datasets/finding_annotations.csv}"
PAIRED_NAME="${PAIRED_NAME:-bilateral_CUT}"
UNPAIRED_NAME="${UNPAIRED_NAME:-unpaired_bilateral_CUT}"
SPLIT="${SPLIT:-test}"
EPOCH="${EPOCH:-latest}"
NUM_TEST="${NUM_TEST:-200}"
RESULTS_DIR="${RESULTS_DIR:-./results}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-./checkpoints}"
BILATERAL_SIZE="${BILATERAL_SIZE:-512 384}"

cd "$(dirname "$0")"

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
