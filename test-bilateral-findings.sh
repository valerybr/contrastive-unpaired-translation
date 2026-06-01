#!/bin/bash
#SBATCH --job-name=cut_test_findings
#SBATCH --output=/home/valeryb/logs/cut_test_findings.out
#SBATCH --error=/home/valeryb/logs/cut_test_findings.err
#SBATCH --partition=gpu-rtx
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#
# Test all three trained checkpoints (paired / unpaired / scheduled) on CC
# pairs WHERE THE RIGHT BREAST HAS A FINDING, via the deterministic paired
# `bilateral` adapter (--finding_filter right_finding). Both breasts are fixed
# and unshuffled, so every model sees identical inputs.
#
# Direction note: with --direction AtoB (default, as the models were trained)
# the generator translates A = LEFT CC and the with-finding RIGHT CC is the
# reference (B) shown alongside. Set DIRECTION=BtoA to instead feed the
# with-finding right image through G (out-of-distribution for a left->right
# generator, but useful to check whether findings survive translation).
#
# Results land under a separate dir (cut_results_findings) so they don't clobber
# the normal-anatomy test outputs.
#
# Submit:   sbatch test-bilateral-findings.sh
# Override: sbatch --export=ALL,EPOCH=80,NUM_TEST=500,DIRECTION=BtoA test-bilateral-findings.sh

set -euo pipefail

source /home/valeryb/.bashrc
conda activate mgdetect
cd /home/valeryb/contrastive-unpaired-translation

DATAROOT="${DATAROOT:-/home/management/projects/gilba/valeryb/data/vindr/images}"
ANNOTATIONS_CSV="${ANNOTATIONS_CSV:-/home/management/projects/gilba/valeryb/data/vindr/finding_annotations.csv}"
NAME1="${PAIRED_NAME:-vindr_scheduled_ddp_g1}"
NAME2="${UNPAIRED_NAME:-vindr_scheduled_ddp_g3}"
NAME3="${SCHEDULED_NAME:-vindr_scheduled_ddp_g5}"
SPLIT="${SPLIT:-test}"
EPOCH="${EPOCH:-latest}"
NUM_TEST="${NUM_TEST:-200}"
FINDING_FILTER="${FINDING_FILTER:-right_finding}"
DIRECTION="${DIRECTION:-AtoB}"
RESULTS_DIR="${RESULTS_DIR:-/home/management/projects/gilba/valeryb/cut_results_findings}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-/home/management/projects/gilba/valeryb/cut_checkpoints}"
BILATERAL_SIZE="${BILATERAL_SIZE:-512 384}"

run_test() {
  local name="$1"
  echo "==> Testing name=${name} finding_filter=${FINDING_FILTER} direction=${DIRECTION}"
  python test.py \
    --dataroot "${DATAROOT}" \
    --annotations_csv "${ANNOTATIONS_CSV}" \
    --dataset_mode bilateral \
    --finding_filter "${FINDING_FILTER}" \
    --name "${name}" \
    --model cut --CUT_mode CUT \
    --direction "${DIRECTION}" \
    --split "${SPLIT}" \
    --phase test \
    --epoch "${EPOCH}" \
    --num_test "${NUM_TEST}" \
    --results_dir "${RESULTS_DIR}" \
    --checkpoints_dir "${CHECKPOINTS_DIR}" \
    --bilateral_size ${BILATERAL_SIZE}

  # Emit findings.json alongside the saved images so util/overlay.html can draw
  # the finding bounding boxes. Mirrors the geometry args above so the boxes line
  # up with the saved (flipped/cropped) PNGs.
  python -m util.write_findings \
    --dataroot "${DATAROOT}" \
    --annotations_csv "${ANNOTATIONS_CSV}" \
    --finding_filter "${FINDING_FILTER}" \
    --name "${name}" \
    --direction "${DIRECTION}" \
    --split "${SPLIT}" \
    --phase test \
    --epoch "${EPOCH}" \
    --results_dir "${RESULTS_DIR}" \
    --bilateral_size ${BILATERAL_SIZE}
}

# All three checkpoints use the paired `bilateral` adapter with the findings
# filter (deterministic, no shuffling on either breast); only --name differs.
run_test "${NAME1}"
run_test "${NAME2}"
run_test "${NAME3}"

echo "Results written under ${RESULTS_DIR}/{${NAME1},${NAME2},${NAME3}}/test_${EPOCH}/"
