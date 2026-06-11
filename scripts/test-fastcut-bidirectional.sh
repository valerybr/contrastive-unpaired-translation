#!/bin/bash
#SBATCH --job-name=test_fastcut_bidir
#SBATCH --output=/home/valeryb/logs/fastcut_test_bidir.out
#SBATCH --error=/home/valeryb/logs/fastcut_test_bidir.err
#SBATCH --partition=gpu-rtx
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#
# Run test.py for a bidirectional shared-G checkpoint through the deterministic
# *paired* `bilateral` adapter. With --bidirectional the single shared G is
# applied in BOTH directions on each study, so every saved sample shows four
# images side by side:
#
#     real_A  = L            (left CC, the canonical-oriented domain)
#     fake_B  = G(L) = fake_R  (L translated to the right domain)
#     real_B  = R            (paired right CC, flipped to L orientation)
#     fake_L  = G(R) = fake_L  (R translated back to the left domain)
#
# The paired adapter keeps both breasts fixed and unshuffled, so the L->R and
# R->L outputs line up with their true contralateral reference.
#
# Submit:   sbatch test-fastcut-bidirectional.sh
# Override any variable via --export, e.g.:
#   sbatch --export=ALL,EPOCH=80,NUM_TEST=500 test-fastcut-bidirectional.sh

set -euo pipefail

source /home/valeryb/.bashrc
conda activate mgdetect
cd /home/valeryb/contrastive-unpaired-translation

DATAROOT="${DATAROOT:-/home/management/projects/gilba/valeryb/data/vindr-masks/images}"
ANNOTATIONS_CSV="${ANNOTATIONS_CSV:-/home/management/projects/gilba/valeryb/data/vindr-masks/finding_annotations.csv}"
NAME1="${NAME1:-vindr_bidirectional_ddp_g1_nce5_masks}"
SPLIT="${SPLIT:-training}"
EPOCH="${EPOCH:-latest}"
NUM_TEST="${NUM_TEST:-200}"
RESULTS_DIR="${RESULTS_DIR:-/home/management/projects/gilba/valeryb/fastcut_results}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-/home/management/projects/gilba/valeryb/fastcut_checkpoints}"
BILATERAL_SIZE="${BILATERAL_SIZE:-512 360}"
# either_finding so a finding on the left (paired with fake_R) OR the right
# (paired with fake_L) yields non-empty boxes in the overlay. Override e.g.
# FINDING_FILTER=no_finding for the original normals-only behaviour.
FINDING_FILTER="${FINDING_FILTER:-either_finding}"

run_test() {
  local mode="$1" name="$2"
  echo "==> Testing dataset_mode=${mode} name=${name} finding_filter=${FINDING_FILTER} (bidirectional)"
  python test.py \
    --dataroot "${DATAROOT}" \
    --annotations_csv "${ANNOTATIONS_CSV}" \
    --dataset_mode "${mode}" \
    --finding_filter "${FINDING_FILTER}" \
    --name "${name}" \
    --model cut --CUT_mode CUT \
    --bidirectional True \
    --flip_right \
    --masked_loss True \
    --split "${SPLIT}" \
    --phase test \
    --epoch "${EPOCH}" \
    --num_test "${NUM_TEST}" \
    --results_dir "${RESULTS_DIR}" \
    --checkpoints_dir "${CHECKPOINTS_DIR}" \
    --bilateral_size ${BILATERAL_SIZE}

  # Emit findings.json alongside the saved images so util/overlay-bd.html can
  # draw the finding bounding boxes on both lateralities. Mirrors the geometry
  # args above (same bilateral_size / finding_filter / --flip_right) so the
  # real_B boxes land in the same flipped L-canonical frame as the saved real_B
  # PNG — without --flip_right they'd be mirrored relative to the image.
  python -m util.write_findings \
    --dataroot "${DATAROOT}" \
    --annotations_csv "${ANNOTATIONS_CSV}" \
    --finding_filter "${FINDING_FILTER}" \
    --name "${name}" \
    --split "${SPLIT}" \
    --phase test \
    --epoch "${EPOCH}" \
    --results_dir "${RESULTS_DIR}" \
    --bilateral_size ${BILATERAL_SIZE} \
    --flip_right
}

# Paired `bilateral` adapter (deterministic, no shuffling on either breast) so
# L, R and their two translations are mutually aligned per study.
run_test bilateral "${NAME1}"

echo "Results written under ${RESULTS_DIR}/${NAME1}/test_${EPOCH}/"
