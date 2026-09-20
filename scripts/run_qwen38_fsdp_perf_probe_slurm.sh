#!/usr/bin/env bash
#SBATCH -J q38-fsdp-perf
#SBATCH -A etur92
#SBATCH -q acc_ehpc
#SBATCH -t 01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4
#SBATCH -o /dev/null
#SBATCH -e /dev/null

# FSDP throughput and per-rank memory sweep for the Qwen3.8 text-only recipe.
#
# Runs the real DAIC long examples at per_device_train_batch_size 1 / 2 / 4 with
# the accumulation that keeps the effective global batch at 128, and audits the
# gradient-accumulation sync behaviour. Correctness gates must have passed first;
# this only chooses a setting, it never changes the recipe.
#
# Required env: MANIFEST (the DAIC manifest jsonl built by the training job).
# Optional: PROJECT_ROOT, VENV_DIR, CONFIG, LOG_ROOT, OUTPUT, BATCH_SIZES,
#           LONGEST, STEPS, OVERRIDES.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
VENV_DIR="${VENV_DIR:-/gpfs/projects/etur92/ozu647717/venvs/gemma4_12b_tf5_14_1}"
CONFIG="${CONFIG:-$PROJECT_ROOT/configs/main/daic_text_only_harmonized_selmacrof1_likelihood_v1_qwen38_27b.yaml}"
LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/logs/qwen38_fsdp_perf}"
OUTPUT="${OUTPUT:-$LOG_ROOT/qwen38_fsdp_perf_${SLURM_JOB_ID:-local}.json}"
LOG_FILE="$LOG_ROOT/qwen38_fsdp_perf_${SLURM_JOB_ID:-local}.log"
MANIFEST="${MANIFEST:?Set MANIFEST to the DAIC manifest jsonl}"

module purge
module load bsc/1.0
module load miniforge/24.3.0-0

mkdir -p "$LOG_ROOT"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "========================================"
echo "Qwen3.8 FSDP performance probe"
echo "Timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-} | node: $(hostname) | nodes: ${SLURM_NNODES:-1}"
echo "PROJECT_ROOT: $PROJECT_ROOT"
echo "MANIFEST: $MANIFEST"
echo "========================================"

if [ ! -x "$VENV_DIR/bin/python" ]; then
  echo "FAILED: no python at $VENV_DIR/bin/python" >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PROJECT_ROOT

cd "$PROJECT_ROOT"

set +e
torchrun --nproc_per_node="${NPROC_PER_NODE:-4}" scripts/qwen38_fsdp_perf_probe.py \
  --config "$CONFIG" \
  --manifest "$MANIFEST" \
  --output "$OUTPUT" \
  --batch-sizes ${BATCH_SIZES:-1 2 4} \
  --longest "${LONGEST:-4}" \
  --steps "${STEPS:-2}" \
  ${OVERRIDES:-}
STATUS=$?
set -e

if [ "$STATUS" -ne 0 ]; then
  echo "FAILED: probe exited with status $STATUS (report: $OUTPUT)" >&2
  exit "$STATUS"
fi
echo "OK: FSDP performance probe finished; report: $OUTPUT"
