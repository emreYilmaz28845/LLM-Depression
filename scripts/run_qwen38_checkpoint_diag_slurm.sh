#!/usr/bin/env bash
#SBATCH -J q38-ckpt-diag
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

# Gradient-checkpointing diagnosis for the Qwen3.8 FSDP run: a controlled memory
# ladder over one DAIC example (median by default), plus runtime evidence about
# the checkpointing flags, the saved tensors and the FSDP shard geometry.
#
# Required env: MANIFEST.
# Optional: PROJECT_ROOT, VENV_DIR, CONFIG, LOG_ROOT, OUTPUT, EXAMPLE_INDEX,
#           OVERRIDES.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
VENV_DIR="${VENV_DIR:-/gpfs/projects/etur92/ozu647717/venvs/gemma4_12b_tf5_14_1}"
CONFIG="${CONFIG:-$PROJECT_ROOT/configs/main/daic_text_only_harmonized_selmacrof1_likelihood_v1_qwen38_27b.yaml}"
LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/logs/qwen38_checkpoint_diag}"
OUTPUT="${OUTPUT:-$LOG_ROOT/qwen38_checkpoint_diag_${SLURM_JOB_ID:-local}.json}"
LOG_FILE="$LOG_ROOT/qwen38_checkpoint_diag_${SLURM_JOB_ID:-local}.log"
MANIFEST="${MANIFEST:?Set MANIFEST to the DAIC manifest jsonl}"

module purge
module load bsc/1.0
module load miniforge/24.3.0-0

mkdir -p "$LOG_ROOT"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "========================================"
echo "Qwen3.8 gradient-checkpointing diagnosis"
echo "Timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-} | node: $(hostname)"
echo "MANIFEST: $MANIFEST"
echo "VENV_DIR: $VENV_DIR"
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
"$VENV_DIR/bin/python" -m torch.distributed.run --nproc_per_node="${NPROC_PER_NODE:-4}" scripts/qwen38_checkpoint_diag.py \
  --config "$CONFIG" \
  --manifest "$MANIFEST" \
  --output "$OUTPUT" \
  --example-index "${EXAMPLE_INDEX:-94}" \
  ${OVERRIDES:-}
STATUS=$?
set -e

if [ "$STATUS" -ne 0 ]; then
  echo "FAILED: diagnosis exited with status $STATUS (report: $OUTPUT)" >&2
else
  echo "OK: diagnosis finished; report: $OUTPUT"
fi
exit "$STATUS"
