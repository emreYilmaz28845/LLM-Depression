#!/usr/bin/env bash
#SBATCH -J q38-mem-probe
#SBATCH -A etur92
#SBATCH -q acc_ehpc
#SBATCH -t 00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

# Qwen3.8-27B text-only backend: per-GPU memory measurement on one H100.
#
# Loads the pinned BF16 checkpoint, runs one training step and one likelihood
# pass through the real backend, and writes a JSON report with the CUDA peak of
# each phase. The result decides whether the existing 4-GPU DDP training lane
# and single-GPU evaluation lane can hold the model, or whether the repository
# needs a separate FSDP / multi-GPU evaluation infrastructure change.
#
# Never run this on a login or transfer node: the model is 52 GB on disk.
#
# Required env: none.
# Optional: PROJECT_ROOT, VENV_DIR, CONFIG, LOG_ROOT, PROMPT_TOKENS, OVERRIDES.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
VENV_DIR="${VENV_DIR:-/gpfs/projects/etur92/ozu647717/venvs/gemma4_12b_tf5_14_1}"
CONFIG="${CONFIG:-$PROJECT_ROOT/configs/main/daic_text_only_harmonized_selmacrof1_likelihood_v1_qwen38_27b.yaml}"
LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/logs/qwen38_memory_probe}"
PROMPT_TOKENS="${PROMPT_TOKENS:-4096}"
OUTPUT="${OUTPUT:-$LOG_ROOT/qwen38_memory_probe_${SLURM_JOB_ID:-local}.json}"
LOG_FILE="$LOG_ROOT/qwen38_memory_probe_${SLURM_JOB_ID:-local}.log"

module purge
module load bsc/1.0
module load miniforge/24.3.0-0

mkdir -p "$LOG_ROOT"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "========================================"
echo "Qwen3.8 memory probe"
echo "Timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-}"
echo "Hostname: $(hostname)"
echo "PROJECT_ROOT: $PROJECT_ROOT"
echo "VENV_DIR: $VENV_DIR"
echo "CONFIG: $CONFIG"
echo "PROMPT_TOKENS: $PROMPT_TOKENS"
echo "========================================"

if [ ! -x "$VENV_DIR/bin/python" ]; then
  echo "FAILED: no python at $VENV_DIR/bin/python" >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PROJECT_ROOT

"$VENV_DIR/bin/python" -c 'import torch, transformers; print("torch", torch.__version__, "| transformers", transformers.__version__)'

cd "$PROJECT_ROOT"

set +e
"$VENV_DIR/bin/python" scripts/qwen38_memory_probe.py \
  --config "$CONFIG" \
  --output "$OUTPUT" \
  --prompt-tokens "$PROMPT_TOKENS" \
  ${OVERRIDES:-}
STATUS=$?
set -e

echo "----------------------------------------"
if [ "$STATUS" -ne 0 ]; then
  echo "FAILED: probe exited with status $STATUS; report: $OUTPUT" >&2
  echo "----- report -----"
  cat "$OUTPUT" || true
  exit "$STATUS"
fi
echo "OK: memory probe finished; report: $OUTPUT"
echo "log: $LOG_FILE"
