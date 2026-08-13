#!/bin/bash
#SBATCH -J daic-odv-smoke-heads
#SBATCH -A etur92
#SBATCH -q acc_ehpc
#SBATCH -t 04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#SBATCH --chdir=/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression

set -euo pipefail
module purge
module load bsc/1.0
module load miniforge/24.3.0-0

PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
ENV_ACTIVATE="${ENV_ACTIVATE:-/gpfs/projects/etur92/ozu647717/venvs/qwen_mn5_rebuilt/bin/activate}"
ATTEMPT_DIR="${ATTEMPT_DIR:?ATTEMPT_DIR is required}"
PARENT_FOLD_DIR="${PARENT_FOLD_DIR:?PARENT_FOLD_DIR is required}"
BACKBONE="${BACKBONE:?BACKBONE is required (qwen|gemma4)}"
MODALITY="${MODALITY:?MODALITY is required}"
SMOKE_NAME="${SMOKE_NAME:?SMOKE_NAME is required}"

if [ ! -f "$ENV_ACTIVATE" ]; then
    echo "Qwen environment activate script not found: $ENV_ACTIVATE" >&2
    exit 1
fi
# shellcheck disable=SC1090
source "$ENV_ACTIVATE"
cd "$PROJECT_ROOT"
export PROJECT_ROOT
export PYTHONPATH="$PROJECT_ROOT/.deps/qwen_hidden:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# CPU-only classifier job; no GPU requested.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

LOG_ROOT="$PROJECT_ROOT/logs/daic_officialdev_smokes"
mkdir -p "$LOG_ROOT"
exec > >(tee -a "$LOG_ROOT/smoke-heads-${SLURM_JOB_ID}.out")
exec 2> >(tee -a "$LOG_ROOT/smoke-heads-${SLURM_JOB_ID}.err" >&2)

python baselines/qwen_hidden_classifier.py \
    --cache-dir "$ATTEMPT_DIR/hidden_features" \
    --output-dir "$ATTEMPT_DIR/hidden_classifiers" \
    --variants logreg_raw xgb_raw \
    --seed 1337 \
    --sampling-mode legacy

python scripts/audit_daic_officialdev_smoke.py \
    --attempt-dir "$ATTEMPT_DIR" \
    --parent-fold-dir "$PARENT_FOLD_DIR" \
    --backbone "$BACKBONE" \
    --modality "$MODALITY" \
    --output "$ATTEMPT_DIR/smoke_audit.json"
