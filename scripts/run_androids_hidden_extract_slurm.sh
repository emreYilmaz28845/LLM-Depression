#!/bin/bash
#SBATCH -J and-hid-ext
#SBATCH -A etur92
#SBATCH -q acc_ehpc
#SBATCH -t 24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#SBATCH --chdir=/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression

set -euo pipefail
module purge
module load bsc/1.0
module load miniforge/24.3.0-0

PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
ENV_ACTIVATE="${ENV_ACTIVATE:-/gpfs/projects/etur92/ozu647717/venvs/qwen_mn5_rebuilt/bin/activate}"
RUN_ID="${RUN_ID:?RUN_ID is required}"
MODALITY="${MODALITY:?MODALITY is required}"
FOLD="${FOLD:?FOLD is required}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:?CHECKPOINT_DIR is required}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR is required}"
MANIFEST_PATH="${MANIFEST_PATH:?MANIFEST_PATH is required}"
SOURCE_COMMIT="${SOURCE_COMMIT:?SOURCE_COMMIT is required}"
SOURCE_RUN_ID="${SOURCE_RUN_ID:?SOURCE_RUN_ID is required}"
MAX_EXAMPLES="${MAX_EXAMPLES:-}"

if [ ! -f "$ENV_ACTIVATE" ]; then
    echo "Environment activate script not found: $ENV_ACTIVATE" >&2
    exit 1
fi
# shellcheck disable=SC1090
source "$ENV_ACTIVATE"
cd "$PROJECT_ROOT"
export PROJECT_ROOT
export PYTHONPATH="$PROJECT_ROOT/.deps/qwen_hidden:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
LOG_ROOT="$PROJECT_ROOT/logs/slurm_androids_hidden/$RUN_ID"
mkdir -p "$LOG_ROOT"
exec > >(tee -a "$LOG_ROOT/extract-${MODALITY}-fold${FOLD}-${SLURM_JOB_ID}.out")
exec 2> >(tee -a "$LOG_ROOT/extract-${MODALITY}-fold${FOLD}-${SLURM_JOB_ID}.err" >&2)

CMD=(python baselines/extract_androids_hidden.py
    --checkpoint-dir "$CHECKPOINT_DIR"
    --output-dir "$OUTPUT_DIR"
    --manifest-path "$MANIFEST_PATH"
    --modality "$MODALITY"
    --source-commit "$SOURCE_COMMIT"
    --source-run-id "$SOURCE_RUN_ID")
if [ -n "$MAX_EXAMPLES" ]; then
    CMD+=(--max-examples "$MAX_EXAMPLES")
fi
"${CMD[@]}"
