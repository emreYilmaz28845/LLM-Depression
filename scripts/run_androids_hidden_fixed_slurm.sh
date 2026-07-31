#!/bin/bash
#SBATCH -J and-hid-fix
#SBATCH -A etur92
#SBATCH -q acc_ehpc
#SBATCH -t 04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
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
RUN_ID="${RUN_ID:?RUN_ID is required}"
MODALITY="${MODALITY:?MODALITY is required}"
FOLD="${FOLD:?FOLD is required}"
CACHE_DIR="${CACHE_DIR:?CACHE_DIR is required}"
OUTPUT_ROOT="${OUTPUT_ROOT:?OUTPUT_ROOT is required}"
SOURCE_COMMIT="${SOURCE_COMMIT:?SOURCE_COMMIT is required}"
SEED="${SEED:-1337}"

# shellcheck disable=SC1090
source "$ENV_ACTIVATE"
cd "$PROJECT_ROOT"
export PROJECT_ROOT
export PYTHONPATH="$PROJECT_ROOT/.deps/qwen_hidden:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
LOG_ROOT="$PROJECT_ROOT/logs/slurm_androids_hidden/$RUN_ID"
mkdir -p "$LOG_ROOT"
exec > >(tee -a "$LOG_ROOT/fixed-${MODALITY}-fold${FOLD}-${SLURM_JOB_ID}.out")
exec 2> >(tee -a "$LOG_ROOT/fixed-${MODALITY}-fold${FOLD}-${SLURM_JOB_ID}.err" >&2)

python baselines/androids_hidden_classifier.py \
    --cache-dir "$CACHE_DIR" \
    --output-root "$OUTPUT_ROOT" \
    --modality "$MODALITY" \
    --fold "$FOLD" \
    --source-commit "$SOURCE_COMMIT" \
    --heads logreg_raw xgb_raw \
    --seed "$SEED"
