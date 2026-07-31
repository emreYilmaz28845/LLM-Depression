#!/bin/bash
#SBATCH -J and-hid-opt
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
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR is required}"
SOURCE_COMMIT="${SOURCE_COMMIT:?SOURCE_COMMIT is required}"
TARGET_TRIALS="${TARGET_TRIALS:-150}"
INNER_FOLDS="${INNER_FOLDS:-3}"
SEED="${SEED:-1337}"
INNER_SEED="${INNER_SEED:-1337}"
XGB_THREADS="${XGB_THREADS:-20}"

# shellcheck disable=SC1090
source "$ENV_ACTIVATE"
cd "$PROJECT_ROOT"
export PROJECT_ROOT
export PYTHONPATH="$PROJECT_ROOT/.deps/qwen_hidden:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
LOG_ROOT="$PROJECT_ROOT/logs/slurm_androids_hidden/$RUN_ID"
mkdir -p "$LOG_ROOT"
exec > >(tee -a "$LOG_ROOT/optuna-${MODALITY}-fold${FOLD}-${SLURM_JOB_ID}.out")
exec 2> >(tee -a "$LOG_ROOT/optuna-${MODALITY}-fold${FOLD}-${SLURM_JOB_ID}.err" >&2)

python baselines/androids_hidden_xgb_optuna.py \
    --cache-dir "$CACHE_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --modality "$MODALITY" \
    --fold "$FOLD" \
    --run-id "$RUN_ID" \
    --source-commit "$SOURCE_COMMIT" \
    --target-trials "$TARGET_TRIALS" \
    --inner-folds "$INNER_FOLDS" \
    --seed "$SEED" \
    --inner-seed "$INNER_SEED" \
    --xgb-threads "$XGB_THREADS"
