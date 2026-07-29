#!/bin/bash
#SBATCH -J d3tec-hidden-smoke
#SBATCH -A etur92
#SBATCH -q acc_ehpc
#SBATCH -t 01:00:00
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
ENV_ACTIVATE="${ENV_ACTIVATE:-/gpfs/projects/etur92/ozu647717/venvs/qwen_mn5_rebuilt/bin/activate}"
source "$ENV_ACTIVATE"

PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
MODE="${MODE:?Set MODE=fixed or MODE=optuna}"
RUN_ID="${RUN_ID:?Set RUN_ID}"
CACHE_DIR="${CACHE_DIR:?Set CACHE_DIR}"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR}"
QWEN_HIDDEN_DEPS="${QWEN_HIDDEN_DEPS:-$PROJECT_ROOT/.deps/qwen_hidden}"
export PYTHONPATH="$QWEN_HIDDEN_DEPS:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT_ROOT"

LOG_ROOT="$PROJECT_ROOT/logs/slurm_d3tec_hidden/$RUN_ID"
mkdir -p "$LOG_ROOT"
exec > >(tee -a "$LOG_ROOT/smoke-$MODE-${SLURM_JOB_ID}.out")
exec 2> >(tee -a "$LOG_ROOT/smoke-$MODE-${SLURM_JOB_ID}.err" >&2)

python "$PROJECT_ROOT/scripts/create_d3tec_hidden_smoke_cache.py" \
  --output-dir "$CACHE_DIR"

if [ "$MODE" = "fixed" ]; then
  python "$PROJECT_ROOT/baselines/qwen_hidden_classifier.py" \
    --cache-dir "$CACHE_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --variants logreg_raw xgb_raw majority_class xgb_raw_shuffled_labels \
    --seed 1337
elif [ "$MODE" = "optuna" ]; then
  python "$PROJECT_ROOT/baselines/qwen_hidden_xgb_optuna.py" \
    --cache-dir "$CACHE_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --objective macro_f1 \
    --target-trials 2 \
    --inner-folds 3 \
    --seed 1337 \
    --inner-seed 1337 \
    --experiment-id xgb_optuna_raw_smoke_t2_seed1337 \
    --search-profile standard_d6 \
    --xgb-threads 20
else
  echo "Unsupported MODE=$MODE" >&2
  exit 1
fi
