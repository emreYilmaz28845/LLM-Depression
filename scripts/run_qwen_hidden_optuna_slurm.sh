#!/bin/bash
#SBATCH -J qwen-hidden-optuna
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

ENV_ACTIVATE="${ENV_ACTIVATE:-/gpfs/projects/etur92/ozu647717/venvs/qwen_mn5_rebuilt/bin/activate}"
if [ ! -f "$ENV_ACTIVATE" ]; then
  echo "Environment activate script not found: $ENV_ACTIVATE" >&2
  exit 1
fi
# shellcheck disable=SC1090
source "$ENV_ACTIVATE"

PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
CACHE_DIR="${CACHE_DIR:?Set CACHE_DIR}"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR}"
OBJECTIVE="${OBJECTIVE:?Set OBJECTIVE}"
TARGET_TRIALS="${TARGET_TRIALS:-50}"
INNER_FOLDS="${INNER_FOLDS:-3}"
SEED="${SEED:-1337}"
INNER_SEED="${INNER_SEED:-$SEED}"
EXPERIMENT_ID="${EXPERIMENT_ID:-xgb_optuna_raw}"
SEARCH_PROFILE="${SEARCH_PROFILE:-standard_d6}"
XGB_THREADS="${XGB_THREADS:-20}"
QWEN_HIDDEN_DEPS="${QWEN_HIDDEN_DEPS:-$PROJECT_ROOT/.deps/qwen_hidden}"
LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/logs/slurm_qwen_hidden_optuna/$EXPERIMENT_ID}"

export PROJECT_ROOT
export PYTHONPATH="$QWEN_HIDDEN_DEPS:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT_ROOT"
mkdir -p "$LOG_ROOT"

STDOUT_FILE="$LOG_ROOT/qwen-hidden-optuna-${SLURM_JOB_ID:-manual}.out"
STDERR_FILE="$LOG_ROOT/qwen-hidden-optuna-${SLURM_JOB_ID:-manual}.err"
exec > >(tee -a "$STDOUT_FILE")
exec 2> >(tee -a "$STDERR_FILE" >&2)

CMD=(
  python
  "$PROJECT_ROOT/baselines/qwen_hidden_xgb_optuna.py"
  --cache-dir "$CACHE_DIR"
  --output-dir "$OUTPUT_DIR"
  --objective "$OBJECTIVE"
  --target-trials "$TARGET_TRIALS"
  --inner-folds "$INNER_FOLDS"
  --seed "$SEED"
  --inner-seed "$INNER_SEED"
  --experiment-id "$EXPERIMENT_ID"
  --search-profile "$SEARCH_PROFILE"
  --xgb-threads "$XGB_THREADS"
)

echo "Timestamp: $(date +%Y-%m-%d_%H:%M:%S)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-}"
echo "Hostname: $(hostname)"
echo "Project Root: $PROJECT_ROOT"
echo "Cache Dir: $CACHE_DIR"
echo "Output Dir: $OUTPUT_DIR"
echo "Objective: $OBJECTIVE"
echo "Target Trials: $TARGET_TRIALS"
echo "Inner Folds: $INNER_FOLDS"
echo "Seed: $SEED"
echo "Inner Seed: $INNER_SEED"
echo "Experiment ID: $EXPERIMENT_ID"
echo "Search Profile: $SEARCH_PROFILE"
echo "XGB Threads: $XGB_THREADS"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-}"
python -V
python -c "
import optuna, xgboost, sklearn
assert optuna.__version__ == '4.4.0', optuna.__version__
assert xgboost.__version__ == '2.1.4', xgboost.__version__
print('optuna', optuna.__version__)
print('xgboost', xgboost.__version__)
print('sklearn', sklearn.__version__)
"
printf 'Launch command: '
printf '%q ' "${CMD[@]}"
printf '\n'

"${CMD[@]}"
