#!/bin/bash
# CPU-only fixed-head classifier worker for an existing hidden-state cache.
# The extraction job owns the GPU; this job only fits and evaluates the fixed
# Logistic Regression / XGBoost heads on the cache both jobs share.
#SBATCH -J qwen-hidden-clf
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
CACHE_DIR="${CACHE_DIR:?Set CACHE_DIR to the completed hidden-feature cache}"
CLASSIFIER_DIR="${CLASSIFIER_DIR:?Set CLASSIFIER_DIR for the fitted-head outputs}"
CLASSIFIER_VARIANTS="${CLASSIFIER_VARIANTS:-}"
SEED="${SEED:-1337}"
QWEN_HIDDEN_DEPS="${QWEN_HIDDEN_DEPS:-$PROJECT_ROOT/.deps/qwen_hidden}"
LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/logs/slurm_qwen_hidden_classifier}"

export PROJECT_ROOT
export PYTHONPATH="$QWEN_HIDDEN_DEPS:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT_ROOT"
mkdir -p "$LOG_ROOT"
exec > >(tee -a "$LOG_ROOT/classifier-${SLURM_JOB_ID}.out")
exec 2> >(tee -a "$LOG_ROOT/classifier-${SLURM_JOB_ID}.err" >&2)

python -V
python -c 'import numpy, sklearn, xgboost; print("versions", numpy.__version__, sklearn.__version__, xgboost.__version__)'

CMD=(python "$PROJECT_ROOT/baselines/qwen_hidden_classifier.py" \
  --cache-dir "$CACHE_DIR" \
  --output-dir "$CLASSIFIER_DIR" \
  --seed "$SEED")
if [ -n "$CLASSIFIER_VARIANTS" ]; then
  IFS=':' read -r -a classifier_variant_args <<< "$CLASSIFIER_VARIANTS"
  CMD+=(--variants "${classifier_variant_args[@]}")
fi
printf 'Classifier command: '; printf '%q ' "${CMD[@]}"; printf '\n'
"${CMD[@]}"
