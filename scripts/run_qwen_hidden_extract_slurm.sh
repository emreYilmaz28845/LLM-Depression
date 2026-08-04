#!/bin/bash
#SBATCH -J qwen-hidden
#SBATCH -A etur92
#SBATCH -q acc_ehpc
#SBATCH -t 24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
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
export PROJECT_ROOT
cd "$PROJECT_ROOT"

CHECKPOINT_DIR="${CHECKPOINT_DIR:?Set CHECKPOINT_DIR to a fold best_model directory}"
CACHE_DIR="${CACHE_DIR:?Set CACHE_DIR}"
CLASSIFIER_DIR="${CLASSIFIER_DIR:-${CACHE_DIR/hidden_features/hidden_classifiers}}"
MODEL_PATH="${MODEL_PATH:-}"
MAX_EXAMPLES="${MAX_EXAMPLES:-}"
CONDITION="${CONDITION:-}"
EMOTION_SOURCE="${EMOTION_SOURCE:-}"
EMOTION_LANGUAGE="${EMOTION_LANGUAGE:-}"
SKIP_CLASSIFIERS="${SKIP_CLASSIFIERS:-0}"
CLASSIFIER_VARIANTS="${CLASSIFIER_VARIANTS:-}"
QWEN_HIDDEN_DEPS="${QWEN_HIDDEN_DEPS:-$PROJECT_ROOT/.deps/qwen_hidden}"
export PYTHONPATH="$QWEN_HIDDEN_DEPS:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/logs/slurm_qwen_hidden}"
mkdir -p "$LOG_ROOT"
exec > >(tee -a "$LOG_ROOT/hidden-${SLURM_JOB_ID}.out")
exec 2> >(tee -a "$LOG_ROOT/hidden-${SLURM_JOB_ID}.err" >&2)

python -c 'import torch, transformers, peft, sklearn, xgboost; print("versions", torch.__version__, transformers.__version__, peft.__version__, sklearn.__version__, xgboost.__version__)'
nvidia-smi

CMD=(python "$PROJECT_ROOT/src/features/extract_qwen_hidden.py" --checkpoint-dir "$CHECKPOINT_DIR" --output-dir "$CACHE_DIR")
if [ -n "$MODEL_PATH" ]; then CMD+=(--model-name-or-path "$MODEL_PATH"); fi
if [ -n "$MAX_EXAMPLES" ]; then CMD+=(--max-examples "$MAX_EXAMPLES"); fi
if [ -n "$CONDITION" ]; then CMD+=(--condition "$CONDITION"); fi
if [ -n "$EMOTION_SOURCE" ]; then CMD+=(--emotion-source "$EMOTION_SOURCE"); fi
if [ -n "$EMOTION_LANGUAGE" ]; then CMD+=(--emotion-language "$EMOTION_LANGUAGE"); fi
printf 'Extraction command: '; printf '%q ' "${CMD[@]}"; printf '\n'
"${CMD[@]}"

if [ "$SKIP_CLASSIFIERS" = "1" ]; then
  echo "Skipping classifiers for extraction-only smoke test."
else
  CLASSIFIER_CMD=(python "$PROJECT_ROOT/baselines/qwen_hidden_classifier.py" \
    --cache-dir "$CACHE_DIR" \
    --output-dir "$CLASSIFIER_DIR")
  if [ -n "$CLASSIFIER_VARIANTS" ]; then
    IFS=':' read -r -a classifier_variant_args <<< "$CLASSIFIER_VARIANTS"
    CLASSIFIER_CMD+=(--variants "${classifier_variant_args[@]}")
  fi
  printf 'Classifier command: '; printf '%q ' "${CLASSIFIER_CMD[@]}"; printf '\n'
  "${CLASSIFIER_CMD[@]}"
fi
