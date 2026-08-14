#!/bin/bash
#SBATCH -J gemma4-hid-harm
#SBATCH -A etur92
#SBATCH -q acc_ehpc
#SBATCH -t 48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#SBATCH --chdir=/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression

# Harmonized Gemma hidden job: prompt-only hidden-state extraction (Gemma env,
# 1 H100) followed by the fixed Logistic Regression head (Qwen env + hidden
# dependency dir, CPU). One Slurm job per fold.
#
# XGBoost is deliberately NOT fitted here: every new Gemma XGBoost result
# comes from the standardized Optuna-100 protocol, never from a fixed head.
set -euo pipefail
module purge
module load bsc/1.0
module load miniforge/24.3.0-0

PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
export PROJECT_ROOT
cd "$PROJECT_ROOT"

GEMMA_ENV="${GEMMA_ENV:-/gpfs/projects/etur92/ozu647717/venvs/gemma4_12b_tf5_14_1}"
ENV_ACTIVATE="${ENV_ACTIVATE:-$GEMMA_ENV/bin/activate}"
QWEN_ENV="${QWEN_ENV:-/gpfs/projects/etur92/ozu647717/venvs/qwen_mn5_rebuilt}"
QWEN_HIDDEN_DEPS="${QWEN_HIDDEN_DEPS:-$PROJECT_ROOT/.deps/qwen_hidden}"

CHECKPOINT_DIR="${CHECKPOINT_DIR:?Set CHECKPOINT_DIR to a fold best_model directory}"
CACHE_DIR="${CACHE_DIR:?Set CACHE_DIR}"
CLASSIFIER_DIR="${CLASSIFIER_DIR:-${CACHE_DIR/hidden_features/hidden_classifiers}}"
MODEL_PATH="${MODEL_PATH:-}"
CONDITION="${CONDITION:-}"
MAX_EXAMPLES="${MAX_EXAMPLES:-}"
CLASSIFIER_VARIANTS="${CLASSIFIER_VARIANTS:-logreg_raw}"

if [ ! -f "$ENV_ACTIVATE" ]; then
    echo "Gemma environment activate script not found: $ENV_ACTIVATE" >&2
    exit 1
fi
# shellcheck disable=SC1090
source "$ENV_ACTIVATE"

# MN5 has no outbound internet: force offline everywhere and never fall back
# to a remote package/model API.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/logs/gemma4_harmonized_hidden}"
mkdir -p "$LOG_ROOT"
exec > >(tee -a "$LOG_ROOT/hidden-${SLURM_JOB_ID}.out")
exec 2> >(tee -a "$LOG_ROOT/hidden-${SLURM_JOB_ID}.err" >&2)

python -c 'import torch, transformers, peft; print("gemma-env versions", torch.__version__, transformers.__version__, peft.__version__)'
nvidia-smi >/dev/null 2>&1 || true

CMD=(python "$PROJECT_ROOT/src/features/extract_qwen_hidden.py"
    --checkpoint-dir "$CHECKPOINT_DIR"
    --output-dir "$CACHE_DIR")
if [ -n "$MODEL_PATH" ]; then CMD+=(--model-name-or-path "$MODEL_PATH"); fi
if [ -n "$MAX_EXAMPLES" ]; then CMD+=(--max-examples "$MAX_EXAMPLES"); fi
if [ -n "$CONDITION" ]; then CMD+=(--condition "$CONDITION"); fi
printf 'Extraction command: '; printf '%q ' "${CMD[@]}"; printf '\n'
"${CMD[@]}"

# Fixed LR head in the Qwen environment: sklearn 1.7.0 lives in the Qwen venv,
# xgboost 2.1.4 (unused here but required by the locked import guards) in the
# project-local dependency directory. The Qwen interpreter is invoked by
# absolute path; no second Slurm job is needed.
export PYTHONPATH="$QWEN_HIDDEN_DEPS:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
QWEN_PYTHON="$QWEN_ENV/bin/python"
[ -x "$QWEN_PYTHON" ] || { echo "Qwen python not found: $QWEN_PYTHON" >&2; exit 1; }

CLASSIFIER_CMD=("$QWEN_PYTHON" "$PROJECT_ROOT/baselines/qwen_hidden_classifier.py"
    --cache-dir "$CACHE_DIR"
    --output-dir "$CLASSIFIER_DIR")
IFS=':' read -r -a classifier_variant_args <<< "$CLASSIFIER_VARIANTS"
CLASSIFIER_CMD+=(--variants "${classifier_variant_args[@]}")
printf 'Classifier command: '; printf '%q ' "${CLASSIFIER_CMD[@]}"; printf '\n'
"${CLASSIFIER_CMD[@]}"
