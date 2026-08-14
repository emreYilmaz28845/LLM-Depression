#!/usr/bin/env bash
# Emit backend-resolved harmonized launcher settings for one config file.
#
# Usage:
#   backend_vars="$(bash scripts/harmonized_backend_env.sh <config-path> [<project-root>])"
#   eval "$backend_vars"
#
# The launcher then uses $ENV_ACTIVATE, $MODEL_PATH, $HIDDEN_WORKER, and
# $CLASSIFIER_VARIANTS when building each cell's sbatch commands. Qwen cells
# keep today's behavior; gemma4 cells select the dedicated Gemma environment,
# the pinned Gemma base-model path, the Gemma hidden worker, and the
# standardized fixed-LogReg-only head (XGBoost arrives later through the
# Optuna-100 protocol, never as a fixed head for Gemma).
set -euo pipefail

CONFIG_PATH="${1:?config path required}"
PROJECT_ROOT="${2:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
GEMMA_ENV="${GEMMA_ENV:-/gpfs/projects/etur92/ozu647717/venvs/gemma4_12b_tf5_14_1}"
GEMMA4_MODEL_PATH="${GEMMA4_MODEL_PATH:-/gpfs/projects/etur92/ozu647717/models/gemma-4-12B-it/707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7}"
QWEN_ENV_ACTIVATE="${QWEN_ENV_ACTIVATE:-/gpfs/projects/etur92/ozu647717/venvs/qwen_mn5_rebuilt/bin/activate}"

MODEL_BACKEND="$(python - "$CONFIG_PATH" "$PROJECT_ROOT" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[2])
from src.utils import load_yaml_with_overrides, resolve_model_backend
config = load_yaml_with_overrides(Path(sys.argv[1]), [])
print(resolve_model_backend(config) or "")
PY
)"

case "$MODEL_BACKEND" in
  gemma4)
    printf 'MODEL_BACKEND=gemma4\n'
    printf 'ENV_ACTIVATE=%s/bin/activate\n' "$GEMMA_ENV"
    printf 'MODEL_PATH=%s\n' "$GEMMA4_MODEL_PATH"
    printf 'HIDDEN_WORKER=%s/scripts/run_gemma4_harmonized_hidden_slurm.sh\n' "$PROJECT_ROOT"
    printf 'CLASSIFIER_VARIANTS=logreg_raw\n'
    ;;
  qwen2audio|qwen_text|text|qwen3omni|"")
    printf 'MODEL_BACKEND=qwen\n'
    printf 'ENV_ACTIVATE=%s\n' "$QWEN_ENV_ACTIVATE"
    printf 'MODEL_PATH=\n'
    printf 'HIDDEN_WORKER=%s/scripts/run_qwen_hidden_extract_slurm.sh\n' "$PROJECT_ROOT"
    printf 'CLASSIFIER_VARIANTS=logreg_raw:xgb_raw\n'
    ;;
  *)
    echo "Unsupported model_backend: $MODEL_BACKEND" >&2
    exit 2
    ;;
esac
