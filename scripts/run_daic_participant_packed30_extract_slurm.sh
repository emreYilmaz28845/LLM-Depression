#!/bin/bash
#SBATCH -J p30-extract
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

CONFIG="${CONFIG:?Set CONFIG}"
FOLD="${FOLD:-0}"
RUN_NAME="${RUN_NAME:?Set RUN_NAME}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:?Set CHECKPOINT_DIR}"
CACHE_DIR="${CACHE_DIR:?Set CACHE_DIR}"
CONDITION="${CONDITION:?Set CONDITION}"
EXTRACTION_INFERENCE_DTYPE="${EXTRACTION_INFERENCE_DTYPE:-}"
QWEN_HIDDEN_DEPS="${QWEN_HIDDEN_DEPS:-$PROJECT_ROOT/.deps/qwen_hidden}"
export PYTHONPATH="$QWEN_HIDDEN_DEPS:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/logs/slurm_daic_participant_packed30}"
mkdir -p "$LOG_ROOT"
exec > >(tee -a "$LOG_ROOT/extract-${SLURM_JOB_ID}.out")
exec 2> >(tee -a "$LOG_ROOT/extract-${SLURM_JOB_ID}.err" >&2)

# Locked dependency versions; a mismatch is a STOP condition.
python - <<'PY'
import platform, importlib.metadata as md
required = {"python": "3.10.14", "scikit-learn": "1.7.0", "xgboost": "2.1.4"}
actual = {
    "python": platform.python_version(),
    "scikit-learn": md.version("scikit-learn"),
    "xgboost": md.version("xgboost"),
}
print("versions", actual)
if actual != required:
    raise SystemExit(f"STOP: locked hidden/head dependency versions required {required}, got {actual}")
PY

if [ ! -f "$CHECKPOINT_DIR/adapter_model.safetensors" ]; then
    echo "Refusing extraction: best_model is missing at $CHECKPOINT_DIR" >&2
    exit 1
fi

CMD=(
  python "$PROJECT_ROOT/src/features/extract_qwen_hidden.py"
  --checkpoint-dir "$CHECKPOINT_DIR"
  --output-dir "$CACHE_DIR"
  --condition "$CONDITION"
)
if [ -n "$EXTRACTION_INFERENCE_DTYPE" ]; then
    export EXTRACTION_INFERENCE_DTYPE
fi
# The base model is resolved from the saved checkpoint's run_config; do not
# pass model flags that could override the text-only YAML default.

echo "== packed30 hidden extraction =="
echo "checkpoint=$CHECKPOINT_DIR cache=$CACHE_DIR condition=$CONDITION"
printf 'Launch: '; printf '%q ' "${CMD[@]}"; printf '\n'
"${CMD[@]}"
echo "== packed30 hidden extraction finished =="
