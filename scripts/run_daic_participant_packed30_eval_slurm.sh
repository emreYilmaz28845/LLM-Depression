#!/bin/bash
#SBATCH -J p30-eval
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
SEED="${SEED:-1337}"
SMOKE_SUBJECT_LIMIT="${SMOKE_SUBJECT_LIMIT:-}"
MODEL_PATH="${MODEL_PATH:-}"
TEXT_MODEL_PATH="${TEXT_MODEL_PATH:-}"
DAIC_UNPROCESSED_ROOT="${DAIC_UNPROCESSED_ROOT:?Set DAIC_UNPROCESSED_ROOT}"
DAIC_LABEL_ROOT="${DAIC_LABEL_ROOT:?Set DAIC_LABEL_ROOT}"
export DAIC_UNPROCESSED_ROOT DAIC_LABEL_ROOT

LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/logs/slurm_daic_participant_packed30}"
mkdir -p "$LOG_ROOT"
exec > >(tee -a "$LOG_ROOT/eval-${SLURM_JOB_ID}.out")
exec 2> >(tee -a "$LOG_ROOT/eval-${SLURM_JOB_ID}.err" >&2)

# Official-test evaluation of the selected checkpoint only. Refuse a missing
# best_model; last_model must never be substituted.
RUN_ROOT="$(python - "$CONFIG" <<PY
import sys
from pathlib import Path
sys.path.insert(0, "$PROJECT_ROOT")
from src.utils import load_yaml_with_overrides
config = load_yaml_with_overrides(Path("$CONFIG"), [])
print(Path(config["output_dirs"]["run_root"]))
PY
)"
BEST_DIR="$RUN_ROOT/$RUN_NAME/fold_$FOLD/best_model"
if [ ! -f "$BEST_DIR/adapter_model.safetensors" ] || [ ! -f "$BEST_DIR/adapter_config.json" ]; then
    echo "Refusing evaluation: best_model is missing at $BEST_DIR" >&2
    exit 1
fi

CMD=(
  python "$PROJECT_ROOT/src/evaluate.py"
  --config "$CONFIG"
  --fold "$FOLD"
  --checkpoint_dir "$BEST_DIR"
  --set "seed=$SEED"
  --set "split.seed=1337"
)
if [ -n "$SMOKE_SUBJECT_LIMIT" ]; then CMD+=(--set "split.smoke_subject_limit=$SMOKE_SUBJECT_LIMIT"); fi
if [ -n "$MODEL_PATH" ]; then CMD+=(--model_name_or_path "$MODEL_PATH"); fi

echo "== packed30 official-test evaluation =="
echo "checkpoint=$BEST_DIR config=$CONFIG"
printf 'Launch: '; printf '%q ' "${CMD[@]}"; printf '\n'
"${CMD[@]}"
echo "== packed30 evaluation finished =="
