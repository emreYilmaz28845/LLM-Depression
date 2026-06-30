#!/bin/bash
#SBATCH -J llm-depression-optuna
#SBATCH -A etur92
#SBATCH -q acc_ehpc
#SBATCH -t 72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#SBATCH --chdir=/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression

set -e
set -o pipefail

module purge
module load bsc/1.0
module load miniforge/24.3.0-0

ENV_ACTIVATE="${ENV_ACTIVATE:-/gpfs/projects/etur92/ozu647717/venvs/qwen_mn5_rebuilt/bin/activate}"
if [ -f "$ENV_ACTIVATE" ]; then
    # shellcheck disable=SC1090
    source "$ENV_ACTIVATE"
else
    echo "Environment activate script not found: $ENV_ACTIVATE"
    exit 1
fi

PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
export PROJECT_ROOT
cd "$PROJECT_ROOT"

DATASET_BASE_ROOT="${DATASET_BASE_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/Datasets}"
export DAIC_DATASET_ROOT="${DAIC_DATASET_ROOT:-$DATASET_BASE_ROOT/DAIC-WOZ/preprocessed}"
export CMDC_DATASET_ROOT="${CMDC_DATASET_ROOT:-$DATASET_BASE_ROOT/CMDC}"
export EATD_DATASET_ROOT="${EATD_DATASET_ROOT:-$DATASET_BASE_ROOT/EATD-Corpus}"

CONFIG="${CONFIG:-$PROJECT_ROOT/configs/main/daic_audio_text_selposf1_tf.yaml}"
FOLD="${FOLD:-0}"
MODEL_PATH="${MODEL_PATH:-}"
N_TRIALS="${N_TRIALS:-40}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
STUDY_NAME="${STUDY_NAME:-}"
EXTRA_HPO_ARGS="${EXTRA_HPO_ARGS:-}"
DATASET_NAME="$(python - <<PY
import sys
from pathlib import Path
sys.path.insert(0, "$PROJECT_ROOT")
from src.utils import load_yaml
config = load_yaml(Path("$CONFIG"))
print(config["dataset"])
PY
)"

LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/logs/slurm_optuna/$DATASET_NAME}"
mkdir -p "$LOG_ROOT"

SLURM_STDOUT_FILE="$LOG_ROOT/optuna-${SLURM_JOB_ID}.out"
SLURM_STDERR_FILE="$LOG_ROOT/optuna-${SLURM_JOB_ID}.err"
exec > >(tee -a "$SLURM_STDOUT_FILE")
exec 2> >(tee -a "$SLURM_STDERR_FILE" >&2)

TIMESTAMP="$(date +%Y-%m-%d_%H:%M:%S)"
RUN_LOG_FILE="$LOG_ROOT/optuna-${SLURM_JOB_ID}-${TIMESTAMP}.log"

echo "========================================" | tee -a "$RUN_LOG_FILE"
echo "LLM-Depression Optuna Job" | tee -a "$RUN_LOG_FILE"
echo "========================================" | tee -a "$RUN_LOG_FILE"
echo "Timestamp: $TIMESTAMP" | tee -a "$RUN_LOG_FILE"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-}" | tee -a "$RUN_LOG_FILE"
echo "Project Root: $PROJECT_ROOT" | tee -a "$RUN_LOG_FILE"
echo "Config: $CONFIG" | tee -a "$RUN_LOG_FILE"
echo "Fold: $FOLD" | tee -a "$RUN_LOG_FILE"
echo "N_TRIALS: $N_TRIALS" | tee -a "$RUN_LOG_FILE"
echo "NPROC_PER_NODE: $NPROC_PER_NODE" | tee -a "$RUN_LOG_FILE"
echo "MODEL_PATH: ${MODEL_PATH:-<from YAML>}" | tee -a "$RUN_LOG_FILE"
echo "STUDY_NAME: ${STUDY_NAME:-<auto>}" | tee -a "$RUN_LOG_FILE"
echo "EXTRA_HPO_ARGS: ${EXTRA_HPO_ARGS:-<none>}" | tee -a "$RUN_LOG_FILE"
echo "DATASET_BASE_ROOT: $DATASET_BASE_ROOT" | tee -a "$RUN_LOG_FILE"
echo "DAIC_DATASET_ROOT: $DAIC_DATASET_ROOT" | tee -a "$RUN_LOG_FILE"
echo "CMDC_DATASET_ROOT: $CMDC_DATASET_ROOT" | tee -a "$RUN_LOG_FILE"
echo "EATD_DATASET_ROOT: $EATD_DATASET_ROOT" | tee -a "$RUN_LOG_FILE"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-}" | tee -a "$RUN_LOG_FILE"
echo "Hostname: $(hostname)" | tee -a "$RUN_LOG_FILE"
echo "Working Directory: $(pwd)" | tee -a "$RUN_LOG_FILE"
echo "========================================" | tee -a "$RUN_LOG_FILE"

nvidia-smi | tee -a "$RUN_LOG_FILE" || true
python -V | tee -a "$RUN_LOG_FILE"
python -c "import optuna, torch, transformers, accelerate, peft; print('optuna', optuna.__version__); print('torch', torch.__version__, 'cuda', torch.version.cuda); print('transformers', transformers.__version__); print('accelerate', accelerate.__version__); print('peft', peft.__version__)" | tee -a "$RUN_LOG_FILE"

MANIFEST_CMD=(
    python
    "$PROJECT_ROOT/src/data/build_manifest.py"
    --config "$CONFIG"
)
if [ -n "$EXTRA_HPO_ARGS" ]; then
    # shellcheck disable=SC2206
    EXTRA_ARGS_ARRAY=($EXTRA_HPO_ARGS)
    for ((i=0; i<${#EXTRA_ARGS_ARRAY[@]}; i++)); do
        if [ "${EXTRA_ARGS_ARRAY[$i]}" = "--set" ] && [ $((i + 1)) -lt ${#EXTRA_ARGS_ARRAY[@]} ]; then
            MANIFEST_CMD+=("${EXTRA_ARGS_ARRAY[$i]}" "${EXTRA_ARGS_ARRAY[$((i + 1))]}")
            i=$((i + 1))
        fi
    done
fi

echo "Building or refreshing manifests before Optuna study" | tee -a "$RUN_LOG_FILE"
printf 'Manifest command: ' | tee -a "$RUN_LOG_FILE"
printf '%q ' "${MANIFEST_CMD[@]}" | tee -a "$RUN_LOG_FILE"
printf '\n' | tee -a "$RUN_LOG_FILE"
"${MANIFEST_CMD[@]}" 2>&1 | tee -a "$RUN_LOG_FILE"

CMD=(
    python
    "$PROJECT_ROOT/src/hpo.py"
    --config "$CONFIG"
    --fold "$FOLD"
    --n-trials "$N_TRIALS"
    --nproc-per-node "$NPROC_PER_NODE"
)

if [ -n "$MODEL_PATH" ]; then
    CMD+=(--model_name_or_path "$MODEL_PATH")
fi

if [ -n "$STUDY_NAME" ]; then
    CMD+=(--study-name "$STUDY_NAME")
fi

if [ -n "$EXTRA_HPO_ARGS" ]; then
    # shellcheck disable=SC2206
    EXTRA_ARGS_ARRAY=($EXTRA_HPO_ARGS)
    CMD+=("${EXTRA_ARGS_ARRAY[@]}")
fi

printf 'Launch command: ' | tee -a "$RUN_LOG_FILE"
printf '%q ' "${CMD[@]}" | tee -a "$RUN_LOG_FILE"
printf '\n' | tee -a "$RUN_LOG_FILE"

"${CMD[@]}" 2>&1 | tee -a "$RUN_LOG_FILE"
