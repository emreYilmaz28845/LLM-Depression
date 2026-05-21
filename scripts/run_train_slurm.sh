#!/bin/bash
#SBATCH -J llm-depression-train
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

CONFIG="${CONFIG:-$PROJECT_ROOT/configs/daic_audio_text.yaml}"
FOLD="${FOLD:-0}"
RUN_NAME="${RUN_NAME:-mn5_reproduction}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
MODEL_PATH="${MODEL_PATH:-}"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"
ENABLE_LABEL_MASK_DEBUG="${ENABLE_LABEL_MASK_DEBUG:-0}"

LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/logs/slurm_train}"
mkdir -p "$LOG_ROOT"

SLURM_STDOUT_FILE="$LOG_ROOT/train-${SLURM_JOB_ID}.out"
SLURM_STDERR_FILE="$LOG_ROOT/train-${SLURM_JOB_ID}.err"
exec > >(tee -a "$SLURM_STDOUT_FILE")
exec 2> >(tee -a "$SLURM_STDERR_FILE" >&2)

TIMESTAMP="$(date +%Y-%m-%d_%H:%M:%S)"
RUN_LOG_FILE="$LOG_ROOT/train-${SLURM_JOB_ID}-${TIMESTAMP}.log"

echo "========================================" | tee -a "$RUN_LOG_FILE"
echo "LLM-Depression Training Job" | tee -a "$RUN_LOG_FILE"
echo "========================================" | tee -a "$RUN_LOG_FILE"
echo "Timestamp: $TIMESTAMP" | tee -a "$RUN_LOG_FILE"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-}" | tee -a "$RUN_LOG_FILE"
echo "Project Root: $PROJECT_ROOT" | tee -a "$RUN_LOG_FILE"
echo "Config: $CONFIG" | tee -a "$RUN_LOG_FILE"
echo "Fold: $FOLD" | tee -a "$RUN_LOG_FILE"
echo "Run Name: $RUN_NAME" | tee -a "$RUN_LOG_FILE"
echo "NPROC_PER_NODE: $NPROC_PER_NODE" | tee -a "$RUN_LOG_FILE"
echo "MODEL_PATH: ${MODEL_PATH:-<from YAML>}" | tee -a "$RUN_LOG_FILE"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-}" | tee -a "$RUN_LOG_FILE"
echo "Hostname: $(hostname)" | tee -a "$RUN_LOG_FILE"
echo "Working Directory: $(pwd)" | tee -a "$RUN_LOG_FILE"
echo "========================================" | tee -a "$RUN_LOG_FILE"

nvidia-smi | tee -a "$RUN_LOG_FILE" || true
python -V | tee -a "$RUN_LOG_FILE"
python -c "import torch, transformers, accelerate, peft; print('torch', torch.__version__, 'cuda', torch.version.cuda); print('transformers', transformers.__version__); print('accelerate', accelerate.__version__); print('peft', peft.__version__)" | tee -a "$RUN_LOG_FILE"

python - <<PY | tee -a "$RUN_LOG_FILE"
import json
import sys
from pathlib import Path
sys.path.insert(0, "$PROJECT_ROOT")
from src.utils import load_yaml, resolve_project_path

config = load_yaml(Path("$CONFIG"))
dataset = config["dataset"]
per_device = int(config["training"]["per_device_train_batch_size"])
grad_acc = int(config["training"]["gradient_accumulation_steps"])
world_size = int("$NPROC_PER_NODE")
effective_batch_size = per_device * grad_acc * world_size

print("dataset", dataset)
print("effective_batch_size", effective_batch_size)
print("dataset_root", config["dataset_root"])

split_dir = resolve_project_path(config["output_dirs"]["split_dir"])
metadata_path = split_dir / f"{dataset}_manifest_metadata.json"
if metadata_path.exists():
    metadata = json.loads(metadata_path.read_text())
    print("manifest_metadata", str(metadata_path))
    print("manifest_path", metadata.get("manifest_path", ""))
    print("manifest_hash", metadata.get("manifest_hash", ""))
else:
    print("manifest_metadata", f"MISSING:{metadata_path}")
PY

echo "Building or refreshing manifests before torchrun" | tee -a "$RUN_LOG_FILE"
python "$PROJECT_ROOT/src/data/build_manifest.py" --config "$CONFIG" 2>&1 | tee -a "$RUN_LOG_FILE"

LABEL_MASK_FLAG=""
if [ "$ENABLE_LABEL_MASK_DEBUG" = "1" ]; then
    LABEL_MASK_FLAG="--label_mask_debug"
fi

CMD=(
    torchrun
    --nproc_per_node="$NPROC_PER_NODE"
    "$PROJECT_ROOT/src/train.py"
    --config "$CONFIG"
    --fold "$FOLD"
    --run_name "$RUN_NAME"
)

if [ -n "$MODEL_PATH" ]; then
    CMD+=(--model_name_or_path "$MODEL_PATH")
fi

if [ -n "$LABEL_MASK_FLAG" ]; then
    CMD+=("$LABEL_MASK_FLAG")
fi

if [ -n "$EXTRA_TRAIN_ARGS" ]; then
    # Intentionally allow extra CLI overrides from sbatch --export.
    # shellcheck disable=SC2206
    EXTRA_ARGS_ARRAY=($EXTRA_TRAIN_ARGS)
    CMD+=("${EXTRA_ARGS_ARRAY[@]}")
fi

printf 'Launch command: ' | tee -a "$RUN_LOG_FILE"
printf '%q ' "${CMD[@]}" | tee -a "$RUN_LOG_FILE"
printf '\n' | tee -a "$RUN_LOG_FILE"

"${CMD[@]}" 2>&1 | tee -a "$RUN_LOG_FILE"
