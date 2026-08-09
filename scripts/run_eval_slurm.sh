#!/bin/bash
#SBATCH -J llm-depression-eval
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
export TURKISH_DATASET_ROOT="${TURKISH_DATASET_ROOT:-$DATASET_BASE_ROOT/Turkish}"

CONFIG="${CONFIG:-$PROJECT_ROOT/configs/main/daic_audio_text_harmonized_selmacrof1_tf.yaml}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:?Set CHECKPOINT_DIR}"
FOLD="${FOLD:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-$CHECKPOINT_DIR/standalone_eval}"
MODEL_PATH="${MODEL_PATH:-}"
EXTRA_EVAL_ARGS="${EXTRA_EVAL_ARGS:-}"
EXPERIMENT_CONTEXT="${EXPERIMENT_CONTEXT:-}"
SKIP_MANIFEST_BUILD="${SKIP_MANIFEST_BUILD:-0}"
DATASET_NAME="$(python - <<PY
import sys
from pathlib import Path
sys.path.insert(0, "$PROJECT_ROOT")
from src.utils import load_yaml
config = load_yaml(Path("$CONFIG"))
print(config["dataset"])
PY
)"

LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/logs/slurm_eval/$DATASET_NAME}"
mkdir -p "$LOG_ROOT"

SLURM_STDOUT_FILE="$LOG_ROOT/eval-${SLURM_JOB_ID}.out"
SLURM_STDERR_FILE="$LOG_ROOT/eval-${SLURM_JOB_ID}.err"
exec > >(tee -a "$SLURM_STDOUT_FILE")
exec 2> >(tee -a "$SLURM_STDERR_FILE" >&2)

TIMESTAMP="$(date +%Y-%m-%d_%H:%M:%S)"
RUN_LOG_FILE="$LOG_ROOT/eval-${SLURM_JOB_ID}-${TIMESTAMP}.log"

echo "========================================" | tee -a "$RUN_LOG_FILE"
echo "LLM-Depression Evaluation Job" | tee -a "$RUN_LOG_FILE"
echo "========================================" | tee -a "$RUN_LOG_FILE"
echo "Timestamp: $TIMESTAMP" | tee -a "$RUN_LOG_FILE"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-}" | tee -a "$RUN_LOG_FILE"
echo "Project Root: $PROJECT_ROOT" | tee -a "$RUN_LOG_FILE"
echo "Config: $CONFIG" | tee -a "$RUN_LOG_FILE"
echo "Fold: $FOLD" | tee -a "$RUN_LOG_FILE"
echo "Checkpoint Dir: $CHECKPOINT_DIR" | tee -a "$RUN_LOG_FILE"
echo "Output Dir: $OUTPUT_DIR" | tee -a "$RUN_LOG_FILE"
echo "MODEL_PATH: ${MODEL_PATH:-<from YAML or checkpoint base model>}" | tee -a "$RUN_LOG_FILE"
echo "EXTRA_EVAL_ARGS: ${EXTRA_EVAL_ARGS:-<none>}" | tee -a "$RUN_LOG_FILE"
echo "SKIP_MANIFEST_BUILD: $SKIP_MANIFEST_BUILD" | tee -a "$RUN_LOG_FILE"
echo "DATASET_BASE_ROOT: $DATASET_BASE_ROOT" | tee -a "$RUN_LOG_FILE"
echo "DAIC_DATASET_ROOT: $DAIC_DATASET_ROOT" | tee -a "$RUN_LOG_FILE"
echo "CMDC_DATASET_ROOT: $CMDC_DATASET_ROOT" | tee -a "$RUN_LOG_FILE"
echo "EATD_DATASET_ROOT: $EATD_DATASET_ROOT" | tee -a "$RUN_LOG_FILE"
echo "TURKISH_DATASET_ROOT: $TURKISH_DATASET_ROOT" | tee -a "$RUN_LOG_FILE"
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
from src.utils import load_yaml_with_overrides, resolve_project_path

override_args = """$EXTRA_EVAL_ARGS""".split()
config = load_yaml_with_overrides(Path("$CONFIG"), override_args)
dataset = config["dataset"]

print("dataset", dataset)
print("dataset_root", config["dataset_root"])
print("transcript_file", config.get("transcript_file", "<default>"))
print("audio_adapter", json.dumps(config.get("audio_adapter", {}), sort_keys=True))

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

MANIFEST_CMD=(
    python
    "$PROJECT_ROOT/src/data/build_manifest.py"
    --config "$CONFIG"
)
if [ -n "$EXTRA_EVAL_ARGS" ]; then
    # shellcheck disable=SC2206
    EXTRA_ARGS_ARRAY=($EXTRA_EVAL_ARGS)
    for ((i=0; i<${#EXTRA_ARGS_ARRAY[@]}; i++)); do
        if [ "${EXTRA_ARGS_ARRAY[$i]}" = "--set" ] && [ $((i + 1)) -lt ${#EXTRA_ARGS_ARRAY[@]} ]; then
            MANIFEST_CMD+=("${EXTRA_ARGS_ARRAY[$i]}" "${EXTRA_ARGS_ARRAY[$((i + 1))]}")
            i=$((i + 1))
        fi
    done
fi

if [ "$SKIP_MANIFEST_BUILD" = "1" ]; then
    echo "Skipping manifest rebuild; using the manifest prepared by the orchestration job." | tee -a "$RUN_LOG_FILE"
else
    echo "Building or refreshing manifests before evaluation" | tee -a "$RUN_LOG_FILE"
    printf 'Manifest command: ' | tee -a "$RUN_LOG_FILE"
    printf '%q ' "${MANIFEST_CMD[@]}" | tee -a "$RUN_LOG_FILE"
    printf '\n' | tee -a "$RUN_LOG_FILE"
    "${MANIFEST_CMD[@]}" 2>&1 | tee -a "$RUN_LOG_FILE"
fi

CMD=(
    python
    "$PROJECT_ROOT/src/evaluate.py"
    --config "$CONFIG"
    --fold "$FOLD"
    --checkpoint_dir "$CHECKPOINT_DIR"
    --output_dir "$OUTPUT_DIR"
)

if [ -n "$MODEL_PATH" ]; then
    CMD+=(--model_name_or_path "$MODEL_PATH")
fi

if [ -n "$EXTRA_EVAL_ARGS" ]; then
    # Intentionally allow extra CLI overrides from sbatch --export.
    # shellcheck disable=SC2206
    EXTRA_ARGS_ARRAY=($EXTRA_EVAL_ARGS)
    CMD+=("${EXTRA_ARGS_ARRAY[@]}")
fi

if [ -n "$EXPERIMENT_CONTEXT" ]; then
    CMD+=(--experiment-context "$EXPERIMENT_CONTEXT")
fi

printf 'Launch command: ' | tee -a "$RUN_LOG_FILE"
printf '%q ' "${CMD[@]}" | tee -a "$RUN_LOG_FILE"
printf '\n' | tee -a "$RUN_LOG_FILE"

"${CMD[@]}" 2>&1 | tee -a "$RUN_LOG_FILE"
