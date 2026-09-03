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

# MN5 has no outbound internet: force offline everywhere, and hard-fail when a
# package or model would try to reach the network. Qwen jobs load local GPFS
# models too, so this is safe for the existing environment as well.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

# Memory-allocator strategy (same class of runtime env as
# CUBLAS_WORKSPACE_CONFIG): expandable segments avoid the fragmentation that
# otherwise OOMs the intended BF16 LoRA audio+text configuration on H100s
# (verified: 50.4 GiB peak with real 6490-token sequences). This changes no
# scientific configuration.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
export PROJECT_ROOT
cd "$PROJECT_ROOT"

DATASET_BASE_ROOT="${DATASET_BASE_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/Datasets}"
export DAIC_DATASET_ROOT="${DAIC_DATASET_ROOT:-$DATASET_BASE_ROOT/DAIC-WOZ/preprocessed}"
export DAIC_UNPROCESSED_ROOT="${DAIC_UNPROCESSED_ROOT:-$DATASET_BASE_ROOT/DAIC-WOZ/unprocessed}"
export DAIC_LABEL_ROOT="${DAIC_LABEL_ROOT:-$DATASET_BASE_ROOT/DAIC-WOZ/minimal_zips}"
export CMDC_DATASET_ROOT="${CMDC_DATASET_ROOT:-$DATASET_BASE_ROOT/CMDC}"
export EATD_DATASET_ROOT="${EATD_DATASET_ROOT:-$DATASET_BASE_ROOT/EATD-Corpus}"
export TURKISH_DATASET_ROOT="${TURKISH_DATASET_ROOT:-$DATASET_BASE_ROOT/Turkish}"
export TURKISH_POS_ONLY_DATASET_ROOT="${TURKISH_POS_ONLY_DATASET_ROOT:-$DATASET_BASE_ROOT/Turkish_Positive_Only}"
export D3TEC_DATASET_ROOT="${D3TEC_DATASET_ROOT:-$DATASET_BASE_ROOT/D3TEC DATASET/D3TEC DATASET}"
export D3TEC_FULL_TRANSCRIPTS="${D3TEC_FULL_TRANSCRIPTS:-$D3TEC_DATASET_ROOT/transcripts_qwen3_asr_spanish.jsonl}"
export D3TEC_SEGMENT_TRANSCRIPTS="${D3TEC_SEGMENT_TRANSCRIPTS:-$D3TEC_DATASET_ROOT/transcripts_qwen3_asr_spanish_segments.jsonl}"
export ANDROIDS_DATASET_ROOT="${ANDROIDS_DATASET_ROOT:-$DATASET_BASE_ROOT/Androids-Corpus/Androids-Corpus}"
export ANDROIDS_INTERVIEW_FULL_TRANSCRIPTS="${ANDROIDS_INTERVIEW_FULL_TRANSCRIPTS:-$ANDROIDS_DATASET_ROOT/interview_transcripts_qwen3_asr_italian.jsonl}"
export ANDROIDS_INTERVIEW_SEGMENT_TRANSCRIPTS="${ANDROIDS_INTERVIEW_SEGMENT_TRANSCRIPTS:-$ANDROIDS_DATASET_ROOT/interview_transcripts_qwen3_asr_italian_segments.jsonl}"

CONFIG="${CONFIG:-$PROJECT_ROOT/configs/main/daic_audio_text_harmonized_selmacrof1_tf.yaml}"
FOLD="${FOLD:-0}"
RUN_NAME="${RUN_NAME:-mn5_reproduction}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
MODEL_PATH="${MODEL_PATH:-}"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"
OVERRIDES_JSON_B64="${OVERRIDES_JSON_B64:-}"
EXPERIMENT_CONTEXT="${EXPERIMENT_CONTEXT:-}"
SKIP_MANIFEST_BUILD="${SKIP_MANIFEST_BUILD:-0}"
ENABLE_LABEL_MASK_DEBUG="${ENABLE_LABEL_MASK_DEBUG:-0}"

# Lossless common overrides: decode the base64 JSON token array when present
# (authoritative); otherwise fall back to whitespace-splitting EXTRA_TRAIN_ARGS.
if [ -n "$OVERRIDES_JSON_B64" ]; then
    mapfile -t -d '' OVERRIDE_ARGS < <(python - "$OVERRIDES_JSON_B64" <<'PY'
import base64, json, sys
tokens = json.loads(base64.b64decode(sys.argv[1]).decode("utf-8"))
sys.stdout.write("\0".join(tokens))
PY
)
else
    # shellcheck disable=SC2206
    OVERRIDE_ARGS=($EXTRA_TRAIN_ARGS)
fi
DATASET_NAME="$(python - <<PY
import sys
from pathlib import Path
sys.path.insert(0, "$PROJECT_ROOT")
from src.utils import load_yaml
config = load_yaml(Path("$CONFIG"))
print(config["dataset"])
PY
)"

LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/logs/slurm_train/$DATASET_NAME}"
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
echo "EXTRA_TRAIN_ARGS: ${EXTRA_TRAIN_ARGS:-<none>}" | tee -a "$RUN_LOG_FILE"
echo "SKIP_MANIFEST_BUILD: $SKIP_MANIFEST_BUILD" | tee -a "$RUN_LOG_FILE"
echo "DATASET_BASE_ROOT: $DATASET_BASE_ROOT" | tee -a "$RUN_LOG_FILE"
echo "DAIC_DATASET_ROOT: $DAIC_DATASET_ROOT" | tee -a "$RUN_LOG_FILE"
echo "DAIC_UNPROCESSED_ROOT: $DAIC_UNPROCESSED_ROOT" | tee -a "$RUN_LOG_FILE"
echo "DAIC_LABEL_ROOT: $DAIC_LABEL_ROOT" | tee -a "$RUN_LOG_FILE"
echo "CMDC_DATASET_ROOT: $CMDC_DATASET_ROOT" | tee -a "$RUN_LOG_FILE"
echo "EATD_DATASET_ROOT: $EATD_DATASET_ROOT" | tee -a "$RUN_LOG_FILE"
echo "TURKISH_DATASET_ROOT: $TURKISH_DATASET_ROOT" | tee -a "$RUN_LOG_FILE"
echo "D3TEC_DATASET_ROOT: $D3TEC_DATASET_ROOT" | tee -a "$RUN_LOG_FILE"
echo "ANDROIDS_DATASET_ROOT: $ANDROIDS_DATASET_ROOT" | tee -a "$RUN_LOG_FILE"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-}" | tee -a "$RUN_LOG_FILE"
echo "Hostname: $(hostname)" | tee -a "$RUN_LOG_FILE"
echo "Working Directory: $(pwd)" | tee -a "$RUN_LOG_FILE"
echo "========================================" | tee -a "$RUN_LOG_FILE"

nvidia-smi | tee -a "$RUN_LOG_FILE" || true
python -V | tee -a "$RUN_LOG_FILE"
python -c "import torch, transformers, accelerate, peft; print('torch', torch.__version__, 'cuda', torch.version.cuda); print('transformers', transformers.__version__); print('accelerate', accelerate.__version__); print('peft', peft.__version__)" | tee -a "$RUN_LOG_FILE"

# Tier 0 provenance: stamp the shipped code snapshot (commit + uncommitted diff,
# captured locally by scripts/capture_provenance.sh) and freeze the live env into
# a per-job folder. All offline; no .git needed on the cluster.
PROV_OUT="$LOG_ROOT/provenance-${SLURM_JOB_ID}"
mkdir -p "$PROV_OUT"
if [ -d "$PROJECT_ROOT/.provenance" ]; then
    cp -a "$PROJECT_ROOT/.provenance/." "$PROV_OUT/" || true
    echo "Code provenance: commit $(cat "$PROJECT_ROOT/.provenance/git_commit.txt" 2>/dev/null || echo '?')" | tee -a "$RUN_LOG_FILE"
else
    echo "WARNING: $PROJECT_ROOT/.provenance missing; run scripts/capture_provenance.sh before sync." | tee -a "$RUN_LOG_FILE"
fi
pip freeze > "$PROV_OUT/pip_freeze.txt" 2>/dev/null || true
echo "Provenance stamped -> $PROV_OUT" | tee -a "$RUN_LOG_FILE"

python - <<PY | tee -a "$RUN_LOG_FILE"
import base64, json
import sys
from pathlib import Path
sys.path.insert(0, "$PROJECT_ROOT")
from src.utils import load_yaml_with_overrides, resolve_project_path

if """$OVERRIDES_JSON_B64""":
    override_args = json.loads(base64.b64decode("""$OVERRIDES_JSON_B64""").decode("utf-8"))
else:
    override_args = """$EXTRA_TRAIN_ARGS""".split()
config = load_yaml_with_overrides(Path("$CONFIG"), override_args)
dataset = config["dataset"]
per_device = int(config["training"]["per_device_train_batch_size"])
grad_acc = int(config["training"]["gradient_accumulation_steps"])
world_size = int("$NPROC_PER_NODE")
effective_batch_size = per_device * grad_acc * world_size

print("dataset", dataset)
print("effective_batch_size", effective_batch_size)
print("dataset_root", config["dataset_root"])
print("transcript_file", config.get("transcript_file", "<default>"))
print("sample_prediction_mode", config["evaluation"]["sample_prediction_mode"])
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
if [ "${#OVERRIDE_ARGS[@]}" -gt 0 ]; then
    i=0
    while [ $i -lt ${#OVERRIDE_ARGS[@]} ]; do
        if [ "${OVERRIDE_ARGS[$i]}" = "--set" ] && [ $((i + 1)) -lt ${#OVERRIDE_ARGS[@]} ]; then
            MANIFEST_CMD+=("${OVERRIDE_ARGS[$i]}" "${OVERRIDE_ARGS[$((i + 1))]}")
            i=$((i + 2))
        else
            i=$((i + 1))
        fi
    done
fi

if [ "$SKIP_MANIFEST_BUILD" = "1" ]; then
    echo "Skipping manifest rebuild; using the manifest prepared by the orchestration job." | tee -a "$RUN_LOG_FILE"
else
    echo "Building or refreshing manifests before torchrun" | tee -a "$RUN_LOG_FILE"
    printf 'Manifest command: ' | tee -a "$RUN_LOG_FILE"
    printf '%q ' "${MANIFEST_CMD[@]}" | tee -a "$RUN_LOG_FILE"
    printf '\n' | tee -a "$RUN_LOG_FILE"
    "${MANIFEST_CMD[@]}" 2>&1 | tee -a "$RUN_LOG_FILE"
fi

LABEL_MASK_FLAG=""
if [ "$ENABLE_LABEL_MASK_DEBUG" = "1" ]; then
    LABEL_MASK_FLAG="--label_mask_debug"
fi

# Tier 1 determinism: these MUST be set in the shell before the training process
# starts. cuBLAS reads CUBLAS_WORKSPACE_CONFIG at handle creation (before our
# Python runs), and PYTHONHASHSEED must precede interpreter start. set_seed() in
# src/utils.py then flips the torch determinism flags. Harmless when the config
# sets training.deterministic=false. Override CUBLAS_WORKSPACE_CONFIG/PYTHONHASHSEED
# from the environment if you need different values.
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
echo "Determinism env | CUBLAS_WORKSPACE_CONFIG=$CUBLAS_WORKSPACE_CONFIG PYTHONHASHSEED=$PYTHONHASHSEED" | tee -a "$RUN_LOG_FILE"

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

if [ "${#OVERRIDE_ARGS[@]}" -gt 0 ]; then
    # Lossless common override array (from OVERRIDES_JSON_B64 or legacy split).
    CMD+=("${OVERRIDE_ARGS[@]}")
fi

if [ -n "$EXPERIMENT_CONTEXT" ]; then
    CMD+=(--experiment-context "$EXPERIMENT_CONTEXT")
fi

printf 'Launch command: ' | tee -a "$RUN_LOG_FILE"
printf '%q ' "${CMD[@]}" | tee -a "$RUN_LOG_FILE"
printf '\n' | tee -a "$RUN_LOG_FILE"

"${CMD[@]}" 2>&1 | tee -a "$RUN_LOG_FILE"
