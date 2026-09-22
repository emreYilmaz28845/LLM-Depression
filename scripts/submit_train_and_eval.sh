#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
CONFIG="${CONFIG:-$PROJECT_ROOT/configs/main/daic_audio_text_harmonized_selmacrof1_likelihood_v1.yaml}"
FOLD="${FOLD:-0}"
RUN_NAME="${RUN_NAME:-mn5_reproduction}"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"
EXTRA_EVAL_ARGS="${EXTRA_EVAL_ARGS:-}"
LOG_ROOT="${LOG_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/experiment_runtime/parallel_workflow_smoke_v1/logs/slurm_train}"
EXPERIMENT_CONTEXT="${EXPERIMENT_CONTEXT:-}"
# Lossless common override transport: base64(JSON array of override tokens).
# When set, it is authoritative for both training and evaluation; the
# whitespace-split EXTRA_*_ARGS strings remain only as a legacy fallback.
OVERRIDES_JSON_B64="${OVERRIDES_JSON_B64:-}"
SKIP_MANIFEST_BUILD="${SKIP_MANIFEST_BUILD:-0}"
# Optional task-scoped sbatch options.  The native-en launcher uses this to
# carry its node exclusion into both the training and dependent evaluation
# submissions without changing the requested resource shape.
SBATCH_EXTRA_ARGS="${SBATCH_EXTRA_ARGS:-}"
# Train job shape. The default is today's single 4-GPU lane; the FSDP strategy can
# use 2 nodes x 4 GPUs (8 GPUs) when a model does not fit one card. The evaluation
# job keeps its own single-GPU shape regardless.
TRAIN_NODES="${TRAIN_NODES:-1}"
TRAIN_GPUS_PER_NODE="${TRAIN_GPUS_PER_NODE:-4}"
TRAIN_CPUS_PER_GPU="${TRAIN_CPUS_PER_GPU:-20}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$PROJECT_ROOT/scripts/run_train_slurm.sh}"
EVAL_SCRIPT="${EVAL_SCRIPT:-$PROJECT_ROOT/scripts/run_eval_slurm.sh}"
if [ -f "/gpfs/projects/etur92/ozu647717/venvs/qwen_mn5_rebuilt/bin/activate" ]; then
    source "/gpfs/projects/etur92/ozu647717/venvs/qwen_mn5_rebuilt/bin/activate"
fi
echo "Resolving workflow configuration with common overrides..."
echo "  project_root: $PROJECT_ROOT"
echo "  config: $CONFIG"
echo "  fold: $FOLD"
echo "  run_name: $RUN_NAME"
# Extract overrides helper
extract_set_override() {
    local args="$1"
    local target_key="$2"
    local prev=""
    local token=""
    for token in $args; do
        if [ "$prev" = "--set" ]; then
            case "$token" in
                "$target_key"=*)
                    printf '%s\n' "${token#"$target_key"=}"
                    return 0
                    ;;
            esac
            prev=""
            continue
        fi
        if [ "$token" = "--set" ]; then
            prev="--set"
        fi
    done
    return 1
}
DATASET_NAME="$(python - "$CONFIG" "$PROJECT_ROOT" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[2])
from src.utils import load_yaml
config = load_yaml(Path(sys.argv[1]))
print(config["dataset"])
PY
)"
# Resolve run_root with overrides
CONFIG_VALUES="$(python - "$CONFIG" "$PROJECT_ROOT" "$EXTRA_TRAIN_ARGS" "$OVERRIDES_JSON_B64" <<'PY'
import base64, json, sys, shlex
from pathlib import Path
sys.path.insert(0, sys.argv[2])
from src.utils import load_yaml
from src.utils import load_yaml_with_overrides
config_path = Path(sys.argv[1])
project_root = sys.argv[2]
extra_str = sys.argv[3] if len(sys.argv) > 3 else ""
b64 = sys.argv[4] if len(sys.argv) > 4 else ""
if b64:
    args = json.loads(base64.b64decode(b64).decode("utf-8"))
else:
    args = shlex.split(extra_str) if extra_str else []
try:
    config = load_yaml_with_overrides(config_path, args)
except Exception:
    from src.utils import load_yaml
    config = load_yaml(config_path)
run_root = str(config["output_dirs"]["run_root"]).replace("${PROJECT_ROOT}", project_root)
manifest_dir = str(config["output_dirs"]["manifest_dir"]).replace("${PROJECT_ROOT}", project_root)
split_dir = str(config["output_dirs"]["split_dir"]).replace("${PROJECT_ROOT}", project_root)
split = config.get("split", {})
print(json.dumps({
    "run_root": run_root,
    "split_mode": split.get("mode", "fixed"),
    "cv_protocol": split.get("cv_protocol"),
    "dataset": str(config["dataset"]),
    "dataset_variant": str(config.get("dataset_variant", "") or ""),
    "manifest_dir": manifest_dir,
    "split_dir": split_dir,
}))
PY
)"
RUN_ROOT_REL="$(printf '%s' "$CONFIG_VALUES" | python -c 'import json,sys; print(json.load(sys.stdin)["run_root"])')"
DATASET_VARIANT="$(printf '%s' "$CONFIG_VALUES" | python -c 'import json,sys; print(json.load(sys.stdin)["dataset_variant"])')"
MANIFEST_DIR="$(printf '%s' "$CONFIG_VALUES" | python -c 'import json,sys; print(json.load(sys.stdin)["manifest_dir"])')"
SPLIT_DIR="$(printf '%s' "$CONFIG_VALUES" | python -c 'import json,sys; print(json.load(sys.stdin)["split_dir"])')"
# Manifest route guard (fail closed before any sbatch):
# - the pooled Turkish recipe must never let the worker rebuild its manifest;
# - a prebuilt submission must already provide every file the workers read.
if [ "$DATASET_VARIANT" = "pooled_t17" ] && [ "$SKIP_MANIFEST_BUILD" != "1" ]; then
    echo "ERROR: dataset_variant=pooled_t17 requires a prebuilt manifest" >&2
    echo "       (set manifest_policy=prebuilt / SKIP_MANIFEST_BUILD=1)" >&2
    exit 1
fi
if [ "$SKIP_MANIFEST_BUILD" = "1" ]; then
    PREBUILT_FILES=(
        "$MANIFEST_DIR/${DATASET_NAME}_manifest.jsonl"
        "$MANIFEST_DIR/${DATASET_NAME}_manifest.csv"
        "$SPLIT_DIR/${DATASET_NAME}_folds.json"
        "$SPLIT_DIR/${DATASET_NAME}_manifest_metadata.json"
    )
    for prebuilt in "${PREBUILT_FILES[@]}"; do
        if [ ! -f "$prebuilt" ]; then
            echo "ERROR: prebuilt manifest file missing: $prebuilt" >&2
            exit 1
        fi
    done
    echo "  prebuilt_manifest: verified ${#PREBUILT_FILES[@]} files under $MANIFEST_DIR and $SPLIT_DIR"
fi
RUN_ROOT="${RUN_ROOT_REL}"
FOLD_DIR="$RUN_ROOT/$RUN_NAME/fold_$FOLD"
BEST_CHECKPOINT_DIR="$FOLD_DIR/best_model"
# Collision check
if [ -e "$FOLD_DIR" ] && [ -e "$FOLD_DIR/run_config.yaml" ]; then
    echo "ERROR: run directory already exists: $FOLD_DIR (collision)" >&2
    exit 1
fi
# Evaluation view check
EVAL_VIEW_TRAIN="$(extract_set_override "$EXTRA_TRAIN_ARGS" "evaluation.evaluation_view" || true)"
EVAL_VIEW_EVAL="$(extract_set_override "$EXTRA_EVAL_ARGS" "evaluation.evaluation_view" || true)"
EVAL_VIEW="${EVAL_VIEW_TRAIN:-$EVAL_VIEW_EVAL}"
if [ -z "$EVAL_VIEW" ]; then
    HAS_VIEW="$(python - "$CONFIG" "$PROJECT_ROOT" "$EXTRA_TRAIN_ARGS" "$OVERRIDES_JSON_B64" <<'PY'
import base64, json, sys, shlex
from pathlib import Path
sys.path.insert(0, sys.argv[2])
from src.utils import load_yaml
from src.utils import load_yaml_with_overrides
config_path = Path(sys.argv[1])
extra = shlex.split(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] else []
b64 = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] else ""
if b64:
    extra = json.loads(base64.b64decode(b64).decode("utf-8"))
try:
    config = load_yaml_with_overrides(config_path, extra)
except Exception:
    config = load_yaml(config_path)
view = config.get("evaluation", {}).get("evaluation_view")
print(view if view else "")
PY
)"
    if [ -z "$HAS_VIEW" ]; then
        echo "ERROR: evaluation.evaluation_view is required for production" >&2
        exit 1
    fi
    EVAL_VIEW="$HAS_VIEW"
fi
echo "  dataset: $DATASET_NAME"
echo "  fold_dir: $FOLD_DIR"
echo "  evaluation_view: $EVAL_VIEW"
echo "  log_root: $LOG_ROOT"
# Resolve the training strategy and the effective global batch the run will use.
read -r SHAPE_STRATEGY PER_DEVICE_BATCH GRAD_ACCUM <<< "$(python - "$CONFIG" "$PROJECT_ROOT" "$EXTRA_TRAIN_ARGS" "$OVERRIDES_JSON_B64" <<'PY'
import base64, json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[2])
from src.utils import load_yaml, load_yaml_with_overrides
extra = []
if len(sys.argv) > 4 and sys.argv[4]:
    extra = json.loads(base64.b64decode(sys.argv[4]).decode("utf-8"))
elif len(sys.argv) > 3 and sys.argv[3]:
    extra = sys.argv[3].split()
try:
    config = load_yaml_with_overrides(Path(sys.argv[1]), extra or None)
except Exception:
    config = load_yaml(Path(sys.argv[1]))
training = config.get("training", {})
print(
    training.get("strategy", "ddp"),
    training.get("per_device_train_batch_size", 1),
    training.get("gradient_accumulation_steps", 1),
)
PY
)"
WORLD_SIZE=$((TRAIN_NODES * TRAIN_GPUS_PER_NODE))
EFFECTIVE_BATCH=$((PER_DEVICE_BATCH * GRAD_ACCUM * WORLD_SIZE))
echo "  training_strategy: $SHAPE_STRATEGY"
echo "  train_shape: ${TRAIN_NODES} node(s) x ${TRAIN_GPUS_PER_NODE} GPU(s) = ${WORLD_SIZE} rank(s)"
echo "  effective_global_batch_size: $EFFECTIVE_BATCH (per_device=${PER_DEVICE_BATCH} x accumulation=${GRAD_ACCUM} x world_size=${WORLD_SIZE})"
if [ "$SHAPE_STRATEGY" = "fsdp" ] && [ "$EFFECTIVE_BATCH" -ne 128 ]; then
    SUGGESTED_ACCUM=$((128 / (PER_DEVICE_BATCH * WORLD_SIZE)))
    echo "ERROR: the fsdp recipe keeps an effective global batch of 128; ${WORLD_SIZE} rank(s) give ${EFFECTIVE_BATCH}." >&2
    echo "       Pass EXTRA_TRAIN_ARGS=\"--set training.gradient_accumulation_steps=${SUGGESTED_ACCUM}\" so the change is recorded in provenance." >&2
    exit 1
fi
# Ensure log root exists
mkdir -p "$LOG_ROOT"
EXPORT_ARGS="ALL,PROJECT_ROOT=$PROJECT_ROOT,CONFIG=$CONFIG,FOLD=$FOLD,RUN_NAME=$RUN_NAME,EXTRA_TRAIN_ARGS=$EXTRA_TRAIN_ARGS,EXTRA_EVAL_ARGS=$EXTRA_EVAL_ARGS,EXPERIMENT_CONTEXT=${EXPERIMENT_CONTEXT:-},LOG_ROOT=$LOG_ROOT,OVERRIDES_JSON_B64=${OVERRIDES_JSON_B64:-},ENV_ACTIVATE=${ENV_ACTIVATE:-},MODEL_PATH=${MODEL_PATH:-},SKIP_MANIFEST_BUILD=$SKIP_MANIFEST_BUILD"
SBATCH_BASE_ARGS=()
if [ -n "$SBATCH_EXTRA_ARGS" ]; then
    # shellcheck disable=SC2206
    read -r -a SBATCH_BASE_ARGS <<< "$SBATCH_EXTRA_ARGS"
fi
TRAIN_SBATCH_ARGS=(
    --nodes="$TRAIN_NODES"
    --ntasks="$WORLD_SIZE"
    --ntasks-per-node="$TRAIN_GPUS_PER_NODE"
    --cpus-per-task="$TRAIN_CPUS_PER_GPU"
    --gres="gpu:$TRAIN_GPUS_PER_NODE"
)
EVAL_SBATCH_ARGS=(
    --nodes=1
    --ntasks=1
    --ntasks-per-node=1
    --cpus-per-task="$TRAIN_CPUS_PER_GPU"
    --gres="gpu:1"
)
echo "Submitting workflow with --chdir=$PROJECT_ROOT"
TRAIN_JOB_RAW="$(sbatch --parsable --chdir="$PROJECT_ROOT" "${SBATCH_BASE_ARGS[@]}" "${TRAIN_SBATCH_ARGS[@]}" --export="$EXPORT_ARGS,NNODES=$TRAIN_NODES,NPROC_PER_NODE=$TRAIN_GPUS_PER_NODE" "$TRAIN_SCRIPT")"
TRAIN_JOB_ID="${TRAIN_JOB_RAW%%;*}"
echo "Submitted training job: $TRAIN_JOB_ID"
BEST_OUTPUT_DIR="$BEST_CHECKPOINT_DIR/standalone_eval"
BEST_JOB_RAW="$(sbatch --parsable --chdir="$PROJECT_ROOT" "${SBATCH_BASE_ARGS[@]}" "${EVAL_SBATCH_ARGS[@]}" --dependency=afterok:$TRAIN_JOB_ID --export="$EXPORT_ARGS,CHECKPOINT_DIR=$BEST_CHECKPOINT_DIR,OUTPUT_DIR=$BEST_OUTPUT_DIR" "$EVAL_SCRIPT")"
BEST_JOB_ID="${BEST_JOB_RAW%%;*}"
echo "Submitted best-checkpoint eval job: $BEST_JOB_ID"
if [ -n "${EXPERIMENT_CONTEXT:-}" ] && [ -f "$EXPERIMENT_CONTEXT" ]; then
    mkdir -p "$FOLD_DIR"
    python - "$FOLD_DIR" "$EXPERIMENT_CONTEXT" "$TRAIN_JOB_ID" "${BEST_JOB_ID:-}" "$PROJECT_ROOT" <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[4])
from src.experiment_tracking import lifecycle
fold_dir, context_path, train_job = sys.argv[1], sys.argv[2], sys.argv[3]
best_job = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] else None
context = json.loads(Path(context_path).read_text(encoding="utf-8"))
attempt_id = context["attempt_id"]
fold = int(context["fold"])
jobs_path = Path(fold_dir) / "jobs.jsonl"
events = []
if train_job:
    events.append(lifecycle.new_job_event(job_key="train", job_type="train", event_type="SUBMITTED", attempt_id=attempt_id, fold=fold, slurm_job_id=train_job, status="PENDING"))
if best_job:
    events.append(lifecycle.new_job_event(job_key="best_eval", job_type="evaluation", event_type="SUBMITTED", attempt_id=attempt_id, fold=fold, slurm_job_id=best_job, dependency_job_ids=[train_job], status="PENDING"))
for event in events:
    lifecycle.append_job_event(jobs_path, event)
print(f"recorded {len(events)} SUBMITTED job events -> {jobs_path}")
PY
fi
