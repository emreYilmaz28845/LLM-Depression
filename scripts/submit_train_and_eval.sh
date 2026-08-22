#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
CONFIG="${CONFIG:-$PROJECT_ROOT/configs/main/daic_audio_text_harmonized_selmacrof1_tf.yaml}"
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
split = config.get("split", {})
print(json.dumps({"run_root": run_root, "split_mode": split.get("mode", "fixed"), "cv_protocol": split.get("cv_protocol")}))
PY
)"
RUN_ROOT_REL="$(printf '%s' "$CONFIG_VALUES" | python -c 'import json,sys; print(json.load(sys.stdin)["run_root"])')"
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
# Ensure log root exists
mkdir -p "$LOG_ROOT"
EXPORT_ARGS="ALL,PROJECT_ROOT=$PROJECT_ROOT,CONFIG=$CONFIG,FOLD=$FOLD,RUN_NAME=$RUN_NAME,EXTRA_TRAIN_ARGS=$EXTRA_TRAIN_ARGS,EXTRA_EVAL_ARGS=$EXTRA_EVAL_ARGS,EXPERIMENT_CONTEXT=${EXPERIMENT_CONTEXT:-},LOG_ROOT=$LOG_ROOT,OVERRIDES_JSON_B64=${OVERRIDES_JSON_B64:-},ENV_ACTIVATE=${ENV_ACTIVATE:-},MODEL_PATH=${MODEL_PATH:-},SKIP_MANIFEST_BUILD=$SKIP_MANIFEST_BUILD"
SBATCH_BASE_ARGS=()
if [ -n "$SBATCH_EXTRA_ARGS" ]; then
    # shellcheck disable=SC2206
    read -r -a SBATCH_BASE_ARGS <<< "$SBATCH_EXTRA_ARGS"
fi
echo "Submitting workflow with --chdir=$PROJECT_ROOT"
TRAIN_JOB_RAW="$(sbatch --parsable --chdir="$PROJECT_ROOT" "${SBATCH_BASE_ARGS[@]}" --export="$EXPORT_ARGS" "$TRAIN_SCRIPT")"
TRAIN_JOB_ID="${TRAIN_JOB_RAW%%;*}"
echo "Submitted training job: $TRAIN_JOB_ID"
BEST_OUTPUT_DIR="$BEST_CHECKPOINT_DIR/standalone_eval"
BEST_JOB_RAW="$(sbatch --parsable --chdir="$PROJECT_ROOT" --dependency=afterok:$TRAIN_JOB_ID --export="$EXPORT_ARGS,CHECKPOINT_DIR=$BEST_CHECKPOINT_DIR,OUTPUT_DIR=$BEST_OUTPUT_DIR" "$EVAL_SCRIPT")"
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
