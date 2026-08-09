#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
CONFIG="${CONFIG:-$PROJECT_ROOT/configs/main/daic_audio_text_harmonized_selmacrof1_tf.yaml}"
FOLD="${FOLD:-0}"
RUN_NAME="${RUN_NAME:-mn5_reproduction}"
SUBMIT_BEST_EVAL_SPECIFIED=0
if [ -n "${SUBMIT_BEST_EVAL+x}" ]; then
    SUBMIT_BEST_EVAL_SPECIFIED=1
else
    SUBMIT_BEST_EVAL=1
fi
SUBMIT_LAST_EVAL_SPECIFIED=0
if [ -n "${SUBMIT_LAST_EVAL+x}" ]; then
    SUBMIT_LAST_EVAL_SPECIFIED=1
else
    SUBMIT_LAST_EVAL=0
fi
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"
EXTRA_EVAL_ARGS="${EXTRA_EVAL_ARGS:-}"
ENV_ACTIVATE="${ENV_ACTIVATE:-/gpfs/projects/etur92/ozu647717/venvs/qwen_mn5_rebuilt/bin/activate}"
SBATCH_DEPENDENCY="${SBATCH_DEPENDENCY:-}"
SBATCH_JOB_NAME="${SBATCH_JOB_NAME:-}"
SKIP_MANIFEST_BUILD="${SKIP_MANIFEST_BUILD:-0}"

TRAIN_SCRIPT="${TRAIN_SCRIPT:-$PROJECT_ROOT/scripts/run_train_slurm.sh}"
EVAL_SCRIPT="${EVAL_SCRIPT:-$PROJECT_ROOT/scripts/run_eval_slurm.sh}"

if [ -f "$ENV_ACTIVATE" ]; then
    # shellcheck disable=SC1090
    source "$ENV_ACTIVATE"
fi

echo "Resolving workflow configuration..."
echo "  project_root: $PROJECT_ROOT"
echo "  config: $CONFIG"
echo "  fold: $FOLD"
echo "  run_name: $RUN_NAME"
echo "  extra_train_args: ${EXTRA_TRAIN_ARGS:-<none>}"
echo "  extra_eval_args: ${EXTRA_EVAL_ARGS:-<none>}"
echo "  sbatch_dependency: ${SBATCH_DEPENDENCY:-<none>}"

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

if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "Training script not found: $TRAIN_SCRIPT"
    exit 1
fi

if [ ! -f "$EVAL_SCRIPT" ]; then
    echo "Evaluation script not found: $EVAL_SCRIPT"
    exit 1
fi

DATASET_NAME="$(awk -F': *' '/^dataset:/{print $2; exit}' "$CONFIG" | tr -d '"' | tr -d "'")"
RUN_ROOT_REL="$(awk '
  /^output_dirs:/ {in_block=1; next}
  in_block && /^[^[:space:]]/ {in_block=0}
  in_block && /^[[:space:]]+run_root:/ {
    sub(/^[[:space:]]+run_root:[[:space:]]*/, "", $0)
    print $0
    exit
  }
' "$CONFIG" | tr -d '"' | tr -d "'")"
CONFIG_SPLIT_MODE="$(awk '
  /^split:/ {in_block=1; next}
  in_block && /^[^[:space:]]/ {in_block=0}
  in_block && /^[[:space:]]+mode:/ {
    sub(/^[[:space:]]+mode:[[:space:]]*/, "", $0)
    print $0
    exit
  }
' "$CONFIG" | tr -d '"' | tr -d "'")"
CONFIG_CV_PROTOCOL="$(awk '
  /^split:/ {in_block=1; next}
  in_block && /^[^[:space:]]/ {in_block=0}
  in_block && /^[[:space:]]+cv_protocol:/ {
    sub(/^[[:space:]]+cv_protocol:[[:space:]]*/, "", $0)
    print $0
    exit
  }
' "$CONFIG" | tr -d '"' | tr -d "'")"

if [ -z "$DATASET_NAME" ]; then
    echo "Could not parse dataset from config: $CONFIG"
    exit 1
fi

if [ -z "$RUN_ROOT_REL" ]; then
    echo "Could not parse output_dirs.run_root from config: $CONFIG"
    exit 1
fi

RUN_ROOT="${RUN_ROOT_REL//\$\{PROJECT_ROOT\}/$PROJECT_ROOT}"
FOLD_DIR="$RUN_ROOT/$RUN_NAME/fold_$FOLD"
BEST_CHECKPOINT_DIR="$FOLD_DIR/best_model"
LAST_CHECKPOINT_DIR="$FOLD_DIR/last_model"

TRAIN_SAMPLE_PREDICTION_MODE="$(extract_set_override "$EXTRA_TRAIN_ARGS" "evaluation.sample_prediction_mode" || true)"
EVAL_SAMPLE_PREDICTION_MODE="$(extract_set_override "$EXTRA_EVAL_ARGS" "evaluation.sample_prediction_mode" || true)"
TRAIN_SPLIT_MODE="$(extract_set_override "$EXTRA_TRAIN_ARGS" "split.mode" || true)"
TRAIN_SPLIT_MODE="${TRAIN_SPLIT_MODE:-$CONFIG_SPLIT_MODE}"
TRAIN_CV_PROTOCOL="$(extract_set_override "$EXTRA_TRAIN_ARGS" "split.cv_protocol" || true)"
TRAIN_CV_PROTOCOL="${TRAIN_CV_PROTOCOL:-$CONFIG_CV_PROTOCOL}"
if [ -z "$TRAIN_SPLIT_MODE" ]; then
    TRAIN_SPLIT_MODE="fixed"
fi
if [ -z "$TRAIN_CV_PROTOCOL" ]; then
    if [ "$DATASET_NAME" = "turkish" ]; then
        TRAIN_CV_PROTOCOL="train_val"
    else
        TRAIN_CV_PROTOCOL="train_val_test"
    fi
fi

if [ -n "$TRAIN_SAMPLE_PREDICTION_MODE" ] && [ -z "$EVAL_SAMPLE_PREDICTION_MODE" ]; then
    EXTRA_EVAL_ARGS="${EXTRA_EVAL_ARGS:+$EXTRA_EVAL_ARGS }--set evaluation.sample_prediction_mode=$TRAIN_SAMPLE_PREDICTION_MODE"
    EVAL_SAMPLE_PREDICTION_MODE="$TRAIN_SAMPLE_PREDICTION_MODE"
fi

EVAL_CV_PROTOCOL="$(extract_set_override "$EXTRA_EVAL_ARGS" "split.cv_protocol" || true)"
if [ -n "$TRAIN_CV_PROTOCOL" ] && [ -z "$EVAL_CV_PROTOCOL" ]; then
    EXTRA_EVAL_ARGS="${EXTRA_EVAL_ARGS:+$EXTRA_EVAL_ARGS }--set split.cv_protocol=$TRAIN_CV_PROTOCOL"
fi

if [ -n "${EXPERIMENT_CONTEXT:-}" ] && [ -f "$EXPERIMENT_CONTEXT" ]; then
    echo "experiment context: $EXPERIMENT_CONTEXT"
fi

if [ "$TRAIN_SPLIT_MODE" = "full_train" ] && [ "$SUBMIT_BEST_EVAL_SPECIFIED" = "0" ]; then
    SUBMIT_BEST_EVAL=0
fi
if [ "$TRAIN_SPLIT_MODE" = "cv" ] && [ "$TRAIN_CV_PROTOCOL" = "train_val" ] && [ "$SUBMIT_BEST_EVAL_SPECIFIED" = "0" ]; then
    SUBMIT_BEST_EVAL=0
fi

EXPORT_ARGS="ALL,PROJECT_ROOT=$PROJECT_ROOT,CONFIG=$CONFIG,FOLD=$FOLD,RUN_NAME=$RUN_NAME,EXTRA_TRAIN_ARGS=$EXTRA_TRAIN_ARGS,EXTRA_EVAL_ARGS=$EXTRA_EVAL_ARGS,SKIP_MANIFEST_BUILD=$SKIP_MANIFEST_BUILD,EXPERIMENT_CONTEXT=${EXPERIMENT_CONTEXT:-}"
SBATCH_BASE_ARGS=()
if [ -n "$SBATCH_DEPENDENCY" ]; then
    SBATCH_BASE_ARGS+=(--dependency="$SBATCH_DEPENDENCY")
fi
if [ -n "$SBATCH_JOB_NAME" ]; then
    SBATCH_BASE_ARGS+=(--job-name="$SBATCH_JOB_NAME")
fi

echo "Submitting workflow"
echo "  dataset: $DATASET_NAME"
echo "  config: $CONFIG"
echo "  fold: $FOLD"
echo "  run_name: $RUN_NAME"
echo "  fold_dir: $FOLD_DIR"
echo "  train_split_mode: $TRAIN_SPLIT_MODE"
echo "  train_cv_protocol: $TRAIN_CV_PROTOCOL"
echo "  best_checkpoint_dir: $BEST_CHECKPOINT_DIR"
echo "  last_checkpoint_dir: $LAST_CHECKPOINT_DIR"
echo "  train_sample_prediction_mode: ${TRAIN_SAMPLE_PREDICTION_MODE:-<from config>}"
echo "  eval_sample_prediction_mode: ${EVAL_SAMPLE_PREDICTION_MODE:-<from config>}"
echo "  synced_extra_eval_args: ${EXTRA_EVAL_ARGS:-<none>}"
echo "  skip_manifest_build: $SKIP_MANIFEST_BUILD"

TRAIN_JOB_RAW="$(sbatch --parsable "${SBATCH_BASE_ARGS[@]}" --export="$EXPORT_ARGS" "$TRAIN_SCRIPT")"
TRAIN_JOB_ID="${TRAIN_JOB_RAW%%;*}"
echo "Submitted training job: $TRAIN_JOB_ID"

if [ "$SUBMIT_BEST_EVAL" = "1" ]; then
    BEST_OUTPUT_DIR="$BEST_CHECKPOINT_DIR/standalone_eval"
    BEST_JOB_RAW="$(sbatch --parsable --dependency=afterok:$TRAIN_JOB_ID --export="$EXPORT_ARGS,CHECKPOINT_DIR=$BEST_CHECKPOINT_DIR,OUTPUT_DIR=$BEST_OUTPUT_DIR" "$EVAL_SCRIPT")"
    BEST_JOB_ID="${BEST_JOB_RAW%%;*}"
    echo "Submitted best-checkpoint eval job: $BEST_JOB_ID"
fi

if [ "$SUBMIT_LAST_EVAL" = "1" ]; then
    LAST_OUTPUT_DIR="$LAST_CHECKPOINT_DIR/standalone_eval"
    LAST_JOB_RAW="$(sbatch --parsable --dependency=afterok:$TRAIN_JOB_ID --export="$EXPORT_ARGS,CHECKPOINT_DIR=$LAST_CHECKPOINT_DIR,OUTPUT_DIR=$LAST_OUTPUT_DIR" "$EVAL_SCRIPT")"
    LAST_JOB_ID="${LAST_JOB_RAW%%;*}"
    echo "Submitted last-checkpoint eval job: $LAST_JOB_ID"
fi

if [ -n "${EXPERIMENT_CONTEXT:-}" ] && [ -f "$EXPERIMENT_CONTEXT" ]; then
    mkdir -p "$FOLD_DIR"
    python - "$FOLD_DIR" "$EXPERIMENT_CONTEXT" "$TRAIN_JOB_ID" "${BEST_JOB_ID:-}" "${LAST_JOB_ID:-}" "$PROJECT_ROOT" <<'PY'
import json
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[6])
from src.experiment_tracking import lifecycle

fold_dir, context_path, train_job = sys.argv[1], sys.argv[2], sys.argv[3]
best_job = sys.argv[4] or None
last_job = sys.argv[5] or None
context = json.loads(Path(context_path).read_text(encoding="utf-8"))
attempt_id = context["attempt_id"]
fold = int(context["fold"])
jobs_path = Path(fold_dir) / "jobs.jsonl"
events = []
if train_job:
    events.append(
        lifecycle.new_job_event(
            job_key="train", job_type="train", event_type="SUBMITTED",
            attempt_id=attempt_id, fold=fold, slurm_job_id=train_job, status="PENDING",
        )
    )
for job_id, key in ((best_job, "best_eval"), (last_job, "last_eval")):
    if job_id:
        events.append(
            lifecycle.new_job_event(
                job_key=key, job_type="evaluation", event_type="SUBMITTED",
                attempt_id=attempt_id, fold=fold, slurm_job_id=job_id,
                dependency_job_ids=[train_job], status="PENDING",
            )
        )
for event in events:
    lifecycle.append_job_event(jobs_path, event)
print(f"recorded {len(events)} SUBMITTED job events -> {jobs_path}")
PY
fi
