#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
CONFIG="${CONFIG:-$PROJECT_ROOT/configs/daic_audio_text.yaml}"
FOLD="${FOLD:-0}"
RUN_NAME="${RUN_NAME:-mn5_reproduction}"
SUBMIT_BEST_EVAL="${SUBMIT_BEST_EVAL:-1}"
SUBMIT_LAST_EVAL="${SUBMIT_LAST_EVAL:-1}"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"
EXTRA_EVAL_ARGS="${EXTRA_EVAL_ARGS:-}"
ENV_ACTIVATE="${ENV_ACTIVATE:-/gpfs/projects/etur92/ozu647717/venvs/qwen_mn5_rebuilt/bin/activate}"

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

extract_sample_prediction_mode() {
    local args="$1"
    local prev=""
    local token=""
    for token in $args; do
        if [ "$prev" = "--set" ]; then
            case "$token" in
                evaluation.sample_prediction_mode=*)
                    printf '%s\n' "${token#evaluation.sample_prediction_mode=}"
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

TRAIN_SAMPLE_PREDICTION_MODE="$(extract_sample_prediction_mode "$EXTRA_TRAIN_ARGS" || true)"
EVAL_SAMPLE_PREDICTION_MODE="$(extract_sample_prediction_mode "$EXTRA_EVAL_ARGS" || true)"

if [ -n "$TRAIN_SAMPLE_PREDICTION_MODE" ] && [ -z "$EVAL_SAMPLE_PREDICTION_MODE" ]; then
    EXTRA_EVAL_ARGS="${EXTRA_EVAL_ARGS:+$EXTRA_EVAL_ARGS }--set evaluation.sample_prediction_mode=$TRAIN_SAMPLE_PREDICTION_MODE"
    EVAL_SAMPLE_PREDICTION_MODE="$TRAIN_SAMPLE_PREDICTION_MODE"
fi

EXPORT_ARGS="ALL,PROJECT_ROOT=$PROJECT_ROOT,CONFIG=$CONFIG,FOLD=$FOLD,RUN_NAME=$RUN_NAME,EXTRA_TRAIN_ARGS=$EXTRA_TRAIN_ARGS,EXTRA_EVAL_ARGS=$EXTRA_EVAL_ARGS"

echo "Submitting workflow"
echo "  dataset: $DATASET_NAME"
echo "  config: $CONFIG"
echo "  fold: $FOLD"
echo "  run_name: $RUN_NAME"
echo "  fold_dir: $FOLD_DIR"
echo "  best_checkpoint_dir: $BEST_CHECKPOINT_DIR"
echo "  last_checkpoint_dir: $LAST_CHECKPOINT_DIR"
echo "  train_sample_prediction_mode: ${TRAIN_SAMPLE_PREDICTION_MODE:-<from config>}"
echo "  eval_sample_prediction_mode: ${EVAL_SAMPLE_PREDICTION_MODE:-<from config>}"
echo "  synced_extra_eval_args: ${EXTRA_EVAL_ARGS:-<none>}"

TRAIN_JOB_RAW="$(sbatch --parsable --export="$EXPORT_ARGS" "$TRAIN_SCRIPT")"
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
