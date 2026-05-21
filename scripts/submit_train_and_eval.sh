#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
CONFIG="${CONFIG:-$PROJECT_ROOT/configs/daic_audio_text.yaml}"
FOLD="${FOLD:-0}"
RUN_NAME="${RUN_NAME:-mn5_reproduction}"
SUBMIT_BEST_EVAL="${SUBMIT_BEST_EVAL:-1}"
SUBMIT_LAST_EVAL="${SUBMIT_LAST_EVAL:-1}"

TRAIN_SCRIPT="${TRAIN_SCRIPT:-$PROJECT_ROOT/scripts/run_train_slurm.sh}"
EVAL_SCRIPT="${EVAL_SCRIPT:-$PROJECT_ROOT/scripts/run_eval_slurm.sh}"

if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "Training script not found: $TRAIN_SCRIPT"
    exit 1
fi

if [ ! -f "$EVAL_SCRIPT" ]; then
    echo "Evaluation script not found: $EVAL_SCRIPT"
    exit 1
fi

readarray -t CONFIG_INFO < <(
    python - <<PY
import sys
from pathlib import Path
sys.path.insert(0, "$PROJECT_ROOT")
from src.utils import load_yaml, resolve_project_path

config = load_yaml(Path("$CONFIG"))
dataset = str(config["dataset"])
run_root = resolve_project_path(config["output_dirs"]["run_root"])
fold_dir = run_root / "$RUN_NAME" / f"fold_{int('$FOLD')}"
print(dataset)
print(run_root)
print(fold_dir)
print(fold_dir / "best_model")
print(fold_dir / "last_model")
PY
)

DATASET_NAME="${CONFIG_INFO[0]}"
RUN_ROOT="${CONFIG_INFO[1]}"
FOLD_DIR="${CONFIG_INFO[2]}"
BEST_CHECKPOINT_DIR="${CONFIG_INFO[3]}"
LAST_CHECKPOINT_DIR="${CONFIG_INFO[4]}"

EXPORT_ARGS="ALL,PROJECT_ROOT=$PROJECT_ROOT,CONFIG=$CONFIG,FOLD=$FOLD,RUN_NAME=$RUN_NAME"

echo "Submitting workflow"
echo "  dataset: $DATASET_NAME"
echo "  config: $CONFIG"
echo "  fold: $FOLD"
echo "  run_name: $RUN_NAME"
echo "  fold_dir: $FOLD_DIR"
echo "  best_checkpoint_dir: $BEST_CHECKPOINT_DIR"
echo "  last_checkpoint_dir: $LAST_CHECKPOINT_DIR"

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
