#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
ATTEMPT_DIR="${ATTEMPT_DIR:?ATTEMPT_DIR is required}"
PARENT_FOLD_DIR="${PARENT_FOLD_DIR:?PARENT_FOLD_DIR is required}"
MODEL_PATH="${MODEL_PATH:?MODEL_PATH is required}"
DRY_RUN="${DRY_RUN:-0}"

GEMMA_ENV="/gpfs/projects/etur92/ozu647717/venvs/gemma4_12b_tf5_14_1/bin/activate"
QWEN_ENV="/gpfs/projects/etur92/ozu647717/venvs/qwen_mn5_rebuilt/bin/activate"

if [ ! -f "$GEMMA_ENV" ]; then
    echo "Gemma environment not found: $GEMMA_ENV" >&2
    exit 1
fi
if [ ! -f "$QWEN_ENV" ]; then
    echo "Qwen environment not found: $QWEN_ENV" >&2
    exit 1
fi
if [ ! -f "$PROJECT_ROOT/tools/gemma4_hidden_campaign.py" ]; then
    echo "Campaign CLI not found in $PROJECT_ROOT" >&2
    exit 1
fi

campaign() {
    python "$PROJECT_ROOT/tools/gemma4_hidden_campaign.py" "$1" --attempt-dir "$ATTEMPT_DIR" "${@:2}"
}

if [ -n "${SUBJECT_SELECTION:-}" ]; then
    echo "refusing production submission with SUBJECT_SELECTION set" >&2
    exit 1
fi

echo "Submitting Gemma 4 DAIC fixed-head attempt:"
echo "  attempt_dir: $ATTEMPT_DIR"
echo "  parent_fold_dir: $PARENT_FOLD_DIR"
echo "  model_path: $MODEL_PATH"

if [ "$DRY_RUN" = "1" ]; then
    echo "DRY_RUN: extraction sbatch + afterok heads sbatch would be submitted"
    exit 0
fi

campaign transition --to-state DEPLOYED --reason "source deployed to MN5" > /dev/null

EXPORT_COMMON="PROJECT_ROOT=$PROJECT_ROOT,ATTEMPT_DIR=$ATTEMPT_DIR,PARENT_FOLD_DIR=$PARENT_FOLD_DIR,MODEL_PATH=$MODEL_PATH"

EXTRACT_JOB_RAW="$(sbatch --parsable \
    --export="$EXPORT_COMMON,ENV_ACTIVATE=$GEMMA_ENV" \
    "$PROJECT_ROOT/scripts/run_gemma4_hidden_extract_slurm.sh")"
EXTRACT_JOB_ID="$(printf '%s' "$EXTRACT_JOB_RAW" | tail -n 1 | tr -d ' ')"
echo "extraction job: $EXTRACT_JOB_ID"

campaign record-job \
    --job-key extract --job-type hidden_extraction --event-type SUBMITTED \
    --slurm-job-id "$EXTRACT_JOB_ID" --status PENDING \
    --reason "extraction sbatch submitted"

campaign transition --to-state SUBMITTED \
    --reason "extraction and head jobs submitted" > /dev/null

HEADS_JOB_RAW="$(sbatch --parsable \
    --dependency="afterok:${EXTRACT_JOB_ID}" \
    --export="$EXPORT_COMMON,ENV_ACTIVATE=$QWEN_ENV" \
    "$PROJECT_ROOT/scripts/run_gemma4_hidden_heads_slurm.sh")"
HEADS_JOB_ID="$(printf '%s' "$HEADS_JOB_RAW" | tail -n 1 | tr -d ' ')"
echo "heads job: $HEADS_JOB_ID (afterok:$EXTRACT_JOB_ID)"

campaign record-job \
    --job-key heads --job-type hidden_classifier --event-type SUBMITTED \
    --slurm-job-id "$HEADS_JOB_ID" --status PENDING \
    --dependency-job-ids "$EXTRACT_JOB_ID" \
    --reason "heads sbatch submitted afterok extraction"

echo "submitted attempt job graph: extract=$EXTRACT_JOB_ID heads=$HEADS_JOB_ID"
