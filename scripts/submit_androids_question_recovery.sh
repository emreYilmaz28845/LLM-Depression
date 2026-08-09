#!/usr/bin/env bash
# Submit one Androids interviewer-context recovery job.
#
# Required:
#   RUN_ID=<unique identifier>
#
# Optional:
#   LIMIT=8             smoke only; empty means all 874 contexts
#   DRY_RUN=1           print the exact sbatch command
#   RESUME=1            continue an interrupted output
#   OVERWRITE=1         replace an incompatible existing output
#   BATCH_SIZE=16

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
RUN_ID="${RUN_ID:?Set RUN_ID}"
LIMIT="${LIMIT:-}"
DRY_RUN="${DRY_RUN:-0}"
RESUME="${RESUME:-0}"
OVERWRITE="${OVERWRITE:-0}"
BATCH_SIZE="${BATCH_SIZE:-16}"
OUTPUT_DIR="$PROJECT_ROOT/outputs/androids_question_recovery/$RUN_ID"
LOG_ROOT="$PROJECT_ROOT/logs/slurm_androids_question_recovery/$RUN_ID"

if [[ "$RESUME" != "1" && "$OVERWRITE" != "1" && -e "$OUTPUT_DIR" ]]; then
    echo "Refusing existing output directory without RESUME=1 or OVERWRITE=1: $OUTPUT_DIR" >&2
    exit 1
fi

mkdir -p "$LOG_ROOT"

EXPORTS="ALL,PROJECT_ROOT=$PROJECT_ROOT,RUN_ID=$RUN_ID,LIMIT=$LIMIT,RESUME=$RESUME,OVERWRITE=$OVERWRITE,BATCH_SIZE=$BATCH_SIZE"
COMMAND=(
    sbatch
    --parsable
    --export="$EXPORTS"
    "$PROJECT_ROOT/scripts/run_androids_question_recovery_slurm.sh"
)

printf 'submission:'
printf ' %q' "${COMMAND[@]}"
printf '\n'
echo "output_dir=$OUTPUT_DIR"
echo "log_dir=$LOG_ROOT"

if [[ "$DRY_RUN" == "1" ]]; then
    exit 0
fi

JOB_ID="$("${COMMAND[@]}")"
echo "job_id=$JOB_ID"
