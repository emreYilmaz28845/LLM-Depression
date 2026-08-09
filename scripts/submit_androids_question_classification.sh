#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
RUN_ID="${RUN_ID:?Set a unique RUN_ID}"
SOURCE_RUN_ID="${SOURCE_RUN_ID:-androids_qctx_full_20260809T124731Z}"
LIMIT="${LIMIT:-}"
RESUME="${RESUME:-0}"
DRY_RUN="${DRY_RUN:-0}"
OUT_DIR="$PROJECT_ROOT/outputs/androids_question_classification/$RUN_ID"
LOG_DIR="$PROJECT_ROOT/logs/slurm_androids_question_classification/$RUN_ID"

if [ -e "$OUT_DIR" ] && [ "$RESUME" != "1" ]; then
  echo "Refusing existing output directory without RESUME=1: $OUT_DIR" >&2
  exit 1
fi

CMD=(
  sbatch --parsable
  --export="ALL,PROJECT_ROOT=$PROJECT_ROOT,RUN_ID=$RUN_ID,SOURCE_RUN_ID=$SOURCE_RUN_ID,LIMIT=$LIMIT,RESUME=$RESUME"
  "$PROJECT_ROOT/scripts/run_androids_question_classification_slurm.sh"
)
printf 'submission:'
printf ' %q' "${CMD[@]}"
printf '\noutput_dir=%s\nlog_dir=%s\n' "$OUT_DIR" "$LOG_DIR"
if [ "$DRY_RUN" = "1" ]; then exit 0; fi
mkdir -p "$OUT_DIR" "$LOG_DIR"
JOB_ID="$("${CMD[@]}")"
echo "job_id=$JOB_ID"
