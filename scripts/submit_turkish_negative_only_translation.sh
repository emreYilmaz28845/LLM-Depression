#!/usr/bin/env bash
# Submit an isolated Qwen3.6 translation smoke or the complete negative-only run.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
TRANSLATION_ROOT="${TRANSLATION_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/translations}"
STAGE="${STAGE:-smoke}"
RUN_ID="${RUN_ID:-turkish-negative-only-qwen36-v1}"
DRY_RUN="${DRY_RUN:-1}"
RESUME="${RESUME:-0}"
PRODUCTION_RUN_ROOT="${PRODUCTION_RUN_ROOT:-$TRANSLATION_ROOT/harmonized_en_complete_v1/turkish_negative_only_t17}"
WORKER="${WORKER:-$PROJECT_ROOT/scripts/run_translation_slurm.sh}"
MANIFEST_CONFIG="${MANIFEST_CONFIG:-$PROJECT_ROOT/configs/main/turkish_negative_only_t17_audio_text_harmonized_selmacrof1_tf_qwen3asr.yaml}"

if [ "$DRY_RUN" != "0" ] && [ "$DRY_RUN" != "1" ]; then
  echo "DRY_RUN must be 0 or 1." >&2
  exit 1
fi
if [ "$RESUME" != "0" ] && [ "$RESUME" != "1" ]; then
  echo "RESUME must be 0 or 1." >&2
  exit 1
fi
if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "RUN_ID may contain only letters, numbers, dot, underscore, and hyphen." >&2
  exit 1
fi

case "$STAGE" in
  smoke)
    RUN_ROOT="$TRANSLATION_ROOT/smokes/$RUN_ID"
    UNIT_LIMIT=4
    EXPECTED_UNIT_COUNT=4
    JOB_NAME="tr-neg-en-smoke"
    ;;
  production)
    RUN_ROOT="$PRODUCTION_RUN_ROOT"
    UNIT_LIMIT=0
    EXPECTED_UNIT_COUNT=1170
    JOB_NAME="tr-neg-en-full"
    ;;
  *)
    echo "STAGE must be smoke or production." >&2
    exit 1
    ;;
esac

if [ "$DRY_RUN" = "0" ]; then
  test -f "$WORKER" || { echo "Worker not found: $WORKER" >&2; exit 1; }
  test -f "$MANIFEST_CONFIG" || { echo "Manifest config not found: $MANIFEST_CONFIG" >&2; exit 1; }
  if [ -e "$RUN_ROOT" ] && [ "$RESUME" != "1" ]; then
    echo "Run root already exists; set RESUME=1 only after inspecting it: $RUN_ROOT" >&2
    exit 1
  fi
fi

EXPORTS="ALL,PROJECT_ROOT=$PROJECT_ROOT,DATASET=turkish,MANIFEST_CONFIG=$MANIFEST_CONFIG,TRANSLATION_ROOT=$TRANSLATION_ROOT,TRANSLATION_RUN_ROOT=$RUN_ROOT,UNIT_LIMIT=$UNIT_LIMIT,EXPECTED_UNIT_COUNT=$EXPECTED_UNIT_COUNT,REQUIRE_COMPLETE=1"
COMMAND=(
  sbatch
  --parsable
  --job-name="$JOB_NAME"
  --chdir="$PROJECT_ROOT"
  --export="$EXPORTS"
  "$WORKER"
)

echo "stage=$STAGE"
echo "run_id=$RUN_ID"
echo "run_root=$RUN_ROOT"
echo "expected_units=$EXPECTED_UNIT_COUNT"
echo "model=Qwen/Qwen3.6-27B"
echo "resources=1 node, 1 task, 2 H100s, 40 CPUs, 24 hours"
if [ "$DRY_RUN" = "1" ]; then
  printf 'DRY_RUN'
  printf ' %q' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi

RAW="$("${COMMAND[@]}")"
JOB_ID="${RAW%%;*}"
if [[ ! "$JOB_ID" =~ ^[0-9]+$ ]]; then
  echo "Could not parse Slurm job ID: $RAW" >&2
  exit 1
fi
echo "submitted_job_id=$JOB_ID"
