#!/bin/bash
set -euo pipefail

# Submit the native-versus-English preflight audit job (CPU-only).

RUN_ID="${RUN_ID:?Set RUN_ID}"
DRY_RUN="${DRY_RUN:-1}"
CODE="${CODE:?Set CODE to the deployed code path}"
SOURCE_COMMIT="${SOURCE_COMMIT:?Set SOURCE_COMMIT}"
PERMANENT="${PERMANENT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
PREFLIGHT_ROOT="${PREFLIGHT_ROOT:-$PERMANENT/outputs/native_en_text_heads_preflight}"
BUILD_MERGED_PROTOCOLS="${BUILD_MERGED_PROTOCOLS:-1}"
WITH_TOKENIZER="${WITH_TOKENIZER:-0}"
RUN_NAMES_FILE="${RUN_NAMES_FILE:-}"
MERGED_RUN_IDS_FILE="${MERGED_RUN_IDS_FILE:-}"

mkdir -p "$PREFLIGHT_ROOT/$RUN_ID"

EXPORT_LINE="ALL,CODE=$CODE,SOURCE_COMMIT=$SOURCE_COMMIT,PERMANENT=$PERMANENT,PREFLIGHT_OUTPUT=$PREFLIGHT_ROOT/$RUN_ID/audit.json,BUILD_MERGED_PROTOCOLS=$BUILD_MERGED_PROTOCOLS,WITH_TOKENIZER=$WITH_TOKENIZER"
[ -n "$RUN_NAMES_FILE" ] && EXPORT_LINE="$EXPORT_LINE,RUN_NAMES_FILE=$RUN_NAMES_FILE"
[ -n "$MERGED_RUN_IDS_FILE" ] && EXPORT_LINE="$EXPORT_LINE,MERGED_RUN_IDS_FILE=$MERGED_RUN_IDS_FILE"

CMD=(sbatch --parsable --chdir="$PERMANENT" --export="$EXPORT_LINE" scripts/run_native_en_preflight_slurm.sh)

if [ "$DRY_RUN" = "1" ]; then
    echo "dry run:"
    printf '  %q ' "${CMD[@]}"
    echo
    echo "output: $PREFLIGHT_ROOT/$RUN_ID/audit.json"
    exit 0
fi

JOB_ID=$("${CMD[@]}")
echo "Submitted preflight job: $JOB_ID"
echo "output: $PREFLIGHT_ROOT/$RUN_ID/audit.json"
