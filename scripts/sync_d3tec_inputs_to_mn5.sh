#!/bin/bash
set -euo pipefail

DRY_RUN="${DRY_RUN:-1}"
TRANSFER_HOST="${TRANSFER_HOST:-transfer1}"
LOCAL_ROOT="${LOCAL_ROOT:-/media/emre/Backup/AudioLLM/Datasets/D3TEC DATASET/D3TEC DATASET}"
REMOTE_ROOT="${REMOTE_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/Datasets/D3TEC DATASET/D3TEC DATASET}"
FILES=(
    "transcripts_qwen3_asr_spanish.jsonl"
    "transcripts_qwen3_asr_spanish.report.json"
    "transcripts_qwen3_asr_spanish_segments.jsonl"
    "transcripts_qwen3_asr_spanish_segments.report.json"
)

if [ "$DRY_RUN" != "0" ] && [ "$DRY_RUN" != "1" ]; then
    echo "DRY_RUN must be 0 or 1." >&2
    exit 1
fi
for filename in "${FILES[@]}"; do
    if [ ! -s "$LOCAL_ROOT/$filename" ]; then
        echo "Missing local D3TEC input: $LOCAL_ROOT/$filename" >&2
        exit 1
    fi
done

RSYNC_ARGS=(-av --protect-args)
if [ "$DRY_RUN" = "1" ]; then
    RSYNC_ARGS+=(--dry-run --itemize-changes)
fi
for filename in "${FILES[@]}"; do
    rsync "${RSYNC_ARGS[@]}" \
        "$LOCAL_ROOT/$filename" \
        "$TRANSFER_HOST:$REMOTE_ROOT/$filename"
done

echo "Local SHA-256:"
(
    cd "$LOCAL_ROOT"
    sha256sum "${FILES[@]}"
)
if [ "$DRY_RUN" = "0" ]; then
    printf -v remote_files ' %q' "${FILES[@]}"
    ssh "$TRANSFER_HOST" "cd $(printf '%q' "$REMOTE_ROOT") && sha256sum$remote_files"
else
    echo "Dry run only; remote hashes were not queried."
fi
