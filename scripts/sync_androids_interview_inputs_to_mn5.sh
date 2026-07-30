#!/bin/bash
set -euo pipefail

DRY_RUN="${DRY_RUN:-1}"
TRANSFER_HOST="${TRANSFER_HOST:-ozu647717@transfer1.bsc.es}"
LOCAL_ROOT="${LOCAL_ROOT:-/media/emre/Backup/AudioLLM/Datasets/Androids-Corpus/Androids-Corpus}"
REMOTE_ROOT="${REMOTE_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/Datasets/Androids-Corpus/Androids-Corpus}"
FILES=(
    "interview_transcripts_qwen3_asr_italian.jsonl"
    "interview_transcripts_qwen3_asr_italian.report.json"
    "interview_transcripts_qwen3_asr_italian_segments.jsonl"
    "interview_transcripts_qwen3_asr_italian_segments.report.json"
)

if [ "$DRY_RUN" != "0" ] && [ "$DRY_RUN" != "1" ]; then
    echo "DRY_RUN must be 0 or 1." >&2
    exit 1
fi
for filename in "${FILES[@]}"; do
    if [ ! -s "$LOCAL_ROOT/$filename" ]; then
        echo "Missing local ANDROIDS input: $LOCAL_ROOT/$filename" >&2
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
    ssh "$TRANSFER_HOST" \
        "cd $(printf '%q' "$REMOTE_ROOT") && sha256sum$remote_files && find Interview-Task/audio_clip -type f -name '*.wav' | wc -l"
else
    echo "Dry run only; remote hashes and the required 874-WAV count were not queried."
fi
