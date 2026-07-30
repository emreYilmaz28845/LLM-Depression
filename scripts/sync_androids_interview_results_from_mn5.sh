#!/bin/bash
set -euo pipefail

DRY_RUN="${DRY_RUN:-1}"
RUN_ID="${RUN_ID:?RUN_ID is required}"
TRANSFER_HOST="${TRANSFER_HOST:-ozu647717@transfer1.bsc.es}"
REMOTE_PROJECT_ROOT="${REMOTE_PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
LOCAL_PROJECT_ROOT="${LOCAL_PROJECT_ROOT:-/home/emre/Projects/AudioLLM/LLM-Depression}"
RUN_VARIANTS=(
    "audio_only"
    "audio_text_segment_aligned"
    "audio_text_full_turn"
    "text_only"
)

if [ "$DRY_RUN" != "0" ] && [ "$DRY_RUN" != "1" ]; then
    echo "DRY_RUN must be 0 or 1." >&2
    exit 1
fi
if [ ! -d "$LOCAL_PROJECT_ROOT" ]; then
    echo "Local project root does not exist: $LOCAL_PROJECT_ROOT" >&2
    exit 1
fi

RSYNC_ARGS=(-av --protect-args --exclude=best_model/ --exclude=last_model/)
if [ "$DRY_RUN" = "1" ]; then
    RSYNC_ARGS+=(--dry-run --itemize-changes)
fi
for variant in "${RUN_VARIANTS[@]}"; do
    remote_run="$REMOTE_PROJECT_ROOT/output_model/experiments/androids_interview/$variant/${RUN_ID}_androids_interview_$variant/"
    local_parent="$LOCAL_PROJECT_ROOT/output_model/experiments/androids_interview/$variant/"
    mkdir -p "$local_parent"
    rsync "${RSYNC_ARGS[@]}" "$TRANSFER_HOST:$remote_run" "$local_parent"
done

ARTIFACT_DIRS=(
    "outputs/manifests_androids_interview/"
    "outputs/splits_androids_interview/"
    "outputs/androids_interview_jobs/"
    "outputs/androids_interview_matrix/$RUN_ID/"
    "logs/slurm_androids_interview/"
)
for relative in "${ARTIFACT_DIRS[@]}"; do
    mkdir -p "$LOCAL_PROJECT_ROOT/$(dirname "$relative")"
    rsync "${RSYNC_ARGS[@]}" \
        "$TRANSFER_HOST:$REMOTE_PROJECT_ROOT/$relative" \
        "$LOCAL_PROJECT_ROOT/$relative"
done

if [ "$DRY_RUN" = "1" ]; then
    echo "Dry run only; no result artifacts were transferred."
else
    echo "Retrieved Androids Interview run $RUN_ID without model checkpoints."
    find "$LOCAL_PROJECT_ROOT/output_model/experiments/androids_interview" \
        -path "*/${RUN_ID}_androids_interview_*/fold_*/eval/best_checkpoint/predictions_subject_level.csv" \
        -type f -print | sort
    find "$LOCAL_PROJECT_ROOT/output_model/experiments/androids_interview" \
        -path "*/${RUN_ID}_androids_interview_*" -type f -print0 \
        | sort -z | xargs -0 sha256sum \
        > "$LOCAL_PROJECT_ROOT/outputs/androids_interview_matrix/$RUN_ID/local_result_hashes.sha256"
fi
