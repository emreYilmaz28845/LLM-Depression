#!/usr/bin/env bash
set -euo pipefail

# Retrieve only reproducibility/reporting artifacts. Heavy checkpoints, hidden
# arrays, classifier joblibs, and Optuna databases remain authoritative on GPFS.
DRY_RUN="${DRY_RUN:-1}"
RUN_ID="${RUN_ID:?RUN_ID is required}"
TRANSFER_HOST="${TRANSFER_HOST:-ozu647717@transfer1.bsc.es}"
REMOTE_PROJECT_ROOT="${REMOTE_PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
LOCAL_PROJECT_ROOT="${LOCAL_PROJECT_ROOT:-/home/emre/Projects/AudioLLM/LLM-Depression}"
MODALITIES=(audio_text audio_only text_only)

if [[ "$DRY_RUN" != "0" && "$DRY_RUN" != "1" ]]; then
    echo "DRY_RUN must be 0 or 1." >&2
    exit 1
fi
if [[ ! -d "$LOCAL_PROJECT_ROOT" ]]; then
    echo "Local project root does not exist: $LOCAL_PROJECT_ROOT" >&2
    exit 1
fi

RSYNC_ARGS=(
    -avc --protect-args --itemize-changes
    --exclude='best_model/'
    --exclude='last_model/'
    --exclude='*.safetensors'
    --exclude='*.bin'
    --exclude='*.pt'
    # Keep feature provenance rows for local acceptance audits; exclude only
    # the dense hidden-vector arrays.
    --exclude='features/*.npz'
    --exclude='**/classifier.joblib'
    --exclude='**/*.db'
)
if [[ "$DRY_RUN" == "1" ]]; then
    RSYNC_ARGS+=(--dry-run)
fi

for modality in "${MODALITIES[@]}"; do
    mkdir -p "$LOCAL_PROJECT_ROOT/outputs/symmetric_merged/$modality" \
        "$LOCAL_PROJECT_ROOT/output_model/symmetric_merged/$modality"
    for artifact in merged_manifest.jsonl merged_protocol.json; do
        rsync "${RSYNC_ARGS[@]}" \
            "$TRANSFER_HOST:$REMOTE_PROJECT_ROOT/outputs/symmetric_merged/$modality/$artifact" \
            "$LOCAL_PROJECT_ROOT/outputs/symmetric_merged/$modality/$artifact"
    done
    rsync "${RSYNC_ARGS[@]}" \
        "$TRANSFER_HOST:$REMOTE_PROJECT_ROOT/outputs/symmetric_merged/$modality/$RUN_ID/" \
        "$LOCAL_PROJECT_ROOT/outputs/symmetric_merged/$modality/$RUN_ID/"
    rsync "${RSYNC_ARGS[@]}" \
        "$TRANSFER_HOST:$REMOTE_PROJECT_ROOT/output_model/symmetric_merged/$modality/$RUN_ID/" \
        "$LOCAL_PROJECT_ROOT/output_model/symmetric_merged/$modality/$RUN_ID/"
done

mkdir -p "$LOCAL_PROJECT_ROOT/outputs/symmetric_merged_jobs" "$LOCAL_PROJECT_ROOT/logs/symmetric_merged"
rsync "${RSYNC_ARGS[@]}" \
    "$TRANSFER_HOST:$REMOTE_PROJECT_ROOT/outputs/symmetric_merged_jobs/$RUN_ID.json" \
    "$LOCAL_PROJECT_ROOT/outputs/symmetric_merged_jobs/$RUN_ID.json"
rsync "${RSYNC_ARGS[@]}" \
    "$TRANSFER_HOST:$REMOTE_PROJECT_ROOT/logs/symmetric_merged/" \
    "$LOCAL_PROJECT_ROOT/logs/symmetric_merged/"

if [[ "$DRY_RUN" == "1" ]]; then
    echo "Dry run only; no symmetric merged result artifacts were transferred."
else
    echo "Retrieved symmetric merged reporting artifacts for $RUN_ID without heavy model/feature artifacts."
    find "$LOCAL_PROJECT_ROOT/outputs/symmetric_merged" \
        "$LOCAL_PROJECT_ROOT/output_model/symmetric_merged" \
        -type f -path "*" -not -name '*.npz' -not -name '*.safetensors' \
        -not -name '*.joblib' -not -name '*.db' -print0 \
        | sort -z | xargs -0 sha256sum > "$LOCAL_PROJECT_ROOT/outputs/symmetric_merged_jobs/${RUN_ID}_local_result_hashes.sha256"
fi
