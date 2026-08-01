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

hash_file() {
    local path="$1"
    local prefix="$2"
    [[ -f "$path" ]] || {
        echo "Cannot hash missing result file: $path" >&2
        return 1
    }
    local digest
    digest="$(sha256sum -- "$path" | awk '{print $1}')"
    printf '%s  %s\n' "$digest" "$prefix"
}

hash_tree() {
    local root="$1"
    local prefix="$2"
    [[ -d "$root" ]] || {
        echo "Cannot hash missing result directory: $root" >&2
        return 1
    }
    (
        cd "$root"
        find . -type f \
            ! -path '*/best_model/*' \
            ! -path '*/last_model/*' \
            ! -name '*.safetensors' \
            ! -name '*.bin' \
            ! -name '*.pt' \
            ! -path '*/features/*.npz' \
            ! -name 'classifier.joblib' \
            ! -name '*.db' \
            -print0 \
            | sort -z \
            | while IFS= read -r -d '' relative_path; do
                local digest
                digest="$(sha256sum -- "$relative_path" | awk '{print $1}')"
                printf '%s  %s/%s\n' "$digest" "$prefix" "${relative_path#./}"
            done
    )
}

collect_hashes() {
    local project_root="$1"
    local run_id="$2"
    {
        for modality in "${MODALITIES[@]}"; do
            hash_file "$project_root/outputs/symmetric_merged/$modality/merged_manifest.jsonl" "outputs/symmetric_merged/$modality/merged_manifest.jsonl"
            hash_file "$project_root/outputs/symmetric_merged/$modality/merged_protocol.json" "outputs/symmetric_merged/$modality/merged_protocol.json"
            hash_tree "$project_root/outputs/symmetric_merged/$modality/$run_id" "outputs/symmetric_merged/$modality/$run_id"
            hash_tree "$project_root/output_model/symmetric_merged/$modality/$run_id" "output_model/symmetric_merged/$modality/$run_id"
        done
        hash_file "$project_root/outputs/symmetric_merged_jobs/$run_id.json" "outputs/symmetric_merged_jobs/$run_id.json"
        hash_tree "$project_root/logs/symmetric_merged" "logs/symmetric_merged"
    } | LC_ALL=C sort
}

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
    hash_tmp_dir="$(mktemp -d)"
    trap 'rm -rf "$hash_tmp_dir"' EXIT
    remote_hashes="$hash_tmp_dir/remote.sha256"
    local_hashes="$hash_tmp_dir/local.sha256"
    ssh "$TRANSFER_HOST" bash -s -- "$REMOTE_PROJECT_ROOT" "$RUN_ID" > "$remote_hashes" <<'REMOTE_HASH'
set -euo pipefail
REMOTE_PROJECT_ROOT="$1"
RUN_ID="$2"
MODALITIES=(audio_text audio_only text_only)

hash_file() {
    local path="$1"
    local prefix="$2"
    [[ -f "$path" ]] || { echo "Cannot hash missing result file: $path" >&2; return 1; }
    local digest
    digest="$(sha256sum -- "$path" | awk '{print $1}')"
    printf '%s  %s\n' "$digest" "$prefix"
}

hash_tree() {
    local root="$1"
    local prefix="$2"
    [[ -d "$root" ]] || { echo "Cannot hash missing result directory: $root" >&2; return 1; }
    (
        cd "$root"
        find . -type f \
            ! -path '*/best_model/*' \
            ! -path '*/last_model/*' \
            ! -name '*.safetensors' \
            ! -name '*.bin' \
            ! -name '*.pt' \
            ! -path '*/features/*.npz' \
            ! -name 'classifier.joblib' \
            ! -name '*.db' \
            -print0 \
            | sort -z \
            | while IFS= read -r -d '' relative_path; do
                digest="$(sha256sum -- "$relative_path" | awk '{print $1}')"
                printf '%s  %s/%s\n' "$digest" "$prefix" "${relative_path#./}"
            done
    )
}

collect_hashes() {
    {
        for modality in "${MODALITIES[@]}"; do
            hash_file "$REMOTE_PROJECT_ROOT/outputs/symmetric_merged/$modality/merged_manifest.jsonl" "outputs/symmetric_merged/$modality/merged_manifest.jsonl"
            hash_file "$REMOTE_PROJECT_ROOT/outputs/symmetric_merged/$modality/merged_protocol.json" "outputs/symmetric_merged/$modality/merged_protocol.json"
            hash_tree "$REMOTE_PROJECT_ROOT/outputs/symmetric_merged/$modality/$RUN_ID" "outputs/symmetric_merged/$modality/$RUN_ID"
            hash_tree "$REMOTE_PROJECT_ROOT/output_model/symmetric_merged/$modality/$RUN_ID" "output_model/symmetric_merged/$modality/$RUN_ID"
        done
        hash_file "$REMOTE_PROJECT_ROOT/outputs/symmetric_merged_jobs/$RUN_ID.json" "outputs/symmetric_merged_jobs/$RUN_ID.json"
        hash_tree "$REMOTE_PROJECT_ROOT/logs/symmetric_merged" "logs/symmetric_merged"
    } | LC_ALL=C sort
}

collect_hashes
REMOTE_HASH
    collect_hashes "$LOCAL_PROJECT_ROOT" "$RUN_ID" > "$local_hashes"
    missing_hashes="$(comm -23 "$remote_hashes" "$local_hashes")"
    if [[ -n "$missing_hashes" ]]; then
        echo "Transferred result checksum verification failed; remote entries missing or mismatched locally:" >&2
        printf '%s\n' "$missing_hashes" >&2
        exit 1
    fi
    cp "$local_hashes" "$LOCAL_PROJECT_ROOT/outputs/symmetric_merged_jobs/${RUN_ID}_local_result_hashes.sha256"
    echo "Retrieved and checksum-verified symmetric merged reporting artifacts for $RUN_ID without heavy model/feature artifacts."
fi
