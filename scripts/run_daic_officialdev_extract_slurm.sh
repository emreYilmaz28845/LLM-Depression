#!/bin/bash
#SBATCH -J daic-odv-extract
#SBATCH -A etur92
#SBATCH -q acc_ehpc
#SBATCH -t 24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#SBATCH --chdir=/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression

set -euo pipefail
module purge
module load bsc/1.0
module load miniforge/24.3.0-0

PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
# Backend-correct environment: the launcher exports ENV_ACTIVATE per cell
# (gemma4_12b_tf5_14_1 for Gemma, qwen_mn5_rebuilt for Qwen).
ENV_ACTIVATE="${ENV_ACTIVATE:?ENV_ACTIVATE is required}"
ATTEMPT_DIR="${ATTEMPT_DIR:?ATTEMPT_DIR is required}"
PARENT_FOLD_DIR="${PARENT_FOLD_DIR:?PARENT_FOLD_DIR is required}"
PARENT_ATTEMPT_ID="${PARENT_ATTEMPT_ID:-}"
MODEL_PATH="${MODEL_PATH:?MODEL_PATH is required}"
MODALITY="${MODALITY:?MODALITY is required}"
BACKBONE="${BACKBONE:?BACKBONE is required}"
RUN_NAME="${RUN_NAME:?RUN_NAME is required}"
GROUP_ID="${GROUP_ID:?GROUP_ID is required}"
MERGED_SHA="${MERGED_SHA:?MERGED_SHA is required}"
BRANCH="${BRANCH:-main}"
PR_NUMBER="${PR_NUMBER:-}"
CONDITION="${CONDITION:-}"
SUBJECT_SELECTION="${SUBJECT_SELECTION:-}"

if [ ! -f "$ENV_ACTIVATE" ]; then
    echo "Environment activate script not found: $ENV_ACTIVATE" >&2
    exit 1
fi
# shellcheck disable=SC1090
source "$ENV_ACTIVATE"
cd "$PROJECT_ROOT"
export PROJECT_ROOT

# MN5 has no outbound internet: force offline everywhere and never fall back
# to a remote package/model API.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

LOG_ROOT="$PROJECT_ROOT/logs/daic_officialdev"
mkdir -p "$LOG_ROOT"
OUT_LOG="$LOG_ROOT/extract-${SLURM_JOB_ID}.out"
ERR_LOG="$LOG_ROOT/extract-${SLURM_JOB_ID}.err"
exec > >(tee -a "$OUT_LOG")
exec 2> >(tee -a "$ERR_LOG" >&2)

campaign_record() {
    python tools/gemma4_hidden_campaign.py "$1" --attempt-dir "$ATTEMPT_DIR" "${@:2}"
}
campaign_transition() {
    local to_state="$1"
    local reason="$2"
    if ! python tools/gemma4_hidden_campaign.py transition --attempt-dir "$ATTEMPT_DIR" \
        --to-state "$to_state" --reason "$reason" > /dev/null 2>&1; then
        # The job can start before the submission script records SUBMITTED.
        # A STARTED event already proves submission, so normalize the state
        # first and then retry the requested transition.
        python tools/gemma4_hidden_campaign.py transition --attempt-dir "$ATTEMPT_DIR" \
            --to-state SUBMITTED --reason "job start implies submission" > /dev/null 2>&1 || true
        python tools/gemma4_hidden_campaign.py transition --attempt-dir "$ATTEMPT_DIR" \
            --to-state "$to_state" --reason "$reason" > /dev/null
    fi
}

PR_ARGS=()
if [ -n "$PR_NUMBER" ]; then
    PR_ARGS+=(--pr-number "$PR_NUMBER")
fi
if [ -z "$PARENT_ATTEMPT_ID" ]; then
    # The parent attempt ID is minted when the training job starts; read it
    # from the completed parent fold's metadata.json.
    PARENT_ATTEMPT_ID="$(python - "$PARENT_FOLD_DIR" <<'PY'
import json, sys
from pathlib import Path
metadata_path = Path(sys.argv[1]) / "metadata.json"
if not metadata_path.is_file():
    raise SystemExit("parent fold has no metadata.json; PARENT_ATTEMPT_ID is required")
attempt_id = json.loads(metadata_path.read_text(encoding="utf-8")).get("attempt_id")
if not attempt_id:
    raise SystemExit("parent metadata.json has no attempt_id")
print(attempt_id)
PY
)"
fi
python tools/gemma4_hidden_campaign.py create-attempt \
    --repo-root "$PROJECT_ROOT" \
    --attempt-dir "$ATTEMPT_DIR" \
    --modality "$MODALITY" \
    --run-name "$RUN_NAME" \
    --group-id "$GROUP_ID" \
    --parent-fold-dir "$PARENT_FOLD_DIR" \
    --parent-attempt-id "$PARENT_ATTEMPT_ID" \
    --merged-sha "$MERGED_SHA" \
    --branch "$BRANCH" \
    --backbone "$BACKBONE" \
    "${PR_ARGS[@]}"

campaign_record record-job \
    --job-key extract --job-type hidden_extraction --event-type SUBMITTED \
    --slurm-job-id "${SLURM_JOB_ID:-}" --status PENDING \
    --reason "extraction job submitted by campaign launcher"
campaign_transition SUBMITTED "extraction job submitted"
campaign_record record-job \
    --job-key extract --job-type hidden_extraction --event-type STARTED \
    --slurm-job-id "${SLURM_JOB_ID:-}" --status RUNNING \
    --reason "extraction job started on ${SLURMD_NODENAME:-unknown}"
campaign_transition RUNNING "extraction job started"

cleanup() {
    local exit_code=$?
    if [ "$exit_code" -eq 0 ]; then
        campaign_record record-job \
            --job-key extract --job-type hidden_extraction --event-type COMPLETED \
            --slurm-job-id "${SLURM_JOB_ID:-}" --status COMPLETED \
            --reason "extraction job completed"
    else
        campaign_record record-job \
            --job-key extract --job-type hidden_extraction --event-type FAILED \
            --slurm-job-id "${SLURM_JOB_ID:-}" --status FAILED \
            --reason "extraction job failed with exit $exit_code"
        campaign_transition FAILED "extraction job failed" || true
    fi
    exit "$exit_code"
}
trap cleanup EXIT

CMD=(python src/features/extract_qwen_hidden.py
    --checkpoint-dir "$PARENT_FOLD_DIR/best_model"
    --output-dir "$ATTEMPT_DIR/hidden_features"
    --model-name-or-path "$MODEL_PATH"
    --condition "${CONDITION:-daic_officialdev}")
if [ -n "$SUBJECT_SELECTION" ]; then
    # Smoke-only: hashes into the cache identity so a smoke cache can never
    # collide with a production cache.
    CMD+=(--subject-selection "$SUBJECT_SELECTION")
fi
"${CMD[@]}"
