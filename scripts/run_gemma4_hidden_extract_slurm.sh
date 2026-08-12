#!/bin/bash
#SBATCH -J gemma4-hid-ext
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
ENV_ACTIVATE="${ENV_ACTIVATE:-/gpfs/projects/etur92/ozu647717/venvs/gemma4_12b_tf5_14_1/bin/activate}"
ATTEMPT_DIR="${ATTEMPT_DIR:?ATTEMPT_DIR is required}"
PARENT_FOLD_DIR="${PARENT_FOLD_DIR:?PARENT_FOLD_DIR is required}"
MODEL_PATH="${MODEL_PATH:?MODEL_PATH is required}"
SUBJECT_SELECTION="${SUBJECT_SELECTION:-}"

if [ ! -f "$ENV_ACTIVATE" ]; then
    echo "Gemma environment activate script not found: $ENV_ACTIVATE" >&2
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

LOG_ROOT="$PROJECT_ROOT/logs/gemma4_hidden_fixed_heads"
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
    --condition "gemma4_daic_fixed_heads")
if [ -n "$SUBJECT_SELECTION" ]; then
    CMD+=(--subject-selection "$SUBJECT_SELECTION")
fi
"${CMD[@]}"
