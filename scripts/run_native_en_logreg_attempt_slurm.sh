#!/bin/bash
#SBATCH -J nat-en-logreg
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

# Native-versus-English standalone hidden-state extraction + raw LogReg head
# with a self-created tracked attempt. One GPU job per parent fold. The
# attempt is created after feature extraction completes so the cache identity
# can be verified before any evidence is written. Classifier fitting and all
# tracking commands always run in the Qwen environment (sklearn lives there);
# only the extraction stage uses the backbone environment.
set -euo pipefail
module purge
module load bsc/1.0
module load miniforge/24.3.0-0

PROJECT_ROOT="${PROJECT_ROOT:?Set PROJECT_ROOT to the deployed code path}"
export PROJECT_ROOT
cd "$PROJECT_ROOT"

ENV_ACTIVATE="${ENV_ACTIVATE:-/gpfs/projects/etur92/ozu647717/venvs/qwen_mn5_rebuilt/bin/activate}"
GEMMA_ENV="${GEMMA_ENV:-/gpfs/projects/etur92/ozu647717/venvs/gemma4_12b_tf5_14_1/bin/activate}"
QWEN_ENV="${QWEN_ENV:-/gpfs/projects/etur92/ozu647717/venvs/qwen_mn5_rebuilt/bin/activate}"
BACKBONE="${BACKBONE:-qwen}"
MODEL_PATH="${MODEL_PATH:-}"
PARENT_FOLD_DIR="${PARENT_FOLD_DIR:?Set PARENT_FOLD_DIR}"
CACHE_DIR="${CACHE_DIR:?Set CACHE_DIR}"
ATTEMPT_DIR="${ATTEMPT_DIR:?Set ATTEMPT_DIR}"
TASK_SPEC_PATH="${TASK_SPEC_PATH:?Set TASK_SPEC_PATH}"
LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/logs/native_en_logreg}"

if [ "$BACKBONE" = "gemma4" ]; then
    if [ ! -f "$GEMMA_ENV" ]; then
        echo "Gemma environment activate script not found: $GEMMA_ENV" >&2
        exit 1
    fi
    : "${MODEL_PATH:?BACKBONE=gemma4 requires MODEL_PATH}"
fi
if [ ! -f "$QWEN_ENV" ]; then
    echo "Qwen environment activate script not found: $QWEN_ENV" >&2
    exit 1
fi

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1

mkdir -p "$LOG_ROOT"
exec > >(tee -a "$LOG_ROOT/logreg-${SLURM_JOB_ID}.out")
exec 2> >(tee -a "$LOG_ROOT/logreg-${SLURM_JOB_ID}.err" >&2)

campaign() {
    python tools/logreg_head_campaign.py "$1" --attempt-dir "$ATTEMPT_DIR" "${@:2}"
}

cleanup() {
    local exit_code=$?
    source "$QWEN_ENV"
    QWEN_HIDDEN_DEPS="${QWEN_HIDDEN_DEPS:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression/.deps/qwen_hidden}"
    export PYTHONPATH="$QWEN_HIDDEN_DEPS:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
    if [ "$exit_code" -eq 0 ]; then
        campaign record-job \
            --job-key logreg --job-type hidden_classifier --event-type COMPLETED \
            --slurm-job-id "${SLURM_JOB_ID:-}" --job-status COMPLETED \
            --reason "logreg attempt completed"
    else
        campaign record-job \
            --job-key logreg --job-type hidden_classifier --event-type FAILED \
            --slurm-job-id "${SLURM_JOB_ID:-}" --job-status FAILED \
            --reason "logreg attempt failed with exit $exit_code" || true
        campaign transition --to-state FAILED \
            --reason "logreg attempt failed" || true
    fi
    exit "$exit_code"
}
trap cleanup EXIT

# --- Stage 1: hidden-state extraction under the backbone environment ---
source "$ENV_ACTIVATE"
mkdir -p "$CACHE_DIR"
EXTRACT_ARGS=(--checkpoint-dir "$PARENT_FOLD_DIR/best_model" --output-dir "$CACHE_DIR")
if [ -n "$MODEL_PATH" ]; then
    EXTRACT_ARGS+=(--model-name-or-path "$MODEL_PATH")
fi
python src/features/extract_qwen_hidden.py "${EXTRACT_ARGS[@]}"

# --- Stage 2: attempt creation, LogReg fit, evidence materialization ---
source "$QWEN_ENV"
QWEN_HIDDEN_DEPS="${QWEN_HIDDEN_DEPS:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression/.deps/qwen_hidden}"
export PYTHONPATH="$QWEN_HIDDEN_DEPS:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

campaign create-attempt --task-spec "$TASK_SPEC_PATH"
campaign mark-deployed --reason "extraction finished on $(hostname)"
campaign record-job \
    --job-key logreg --job-type hidden_classifier --event-type SUBMITTED \
    --slurm-job-id "${SLURM_JOB_ID:-}" --job-status PENDING \
    --reason "self-recorded at job start after extraction"
campaign transition --to-state SUBMITTED --reason "job started"
campaign transition --to-state RUNNING --reason "logreg fit started"

STAGING_DIR="$CACHE_DIR/../logreg_staging_$$"
python baselines/qwen_hidden_classifier.py \
    --cache-dir "$CACHE_DIR" \
    --output-dir "$STAGING_DIR" \
    --variants logreg_raw \
    --seed 1337 \
    --sampling-mode none \
    --head-backend-policy harmonized_hidden_logreg_raw_v1

mv "$STAGING_DIR"/logreg_raw/* "$ATTEMPT_DIR"/
mv "$STAGING_DIR"/variant_summary.json* "$ATTEMPT_DIR"/ 2>/dev/null || true
find "$STAGING_DIR" -mindepth 1 -delete
rmdir "$STAGING_DIR"

campaign materialize-mn5-evidence
