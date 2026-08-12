#!/bin/bash
#SBATCH -J gemma4-hid-heads
#SBATCH -A etur92
#SBATCH -q acc_ehpc
#SBATCH -t 12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#SBATCH --chdir=/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression

set -euo pipefail
module purge
module load bsc/1.0
module load miniforge/24.3.0-0

PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
ENV_ACTIVATE="${ENV_ACTIVATE:-/gpfs/projects/etur92/ozu647717/venvs/qwen_mn5_rebuilt/bin/activate}"
ATTEMPT_DIR="${ATTEMPT_DIR:?ATTEMPT_DIR is required}"
PARENT_FOLD_DIR="${PARENT_FOLD_DIR:?PARENT_FOLD_DIR is required}"

if [ ! -f "$ENV_ACTIVATE" ]; then
    echo "Qwen environment activate script not found: $ENV_ACTIVATE" >&2
    exit 1
fi
# shellcheck disable=SC1090
source "$ENV_ACTIVATE"
cd "$PROJECT_ROOT"
export PROJECT_ROOT
export PYTHONPATH="$PROJECT_ROOT/.deps/qwen_hidden:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# CPU-only classifier job. No GPU is requested and XGBoost stays on one
# thread (n_jobs=1 in the locked implementation).
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

LOG_ROOT="$PROJECT_ROOT/logs/gemma4_hidden_fixed_heads"
mkdir -p "$LOG_ROOT"
OUT_LOG="$LOG_ROOT/heads-${SLURM_JOB_ID}.out"
ERR_LOG="$LOG_ROOT/heads-${SLURM_JOB_ID}.err"
exec > >(tee -a "$OUT_LOG")
exec 2> >(tee -a "$ERR_LOG" >&2)

CAMPAIGN=(python tools/gemma4_hidden_campaign.py --attempt-dir "$ATTEMPT_DIR")

"${CAMPAIGN[@]}" record-job \
    --job-key heads --job-type hidden_classifier --event-type STARTED \
    --slurm-job-id "${SLURM_JOB_ID:-}" --status RUNNING \
    --reason "fixed-head job started on ${SLURMD_NODENAME:-unknown}"

cleanup() {
    local exit_code=$?
    if [ "$exit_code" -eq 0 ]; then
        "${CAMPAIGN[@]}" record-job \
            --job-key heads --job-type hidden_classifier --event-type COMPLETED \
            --slurm-job-id "${SLURM_JOB_ID:-}" --status COMPLETED \
            --reason "fixed-head job completed"
        "${CAMPAIGN[@]}" materialize-mn5-evidence \
            --attempt-dir "$ATTEMPT_DIR" --parent-fold-dir "$PARENT_FOLD_DIR"
    else
        "${CAMPAIGN[@]}" record-job \
            --job-key heads --job-type hidden_classifier --event-type FAILED \
            --slurm-job-id "${SLURM_JOB_ID:-}" --status FAILED \
            --reason "fixed-head job failed with exit $exit_code"
        "${CAMPAIGN[@]}" transition --to-state FAILED --reason "fixed-head job failed" || true
    fi
    exit "$exit_code"
}
trap cleanup EXIT

python baselines/qwen_hidden_classifier.py \
    --cache-dir "$ATTEMPT_DIR/hidden_features" \
    --output-dir "$ATTEMPT_DIR/hidden_classifiers" \
    --variants logreg_raw xgb_raw \
    --seed 1337 \
    --sampling-mode legacy
