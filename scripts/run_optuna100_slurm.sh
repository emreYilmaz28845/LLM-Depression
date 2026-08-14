#!/bin/bash
#SBATCH -J optuna100
#SBATCH -A etur92
#SBATCH -q acc_ehpc
#SBATCH -t 04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#SBATCH --chdir=/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression

# CPU-only Optuna-100 XGBoost study worker. One study per job, exactly 100
# completed trials under the harmonized_optuna100_v1 protocol. The attempt
# directory is created by the submission wrapper before this job starts; this
# worker records lifecycle/job events, runs the study, materializes evidence,
# and transitions to COMPLETED_ON_MN5.
set -euo pipefail
module purge
module load bsc/1.0
module load miniforge/24.3.0-0

PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
export PROJECT_ROOT
cd "$PROJECT_ROOT"

ENV_ACTIVATE="${ENV_ACTIVATE:-/gpfs/projects/etur92/ozu647717/venvs/qwen_mn5_rebuilt/bin/activate}"
QWEN_HIDDEN_DEPS="${QWEN_HIDDEN_DEPS:-$PROJECT_ROOT/.deps/qwen_hidden}"
ATTEMPT_DIR="${ATTEMPT_DIR:?Set ATTEMPT_DIR}"
CACHE_DIR="${CACHE_DIR:?Set CACHE_DIR}"
EXPERIMENT_ID="${EXPERIMENT_ID:-xgb_optuna100_harmonized_v1}"
OBJECTIVE="${OBJECTIVE:-macro_f1}"
TARGET_TRIALS="${TARGET_TRIALS:-100}"
XGB_THREADS="${XGB_THREADS:-20}"

if [ ! -f "$ENV_ACTIVATE" ]; then
    echo "Qwen environment activate script not found: $ENV_ACTIVATE" >&2
    exit 1
fi
# shellcheck disable=SC1090
source "$ENV_ACTIVATE"
export PYTHONPATH="$QWEN_HIDDEN_DEPS:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# MN5 has no outbound internet: force offline everywhere.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/logs/optuna100}"
mkdir -p "$LOG_ROOT"
exec > >(tee -a "$LOG_ROOT/optuna-${SLURM_JOB_ID}.out")
exec 2> >(tee -a "$LOG_ROOT/optuna-${SLURM_JOB_ID}.err" >&2)

python -c 'import optuna, xgboost, sklearn; print("versions", optuna.__version__, xgboost.__version__, sklearn.__version__)'

campaign_record() {
    python tools/posthoc_head_campaign.py "$1" --attempt-dir "$ATTEMPT_DIR" "${@:2}"
}
campaign_transition() {
    local to_state="$1"
    local reason="$2"
    if ! python tools/posthoc_head_campaign.py transition --attempt-dir "$ATTEMPT_DIR" \
        --to-state "$to_state" --reason "$reason" > /dev/null 2>&1; then
        python tools/posthoc_head_campaign.py transition --attempt-dir "$ATTEMPT_DIR" \
            --to-state SUBMITTED --reason "job start implies submission" > /dev/null 2>&1 || true
        python tools/posthoc_head_campaign.py transition --attempt-dir "$ATTEMPT_DIR" \
            --to-state "$to_state" --reason "$reason" > /dev/null
    fi
}

campaign_record record-job \
    --job-key optuna --job-type hidden_classifier --event-type STARTED \
    --slurm-job-id "${SLURM_JOB_ID:-}" --job-status RUNNING \
    --reason "optuna study started on ${SLURMD_NODENAME:-unknown}"

campaign_transition RUNNING "optuna study started"

cleanup() {
    local exit_code=$?
    if [ "$exit_code" -eq 0 ]; then
        campaign_record record-job \
            --job-key optuna --job-type hidden_classifier --event-type COMPLETED \
            --slurm-job-id "${SLURM_JOB_ID:-}" --job-status COMPLETED \
            --reason "optuna study completed"
        python tools/posthoc_head_campaign.py materialize-mn5-evidence \
            --attempt-dir "$ATTEMPT_DIR"
    else
        campaign_record record-job \
            --job-key optuna --job-type hidden_classifier --event-type FAILED \
            --slurm-job-id "${SLURM_JOB_ID:-}" --job-status FAILED \
            --reason "optuna study failed with exit $exit_code"
        campaign_transition FAILED "optuna study failed" || true
    fi
    exit "$exit_code"
}
trap cleanup EXIT

python baselines/qwen_hidden_xgb_optuna.py \
    --cache-dir "$CACHE_DIR" \
    --output-dir "$ATTEMPT_DIR" \
    --objective "$OBJECTIVE" \
    --target-trials "$TARGET_TRIALS" \
    --inner-folds 3 \
    --seed 1337 \
    --inner-seed 1337 \
    --sampling-mode none \
    --experiment-id "$EXPERIMENT_ID" \
    --protocol-profile harmonized_optuna100_v1 \
    --xgb-threads "$XGB_THREADS"
