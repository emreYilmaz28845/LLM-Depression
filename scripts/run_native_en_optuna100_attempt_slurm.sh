#!/bin/bash
#SBATCH -J nat-en-optuna100
#SBATCH -A etur92
#SBATCH -q acc_ehpc
#SBATCH -t 04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH -o /dev/null
#SBATCH -e /dev/null

# Native-versus-English Optuna-100 XGBoost study worker with a self-created
# tracked attempt. CPU-only; one study per job. The attempt is created at job
# start (the feature cache already exists by then thanks to the Slurm
# dependency on the extraction/postprocess job), so the SUBMITTED event is
# self-recorded with this job's own Slurm ID. Exactly TARGET_TRIALS completed
# trials are enforced by the study code (100 production, 2 smoke).
set -euo pipefail
module purge
module load bsc/1.0
module load miniforge/24.3.0-0

PROJECT_ROOT="${PROJECT_ROOT:?Set PROJECT_ROOT to the deployed code path}"
export PROJECT_ROOT
cd "$PROJECT_ROOT"

ENV_ACTIVATE="${ENV_ACTIVATE:-/gpfs/projects/etur92/ozu647717/venvs/qwen_mn5_rebuilt/bin/activate}"
QWEN_HIDDEN_DEPS="${QWEN_HIDDEN_DEPS:-$PROJECT_ROOT/.deps/qwen_hidden}"
ATTEMPT_DIR="${ATTEMPT_DIR:?Set ATTEMPT_DIR}"
CACHE_DIR="${CACHE_DIR:?Set CACHE_DIR}"
TASK_SPEC_PATH="${TASK_SPEC_PATH:?Set TASK_SPEC_PATH}"
TARGET_TRIALS="${TARGET_TRIALS:-100}"
MODE="${MODE:-standalone}"
MERGED_CONFIG="${MERGED_CONFIG:-}"
STAGE="${STAGE:-cv}"
FOLD="${FOLD:-0}"
RUN_ID="${RUN_ID:-}"
LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/logs/native_en_optuna100}"

if [ ! -f "$ENV_ACTIVATE" ]; then
    echo "Environment activate script not found: $ENV_ACTIVATE" >&2
    exit 1
fi

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

mkdir -p "$LOG_ROOT"
exec > >(tee -a "$LOG_ROOT/optuna-${SLURM_JOB_ID}.out")
exec 2> >(tee -a "$LOG_ROOT/optuna-${SLURM_JOB_ID}.err" >&2)

source "$ENV_ACTIVATE"
export PYTHONPATH="$QWEN_HIDDEN_DEPS:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

python -c 'import optuna, xgboost, sklearn; print("versions", optuna.__version__, xgboost.__version__, sklearn.__version__)'

campaign() {
    python tools/posthoc_head_campaign.py "$1" --attempt-dir "$ATTEMPT_DIR" "${@:2}"
}

cleanup() {
    local exit_code=$?
    if [ "$exit_code" -eq 0 ]; then
        campaign record-job \
            --job-key optuna --job-type hidden_classifier --event-type COMPLETED \
            --slurm-job-id "${SLURM_JOB_ID:-}" --job-status COMPLETED \
            --reason "optuna study completed"
    else
        campaign record-job \
            --job-key optuna --job-type hidden_classifier --event-type FAILED \
            --slurm-job-id "${SLURM_JOB_ID:-}" --job-status FAILED \
            --reason "optuna study failed with exit $exit_code" || true
        campaign transition --to-state FAILED \
            --reason "optuna study failed" || true
    fi
    exit "$exit_code"
}
trap cleanup EXIT

campaign create-attempt --task-spec "$TASK_SPEC_PATH"
campaign mark-deployed --reason "features present; job $(hostname)"
campaign record-job \
    --job-key optuna --job-type hidden_classifier --event-type SUBMITTED \
    --slurm-job-id "${SLURM_JOB_ID:-}" --job-status PENDING \
    --reason "self-recorded at job start"
campaign transition --to-state SUBMITTED --reason "job started"
campaign transition --to-state RUNNING --reason "optuna study started"

if [ "$MODE" = "merged" ]; then
    : "${MERGED_CONFIG:?MODE=merged requires MERGED_CONFIG}"
    : "${RUN_ID:?MODE=merged requires RUN_ID}"
    python src/merged/optuna100.py \
        --features-dir "$CACHE_DIR" \
        --output-dir "$ATTEMPT_DIR" \
        --merged-config "$MERGED_CONFIG" \
        --stage "$STAGE" \
        --fold "$FOLD" \
        --run-id "$RUN_ID" \
        --target-trials "$TARGET_TRIALS"
else
    python baselines/qwen_hidden_xgb_optuna.py \
        --cache-dir "$CACHE_DIR" \
        --output-dir "$ATTEMPT_DIR" \
        --objective macro_f1 \
        --target-trials "$TARGET_TRIALS" \
        --inner-folds 3 \
        --seed 1337 \
        --inner-seed 1337 \
        --sampling-mode none \
        --experiment-id xgb_optuna100_harmonized_v1 \
        --protocol-profile harmonized_optuna100_v1 \
        --xgb-threads 20
fi

campaign materialize-mn5-evidence
