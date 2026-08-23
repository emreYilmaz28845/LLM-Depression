#!/usr/bin/env bash
# CPU-only fixed Optuna-100 (or two-trial smoke) head for one hidden cache.
#SBATCH -J tqcond-xgb
#SBATCH -A etur92
#SBATCH -q acc_ehpc
#SBATCH -t 24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#SBATCH --chdir=/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression

set -Eeuo pipefail
module purge
module load bsc/1.0
module load miniforge/24.3.0-0

PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
ATTEMPT_DIR="${ATTEMPT_DIR:?ATTEMPT_DIR is required}"
CONTEXT_JSON="${CONTEXT_JSON:?CONTEXT_JSON is required}"
CONFIG_JSON="${CONFIG_JSON:?CONFIG_JSON is required}"
PARENT_JSON="${PARENT_JSON:?PARENT_JSON is required}"
CACHE_DIR="${CACHE_DIR:?CACHE_DIR is required}"
TRIALS="${TRIALS:?TRIALS is required}"
STAGE="${STAGE:?STAGE is required}"
BACKBONE="${BACKBONE:?BACKBONE is required}"
LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/experiment_runtime/tqcond/logs/xgb}"
QWEN_ENV_ACTIVATE="${QWEN_ENV_ACTIVATE:-/gpfs/projects/etur92/ozu647717/venvs/qwen_mn5_rebuilt/bin/activate}"

# XGBoost and Optuna are run in the project environment; this worker has no
# GPU request and remains downstream of the one-GPU hidden extraction job.
# shellcheck disable=SC1090
source "$QWEN_ENV_ACTIVATE"
cd "$PROJECT_ROOT"
mkdir -p "$LOG_ROOT"
exec > >(tee -a "$LOG_ROOT/xgb-${SLURM_JOB_ID:-local}.out")
exec 2> >(tee -a "$LOG_ROOT/xgb-${SLURM_JOB_ID:-local}.err" >&2)

WORKER=(python "$PROJECT_ROOT/tools/turkish_question_condition_worker.py")
if [ ! -f "$ATTEMPT_DIR/metadata.json" ]; then
    "${WORKER[@]}" init --attempt-dir "$ATTEMPT_DIR" --context "$CONTEXT_JSON" --config "$CONFIG_JSON" --parent "$PARENT_JSON"
fi

on_error() {
    code=$?
    set +e
    "${WORKER[@]}" record --attempt-dir "$ATTEMPT_DIR" --job-key head --job-type hidden_classifier \
        --event-type FAILED --slurm-job-id "${SLURM_JOB_ID:-}" --status FAILED \
        --reason "worker exit $code" --exit-code "${code}:0" || true
    "${WORKER[@]}" transition --attempt-dir "$ATTEMPT_DIR" --to-state FAILED \
        --reason "Turkish question-condition Optuna worker failed" || true
    exit "$code"
}
trap on_error ERR

"${WORKER[@]}" transition --attempt-dir "$ATTEMPT_DIR" --to-state RUNNING --reason "Optuna XGBoost worker started"
"${WORKER[@]}" record --attempt-dir "$ATTEMPT_DIR" --job-key head --job-type hidden_classifier \
    --event-type STARTED --slurm-job-id "${SLURM_JOB_ID:-}" --status RUNNING

export NATIVE_EN_TEXT_HEADS_STAGE="$STAGE"
export NATIVE_EN_TEXT_HEADS_V2_ATTEMPT_DIR="$ATTEMPT_DIR"
OUT_DIR="$ATTEMPT_DIR/xgb_optuna100_harmonized_v1"
python "$PROJECT_ROOT/baselines/qwen_hidden_xgb_optuna.py" \
    --cache-dir "$CACHE_DIR" --output-dir "$OUT_DIR" --objective macro_f1 \
    --target-trials "$TRIALS" --inner-folds 3 --seed 1337 --inner-seed 1337 \
    --sampling-mode none --experiment-id xgb_optuna100_harmonized_v1 \
    --protocol-profile harmonized_optuna100_v1 --xgb-threads 20

PREDICTIONS="$OUT_DIR/predictions_subject_level.jsonl"
METRICS="$OUT_DIR/metrics.json"
CHECKPOINT_DIR="$(python - "$PARENT_JSON" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["parent_checkpoint_path"])
PY
)"
"${WORKER[@]}" materialize --attempt-dir "$ATTEMPT_DIR" --predictions "$PREDICTIONS" \
    --metrics "$METRICS" --checkpoint-path "$CHECKPOINT_DIR"
"${WORKER[@]}" record --attempt-dir "$ATTEMPT_DIR" --job-key head --job-type hidden_classifier \
    --event-type COMPLETED --slurm-job-id "${SLURM_JOB_ID:-}" --status COMPLETED --exit-code 0:0
echo "Turkish question-condition XGBoost worker completed: attempt=$ATTEMPT_DIR trials=$TRIALS"
