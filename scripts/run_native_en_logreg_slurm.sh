#!/usr/bin/env bash
# One-GPU hidden extraction plus raw Logistic Regression for one v2 fold.
# The classifier is deliberately a separate managed attempt from the backbone.
#SBATCH -J native-en-logreg
#SBATCH -A etur92
#SBATCH -q acc_ehpc
#SBATCH -t 48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
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
CONFIG="${CONFIG:?CONFIG is required}"
ATTEMPT_DIR="${ATTEMPT_DIR:?ATTEMPT_DIR is required}"
CONTEXT_JSON="${CONTEXT_JSON:?CONTEXT_JSON is required}"
CONFIG_JSON="${CONFIG_JSON:?CONFIG_JSON is required}"
PARENT_JSON="${PARENT_JSON:?PARENT_JSON is required}"
CACHE_DIR="${CACHE_DIR:?CACHE_DIR is required}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:?CHECKPOINT_DIR is required}"
CONDITION="${CONDITION:?CONDITION is required}"
BACKBONE="${BACKBONE:?BACKBONE is required}"
LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/logs/native_en_text_heads_v2/logreg}"

source "$PROJECT_ROOT/scripts/native_en_text_heads_env.sh"
if [ ! -f "$ENV_ACTIVATE" ]; then
    echo "Environment activate script not found: $ENV_ACTIVATE" >&2
    exit 1
fi
# shellcheck disable=SC1090
source "$ENV_ACTIVATE"
cd "$PROJECT_ROOT"
mkdir -p "$LOG_ROOT" "$CACHE_DIR"
exec > >(tee -a "$LOG_ROOT/logreg-${SLURM_JOB_ID}.out")
exec 2> >(tee -a "$LOG_ROOT/logreg-${SLURM_JOB_ID}.err" >&2)

WORKER=("$QWEN_PYTHON" "$PROJECT_ROOT/tools/native_en_text_heads_worker.py")
if [ ! -f "$ATTEMPT_DIR/metadata.json" ]; then
    "${WORKER[@]}" init --attempt-dir "$ATTEMPT_DIR" --context "$CONTEXT_JSON" --config "$CONFIG_JSON" --parent "$PARENT_JSON"
fi

on_error() {
    code=$?
    set +e
    "${WORKER[@]}" record --attempt-dir "$ATTEMPT_DIR" --job-key head --job-type hidden_classifier \
        --event-type FAILED --slurm-job-id "${SLURM_JOB_ID:-}" --status FAILED \
        --reason "worker exit $code" --exit-code "${code}:0"
    "${WORKER[@]}" transition --attempt-dir "$ATTEMPT_DIR" --to-state FAILED --reason "hidden LogReg worker failed"
    exit "$code"
}
trap on_error ERR

"${WORKER[@]}" transition --attempt-dir "$ATTEMPT_DIR" --to-state RUNNING --reason "hidden LogReg worker started"
"${WORKER[@]}" record --attempt-dir "$ATTEMPT_DIR" --job-key head --job-type hidden_classifier \
    --event-type STARTED --slurm-job-id "${SLURM_JOB_ID:-}" --status RUNNING

EXTRACT=(python "$PROJECT_ROOT/src/features/extract_qwen_hidden.py"
    --checkpoint-dir "$CHECKPOINT_DIR" --output-dir "$CACHE_DIR" --condition "$CONDITION")
if [ -n "${MODEL_PATH:-}" ]; then EXTRACT+=(--model-name-or-path "$MODEL_PATH"); fi
echo "Extraction command: ${EXTRACT[*]}"
"${EXTRACT[@]}"

CLASSIFIER_PYTHON="$QWEN_PYTHON"
"$CLASSIFIER_PYTHON" "$PROJECT_ROOT/baselines/qwen_hidden_classifier.py" \
    --cache-dir "$CACHE_DIR" --output-dir "$ATTEMPT_DIR/classifier" \
    --variants logreg_raw --seed 1337 --sampling-mode none \
    --protocol-backend-mode native_en_text_heads_v2

PREDICTIONS="$ATTEMPT_DIR/classifier/logreg_raw/predictions_subject_level.jsonl"
METRICS="$ATTEMPT_DIR/classifier/logreg_raw/metrics.json"
"${WORKER[@]}" materialize --attempt-dir "$ATTEMPT_DIR" --predictions "$PREDICTIONS" \
    --metrics "$METRICS" --checkpoint-path "$CHECKPOINT_DIR"
"${WORKER[@]}" record --attempt-dir "$ATTEMPT_DIR" --job-key head --job-type hidden_classifier \
    --event-type COMPLETED --slurm-job-id "${SLURM_JOB_ID:-}" --status COMPLETED --exit-code 0:0
echo "v2 LogReg worker completed: attempt=$ATTEMPT_DIR"
