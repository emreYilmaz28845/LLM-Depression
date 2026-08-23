#!/usr/bin/env bash
# One-GPU hidden extraction plus fixed raw Logistic Regression for one fold.
# The attempt is a child of the fresh backbone fold and keeps compact evidence
# separate from the backbone output.
#SBATCH -J tqcond-logreg
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

set -Eeuo pipefail
module purge
module load bsc/1.0
module load miniforge/24.3.0-0

PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
CONFIG="${CONFIG:?CONFIG is required}"
ATTEMPT_DIR="${ATTEMPT_DIR:?ATTEMPT_DIR is required}"
CONTEXT_JSON="${CONTEXT_JSON:?CONTEXT_JSON is required}"
CONFIG_JSON="${CONFIG_JSON:?CONFIG_JSON is required}"
PARENT_JSON="${PARENT_JSON:?PARENT_JSON is required}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:?CHECKPOINT_DIR is required}"
CACHE_DIR="${CACHE_DIR:-$ATTEMPT_DIR/hidden_cache}"
CONDITION="${CONDITION:?CONDITION is required}"
BACKBONE="${BACKBONE:?BACKBONE is required}"
MODEL_PATH="${MODEL_PATH:-}"
LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/experiment_runtime/tqcond/logs/logreg}"
QWEN_ENV_ACTIVATE="${QWEN_ENV_ACTIVATE:-/gpfs/projects/etur92/ozu647717/venvs/qwen_mn5_rebuilt/bin/activate}"
GEMMA_ENV_ACTIVATE="${GEMMA_ENV_ACTIVATE:-/gpfs/projects/etur92/ozu647717/venvs/gemma4_12b_tf5_14_1/bin/activate}"
QWEN_DEPS_ROOT="${QWEN_DEPS_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression/.deps/qwen_hidden}"

if [ "$BACKBONE" = "gemma4" ]; then
    # shellcheck disable=SC1090
    source "$GEMMA_ENV_ACTIVATE"
else
    # shellcheck disable=SC1090
    source "$QWEN_ENV_ACTIVATE"
fi
cd "$PROJECT_ROOT"
export PYTHONPATH="$QWEN_DEPS_ROOT:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$LOG_ROOT" "$CACHE_DIR"
exec > >(tee -a "$LOG_ROOT/logreg-${SLURM_JOB_ID:-local}.out")
exec 2> >(tee -a "$LOG_ROOT/logreg-${SLURM_JOB_ID:-local}.err" >&2)

WORKER=(python "$PROJECT_ROOT/tools/turkish_question_condition_worker.py")
if [ ! -f "$ATTEMPT_DIR/metadata.json" ]; then
    "${WORKER[@]}" init --attempt-dir "$ATTEMPT_DIR" --context "$CONTEXT_JSON" --config "$CONFIG_JSON" --parent "$PARENT_JSON"
fi

on_error() {
    code=$?
    set +e
    "${WORKER[@]}" record --attempt-dir "$ATTEMPT_DIR" --job-key head --job-type hidden_extraction \
        --event-type FAILED --slurm-job-id "${SLURM_JOB_ID:-}" --status FAILED \
        --reason "worker exit $code" --exit-code "${code}:0" || true
    "${WORKER[@]}" transition --attempt-dir "$ATTEMPT_DIR" --to-state FAILED \
        --reason "Turkish question-condition LogReg worker failed" || true
    exit "$code"
}
trap on_error ERR

"${WORKER[@]}" transition --attempt-dir "$ATTEMPT_DIR" --to-state RUNNING --reason "hidden extraction and LogReg worker started"
"${WORKER[@]}" record --attempt-dir "$ATTEMPT_DIR" --job-key head --job-type hidden_extraction \
    --event-type STARTED --slurm-job-id "${SLURM_JOB_ID:-}" --status RUNNING

EXTRACT=(python "$PROJECT_ROOT/src/features/extract_qwen_hidden.py" \
    --checkpoint-dir "$CHECKPOINT_DIR" --output-dir "$CACHE_DIR" --condition "$CONDITION")
if [ -n "$MODEL_PATH" ]; then EXTRACT+=(--model-name-or-path "$MODEL_PATH"); fi
echo "Extraction command: ${EXTRACT[*]}"
"${EXTRACT[@]}"

python "$PROJECT_ROOT/baselines/qwen_hidden_classifier.py" \
    --cache-dir "$CACHE_DIR" --output-dir "$ATTEMPT_DIR/classifier" \
    --variants logreg_raw --seed 1337 --sampling-mode none \
    --protocol-backend-mode native_en_text_heads_v2

PREDICTIONS="$ATTEMPT_DIR/classifier/logreg_raw/predictions_subject_level.jsonl"
METRICS="$ATTEMPT_DIR/classifier/logreg_raw/metrics.json"
"${WORKER[@]}" materialize --attempt-dir "$ATTEMPT_DIR" --predictions "$PREDICTIONS" \
    --metrics "$METRICS" --checkpoint-path "$CHECKPOINT_DIR"
"${WORKER[@]}" record --attempt-dir "$ATTEMPT_DIR" --job-key head --job-type hidden_extraction \
    --event-type COMPLETED --slurm-job-id "${SLURM_JOB_ID:-}" --status COMPLETED --exit-code 0:0
echo "Turkish question-condition LogReg worker completed: attempt=$ATTEMPT_DIR"
