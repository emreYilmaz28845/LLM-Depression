#!/usr/bin/env bash
# CPU-only merged LogReg or Optuna-100 head for one merged fold.
#SBATCH -J native-en-merged-head
#SBATCH -A etur92
#SBATCH -q acc_ehpc
#SBATCH -t 24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
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
FEATURES_DIR="${FEATURES_DIR:?FEATURES_DIR is required}"
STAGE="${STAGE:?STAGE is required}"
FOLD="${FOLD:?FOLD is required}"
RUN_ID="${RUN_ID:?RUN_ID is required}"
METHOD="${METHOD:?METHOD is required}"
TRIALS="${TRIALS:-}"
OVERRIDES_JSON_B64="${OVERRIDES_JSON_B64:-}"
LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/logs/native_en_text_heads_v2/merged_head}"

source "$PROJECT_ROOT/scripts/native_en_text_heads_env.sh"
source "$QWEN_ENV_ACTIVATE"
cd "$PROJECT_ROOT"
mkdir -p "$LOG_ROOT"
exec > >(tee -a "$LOG_ROOT/${METHOD}-${SLURM_JOB_ID}.out")
exec 2> >(tee -a "$LOG_ROOT/${METHOD}-${SLURM_JOB_ID}.err" >&2)

WORKER=(python "$PROJECT_ROOT/tools/native_en_text_heads_worker.py")
if [ ! -f "$ATTEMPT_DIR/metadata.json" ]; then
    "${WORKER[@]}" init --attempt-dir "$ATTEMPT_DIR" --context "$CONTEXT_JSON" --config "$CONFIG_JSON" --parent "$PARENT_JSON"
fi
on_error() {
    code=$?
    set +e
    "${WORKER[@]}" record --attempt-dir "$ATTEMPT_DIR" --job-key head --job-type hidden_classifier \
        --event-type FAILED --slurm-job-id "${SLURM_JOB_ID:-}" --status FAILED \
        --reason "worker exit $code" --exit-code "${code}:0"
    "${WORKER[@]}" transition --attempt-dir "$ATTEMPT_DIR" --to-state FAILED --reason "merged head worker failed"
    exit "$code"
}
trap on_error ERR

"${WORKER[@]}" transition --attempt-dir "$ATTEMPT_DIR" --to-state RUNNING --reason "merged head worker started"
"${WORKER[@]}" record --attempt-dir "$ATTEMPT_DIR" --job-key head --job-type hidden_classifier \
    --event-type STARTED --slurm-job-id "${SLURM_JOB_ID:-}" --status RUNNING

OVERRIDE_ARGS=()
if [ -n "$OVERRIDES_JSON_B64" ]; then
    mapfile -t OVERRIDE_ARGS < <(python - "$OVERRIDES_JSON_B64" <<'PY'
import base64, json, sys
for token in json.loads(base64.b64decode(sys.argv[1]).decode("utf-8")):
    print(token)
PY
)
fi
CMD=(python -m src.merged.heads --config "$CONFIG" --stage "$STAGE" --fold "$FOLD" \
    --run-id "$RUN_ID" --features-dir "$FEATURES_DIR" --method "$METHOD" \
    --output-root "$ATTEMPT_DIR")
if [ -n "$TRIALS" ]; then CMD+=(--trials "$TRIALS"); fi
for token in "${OVERRIDE_ARGS[@]}"; do CMD+=(--override="$token"); done
echo "Merged head command: ${CMD[*]}"
"${CMD[@]}"

PREDICTIONS="$ATTEMPT_DIR/predictions_subject_level.jsonl"
METRICS="$ATTEMPT_DIR/metrics_by_dataset.json"
"${WORKER[@]}" materialize --attempt-dir "$ATTEMPT_DIR" --predictions "$PREDICTIONS" \
    --metrics "$METRICS" --checkpoint-path "$(python - "$PARENT_JSON" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["parent_checkpoint_path"])
PY
)"
"${WORKER[@]}" record --attempt-dir "$ATTEMPT_DIR" --job-key head --job-type hidden_classifier \
    --event-type COMPLETED --slurm-job-id "${SLURM_JOB_ID:-}" --status COMPLETED --exit-code 0:0
echo "v2 merged $METHOD worker completed: attempt=$ATTEMPT_DIR"
