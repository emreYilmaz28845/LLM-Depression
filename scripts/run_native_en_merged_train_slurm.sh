#!/usr/bin/env bash
# Managed four-GPU merged training worker with modern lifecycle evidence.
#SBATCH -J native-en-merged-train
#SBATCH -A etur92
#SBATCH -q acc_ehpc
#SBATCH -t 72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
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
STAGE="${STAGE:?STAGE is required}"
FOLD="${FOLD:?FOLD is required}"
RUN_ID="${RUN_ID:?RUN_ID is required}"
EPOCHS="${EPOCHS:-}"
SUBJECTS_PER_CLASS="${SUBJECTS_PER_CLASS:-}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
OVERRIDES_JSON_B64="${OVERRIDES_JSON_B64:-}"
LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/logs/native_en_text_heads_v2/merged_train}"

source "$PROJECT_ROOT/scripts/native_en_text_heads_env.sh"
source "$ENV_ACTIVATE"
cd "$PROJECT_ROOT"
if [ "$NPROC_PER_NODE" -ne 4 ]; then
    echo "Merged training requires exactly four processes; got $NPROC_PER_NODE" >&2
    exit 1
fi
mkdir -p "$LOG_ROOT"
exec > >(tee -a "$LOG_ROOT/train-${SLURM_JOB_ID}.out")
exec 2> >(tee -a "$LOG_ROOT/train-${SLURM_JOB_ID}.err" >&2)

WORKER=("$QWEN_PYTHON" "$PROJECT_ROOT/tools/native_en_text_heads_worker.py")
if [ ! -f "$ATTEMPT_DIR/metadata.json" ]; then
    "${WORKER[@]}" init --attempt-dir "$ATTEMPT_DIR" --context "$CONTEXT_JSON" --config "$CONFIG_JSON" --parent "$PARENT_JSON"
fi
on_error() {
    code=$?
    set +e
    "${WORKER[@]}" record --attempt-dir "$ATTEMPT_DIR" --job-key train --job-type train \
        --event-type FAILED --slurm-job-id "${SLURM_JOB_ID:-}" --status FAILED \
        --reason "worker exit $code" --exit-code "${code}:0"
    "${WORKER[@]}" transition --attempt-dir "$ATTEMPT_DIR" --to-state FAILED --reason "merged training worker failed"
    exit "$code"
}
trap on_error ERR

"${WORKER[@]}" transition --attempt-dir "$ATTEMPT_DIR" --to-state RUNNING --reason "merged training worker started"
"${WORKER[@]}" record --attempt-dir "$ATTEMPT_DIR" --job-key train --job-type train \
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
CMD=(torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" -m src.merged.train \
    --config "$CONFIG" --stage "$STAGE" --fold "$FOLD" --run-id "$RUN_ID")
if [ -n "$EPOCHS" ]; then CMD+=(--epochs "$EPOCHS"); fi
if [ -n "$SUBJECTS_PER_CLASS" ]; then CMD+=(--subjects-per-class "$SUBJECTS_PER_CLASS"); fi
for token in "${OVERRIDE_ARGS[@]}"; do CMD+=(--override "$token"); done
"${CMD[@]}"

ARTIFACTS=(training_identity.json resolved_merged_config.json slurm_provenance.json training_complete.json \
    logs/training_history.json logs/composition.json logs/selected_checkpoint.json)
ARGS=(job-materialize --attempt-dir "$ATTEMPT_DIR")
for artifact in "${ARTIFACTS[@]}"; do [ -f "$ATTEMPT_DIR/$artifact" ] && ARGS+=(--artifact "$artifact"); done
"${WORKER[@]}" "${ARGS[@]}"
"${WORKER[@]}" record --attempt-dir "$ATTEMPT_DIR" --job-key train --job-type train \
    --event-type COMPLETED --slurm-job-id "${SLURM_JOB_ID:-}" --status COMPLETED --exit-code 0:0
echo "v2 merged training completed: attempt=$ATTEMPT_DIR"
