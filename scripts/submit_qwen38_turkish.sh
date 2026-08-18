#!/usr/bin/env bash
# Submit and monitor the Qwen3.8 Turkish question-recovery job
# (runbook sections 17, 20, 21, 22).
#
# Required env:
#   DEPLOYMENT_ID, SOURCE_COMMIT
# Optional env:
#   PROJECT_ROOT, DEPLOY_ROOT, MODEL_DIR, WHEELHOUSE_DIR, VENV_DIR,
#   MODEL_REVISION, TRANSCRIPT_PATH, DRY_RUN (default 0), WAIT (default 1),
#   MAX_POLL_SECONDS (default 9000).
#
# The launcher reads serving_selection.json, submits one job with the
# selected TP (CPU = 20 * TP, GPUs = TP, two-hour wall time, one node),
# waits for terminal accounting, writes the Slurm metadata from sacct,
# regenerates audit.json with the terminal state, and validates the final
# audit. The job ID is appended to
# $DEPLOY_ROOT/$DEPLOYMENT_ID/jobs.jsonl.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
DEPLOY_ROOT="${DEPLOY_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/deployments/qwen38_inference}"
MODEL_DIR="${MODEL_DIR:-/gpfs/projects/etur92/ozu647717/models/Qwen3.8-27B}"
WHEELHOUSE_DIR="${WHEELHOUSE_DIR:-/gpfs/projects/etur92/ozu647717/wheelhouses/qwen38_vllm_0.25.1_tf5.8.0_cu130_py310}"
VENV_DIR="${VENV_DIR:-/gpfs/projects/etur92/ozu647717/venvs/qwen38_inference}"
MODEL_REVISION="${MODEL_REVISION:-1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0}"
TRANSCRIPT_PATH="${TRANSCRIPT_PATH:-/gpfs/projects/etur92/ozu647717/AudioLLM/private_inputs/turkish_question_recovery/whisper_transcripts_qwen3_asr.jsonl}"
DRY_RUN="${DRY_RUN:-0}"
WAIT="${WAIT:-1}"
MAX_POLL_SECONDS="${MAX_POLL_SECONDS:-9000}"

DEPLOYMENT_ID="${DEPLOYMENT_ID:?Set DEPLOYMENT_ID}"
SOURCE_COMMIT="${SOURCE_COMMIT:?Set SOURCE_COMMIT}"
JOBS_FILE="$DEPLOY_ROOT/$DEPLOYMENT_ID/jobs.jsonl"
mkdir -p "$DEPLOY_ROOT/$DEPLOYMENT_ID"
cd "$PROJECT_ROOT"

SELECTION_FILE="$DEPLOY_ROOT/$DEPLOYMENT_ID/serving_selection.json"
if [ ! -f "$SELECTION_FILE" ]; then
  echo "FAILED: serving_selection.json missing at $SELECTION_FILE" >&2
  exit 1
fi

SELECTED_TP="$(python - "$SELECTION_FILE" <<'PY'
import json
import sys
selection = json.load(open(sys.argv[1], encoding="utf-8"))
selected = selection.get("selected_tp")
if selected not in (1, 2, 4):
    print("FAILED: invalid selected_tp in serving_selection.json", file=sys.stderr)
    raise SystemExit(1)
print(selected)
PY
)"
echo "selected_tp=$SELECTED_TP from $SELECTION_FILE"

RUN_ROOT="$PROJECT_ROOT/outputs/turkish_question_recovery/qwen38_${DEPLOYMENT_ID}"
CPUS=$((20 * SELECTED_TP))

echo "Qwen3.8 Turkish launcher"
echo "deployment_id=$DEPLOYMENT_ID source_commit=$SOURCE_COMMIT selected_tp=$SELECTED_TP"
echo "cpus=$CPUS gpus=$SELECTED_TP wall=02:00:00"
echo "run_root=$RUN_ROOT"

COMMAND=(
  sbatch --parsable --export=ALL \
    --job-name=q38-turkish \
    -A etur92 -q acc_ehpc -t 02:00:00 \
    --cpus-per-task="$CPUS" --gres="gpu:${SELECTED_TP}" \
    --export=ALL,DEPLOYMENT_ID="$DEPLOYMENT_ID",PROJECT_ROOT="$PROJECT_ROOT",DEPLOY_ROOT="$DEPLOY_ROOT",MODEL_DIR="$MODEL_DIR",WHEELHOUSE_DIR="$WHEELHOUSE_DIR",VENV_DIR="$VENV_DIR",MODEL_REVISION="$MODEL_REVISION",SOURCE_COMMIT="$SOURCE_COMMIT",SELECTED_TP="$SELECTED_TP",TRANSCRIPT_PATH="$TRANSCRIPT_PATH" \
    "$PROJECT_ROOT/scripts/run_qwen38_turkish_slurm.sh"
)
printf 'submission:'
printf ' %q' "${COMMAND[@]}"
printf '\n'

if [ "$DRY_RUN" = "1" ]; then
  echo "dry run: no job submitted"
  exit 0
fi

JOB_ID="$("${COMMAND[@]}")"
echo "job_id=$JOB_ID"

python - "$JOBS_FILE" "$DEPLOYMENT_ID" "$JOB_ID" "$SOURCE_COMMIT" "$SELECTED_TP" <<'PY'
import json
import sys
import time

path, deployment_id, job_id, source_commit, selected_tp = sys.argv[1:]
record = {
    "deployment_id": deployment_id,
    "stage": "q38-turkish",
    "slurm_job_id": job_id,
    "state": "SUBMITTED",
    "exit_code": "unknown",
    "attempt": "1",
    "selected_tp": int(selected_tp),
    "source_commit": source_commit,
    "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
with open(path, "a", encoding="utf-8") as handle:
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
PY

if [ "$WAIT" != "1" ]; then
  echo "submitted; WAIT=0 so no monitoring in this invocation"
  exit 0
fi

deadline=$(( $(date +%s) + MAX_POLL_SECONDS ))
while :; do
  queue_state="$(squeue -j "$JOB_ID" -h -o '%T' 2>/dev/null || true)"
  if [ -n "$queue_state" ]; then
    echo "job $JOB_ID queue_state=$queue_state"
  else
    break
  fi
  if [ "$(date +%s)" -ge "$deadline" ]; then
    echo "FAILED: poll deadline exceeded for $JOB_ID" >&2
    exit 2
  fi
  sleep 30
done

STATE="$(sacct -j "$JOB_ID" -n -X -o State 2>/dev/null | head -1 | tr -d ' ' || true)"
EXIT_CODE="$(sacct -j "$JOB_ID" -n -X -o ExitCode 2>/dev/null | head -1 | tr -d ' ' || true)"
ELAPSED="$(sacct -j "$JOB_ID" -n -X -o Elapsed 2>/dev/null | head -1 | tr -d ' ' || true)"
NODE="$(sacct -j "$JOB_ID" -n -X -o NodeList 2>/dev/null | head -1 | tr -d ' ' || true)"
START_TIME="$(sacct -j "$JOB_ID" -n -X -o Start 2>/dev/null | head -1 | tr -d ' ' || true)"
END_TIME="$(sacct -j "$JOB_ID" -n -X -o End 2>/dev/null | head -1 | tr -d ' ' || true)"

python - "$JOBS_FILE" "$DEPLOYMENT_ID" "$JOB_ID" "$STATE" "$EXIT_CODE" <<'PY'
import json
import sys
import time

path, deployment_id, job_id, state, exit_code = sys.argv[1:]
record = {
    "deployment_id": deployment_id,
    "stage": "q38-turkish",
    "slurm_job_id": job_id,
    "state": state,
    "exit_code": exit_code,
    "attempt": "1",
    "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
with open(path, "a", encoding="utf-8") as handle:
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
PY

echo "job $JOB_ID terminal: state=$STATE exit=$EXIT_CODE elapsed=$ELAPSED node=$NODE"
if [ "$STATE" != "COMPLETED" ] || [ "$EXIT_CODE" != "0:0" ]; then
  case "$STATE" in
    NODE_FAIL|PREEMPTED|BOOT_FAIL)
      echo "transient infrastructure failure; retry requires a manual attempt2 submission with the same deployment ID" >&2
      ;;
    *)
      echo "FAILED: job $JOB_ID did not complete cleanly" >&2
      ;;
  esac
  exit 1
fi

# Regenerate audit.json with the terminal Slurm accounting record.
python - "$RUN_ROOT/slurm_run_metadata.json" "$JOB_ID" "$STATE" "$EXIT_CODE" "$NODE" "$START_TIME" "$END_TIME" <<'PY'
import json
import sys

path, job_id, state, exit_code, node, start_time, end_time = sys.argv[1:]
metadata = {
    "job_id": job_id,
    "state": state,
    "exit_code": exit_code,
    "node": node,
    "start_time": start_time,
    "end_time": end_time,
}
with open(path, "w", encoding="utf-8") as handle:
    json.dump(metadata, handle, indent=2)
    handle.write("\n")
PY
chmod 600 "$RUN_ROOT/slurm_run_metadata.json"

python scripts/qwen38_audit.py turkish \
  --run-dir "$RUN_ROOT" \
  --transcript "$TRANSCRIPT_PATH" \
  --deploy-dir "$DEPLOY_ROOT" \
  --deployment-id "$DEPLOYMENT_ID" \
  --model-dir "$MODEL_DIR" \
  --wheelhouse-dir "$WHEELHOUSE_DIR" \
  --source-commit "$SOURCE_COMMIT" \
  --slurm-metadata "$RUN_ROOT/slurm_run_metadata.json" \
  --out "$RUN_ROOT/audit.json"
chmod 600 "$RUN_ROOT/audit.json"

echo "OK: Turkish question recovery finished for $DEPLOYMENT_ID"
echo "Run root: $RUN_ROOT"
echo "Compact artifacts: turkish_inferred_questions.{csv,json,md} + audit.json"
