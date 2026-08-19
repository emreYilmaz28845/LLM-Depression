#!/usr/bin/env bash
# Submit and monitor the Qwen3.8 validation sequence (runbook sections 14-17).
#
# Stage order: environment build (CPU) -> TP=2 acceptance smoke -> TP=1 and
# TP=4 bounded comparisons (concurrent) -> serving-configuration selection.
#
# Required env:
#   DEPLOYMENT_ID
# Optional env:
#   PROJECT_ROOT, DEPLOY_ROOT, MODEL_DIR, WHEELHOUSE_DIR, VENV_DIR,
#   MODEL_REVISION, SOURCE_COMMIT, DRY_RUN (default 0), WAIT (default 1),
#   SKIP_ENV_BUILD (default 0), SKIP_TP1 (default 0), SKIP_TP4 (default 0),
#   SKIP_SELECTION (default 0), MAX_POLL_SECONDS (default 5400 per stage).
#
# Every submitted job ID is appended to
# $DEPLOY_ROOT/$DEPLOYMENT_ID/jobs.jsonl. A failed job is retried once with
# attempt=2 only for transient infrastructure states (NODE_FAIL, PREEMPTED,
# BOOT_FAIL), per runbook section 22.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
DEPLOY_ROOT="${DEPLOY_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/deployments/qwen38_inference}"
MODEL_DIR="${MODEL_DIR:-/gpfs/projects/etur92/ozu647717/models/Qwen3.8-27B}"
WHEELHOUSE_DIR="${WHEELHOUSE_DIR:-/gpfs/projects/etur92/ozu647717/wheelhouses/qwen38_vllm_0.25.1_tf5.8.0_cu130_py310}"
VENV_DIR="${VENV_DIR:-/gpfs/projects/etur92/ozu647717/venvs/qwen38_inference}"
MODEL_REVISION="${MODEL_REVISION:-1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0}"
SOURCE_COMMIT="${SOURCE_COMMIT:-unknown}"
DRY_RUN="${DRY_RUN:-0}"
WAIT="${WAIT:-1}"
SKIP_ENV_BUILD="${SKIP_ENV_BUILD:-0}"
SKIP_TP1="${SKIP_TP1:-0}"
SKIP_TP4="${SKIP_TP4:-0}"
SKIP_SELECTION="${SKIP_SELECTION:-0}"
MAX_POLL_SECONDS="${MAX_POLL_SECONDS:-5400}"

DEPLOYMENT_ID="${DEPLOYMENT_ID:?Set DEPLOYMENT_ID}"
JOBS_FILE="$DEPLOY_ROOT/$DEPLOYMENT_ID/jobs.jsonl"
mkdir -p "$DEPLOY_ROOT/$DEPLOYMENT_ID"
cd "$PROJECT_ROOT"

record_job() {
  local deployment_id="$1" stage="$2" job_id="$3" state="$4" exit_code="$5" attempt="$6"
  python - "$JOBS_FILE" "$deployment_id" "$stage" "$job_id" "$state" "$exit_code" "$attempt" <<'PY'
import json
import sys
import time

path, deployment_id, stage, job_id, state, exit_code, attempt = sys.argv[1:]
record = {
    "deployment_id": deployment_id,
    "stage": stage,
    "slurm_job_id": job_id,
    "state": state,
    "exit_code": exit_code,
    "attempt": attempt,
    "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
with open(path, "a", encoding="utf-8") as handle:
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
PY
}

poll_job() {
  local job_id="$1" stage="$2" attempt="$3"
  local deadline=$(( $(date +%s) + MAX_POLL_SECONDS ))
  while :; do
    local queue_state
    queue_state="$(squeue -j "$job_id" -h -o '%T' 2>/dev/null || true)"
    if [ -z "$queue_state" ]; then
      local sacct_state exit_code elapsed
      sacct_state="$(sacct -j "$job_id" -n -X -o State 2>/dev/null | head -1 | tr -d ' ' || true)"
      exit_code="$(sacct -j "$job_id" -n -X -o ExitCode 2>/dev/null | head -1 | tr -d ' ' || true)"
      elapsed="$(sacct -j "$job_id" -n -X -o Elapsed 2>/dev/null | head -1 | tr -d ' ' || true)"
      if [ -n "$sacct_state" ] && [[ "$sacct_state" == *"COMPLETED"* || "$sacct_state" == *"FAILED"* || "$sacct_state" == *"CANCELLED"* || "$sacct_state" == *"TIMEOUT"* || "$sacct_state" == *"OUT_OF_MEMORY"* || "$sacct_state" == *"NODE_FAIL"* || "$sacct_state" == *"PREEMPTED"* || "$sacct_state" == *"BOOT_FAIL"* || "$sacct_state" == *"OUT_OF_"* ]]; then
        record_job "$DEPLOYMENT_ID" "$stage" "$job_id" "$sacct_state" "${exit_code:-unknown}" "$attempt"
        echo "job $job_id ($stage) terminal: state=$sacct_state exit=$exit_code elapsed=$elapsed"
        if [ "$sacct_state" = "COMPLETED" ]; then
          return 0
        fi
        echo "$sacct_state"
        return 1
      fi
      echo "job $job_id ($stage): left queue, accounting pending (state=$sacct_state)" >&2
    else
      echo "job $job_id ($stage): queue_state=$queue_state"
    fi
    if [ "$(date +%s)" -ge "$deadline" ]; then
      echo "job $job_id ($stage): poll deadline exceeded" >&2
      return 2
    fi
    sleep 30
  done
}

submit_and_wait() {
  local stage="$1" attempt="$2"
  shift 2
  local job_id
  local command=(sbatch --parsable --export=ALL \
    --job-name="$stage" \
    "$@")
  printf 'submission (%s, attempt %s):' "$stage" "$attempt"
  printf ' %q' "${command[@]}"
  printf '\n'
  if [ "$DRY_RUN" = "1" ]; then
    return 0
  fi
  job_id="$("${command[@]}")"
  echo "job_id=$job_id stage=$stage attempt=$attempt"
  if [ "$WAIT" = "1" ]; then
    if poll_job "$job_id" "$stage" "$attempt"; then
      echo "RESULT $stage attempt=$attempt job=$job_id PASSED"
      return 0
    else
      echo "RESULT $stage attempt=$attempt job=$job_id FAILED" >&2
      return 1
    fi
  fi
  record_job "$DEPLOYMENT_ID" "$stage" "$job_id" "SUBMITTED" "unknown" "$attempt"
  return 0
}

run_with_transient_retry() {
  local stage="$1"
  shift
  if submit_and_wait "$stage" "1" "$@"; then
    return 0
  fi
  # Attempt 2 is allowed only for transient infrastructure failures; the
  # poll helper reports the sacct state on stderr so the operator can verify.
  local state
  state="$(sacct -j "$(python - "$JOBS_FILE" "$stage" <<'PY'
import json
import sys
path, stage = sys.argv[1:]
last = None
for line in open(path, encoding="utf-8"):
    record = json.loads(line)
    if record.get("stage") == stage:
        last = record
print(last["slurm_job_id"] if last else "")
PY
)" -n -X -o State 2>/dev/null | head -1 | tr -d ' ' || true)"
  case "$state" in
    NODE_FAIL|PREEMPTED|BOOT_FAIL)
      echo "transient infrastructure failure ($state); retrying $stage once as attempt2" >&2
      if submit_and_wait "$stage" "2" "$@"; then
        return 0
      fi
      echo "attempt2 also failed; stopping" >&2
      return 1
      ;;
    *)
      echo "non-transient failure state ($state); no retry" >&2
      return 1
      ;;
  esac
}

echo "Qwen3.8 validation launcher"
echo "deployment_id=$DEPLOYMENT_ID source_commit=$SOURCE_COMMIT dry_run=$DRY_RUN wait=$WAIT"
echo "project_root=$PROJECT_ROOT"
echo "deploy_root=$DEPLOY_ROOT"
echo "model_dir=$MODEL_DIR"
echo "wheelhouse_dir=$WHEELHOUSE_DIR"
echo "venv_dir=$VENV_DIR"
echo "jobs_file=$JOBS_FILE"

# --- Stage 1: environment build ----------------------------------------------
ENV_DIR="$DEPLOY_ROOT/$DEPLOYMENT_ID/environment"
if [ "$SKIP_ENV_BUILD" != "1" ] && [ ! -f "$ENV_DIR/environment_acceptance.json" ]; then
  echo "[stage:env-build]"
  run_with_transient_retry "q38-env-build" \
    -A etur92 -q acc_ehpc -t 00:30:00 \
    --cpus-per-task=4 --gres=none \
    --export=ALL,DEPLOYMENT_ID="$DEPLOYMENT_ID",PROJECT_ROOT="$PROJECT_ROOT",DEPLOY_ROOT="$DEPLOY_ROOT",MODEL_DIR="$MODEL_DIR",WHEELHOUSE_DIR="$WHEELHOUSE_DIR",VENV_DIR="$VENV_DIR",MODEL_REVISION="$MODEL_REVISION",SOURCE_COMMIT="$SOURCE_COMMIT" \
    "$PROJECT_ROOT/scripts/run_qwen38_env_build_slurm.sh"
  if [ "$DRY_RUN" = "0" ] && [ ! -f "$ENV_DIR/environment_acceptance.json" ]; then
    echo "FAILED: environment build did not produce environment_acceptance.json" >&2
    exit 1
  fi
else
  echo "[stage:env-build] skipped (SKIP_ENV_BUILD=$SKIP_ENV_BUILD or acceptance exists)"
fi

# --- Stage 2: mandatory TP=2 acceptance smoke --------------------------------
TP2_ACCEPTANCE="$DEPLOY_ROOT/$DEPLOYMENT_ID/validation/tp2/attempt1/acceptance.json"
if [ -f "$TP2_ACCEPTANCE" ] && python - "$TP2_ACCEPTANCE" <<'PY'
import json
import sys
if json.load(open(sys.argv[1], encoding="utf-8")).get("passed"):
    raise SystemExit(0)
raise SystemExit(1)
PY
then
  echo "[stage:q38-tp2-smoke] skipped (acceptance already passed)"
else
  echo "[stage:q38-tp2-smoke]"
  run_with_transient_retry "q38-tp2-smoke" \
    -A etur92 -q acc_ehpc -t 00:30:00 \
    --cpus-per-task=40 --gres=gpu:2 --exclusive \
    --export=ALL,DEPLOYMENT_ID="$DEPLOYMENT_ID",PROJECT_ROOT="$PROJECT_ROOT",DEPLOY_ROOT="$DEPLOY_ROOT",MODEL_DIR="$MODEL_DIR",WHEELHOUSE_DIR="$WHEELHOUSE_DIR",VENV_DIR="$VENV_DIR",MODEL_REVISION="$MODEL_REVISION",SOURCE_COMMIT="$SOURCE_COMMIT",TP_SIZE=2,ATTEMPT=1,RUN_LABEL=q38-tp2-smoke \
    "$PROJECT_ROOT/scripts/run_qwen38_validation_slurm.sh"
  if [ "$DRY_RUN" = "0" ]; then
    if [ ! -f "$TP2_ACCEPTANCE" ] || ! python - "$TP2_ACCEPTANCE" <<'PY'
import json
import sys
if not json.load(open(sys.argv[1], encoding="utf-8")).get("passed"):
    raise SystemExit(1)
PY
    then
      echo "FAILED: TP=2 acceptance gate did not pass; TP=1/TP=4 and selection are blocked" >&2
      exit 1
    fi
  fi
fi

# --- Stage 3: bounded TP=1 and TP=4 comparisons ------------------------------
# Per runbook section 16, a TP=1 capacity failure (OOM / insufficient memory /
# architecture refusal) is CAPACITY_INELIGIBLE and a TP=4 failure is recorded
# and excluded from selection; neither hard-stops the pipeline. Only TP=2 gates
# the deployment. Failures are recorded in jobs.jsonl by the poll helper and a
# capacity/ineligible marker is written when the worker log shows an OOM.

record_capacity() {
  local tp="$1" label="$2"
  local marker="$DEPLOY_ROOT/$DEPLOYMENT_ID/validation/tp${tp}/attempt1/capacity_ineligible.json"
  mkdir -p "$(dirname "$marker")"
  python - "$marker" "$tp" "$label" <<'PY'
import json
import sys
import time
marker, tp, reason = sys.argv[1:]
payload = {
    "tp": int(tp),
    "capacity_ineligible": True,
    "reason": reason,
    "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
with open(marker, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2)
    handle.write("\n")
PY
  echo "recorded $label as CAPACITY_INELIGIBLE ($marker)"
}

if [ "$SKIP_TP1" != "1" ]; then
  TP1_DONE="$DEPLOY_ROOT/$DEPLOYMENT_ID/validation/tp1/attempt1/acceptance.json"
  TP1_CAP="$DEPLOY_ROOT/$DEPLOYMENT_ID/validation/tp1/attempt1/capacity_ineligible.json"
  if [ -f "$TP1_DONE" ] || [ -f "$TP1_CAP" ]; then
    echo "[stage:q38-tp1-compare] skipped (acceptance or capacity record exists)"
  else
    echo "[stage:q38-tp1-compare]"
    if run_with_transient_retry "q38-tp1-compare" \
      -A etur92 -q acc_ehpc -t 00:30:00 \
      --cpus-per-task=20 --gres=gpu:1 --exclusive \
      --export=ALL,DEPLOYMENT_ID="$DEPLOYMENT_ID",PROJECT_ROOT="$PROJECT_ROOT",DEPLOY_ROOT="$DEPLOY_ROOT",MODEL_DIR="$MODEL_DIR",WHEELHOUSE_DIR="$WHEELHOUSE_DIR",VENV_DIR="$VENV_DIR",MODEL_REVISION="$MODEL_REVISION",SOURCE_COMMIT="$SOURCE_COMMIT",TP_SIZE=1,ATTEMPT=1,RUN_LABEL=q38-tp1-compare \
      "$PROJECT_ROOT/scripts/run_qwen38_validation_slurm.sh"; then
      :
    else
      echo "TP=1 comparison did not pass; continuing to TP=4 and selection"
      if [ "$DRY_RUN" = "0" ]; then
        for LOG in "$DEPLOY_ROOT/$DEPLOYMENT_ID"/logs/validation_tp1_attempt1_*.log; do
          if [ -f "$LOG" ] && grep -qE "CUDA out of memory|OutOfMemoryError|AcceleratorError" "$LOG"; then
            record_capacity 1 "TP=1"
            break
          fi
        done
      fi
    fi
  fi
else
  echo "[stage:q38-tp1-compare] skipped (SKIP_TP1=$SKIP_TP1)"
fi

if [ "$SKIP_TP4" != "1" ]; then
  TP4_DONE="$DEPLOY_ROOT/$DEPLOYMENT_ID/validation/tp4/attempt1/acceptance.json"
  if [ -f "$TP4_DONE" ]; then
    echo "[stage:q38-tp4-compare] skipped (acceptance exists)"
  else
    echo "[stage:q38-tp4-compare]"
    if run_with_transient_retry "q38-tp4-compare" \
      -A etur92 -q acc_ehpc -t 00:30:00 \
      --cpus-per-task=80 --gres=gpu:4 --exclusive \
      --export=ALL,DEPLOYMENT_ID="$DEPLOYMENT_ID",PROJECT_ROOT="$PROJECT_ROOT",DEPLOY_ROOT="$DEPLOY_ROOT",MODEL_DIR="$MODEL_DIR",WHEELHOUSE_DIR="$WHEELHOUSE_DIR",VENV_DIR="$VENV_DIR",MODEL_REVISION="$MODEL_REVISION",SOURCE_COMMIT="$SOURCE_COMMIT",TP_SIZE=4,ATTEMPT=1,RUN_LABEL=q38-tp4-compare \
      "$PROJECT_ROOT/scripts/run_qwen38_validation_slurm.sh"; then
      :
    else
      echo "TP=4 comparison did not pass; recording and continuing to selection"
    fi
  fi
else
  echo "[stage:q38-tp4-compare] skipped (SKIP_TP4=$SKIP_TP4)"
fi

# --- Stage 4: serving-configuration selection ---------------------------------
if [ "$SKIP_SELECTION" != "1" ]; then
  echo "[stage:select]"
  if [ "$DRY_RUN" = "1" ]; then
    echo "would run: python scripts/qwen38_validation_client.py select --deploy-dir $DEPLOY_ROOT --deployment-id $DEPLOYMENT_ID --source-commit $SOURCE_COMMIT"
  else
    python scripts/qwen38_validation_client.py select \
      --deploy-dir "$DEPLOY_ROOT" \
      --deployment-id "$DEPLOYMENT_ID" \
      --source-commit "$SOURCE_COMMIT"
    echo "serving_selection.json: $DEPLOY_ROOT/$DEPLOYMENT_ID/serving_selection.json"
  fi
else
  echo "[stage:select] skipped (SKIP_SELECTION=$SKIP_SELECTION)"
fi

echo "OK: validation sequence finished for $DEPLOYMENT_ID"
