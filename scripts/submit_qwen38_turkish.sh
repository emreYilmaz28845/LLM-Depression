#!/usr/bin/env bash
# Submit and monitor the Qwen3.8 Turkish question-recovery job
# (runbook sections 17, 20, 21, 22).
#
# Required env:
#   DEPLOYMENT_ID, TURKISH_RUN_ID, SOURCE_COMMIT, SELECTION_FILE
# Optional env:
#   PROJECT_ROOT, DEPLOY_ROOT, MODEL_DIR, WHEELHOUSE_DIR, VENV_DIR,
#   MODEL_REVISION, TRANSCRIPT_PATH, DRY_RUN (default 0), WAIT (default 1),
#   MAX_POLL_SECONDS (default 9000).
#
# The launcher reads serving_selection_v2.json, submits one job with the
# selected TP (CPU = 20 * TP, GPUs = TP, two-hour wall time, one node),
# waits for terminal accounting, writes the Slurm metadata from sacct,
# regenerates audit.json with the terminal state, and validates the final
# audit. New attempt events are appended only to
# $DEPLOY_ROOT/$DEPLOYMENT_ID/turkish_jobs.jsonl.

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
TURKISH_RUN_ID="${TURKISH_RUN_ID:?Set TURKISH_RUN_ID}"
SOURCE_COMMIT="${SOURCE_COMMIT:?Set SOURCE_COMMIT}"
SELECTION_FILE="${SELECTION_FILE:?Set SELECTION_FILE}"
ANALYSIS_ATTEMPT="${ANALYSIS_ATTEMPT:-1}"
SUPERSEDES_JOB_IDS="${SUPERSEDES_JOB_IDS:-44797563,44797605,44799622}"
TURKISH_LEDGER="$DEPLOY_ROOT/$DEPLOYMENT_ID/turkish_jobs.jsonl"
VALIDATION_LEDGER="$DEPLOY_ROOT/$DEPLOYMENT_ID/jobs.jsonl"
RECONCILIATION_FILE="$DEPLOY_ROOT/$DEPLOYMENT_ID/turkish_job_reconciliation_${TURKISH_RUN_ID}.json"
cd "$PROJECT_ROOT"

if [ ! -f "$SELECTION_FILE" ]; then
  echo "FAILED: serving_selection_v2.json missing at $SELECTION_FILE" >&2
  exit 1
fi

if [[ ! "$TURKISH_RUN_ID" =~ ^q38tr_[0-9a-f]{12}_attempt[0-9]+$ ]]; then
  echo "FAILED: invalid TURKISH_RUN_ID=$TURKISH_RUN_ID" >&2
  exit 1
fi
ORIGIN_MAIN_SHA="${LOCAL_ORIGIN_MAIN_SHA:-$(git rev-parse origin/main 2>/dev/null || true)}"
REMOTE_PROVENANCE_SHA="$(tr -d '[:space:]' < "$PROJECT_ROOT/.provenance/git_commit.txt" 2>/dev/null || true)"
if [ "$SOURCE_COMMIT" != "$ORIGIN_MAIN_SHA" ] || [ "$SOURCE_COMMIT" != "$REMOTE_PROVENANCE_SHA" ]; then
  echo "FAILED: source identity mismatch (SOURCE_COMMIT=$SOURCE_COMMIT origin/main=$ORIGIN_MAIN_SHA provenance=$REMOTE_PROVENANCE_SHA)" >&2
  exit 1
fi

SELECTED_TP="$(python - "$SELECTION_FILE" "$SOURCE_COMMIT" <<'PY'
import json
import sys
selection = json.load(open(sys.argv[1], encoding="utf-8"))
if selection.get("selection_version") != 2:
    raise SystemExit("selection file is not version 2")
if selection.get("source_commit") != sys.argv[2]:
    raise SystemExit("selection v2 source commit does not match SOURCE_COMMIT")
selected = selection.get("selected_tp")
if selected not in (1, 2, 4):
    raise SystemExit("selection v2 has invalid selected_tp")
print(selected)
PY
)"
echo "selected_tp=$SELECTED_TP from $SELECTION_FILE"

RUN_ROOT="$PROJECT_ROOT/outputs/turkish_question_recovery/$TURKISH_RUN_ID"
if [ -e "$RUN_ROOT" ]; then
  if find "$RUN_ROOT" -type f \( -name 'prepared_sequences.jsonl' -o -path '*/subject_inferences/*' -o -path '*/consolidation_batches/*' -o -name 'turkish_inferred_questions.*' -o -name 'audit*.json' -o -name 'slurm_run_metadata.json' \) -print -quit | grep -q .; then
    echo "FAILED: new Turkish run root already contains analysis evidence: $RUN_ROOT" >&2
  else
    echo "FAILED: refusing to reuse existing Turkish run root: $RUN_ROOT" >&2
  fi
  exit 1
fi
CPUS=$((20 * SELECTED_TP))
if [ -e "$RECONCILIATION_FILE" ]; then
  echo "FAILED: reconciliation already exists for TURKISH_RUN_ID=$TURKISH_RUN_ID" >&2
  exit 1
fi
if [ -f "$TURKISH_LEDGER" ] && python - "$TURKISH_LEDGER" "$TURKISH_RUN_ID" <<'PY'
import json
import sys

path, run_id = sys.argv[1:]
for line in open(path, encoding="utf-8"):
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        continue
    if record.get("turkish_run_id") == run_id:
        raise SystemExit(0)
raise SystemExit(1)
PY
then
  echo "FAILED: Turkish ledger already contains TURKISH_RUN_ID=$TURKISH_RUN_ID" >&2
  exit 1
fi
PROMPT_CONTRACT_SHA256="$(python - <<'PY'
from src.qwen38.turkish_questions import prompt_contract_sha256
print(prompt_contract_sha256())
PY
)"

append_turkish_event() {
  local stage="$1" state="$2" exit_code="$3" job_id="$4" manifest_hash="${5:-}"
  python - "$TURKISH_LEDGER" "$TURKISH_RUN_ID" "$ANALYSIS_ATTEMPT" "$DEPLOYMENT_ID" "$stage" "$job_id" "$state" "$exit_code" "$SOURCE_COMMIT" "$PROMPT_CONTRACT_SHA256" "$manifest_hash" "$SELECTED_TP" "$SUPERSEDES_JOB_IDS" <<'PY'
import json
import sys
import time

(path, run_id, attempt, deployment_id, stage, job_id, state, exit_code,
 source_commit, prompt_contract, manifest_hash, selected_tp, supersedes) = sys.argv[1:]
record = {
    "turkish_run_id": run_id,
    "analysis_attempt": int(attempt),
    "deployment_id": deployment_id,
    "stage": stage,
    "slurm_job_id": job_id,
    "state": state,
    "exit_code": exit_code,
    "source_commit": source_commit,
    "prompt_contract_sha256": prompt_contract,
    "run_manifest_sha256": manifest_hash or None,
    "selected_tp": int(selected_tp),
    "supersedes_job_ids": [item for item in supersedes.split(",") if item],
    "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
with open(path, "a", encoding="utf-8") as handle:
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
PY
}

write_reconciliation() {
  local prior_ids="44786675,44787044,44787058,44787265,44787858,44788198,44788385,44788761,44789295,44789857,44790166,44790785,44791198,44791795,44792293,44792525,44792690,44793073,44793552,44793816,44794141,44794513,44794838,44795227,44795599,44796268,44796478,44797033,44797196,44797499,44797563,44797605,44798109,44798371,44798414,44798606,44798835,44798943,44799225,44799432,44799622"
  local current_job_id="${1:?current Turkish job ID is required}"
  python - "$RECONCILIATION_FILE" "$prior_ids" "$current_job_id" "$TURKISH_RUN_ID" "$ANALYSIS_ATTEMPT" "$DEPLOYMENT_ID" "$SOURCE_COMMIT" "$SUPERSEDES_JOB_IDS" "$VALIDATION_LEDGER" "$TURKISH_LEDGER" <<'PY'
import json
import os
import subprocess
import sys
import time

(
    path,
    ids_csv,
    current_job_id,
    turkish_run_id,
    analysis_attempt,
    deployment_id,
    source_commit,
    supersedes_job_ids,
    *ledger_paths,
) = sys.argv[1:]
ids = []
seen = set()

def add_job_id(value):
    value = str(value).strip()
    if value and value not in seen:
        seen.add(value)
        ids.append(value)

for item in ids_csv.split(","):
    add_job_id(item)
add_job_id(current_job_id)
for ledger_path in ledger_paths:
    if not os.path.isfile(ledger_path):
        continue
    with open(ledger_path, encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            job_id = record.get("slurm_job_id")
            if job_id not in (None, "", "unknown"):
                add_job_id(job_id)
if os.path.exists(path):
    raise SystemExit(f"refusing to overwrite reconciliation: {path}")
result = subprocess.run(
    ["sacct", "-X", "-j", ",".join(ids), "-n", "-P",
     "-o", "JobIDRaw,JobName,State,ExitCode,Elapsed,NodeList,Start,End"],
    capture_output=True, text=True, check=False,
)
records = []
for line in result.stdout.splitlines():
    fields = line.split("|")
    if len(fields) != 8:
        continue
    records.append(dict(zip(
        ("job_id", "job_name", "state", "exit_code", "elapsed", "node", "start", "end"),
        fields,
    )))
found = {record["job_id"] for record in records}
payload = {
    "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "turkish_run_id": turkish_run_id,
    "analysis_attempt": int(analysis_attempt),
    "deployment_id": deployment_id,
    "source_commit": source_commit,
    "supersedes_job_ids": [item for item in supersedes_job_ids.split(",") if item],
    "prior_job_ids": ids,
    "current_turkish_job_id": current_job_id,
    "ledger_sources": ledger_paths,
    "records": records,
    "missing_job_ids": [job_id for job_id in ids if job_id not in found],
    "sacct_returncode": result.returncode,
    "history": {
        "malformed_original_environment_event": {
            "job_id": "44786675",
            "state": "COMPLETEDCOMPLETED",
            "exit_code": "0:00:0",
        },
        "authoritative_correction": {
            "job_id": "44786675",
            "state": "COMPLETED",
            "exit_code": "0:0",
        },
    },
}
tmp = path + ".tmp"
with open(tmp, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2)
    handle.write("\n")
os.replace(tmp, path)
PY
}

echo "Qwen3.8 Turkish launcher"
echo "deployment_id=$DEPLOYMENT_ID source_commit=$SOURCE_COMMIT selected_tp=$SELECTED_TP"
echo "cpus=$CPUS gpus=$SELECTED_TP wall=02:00:00"
echo "run_root=$RUN_ROOT"

COMMAND=(
  sbatch --parsable --export=ALL \
    --job-name=q38-turkish \
    -A etur92 -q acc_ehpc -t 02:00:00 \
    --cpus-per-task="$CPUS" --gres="gpu:${SELECTED_TP}" \
    --export=ALL,DEPLOYMENT_ID="$DEPLOYMENT_ID",TURKISH_RUN_ID="$TURKISH_RUN_ID",ANALYSIS_ATTEMPT="$ANALYSIS_ATTEMPT",SUPERSEDES_JOB_IDS="$SUPERSEDES_JOB_IDS",LOCAL_ORIGIN_MAIN_SHA="$ORIGIN_MAIN_SHA",PROJECT_ROOT="$PROJECT_ROOT",DEPLOY_ROOT="$DEPLOY_ROOT",MODEL_DIR="$MODEL_DIR",WHEELHOUSE_DIR="$WHEELHOUSE_DIR",VENV_DIR="$VENV_DIR",MODEL_REVISION="$MODEL_REVISION",SOURCE_COMMIT="$SOURCE_COMMIT",SELECTED_TP="$SELECTED_TP",SELECTION_FILE="$SELECTION_FILE",TRANSCRIPT_PATH="$TRANSCRIPT_PATH" \
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

append_turkish_event "submit" "SUBMITTED" "unknown" "$JOB_ID"

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

append_turkish_event "terminal" "$STATE" "$EXIT_CODE" "$JOB_ID"
write_reconciliation "$JOB_ID"

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

# Attach the authoritative terminal Slurm accounting record without touching
# any restricted intermediate.
RUN_MANIFEST_SHA256="$(sha256sum "$RUN_ROOT/run_manifest.json" | awk '{print $1}')"
python - "$RUN_ROOT/slurm_run_metadata.json" "$JOB_ID" "$STATE" "$EXIT_CODE" "$NODE" "$START_TIME" "$END_TIME" "$TURKISH_RUN_ID" "$ANALYSIS_ATTEMPT" "$SOURCE_COMMIT" "$SELECTED_TP" "$SELECTION_FILE" "$RUN_MANIFEST_SHA256" <<'PY'
import json
import os
import sys

path, job_id, state, exit_code, node, start_time, end_time, run_id, attempt, source_commit, selected_tp, selection_file, manifest_hash = sys.argv[1:]
metadata = {
    "job_id": job_id,
    "state": state,
    "exit_code": exit_code,
    "node": node,
    "start_time": start_time,
    "end_time": end_time,
    "turkish_run_id": run_id,
    "analysis_attempt": int(attempt),
    "source_commit": source_commit,
    "selected_tp": int(selected_tp),
    "selection_file": selection_file,
    "run_manifest_sha256": manifest_hash,
}
tmp = path + ".tmp"
with open(tmp, "w", encoding="utf-8") as handle:
    json.dump(metadata, handle, indent=2)
    handle.write("\n")
os.replace(tmp, path)
PY
chmod 600 "$RUN_ROOT/slurm_run_metadata.json"

python scripts/qwen38_audit.py turkish \
  --run-dir "$RUN_ROOT" \
  --turkish-run-id "$TURKISH_RUN_ID" \
  --transcript "$TRANSCRIPT_PATH" \
  --deploy-dir "$DEPLOY_ROOT" \
  --deployment-id "$DEPLOYMENT_ID" \
  --model-dir "$MODEL_DIR" \
  --wheelhouse-dir "$WHEELHOUSE_DIR" \
  --source-commit "$SOURCE_COMMIT" \
  --selection-file "$SELECTION_FILE" \
  --slurm-metadata "$RUN_ROOT/slurm_run_metadata.json" \
  --out "$RUN_ROOT/audit.json"
chmod 600 "$RUN_ROOT/audit.json"
sha256sum "$RUN_ROOT/audit.json" > "$RUN_ROOT/audit.json.sha256"
chmod 600 "$RUN_ROOT/audit.json.sha256"
append_turkish_event "audit" "COMPLETED" "0:0" "$JOB_ID" "$RUN_MANIFEST_SHA256"

echo "OK: Turkish question recovery finished for $DEPLOYMENT_ID"
echo "Run root: $RUN_ROOT"
echo "Compact artifacts: turkish_inferred_questions.{csv,json,md} + audit.json"
