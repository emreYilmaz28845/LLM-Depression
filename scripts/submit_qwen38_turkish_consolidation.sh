#!/usr/bin/env bash
# Submit exactly one TP=2 consolidation-only recovery job and wait for sacct.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
DEPLOY_ROOT="${DEPLOY_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/deployments/qwen38_inference}"
DEPLOYMENT_ID="${DEPLOYMENT_ID:?Set DEPLOYMENT_ID}"
DERIVED_RUN_ID="${DERIVED_RUN_ID:?Set DERIVED_RUN_ID}"
SOURCE_COMMIT="${SOURCE_COMMIT:?Set SOURCE_COMMIT}"
LEDGER="$DEPLOY_ROOT/$DEPLOYMENT_ID/turkish_consolidation_jobs.jsonl"
RUN_ROOT="$PROJECT_ROOT/outputs/turkish_question_recovery_derived/$DERIVED_RUN_ID"
cd "$PROJECT_ROOT"

if [ -e "$RUN_ROOT" ]; then
  echo "FAILED: refusing existing derived run root $RUN_ROOT" >&2
  exit 1
fi
if [ -f "$LEDGER" ] && grep -Fq "\"derived_run_id\": \"$DERIVED_RUN_ID\"" "$LEDGER"; then
  echo "FAILED: derived run ID already exists in ledger" >&2
  exit 1
fi

JOB_ID="$(sbatch --parsable --job-name=q38tc --cpus-per-task=40 --gpus=2 --export=ALL scripts/run_qwen38_turkish_consolidation_slurm.sh)"
python - "$LEDGER" "$DERIVED_RUN_ID" "$JOB_ID" "$SOURCE_COMMIT" <<'PY'
import json, sys, time
path, run_id, job_id, source = sys.argv[1:]
with open(path, "a", encoding="utf-8") as handle:
    handle.write(json.dumps({"derived_run_id": run_id, "slurm_job_id": job_id, "state": "SUBMITTED", "source_commit": source, "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, sort_keys=True) + "\n")
PY
echo "submitted $JOB_ID"

while true; do
  LINE="$(sacct -X -j "$JOB_ID" -n -P -o JobIDRaw,State,ExitCode,Elapsed,NodeList,Start,End | awk -F'|' -v id="$JOB_ID" '$1==id {print; exit}')"
  STATE="$(printf '%s' "$LINE" | cut -d'|' -f2 | cut -d'+' -f1)"
  case "$STATE" in
    COMPLETED|FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED) break ;;
  esac
  sleep 20
done
python - "$LEDGER" "$DERIVED_RUN_ID" "$JOB_ID" "$SOURCE_COMMIT" "$LINE" "$RUN_ROOT/slurm_run_metadata.json" <<'PY'
import json, pathlib, sys, time
ledger, run_id, job_id, source, line, metadata_path = sys.argv[1:]
fields = line.split("|")
record = dict(zip(("job_id", "state", "exit_code", "elapsed", "node", "start_time", "end_time"), fields))
record.update({"derived_run_id": run_id, "source_commit": source})
path = pathlib.Path(metadata_path)
if path.parent.is_dir():
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
with open(ledger, "a", encoding="utf-8") as handle:
    handle.write(json.dumps({**record, "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, sort_keys=True) + "\n")
print(json.dumps(record, sort_keys=True))
if record.get("state") != "COMPLETED" or record.get("exit_code") != "0:0":
    raise SystemExit(1)
PY
