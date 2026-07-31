#!/bin/bash
#SBATCH -J and-hid-audit
#SBATCH -A etur92
#SBATCH -q acc_ehpc
#SBATCH -t 01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#SBATCH --chdir=/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression

set -euo pipefail
module purge
module load bsc/1.0
module load miniforge/24.3.0-0

PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
ENV_ACTIVATE="${ENV_ACTIVATE:-/gpfs/projects/etur92/ozu647717/venvs/qwen_mn5_rebuilt/bin/activate}"
MODE="${MODE:?MODE is required}"
RUN_ID="${RUN_ID:?RUN_ID is required}"
SOURCE_COMMIT="${SOURCE_COMMIT:?SOURCE_COMMIT is required}"
AUDIT_OUT="${AUDIT_OUT:?AUDIT_OUT is required}"
JOB_REGISTRY="${JOB_REGISTRY:-}"
JOB_ACCOUNTING_OUT="${JOB_ACCOUNTING_OUT:-$(dirname "${AUDIT_OUT}")/job_accounting.json}"
CACHE_ROOT="${CACHE_ROOT:-}"
CLASSIFIER_ROOT="${CLASSIFIER_ROOT:-}"
SMOKE_EXTRACTION_DIR="${SMOKE_EXTRACTION_DIR:-}"
SMOKE_FIXED_ROOT="${SMOKE_FIXED_ROOT:-}"
SMOKE_OPTUNA_DIR="${SMOKE_OPTUNA_DIR:-}"

# shellcheck disable=SC1090
source "$ENV_ACTIVATE"
cd "$PROJECT_ROOT"
export PROJECT_ROOT
export PYTHONPATH="$PROJECT_ROOT/.deps/qwen_hidden:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
LOG_ROOT="$PROJECT_ROOT/logs/slurm_androids_hidden/$RUN_ID"
mkdir -p "$LOG_ROOT"
mkdir -p "$(dirname "$AUDIT_OUT")"
exec > >(tee -a "$LOG_ROOT/audit-${SLURM_JOB_ID}.out")
exec 2> >(tee -a "$LOG_ROOT/audit-${SLURM_JOB_ID}.err" >&2)

ACCOUNTING_TEXT="$LOG_ROOT/sacct_${SLURM_JOB_ID}.txt"
: > "$ACCOUNTING_TEXT"
if [ "$MODE" = "production" ] && [ -n "$JOB_REGISTRY" ]; then
    ids="$(awk -F '\t' 'NR > 1 && $2 != "audit" {print $3}' "$JOB_REGISTRY" | paste -sd, -)"
    if [ -n "$ids" ]; then
        accounting="$(sacct -X -n -P -j "$ids" --format=JobIDRaw,State,ExitCode)"
        printf '%s\n' "$accounting" > "$ACCOUNTING_TEXT"
        while IFS='|' read -r job_id state exit_code; do
            [ -z "$job_id" ] && continue
            if [ "$state" != "COMPLETED" ] || [ "$exit_code" != "0:0" ]; then
                echo "Upstream Androids hidden job did not complete cleanly: $job_id $state $exit_code" >&2
                exit 1
            fi
        done <<< "$accounting"
    fi
fi

python - "$ACCOUNTING_TEXT" "$JOB_ACCOUNTING_OUT" "$PROJECT_ROOT" <<'PY'
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

accounting_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
project_root = sys.argv[3]
jobs = []
for line in accounting_path.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    job_id, state, exit_code, elapsed = (line.split("|", 3) + [None] * 4)[:4]
    jobs.append({"job_id": job_id, "state": state, "exit_code": exit_code, "elapsed": elapsed})
storage = {}
try:
    lines = subprocess.check_output(["df", "-P", "-B1", project_root], text=True).splitlines()
    if len(lines) >= 2:
        fields = lines[-1].split()
        storage = {
            "filesystem": fields[0],
            "size_bytes": int(fields[1]),
            "used_bytes": int(fields[2]),
            "available_bytes": int(fields[3]),
            "use_percent": fields[4],
            "mountpoint": fields[5],
        }
except Exception as exc:
    storage = {"error": str(exc)}
payload = {
    "schema_version": "androids_hidden_job_accounting.v1",
    "captured_at_utc": datetime.now(timezone.utc).isoformat(),
    "jobs": jobs,
    "storage": storage,
}
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

CMD=(python scripts/audit_androids_hidden.py
    --mode "$MODE"
    --run-id "$RUN_ID"
    --out "$AUDIT_OUT"
    --source-commit "$SOURCE_COMMIT")
if [ -n "$JOB_REGISTRY" ]; then CMD+=(--job-registry "$JOB_REGISTRY"); fi
if [ -n "$JOB_ACCOUNTING_OUT" ]; then CMD+=(--job-accounting "$JOB_ACCOUNTING_OUT"); fi
if [ "$MODE" = "production" ]; then
    CMD+=(--cache-root "$CACHE_ROOT" --classifier-root "$CLASSIFIER_ROOT")
else
    CMD+=(--smoke-extraction-dir "$SMOKE_EXTRACTION_DIR"
          --smoke-fixed-root "$SMOKE_FIXED_ROOT"
          --smoke-optuna-dir "$SMOKE_OPTUNA_DIR")
fi
"${CMD[@]}"

if [ "$MODE" = "production" ]; then
    python scripts/summarize_androids_hidden.py \
        --acceptance "$AUDIT_OUT" \
        --output-dir "$(dirname "$AUDIT_OUT")"
fi
