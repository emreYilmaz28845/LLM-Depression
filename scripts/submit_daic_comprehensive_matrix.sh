#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
RUN_ID="${RUN_ID:?Set a unique RUN_ID}"
STAGE="${STAGE:-smoke}"
DRY_RUN="${DRY_RUN:-1}"
MAX_CONCURRENT_TRAIN="${MAX_CONCURRENT_TRAIN:-4}"
MAX_CONCURRENT_EVAL="${MAX_CONCURRENT_EVAL:-16}"
MAX_CONCURRENT_HIDDEN="${MAX_CONCURRENT_HIDDEN:-16}"
MAX_CONCURRENT_CLASSICAL="${MAX_CONCURRENT_CLASSICAL:-8}"
RESUME="${RESUME:-0}"
MATRIX_PATH="${MATRIX_PATH:-$PROJECT_ROOT/outputs/daic_chunking_comprehensive/$RUN_ID/matrix_${STAGE}.json}"
export MAX_CONCURRENT_TRAIN MAX_CONCURRENT_EVAL MAX_CONCURRENT_HIDDEN MAX_CONCURRENT_CLASSICAL
case "$STAGE" in smoke|core|focused|final) ;; *) echo "Invalid STAGE=$STAGE" >&2; exit 2;; esac
case "$DRY_RUN" in 0|1) ;; *) echo "DRY_RUN must be 0 or 1" >&2; exit 2;; esac
case "$RESUME" in 0|1) ;; *) echo "RESUME must be 0 or 1" >&2; exit 2;; esac
case "$MAX_CONCURRENT_TRAIN" in ''|*[!0-9]*) echo "MAX_CONCURRENT_TRAIN must be a positive integer" >&2; exit 2;; esac
[ "$MAX_CONCURRENT_TRAIN" -ge 1 ] || { echo "MAX_CONCURRENT_TRAIN must be a positive integer" >&2; exit 2; }
for pair in MAX_CONCURRENT_EVAL:$MAX_CONCURRENT_EVAL MAX_CONCURRENT_HIDDEN:$MAX_CONCURRENT_HIDDEN MAX_CONCURRENT_CLASSICAL:$MAX_CONCURRENT_CLASSICAL; do
  name="${pair%%:*}"; value="${pair#*:}"
  case "$value" in ''|*[!0-9]*) echo "$name must be a positive integer" >&2; exit 2;; esac
  [ "$value" -ge 1 ] || { echo "$name must be a positive integer" >&2; exit 2; }
done
[ -f "$MATRIX_PATH" ] || { echo "Missing matrix: $MATRIX_PATH" >&2; exit 3; }

python - "$MATRIX_PATH" <<'PY'
import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1] if __file__ != "<stdin>" else Path.cwd()
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
from src.daic_comprehensive_audit import audit_matrix, validate_final_test_authorization

matrix_path = Path(sys.argv[1])
matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
failures = audit_matrix(matrix)
if str(matrix.get("stage")) == "final":
    marker_value = (matrix.get("test_authorization") or {}).get("path")
    marker = Path(marker_value) if marker_value else matrix_path.parent / "FINAL_TEST_AUTHORIZED.json"
    if not marker.is_absolute():
        marker = matrix_path.parent / marker
    ok, gate_failures, _ = validate_final_test_authorization(
        marker,
        selection_hash=matrix.get("selection_hash"),
        implementation_commit=matrix.get("implementation_commit"),
        spec_hash=matrix.get("spec_hash"),
    )
    if not ok:
        failures.extend(gate_failures)
if failures:
    raise SystemExit("Matrix rejected: " + ", ".join(failures))
PY

readarray -t COUNTS < <(python - "$MATRIX_PATH" <<'PY'
import json, sys
p=json.load(open(sys.argv[1]))
print(p["kind_counts"]["train"])
print(p["kind_counts"]["evaluation"])
print(p["kind_counts"]["hidden"])
print(p["kind_counts"]["classical"])
train=[t for t in p["tasks"] if t["kind"] == "train"]
regular=[str(i) for i,t in enumerate(train) if t["overrides"].get("training.objective", "token_ce") != "subject_mean_margin_mil"]
mil=[str(i) for i,t in enumerate(train) if t["overrides"].get("training.objective") == "subject_mean_margin_mil"]
print(",".join(regular))
print(",".join(mil))
PY
)
submit() {
  if [ "$DRY_RUN" = 1 ]; then
    printf 'DRY_RUN ' >&2; printf '%q ' "$@" >&2; printf '\n' >&2
    printf 'dry_%s\n' "$RANDOM"
  else
    "$@"
  fi
}
job_id() { printf '%s' "${1%%;*}"; }
LOG_ROOT="$PROJECT_ROOT/logs/daic_chunking_comprehensive/$RUN_ID/$STAGE/arrays"
train_jobs=()
if [ -n "${COUNTS[4]}" ]; then
train_raw="$(submit sbatch --parsable --array="${COUNTS[4]}%$MAX_CONCURRENT_TRAIN" --gres=gpu:4 --ntasks=4 --ntasks-per-node=4 --cpus-per-task=20 \
  --export="ALL,PROJECT_ROOT=$PROJECT_ROOT,MATRIX_PATH=$MATRIX_PATH,TASK_KIND=train,ARRAY_LOG_ROOT=$LOG_ROOT,RESUME=$RESUME" \
  "$PROJECT_ROOT/scripts/run_daic_comprehensive_array_slurm.sh")"
train_jobs+=("$(job_id "$train_raw")")
fi
if [ -n "${COUNTS[5]}" ]; then
mil_raw="$(submit sbatch --parsable --array="${COUNTS[5]}%1" --gres=gpu:1 --ntasks=1 --cpus-per-task=20 \
  --export="ALL,PROJECT_ROOT=$PROJECT_ROOT,MATRIX_PATH=$MATRIX_PATH,TASK_KIND=train,ARRAY_LOG_ROOT=$LOG_ROOT,RESUME=$RESUME" \
  "$PROJECT_ROOT/scripts/run_daic_comprehensive_array_slurm.sh")"
train_jobs+=("$(job_id "$mil_raw")")
fi
dependency="afterok:$(IFS=:; echo "${train_jobs[*]}")"
eval_raw="$(submit sbatch --parsable --dependency="$dependency" --array="0-$((${COUNTS[1]}-1))%$MAX_CONCURRENT_EVAL" --gres=gpu:1 --time=24:00:00 \
  --export="ALL,PROJECT_ROOT=$PROJECT_ROOT,MATRIX_PATH=$MATRIX_PATH,TASK_KIND=evaluation,ARRAY_LOG_ROOT=$LOG_ROOT,RESUME=$RESUME" \
  "$PROJECT_ROOT/scripts/run_daic_comprehensive_array_slurm.sh")"
evaluation_job="$(job_id "$eval_raw")"
hidden_raw="$(submit sbatch --parsable --dependency="afterok:$evaluation_job" --array="0-$((${COUNTS[2]}-1))%$MAX_CONCURRENT_HIDDEN" --gres=gpu:1 --time=24:00:00 \
  --export="ALL,PROJECT_ROOT=$PROJECT_ROOT,MATRIX_PATH=$MATRIX_PATH,TASK_KIND=hidden,ARRAY_LOG_ROOT=$LOG_ROOT,RESUME=$RESUME" \
  "$PROJECT_ROOT/scripts/run_daic_comprehensive_array_slurm.sh")"
hidden_job="$(job_id "$hidden_raw")"
classical_raw="$(submit sbatch --parsable --dependency="afterok:$hidden_job" --array="0-$((${COUNTS[3]}-1))%$MAX_CONCURRENT_CLASSICAL" --time=02:00:00 \
  --export="ALL,PROJECT_ROOT=$PROJECT_ROOT,MATRIX_PATH=$MATRIX_PATH,TASK_KIND=classical,ARRAY_LOG_ROOT=$LOG_ROOT,RESUME=$RESUME" \
  "$PROJECT_ROOT/scripts/run_daic_comprehensive_array_slurm.sh")"
classical_job="$(job_id "$classical_raw")"
SUBMISSION_PATH="$(dirname "$MATRIX_PATH")/submission_${STAGE}.json"
export SUBMISSION_PATH RUN_ID STAGE MATRIX_PATH DRY_RUN
export TRAIN_JOB_IDS="${train_jobs[*]}" REGULAR_INDICES="${COUNTS[4]}" MIL_INDICES="${COUNTS[5]}"
export EVALUATION_JOB_ID="$evaluation_job" HIDDEN_JOB_ID="$hidden_job" CLASSICAL_JOB_ID="$classical_job"
python - <<'PY'
import json, os
from pathlib import Path
matrix = json.loads(Path(os.environ["MATRIX_PATH"]).read_text(encoding="utf-8"))
payload = {
  "run_id": os.environ["RUN_ID"], "stage": os.environ["STAGE"],
  "matrix_path": os.environ["MATRIX_PATH"], "dry_run": os.environ["DRY_RUN"] == "1",
  "matrix_hash": matrix.get("matrix_hash"),
  "task_count": int(matrix.get("task_count", 0)),
  "kind_counts": matrix.get("kind_counts", {}),
  "maximum_concurrent_train": int(os.environ["MAX_CONCURRENT_TRAIN"]),
  "maximum_concurrent_evaluation": int(os.environ["MAX_CONCURRENT_EVAL"]),
  "maximum_concurrent_hidden": int(os.environ["MAX_CONCURRENT_HIDDEN"]),
  "maximum_concurrent_classical": int(os.environ["MAX_CONCURRENT_CLASSICAL"]),
  "arrays": {
    "train": {"job_ids": os.environ["TRAIN_JOB_IDS"].split(), "regular_indices": os.environ["REGULAR_INDICES"], "mil_indices": os.environ["MIL_INDICES"]},
    "evaluation": {"job_ids": [os.environ["EVALUATION_JOB_ID"]]},
    "hidden": {"job_ids": [os.environ["HIDDEN_JOB_ID"]]},
    "classical": {"job_ids": [os.environ["CLASSICAL_JOB_ID"]]},
  },
}
with open(os.environ["SUBMISSION_PATH"], "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True); handle.write("\n")
PY
printf 'train=%s\nevaluation=%s\nhidden=%s\nclassical=%s\nsubmission=%s\n' "${train_jobs[*]}" "$evaluation_job" "$hidden_job" "$classical_job" "$SUBMISSION_PATH"
