#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
MATRIX="${MATRIX:-$PROJECT_ROOT/configs/features/turkish_oversampling_hidden_matrix.yaml}"
DRY_RUN="${DRY_RUN:-0}"
cd "$PROJECT_ROOT"

mapfile -t jobs < <(python - "$MATRIX" <<'PY'
import sys
import yaml

matrix = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
rows = []
for item in matrix["experiments"]:
    for fold in item["folds"]:
        rows.append(
            (
                item["dataset"],
                item["condition"],
                str(fold),
                item["run_dir"],
                matrix["experiment_id"],
            )
        )
if len(rows) != int(matrix["expected_jobs"]):
    raise SystemExit(f"Expected {matrix['expected_jobs']} jobs, found {len(rows)}")
if len({(row[1], row[2], row[4]) for row in rows}) != len(rows):
    raise SystemExit("Duplicate screening job identity")
for row in rows:
    print("\t".join(row))
PY
)

submitted=0
skipped=0
for job in "${jobs[@]}"; do
  IFS=$'\t' read -r dataset condition fold run_dir experiment_id <<< "$job"
  run_name="$(basename "$run_dir")"
  cache="$PROJECT_ROOT/outputs/hidden_features/$dataset/$condition/$run_name/fold_$fold"
  output="$PROJECT_ROOT/outputs/hidden_classifiers/$dataset/$condition/$run_name/fold_$fold/$experiment_id"
  for required in outer_train.npz outer_train_rows.jsonl extraction_metadata.json; do
    test -f "$cache/$required" || { echo "Missing cache input: $cache/$required" >&2; exit 1; }
  done
  if [ -f "$output/completion.json" ]; then
    python - "$output/completion.json" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
assert payload["status"] == "complete"
assert payload["expected_inner_fits"] == 42
assert payload["observed_sampling_audits"] == 42
PY
    printf 'SKIP complete\t%s\t%s\t%s\n' "$condition" "$fold" "$experiment_id"
    skipped=$((skipped + 1))
    continue
  fi
  command=(sbatch --parsable --export="ALL,CACHE_DIR=$cache,OUTPUT_DIR=$output,EXPERIMENT_ID=$experiment_id" "$PROJECT_ROOT/scripts/run_turkish_oversampling_screen_slurm.sh")
  if [ "$DRY_RUN" = "1" ]; then
    printf '%q ' "${command[@]}"; printf '\n'
  else
    job_id="$("${command[@]}")"
    printf '%s\t%s\t%s\t%s\n' "$job_id" "$condition" "$fold" "$experiment_id"
  fi
  submitted=$((submitted + 1))
done
test $((submitted + skipped)) -eq "${#jobs[@]}"
echo "Handled ${#jobs[@]} jobs: submitted=$submitted skipped=$skipped"
