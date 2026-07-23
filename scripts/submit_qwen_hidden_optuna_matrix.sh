#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
MATRIX="${MATRIX:-$PROJECT_ROOT/configs/features/optuna_raw_matrix.yaml}"
DRY_RUN="${DRY_RUN:-0}"
TARGET_TRIALS="${TARGET_TRIALS:-50}"
INNER_FOLDS="${INNER_FOLDS:-3}"
SEED="${SEED:-1337}"
XGB_THREADS="${XGB_THREADS:-20}"
EXPECTED_JOBS="${EXPECTED_JOBS:-33}"
cd "$PROJECT_ROOT"

mapfile -t jobs < <(python - "$MATRIX" <<'PY'
import sys
from pathlib import Path

import yaml

matrix = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
expected = int(matrix.get("expected_jobs", 33))
rows = []
for item in matrix["experiments"]:
    objective = item["objective"]
    if item["dataset"] in {"daic", "cmdc"} and objective != "positive_f1":
        raise SystemExit(f"{item['dataset']} requires positive_f1, found {objective}")
    if item["dataset"] == "turkish" and objective != "macro_f1":
        raise SystemExit(f"turkish requires macro_f1, found {objective}")
    condition = item.get("condition", item["modality"])
    for fold in item["folds"]:
        rows.append(
            (
                item["dataset"],
                condition,
                item["modality"],
                str(fold),
                item["run_dir"],
                objective,
            )
        )
if len(rows) != expected:
    raise SystemExit(f"Expected {expected} jobs, found {len(rows)}")
for row in rows:
    print("\t".join(row))
PY
)

if [ "${#jobs[@]}" -ne "$EXPECTED_JOBS" ]; then
  echo "Expected $EXPECTED_JOBS jobs, found ${#jobs[@]}" >&2
  exit 1
fi

submitted=0
for job in "${jobs[@]}"; do
  IFS=$'\t' read -r dataset condition modality fold run_dir objective <<< "$job"
  run_name="$(basename "$run_dir")"
  cache="$PROJECT_ROOT/outputs/hidden_features/$dataset/$condition/$run_name/fold_$fold"
  output="$PROJECT_ROOT/outputs/hidden_classifiers/$dataset/$condition/$run_name/fold_$fold/xgb_optuna_raw"
  for required in outer_train.npz outer_train_rows.jsonl final_eval.npz final_eval_rows.jsonl extraction_metadata.json; do
    if [ ! -f "$cache/$required" ]; then
      echo "Missing cache input: $cache/$required" >&2
      exit 1
    fi
  done
  command=(
    sbatch
    --parsable
    --export="ALL,CACHE_DIR=$cache,OUTPUT_DIR=$output,OBJECTIVE=$objective,TARGET_TRIALS=$TARGET_TRIALS,INNER_FOLDS=$INNER_FOLDS,SEED=$SEED,XGB_THREADS=$XGB_THREADS"
    "$PROJECT_ROOT/scripts/run_qwen_hidden_optuna_slurm.sh"
  )
  if [ "$DRY_RUN" = "1" ]; then
    printf '%q ' "${command[@]}"
    printf '\n'
  else
    job_id="$("${command[@]}")"
    printf '%s\t%s\t%s\t%s\t%s\n' "$job_id" "$dataset" "$condition" "$fold" "$objective"
  fi
  submitted=$((submitted + 1))
done

if [ "$submitted" -ne "$EXPECTED_JOBS" ]; then
  echo "Expected $EXPECTED_JOBS submitted commands, got $submitted" >&2
  exit 1
fi

if [ "$DRY_RUN" = "1" ]; then
  echo "DRY_RUN command count: $submitted"
else
  echo "Submitted job count: $submitted"
fi
