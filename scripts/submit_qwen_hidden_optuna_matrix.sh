#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
MATRIX="${MATRIX:-$PROJECT_ROOT/configs/features/optuna_raw_matrix.yaml}"
DRY_RUN="${DRY_RUN:-0}"
TARGET_TRIALS="${TARGET_TRIALS:-50}"
INNER_FOLDS="${INNER_FOLDS:-3}"
SEED="${SEED:-1337}"
INNER_SEED="${INNER_SEED:-$SEED}"
EXPERIMENT_ID="${EXPERIMENT_ID:-xgb_optuna_raw}"
SEARCH_PROFILE="${SEARCH_PROFILE:-standard_d6}"
XGB_THREADS="${XGB_THREADS:-20}"
cd "$PROJECT_ROOT"

matrix_expected="$(python - "$MATRIX" <<'PY'
import sys
import yaml
matrix = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
print(int(matrix.get("expected_jobs", 33)))
PY
)"
EXPECTED_JOBS="${EXPECTED_JOBS:-$matrix_expected}"

mapfile -t jobs < <(python - "$MATRIX" <<'PY'
import sys
import re
from pathlib import Path

import yaml

matrix = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
expected = int(matrix.get("expected_jobs", 33))
rows = []
if "jobs" in matrix:
    expanded = matrix["jobs"]
else:
    expanded = []
    for item in matrix["experiments"]:
        condition = item.get("condition", item["modality"])
        for fold in item["folds"]:
            expanded.append({**item, "condition": condition, "fold": fold})
for item in expanded:
    objective = item["objective"]
    if item["dataset"] in {"daic", "cmdc"} and objective != "positive_f1":
        raise SystemExit(f"{item['dataset']} requires positive_f1, found {objective}")
    if item["dataset"] == "turkish" and objective != "macro_f1":
        raise SystemExit(f"turkish requires macro_f1, found {objective}")
    condition = item.get("condition", item["modality"])
    experiment_id = str(item.get("experiment_id", "-"))
    if experiment_id != "-" and not re.fullmatch(r"[a-z0-9]+(?:_[a-z0-9]+)*", experiment_id):
        raise SystemExit(f"Unsafe experiment_id: {experiment_id!r}")
    search_profile = str(item.get("search_profile", "-"))
    if search_profile not in {"-", "standard_d6", "depth8"}:
        raise SystemExit(f"Unsupported search_profile: {search_profile!r}")
    rows.append(
        (
            item["dataset"],
            condition,
            item["modality"],
            str(item["fold"]),
            item["run_dir"],
            objective,
            str(item.get("target_trials", "-")),
            str(item.get("inner_folds", "-")),
            str(item.get("seed", "-")),
            str(item.get("inner_seed", "-")),
            str(item.get("xgb_threads", "-")),
            experiment_id,
            search_profile,
            str(item.get("sampling_mode", "-")),
            str(item.get("oversampling_ratio", "-")),
            str(item.get("oversampling_seed", "-")),
        )
    )
if len(rows) != expected:
    raise SystemExit(f"Expected {expected} jobs, found {len(rows)}")
identities = [(row[0], row[1], row[3], row[11]) for row in rows]
if len(identities) != len(set(identities)):
    raise SystemExit("Matrix contains duplicate dataset/condition/fold/experiment identities")
for row in rows:
    print("\t".join(row))
PY
)

if [ "${#jobs[@]}" -ne "$EXPECTED_JOBS" ]; then
  echo "Expected $EXPECTED_JOBS jobs, found ${#jobs[@]}" >&2
  exit 1
fi

submitted=0
skipped=0
for job in "${jobs[@]}"; do
  IFS=$'\t' read -r dataset condition modality fold run_dir objective row_trials row_folds row_seed row_inner_seed row_threads row_experiment row_profile row_sampling_mode row_oversampling_ratio row_oversampling_seed <<< "$job"
  target_trials="$row_trials"; [ "$target_trials" = "-" ] && target_trials="$TARGET_TRIALS"
  inner_folds="$row_folds"; [ "$inner_folds" = "-" ] && inner_folds="$INNER_FOLDS"
  seed="$row_seed"; [ "$seed" = "-" ] && seed="$SEED"
  inner_seed="$row_inner_seed"; [ "$inner_seed" = "-" ] && inner_seed="$INNER_SEED"
  xgb_threads="$row_threads"; [ "$xgb_threads" = "-" ] && xgb_threads="$XGB_THREADS"
  experiment_id="$row_experiment"; [ "$experiment_id" = "-" ] && experiment_id="$EXPERIMENT_ID"
  search_profile="$row_profile"; [ "$search_profile" = "-" ] && search_profile="$SEARCH_PROFILE"
  sampling_mode="$row_sampling_mode"
  oversampling_ratio="$row_oversampling_ratio"
  oversampling_seed="$row_oversampling_seed"; [ "$oversampling_seed" = "-" ] && oversampling_seed="$seed"
  run_name="$(basename "$run_dir")"
  cache="$PROJECT_ROOT/outputs/hidden_features/$dataset/$condition/$run_name/fold_$fold"
  output="$PROJECT_ROOT/outputs/hidden_classifiers/$dataset/$condition/$run_name/fold_$fold/$experiment_id"
  for required in outer_train.npz outer_train_rows.jsonl final_eval.npz final_eval_rows.jsonl extraction_metadata.json; do
    if [ ! -f "$cache/$required" ]; then
      echo "Missing cache input: $cache/$required" >&2
      exit 1
    fi
  done
  if [ -d "$output" ]; then
    state="$(python - "$output" "$experiment_id" "$target_trials" "$seed" "$inner_seed" "$search_profile" "$inner_folds" "$xgb_threads" "$objective" "$sampling_mode" "$oversampling_ratio" "$oversampling_seed" <<'PY'
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
expected = {
    "experiment_id": sys.argv[2],
    "target_trials": int(sys.argv[3]),
    "seed": int(sys.argv[4]),
    "inner_seed": int(sys.argv[5]),
    "search_profile": sys.argv[6],
    "inner_fold_count": int(sys.argv[7]),
    "objective": sys.argv[9],
}
if sys.argv[10] != "-":
    expected.update(
        {
            "sampling_mode": sys.argv[10],
            "oversampling_ratio": None if sys.argv[11] == "-" else float(sys.argv[11]),
            "oversampling_seed": int(sys.argv[12]),
        }
    )
config_path = output / "study_config.json"
if not config_path.is_file():
    if any(output.iterdir()):
        raise SystemExit(f"Non-empty output has no study_config.json: {output}")
    print("new")
    raise SystemExit
payload = json.loads(config_path.read_text(encoding="utf-8"))["canonical_config"]
for key, value in expected.items():
    if payload.get(key) != value:
        raise SystemExit(
            f"Incompatible existing output {output}: {key}={payload.get(key)!r}, expected {value!r}"
        )
if int(payload.get("fixed_xgb_params", {}).get("n_jobs", -1)) != int(sys.argv[8]):
    raise SystemExit(
        f"Incompatible existing output {output}: XGBoost thread count differs"
    )
metadata_path = output / "classifier_metadata.json"
if metadata_path.is_file():
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    final_artifacts = (
        "pipeline.joblib",
        "predictions_sample_level.jsonl",
        "predictions_sample_level.csv",
        "predictions_subject_level.jsonl",
        "predictions_subject_level.csv",
        "metrics.json",
        "classifier_metadata.json",
    )
    if (
        metadata.get("experiment_id") == expected["experiment_id"]
        and int(metadata.get("completed_trials", -1)) == expected["target_trials"]
        and all((output / name).is_file() for name in final_artifacts)
    ):
        print("complete")
        raise SystemExit
print("resume")
PY
)"
    if [ "$state" = "complete" ]; then
      printf 'SKIP complete\t%s\t%s\t%s\t%s\n' "$dataset" "$condition" "$fold" "$experiment_id"
      skipped=$((skipped + 1))
      continue
    fi
  fi
  command=(
    sbatch
    --parsable
    --export="ALL,CACHE_DIR=$cache,OUTPUT_DIR=$output,OBJECTIVE=$objective,TARGET_TRIALS=$target_trials,INNER_FOLDS=$inner_folds,SEED=$seed,INNER_SEED=$inner_seed,XGB_THREADS=$xgb_threads,EXPERIMENT_ID=$experiment_id,SEARCH_PROFILE=$search_profile,SAMPLING_MODE=$sampling_mode,OVERSAMPLING_RATIO=$oversampling_ratio,OVERSAMPLING_SEED=$oversampling_seed"
    "$PROJECT_ROOT/scripts/run_qwen_hidden_optuna_slurm.sh"
  )
  if [ "$DRY_RUN" = "1" ]; then
    printf '%q ' "${command[@]}"
    printf '\n'
  else
    job_id="$("${command[@]}")"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$job_id" "$dataset" "$condition" "$fold" "$objective" "$experiment_id"
  fi
  submitted=$((submitted + 1))
done

if [ $((submitted + skipped)) -ne "$EXPECTED_JOBS" ]; then
  echo "Expected $EXPECTED_JOBS handled jobs, got submitted=$submitted skipped=$skipped" >&2
  exit 1
fi

if [ "$DRY_RUN" = "1" ]; then
  echo "DRY_RUN command count: $submitted (complete skipped: $skipped)"
else
  echo "Submitted job count: $submitted (complete skipped: $skipped)"
fi
