#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
MATRIX="${MATRIX:?Set MATRIX}"
DRY_RUN="${DRY_RUN:-0}"
cd "$PROJECT_ROOT"

mapfile -t jobs < <(python - "$MATRIX" <<'PY'
import sys, yaml
matrix = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
jobs = matrix["jobs"]
assert len(jobs) == int(matrix["expected_jobs"])
identities = [(job["run_name"], int(job["fold"])) for job in jobs]
assert len(identities) == len(set(identities))
for job in jobs:
    print(
        job["modality"], job["config"], job["fold"], job["profile"],
        job["run_name"], job["sampling_mode"],
        "-" if job.get("oversampling_ratio") is None else job["oversampling_ratio"],
        "-" if job.get("oversampling_seed") is None else job["oversampling_seed"],
        job["chain_key"], sep="\t"
    )
PY
)

declare -A dependencies
submitted=0
skipped=0
for job in "${jobs[@]}"; do
  IFS=$'\t' read -r modality config_rel fold profile run_name sampling_mode ratio os_seed chain_key <<< "$job"
  config="$PROJECT_ROOT/$config_rel"
  run_root="$(python - "$config" <<'PY'
import sys
sys.path.insert(0, ".")
from src.utils import load_yaml, resolve_project_path
print(resolve_project_path(load_yaml(sys.argv[1])["output_dirs"]["run_root"]))
PY
)"
  output="$run_root/$run_name/fold_$fold"
  extra_args="--set training.class_balance=$sampling_mode"
  if [ "$profile" = "oversampled" ]; then
    extra_args+=" --set training.oversampling_ratio=$ratio --set training.oversampling_seed=$os_seed"
  fi
  if [ -f "$output/run_config.yaml" ] && [ -f "$output/eval/best_validation/metrics_original_teacher_forced.json" ]; then
    python - "$output/run_config.yaml" "$sampling_mode" "$ratio" "$os_seed" <<'PY'
import sys, yaml
config = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))["config"]["training"]
assert config["class_balance"] == sys.argv[2]
if sys.argv[2] == "minority_subject_oversample":
    assert float(config["oversampling_ratio"]) == float(sys.argv[3])
    assert int(config["oversampling_seed"]) == int(sys.argv[4])
PY
    echo "SKIP complete $run_name fold=$fold"
    skipped=$((skipped + 1))
    dependencies["$chain_key"]=""
    continue
  fi
  dependency="${dependencies[$chain_key]:-}"
  log_root="$PROJECT_ROOT/logs/slurm_turkish_oversampling/$run_name"
  if [ "$DRY_RUN" = "1" ]; then
    printf 'PROJECT_ROOT=%q CONFIG=%q FOLD=%q RUN_NAME=%q EXTRA_TRAIN_ARGS=%q LOG_ROOT=%q SBATCH_DEPENDENCY=%q bash %q\n' \
      "$PROJECT_ROOT" "$config" "$fold" "$run_name" "$extra_args" "$log_root" "$dependency" \
      "$PROJECT_ROOT/scripts/submit_train_and_eval.sh"
    submitted=$((submitted + 1))
    continue
  fi
  submission="$(env PROJECT_ROOT="$PROJECT_ROOT" CONFIG="$config" FOLD="$fold" RUN_NAME="$run_name" \
    EXTRA_TRAIN_ARGS="$extra_args" LOG_ROOT="$log_root" SBATCH_DEPENDENCY="$dependency" \
    SUBMIT_BEST_EVAL=0 SUBMIT_LAST_EVAL=0 SKIP_MANIFEST_BUILD=0 \
    bash "$PROJECT_ROOT/scripts/submit_train_and_eval.sh")"
  printf '%s\n' "$submission"
  train_job="$(printf '%s\n' "$submission" | awk '/Submitted training job:/{print $NF; exit}')"
  test -n "$train_job" || { echo "Could not parse training job ID" >&2; exit 1; }
  dependencies["$chain_key"]="afterok:$train_job"
  printf '%s\t%s\t%s\t%s\n' "$train_job" "$run_name" "$fold" "$profile"
  submitted=$((submitted + 1))
done
test $((submitted + skipped)) -eq "${#jobs[@]}"
echo "Handled ${#jobs[@]} Qwen jobs: submitted=$submitted skipped=$skipped"
