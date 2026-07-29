#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
STAGE="${STAGE:-extract_fixed}"
RUN_ID="${RUN_ID:?Set a unique UTC RUN_ID}"
DRY_RUN="${DRY_RUN:-1}"
SOURCE_COMMIT="${SOURCE_COMMIT:-$(git -C "$PROJECT_ROOT" rev-parse HEAD)}"
REGISTRY="$PROJECT_ROOT/outputs/d3tec_hidden_jobs/$RUN_ID.tsv"
mkdir -p "$(dirname "$REGISTRY")"

record() {
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$1" "$2" "$3" "$4" "$5" "$6" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$SOURCE_COMMIT" \
    >> "$REGISTRY"
}

if [ ! -f "$REGISTRY" ]; then
  printf 'job_id\tstage\tmodality\tfold\texperiment_id\tpath\ttimestamp_utc\tsource_commit\n' > "$REGISTRY"
fi

if [ "$STAGE" = "extract_fixed" ]; then
  MATRIX="${MATRIX:-$PROJECT_ROOT/configs/features/d3tec_hidden_matrix.yaml}"
  mapfile -t jobs < <(python - "$MATRIX" <<'PY'
import sys, yaml
matrix = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
rows = []
for item in matrix["experiments"]:
    for fold in item["folds"]:
        rows.append((
            item["modality"], item["condition"], str(fold), item["run_dir"],
            " ".join(item["variants"]),
        ))
if len(rows) != int(matrix["expected_jobs"]) or len(rows) != 15:
    raise SystemExit(f"Expected 15 extraction/fixed identities, found {len(rows)}")
for row in rows:
    print("\t".join(row))
PY
)
  for job in "${jobs[@]}"; do
    IFS=$'\t' read -r modality condition fold run_dir variants <<< "$job"
    run_name="$(basename "$run_dir")"
    checkpoint="$PROJECT_ROOT/$run_dir/fold_$fold/best_model"
    cache="$PROJECT_ROOT/outputs/hidden_features/d3tec/$condition/$run_name/fold_$fold"
    output="$PROJECT_ROOT/outputs/hidden_classifiers/d3tec/$condition/$run_name/fold_$fold"
    [ -f "$checkpoint/adapter_model.safetensors" ] || {
      echo "Missing checkpoint: $checkpoint" >&2
      exit 1
    }
    extract=(sbatch --parsable
      --export="ALL,CHECKPOINT_DIR=$checkpoint,CACHE_DIR=$cache,CONDITION=$condition,SKIP_CLASSIFIERS=1,LOG_ROOT=$PROJECT_ROOT/logs/slurm_d3tec_hidden/$RUN_ID"
      "$PROJECT_ROOT/scripts/run_qwen_hidden_extract_slurm.sh")
    if [ "$DRY_RUN" = "1" ]; then
      printf 'DRY RUN extraction: '; printf '%q ' "${extract[@]}"; printf '\n'
      printf 'DRY RUN fixed dependency for %s fold %s\n' "$condition" "$fold"
      continue
    fi
    extract_id="$("${extract[@]}")"
    record "$extract_id" extraction "$modality" "$fold" hidden_cache "$cache"
    fixed_id="$(sbatch --parsable --dependency="afterok:$extract_id" \
      --export="ALL,CACHE_DIR=$cache,OUTPUT_DIR=$output,VARIANTS=$variants,RUN_ID=$RUN_ID" \
      "$PROJECT_ROOT/scripts/run_d3tec_hidden_fixed_slurm.sh")"
    record "$fixed_id" fixed_heads "$modality" "$fold" fixed_matrix "$output"
    printf '%s\t%s\t%s\t%s\n' "$extract_id" "$fixed_id" "$condition" "$fold"
  done
elif [ "$STAGE" = "optuna" ]; then
  MATRIX="${MATRIX:?Set MATRIX to a generated D3TEC Optuna stage manifest}"
  mapfile -t jobs < <(python - "$MATRIX" <<'PY'
import sys, yaml
matrix = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
rows = matrix["jobs"]
if len(rows) != int(matrix["expected_jobs"]):
    raise SystemExit("Optuna job count differs from expected_jobs")
for row in rows:
    print(
        row["modality"], row["condition"], row["fold"], row["run_dir"],
        row["objective"], row["target_trials"], row["inner_folds"], row["seed"],
        row["inner_seed"], row["experiment_id"], row["search_profile"],
        row["xgb_threads"], sep="\t",
    )
PY
)
  for job in "${jobs[@]}"; do
    IFS=$'\t' read -r modality condition fold run_dir objective trials inner_folds seed inner_seed experiment_id profile threads <<< "$job"
    run_name="$(basename "$run_dir")"
    cache="$PROJECT_ROOT/outputs/hidden_features/d3tec/$condition/$run_name/fold_$fold"
    output="$PROJECT_ROOT/outputs/hidden_classifiers/d3tec/$condition/$run_name/fold_$fold/$experiment_id"
    [ -f "$cache/extraction_metadata.json" ] || {
      echo "Missing cache: $cache" >&2
      exit 1
    }
    command=(sbatch --parsable
      --export="ALL,CACHE_DIR=$cache,OUTPUT_DIR=$output,OBJECTIVE=$objective,TARGET_TRIALS=$trials,INNER_FOLDS=$inner_folds,SEED=$seed,INNER_SEED=$inner_seed,XGB_THREADS=$threads,EXPERIMENT_ID=$experiment_id,SEARCH_PROFILE=$profile,LOG_ROOT=$PROJECT_ROOT/logs/slurm_d3tec_hidden/$RUN_ID/$experiment_id"
      "$PROJECT_ROOT/scripts/run_qwen_hidden_optuna_slurm.sh")
    if [ "$DRY_RUN" = "1" ]; then
      printf 'DRY RUN Optuna: '; printf '%q ' "${command[@]}"; printf '\n'
      continue
    fi
    job_id="$("${command[@]}")"
    record "$job_id" optuna "$modality" "$fold" "$experiment_id" "$output"
    printf '%s\t%s\t%s\t%s\n' "$job_id" "$condition" "$fold" "$experiment_id"
  done
else
  echo "Unsupported STAGE=$STAGE" >&2
  exit 1
fi
