#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
MATRIX="${MATRIX:-$PROJECT_ROOT/configs/features/primary_matrix.yaml}"
DRY_RUN="${DRY_RUN:-0}"
cd "$PROJECT_ROOT"

python - "$MATRIX" <<'PY' | while IFS=$'\t' read -r dataset modality fold run_dir; do
import sys, yaml
matrix = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
for item in matrix["experiments"]:
    for fold in item["folds"]:
        print(item["dataset"], item["modality"], fold, item["run_dir"], sep="\t")
PY
  checkpoint="$PROJECT_ROOT/$run_dir/fold_$fold/best_model"
  cache="$PROJECT_ROOT/outputs/hidden_features/$dataset/$modality/$(basename "$run_dir")/fold_$fold"
  classifiers="$PROJECT_ROOT/outputs/hidden_classifiers/$dataset/$modality/$(basename "$run_dir")/fold_$fold"
  if [ ! -f "$checkpoint/adapter_model.safetensors" ]; then
    echo "Missing checkpoint: $checkpoint" >&2
    exit 1
  fi
  command=(sbatch --parsable --export="ALL,CHECKPOINT_DIR=$checkpoint,CACHE_DIR=$cache,CLASSIFIER_DIR=$classifiers" "$PROJECT_ROOT/scripts/run_qwen_hidden_extract_slurm.sh")
  if [ "$DRY_RUN" = "1" ]; then
    printf 'DRY RUN: '; printf '%q ' "${command[@]}"; printf '\n'
  else
    job_id="$("${command[@]}")"
    printf '%s\t%s\t%s\t%s\n' "$job_id" "$dataset" "$modality" "$fold"
  fi
done
