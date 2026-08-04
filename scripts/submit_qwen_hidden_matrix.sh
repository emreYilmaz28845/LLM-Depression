#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
MATRIX="${MATRIX:-$PROJECT_ROOT/configs/features/primary_matrix.yaml}"
DRY_RUN="${DRY_RUN:-0}"
cd "$PROJECT_ROOT"

python - "$MATRIX" <<'PY' | while IFS=$'\t' read -r dataset condition modality fold run_dir emotion_source emotion_language classifier_variants; do
import sys, yaml
matrix = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
variants = matrix.get("variants") or []
if not isinstance(variants, list) or any(not isinstance(value, str) or not value for value in variants):
    raise SystemExit("matrix.variants must be a list of non-empty classifier names")
variant_csv = ":".join(variants) if variants else "-"
for item in matrix["experiments"]:
    for fold in item["folds"]:
        print(
            item["dataset"],
            item.get("condition", item["modality"]),
            item["modality"],
            fold,
            item["run_dir"],
            item.get("emotion_source", "-"),
            item.get("emotion_language", "-"),
            variant_csv,
            sep="\t",
        )
PY
  if [ "$emotion_source" = "-" ]; then emotion_source=""; fi
  if [ "$emotion_language" = "-" ]; then emotion_language=""; fi
  if [ "$classifier_variants" = "-" ]; then classifier_variants=""; fi
  checkpoint="$PROJECT_ROOT/$run_dir/fold_$fold/best_model"
  cache="$PROJECT_ROOT/outputs/hidden_features/$dataset/$condition/$(basename "$run_dir")/fold_$fold"
  classifiers="$PROJECT_ROOT/outputs/hidden_classifiers/$dataset/$condition/$(basename "$run_dir")/fold_$fold"
  if [ ! -f "$checkpoint/adapter_model.safetensors" ]; then
    echo "Missing checkpoint: $checkpoint" >&2
    exit 1
  fi
  command=(sbatch --parsable --export="ALL,CHECKPOINT_DIR=$checkpoint,CACHE_DIR=$cache,CLASSIFIER_DIR=$classifiers,CONDITION=$condition,EMOTION_SOURCE=$emotion_source,EMOTION_LANGUAGE=$emotion_language,CLASSIFIER_VARIANTS=$classifier_variants" "$PROJECT_ROOT/scripts/run_qwen_hidden_extract_slurm.sh")
  if [ "$DRY_RUN" = "1" ]; then
    printf 'DRY RUN: '; printf '%q ' "${command[@]}"; printf '\n'
  else
    job_id="$("${command[@]}")"
    printf '%s\t%s\t%s\t%s\n' "$job_id" "$dataset" "$condition" "$fold"
  fi
done
