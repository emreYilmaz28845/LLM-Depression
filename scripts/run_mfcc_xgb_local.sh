#!/usr/bin/env bash
# Local CPU campaign: openSMILE MFCC functionals on the harmonized LLM windows,
# then fixed XGBoost and Optuna-100 XGBoost heads per fold.
#
#   bash scripts/run_mfcc_xgb_local.sh <androids|androids_standalone|cmdc|d3tec|daic|turkish_pooled> [...]
#
# Resumable: extraction is skipped when every fold cache exists, fixed heads
# when metrics.json exists, and Optuna resumes its own study.sqlite3.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

INPUTS_ROOT="${MFCC_INPUTS_ROOT:-/home/emre/Projects/AudioLLM/LLM-Depression}"
OUTPUT_ROOT="${MFCC_OUTPUT_ROOT:-$PROJECT_ROOT/outputs/mfcc_xgb}"
STAGED="$PROJECT_ROOT/outputs/mfcc_inputs"
SMILE_BIN="${SMILE_BIN:-/usr/local/bin/SMILExtract}"
SMILE_CONFIG="$PROJECT_ROOT/configs/mffcc_config/avec2013_3.conf"
WORKERS="${MFCC_WORKERS:-8}"
XGB_THREADS="${MFCC_XGB_THREADS:-8}"
GPFS_DATASETS=/gpfs/projects/etur92/ozu647717/AudioLLM/Datasets
LOCAL_DATASETS=/media/emre/Backup/AudioLLM/Datasets

dataset_args() {
  case "$1" in
    androids)
      CONFIG=configs/main/androids_audio_only_harmonized_selmacrof1_likelihood_v1.yaml
      FOLDS=(0 1 2 3 4)
      EXTRA=(--manifest-metadata "$INPUTS_ROOT/outputs/splits_harmonized/androids/androids_interview_manifest_metadata.json"
             --locked-folds "$STAGED/androids/androids_merged_folds_v1.json"
             --locked-folds-sha256 4e15cbbf35e6d426b17fd0c8020830a2ccefed421662de8800556553b331b46b) ;;
    androids_standalone)
      # Original standalone folds (androids_interview_folds.json), the split the
      # standalone Qwen2-Audio and Qwen3-Omni Androids runs were scored on.
      CONFIG=configs/main/androids_audio_only_harmonized_selmacrof1_likelihood_v1.yaml
      FOLDS=(0 1 2 3 4)
      EXTRA=(--manifest-metadata "$INPUTS_ROOT/outputs/splits_harmonized/androids/androids_interview_manifest_metadata.json") ;;
    cmdc)
      CONFIG=configs/main/cmdc_audio_only_harmonized_selmacrof1_likelihood_v1.yaml
      FOLDS=(0 1 2 3 4)
      EXTRA=(--manifest-metadata "$INPUTS_ROOT/outputs/splits_harmonized/cmdc/cmdc_manifest_metadata.json") ;;
    d3tec)
      CONFIG=configs/main/d3tec_audio_only_harmonized_selmacrof1_likelihood_v1.yaml
      FOLDS=(0 1 2 3 4)
      EXTRA=(--manifest-metadata "$INPUTS_ROOT/outputs/splits_harmonized/d3tec/d3tec_manifest_metadata.json") ;;
    daic)
      # Fixed official split: fit on official train, score the official test.
      CONFIG=configs/main/daic_audio_only_harmonized_selmacrof1_likelihood_v1.yaml
      FOLDS=(0)
      EXTRA=(--manifest-metadata "$INPUTS_ROOT/outputs/splits_harmonized/daic/daic_manifest_metadata.json") ;;
    turkish_pooled)
      CONFIG=configs/main/turkish_pooled_t17_audio_only_harmonized_selmacrof1_likelihood_v1_qwen3asr.yaml
      FOLDS=(0 1 2 3 4)
      EXTRA=(--manifest-metadata "$STAGED/turkish_pooled/splits/turkish_manifest_metadata.json"
             --manifest-path "$STAGED/turkish_pooled/manifests/turkish_manifest.jsonl"
             --folds-path "$STAGED/turkish_pooled/splits/turkish_folds.json"
             --path-map "$GPFS_DATASETS/Turkish/all-files/=$LOCAL_DATASETS/Turkish_Positive_Only/all-files/"
             --path-map "$GPFS_DATASETS/Turkish_Negative_Only/=$LOCAL_DATASETS/Turkish_Negative_Only/") ;;
    *) echo "Unknown dataset: $1" >&2; exit 2 ;;
  esac
}

run_dataset() {
  local dataset="$1"
  dataset_args "$dataset"
  local out="$OUTPUT_ROOT/$dataset"
  mkdir -p "$out"
  local missing=0
  for fold in "${FOLDS[@]}"; do
    [[ -f "$out/fold_$fold/cache/extraction_metadata.json" ]] || missing=1
  done
  if [[ $missing == 1 ]]; then
    echo "[$(date -u +%FT%TZ)] $dataset: extracting MFCC for folds ${FOLDS[*]}"
    python -m src.features.extract_opensmile_mfcc --config "$CONFIG" --inputs-root "$INPUTS_ROOT" \
      "${EXTRA[@]}" --folds "${FOLDS[@]}" --output-dir "$out" \
      --smile-bin "$SMILE_BIN" --smile-config "$SMILE_CONFIG" --workers "$WORKERS"
  fi
  for fold in "${FOLDS[@]}"; do
    local fold_dir="$out/fold_$fold"
    if [[ ! -f "$fold_dir/fixed_heads/xgb_raw/metrics.json" ]]; then
      echo "[$(date -u +%FT%TZ)] $dataset fold $fold: fixed xgb_raw"
      python baselines/qwen_hidden_classifier.py --cache-dir "$fold_dir/cache" \
        --output-dir "$fold_dir/fixed_heads" --variants xgb_raw > "$fold_dir/fixed_heads.log" 2>&1
    fi
    if [[ ! -f "$fold_dir/xgb_optuna100/metrics.json" ]]; then
      echo "[$(date -u +%FT%TZ)] $dataset fold $fold: Optuna-100 XGBoost"
      python baselines/qwen_hidden_xgb_optuna.py --cache-dir "$fold_dir/cache" \
        --output-dir "$fold_dir/xgb_optuna100" --experiment-id xgb_optuna100 \
        --objective macro_f1 --target-trials 100 --inner-folds 3 --seed 1337 --inner-seed 1337 \
        --sampling-mode none --xgb-threads "$XGB_THREADS" \
        --protocol-profile harmonized_optuna100_v1 >> "$fold_dir/xgb_optuna100.log" 2>&1
    fi
    echo "[$(date -u +%FT%TZ)] $dataset fold $fold: done"
  done
}

[[ $# -gt 0 ]] || { echo "usage: $0 <dataset> [...]" >&2; exit 2; }
for dataset in "$@"; do
  run_dataset "$dataset"
done
echo "[$(date -u +%FT%TZ)] all requested datasets finished"
