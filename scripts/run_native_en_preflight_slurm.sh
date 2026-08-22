#!/bin/bash
#SBATCH -J nat-en-preflight
#SBATCH -A etur92
#SBATCH -q acc_ehpc
#SBATCH -t 02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH -o /dev/null
#SBATCH -e /dev/null

# CPU-only preflight for the native-versus-English text-only head study.
# Optionally builds the four merged protocol artifacts first, then writes the
# hashed audit JSON. No training happens here.
#
# Path contract learned on MN5 (see agent journal 2026-08-22):
# - CODE is the deployed code snapshot; it carries configs and src but NOT
#   .deps (gitignored) nor manifests.
# - Vendored pinned deps live under the PERMANENT tree: optuna/xgboost/sklearn
#   resolve via QWEN_HIDDEN_DEPS -> PERMANENT/.deps/qwen_hidden.
# - Protocol builds must run with PROJECT_ROOT=$PERMANENT so the component
#   manifest paths (${PROJECT_ROOT}/outputs/manifests_harmonized…) resolve in
#   the permanent tree, while --config uses absolute $CODE paths so the exact
#   deployed config bytes are what gets built.
set -euo pipefail
module purge
module load bsc/1.0
module load miniforge/24.3.0-0

CODE="${CODE:?Set CODE to the deployed code path}"
PERMANENT="${PERMANENT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
TRANSLATION_ROOT="${TRANSLATION_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/translations}"
EXPECTED_COMMIT="${SOURCE_COMMIT:?Set SOURCE_COMMIT}"
OUTPUT="${PREFLIGHT_OUTPUT:?Set PREFLIGHT_OUTPUT}"
BUILD_MERGED_PROTOCOLS="${BUILD_MERGED_PROTOCOLS:-0}"
WITH_TOKENIZER="${WITH_TOKENIZER:-0}"
RUN_NAMES_FILE="${RUN_NAMES_FILE:-}"
MERGED_RUN_IDS_FILE="${MERGED_RUN_IDS_FILE:-}"

ENV_ACTIVATE="${ENV_ACTIVATE:-/gpfs/projects/etur92/ozu647717/venvs/qwen_mn5_rebuilt/bin/activate}"
QWEN_HIDDEN_DEPS="${QWEN_HIDDEN_DEPS:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression/.deps/qwen_hidden}"
source "$ENV_ACTIVATE"
export PYTHONPATH="$QWEN_HIDDEN_DEPS:$CODE${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

mkdir -p "$(dirname "$OUTPUT")"
LOG_PREFIX="${PREFLIGHT_LOG_PREFIX:-$(dirname "$OUTPUT")/preflight-${SLURM_JOB_ID:-local}}"
exec > >(tee -a "$LOG_PREFIX.out")
exec 2> >(tee -a "$LOG_PREFIX.err" >&2)

cd "$CODE"
if [ "$BUILD_MERGED_PROTOCOLS" = "1" ]; then
    export PROJECT_ROOT="$PERMANENT"
    for variant in native_qwen english_qwen native_gemma4 english_gemma4; do
        cfg="$CODE/configs/experiments/merged/symmetric_merged_text_heads_${variant}.yaml"
        out_dir="$PERMANENT/outputs/symmetric_merged/native_en_text_heads_v1/${variant}_text_only"
        echo "building merged protocol: $cfg -> $out_dir"
        python scripts/build_symmetric_merged_manifest.py \
            --config "$cfg" --output-dir "$out_dir"
    done
    unset PROJECT_ROOT
fi

ARGS=(
    --mode mn5
    --expected-commit "$EXPECTED_COMMIT"
    --project-root "$PERMANENT"
    --translation-root "$TRANSLATION_ROOT"
    --output "$OUTPUT"
)
python - "$PERMANENT" > "$OUTPUT.manifest_pairs.json" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
name_by_dataset = {
    "d3tec": "d3tec_manifest.jsonl",
    "androids_interview": "androids_interview_manifest.jsonl",
    "cmdc": "cmdc_manifest.jsonl",
    "turkish": "turkish_manifest.jsonl",
}
subdir_by_dataset = {
    "d3tec": "d3tec",
    "androids_interview": "androids",
    "cmdc": "cmdc",
    "turkish": "turkish_t17_qwen3asr",
}
pairs = {
    ds: [
        str(root / f"outputs/manifests_harmonized/{subdir}/{name}"),
        str(root / f"outputs/manifests_harmonized_en/{subdir}/{name}"),
    ]
    for ds, name in name_by_dataset.items()
    for subdir in [subdir_by_dataset[ds]]
}
print(json.dumps(pairs))
PY
ARGS+=(--manifest-pairs "$OUTPUT.manifest_pairs.json")

DATASET_BASE_ROOT="${DATASET_BASE_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/Datasets}"
mapfile -t DATASET_ROOT_ARGS < <(python - "$DATASET_BASE_ROOT" <<'PY'
import os, sys
base = sys.argv[1]
defaults = {
    "DAIC_DATASET_ROOT": f"{base}/DAIC-WOZ/preprocessed",
    "D3TEC_DATASET_ROOT": f"{base}/D3TEC DATASET/D3TEC DATASET",
    "CMDC_DATASET_ROOT": f"{base}/CMDC",
    "TURKISH_DATASET_ROOT": f"{base}/Turkish",
}
for key, default in defaults.items():
    print(f"{key}={os.environ.get(key, default)}")
PY
)
if [ "${#DATASET_ROOT_ARGS[@]}" -gt 0 ]; then
    ARGS+=("--dataset-roots" "${DATASET_ROOT_ARGS[@]}")
fi

if [ -n "$RUN_NAMES_FILE" ] && [ -f "$RUN_NAMES_FILE" ]; then
    mapfile -t RUN_NAMES < "$RUN_NAMES_FILE"
    if [ "${#RUN_NAMES[@]}" -gt 0 ]; then
        ARGS+=("--run-names" "${RUN_NAMES[@]}")
    fi
fi
if [ -n "$MERGED_RUN_IDS_FILE" ] && [ -f "$MERGED_RUN_IDS_FILE" ]; then
    mapfile -t MERGED_RUN_IDS < "$MERGED_RUN_IDS_FILE"
    if [ "${#MERGED_RUN_IDS[@]}" -gt 0 ]; then
        ARGS+=("--merged-run-ids" "${MERGED_RUN_IDS[@]}")
    fi
fi
if [ "$WITH_TOKENIZER" = "1" ]; then
    ARGS+=(--with-tokenizer)
fi

python scripts/preflight_native_en_text_heads.py "${ARGS[@]}"
