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
source "$ENV_ACTIVATE"
export PYTHONPATH="$CODE/.deps/qwen_hidden:$CODE${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

mkdir -p "$(dirname "$OUTPUT")"

cd "$CODE"
if [ "$BUILD_MERGED_PROTOCOLS" = "1" ]; then
    for variant in native_qwen english_qwen native_gemma4 english_gemma4; do
        cfg="configs/experiments/merged/symmetric_merged_text_heads_${variant}.yaml"
        out_dir="$PERMANENT/outputs/symmetric_merged/native_en_text_heads_v1/${variant}_text_only"
        echo "building merged protocol: $cfg -> $out_dir"
        python scripts/build_symmetric_merged_manifest.py \
            --config "$cfg" --output-dir "$out_dir"
    done
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
pairs = {
    "d3tec": [
        str(root / "outputs/manifests_harmonized/d3tec/d3tec_manifest.jsonl"),
        str(root / "outputs/manifests_harmonized_en/d3tec/d3tec_manifest.jsonl"),
    ],
    "androids_interview": [
        str(root / "outputs/manifests_harmonized/androids/androids_interview_manifest.jsonl"),
        str(root / "outputs/manifests_harmonized_en/androids/androids_interview_manifest.jsonl"),
    ],
    "cmdc": [
        str(root / "outputs/manifests_harmonized/cmdc/cmdc_manifest.jsonl"),
        str(root / "outputs/manifests_harmonized_en/cmdc/cmdc_manifest.jsonl"),
    ],
    "turkish": [
        str(root / "outputs/manifests_harmonized/turkish_t17_qwen3asr/turkish_manifest.jsonl"),
        str(root / "outputs/manifests_harmonized_en/turkish_t17_qwen3asr/turkish_manifest.jsonl"),
    ],
}
print(json.dumps(pairs))
PY
ARGS+=(--manifest-pairs "$OUTPUT.manifest_pairs.json")

mapfile -t DATASET_ROOT_ARGS < <(python - <<'PY'
import os
for key in ("DAIC_DATASET_ROOT", "D3TEC_DATASET_ROOT", "CMDC_DATASET_ROOT", "TURKISH_DATASET_ROOT"):
    value = os.environ.get(key)
    if value:
        print(f"{key}={value}")
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
