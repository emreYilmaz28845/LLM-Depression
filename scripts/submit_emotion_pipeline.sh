#!/usr/bin/env bash

# Submit the full offline emotion-cache pipeline as chained SLURM jobs:
#
#   1. extract  (SECap zh captions; array job of SHARDS tasks, GPU)
#   2. merge     (only when SHARDS > 1; combine shard files -> *_zh.jsonl)
#   3. translate (zh -> en, NLLB; GPU)         depends on extract (+merge)
#   4. validate  (coverage of *_en.jsonl vs the manifest; CPU)  depends on translate
#
# Each stage waits for the previous via --dependency=afterok. Override anything
# via environment variables, e.g.:
#
#   DATASET=edaic SHARDS=8 SECAP_ROOT=/gpfs/projects/etur92/SECap \
#     scripts/submit_emotion_pipeline.sh
#
# Smoke test on a few clips:  DATASET=daic SHARDS=1 FAST=1 LIMIT=4 scripts/submit_emotion_pipeline.sh

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
DATASET="${DATASET:-daic}"
SHARDS="${SHARDS:-8}"

MANIFEST="${MANIFEST:-$PROJECT_ROOT/outputs/manifests/${DATASET}_manifest.jsonl}"
SECAP_ROOT="${SECAP_ROOT:-/gpfs/projects/etur92/SECap}"
CKPT="${CKPT:-$SECAP_ROOT/model.ckpt}"
OUT_ZH="${OUT_ZH:-$PROJECT_ROOT/outputs/emotion/${DATASET}_secap_zh.jsonl}"
OUT_EN="${OUT_EN:-$PROJECT_ROOT/outputs/emotion/${DATASET}_secap_en.jsonl}"
MERGE_PATTERN="${MERGE_PATTERN:-$PROJECT_ROOT/outputs/emotion/${DATASET}_secap_zh.shard*of*.jsonl}"
TRANSLATE_MODEL="${TRANSLATE_MODEL:-facebook/nllb-200-distilled-600M}"
FAST="${FAST:-0}"
LIMIT="${LIMIT:-0}"
STRICT="${STRICT:-0}"
RUN_VALIDATE="${RUN_VALIDATE:-1}"

# SECap env selection (passed through to every job).
SECAP_ENV_ACTIVATE="${SECAP_ENV_ACTIVATE:-/gpfs/projects/etur92/ozu647717/venvs/secap_rebuilt/bin/activate}"
SECAP_CONDA_ENV="${SECAP_CONDA_ENV:-secap}"

EXTRACT_SCRIPT="${EXTRACT_SCRIPT:-$PROJECT_ROOT/scripts/run_emotion_extract_slurm.sh}"
TRANSLATE_SCRIPT="${TRANSLATE_SCRIPT:-$PROJECT_ROOT/scripts/run_emotion_translate_slurm.sh}"
BUILD_SCRIPT="${BUILD_SCRIPT:-$PROJECT_ROOT/scripts/run_emotion_build_cache_slurm.sh}"

for script in "$EXTRACT_SCRIPT" "$TRANSLATE_SCRIPT" "$BUILD_SCRIPT"; do
    if [ ! -f "$script" ]; then
        echo "Job script not found: $script"
        exit 1
    fi
done

COMMON_EXPORT="PROJECT_ROOT=$PROJECT_ROOT,DATASET=$DATASET,MANIFEST=$MANIFEST,OUT_ZH=$OUT_ZH,OUT_EN=$OUT_EN,SECAP_ENV_ACTIVATE=$SECAP_ENV_ACTIVATE,SECAP_CONDA_ENV=$SECAP_CONDA_ENV"

echo "Submitting emotion pipeline"
echo "  dataset       : $DATASET"
echo "  manifest      : $MANIFEST"
echo "  secap_root    : $SECAP_ROOT"
echo "  ckpt          : $CKPT"
echo "  shards        : $SHARDS"
echo "  out_zh        : $OUT_ZH"
echo "  out_en        : $OUT_EN"
echo "  translate     : $TRANSLATE_MODEL"
echo "  fast/limit    : $FAST / $LIMIT"

# --- 1. extract (array job) --------------------------------------------------
EXTRACT_EXPORT="ALL,$COMMON_EXPORT,SECAP_ROOT=$SECAP_ROOT,CKPT=$CKPT,SHARDS=$SHARDS,FAST=$FAST,LIMIT=$LIMIT"
ARRAY_SPEC="0-$((SHARDS - 1))"
EXTRACT_JOB_RAW="$(sbatch --parsable --array="$ARRAY_SPEC" --export="$EXTRACT_EXPORT" "$EXTRACT_SCRIPT")"
EXTRACT_JOB_ID="${EXTRACT_JOB_RAW%%;*}"
echo "Submitted extract array job: $EXTRACT_JOB_ID (array $ARRAY_SPEC)"

# --- 2. merge (only when sharded) -------------------------------------------
TRANSLATE_DEP="afterok:$EXTRACT_JOB_ID"
if [ "$SHARDS" -gt 1 ]; then
    MERGE_EXPORT="ALL,$COMMON_EXPORT,MODE=merge,MERGE_PATTERN=$MERGE_PATTERN"
    MERGE_JOB_RAW="$(sbatch --parsable --dependency=afterok:"$EXTRACT_JOB_ID" --export="$MERGE_EXPORT" "$BUILD_SCRIPT")"
    MERGE_JOB_ID="${MERGE_JOB_RAW%%;*}"
    echo "Submitted merge job: $MERGE_JOB_ID"
    TRANSLATE_DEP="afterok:$MERGE_JOB_ID"
fi

# --- 3. translate ------------------------------------------------------------
TRANSLATE_EXPORT="ALL,$COMMON_EXPORT,IN_ZH=$OUT_ZH,TRANSLATE_MODEL=$TRANSLATE_MODEL"
TRANSLATE_JOB_RAW="$(sbatch --parsable --dependency="$TRANSLATE_DEP" --export="$TRANSLATE_EXPORT" "$TRANSLATE_SCRIPT")"
TRANSLATE_JOB_ID="${TRANSLATE_JOB_RAW%%;*}"
echo "Submitted translate job: $TRANSLATE_JOB_ID (dep $TRANSLATE_DEP)"

# --- 4. validate -------------------------------------------------------------
if [ "$RUN_VALIDATE" = "1" ]; then
    VALIDATE_EXPORT="ALL,$COMMON_EXPORT,MODE=validate,STRICT=$STRICT"
    VALIDATE_JOB_RAW="$(sbatch --parsable --dependency=afterok:"$TRANSLATE_JOB_ID" --export="$VALIDATE_EXPORT" "$BUILD_SCRIPT")"
    VALIDATE_JOB_ID="${VALIDATE_JOB_RAW%%;*}"
    echo "Submitted validate job: $VALIDATE_JOB_ID (dep afterok:$TRANSLATE_JOB_ID)"
fi

echo "Done. Final cache will be at: $OUT_EN"
