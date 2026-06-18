#!/usr/bin/env bash

# Submit the full offline Qwen2-Audio emotion-cache pipeline as chained SLURM jobs:
#
#   1. extract  (Qwen2-Audio en captions; array job of SHARDS tasks, GPU)
#   2. merge     (only when SHARDS > 1; combine shard files -> canonical *_en.jsonl)
#   3. validate  (coverage of *_en.jsonl vs the manifest; CPU)  depends on extract (+merge)
#
# Unlike the SECap pipeline there is no translate stage: Qwen2-Audio writes
# emotion_en directly. Each stage waits for the previous via --dependency=afterok.
# Override anything via environment variables, e.g.:
#
#   DATASET=edaic SHARDS=8 scripts/submit_emotion_pipeline_qwen2audio.sh
#
# Smoke test on a few clips:  DATASET=daic SHARDS=1 LIMIT=4 scripts/submit_emotion_pipeline_qwen2audio.sh

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
DATASET="${DATASET:-daic}"
SHARDS="${SHARDS:-8}"

MANIFEST="${MANIFEST:-$PROJECT_ROOT/outputs/manifests/${DATASET}_manifest.jsonl}"
QWEN_MODEL="${QWEN_MODEL:-/gpfs/projects/etur92/ozu647717/models/Qwen2-Audio-7B-Instruct}"
OUT_EN="${OUT_EN:-$PROJECT_ROOT/outputs/emotion/${DATASET}_qwen2audio_en.jsonl}"
MERGE_PATTERN="${MERGE_PATTERN:-$PROJECT_ROOT/outputs/emotion/${DATASET}_qwen2audio_en.shard*of*.jsonl}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-48}"
LIMIT="${LIMIT:-0}"
STRICT="${STRICT:-0}"
RUN_VALIDATE="${RUN_VALIDATE:-1}"

# Env activation passed through to every job (qwen env works fine for the pure
# JSONL merge/validate stage too, since it only needs a base python).
ENV_ACTIVATE="${ENV_ACTIVATE:-/gpfs/projects/etur92/ozu647717/venvs/qwen_mn5_rebuilt/bin/activate}"

EXTRACT_SCRIPT="${EXTRACT_SCRIPT:-$PROJECT_ROOT/scripts/run_emotion_extract_qwen2audio_slurm.sh}"
BUILD_SCRIPT="${BUILD_SCRIPT:-$PROJECT_ROOT/scripts/run_emotion_build_cache_slurm.sh}"

for script in "$EXTRACT_SCRIPT" "$BUILD_SCRIPT"; do
    if [ ! -f "$script" ]; then
        echo "Job script not found: $script"
        exit 1
    fi
done

COMMON_EXPORT="PROJECT_ROOT=$PROJECT_ROOT,DATASET=$DATASET,MANIFEST=$MANIFEST,OUT_EN=$OUT_EN,SECAP_ENV_ACTIVATE=$ENV_ACTIVATE,ENV_ACTIVATE=$ENV_ACTIVATE"

echo "Submitting Qwen2-Audio emotion pipeline"
echo "  dataset       : $DATASET"
echo "  manifest      : $MANIFEST"
echo "  qwen_model    : $QWEN_MODEL"
echo "  shards        : $SHARDS"
echo "  out_en        : $OUT_EN"
echo "  max_new_tokens: $MAX_NEW_TOKENS"
echo "  limit         : $LIMIT"
echo "  strict        : $STRICT"

# --- 1. extract (array job) --------------------------------------------------
EXTRACT_EXPORT="ALL,$COMMON_EXPORT,QWEN_MODEL=$QWEN_MODEL,SHARDS=$SHARDS,MAX_NEW_TOKENS=$MAX_NEW_TOKENS,LIMIT=$LIMIT"
ARRAY_SPEC="0-$((SHARDS - 1))"
EXTRACT_JOB_RAW="$(sbatch --parsable --array="$ARRAY_SPEC" --export="$EXTRACT_EXPORT" "$EXTRACT_SCRIPT")"
EXTRACT_JOB_ID="${EXTRACT_JOB_RAW%%;*}"
echo "Submitted extract array job: $EXTRACT_JOB_ID (array $ARRAY_SPEC)"

# --- 2. merge (only when sharded) -------------------------------------------
VALIDATE_DEP="afterok:$EXTRACT_JOB_ID"
if [ "$SHARDS" -gt 1 ]; then
    # run_emotion_build_cache_slurm.sh writes MODE=merge output to OUT_ZH; point
    # that at the canonical OUT_EN path since Qwen2-Audio has no translate stage.
    MERGE_EXPORT="ALL,$COMMON_EXPORT,MODE=merge,MERGE_PATTERN=$MERGE_PATTERN,OUT_ZH=$OUT_EN"
    MERGE_JOB_RAW="$(sbatch --parsable --dependency=afterok:"$EXTRACT_JOB_ID" --export="$MERGE_EXPORT" "$BUILD_SCRIPT")"
    MERGE_JOB_ID="${MERGE_JOB_RAW%%;*}"
    echo "Submitted merge job: $MERGE_JOB_ID"
    VALIDATE_DEP="afterok:$MERGE_JOB_ID"
fi

# --- 3. validate -------------------------------------------------------------
if [ "$RUN_VALIDATE" = "1" ]; then
    VALIDATE_EXPORT="ALL,$COMMON_EXPORT,MODE=validate,STRICT=$STRICT"
    VALIDATE_JOB_RAW="$(sbatch --parsable --dependency="$VALIDATE_DEP" --export="$VALIDATE_EXPORT" "$BUILD_SCRIPT")"
    VALIDATE_JOB_ID="${VALIDATE_JOB_RAW%%;*}"
    echo "Submitted validate job: $VALIDATE_JOB_ID (dep $VALIDATE_DEP)"
fi

echo "Done. Final cache will be at: $OUT_EN"
