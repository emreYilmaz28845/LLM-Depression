#!/bin/bash
# Submission driver for the Worker 1 corrected Merged baseline
# (campaign qwen3_androids_official_folds_20261008).
#
# Run this on the MN5 scheduler login (never a transfer node) from anywhere;
# it calls the specialized merged submitter from the immutable deployment code
# with one common resolved override set per route/seed.
#
# Usage:
#   DEPLOYMENT_CODE=/gpfs/.../deployments/<id>/code \
#   SOURCE_COMMIT=<full-sha> \
#   STAGE=cv|final|smoke [ONLY=slug] [SEEDS="7 1337 2024"] [MODE=--dry-run] \
#   bash scripts/submit_androids_fixed_merged.sh
#
# Environment:
#   DEPLOYMENT_CODE (required) immutable deployment code directory
#   SOURCE_COMMIT   (required) full source commit recorded in the registry
#   STAGE           cv, final or smoke (default cv)
#   ONLY            restrict to one route slug (e.g. native_audio_text)
#   SEEDS           space-separated training seeds (default "7 1337 2024")
#   MODE            --dry-run for planning; empty for real submission
set -euo pipefail

STAGE="${STAGE:-cv}"
DEPLOYMENT_CODE="${DEPLOYMENT_CODE:?DEPLOYMENT_CODE is required}"
SOURCE_COMMIT="${SOURCE_COMMIT:?SOURCE_COMMIT is required}"
ONLY="${ONLY:-}"
SEEDS="${SEEDS:-7 1337 2024}"
MODE="${MODE:-}"

case "$STAGE" in
  cv|final|smoke) ;;
  *) echo "unsupported STAGE: $STAGE" >&2; exit 2 ;;
esac

PERMANENT=/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression
POOLED_RUNTIME=/gpfs/projects/etur92/ozu647717/AudioLLM/experiment_runtime/feat-qwen3-turkish-pooled-defaults-20260929
RUNTIME=/gpfs/projects/etur92/ozu647717/AudioLLM/experiment_runtime/feat-qwen3-androids-official-folds-20261008
REGISTRY_DIR="$RUNTIME/registries"
LOG_ROOT="$RUNTIME/logs/symmetric_merged"
CAMPAIGN_BASE=qwen3_androids_official_folds_20261008

export SYMMETRIC_MERGED_SOURCE_COMMIT="$SOURCE_COMMIT"
export QWEN_HIDDEN_DEPS="$PERMANENT/.deps/qwen_hidden"
mkdir -p "$REGISTRY_DIR" "$LOG_ROOT"

cd "$DEPLOYMENT_CODE"

# slug | route campaign | modality
ROUTES=(
  "native_text_only|${CAMPAIGN_BASE}_native|text_only"
  "native_audio_only|${CAMPAIGN_BASE}_native|audio_only"
  "native_audio_text|${CAMPAIGN_BASE}_native|audio_text"
  "english_text_only|${CAMPAIGN_BASE}_english|text_only"
  "english_audio_text|${CAMPAIGN_BASE}_english|audio_text"
)

SMOKE_ARGS=()
if [ "$STAGE" = "smoke" ]; then
  SMOKE_ARGS=(--smoke-subjects 2 --smoke-epochs 1 --smoke-trials 0)
fi

for spec in "${ROUTES[@]}"; do
  IFS='|' read -r slug campaign modality <<< "$spec"
  if [ -n "$ONLY" ] && [ "$ONLY" != "$slug" ]; then
    continue
  fi
  config="configs/experiments/merged/symmetric_merged_qwen3_pooled_${slug}.yaml"
  for seed in $SEEDS; do
    run_id="qmsm_${slug}_s${seed}"
    registry="$REGISTRY_DIR/${run_id}.json"
    echo "=== ${STAGE} ${slug} seed=${seed} run_id=${run_id} mode=${MODE:-execute} ==="
    python scripts/submit_symmetric_merged.py \
      --stage "$STAGE" \
      --config "$config" \
      --run-id "$run_id" \
      --registry "$registry" \
      --set "seed=${seed}" \
      --set "protocol_settings.split_seed=1337" \
      --set "heads.fixed_seed=1337" \
      --set "output_dirs.merged_root=$PERMANENT/outputs/symmetric_merged/$campaign/$modality" \
      --set "output_dirs.run_root=$PERMANENT/output_model/symmetric_merged/${campaign}_likelihood/$modality" \
      --input-root "$PERMANENT" \
      --pooled-runtime-root "$POOLED_RUNTIME" \
      --log-root "$LOG_ROOT" \
      "${SMOKE_ARGS[@]}" \
      ${MODE:+"$MODE"}
  done
done
