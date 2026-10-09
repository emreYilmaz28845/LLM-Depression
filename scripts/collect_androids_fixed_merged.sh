#!/bin/bash
# Compact collection for the corrected Merged baseline (Worker 1).
#
# Usage: collect_androids_fixed_merged.sh <slug> <seed> <stage>
#   slug: native_text_only|native_audio_only|native_audio_text|english_text_only|english_audio_text
#   stage: cv|final
#
# Collects the run registry, the merged-side compact evidence (excluding adapter
# weights and dense feature arrays), the train-side compact evidence (excluding
# adapter weights) and the stage acceptance audit JSON into the lane worktree,
# mirroring the relative roots the coverage audit and verifier resolve.
set -euo pipefail

SLUG="${1:?usage: collect_androids_fixed_merged.sh <slug> <seed> <stage>}"
SEED="${2:?seed required}"
STAGE="${3:?stage required}"

case "$SLUG" in
  native_*) family=native ;;
  english_*) family=english ;;
  *) echo "unknown route family for $SLUG" >&2; exit 2 ;;
esac
case "$SLUG" in
  *_text_only) modality=text_only ;;
  *_audio_only) modality=audio_only ;;
  *_audio_text) modality=audio_text ;;
  *) echo "unknown modality for $SLUG" >&2; exit 2 ;;
esac

LANE="$(cd "$(dirname "$0")/.." && pwd)"
PERMANENT=/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression
RUNTIME=/gpfs/projects/etur92/ozu647717/AudioLLM/experiment_runtime/feat-qwen3-androids-official-folds-20261008
HOST=ozu647717@transfer1.bsc.es
CAMPAIGN=qwen3_androids_official_folds_20261008
RUN="qmsm_${SLUG}_s${SEED}"

MERGED_SRC="$PERMANENT/outputs/symmetric_merged/${CAMPAIGN}_${family}/${modality}/${RUN}/${STAGE}/"
TRAIN_SRC="$PERMANENT/output_model/symmetric_merged/${CAMPAIGN}_${family}_likelihood/${modality}/${RUN}/${STAGE}/"
MERGED_DST="$LANE/outputs/symmetric_merged/${CAMPAIGN}_${family}/${modality}/${RUN}/${STAGE}/"
TRAIN_DST="$LANE/output_model/symmetric_merged/${CAMPAIGN}_${family}_likelihood/${modality}/${RUN}/${STAGE}/"
REG_DST="$LANE/outputs/${CAMPAIGN}/registries/"

mkdir -p "$MERGED_DST" "$TRAIN_DST" "$REG_DST"
echo "=== collect $RUN $STAGE ==="
rsync -ah --prune-empty-dirs \
  --exclude='best_model/' --exclude='last_model/' --exclude='optuna/' --exclude='*.npz' \
  "$HOST:$MERGED_SRC" "$MERGED_DST"
rsync -ah --prune-empty-dirs \
  --exclude='best_model/' --exclude='last_model/' \
  "$HOST:$TRAIN_SRC" "$TRAIN_DST"
rsync -ah "$HOST:$RUNTIME/registries/${RUN}.json" "$REG_DST/"
if rsync -ah "$HOST:$RUNTIME/audits/${RUN}.${STAGE}.acceptance_audit.json" "$MERGED_DST/acceptance_audit.json"; then
  echo "audit collected: $MERGED_DST/acceptance_audit.json"
else
  echo "WARNING: acceptance audit missing remotely; coverage stays fail-closed" >&2
fi
echo "=== collect done $RUN $STAGE ==="
