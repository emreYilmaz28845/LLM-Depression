#!/usr/bin/env bash
# Submit one fold of one arm of the Turkish four-source versus pooled comparison.
#
#   ARM=baseline|treatment CELL=text_only|audio_only|audio_text FOLD=<n> \
#   RUN_NAME=<unique-name> [MODE=dry-run|execute] [EXTRA_SETS="k=v k=v"] \
#   bash scripts/submit_turkish_geriatri_arm.sh
#
# The two arms must never share a manifest or split directory: the baseline arm
# consumes the prebuilt pooled manifest and split in the lane runtime, while the
# four-source arm builds its own manifest into a separate runtime subdirectory.
# Creating that separation here is what keeps the baseline from silently reading
# the 222-participant manifest.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
EXPERIMENT_ID="${EXPERIMENT_ID:-exp-turkish-geriatri-20260925}"
RUNTIME_ROOT="/gpfs/projects/etur92/ozu647717/AudioLLM/experiment_runtime/${EXPERIMENT_ID}"

ARM="${ARM:?set ARM=baseline or treatment}"
CELL="${CELL:?set CELL=text_only, audio_only, or audio_text}"
FOLD="${FOLD:?set FOLD=<fold index>}"
RUN_NAME="${RUN_NAME:?set RUN_NAME=<unique run name>}"
MODE="${MODE:-dry-run}"
EXTRA_SETS="${EXTRA_SETS:-}"

case "$CELL" in
  text_only)
    BASELINE_CONFIG="configs/main/turkish_pooled_t17_text_only_harmonized_selmacrof1_likelihood_v1_promptcontext_v1_qwen38_27b.yaml"
    TREATMENT_CONFIG="configs/main/turkish_all_geriatri_t17_text_only_harmonized_selmacrof1_likelihood_v1_promptcontext_v1_qwen38_27b.yaml"
    CAMPAIGN="promptcontext_v1_qwen38_likelihood"
    MODALITY="text_only"
    TRAIN_NODES=1
    ;;
  audio_only)
    BASELINE_CONFIG="configs/main/turkish_pooled_t17_audio_only_harmonized_selmacrof1_likelihood_v1_qwen3asr_promptcontext_v1_qwen3omni_30b_a3b.yaml"
    TREATMENT_CONFIG="configs/main/turkish_all_geriatri_t17_audio_only_harmonized_selmacrof1_likelihood_v1_qwen3asr_promptcontext_v1_qwen3omni_30b_a3b.yaml"
    CAMPAIGN="promptcontext_v1_qwen3omni_likelihood"
    MODALITY="audio_only"
    TRAIN_NODES=2
    ;;
  audio_text)
    BASELINE_CONFIG="configs/main/turkish_pooled_t17_audio_text_harmonized_selmacrof1_likelihood_v1_qwen3asr_promptcontext_v1_qwen3omni_30b_a3b.yaml"
    TREATMENT_CONFIG="configs/main/turkish_all_geriatri_t17_audio_text_harmonized_selmacrof1_likelihood_v1_qwen3asr_promptcontext_v1_qwen3omni_30b_a3b.yaml"
    CAMPAIGN="promptcontext_v1_qwen3omni_likelihood"
    MODALITY="audio_text"
    TRAIN_NODES=2
    ;;
  *)
    echo "CELL must be text_only, audio_only, or audio_text" >&2
    exit 2
    ;;
esac

ARGS=(
  submit exp-turkish-geriatri
  --fold "$FOLD"
  --seed 1337
  --run-name "$RUN_NAME"
  --campaign "$CAMPAIGN"
  --modality "$MODALITY"
  --dataset turkish
  --train-nodes "$TRAIN_NODES"
  --train-gpus-per-node 4
)

case "$ARM" in
  baseline)
    ARGS+=(--config "$BASELINE_CONFIG" --manifest-policy prebuilt)
    if [ "$TRAIN_NODES" -eq 2 ]; then
      # The pooled YAML predates the submitted shape: the recorded baseline runs
      # used world size 8 with accumulation 16.
      ARGS+=(--set training.gradient_accumulation_steps=16)
    fi
    ;;
  treatment)
    ARGS+=(--config "$TREATMENT_CONFIG" --manifest-policy build)
    ARGS+=(--set "output_dirs.manifest_dir=$RUNTIME_ROOT/manifests/turkish_all_geriatri")
    ARGS+=(--set "output_dirs.split_dir=$RUNTIME_ROOT/splits/turkish_all_geriatri")
    ;;
  *)
    echo "ARM must be baseline or treatment" >&2
    exit 2
    ;;
esac

for extra in $EXTRA_SETS; do
  ARGS+=(--set "$extra")
done

cd "$PROJECT_ROOT"
echo "arm=$ARM cell=$CELL fold=$FOLD run_name=$RUN_NAME mode=$MODE nodes=$TRAIN_NODES extra_sets='${EXTRA_SETS}'"

case "$MODE" in
  dry-run)
    exec python tools/exp.py "${ARGS[@]}" --dry-run
    ;;
  execute)
    exec python tools/exp.py "${ARGS[@]}" --execute
    ;;
  *)
    echo "MODE must be dry-run or execute" >&2
    exit 2
    ;;
esac
