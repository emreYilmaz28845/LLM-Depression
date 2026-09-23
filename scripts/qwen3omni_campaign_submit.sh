#!/bin/bash
# Submit the Qwen3-Omni standalone prompt-context campaign through the managed
# lane entrypoint (tools/exp.py submit). Run from the lane worktree.
#
#   CELL=<cell id>          (required; see the table below)
#   RUN_SUFFIX=<tag>        (required; run names are qwen3omni_<cell>_f<fold>_<tag>)
#   FOLDS="0 1 2 3 4"       (default; space separated)
#   MODE=dry-run|execute    (default dry-run)
#   SMOKE=1                 (one epoch and split.smoke_subject_limit=6)
#   SUPERSEDES=<attempt-id> (only for a bounded transient-infrastructure retry)
#   SLUG                    (default: the campaign lane)
#   ENV_ACTIVATE            (default: the offline qwen3omni venv)
#   EXTRA_SETS="key=value .." (optional additional --set overrides)
#
# Every cell keeps the campaign contract: seed 1337, the production FSDP shape
# (2 nodes x 4 H100, world size 8, per-rank batch 1, accumulation 16 passed
# explicitly so it lands in provenance, effective batch 128), the 4-GPU
# device-map standalone evaluation declared by the config, and the qwen3omni
# runtime environment. The pooled cells submit with manifest_policy=prebuilt, so
# their manifest and fold files must already be in the attempt runtime paths.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
cd "$PROJECT_ROOT"

CELL="${CELL:?Set CELL to one of the eight campaign cells}"
RUN_SUFFIX="${RUN_SUFFIX:?Set RUN_SUFFIX (for example prod_20260923 or smoke_20260923)}"
FOLDS="${FOLDS:-0 1 2 3 4}"
MODE="${MODE:-dry-run}"
SMOKE="${SMOKE:-0}"
SUPERSEDES="${SUPERSEDES:-}"
SLUG="${SLUG:-feat-qwen3omni-standalone-promptcontext-20260923}"
ENV_ACTIVATE="${ENV_ACTIVATE:-/gpfs/projects/etur92/ozu647717/venvs/qwen3omni/bin/activate}"
EXTRA_SETS="${EXTRA_SETS:-}"
CAMPAIGN="promptcontext_v1_qwen3omni_likelihood"

case "$CELL" in
    d3tec_audio_only)
        CONFIG="configs/main/d3tec_audio_only_harmonized_selmacrof1_likelihood_v1.yaml"
        DATASET="d3tec"; MODALITY="audio_only"; POOLED=0 ;;
    d3tec_audio_text)
        CONFIG="configs/main/d3tec_audio_text_harmonized_selmacrof1_likelihood_v1.yaml"
        DATASET="d3tec"; MODALITY="audio_text"; POOLED=0 ;;
    androids_audio_only)
        CONFIG="configs/main/androids_audio_only_harmonized_selmacrof1_likelihood_v1.yaml"
        DATASET="androids_interview"; MODALITY="audio_only"; POOLED=0 ;;
    androids_audio_text)
        CONFIG="configs/main/androids_audio_text_harmonized_selmacrof1_likelihood_v1.yaml"
        DATASET="androids_interview"; MODALITY="audio_text"; POOLED=0 ;;
    cmdc_audio_only)
        CONFIG="configs/main/cmdc_audio_only_harmonized_selmacrof1_likelihood_v1.yaml"
        DATASET="cmdc"; MODALITY="audio_only"; POOLED=0 ;;
    cmdc_audio_text)
        CONFIG="configs/main/cmdc_audio_text_harmonized_selmacrof1_likelihood_v1.yaml"
        DATASET="cmdc"; MODALITY="audio_text"; POOLED=0 ;;
    turkish_pooled_audio_only)
        CONFIG="configs/main/turkish_pooled_t17_audio_only_harmonized_selmacrof1_likelihood_v1_qwen3asr_promptcontext_v1_qwen3omni_30b_a3b.yaml"
        DATASET="turkish"; MODALITY="audio_only"; POOLED=1 ;;
    turkish_pooled_audio_text)
        CONFIG="configs/main/turkish_pooled_t17_audio_text_harmonized_selmacrof1_likelihood_v1_qwen3asr_promptcontext_v1_qwen3omni_30b_a3b.yaml"
        DATASET="turkish"; MODALITY="audio_text"; POOLED=1 ;;
    *)
        echo "unknown CELL: $CELL" >&2
        exit 2 ;;
esac

case "$MODE" in
    dry-run|execute) ;;
    *) echo "MODE must be dry-run or execute" >&2; exit 2 ;;
esac

ARGS=(
    submit "$SLUG"
    --config "$CONFIG"
    --campaign "$CAMPAIGN"
    --modality "$MODALITY"
    --dataset "$DATASET"
    --seed 1337
    --train-nodes 2
    --train-gpus-per-node 4
    --env-activate "$ENV_ACTIVATE"
    --set "training.gradient_accumulation_steps=16"
)
if [ "$POOLED" = "1" ]; then
    ARGS+=(--manifest-policy prebuilt)
fi
if [ "$SMOKE" = "1" ]; then
    ARGS+=(--set "split.smoke_subject_limit=6" --set "training.num_train_epochs=1")
fi
if [ -n "$SUPERSEDES" ]; then
    ARGS+=(--supersedes-attempt-id "$SUPERSEDES")
fi
for token in $EXTRA_SETS; do
    ARGS+=(--set "$token")
done

for fold in $FOLDS; do
    RUN_NAME="qwen3omni_${CELL}_f${fold}_${RUN_SUFFIX}"
    echo "=== cell=$CELL fold=$fold mode=$MODE run=$RUN_NAME smoke=$SMOKE ==="
    python tools/exp.py "${ARGS[@]}" --fold "$fold" --run-name "$RUN_NAME" "--$MODE"
done
