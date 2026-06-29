#!/usr/bin/env bash
# Fan out the DAIC modalities in one submission. Unlike CMDC/Turkish, DAIC uses
# the official FIXED train/val/test split (split.mode: fixed in the config), so
# there is no cross-validation to average: each modality is a single fold-0
# training run + a best-checkpoint standalone eval on the test partition. The
# metrics in that eval ARE the result (no summary/averaging step).
#
# Delegates per config to submit_train_and_eval.sh, which submits the train job
# and an afterok best-checkpoint eval job. The three modality chains are
# independent and run concurrently.
#
# Usage (login node — submits sbatch jobs and returns):
#   RUN_NAME_PREFIX=daic_tf bash scripts/run_daic_fixed.sh
#
# audio+text and text-only are already on the recipe (their table numbers stand);
# to redo only audio-only, pass a single config:
#   CONFIGS="$PROJECT_ROOT/configs/main/daic_audio_only_selmacrof1_tf.yaml" \
#     RUN_NAME_PREFIX=daic_tf bash scripts/run_daic_fixed.sh

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
CONFIGS="${CONFIGS:-$PROJECT_ROOT/configs/main/daic_audio_only_selmacrof1_tf.yaml $PROJECT_ROOT/configs/main/daic_text_only_selmacrof1_tf.yaml $PROJECT_ROOT/configs/main/daic_audio_text_selmacrof1_tf.yaml}"
FOLD="${FOLD:-0}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-daic_tf}"
SUBMIT_BEST_EVAL="${SUBMIT_BEST_EVAL:-1}"
SUBMIT_LAST_EVAL="${SUBMIT_LAST_EVAL:-0}"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"
EXTRA_EVAL_ARGS="${EXTRA_EVAL_ARGS:-}"
SUBMIT_SCRIPT="${SUBMIT_SCRIPT:-$PROJECT_ROOT/scripts/submit_train_and_eval.sh}"

export PROJECT_ROOT

if [ ! -f "$SUBMIT_SCRIPT" ]; then
    echo "Submit helper not found: $SUBMIT_SCRIPT"
    exit 1
fi

CONFIGS="${CONFIGS//,/ }"
read -r -a CONFIG_VALUES <<< "$CONFIGS"
if [ ${#CONFIG_VALUES[@]} -eq 0 ]; then
    echo "CONFIGS must be non-empty."
    exit 1
fi
for CONFIG in "${CONFIG_VALUES[@]}"; do
    [ -f "$CONFIG" ] || { echo "Config not found: $CONFIG"; exit 1; }
done

echo "========================================"
echo "DAIC fixed-split submission"
echo "  configs: ${CONFIG_VALUES[*]}"
echo "  fold:    $FOLD (fixed split)"
echo "  prefix:  $RUN_NAME_PREFIX"
echo "========================================"

for CONFIG in "${CONFIG_VALUES[@]}"; do
    STEM="$(basename "$CONFIG" .yaml)"
    RUN_NAME="${RUN_NAME_PREFIX}_${STEM}"
    echo "--- submitting: config=$CONFIG run_name=$RUN_NAME ---"
    env \
        PROJECT_ROOT="$PROJECT_ROOT" \
        CONFIG="$CONFIG" \
        FOLD="$FOLD" \
        RUN_NAME="$RUN_NAME" \
        SUBMIT_BEST_EVAL="$SUBMIT_BEST_EVAL" \
        SUBMIT_LAST_EVAL="$SUBMIT_LAST_EVAL" \
        EXTRA_TRAIN_ARGS="$EXTRA_TRAIN_ARGS" \
        EXTRA_EVAL_ARGS="$EXTRA_EVAL_ARGS" \
        bash "$SUBMIT_SCRIPT"
done

echo "========================================"
echo "Submission complete: ${#CONFIG_VALUES[@]} modality runs (fixed split, fold $FOLD)"
echo "========================================"
