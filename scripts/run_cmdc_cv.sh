#!/usr/bin/env bash
# Fan out CMDC 5-fold CV across all three modalities in one submission, mirroring
# scripts/run_turkish_5fold.sh: each config becomes an independent chain (folds
# run sequentially via afterok; the three modality chains run concurrently).
#
# Per config it delegates to submit_cv_then_fulltrain.sh, which submits the CV
# folds (split.mode=cv, best-checkpoint eval per fold) and a dependent summary
# job that writes final_summary.{json,csv} under the config's run_root.
#
# Usage (login node — it submits sbatch jobs and returns):
#   RUN_NAME_PREFIX=cmdc_tf bash scripts/run_cmdc_cv.sh
#
# Override the config set or folds if needed:
#   CONFIGS="$PROJECT_ROOT/configs/main/cmdc_audio_only_selposf1_tf.yaml" \
#     RUN_NAME_PREFIX=cmdc_tf bash scripts/run_cmdc_cv.sh

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
CONFIGS="${CONFIGS:-$PROJECT_ROOT/configs/main/cmdc_audio_only_selposf1_tf.yaml $PROJECT_ROOT/configs/main/cmdc_text_only_selposf1_tf.yaml $PROJECT_ROOT/configs/main/cmdc_audio_text_selposf1_tf.yaml}"
FOLDS="${FOLDS:-0 1 2 3 4}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-cmdc_tf}"
CV_SUBMIT_BEST_EVAL="${CV_SUBMIT_BEST_EVAL:-1}"
CV_SUBMIT_LAST_EVAL="${CV_SUBMIT_LAST_EVAL:-0}"
CV_EXTRA_TRAIN_ARGS="${CV_EXTRA_TRAIN_ARGS:-}"
CV_EXTRA_EVAL_ARGS="${CV_EXTRA_EVAL_ARGS:-}"
CHAIN_SCRIPT="${CHAIN_SCRIPT:-$PROJECT_ROOT/scripts/submit_cv_then_fulltrain.sh}"

export PROJECT_ROOT

if [ ! -f "$CHAIN_SCRIPT" ]; then
    echo "CV chain script not found: $CHAIN_SCRIPT"
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
echo "CMDC three-modality CV submission"
echo "  configs: ${CONFIG_VALUES[*]}"
echo "  folds:   $FOLDS"
echo "  prefix:  $RUN_NAME_PREFIX"
echo "  best_eval per fold: $CV_SUBMIT_BEST_EVAL"
echo "========================================"

for CONFIG in "${CONFIG_VALUES[@]}"; do
    STEM="$(basename "$CONFIG" .yaml)"
    CV_RUN_NAME="${RUN_NAME_PREFIX}_${STEM}"
    echo "--- submitting CV chain: config=$CONFIG run_name=$CV_RUN_NAME ---"
    env \
        PROJECT_ROOT="$PROJECT_ROOT" \
        CONFIG="$CONFIG" \
        CV_RUN_NAME="$CV_RUN_NAME" \
        FOLDS="$FOLDS" \
        CV_SUBMIT_BEST_EVAL="$CV_SUBMIT_BEST_EVAL" \
        CV_SUBMIT_LAST_EVAL="$CV_SUBMIT_LAST_EVAL" \
        CV_EXTRA_TRAIN_ARGS="$CV_EXTRA_TRAIN_ARGS" \
        CV_EXTRA_EVAL_ARGS="$CV_EXTRA_EVAL_ARGS" \
        bash "$CHAIN_SCRIPT"
done

echo "========================================"
echo "Submission complete: ${#CONFIG_VALUES[@]} modality chains × $(echo $FOLDS | wc -w) folds"
echo "========================================"
