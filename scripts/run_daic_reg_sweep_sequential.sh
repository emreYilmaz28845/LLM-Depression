#!/usr/bin/env bash
# Submit the DAIC reg3-reg6 sweep with a barrier between regularization tiers.
#
# The three modalities within a tier run concurrently. The next tier waits for
# every terminal job in the current tier (best/last eval when enabled, otherwise
# training), so the default schedule is:
#
#   reg3 (audio-only, text-only, audio+text) -> reg4 -> reg5 -> reg6
#
# With four GPUs per training job, this limits the sweep to three concurrent
# training jobs / 12 GPUs instead of twelve concurrent jobs / 48 GPUs.
#
# Usage (on the cluster login node):
#   bash scripts/run_daic_reg_sweep_sequential.sh
#
# Inspect the planned configs without submitting anything:
#   DRY_RUN=1 bash scripts/run_daic_reg_sweep_sequential.sh

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
TIERS="${TIERS:-3 4 5 6}"
MODALITIES="${MODALITIES:-audio_only text_only audio_text}"
FOLD="${FOLD:-0}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-daic_regseq}"
SUBMIT_BEST_EVAL="${SUBMIT_BEST_EVAL:-1}"
SUBMIT_LAST_EVAL="${SUBMIT_LAST_EVAL:-0}"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"
EXTRA_EVAL_ARGS="${EXTRA_EVAL_ARGS:-}"
INITIAL_DEPENDENCY="${INITIAL_DEPENDENCY:-}"
DRY_RUN="${DRY_RUN:-0}"
SUBMIT_SCRIPT="${SUBMIT_SCRIPT:-$PROJECT_ROOT/scripts/submit_train_and_eval.sh}"

export PROJECT_ROOT

if [ ! -f "$SUBMIT_SCRIPT" ]; then
    echo "Submit helper not found: $SUBMIT_SCRIPT"
    exit 1
fi

read -r -a TIER_VALUES <<< "${TIERS//,/ }"
read -r -a MODALITY_VALUES <<< "${MODALITIES//,/ }"

if [ ${#TIER_VALUES[@]} -eq 0 ] || [ ${#MODALITY_VALUES[@]} -eq 0 ]; then
    echo "TIERS and MODALITIES must both be non-empty."
    exit 1
fi

config_for() {
    local tier="$1"
    local modality="$2"
    printf '%s/configs/experiments/daic_%s_reg%s_selposf1_tf.yaml\n' \
        "$PROJECT_ROOT" "$modality" "$tier"
}

extract_job_id() {
    local output="$1"
    local pattern="$2"
    printf '%s\n' "$output" | awk -v pattern="$pattern" '$0 ~ pattern {print $NF; exit}'
}

# Validate the whole sweep before submitting its first job.
for TIER in "${TIER_VALUES[@]}"; do
    for MODALITY in "${MODALITY_VALUES[@]}"; do
        CONFIG="$(config_for "$TIER" "$MODALITY")"
        [ -f "$CONFIG" ] || { echo "Config not found: $CONFIG"; exit 1; }
    done
done

echo "========================================"
echo "DAIC tier-sequential regularization sweep"
echo "  tiers:      ${TIER_VALUES[*]}"
echo "  modalities: ${MODALITY_VALUES[*]}"
echo "  fold:       $FOLD (fixed split)"
echo "  prefix:     $RUN_NAME_PREFIX"
echo "  dry run:    $DRY_RUN"
echo "========================================"

if [ "$DRY_RUN" = "1" ]; then
    for TIER in "${TIER_VALUES[@]}"; do
        echo "reg$TIER (concurrent within tier):"
        for MODALITY in "${MODALITY_VALUES[@]}"; do
            echo "  $(config_for "$TIER" "$MODALITY")"
        done
    done
    echo "No jobs submitted."
    exit 0
fi

NEXT_TIER_DEPENDENCY="$INITIAL_DEPENDENCY"

for TIER in "${TIER_VALUES[@]}"; do
    # All modalities in this tier receive the same dependency, so they can run
    # concurrently once the preceding tier has completed.
    TIER_DEPENDENCY="$NEXT_TIER_DEPENDENCY"
    TIER_TERMINAL_IDS=()

    echo "--- submitting reg$TIER tier ---"
    if [ -n "$TIER_DEPENDENCY" ]; then
        echo "  waits for: $TIER_DEPENDENCY"
    fi

    for MODALITY in "${MODALITY_VALUES[@]}"; do
        CONFIG="$(config_for "$TIER" "$MODALITY")"
        STEM="$(basename "$CONFIG" .yaml)"
        RUN_NAME="${RUN_NAME_PREFIX}_${STEM}"

        echo "  submitting: modality=$MODALITY run_name=$RUN_NAME"
        OUTPUT="$(
            env \
                PROJECT_ROOT="$PROJECT_ROOT" \
                CONFIG="$CONFIG" \
                FOLD="$FOLD" \
                RUN_NAME="$RUN_NAME" \
                SUBMIT_BEST_EVAL="$SUBMIT_BEST_EVAL" \
                SUBMIT_LAST_EVAL="$SUBMIT_LAST_EVAL" \
                EXTRA_TRAIN_ARGS="$EXTRA_TRAIN_ARGS" \
                EXTRA_EVAL_ARGS="$EXTRA_EVAL_ARGS" \
                SBATCH_DEPENDENCY="$TIER_DEPENDENCY" \
                bash "$SUBMIT_SCRIPT"
        )"
        printf '%s\n' "$OUTPUT"

        TRAIN_JOB_ID="$(extract_job_id "$OUTPUT" "Submitted training job:")"
        BEST_EVAL_JOB_ID="$(extract_job_id "$OUTPUT" "Submitted best-checkpoint eval job:")"
        LAST_EVAL_JOB_ID="$(extract_job_id "$OUTPUT" "Submitted last-checkpoint eval job:")"

        if [ -z "$TRAIN_JOB_ID" ]; then
            echo "Could not parse the training job id for: $CONFIG"
            exit 1
        fi

        TERMINAL_FOUND=0
        if [ -n "$BEST_EVAL_JOB_ID" ]; then
            TIER_TERMINAL_IDS+=("$BEST_EVAL_JOB_ID")
            TERMINAL_FOUND=1
        fi
        if [ -n "$LAST_EVAL_JOB_ID" ]; then
            TIER_TERMINAL_IDS+=("$LAST_EVAL_JOB_ID")
            TERMINAL_FOUND=1
        fi
        if [ "$TERMINAL_FOUND" = "0" ]; then
            TIER_TERMINAL_IDS+=("$TRAIN_JOB_ID")
        fi
    done

    TERMINAL_JOB_LIST="$(IFS=:; printf '%s' "${TIER_TERMINAL_IDS[*]}")"
    NEXT_TIER_DEPENDENCY="afterok:$TERMINAL_JOB_LIST"
    echo "  reg$TIER barrier: $NEXT_TIER_DEPENDENCY"
done

echo "========================================"
echo "Submission complete: ${#TIER_VALUES[@]} sequential tiers"
echo "Maximum concurrent training jobs: ${#MODALITY_VALUES[@]}"
echo "Final tier terminal jobs: ${NEXT_TIER_DEPENDENCY#afterok:}"
echo "========================================"
