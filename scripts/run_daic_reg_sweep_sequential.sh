#!/usr/bin/env bash
# Submit DAIC baseline/regularization sweeps with barriers between stages.
#
# The three modalities within a stage run concurrently. The next stage waits for
# every terminal job in the current stage (best/last eval when enabled,
# otherwise training), so the default schedule is:
#
#   reg3 (audio-only, text-only, audio+text) -> reg4 -> reg5 -> reg6
#
# SEEDS adds an outer seed sequence. INCLUDE_BASELINE=1 inserts a matched
# 20-epoch, no-early-stopping baseline before the regularization tiers for each
# seed. Seeds listed in BASELINE_ONLY_SEEDS run only that baseline; this is
# useful when their reg3-reg6 jobs already exist.
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
SEEDS="${SEEDS:-1337}"
INCLUDE_BASELINE="${INCLUDE_BASELINE:-1}"
BASELINE_ONLY_SEEDS="${BASELINE_ONLY_SEEDS:-}"
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
read -r -a SEED_VALUES <<< "${SEEDS//,/ }"
read -r -a BASELINE_ONLY_SEED_VALUES <<< "${BASELINE_ONLY_SEEDS//,/ }"

if [ ${#TIER_VALUES[@]} -eq 0 ] || [ ${#MODALITY_VALUES[@]} -eq 0 ] || [ ${#SEED_VALUES[@]} -eq 0 ]; then
    echo "TIERS, MODALITIES, and SEEDS must all be non-empty."
    exit 1
fi

config_for() {
    local stage_kind="$1"
    local modality="$2"
    if [ "$stage_kind" = "baseline" ]; then
        printf '%s/configs/archive/pre_harmonized_posf1_20260809/daic/daic_%s_selposf1_tf.yaml\n' \
            "$PROJECT_ROOT" "$modality"
        return 0
    fi
    printf '%s/configs/experiments/daic_%s_reg%s_selposf1_tf.yaml\n' \
        "$PROJECT_ROOT" "$modality" "$stage_kind"
}

extract_job_id() {
    local output="$1"
    local pattern="$2"
    printf '%s\n' "$output" | awk -v pattern="$pattern" '$0 ~ pattern {print $NF; exit}'
}

join_args() {
    local first="$1"
    local second="$2"
    if [ -n "$first" ] && [ -n "$second" ]; then
        printf '%s %s\n' "$first" "$second"
    elif [ -n "$first" ]; then
        printf '%s\n' "$first"
    else
        printf '%s\n' "$second"
    fi
}

is_baseline_only_seed() {
    local candidate="$1"
    local baseline_seed=""
    for baseline_seed in "${BASELINE_ONLY_SEED_VALUES[@]}"; do
        if [ "$candidate" = "$baseline_seed" ]; then
            return 0
        fi
    done
    return 1
}

# A stage is encoded as seed:kind, where kind is "baseline" or a reg tier.
STAGE_VALUES=()
for SEED in "${SEED_VALUES[@]}"; do
    if [ "$INCLUDE_BASELINE" = "1" ]; then
        STAGE_VALUES+=("$SEED:baseline")
    fi
    if ! is_baseline_only_seed "$SEED"; then
        for TIER in "${TIER_VALUES[@]}"; do
            STAGE_VALUES+=("$SEED:$TIER")
        done
    fi
done

if [ ${#STAGE_VALUES[@]} -eq 0 ]; then
    echo "The requested seed/baseline/tier combination produced no stages."
    exit 1
fi

# Validate the whole sweep before submitting its first job.
for STAGE in "${STAGE_VALUES[@]}"; do
    STAGE_KIND="${STAGE#*:}"
    for MODALITY in "${MODALITY_VALUES[@]}"; do
        CONFIG="$(config_for "$STAGE_KIND" "$MODALITY")"
        [ -f "$CONFIG" ] || { echo "Config not found: $CONFIG"; exit 1; }
    done
done

echo "========================================"
echo "DAIC tier-sequential regularization sweep"
echo "  tiers:      ${TIER_VALUES[*]}"
echo "  modalities: ${MODALITY_VALUES[*]}"
echo "  seeds:      ${SEED_VALUES[*]}"
echo "  baseline:   $INCLUDE_BASELINE"
echo "  baseline-only seeds: ${BASELINE_ONLY_SEED_VALUES[*]:-(none)}"
echo "  stages:     ${#STAGE_VALUES[@]}"
echo "  fold:       $FOLD (fixed split)"
echo "  prefix:     $RUN_NAME_PREFIX"
echo "  dry run:    $DRY_RUN"
echo "========================================"

if [ "$DRY_RUN" = "1" ]; then
    for STAGE in "${STAGE_VALUES[@]}"; do
        SEED="${STAGE%%:*}"
        STAGE_KIND="${STAGE#*:}"
        if [ "$STAGE_KIND" = "baseline" ]; then
            STAGE_LABEL="seed=$SEED baseline"
        else
            STAGE_LABEL="seed=$SEED reg$STAGE_KIND"
        fi
        echo "$STAGE_LABEL (concurrent within stage):"
        for MODALITY in "${MODALITY_VALUES[@]}"; do
            echo "  $(config_for "$STAGE_KIND" "$MODALITY")"
        done
    done
    echo "No jobs submitted."
    exit 0
fi

NEXT_STAGE_DEPENDENCY="$INITIAL_DEPENDENCY"

for STAGE in "${STAGE_VALUES[@]}"; do
    SEED="${STAGE%%:*}"
    STAGE_KIND="${STAGE#*:}"
    if [ "$STAGE_KIND" = "baseline" ]; then
        STAGE_LABEL="seed=$SEED baseline"
    else
        STAGE_LABEL="seed=$SEED reg$STAGE_KIND"
    fi
    # All modalities receive the same dependency, so they can run concurrently
    # once the preceding stage has completed.
    STAGE_DEPENDENCY="$NEXT_STAGE_DEPENDENCY"
    STAGE_TERMINAL_IDS=()
    REQUIRED_TRAIN_ARGS="--set seed=$SEED --set training.num_train_epochs=20 --set training.early_stopping.enabled=false"
    REQUIRED_EVAL_ARGS="--set seed=$SEED"
    STAGE_TRAIN_ARGS="$(join_args "$REQUIRED_TRAIN_ARGS" "$EXTRA_TRAIN_ARGS")"
    STAGE_EVAL_ARGS="$(join_args "$REQUIRED_EVAL_ARGS" "$EXTRA_EVAL_ARGS")"

    echo "--- submitting $STAGE_LABEL stage ---"
    if [ -n "$STAGE_DEPENDENCY" ]; then
        echo "  waits for: $STAGE_DEPENDENCY"
    fi

    for MODALITY in "${MODALITY_VALUES[@]}"; do
        CONFIG="$(config_for "$STAGE_KIND" "$MODALITY")"
        STEM="$(basename "$CONFIG" .yaml)"
        RUN_NAME="${RUN_NAME_PREFIX}_s${SEED}_${STEM}"

        echo "  submitting: modality=$MODALITY run_name=$RUN_NAME"
        OUTPUT="$(
            env \
                PROJECT_ROOT="$PROJECT_ROOT" \
                CONFIG="$CONFIG" \
                FOLD="$FOLD" \
                RUN_NAME="$RUN_NAME" \
                SUBMIT_BEST_EVAL="$SUBMIT_BEST_EVAL" \
                SUBMIT_LAST_EVAL="$SUBMIT_LAST_EVAL" \
                EXTRA_TRAIN_ARGS="$STAGE_TRAIN_ARGS" \
                EXTRA_EVAL_ARGS="$STAGE_EVAL_ARGS" \
                SBATCH_DEPENDENCY="$STAGE_DEPENDENCY" \
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
            STAGE_TERMINAL_IDS+=("$BEST_EVAL_JOB_ID")
            TERMINAL_FOUND=1
        fi
        if [ -n "$LAST_EVAL_JOB_ID" ]; then
            STAGE_TERMINAL_IDS+=("$LAST_EVAL_JOB_ID")
            TERMINAL_FOUND=1
        fi
        if [ "$TERMINAL_FOUND" = "0" ]; then
            STAGE_TERMINAL_IDS+=("$TRAIN_JOB_ID")
        fi
    done

    TERMINAL_JOB_LIST="$(IFS=:; printf '%s' "${STAGE_TERMINAL_IDS[*]}")"
    NEXT_STAGE_DEPENDENCY="afterok:$TERMINAL_JOB_LIST"
    echo "  $STAGE_LABEL barrier: $NEXT_STAGE_DEPENDENCY"
done

echo "========================================"
echo "Submission complete: ${#STAGE_VALUES[@]} sequential stages"
echo "Maximum concurrent training jobs: ${#MODALITY_VALUES[@]}"
echo "Final stage terminal jobs: ${NEXT_STAGE_DEPENDENCY#afterok:}"
echo "========================================"
