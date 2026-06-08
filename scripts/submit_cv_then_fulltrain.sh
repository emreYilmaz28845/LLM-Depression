#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
CONFIG="${CONFIG:-$PROJECT_ROOT/configs/edaic_audio_text.yaml}"
CV_RUN_NAME="${CV_RUN_NAME:-cv_reproduction}"
FINAL_RUN_NAME="${FINAL_RUN_NAME:-fulltrain_reproduction}"
FOLDS="${FOLDS:-0 1 2 3 4}"
FINAL_FOLD="${FINAL_FOLD:-0}"
CV_EXTRA_TRAIN_ARGS="${CV_EXTRA_TRAIN_ARGS:-}"
CV_EXTRA_EVAL_ARGS="${CV_EXTRA_EVAL_ARGS:-}"
FINAL_EXTRA_TRAIN_ARGS="${FINAL_EXTRA_TRAIN_ARGS:-}"
FINAL_EXTRA_EVAL_ARGS="${FINAL_EXTRA_EVAL_ARGS:-}"
CV_SUBMIT_BEST_EVAL="${CV_SUBMIT_BEST_EVAL:-1}"
CV_SUBMIT_LAST_EVAL="${CV_SUBMIT_LAST_EVAL:-1}"
FINAL_SUBMIT_BEST_EVAL="${FINAL_SUBMIT_BEST_EVAL:-0}"
FINAL_SUBMIT_LAST_EVAL="${FINAL_SUBMIT_LAST_EVAL:-1}"
CV_SEQUENTIAL="${CV_SEQUENTIAL:-1}"
SUBMIT_SCRIPT="${SUBMIT_SCRIPT:-$PROJECT_ROOT/scripts/submit_train_and_eval.sh}"

if [ ! -f "$SUBMIT_SCRIPT" ]; then
    echo "Submit helper not found: $SUBMIT_SCRIPT"
    exit 1
fi

join_args() {
    local first="$1"
    local second="$2"
    if [ -n "$first" ] && [ -n "$second" ]; then
        printf '%s %s\n' "$first" "$second"
        return 0
    fi
    if [ -n "$first" ]; then
        printf '%s\n' "$first"
        return 0
    fi
    printf '%s\n' "$second"
}

extract_job_id() {
    local output="$1"
    local pattern="$2"
    printf '%s\n' "$output" | awk -v pattern="$pattern" '$0 ~ pattern {print $NF; exit}'
}

TERMINAL_JOB_IDS=()
PREVIOUS_CV_DEPENDENCY=""

echo "Submitting CV folds"
echo "  config: $CONFIG"
echo "  cv_run_name: $CV_RUN_NAME"
echo "  folds: $FOLDS"
echo "  cv_sequential: $CV_SEQUENTIAL"
echo "  cv_submit_best_eval: $CV_SUBMIT_BEST_EVAL"
echo "  cv_submit_last_eval: $CV_SUBMIT_LAST_EVAL"
echo "  final_run_name: $FINAL_RUN_NAME"
echo "  final_fold: $FINAL_FOLD"
echo "  final_submit_best_eval: $FINAL_SUBMIT_BEST_EVAL"
echo "  final_submit_last_eval: $FINAL_SUBMIT_LAST_EVAL"

for FOLD in $FOLDS; do
    CV_TRAIN_ARGS="$(join_args "--set split.mode=cv" "$CV_EXTRA_TRAIN_ARGS")"
    CV_EVAL_ARGS="$(join_args "--set split.mode=cv" "$CV_EXTRA_EVAL_ARGS")"
    CV_SCHED_DEPENDENCY=""
    if [ "$CV_SEQUENTIAL" = "1" ] && [ -n "$PREVIOUS_CV_DEPENDENCY" ]; then
        CV_SCHED_DEPENDENCY="$PREVIOUS_CV_DEPENDENCY"
    fi
    OUTPUT="$(
        env \
            PROJECT_ROOT="$PROJECT_ROOT" \
            CONFIG="$CONFIG" \
            FOLD="$FOLD" \
            RUN_NAME="$CV_RUN_NAME" \
            SUBMIT_BEST_EVAL="$CV_SUBMIT_BEST_EVAL" \
            SUBMIT_LAST_EVAL="$CV_SUBMIT_LAST_EVAL" \
            EXTRA_TRAIN_ARGS="$CV_TRAIN_ARGS" \
            EXTRA_EVAL_ARGS="$CV_EVAL_ARGS" \
            SBATCH_DEPENDENCY="$CV_SCHED_DEPENDENCY" \
            bash "$SUBMIT_SCRIPT"
    )"
    printf '%s\n' "$OUTPUT"

    TRAIN_JOB_ID="$(extract_job_id "$OUTPUT" "Submitted training job:")"
    BEST_EVAL_JOB_ID="$(extract_job_id "$OUTPUT" "Submitted best-checkpoint eval job:")"
    LAST_EVAL_JOB_ID="$(extract_job_id "$OUTPUT" "Submitted last-checkpoint eval job:")"

    FOLD_TERMINAL_IDS=()
    if [ -n "$BEST_EVAL_JOB_ID" ]; then
        FOLD_TERMINAL_IDS+=("$BEST_EVAL_JOB_ID")
    fi
    if [ -n "$LAST_EVAL_JOB_ID" ]; then
        FOLD_TERMINAL_IDS+=("$LAST_EVAL_JOB_ID")
    fi
    if [ ${#FOLD_TERMINAL_IDS[@]} -eq 0 ]; then
        if [ -z "$TRAIN_JOB_ID" ]; then
            echo "Could not parse the training job id for fold $FOLD."
            exit 1
        fi
        FOLD_TERMINAL_IDS+=("$TRAIN_JOB_ID")
    fi

    TERMINAL_JOB_IDS+=("${FOLD_TERMINAL_IDS[@]}")
    PREVIOUS_CV_DEPENDENCY="afterok:$(IFS=:; printf '%s' "${FOLD_TERMINAL_IDS[*]}")"
done

if [ ${#TERMINAL_JOB_IDS[@]} -eq 0 ]; then
    echo "No CV terminal job ids were collected."
    exit 1
fi

DEPENDENCY="afterok:$(IFS=:; printf '%s' "${TERMINAL_JOB_IDS[*]}")"
if [ "$CV_SEQUENTIAL" = "1" ] && [ -n "$PREVIOUS_CV_DEPENDENCY" ]; then
    DEPENDENCY="$PREVIOUS_CV_DEPENDENCY"
fi
FINAL_TRAIN_ARGS="$(join_args "--set split.mode=full_train" "$FINAL_EXTRA_TRAIN_ARGS")"
FINAL_EVAL_ARGS="$(join_args "--set split.mode=full_train" "$FINAL_EXTRA_EVAL_ARGS")"

echo "Submitting final full-train workflow"
echo "  dependency: $DEPENDENCY"

FINAL_OUTPUT="$(
    env \
        PROJECT_ROOT="$PROJECT_ROOT" \
        CONFIG="$CONFIG" \
        FOLD="$FINAL_FOLD" \
        RUN_NAME="$FINAL_RUN_NAME" \
        SUBMIT_BEST_EVAL="$FINAL_SUBMIT_BEST_EVAL" \
        SUBMIT_LAST_EVAL="$FINAL_SUBMIT_LAST_EVAL" \
        EXTRA_TRAIN_ARGS="$FINAL_TRAIN_ARGS" \
        EXTRA_EVAL_ARGS="$FINAL_EVAL_ARGS" \
        SBATCH_DEPENDENCY="$DEPENDENCY" \
        bash "$SUBMIT_SCRIPT"
)"
printf '%s\n' "$FINAL_OUTPUT"
