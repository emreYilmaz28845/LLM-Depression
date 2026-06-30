#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
CONFIG="${CONFIG:-$PROJECT_ROOT/configs/main/edaic_audio_text_selposf1_tf.yaml}"
CV_RUN_NAME="${CV_RUN_NAME:-cv_reproduction}"
FOLDS="${FOLDS:-0 1 2 3 4}"
CURRENT_STAGE_INDEX="${CURRENT_STAGE_INDEX:-0}"
CV_EXTRA_TRAIN_ARGS="${CV_EXTRA_TRAIN_ARGS:-}"
CV_EXTRA_EVAL_ARGS="${CV_EXTRA_EVAL_ARGS:-}"
CV_SUBMIT_BEST_EVAL="${CV_SUBMIT_BEST_EVAL:-1}"
CV_SUBMIT_LAST_EVAL="${CV_SUBMIT_LAST_EVAL:-0}"
CV_SEQUENTIAL="${CV_SEQUENTIAL:-1}"
SUBMIT_SCRIPT="${SUBMIT_SCRIPT:-$PROJECT_ROOT/scripts/submit_train_and_eval.sh}"
CHAIN_RUNNER_SCRIPT="${CHAIN_RUNNER_SCRIPT:-$PROJECT_ROOT/scripts/run_chain_submit_slurm.sh}"
SUMMARIZE_SCRIPT="${SUMMARIZE_SCRIPT:-$PROJECT_ROOT/src/summarize_runs.py}"

if [ ! -f "$SUBMIT_SCRIPT" ]; then
    echo "Submit helper not found: $SUBMIT_SCRIPT"
    exit 1
fi

if [ ! -f "$CHAIN_RUNNER_SCRIPT" ]; then
    echo "Chain runner script not found: $CHAIN_RUNNER_SCRIPT"
    exit 1
fi

if [ ! -f "$SUMMARIZE_SCRIPT" ]; then
    echo "Summarize script not found: $SUMMARIZE_SCRIPT"
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

resolve_run_root() {
    local config_path="$1"
    local run_name="$2"
    local run_root_rel=""
    run_root_rel="$(awk '
      /^output_dirs:/ {in_block=1; next}
      in_block && /^[^[:space:]]/ {in_block=0}
      in_block && /^[[:space:]]+run_root:/ {
        sub(/^[[:space:]]+run_root:[[:space:]]*/, "", $0)
        print $0
        exit
      }
    ' "$config_path" | tr -d '"' | tr -d "'")"
    if [ -z "$run_root_rel" ]; then
        echo "Could not parse output_dirs.run_root from config: $config_path" >&2
        exit 1
    fi
    run_root_rel="${run_root_rel//\$\{PROJECT_ROOT\}/$PROJECT_ROOT}"
    printf '%s/%s\n' "$run_root_rel" "$run_name"
}

submit_next_stage() {
    local dependency="$1"
    local next_stage_index="$2"
    local export_args=""
    local next_job_raw=""
    local next_job_id=""

    export_args="ALL,PROJECT_ROOT=$PROJECT_ROOT,CONFIG=$CONFIG,CV_RUN_NAME=$CV_RUN_NAME,FOLDS=$FOLDS,CURRENT_STAGE_INDEX=$next_stage_index,CV_EXTRA_TRAIN_ARGS=$CV_EXTRA_TRAIN_ARGS,CV_EXTRA_EVAL_ARGS=$CV_EXTRA_EVAL_ARGS,CV_SUBMIT_BEST_EVAL=$CV_SUBMIT_BEST_EVAL,CV_SUBMIT_LAST_EVAL=$CV_SUBMIT_LAST_EVAL,CV_SEQUENTIAL=$CV_SEQUENTIAL,SUBMIT_SCRIPT=$SUBMIT_SCRIPT,CHAIN_RUNNER_SCRIPT=$CHAIN_RUNNER_SCRIPT,SUMMARIZE_SCRIPT=$SUMMARIZE_SCRIPT,CHAIN_SCRIPT=$PROJECT_ROOT/scripts/submit_cv_then_fulltrain.sh"

    next_job_raw="$(sbatch --parsable --dependency="$dependency" --export="$export_args" "$CHAIN_RUNNER_SCRIPT")"
    next_job_id="${next_job_raw%%;*}"
    echo "Submitted chain continuation job: $next_job_id"
    echo "  next_stage_index: $next_stage_index"
    echo "  dependency: $dependency"
}

echo "Submitting CV stage"
echo "  config: $CONFIG"
echo "  cv_run_name: $CV_RUN_NAME"
echo "  folds: $FOLDS"
echo "  cv_sequential: $CV_SEQUENTIAL"
echo "  current_stage_index: $CURRENT_STAGE_INDEX"
echo "  cv_submit_best_eval: $CV_SUBMIT_BEST_EVAL"
echo "  cv_submit_last_eval: $CV_SUBMIT_LAST_EVAL"

read -r -a FOLD_ARRAY <<< "$FOLDS"
FOLD_COUNT="${#FOLD_ARRAY[@]}"
CV_RUN_ROOT="$(resolve_run_root "$CONFIG" "$CV_RUN_NAME")"

if [ "$CV_SEQUENTIAL" = "1" ]; then
    if [ "$CURRENT_STAGE_INDEX" -lt "$FOLD_COUNT" ]; then
        FOLD="${FOLD_ARRAY[$CURRENT_STAGE_INDEX]}"
        echo "Submitting CV fold $FOLD ($((CURRENT_STAGE_INDEX + 1))/$FOLD_COUNT)"

        CV_TRAIN_ARGS="$(join_args "--set split.mode=cv" "$CV_EXTRA_TRAIN_ARGS")"
        CV_EVAL_ARGS="$(join_args "--set split.mode=cv" "$CV_EXTRA_EVAL_ARGS")"
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

        submit_next_stage "afterok:$(IFS=:; printf '%s' "${FOLD_TERMINAL_IDS[*]}")" "$((CURRENT_STAGE_INDEX + 1))"
        exit 0
    fi

    if [ "$CURRENT_STAGE_INDEX" -eq "$FOLD_COUNT" ]; then
        SUMMARY_LOG="$CV_RUN_ROOT/cv_summary-$(date +%Y-%m-%d_%H:%M:%S).log"
        mkdir -p "$CV_RUN_ROOT"
        {
            echo "Summarizing CV results"
            echo "  run_root: $CV_RUN_ROOT"
            python "$SUMMARIZE_SCRIPT" --run_root "$CV_RUN_ROOT"
            echo "Wrote CV summary to:"
            echo "  $CV_RUN_ROOT/final_summary.json"
            echo "  $CV_RUN_ROOT/final_summary.csv"
            echo "  $CV_RUN_ROOT/final_summary_active.csv"
            if [ -f "$CV_RUN_ROOT/final_summary_active.csv" ]; then
                echo
                echo "Active-backend CV summary:"
                cat "$CV_RUN_ROOT/final_summary_active.csv"
            fi
            echo
            echo "Summary log: $SUMMARY_LOG"
        } 2>&1 | tee "$SUMMARY_LOG"
        exit 0
    fi

    echo "Current stage index $CURRENT_STAGE_INDEX is out of range for folds '$FOLDS'."
    exit 1
fi

TERMINAL_JOB_IDS=()
PREVIOUS_CV_DEPENDENCY=""
echo "Submitting all CV folds in parallel/dependency mode"
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
echo "Summarizing CV results after all folds finish"
echo "  dependency: $DEPENDENCY"
submit_next_stage "$DEPENDENCY" "$FOLD_COUNT"
