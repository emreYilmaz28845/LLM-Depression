#!/usr/bin/env bash
# Submit real joint DAIC+CMDC training and chained final eval jobs.
#
# For each joint primary config this submits:
#   1. one DAIC-primary joint training job;
#   2. DAIC test eval(s) on the selected best checkpoint;
#   3. CMDC fold-k holdout eval(s) on the same selected best checkpoint.
#
# JOINT_CMDC_FOLD controls the CMDC fold used for both joint training and CMDC
# holdout evaluation. FOLD remains the DAIC fixed split fold and should normally
# stay 0.
#
# Usage:
#   JOINT_CMDC_FOLD=0 RUN_NAME_PREFIX=joint_cmdc bash scripts/run_joint_daic_cmdc.sh
#
# Run both subject-level and segment-level evals:
#   FINAL_EVAL_LEVELS="subject segment" bash scripts/run_joint_daic_cmdc.sh
#
# Limit to one modality:
#   JOINT_PAIRS="$PROJECT_ROOT/configs/experiments/joint/daic_cmdc_audio_text_joint_tf.yaml::$PROJECT_ROOT/configs/experiments/joint/cmdc_component_audio_text.yaml" \
#     bash scripts/run_joint_daic_cmdc.sh

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
export PROJECT_ROOT

FOLD="${FOLD:-0}"
JOINT_CMDC_FOLD="${JOINT_CMDC_FOLD:-0}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-joint_cmdc}"
FINAL_EVAL_LEVELS="${FINAL_EVAL_LEVELS:-subject}"
SUBMIT_DAIC_EVAL="${SUBMIT_DAIC_EVAL:-1}"
SUBMIT_CMDC_EVAL="${SUBMIT_CMDC_EVAL:-1}"
SUBMIT_LAST_EVAL="${SUBMIT_LAST_EVAL:-0}"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"
EXTRA_EVAL_ARGS="${EXTRA_EVAL_ARGS:-}"
SKIP_MANIFEST_BUILD="${SKIP_MANIFEST_BUILD:-0}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$PROJECT_ROOT/scripts/run_train_slurm.sh}"
EVAL_SCRIPT="${EVAL_SCRIPT:-$PROJECT_ROOT/scripts/run_eval_slurm.sh}"

DEFAULT_JOINT_PAIRS="$PROJECT_ROOT/configs/experiments/joint/daic_cmdc_audio_only_joint_tf.yaml::$PROJECT_ROOT/configs/experiments/joint/cmdc_component_audio_only.yaml $PROJECT_ROOT/configs/experiments/joint/daic_cmdc_text_only_joint_tf.yaml::$PROJECT_ROOT/configs/experiments/joint/cmdc_component_text_only.yaml $PROJECT_ROOT/configs/experiments/joint/daic_cmdc_audio_text_joint_tf.yaml::$PROJECT_ROOT/configs/experiments/joint/cmdc_component_audio_text.yaml"
JOINT_PAIRS="${JOINT_PAIRS:-$DEFAULT_JOINT_PAIRS}"

if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "Training script not found: $TRAIN_SCRIPT"
    exit 1
fi
if [ ! -f "$EVAL_SCRIPT" ]; then
    echo "Evaluation script not found: $EVAL_SCRIPT"
    exit 1
fi

resolve_run_root() {
    local config="$1"
    local run_root_rel
    run_root_rel="$(awk '
      /^output_dirs:/ {in_block=1; next}
      in_block && /^[^[:space:]]/ {in_block=0}
      in_block && /^[[:space:]]+run_root:/ {
        sub(/^[[:space:]]+run_root:[[:space:]]*/, "", $0)
        print $0
        exit
      }
    ' "$config" | tr -d '"' | tr -d "'")"
    if [ -z "$run_root_rel" ]; then
        echo "Could not parse output_dirs.run_root from config: $config" >&2
        return 1
    fi
    printf '%s\n' "${run_root_rel//\$\{PROJECT_ROOT\}/$PROJECT_ROOT}"
}

modality_from_config() {
    local config="$1"
    local stem
    stem="$(basename "$config" .yaml)"
    stem="${stem#daic_cmdc_}"
    stem="${stem%_joint_tf}"
    printf '%s\n' "$stem"
}

append_level_override() {
    local base_args="$1"
    local level="$2"
    printf '%s\n' "${base_args:+$base_args }--set evaluation.aggregation_level=$level"
}

echo "========================================"
echo "Joint DAIC+CMDC submission"
echo "  project_root:       $PROJECT_ROOT"
echo "  daic fold:          $FOLD"
echo "  cmdc fold:          $JOINT_CMDC_FOLD"
echo "  run_name_prefix:    $RUN_NAME_PREFIX"
echo "  final_eval_levels:  $FINAL_EVAL_LEVELS"
echo "  submit_daic_eval:   $SUBMIT_DAIC_EVAL"
echo "  submit_cmdc_eval:   $SUBMIT_CMDC_EVAL"
echo "  submit_last_eval:   $SUBMIT_LAST_EVAL"
echo "  extra_train_args:   ${EXTRA_TRAIN_ARGS:-<none>}"
echo "  extra_eval_args:    ${EXTRA_EVAL_ARGS:-<none>}"
echo "========================================"

JOINT_PAIRS="${JOINT_PAIRS//,/ }"
read -r -a PAIR_VALUES <<< "$JOINT_PAIRS"
read -r -a EVAL_LEVEL_VALUES <<< "$FINAL_EVAL_LEVELS"

for pair in "${PAIR_VALUES[@]}"; do
    PRIMARY_CONFIG="${pair%%::*}"
    CMDC_CONFIG="${pair##*::}"
    if [ "$PRIMARY_CONFIG" = "$CMDC_CONFIG" ]; then
        echo "Bad JOINT_PAIRS entry, expected primary::cmdc_component: $pair"
        exit 1
    fi
    [ -f "$PRIMARY_CONFIG" ] || { echo "Primary config not found: $PRIMARY_CONFIG"; exit 1; }
    [ -f "$CMDC_CONFIG" ] || { echo "CMDC component config not found: $CMDC_CONFIG"; exit 1; }

    MODALITY="$(modality_from_config "$PRIMARY_CONFIG")"
    RUN_NAME="${RUN_NAME_PREFIX}_cmdc_f${JOINT_CMDC_FOLD}_${MODALITY}"
    RUN_ROOT="$(resolve_run_root "$PRIMARY_CONFIG")"
    FOLD_DIR="$RUN_ROOT/$RUN_NAME/fold_$FOLD"
    BEST_CHECKPOINT_DIR="$FOLD_DIR/best_model"
    LAST_CHECKPOINT_DIR="$FOLD_DIR/last_model"

    EXPORT_ARGS="ALL,PROJECT_ROOT=$PROJECT_ROOT,CONFIG=$PRIMARY_CONFIG,FOLD=$FOLD,RUN_NAME=$RUN_NAME,EXTRA_TRAIN_ARGS=$EXTRA_TRAIN_ARGS,EXTRA_EVAL_ARGS=$EXTRA_EVAL_ARGS,SKIP_MANIFEST_BUILD=$SKIP_MANIFEST_BUILD,JOINT_CMDC_FOLD=$JOINT_CMDC_FOLD"

    echo "--- submitting joint train: modality=$MODALITY run_name=$RUN_NAME ---"
    echo "  primary_config:      $PRIMARY_CONFIG"
    echo "  cmdc_config:         $CMDC_CONFIG"
    echo "  best_checkpoint_dir: $BEST_CHECKPOINT_DIR"

    TRAIN_JOB_RAW="$(sbatch --parsable --export="$EXPORT_ARGS" "$TRAIN_SCRIPT")"
    TRAIN_JOB_ID="${TRAIN_JOB_RAW%%;*}"
    echo "  submitted training job: $TRAIN_JOB_ID"

    for LEVEL in "${EVAL_LEVEL_VALUES[@]}"; do
        if [ "$SUBMIT_DAIC_EVAL" = "1" ]; then
            DAIC_EVAL_ARGS="$(append_level_override "$EXTRA_EVAL_ARGS" "$LEVEL")"
            DAIC_OUTPUT_DIR="$BEST_CHECKPOINT_DIR/daic_test_eval_${LEVEL}"
            DAIC_EXPORT_ARGS="ALL,PROJECT_ROOT=$PROJECT_ROOT,CONFIG=$PRIMARY_CONFIG,FOLD=$FOLD,CHECKPOINT_DIR=$BEST_CHECKPOINT_DIR,OUTPUT_DIR=$DAIC_OUTPUT_DIR,EXTRA_EVAL_ARGS=$DAIC_EVAL_ARGS,JOINT_CMDC_FOLD=$JOINT_CMDC_FOLD"
            DAIC_JOB_RAW="$(sbatch --parsable --dependency=afterok:$TRAIN_JOB_ID --export="$DAIC_EXPORT_ARGS" "$EVAL_SCRIPT")"
            DAIC_JOB_ID="${DAIC_JOB_RAW%%;*}"
            echo "  submitted DAIC best eval ($LEVEL): $DAIC_JOB_ID"
        fi

        if [ "$SUBMIT_CMDC_EVAL" = "1" ]; then
            CMDC_EVAL_ARGS="$(append_level_override "$EXTRA_EVAL_ARGS" "$LEVEL")"
            CMDC_OUTPUT_DIR="$BEST_CHECKPOINT_DIR/cmdc_fold_${JOINT_CMDC_FOLD}_holdout_eval_${LEVEL}"
            CMDC_EXPORT_ARGS="ALL,PROJECT_ROOT=$PROJECT_ROOT,CONFIG=$CMDC_CONFIG,FOLD=$JOINT_CMDC_FOLD,CHECKPOINT_DIR=$BEST_CHECKPOINT_DIR,OUTPUT_DIR=$CMDC_OUTPUT_DIR,EXTRA_EVAL_ARGS=$CMDC_EVAL_ARGS,JOINT_CMDC_FOLD=$JOINT_CMDC_FOLD"
            CMDC_JOB_RAW="$(sbatch --parsable --dependency=afterok:$TRAIN_JOB_ID --export="$CMDC_EXPORT_ARGS" "$EVAL_SCRIPT")"
            CMDC_JOB_ID="${CMDC_JOB_RAW%%;*}"
            echo "  submitted CMDC holdout eval ($LEVEL): $CMDC_JOB_ID"
        fi
    done

    if [ "$SUBMIT_LAST_EVAL" = "1" ]; then
        LAST_OUTPUT_DIR="$LAST_CHECKPOINT_DIR/daic_test_eval_subject"
        LAST_EXPORT_ARGS="ALL,PROJECT_ROOT=$PROJECT_ROOT,CONFIG=$PRIMARY_CONFIG,FOLD=$FOLD,CHECKPOINT_DIR=$LAST_CHECKPOINT_DIR,OUTPUT_DIR=$LAST_OUTPUT_DIR,EXTRA_EVAL_ARGS=$EXTRA_EVAL_ARGS,JOINT_CMDC_FOLD=$JOINT_CMDC_FOLD"
        LAST_JOB_RAW="$(sbatch --parsable --dependency=afterok:$TRAIN_JOB_ID --export="$LAST_EXPORT_ARGS" "$EVAL_SCRIPT")"
        LAST_JOB_ID="${LAST_JOB_RAW%%;*}"
        echo "  submitted DAIC last-checkpoint eval: $LAST_JOB_ID"
    fi
done

echo "========================================"
echo "Joint submission complete."
echo "========================================"
