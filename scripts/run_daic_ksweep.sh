#!/usr/bin/env bash
#
# DAIC K-ablation: does smaller K (stronger combinatorial audio augmentation)
# reduce overfitting on the audio+text path without hurting the test operating
# point? Everything is held at the DECIDED default (frozen encoder + macro-F1
# selection + teacher-forced eval, handoff §8); the ONLY axes are:
#   - K = chunks_per_subject in {2, 3, 4}   (smaller K = more aug, overlap ~ K^2/N)
#   - seed in {1337, 2024, 7}               (drives BOTH LoRA init and the per-epoch
#                                            K-subset sampling -> genuine replicates;
#                                            split.seed stays 1337 so eval data is
#                                            identical across replicates)
# A text_only arm (same recipe/seeds) is the modality reference for the paired
# audio-vs-text McNemar comparison.
#
# Primary readout is the train -> inner-val macro-F1 GAP (overfit signal), NOT the
# noisy N=47 test F1. Report test metrics only as mean +/- std over the 3 seeds.
#
# All jobs are CHAINED: each training job waits on the previous via
# SBATCH_DEPENDENCY=afterany:<prev_train_id> (one trains at a time, kind to the
# 4xH100 queue). Each best-checkpoint eval auto-depends on its own train job
# (afterok) inside submit_train_and_eval.sh. Pass --no-chain to fire in parallel.
#
# Usage:
#   bash scripts/run_daic_ksweep.sh            # chained (default)
#   bash scripts/run_daic_ksweep.sh --no-chain # parallel
#
# NOTE: daic_subject_audio_text_reg1_selposf1_tf (K=4, seed 1337) already has a
# result (handoff §8, F1 0.737). It is included for a complete, uniformly-named
# sweep; re-running reproduces it. Set RUN_SUFFIX to avoid clobbering prior runs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
SUBMIT_SCRIPT="${SUBMIT_SCRIPT:-$PROJECT_ROOT/scripts/submit_train_and_eval.sh}"
RUN_SUFFIX="${RUN_SUFFIX:-}"

CHAIN=1
if [ "${1:-}" = "--no-chain" ]; then
    CHAIN=0
fi

# audio+text K-sweep (3 K x 3 seeds) then the text_only reference (3 seeds).
CONFIGS=(
    daic_subject_audio_text_reg1_selposf1_tf
    daic_subject_audio_text_reg1_selposf1_tf_s2024
    daic_subject_audio_text_reg1_selposf1_tf_s7
    daic_subject_audio_text_reg1_selposf1_tf_k3
    daic_subject_audio_text_reg1_selposf1_tf_k3_s2024
    daic_subject_audio_text_reg1_selposf1_tf_k3_s7
    daic_subject_audio_text_reg1_selposf1_tf_k2
    daic_subject_audio_text_reg1_selposf1_tf_k2_s2024
    daic_subject_audio_text_reg1_selposf1_tf_k2_s7
    daic_text_only_reg1_selposf1_tf
    daic_text_only_reg1_selposf1_tf_s2024
    daic_text_only_reg1_selposf1_tf_s7
)

if [ ! -f "$SUBMIT_SCRIPT" ]; then
    echo "submit script not found: $SUBMIT_SCRIPT" >&2
    exit 1
fi

PREV_TRAIN_ID=""
for name in "${CONFIGS[@]}"; do
    config="$PROJECT_ROOT/configs/$name.yaml"
    if [ ! -f "$config" ]; then
        echo "WARNING: config not found, skipping: $config" >&2
        continue
    fi
    run_name="${name}${RUN_SUFFIX}"

    dep=""
    if [ "$CHAIN" = "1" ] && [ -n "$PREV_TRAIN_ID" ]; then
        dep="afterany:$PREV_TRAIN_ID"
    fi

    echo "=================================================================="
    echo "Submitting: $name  (run_name=$run_name, dependency=${dep:-<none>})"
    echo "=================================================================="

    # Capture submit output so we can parse the train job id to chain the next one.
    out="$(CONFIG="$config" RUN_NAME="$run_name" SBATCH_DEPENDENCY="$dep" \
        bash "$SUBMIT_SCRIPT" 2>&1)"
    echo "$out"

    train_id="$(printf '%s\n' "$out" | sed -n 's/^Submitted training job: //p' | tail -1)"
    if [ -z "$train_id" ]; then
        echo "ERROR: could not parse training job id for $name; aborting chain." >&2
        exit 1
    fi
    echo ">> $name training job id: $train_id"
    PREV_TRAIN_ID="$train_id"
done

echo "All submissions complete. Last training job id: ${PREV_TRAIN_ID:-<none>}"
