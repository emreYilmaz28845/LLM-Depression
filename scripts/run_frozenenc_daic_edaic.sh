#!/usr/bin/env bash
#
# Re-run the DAIC + EDAIC subject-level AUDIO sweep with the audio-encoder freeze
# (DepressInstruct recipe) now default-on in build_lora_config. The previous
# subject_audio* results leaked LoRA into the Whisper audio_tower and are stale.
#
# All jobs are CHAINED: each training job waits on the previous one via
# SBATCH_DEPENDENCY=afterany:<prev_train_id>, so only one trains at a time (kind to
# the 4xH100 queue). Each best-checkpoint eval auto-depends on its own train job
# (afterok) inside submit_train_and_eval.sh. Pass --no-chain to fire them in
# parallel instead.
#
# Usage:
#   bash scripts/run_frozenenc_daic_edaic.sh            # chained (default)
#   bash scripts/run_frozenenc_daic_edaic.sh --no-chain # parallel
#
# text_only configs are intentionally excluded: they have no audio_tower, so the
# freeze does not change them and their reported numbers stay valid.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
SUBMIT_SCRIPT="${SUBMIT_SCRIPT:-$PROJECT_ROOT/scripts/submit_train_and_eval.sh}"
RUN_SUFFIX="${RUN_SUFFIX:-_frozenenc}"

CHAIN=1
if [ "${1:-}" = "--no-chain" ]; then
    CHAIN=0
fi

# Audio paths that leaked and must be re-run. EDAIC has no reg1 subject configs.
CONFIGS=(
    daic_subject_audio_reg1
    daic_subject_audio_reg2
    daic_subject_audio_reg3
    daic_subject_audio_reg4
    daic_subject_audio_text_reg1
    daic_subject_audio_text_reg2
    daic_subject_audio_text_reg3
    daic_subject_audio_text_reg4
    edaic_subject_audio_reg2
    edaic_subject_audio_reg3
    edaic_subject_audio_reg4
    edaic_subject_audio_text_reg2
    edaic_subject_audio_text_reg3
    edaic_subject_audio_text_reg4
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
