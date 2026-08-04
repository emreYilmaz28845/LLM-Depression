#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
RUN_ID="${RUN_ID:?Set a unique RUN_ID}"
CONFIG="${CONFIG:-$PROJECT_ROOT/configs/main/daic_audio_text_selposf1_tf.yaml}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$PROJECT_ROOT/output_model/audio_text/daic/daic_main_k4_control_20260804_f26dd45/fold_0/best_model}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/daic_k4_coverage_audit}"
LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/logs/daic_k4_coverage_audit/$RUN_ID}"
DRY_RUN="${DRY_RUN:-1}"
RESUME="${RESUME:-0}"
EXPECTED_MODALITY="${EXPECTED_MODALITY:-audio_text}"
ALLOW_HISTORICAL_REPLAY_MISMATCH="${ALLOW_HISTORICAL_REPLAY_MISMATCH:-0}"

case "$DRY_RUN" in 0|1) ;; *) echo "DRY_RUN must be 0 or 1" >&2; exit 2 ;; esac
case "$RESUME" in 0|1) ;; *) echo "RESUME must be 0 or 1" >&2; exit 2 ;; esac
case "$EXPECTED_MODALITY" in audio_text|audio_only) ;; *) echo "EXPECTED_MODALITY must be audio_text or audio_only" >&2; exit 2 ;; esac
case "$ALLOW_HISTORICAL_REPLAY_MISMATCH" in 0|1) ;; *) echo "ALLOW_HISTORICAL_REPLAY_MISMATCH must be 0 or 1" >&2; exit 2 ;; esac
if [ "$RESUME" = "0" ] && { [ -e "$OUTPUT_ROOT/$RUN_ID" ] || [ -e "$LOG_ROOT" ]; }; then
  echo "Collision: RUN_ID already exists: $RUN_ID" >&2
  exit 3
fi
if [ ! -d "$CHECKPOINT_DIR" ]; then
  echo "Missing checkpoint: $CHECKPOINT_DIR" >&2
  exit 4
fi

command=(
  sbatch --parsable
  --job-name="dk4cov-$RUN_ID"
  --export="ALL,PROJECT_ROOT=$PROJECT_ROOT,CONFIG=$CONFIG,CHECKPOINT_DIR=$CHECKPOINT_DIR,RUN_ID=$RUN_ID,OUTPUT_ROOT=$OUTPUT_ROOT,LOG_ROOT=$LOG_ROOT,RESUME=$RESUME,EXPECTED_MODALITY=$EXPECTED_MODALITY,ALLOW_HISTORICAL_REPLAY_MISMATCH=$ALLOW_HISTORICAL_REPLAY_MISMATCH"
  "$PROJECT_ROOT/scripts/run_daic_k4_coverage_audit_slurm.sh"
)
if [ "$DRY_RUN" = "1" ]; then
  printf 'DRY_RUN '; printf '%q ' "${command[@]}"; printf '\n'
else
  "${command[@]}"
fi
