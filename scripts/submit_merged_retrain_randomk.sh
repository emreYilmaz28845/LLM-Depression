#!/usr/bin/env bash
# Change-1 experiment: retrain a merged modality with DAIC random-K
# chunk augmentation restored (src/merged/train.py chunk_sampling=random).
# 5 CV folds chained to postprocess (full-coverage DAIC eval) and heads
# (logreg + xgb_fixed, no Optuna). The final stage is submitted separately
# with EPOCHS=<rounded median of this run's CV selected epochs> after CV
# completes (protocol: final_epoch_policy=rounded_median_selected_epoch).
#
# Usage (alogin2):
#   DRY_RUN=1 RUN_ID=merged_retrain_randomk_20260805_<short> bash scripts/submit_merged_retrain_randomk.sh
#   DRY_RUN=0 RUN_ID=... bash scripts/submit_merged_retrain_randomk.sh
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
RUN_ID="${RUN_ID:?Set a unique RUN_ID}"
DRY_RUN="${DRY_RUN:-1}"
STAGE="${STAGE:-cv}"   # cv (default) or final (requires EPOCHS)
EPOCHS="${EPOCHS:-}"
MODALITY="${MODALITY:-audio_text}"  # audio_text (default) or audio_only
case "$DRY_RUN" in 0|1) ;; *) echo "DRY_RUN must be 0 or 1" >&2; exit 2;; esac

CONFIG="$PROJECT_ROOT/configs/experiments/merged/symmetric_merged_${MODALITY}.yaml"
[ -f "$CONFIG" ] || { echo "missing config: $CONFIG" >&2; exit 3; }
if [ "$STAGE" = final ] && [ -z "$EPOCHS" ]; then
  echo "STAGE=final requires EPOCHS (rounded median of this run's CV selected epochs)" >&2; exit 3
fi

submit() {
  if [ "$DRY_RUN" = 1 ]; then
    printf 'DRY_RUN ' >&2; printf '%q ' "$@" >&2; printf '\n' >&2
    printf 'dry_%s\n' "$RANDOM"
  else
    "$@"
  fi
}
job_id() { printf '%s' "${1%%;*}"; }

echo "RUN_ID=$RUN_ID STAGE=$STAGE EPOCHS=${EPOCHS:-auto} DRY_RUN=$DRY_RUN"

folds=(0 1 2 3 4)
[ "$STAGE" = final ] && folds=(0)
for fold in "${folds[@]}"; do
  run_root="$PROJECT_ROOT/output_model/symmetric_merged/$MODALITY/$RUN_ID/$STAGE/fold_$fold"
  post_root="$PROJECT_ROOT/outputs/symmetric_merged/$MODALITY/$RUN_ID/$STAGE/fold_$fold"
  log_root="$PROJECT_ROOT/logs/symmetric_merged/$RUN_ID/$MODALITY/$STAGE/fold_$fold"
  export_args="ALL,PROJECT_ROOT=$PROJECT_ROOT,CONFIG=$CONFIG,STAGE=$STAGE,FOLD=$fold,RUN_ID=$RUN_ID,LOG_ROOT=$log_root"
  if [ -n "$EPOCHS" ]; then export_args="$export_args,EPOCHS=$EPOCHS"; fi
  train_raw="$(submit sbatch --parsable --gres=gpu:4 --ntasks=1 --cpus-per-task=80 --time=72:00:00 \
    --export="$export_args" \
    "$PROJECT_ROOT/scripts/run_symmetric_merged_train_slurm.sh")"
  train_job="$(job_id "$train_raw")"
  post_raw="$(submit sbatch --parsable --dependency="afterok:$train_job" --gres=gpu:1 --ntasks=1 --cpus-per-task=20 --time=48:00:00 \
    --export="ALL,PROJECT_ROOT=$PROJECT_ROOT,CONFIG=$CONFIG,STAGE=$STAGE,FOLD=$fold,RUN_ID=$RUN_ID,CHECKPOINT_DIR=$run_root/best_model,LOG_ROOT=$log_root" \
    "$PROJECT_ROOT/scripts/run_symmetric_merged_postprocess_slurm.sh")"
  post_job="$(job_id "$post_raw")"
  head_raw="$(submit sbatch --parsable --dependency="afterok:$post_job" --ntasks=1 --cpus-per-task=20 --time=12:00:00 \
    --export="ALL,PROJECT_ROOT=$PROJECT_ROOT,CONFIG=$CONFIG,STAGE=$STAGE,FOLD=$fold,RUN_ID=$RUN_ID,FEATURES_DIR=$post_root/features,TRIALS=0,LOG_ROOT=$log_root" \
    "$PROJECT_ROOT/scripts/run_symmetric_merged_head_slurm.sh")"
  head_job="$(job_id "$head_raw")"
  printf 'stage=%s fold=%s train=%s post=%s head=%s\n' "$STAGE" "$fold" "$train_job" "$post_job" "$head_job"
done
echo "done"
