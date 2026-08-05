#!/usr/bin/env bash
# Re-evaluate existing symmetric-merged checkpoints with the new DAIC
# full-coverage eval (balanced_joint_cover). No retraining: postprocess points
# at the OLD run's best_model checkpoints, heads run logreg+xgb_fixed only.
#
# Usage (alogin2):
#   DRY_RUN=1 RUN_ID=symmetric_merged_reeval_20260805_<short> bash scripts/submit_symmetric_merged_reeval.sh
#   DRY_RUN=0 RUN_ID=... bash scripts/submit_symmetric_merged_reeval.sh
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
RUN_ID="${RUN_ID:?Set a unique RUN_ID}"
DRY_RUN="${DRY_RUN:-1}"
OLD_RUN="${OLD_RUN:-symmetric_merged_smoke_6fba6e632653}"
case "$DRY_RUN" in 0|1) ;; *) echo "DRY_RUN must be 0 or 1" >&2; exit 2;; esac

declare -A CONFIGS=(
  [audio_text]="$PROJECT_ROOT/configs/experiments/merged/symmetric_merged_audio_text.yaml"
  [audio_only]="$PROJECT_ROOT/configs/experiments/merged/symmetric_merged_audio_only.yaml"
  [text_only]="$PROJECT_ROOT/configs/experiments/merged/symmetric_merged_text_only.yaml"
)

submit() {
  if [ "$DRY_RUN" = 1 ]; then
    printf 'DRY_RUN ' >&2; printf '%q ' "$@" >&2; printf '\n' >&2
    printf 'dry_%s\n' "$RANDOM"
  else
    "$@"
  fi
}
job_id() { printf '%s' "${1%%;*}"; }

echo "RUN_ID=$RUN_ID OLD_RUN=$OLD_RUN DRY_RUN=$DRY_RUN"

for modality in audio_text audio_only text_only; do
  config="${CONFIGS[$modality]}"
  [ -f "$config" ] || { echo "missing config: $config" >&2; exit 3; }
  for stage in cv final; do
    folds=(0 1 2 3 4)
    [ "$stage" = final ] && folds=(0)
    for fold in "${folds[@]}"; do
      checkpoint="$PROJECT_ROOT/output_model/symmetric_merged/$modality/$OLD_RUN/$stage/fold_$fold/best_model"
      [ -d "$checkpoint" ] || { echo "missing checkpoint: $checkpoint" >&2; exit 3; }
      post_root="$PROJECT_ROOT/outputs/symmetric_merged/$modality/$RUN_ID/$stage/fold_$fold"
      log_root="$PROJECT_ROOT/logs/symmetric_merged/$RUN_ID/$modality/$stage/fold_$fold"
      post_raw="$(submit sbatch --parsable --gres=gpu:1 --ntasks=1 --cpus-per-task=20 --time=48:00:00 \
        --export="ALL,PROJECT_ROOT=$PROJECT_ROOT,CONFIG=$config,STAGE=$stage,FOLD=$fold,RUN_ID=$RUN_ID,CHECKPOINT_DIR=$checkpoint,LOG_ROOT=$log_root" \
        "$PROJECT_ROOT/scripts/run_symmetric_merged_postprocess_slurm.sh")"
      post_job="$(job_id "$post_raw")"
      head_raw="$(submit sbatch --parsable --dependency="afterok:$post_job" --ntasks=1 --cpus-per-task=20 --time=12:00:00 \
        --export="ALL,PROJECT_ROOT=$PROJECT_ROOT,CONFIG=$config,STAGE=$stage,FOLD=$fold,RUN_ID=$RUN_ID,FEATURES_DIR=$post_root/features,TRIALS=0,LOG_ROOT=$log_root" \
        "$PROJECT_ROOT/scripts/run_symmetric_merged_head_slurm.sh")"
      head_job="$(job_id "$head_raw")"
      printf 'modality=%s stage=%s fold=%s post=%s head=%s\n' "$modality" "$stage" "$fold" "$post_job" "$head_job"
    done
  done
done
echo "done"
