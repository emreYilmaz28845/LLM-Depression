#!/usr/bin/env bash
# Coverage-audit head expansion: fit logreg/xgb heads on the canonical K=4
# checkpoints, evaluate on fixed4 (c1_fixed) and complete-coverage (c2_balanced)
# views. No Qwen retraining; extraction is one GPU job per checkpoint, then
# CPU-only classical heads.
#
# Usage (on alogin2):
#   DRY_RUN=1 RUN_ID=daic_coverage_heads_20260805_<short> bash scripts/submit_daic_coverage_heads.sh
#   DRY_RUN=0 RUN_ID=... bash scripts/submit_daic_coverage_heads.sh
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
RUN_ID="${RUN_ID:?Set a unique RUN_ID}"
DRY_RUN="${DRY_RUN:-1}"
case "$DRY_RUN" in 0|1) ;; *) echo "DRY_RUN must be 0 or 1" >&2; exit 2;; esac

declare -A CHECKPOINTS=(
  [audio_text]="$PROJECT_ROOT/output_model/audio_text/daic/daic_main_k4_control_20260804_f26dd45/fold_0/best_model"
  [audio_only]="$PROJECT_ROOT/output_model/subject_audio/daic/daic_replicates_20ep_s1337_daic_audio_only_selposf1_tf/fold_0/best_model"
  [text_only]="$PROJECT_ROOT/output_model/text_only/daic/daic_replicates_20ep_s1337_daic_text_only_selposf1_tf/fold_0/best_model"
)
VARIANTS=(logreg_raw xgb_raw)

submit() {
  if [ "$DRY_RUN" = 1 ]; then
    printf 'DRY_RUN ' >&2; printf '%q ' "$@" >&2; printf '\n' >&2
    printf 'dry_%s\n' "$RANDOM"
  else
    "$@"
  fi
}
job_id() { printf '%s' "${1%%;*}"; }

echo "RUN_ID=$RUN_ID DRY_RUN=$DRY_RUN PROJECT_ROOT=$PROJECT_ROOT"

for modality in "${!CHECKPOINTS[@]}"; do
  checkpoint="${CHECKPOINTS[$modality]}"
  [ -d "$checkpoint" ] || { echo "missing checkpoint: $checkpoint" >&2; exit 3; }
  cache_root="$PROJECT_ROOT/outputs/daic_coverage_heads/$RUN_ID/$modality/hidden"
  classical_root="$PROJECT_ROOT/outputs/daic_coverage_heads/$RUN_ID/$modality/classical"
  log_root="$PROJECT_ROOT/logs/daic_coverage_heads/$RUN_ID/$modality"

  hidden_raw="$(submit sbatch --parsable --gres=gpu:1 --ntasks=1 --cpus-per-task=20 --time=12:00:00 \
    --export="ALL,PROJECT_ROOT=$PROJECT_ROOT,CHECKPOINT_DIR=$checkpoint,CACHE_ROOT=$cache_root,STRATEGY=joint,LOG_ROOT=$log_root" \
    "$PROJECT_ROOT/scripts/run_daic_chunking_hidden_slurm.sh")"
  hidden_job="$(job_id "$hidden_raw")"

  for variant in "${VARIANTS[@]}"; do
    submit sbatch --parsable --dependency="afterok:$hidden_job" --ntasks=1 --cpus-per-task=4 --time=01:00:00 \
      --export="ALL,PROJECT_ROOT=$PROJECT_ROOT,CACHE_ROOT=$cache_root,OUTPUT_ROOT=$classical_root,STRATEGY=joint,VARIANT=$variant,LOG_ROOT=$log_root" \
      "$PROJECT_ROOT/scripts/run_daic_chunking_classical_slurm.sh"
  done
  printf 'modality=%s hidden=%s heads=%s(logreg) %s(xgb)\n' "$modality" "$hidden_job" "${classical_root}/c1" "${classical_root}/c2"
done
echo "submitted. submission record: none (adhoc; see squeue)"
