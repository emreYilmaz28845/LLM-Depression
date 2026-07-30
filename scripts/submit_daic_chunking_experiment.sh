#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
RUN_ID="${RUN_ID:?Set a unique RUN_ID}"
STAGE="${STAGE:-smoke}"
DRY_RUN="${DRY_RUN:-1}"
case "$STAGE" in smoke|production) ;; *) echo "STAGE must be smoke or production" >&2; exit 2;; esac
case "$DRY_RUN" in 0|1) ;; *) echo "DRY_RUN must be 0 or 1" >&2; exit 2;; esac

ROOT="$PROJECT_ROOT/outputs/daic_chunking/$RUN_ID"
LOG_ROOT="$PROJECT_ROOT/logs/daic_chunking/$RUN_ID"
if [ -e "$ROOT" ] || [ -e "$LOG_ROOT" ]; then
  echo "Collision: RUN_ID already exists: $RUN_ID" >&2
  exit 3
fi

declare -A CONFIG=(
  [joint]="$PROJECT_ROOT/configs/experiments/daic_chunking/joint_random_k4.yaml"
  [rotary]="$PROJECT_ROOT/configs/experiments/daic_chunking/independent_rotary_k4.yaml"
  [all]="$PROJECT_ROOT/configs/experiments/daic_chunking/independent_all.yaml"
)
declare -A CONDITION=([joint]=c1 [rotary]=c3 [all]=c4)
EXTRA=""
if [ "$STAGE" = "smoke" ]; then
  EXTRA="--set split.smoke_subject_limit=4 --set training.num_train_epochs=1 --set training.early_stopping.enabled=false"
fi

submit() {
  if [ "$DRY_RUN" = "1" ]; then
    printf 'DRY_RUN ' >&2; printf '%q ' "$@" >&2; printf '\n' >&2
    printf 'dry_%s\n' "$RANDOM"
  else
    "$@"
  fi
}
job_id() { local raw="$1"; printf '%s\n' "${raw%%;*}"; }

declare -A TRAIN_JOB HIDDEN_JOB
for strategy in joint rotary all; do
  run_name="${RUN_ID}_${strategy}"
  fold_dir="$PROJECT_ROOT/output_model/daic_chunking/${strategy/joint/joint_random_k4}/$run_name/fold_0"
  if [ "$strategy" = "rotary" ]; then fold_dir="$PROJECT_ROOT/output_model/daic_chunking/independent_rotary_k4/$run_name/fold_0"; fi
  if [ "$strategy" = "all" ]; then fold_dir="$PROJECT_ROOT/output_model/daic_chunking/independent_all/$run_name/fold_0"; fi
  if [ -e "$fold_dir" ]; then
    echo "Collision: training output already exists: $fold_dir" >&2
    exit 3
  fi
  raw="$(submit sbatch --parsable --job-name="dk-${RUN_ID}-${strategy}-train" \
    --export="ALL,PROJECT_ROOT=$PROJECT_ROOT,CONFIG=${CONFIG[$strategy]},FOLD=0,RUN_NAME=$run_name,EXTRA_TRAIN_ARGS=$EXTRA,SKIP_MANIFEST_BUILD=0,LOG_ROOT=$LOG_ROOT/train/$strategy" \
    "$PROJECT_ROOT/scripts/run_train_slurm.sh")"
  TRAIN_JOB[$strategy]="$(job_id "$raw")"

  eval_extra="$EXTRA"
  raw="$(submit sbatch --parsable --dependency="afterok:${TRAIN_JOB[$strategy]}" \
    --job-name="dk-${RUN_ID}-${strategy}-eval" \
    --export="ALL,PROJECT_ROOT=$PROJECT_ROOT,CONFIG=${CONFIG[$strategy]},FOLD=0,CHECKPOINT_DIR=$fold_dir/best_model,OUTPUT_DIR=$ROOT/qwen/${CONDITION[$strategy]},EXTRA_EVAL_ARGS=$eval_extra,LOG_ROOT=$LOG_ROOT/eval/$strategy" \
    "$PROJECT_ROOT/scripts/run_eval_slurm.sh")"
  echo "qwen_${CONDITION[$strategy]}=$(job_id "$raw")"
  if [ "$strategy" = "joint" ]; then
    raw="$(submit sbatch --parsable --dependency="afterok:${TRAIN_JOB[$strategy]}" \
      --job-name="dk-${RUN_ID}-c2-eval" \
      --export="ALL,PROJECT_ROOT=$PROJECT_ROOT,CONFIG=${CONFIG[$strategy]},FOLD=0,CHECKPOINT_DIR=$fold_dir/best_model,OUTPUT_DIR=$ROOT/qwen/c2,EXTRA_EVAL_ARGS=$eval_extra --set data.eval_chunk_policy=balanced_joint_cover --set data.eval_chunks_per_subject=4,LOG_ROOT=$LOG_ROOT/eval/c2" \
      "$PROJECT_ROOT/scripts/run_eval_slurm.sh")"
    echo "qwen_c2=$(job_id "$raw")"
  fi
  raw="$(submit sbatch --parsable --dependency="afterok:${TRAIN_JOB[$strategy]}" \
    --job-name="dk-${RUN_ID}-${strategy}-hidden" \
    --export="ALL,PROJECT_ROOT=$PROJECT_ROOT,CHECKPOINT_DIR=$fold_dir/best_model,CACHE_ROOT=$ROOT/hidden/$strategy,STRATEGY=$strategy,LOG_ROOT=$LOG_ROOT/hidden/$strategy" \
    "$PROJECT_ROOT/scripts/run_daic_chunking_hidden_slurm.sh")"
  HIDDEN_JOB[$strategy]="$(job_id "$raw")"
done

for strategy in joint rotary all; do
  for variant in logreg_raw xgb_raw; do
    raw="$(submit sbatch --parsable --dependency="afterok:${HIDDEN_JOB[$strategy]}" \
      --job-name="dk-${RUN_ID}-${strategy}-${variant}" \
      --export="ALL,PROJECT_ROOT=$PROJECT_ROOT,CACHE_ROOT=$ROOT/hidden/$strategy,OUTPUT_ROOT=$ROOT/classical,STRATEGY=$strategy,VARIANT=$variant,LOG_ROOT=$LOG_ROOT/classical" \
      "$PROJECT_ROOT/scripts/run_daic_chunking_classical_slurm.sh")"
    echo "classical_${strategy}_${variant}=$(job_id "$raw")"
  done
done
printf 'train_joint=%s\ntrain_rotary=%s\ntrain_all=%s\n' \
  "${TRAIN_JOB[joint]}" "${TRAIN_JOB[rotary]}" "${TRAIN_JOB[all]}"
