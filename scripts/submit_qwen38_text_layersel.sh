#!/bin/bash
# Submit the frozen Qwen3.8-27B all-layer text extraction (lane qwen38-text-layersel-20261106).
#
# Same reference runs, condition names, env and wrapper as the final-layer frozen Qwen
# text caches (exp-frozen-base-standalone-heads-20261024), plus ALL_LAYERS=1. Every job
# is 1 node x 1 H100; all jobs run in ONE dependency chain, so at most 1 GPU is in use.
#
# usage (on an MN5 login node):
#   bash scripts/submit_qwen38_text_layersel.sh <deployment_code_dir> <runtime_dir> smoke|rest [--dry-run]
set -euo pipefail
CODE="${1:?deployment code dir}"; RT="${2:?runtime dir}"; WAVE="${3:?smoke|rest}"; DRY="${4:-}"
PERM=/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression
OM=$PERM/output_model
ENV_ACTIVATE=/gpfs/projects/etur92/ozu647717/venvs/qwen38_fsdp_fastpath_20260921/bin/activate

# dataset|language|part|reference best_model
ROUTES=(
  "d3tec|native|main|$OM/promptcontext_v1_qwen38_likelihood/text_only/d3tec/qwen38_pc_d3tec_f0_20260922/fold_0/best_model"
  "d3tec|english|main|$OM/qwen3_multiseed_english_20261002/text_only/d3tec/d3tec_text_only_english_s1337_f0/fold_0/best_model"
  "daic|native|main|$OM/promptcontext_v1_qwen38_likelihood/text_only/daic/qwen38_pc_daic_f0_20260922/fold_0/best_model"
  "daic|native|dev|$OM/promptcontext_v1_qwen38_likelihood/text_only/daic/qwen38_pc_daic_f0_20260922/fold_0/best_model"
  "androids_interview|native|main|$OM/qwen3_androids_mergedfolds_native_20261015/text_only/androids_interview/q3amf_text_only_native_s1337_f0_r1/fold_0/best_model"
  "androids_interview|english|main|$OM/qwen3_androids_mergedfolds_english_20261015/text_only/androids_interview/q3amf_text_only_english_s1337_f0_r1/fold_0/best_model"
  "cmdc|native|main|$OM/promptcontext_v1_qwen38_likelihood/text_only/cmdc/qwen38_pc_cmdc_f0_20260922/fold_0/best_model"
  "cmdc|english|main|$OM/qwen3_multiseed_english_20261002/text_only/cmdc/cmdc_text_only_english_s1337_f0/fold_0/best_model"
  "turkish|native|main|$OM/promptcontext_v1_qwen38_likelihood/text_only/turkish/turkish_pooled_baseline_text_only_f0_20260925/fold_0/best_model"
  "turkish|english|main|$OM/qwen3_multiseed_english_20261002/text_only/turkish/turkish_text_only_english_s1337_f0/fold_0/best_model"
)
case "$WAVE" in
  smoke) SELECTED=("${ROUTES[0]}") ;;
  rest) SELECTED=("${ROUTES[@]:1}") ;;
  *) echo "wave must be smoke or rest" >&2; exit 2 ;;
esac

[ "$DRY" = "--dry-run" ] || mkdir -p "$RT/logs/extract"
PREV="${AFTER_JOB:-}"
for route in "${SELECTED[@]}"; do
  IFS='|' read -r ds lang part ckpt <<< "$route"
  cache="$RT/caches/$ds/text_only_$lang/$part"
  exports="ALL,PROJECT_ROOT=$CODE,CHECKPOINT_DIR=$ckpt,CACHE_DIR=$cache,CONDITION=text_only_frozen_base_$lang"
  exports+=",BASE_ONLY=1,SKIP_CLASSIFIERS=1,ALL_LAYERS=1,ENV_ACTIVATE=$ENV_ACTIVATE"
  exports+=",QWEN_HIDDEN_DEPS=$PERM/.deps/qwen_hidden,LOG_ROOT=$RT/logs/extract"
  [ "$part" = dev ] && exports+=",FINAL_EVAL_SOURCE=selection"
  name="ql-${ds:0:4}-${lang:0:2}-$part"
  dep=(); [ -n "$PREV" ] && dep=(--dependency="afterany:$PREV")
  cmd=(sbatch --parsable "${dep[@]}" --chdir="$CODE" -J "$name" -t 00:45:00 --gres=gpu:1
       --cpus-per-task=20 --export="$exports" "$CODE/scripts/run_qwen_hidden_extract_slurm.sh")
  if [ "$DRY" = "--dry-run" ]; then
    printf 'DRY %s: ' "$name"; printf '%q ' "${cmd[@]}"; printf '\n'; PREV="DRY_$name"
  else
    jid=$("${cmd[@]}"); echo "JOB $name $jid $cache"; PREV=$jid
  fi
done
