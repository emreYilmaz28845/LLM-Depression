#!/bin/bash
#SBATCH -J w2v2-extract
#SBATCH -A etur92
#SBATCH -q acc_ehpc
#SBATCH -t 02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#SBATCH --chdir=/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression

# Frozen wav2vec2 window features for the audio baseline, all five datasets
# (DAIC writes an official-test and an official-dev cache). Each dataset reuses
# its Qwen3-Omni audio_only frozen-feature reference run (fold 0) for config,
# saved split and manifest; see src/features/extract_wav2vec2.py.
#
# Submit with --chdir=<deployment code dir>. Required: OUTPUT_ROOT, MODEL_PATH.
# SMOKE_SUBJECTS>0 keeps that many subjects per partition (non-reportable).

set -euo pipefail
module purge
module load bsc/1.0
module load miniforge/24.3.0-0

ENV_ACTIVATE="${ENV_ACTIVATE:-/gpfs/projects/etur92/ozu647717/venvs/qwen3omni/bin/activate}"
source "$ENV_ACTIVATE"
PROJECT_ROOT="$(pwd)"
export PROJECT_ROOT
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

OUTPUT_ROOT="${OUTPUT_ROOT:?Set OUTPUT_ROOT}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH}"
MODEL_REVISION="${MODEL_REVISION:-1a640f32ac3e39899438a2931f9924c02f080a54}"
SEGMENT_SECONDS="${SEGMENT_SECONDS:-20}"
SMOKE_SUBJECTS="${SMOKE_SUBJECTS:-0}"
ONLY="${ONLY:-}"
LOG_ROOT="${LOG_ROOT:-$OUTPUT_ROOT/logs}"
mkdir -p "$LOG_ROOT"
exec > >(tee -a "$LOG_ROOT/w2v2-${SLURM_JOB_ID}.out")
exec 2> >(tee -a "$LOG_ROOT/w2v2-${SLURM_JOB_ID}.err" >&2)

OUT=/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression/output_model
# name | reference fold dir | final-eval source
ROUTES=(
  "daic_test|$OUT/promptcontext_v1_qwen3omni_likelihood/audio_only/daic/qwen3omni_daic_audio_only_fold0_prod_20260923/fold_0|saved"
  "daic_dev|$OUT/promptcontext_v1_qwen3omni_likelihood/audio_only/daic/qwen3omni_daic_audio_only_fold0_prod_20260923/fold_0|selection"
  "d3tec|$OUT/promptcontext_v1_qwen3omni_likelihood/audio_only/d3tec/qwen3omni_d3tec_audio_only_f0_prod_20260923/fold_0|saved"
  "androids_interview|$OUT/qwen3_androids_mergedfolds_native_20261015/audio_only/androids_interview/q3amf_audio_only_native_s1337_f0_r1/fold_0|saved"
  "cmdc|$OUT/promptcontext_v1_qwen3omni_likelihood/audio_only/cmdc/qwen3omni_cmdc_audio_only_f0_prod_20260923/fold_0|saved"
  "turkish|$OUT/promptcontext_v1_qwen3omni_likelihood/audio_only/turkish/turkish_pooled_baseline_audio_only_f0_20260925/fold_0|saved"
)

python -c 'import torch, transformers, soundfile; print("versions", torch.__version__, transformers.__version__, soundfile.__version__, torch.cuda.get_device_name(0))'
echo "project_root=$PROJECT_ROOT output_root=$OUTPUT_ROOT segment_seconds=$SEGMENT_SECONDS smoke_subjects=$SMOKE_SUBJECTS"

for route in "${ROUTES[@]}"; do
  IFS='|' read -r name ref source <<< "$route"
  if [ -n "$ONLY" ] && [[ ",$ONLY," != *",$name,"* ]]; then continue; fi
  echo "=== $name $(date -u +%FT%TZ)"
  python "$PROJECT_ROOT/src/features/extract_wav2vec2.py" \
    --reference-fold-dir "$ref" \
    --model-path "$MODEL_PATH" --model-revision "$MODEL_REVISION" \
    --segment-seconds "$SEGMENT_SECONDS" \
    --final-eval-source "$source" \
    --smoke-subjects "$SMOKE_SUBJECTS" \
    --output-root "$OUTPUT_ROOT/$name"
done
echo "=== done $(date -u +%FT%TZ)"
