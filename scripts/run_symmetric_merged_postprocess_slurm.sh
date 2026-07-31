#!/bin/bash
#SBATCH -J sym-merged-eval
#SBATCH -A etur92
#SBATCH -q acc_ehpc
#SBATCH -t 48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
# MN5 allocates 20 CPUs per requested GPU.
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#SBATCH --chdir=/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression

set -euo pipefail
module purge
module load bsc/1.0
module load miniforge/24.3.0-0

PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
ENV_ACTIVATE="${ENV_ACTIVATE:-/gpfs/projects/etur92/ozu647717/venvs/qwen_mn5_rebuilt/bin/activate}"
DATASET_BASE_ROOT="${DATASET_BASE_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/Datasets}"
export DAIC_DATASET_ROOT="${DAIC_DATASET_ROOT:-$DATASET_BASE_ROOT/DAIC-WOZ/preprocessed}"
export CMDC_DATASET_ROOT="${CMDC_DATASET_ROOT:-$DATASET_BASE_ROOT/CMDC}"
export TURKISH_DATASET_ROOT="${TURKISH_DATASET_ROOT:-$DATASET_BASE_ROOT/Turkish}"
export D3TEC_DATASET_ROOT="${D3TEC_DATASET_ROOT:-$DATASET_BASE_ROOT/D3TEC DATASET/D3TEC DATASET}"
export D3TEC_FULL_TRANSCRIPTS="${D3TEC_FULL_TRANSCRIPTS:-$D3TEC_DATASET_ROOT/transcripts_qwen3_asr_spanish.jsonl}"
export D3TEC_SEGMENT_TRANSCRIPTS="${D3TEC_SEGMENT_TRANSCRIPTS:-$D3TEC_DATASET_ROOT/transcripts_qwen3_asr_spanish_segments.jsonl}"
export ANDROIDS_DATASET_ROOT="${ANDROIDS_DATASET_ROOT:-$DATASET_BASE_ROOT/Androids-Corpus/Androids-Corpus}"
export ANDROIDS_INTERVIEW_FULL_TRANSCRIPTS="${ANDROIDS_INTERVIEW_FULL_TRANSCRIPTS:-$ANDROIDS_DATASET_ROOT/interview_transcripts_qwen3_asr_italian.jsonl}"
export ANDROIDS_INTERVIEW_SEGMENT_TRANSCRIPTS="${ANDROIDS_INTERVIEW_SEGMENT_TRANSCRIPTS:-$ANDROIDS_DATASET_ROOT/interview_transcripts_qwen3_asr_italian_segments.jsonl}"
CONFIG="${CONFIG:?CONFIG is required}"
STAGE="${STAGE:?STAGE is required}"
FOLD="${FOLD:-0}"
RUN_ID="${RUN_ID:?RUN_ID is required}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:?CHECKPOINT_DIR is required}"
SUBJECTS_PER_CLASS="${SUBJECTS_PER_CLASS:-}"
LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/logs/symmetric_merged}"

if [ ! -f "$ENV_ACTIVATE" ]; then
    echo "Environment activate script not found: $ENV_ACTIVATE" >&2
    exit 1
fi
# shellcheck disable=SC1090
source "$ENV_ACTIVATE"
cd "$PROJECT_ROOT"
mkdir -p "$LOG_ROOT"
exec > >(tee -a "$LOG_ROOT/postprocess-${SLURM_JOB_ID}.out")
exec 2> >(tee -a "$LOG_ROOT/postprocess-${SLURM_JOB_ID}.err" >&2)
export PROJECT_ROOT PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export PYTHONPATH="$PROJECT_ROOT/.deps/qwen_hidden:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
CMD=(python -m src.merged.postprocess \
    --config "$CONFIG" --stage "$STAGE" --fold "$FOLD" --run-id "$RUN_ID" \
    --checkpoint-dir "$CHECKPOINT_DIR")
if [ -n "$SUBJECTS_PER_CLASS" ]; then CMD+=(--subjects-per-class "$SUBJECTS_PER_CLASS"); fi
"${CMD[@]}"
