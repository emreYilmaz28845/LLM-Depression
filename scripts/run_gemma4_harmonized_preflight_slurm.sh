#!/bin/bash
#SBATCH -J gemma4-harm-preflight
#SBATCH -A etur92
#SBATCH -q acc_ehpc
#SBATCH -t 08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#SBATCH --chdir=/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression

# CPU-only model-free Gemma harmonized preflight in the dedicated Gemma
# environment (only the processor is loaded, never the model weights).
set -euo pipefail
module purge
module load bsc/1.0
module load miniforge/24.3.0-0

PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
export PROJECT_ROOT
cd "$PROJECT_ROOT"

GEMMA_ENV="${GEMMA_ENV:-/gpfs/projects/etur92/ozu647717/venvs/gemma4_12b_tf5_14_1}"
ENV_ACTIVATE="${ENV_ACTIVATE:-$GEMMA_ENV/bin/activate}"
if [ ! -f "$ENV_ACTIVATE" ]; then
    echo "Gemma environment activate script not found: $ENV_ACTIVATE" >&2
    exit 1
fi
# shellcheck disable=SC1090
source "$ENV_ACTIVATE"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

RUN_ID="${RUN_ID:?Set RUN_ID}"
ENGLISH="${ENGLISH:-0}"
MODEL_PATH="${MODEL_PATH:-/gpfs/projects/etur92/ozu647717/models/gemma-4-12B-it/707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7}"
REQUIRED_PATH_PREFIX="${REQUIRED_PATH_PREFIX:-/gpfs/projects/etur92/ozu647717}"

DATASET_BASE_ROOT="${DATASET_BASE_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/Datasets}"
export DAIC_DATASET_ROOT="${DAIC_DATASET_ROOT:-$DATASET_BASE_ROOT/DAIC-WOZ/preprocessed}"
export DAIC_UNPROCESSED_ROOT="${DAIC_UNPROCESSED_ROOT:-$DATASET_BASE_ROOT/DAIC-WOZ/unprocessed}"
export DAIC_LABEL_ROOT="${DAIC_LABEL_ROOT:-$DATASET_BASE_ROOT/DAIC-WOZ/minimal_zips}"
export CMDC_DATASET_ROOT="${CMDC_DATASET_ROOT:-$DATASET_BASE_ROOT/CMDC}"
export EATD_DATASET_ROOT="${EATD_DATASET_ROOT:-$DATASET_BASE_ROOT/EATD-Corpus}"
export TURKISH_DATASET_ROOT="${TURKISH_DATASET_ROOT:-$DATASET_BASE_ROOT/Turkish}"
export D3TEC_DATASET_ROOT="${D3TEC_DATASET_ROOT:-$DATASET_BASE_ROOT/D3TEC DATASET/D3TEC DATASET}"
export D3TEC_FULL_TRANSCRIPTS="${D3TEC_FULL_TRANSCRIPTS:-$D3TEC_DATASET_ROOT/transcripts_qwen3_asr_spanish.jsonl}"
export D3TEC_SEGMENT_TRANSCRIPTS="${D3TEC_SEGMENT_TRANSCRIPTS:-$D3TEC_DATASET_ROOT/transcripts_qwen3_asr_spanish_segments.jsonl}"
export ANDROIDS_DATASET_ROOT="${ANDROIDS_DATASET_ROOT:-$DATASET_BASE_ROOT/Androids-Corpus/Androids-Corpus}"
export ANDROIDS_INTERVIEW_FULL_TRANSCRIPTS="${ANDROIDS_INTERVIEW_FULL_TRANSCRIPTS:-$ANDROIDS_DATASET_ROOT/interview_transcripts_qwen3_asr_italian.jsonl}"
export ANDROIDS_INTERVIEW_SEGMENT_TRANSCRIPTS="${ANDROIDS_INTERVIEW_SEGMENT_TRANSCRIPTS:-$ANDROIDS_DATASET_ROOT/interview_transcripts_qwen3_asr_italian_segments.jsonl}"
export TRANSLATION_ROOT="${TRANSLATION_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/translations}"
export HARMONIZED_SOURCE_COMMIT="${HARMONIZED_SOURCE_COMMIT:-$(tr -d '\n' < "$PROJECT_ROOT/.provenance/git_commit.txt")}"
export HARMONIZED_SOURCE_BRANCH="${HARMONIZED_SOURCE_BRANCH:-$(tr -d '\n' < "$PROJECT_ROOT/.provenance/git_branch.txt")}"

LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/logs/gemma4_harmonized_preflight}"
mkdir -p "$LOG_ROOT"
exec > >(tee -a "$LOG_ROOT/preflight-${SLURM_JOB_ID}.out")
exec 2> >(tee -a "$LOG_ROOT/preflight-${SLURM_JOB_ID}.err" >&2)

ARGS=(--run-id "$RUN_ID" --required-path-prefix "$REQUIRED_PATH_PREFIX" --model-path "$MODEL_PATH")
if [ "$ENGLISH" = "1" ]; then
    ARGS+=(--english)
fi
CMD=(python "$PROJECT_ROOT/scripts/prepare_gemma4_harmonized_mn5.py" "${ARGS[@]}")
printf 'Preflight command: '; printf '%q ' "${CMD[@]}"; printf '\n'
"${CMD[@]}"
