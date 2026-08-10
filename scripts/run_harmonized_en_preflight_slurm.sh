#!/bin/bash
#SBATCH -J harm-en-preflight
#SBATCH -A etur92
#SBATCH -q acc_ehpc
#SBATCH -t 02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
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
TRANSLATION_ROOT="${TRANSLATION_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/translations}"
RUN_ID="${RUN_ID:?RUN_ID is required}"
GITHUB_ISSUE="${GITHUB_ISSUE:?GITHUB_ISSUE is required}"
GITHUB_PR="${GITHUB_PR:?GITHUB_PR is required}"
HARMONIZED_EN_SOURCE_COMMIT="${HARMONIZED_EN_SOURCE_COMMIT:-$(tr -d '\n' < "$PROJECT_ROOT/.provenance/git_commit.txt")}"
HARMONIZED_EN_SOURCE_BRANCH="${HARMONIZED_EN_SOURCE_BRANCH:-$(tr -d '\n' < "$PROJECT_ROOT/.provenance/git_branch.txt")}"

export PROJECT_ROOT DATASET_BASE_ROOT TRANSLATION_ROOT HARMONIZED_EN_SOURCE_COMMIT HARMONIZED_EN_SOURCE_BRANCH
export CMDC_DATASET_ROOT="${CMDC_DATASET_ROOT:-$DATASET_BASE_ROOT/CMDC}"
export TURKISH_DATASET_ROOT="${TURKISH_DATASET_ROOT:-$DATASET_BASE_ROOT/Turkish}"
export D3TEC_DATASET_ROOT="${D3TEC_DATASET_ROOT:-$DATASET_BASE_ROOT/D3TEC DATASET/D3TEC DATASET}"
export D3TEC_FULL_TRANSCRIPTS="${D3TEC_FULL_TRANSCRIPTS:-$D3TEC_DATASET_ROOT/transcripts_qwen3_asr_spanish.jsonl}"
export D3TEC_SEGMENT_TRANSCRIPTS="${D3TEC_SEGMENT_TRANSCRIPTS:-$D3TEC_DATASET_ROOT/transcripts_qwen3_asr_spanish_segments.jsonl}"
export ANDROIDS_DATASET_ROOT="${ANDROIDS_DATASET_ROOT:-$DATASET_BASE_ROOT/Androids-Corpus/Androids-Corpus}"
export ANDROIDS_INTERVIEW_FULL_TRANSCRIPTS="${ANDROIDS_INTERVIEW_FULL_TRANSCRIPTS:-$ANDROIDS_DATASET_ROOT/interview_transcripts_qwen3_asr_italian.jsonl}"
export ANDROIDS_INTERVIEW_SEGMENT_TRANSCRIPTS="${ANDROIDS_INTERVIEW_SEGMENT_TRANSCRIPTS:-$ANDROIDS_DATASET_ROOT/interview_transcripts_qwen3_asr_italian_segments.jsonl}"
export MODEL_PATH="${MODEL_PATH:-/gpfs/projects/etur92/ozu647717/models/Qwen2-Audio-7B-Instruct}"

if [ ! -f "$ENV_ACTIVATE" ]; then
    echo "Environment activate script not found: $ENV_ACTIVATE" >&2
    exit 1
fi
# shellcheck disable=SC1090
source "$ENV_ACTIVATE"
cd "$PROJECT_ROOT"

LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/logs/harmonized_en_mn5_preflight/$RUN_ID}"
mkdir -p "$LOG_ROOT"
exec > >(tee -a "$LOG_ROOT/preflight-${SLURM_JOB_ID}.out")
exec 2> >(tee -a "$LOG_ROOT/preflight-${SLURM_JOB_ID}.err" >&2)

python "$PROJECT_ROOT/scripts/prepare_harmonized_en_mn5.py" \
    --run-id "$RUN_ID" \
    --required-path-prefix "$DATASET_BASE_ROOT" \
    --model-path "$MODEL_PATH" \
    --github-issue "$GITHUB_ISSUE" \
    --github-pr "$GITHUB_PR"
