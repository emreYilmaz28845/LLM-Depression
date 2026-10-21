#!/bin/bash
#SBATCH -J audio-attn-probe
#SBATCH -A etur92
#SBATCH -q acc_ehpc
#SBATCH -t 01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#SBATCH --chdir=/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression

# Audio attention-mass / ablation / prosody probe for one Qwen3-Omni pair of
# checkpoints (audio+text and audio-only, same dataset and fold). The Thinker is
# device-map sharded over the 4 GPUs, the same shape as standalone evaluation.
# Submit with sbatch --chdir=<deployment code dir> and --export with
# AUDIO_TEXT_FOLD_DIR, AUDIO_ONLY_FOLD_DIR, OUTPUT_DIR and LOG_ROOT.

set -e
set -o pipefail
export PYTHONDONTWRITEBYTECODE=1

module purge
module load bsc/1.0
module load miniforge/24.3.0-0

ENV_ACTIVATE="${ENV_ACTIVATE:-/gpfs/projects/etur92/ozu647717/venvs/qwen3omni/bin/activate}"
# shellcheck disable=SC1090
source "$ENV_ACTIVATE"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
export PROJECT_ROOT
cd "$PROJECT_ROOT"

AUDIO_TEXT_FOLD_DIR="${AUDIO_TEXT_FOLD_DIR:?Set AUDIO_TEXT_FOLD_DIR}"
AUDIO_ONLY_FOLD_DIR="${AUDIO_ONLY_FOLD_DIR:?Set AUDIO_ONLY_FOLD_DIR}"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR}"
LOG_ROOT="${LOG_ROOT:?Set LOG_ROOT}"
mkdir -p "$LOG_ROOT"
LOG_FILE="$LOG_ROOT/audio_attention_probe_${SLURM_JOB_ID:-local}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "job=${SLURM_JOB_ID:-} host=$(hostname) project_root=$PROJECT_ROOT"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

python scripts/probe_audio_attention.py \
  --audio-text-fold-dir "$AUDIO_TEXT_FOLD_DIR" \
  --audio-only-fold-dir "$AUDIO_ONLY_FOLD_DIR" \
  --output-dir "$OUTPUT_DIR"
