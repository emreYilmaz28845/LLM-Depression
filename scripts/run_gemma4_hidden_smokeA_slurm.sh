#!/bin/bash
#SBATCH -J gemma4-hid-smokeA
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

set -euo pipefail
module purge
module load bsc/1.0
module load miniforge/24.3.0-0

PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
ENV_ACTIVATE="${ENV_ACTIVATE:-/gpfs/projects/etur92/ozu647717/venvs/gemma4_12b_tf5_14_1/bin/activate}"
MODEL_PATH="${MODEL_PATH:?MODEL_PATH is required}"
ADAPTER_PATH="${ADAPTER_PATH:?ADAPTER_PATH is required}"
MODALITY="${MODALITY:?MODALITY is required}"
OUTPUT="${OUTPUT:?OUTPUT is required}"

if [ ! -f "$ENV_ACTIVATE" ]; then
    echo "Gemma environment not found: $ENV_ACTIVATE" >&2
    exit 1
fi
# shellcheck disable=SC1090
source "$ENV_ACTIVATE"
cd "$PROJECT_ROOT"
export PROJECT_ROOT

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

python scripts/smoke_gemma4_hidden_contract.py \
    --model-path "$MODEL_PATH" \
    --adapter-path "$ADAPTER_PATH" \
    --modality "$MODALITY" \
    --output "$OUTPUT"
