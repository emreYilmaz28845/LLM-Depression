#!/bin/bash
#SBATCH -J daic-odv-contract
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
ENV_ACTIVATE="${ENV_ACTIVATE:?ENV_ACTIVATE is required}"
BACKBONE="${BACKBONE:?BACKBONE is required (qwen|gemma4)}"
MODALITY="${MODALITY:?MODALITY is required}"
MODEL_PATH="${MODEL_PATH:?MODEL_PATH is required}"
ADAPTER_PATH="${ADAPTER_PATH:?ADAPTER_PATH is required}"
OUTPUT="${OUTPUT:?OUTPUT is required}"

if [ ! -f "$ENV_ACTIVATE" ]; then
    echo "Environment activate script not found: $ENV_ACTIVATE" >&2
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

LOG_ROOT="$PROJECT_ROOT/logs/daic_officialdev_smokes"
mkdir -p "$LOG_ROOT"
exec > >(tee -a "$LOG_ROOT/contract-${SLURM_JOB_ID}.out")
exec 2> >(tee -a "$LOG_ROOT/contract-${SLURM_JOB_ID}.err" >&2)

if [ "$BACKBONE" = "gemma4" ]; then
    CONTRACT_SCRIPT="$PROJECT_ROOT/scripts/smoke_gemma4_hidden_contract.py"
else
    CONTRACT_SCRIPT="$PROJECT_ROOT/scripts/smoke_qwen_hidden_contract.py"
fi
python "$CONTRACT_SCRIPT" \
    --model-path "$MODEL_PATH" \
    --adapter-path "$ADAPTER_PATH" \
    --modality "$MODALITY" \
    --output "$OUTPUT"
