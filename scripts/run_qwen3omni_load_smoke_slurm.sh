#!/bin/bash
#SBATCH -J qwen3omni-load-smoke
#SBATCH -A etur92
#SBATCH -q acc_ehpc
#SBATCH -t 01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#SBATCH --chdir=/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression
#
# Load smoke for the Qwen3-Omni environment and the staged 30B-A3B snapshot.
# One process, four GPUs, weights sharded by accelerate device_map. No training,
# no FSDP: this only proves the environment, the weights, the processor, the
# repository collator, and the teacher-forced scoring path work on real H100s.
#
#   PROJECT_ROOT=/gpfs/.../LLM-Depression sbatch --chdir="$PROJECT_ROOT" \
#     scripts/run_qwen3omni_load_smoke_slurm.sh
#
# The four-GPU shape exists because ~66 GB of BF16 weights do not fit one 64 GB
# H100. It is a smoke shape only and does not change any recipe's job shape.

set -e
set -o pipefail

module purge
module load bsc/1.0
module load miniforge/24.3.0-0

ENV_ACTIVATE="${ENV_ACTIVATE:-/gpfs/projects/etur92/ozu647717/venvs/qwen3omni/bin/activate}"
if [ -f "$ENV_ACTIVATE" ]; then
    # shellcheck disable=SC1090
    source "$ENV_ACTIVATE"
else
    echo "Environment activate script not found: $ENV_ACTIVATE" >&2
    exit 1
fi

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
MODEL_DIR="${QWEN3_OMNI_MODEL_DIR:-/gpfs/projects/etur92/ozu647717/models/Qwen3-Omni-30B-A3B-Instruct}"
LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/logs/qwen3omni_smoke}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/qwen3omni_smoke}"

mkdir -p "$LOG_ROOT" "$OUTPUT_DIR"
LOG_FILE="$LOG_ROOT/${SLURM_JOB_NAME:-qwen3omni-load-smoke}_${SLURM_JOB_ID:-local}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

cd "$PROJECT_ROOT"
echo "job=${SLURM_JOB_ID:-local} host=$(hostname) start=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "env_activate=$ENV_ACTIVATE"
echo "model_dir=$MODEL_DIR"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
python -V
python -c "import torch, transformers, peft, accelerate; print('torch', torch.__version__, '| transformers', transformers.__version__, '| peft', peft.__version__, '| accelerate', accelerate.__version__)"

python scripts/smoke_qwen3omni_mn5_load.py \
    --model-dir "$MODEL_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --revision-note "snapshot staged 2026-09-20, 70,530,599,670 bytes"

echo "end=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "log=$LOG_FILE"
