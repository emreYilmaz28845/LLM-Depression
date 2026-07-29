#!/bin/bash
#SBATCH -J d3tec-hidden-fixed
#SBATCH -A etur92
#SBATCH -q acc_ehpc
#SBATCH -t 24:00:00
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
ENV_ACTIVATE="${ENV_ACTIVATE:-/gpfs/projects/etur92/ozu647717/venvs/qwen_mn5_rebuilt/bin/activate}"
source "$ENV_ACTIVATE"

PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
CACHE_DIR="${CACHE_DIR:?Set CACHE_DIR}"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR}"
VARIANTS="${VARIANTS:?Set VARIANTS}"
RUN_ID="${RUN_ID:?Set RUN_ID}"
QWEN_HIDDEN_DEPS="${QWEN_HIDDEN_DEPS:-$PROJECT_ROOT/.deps/qwen_hidden}"
export PYTHONPATH="$QWEN_HIDDEN_DEPS:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT_ROOT"

LOG_ROOT="$PROJECT_ROOT/logs/slurm_d3tec_hidden/$RUN_ID"
mkdir -p "$LOG_ROOT"
exec > >(tee -a "$LOG_ROOT/fixed-${SLURM_JOB_ID}.out")
exec 2> >(tee -a "$LOG_ROOT/fixed-${SLURM_JOB_ID}.err" >&2)
read -r -a variant_args <<< "$VARIANTS"
python "$PROJECT_ROOT/baselines/qwen_hidden_classifier.py" \
  --cache-dir "$CACHE_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --variants "${variant_args[@]}" \
  --seed 1337
