#!/usr/bin/env bash
#SBATCH -J daic-non-k
#SBATCH -A etur92
#SBATCH -q acc_ehpc
#SBATCH -t 72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH -o /dev/null
#SBATCH -e /dev/null
set -euo pipefail
module purge
module load bsc/1.0
module load miniforge/24.3.0-0
PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
source "${ENV_ACTIVATE:-/gpfs/projects/etur92/ozu647717/venvs/qwen_mn5_rebuilt/bin/activate}"
cd "$PROJECT_ROOT"
MATRIX_PATH="${MATRIX_PATH:?Set MATRIX_PATH}"
TASK_KIND="${TASK_KIND:?Set TASK_KIND}"
ARRAY_LOG_ROOT="${ARRAY_LOG_ROOT:?Set ARRAY_LOG_ROOT}"
mkdir -p "$ARRAY_LOG_ROOT"
exec > >(tee -a "$ARRAY_LOG_ROOT/${TASK_KIND}-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out")
exec 2> >(tee -a "$ARRAY_LOG_ROOT/${TASK_KIND}-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err" >&2)
python scripts/run_daic_non_k_task.py --matrix "$MATRIX_PATH" --kind "$TASK_KIND" --index "$SLURM_ARRAY_TASK_ID"
