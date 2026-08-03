#!/usr/bin/env bash
#SBATCH -J daic-comprehensive
#SBATCH -A etur92
#SBATCH -q acc_ehpc
#SBATCH -t 72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH -o /dev/null
#SBATCH -e /dev/null
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
MATRIX_PATH="${MATRIX_PATH:?Set MATRIX_PATH}"
TASK_KIND="${TASK_KIND:?Set TASK_KIND}"
LOG_ROOT="${ARRAY_LOG_ROOT:?Set ARRAY_LOG_ROOT}"
mkdir -p "$LOG_ROOT"
exec > >(tee -a "$LOG_ROOT/${TASK_KIND}-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out")
exec 2> >(tee -a "$LOG_ROOT/${TASK_KIND}-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err" >&2)
python "$PROJECT_ROOT/scripts/run_daic_comprehensive_task.py" \
  --matrix "$MATRIX_PATH" --kind "$TASK_KIND" --index "$SLURM_ARRAY_TASK_ID"
