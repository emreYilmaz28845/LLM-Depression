#!/bin/bash
#SBATCH -J jk4-audit
#SBATCH -A etur92
#SBATCH -q acc_ehpc
#SBATCH -t 01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
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
export PROJECT_ROOT
cd "$PROJECT_ROOT"

RUN_ROOT="${RUN_ROOT:?Set RUN_ROOT}"
MANIFEST_DIR="${MANIFEST_DIR:?Set MANIFEST_DIR}"
SPLIT_DIR="${SPLIT_DIR:?Set SPLIT_DIR}"
SMOKE="${SMOKE:-0}"

LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/logs/slurm_daic_participant_packed30_jointk4}"
mkdir -p "$LOG_ROOT"
exec > >(tee -a "$LOG_ROOT/audit-${SLURM_JOB_ID}.out")
exec 2> >(tee -a "$LOG_ROOT/audit-${SLURM_JOB_ID}.err" >&2)

CMD=(
  python "$PROJECT_ROOT/scripts/audit_daic_participant_packed30_jointk4.py"
  --run-root "$RUN_ROOT"
  --manifest-dir "$MANIFEST_DIR"
  --split-dir "$SPLIT_DIR"
)
if [ "$SMOKE" = "1" ]; then
  CMD+=(--smoke)
fi

echo "== packed30 jointk4 artifact audit =="
printf 'Launch: '; printf '%q ' "${CMD[@]}"; printf '\n'
"${CMD[@]}"
echo "== packed30 jointk4 artifact audit finished =="
