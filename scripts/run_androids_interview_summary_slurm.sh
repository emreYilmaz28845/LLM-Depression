#!/bin/bash
#SBATCH -J androids-summary
#SBATCH -A etur92
#SBATCH -q acc_ehpc
#SBATCH -t 00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -o /dev/null
#SBATCH -e /dev/null

set -euo pipefail
module purge
module load bsc/1.0
module load miniforge/24.3.0-0
PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT is required}"
ENV_ACTIVATE="${ENV_ACTIVATE:?ENV_ACTIVATE is required}"
RUN_ROOT="${RUN_ROOT:?RUN_ROOT is required}"
# shellcheck disable=SC1090
source "$ENV_ACTIVATE"
cd "$PROJECT_ROOT"
python "$PROJECT_ROOT/src/summarize_runs.py" --run_root "$RUN_ROOT"
