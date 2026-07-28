#!/bin/bash
#SBATCH -J d3tec-audit
#SBATCH -A etur92
#SBATCH -q acc_ehpc
#SBATCH -t 01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -o /dev/null
#SBATCH -e /dev/null

set -euo pipefail
module purge
module load bsc/1.0
module load miniforge/24.3.0-0
PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT is required}"
ENV_ACTIVATE="${ENV_ACTIVATE:?ENV_ACTIVATE is required}"
RUN_SPECS="${RUN_SPECS:?RUN_SPECS is required}"
AUDIT_OUT="${AUDIT_OUT:?AUDIT_OUT is required}"
# shellcheck disable=SC1090
source "$ENV_ACTIVATE"
cd "$PROJECT_ROOT"
CMD=(python "$PROJECT_ROOT/scripts/audit_d3tec_matrix.py" --out "$AUDIT_OUT")
IFS=';' read -r -a specs <<< "$RUN_SPECS"
for spec in "${specs[@]}"; do
    CMD+=(--run "$spec")
done
"${CMD[@]}"
