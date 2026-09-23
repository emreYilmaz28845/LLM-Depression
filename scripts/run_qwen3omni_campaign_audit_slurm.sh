#!/bin/bash
#SBATCH -J qwen3omni-audit
#SBATCH -A etur92
#SBATCH -q acc_ehpc
#SBATCH -t 00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#SBATCH --chdir=/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression

# Model-free preflight audit for the Qwen3-Omni standalone prompt-context
# campaign: verifies every runtime manifest against the reference campaign's
# recorded identity and audits counts, labels, leakage units and folds.
#
#   PROJECT_ROOT=<deployed code path>   (required)
#   RUNTIME_ROOT=<experiment runtime>   (required)
#   AUDIT_OUTPUT=<json path>            (required)
#
# CPU only: no GPU is requested and no model weights are loaded.

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
    echo "Environment activate script not found: $ENV_ACTIVATE"
    exit 1
fi

PROJECT_ROOT="${PROJECT_ROOT:?Set PROJECT_ROOT to the deployed code path}"
RUNTIME_ROOT="${RUNTIME_ROOT:?Set RUNTIME_ROOT}"
AUDIT_OUTPUT="${AUDIT_OUTPUT:?Set AUDIT_OUTPUT}"
export PROJECT_ROOT
cd "$PROJECT_ROOT"
mkdir -p "$(dirname "$AUDIT_OUTPUT")"

RUN_LOG="$(dirname "$AUDIT_OUTPUT")/preflight-audit-${SLURM_JOB_ID:-local}.log"
exec > >(tee -a "$RUN_LOG") 2>&1

echo "preflight audit | job=${SLURM_JOB_ID:-} | host=$(hostname)"
echo "project_root=$PROJECT_ROOT"
echo "runtime_root=$RUNTIME_ROOT"
echo "audit_output=$AUDIT_OUTPUT"

python scripts/qwen3omni_campaign_preflight.py \
    --runtime-root "$RUNTIME_ROOT" \
    --output "$AUDIT_OUTPUT"

echo "preflight audit complete: $AUDIT_OUTPUT"
