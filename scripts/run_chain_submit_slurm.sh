#!/bin/bash
#SBATCH -J llm-depression-chain
#SBATCH -A etur92
#SBATCH -q acc_ehpc
#SBATCH -t 00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#SBATCH --chdir=/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
CHAIN_SCRIPT="${CHAIN_SCRIPT:-$PROJECT_ROOT/scripts/submit_cv_then_fulltrain.sh}"

if [ ! -f "$CHAIN_SCRIPT" ]; then
    echo "Chain script not found: $CHAIN_SCRIPT"
    exit 1
fi

exec bash "$CHAIN_SCRIPT"
