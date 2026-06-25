#!/bin/bash
#SBATCH -J turkish-cv-summary
#SBATCH -A etur92
#SBATCH -q acc_ehpc
#SBATCH -t 00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#SBATCH --chdir=/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression

set -euo pipefail

module purge
module load bsc/1.0
module load miniforge/24.3.0-0

PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
CONFIG="${CONFIG:-$PROJECT_ROOT/configs/turkish_audio_text.yaml}"
RUN_NAME="${RUN_NAME:-turkish_audio_text}"
ENV_ACTIVATE="${ENV_ACTIVATE:-/gpfs/projects/etur92/ozu647717/venvs/qwen_mn5_rebuilt/bin/activate}"
DATASET_BASE_ROOT="${DATASET_BASE_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/Datasets}"
export TURKISH_DATASET_ROOT="${TURKISH_DATASET_ROOT:-$DATASET_BASE_ROOT/Turkish}"
export PROJECT_ROOT

if [ -f "$ENV_ACTIVATE" ]; then
    # shellcheck disable=SC1090
    source "$ENV_ACTIVATE"
else
    echo "Environment activate script not found: $ENV_ACTIVATE"
    exit 1
fi

cd "$PROJECT_ROOT"
LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/logs/slurm_turkish}"
mkdir -p "$LOG_ROOT"
exec > >(tee -a "$LOG_ROOT/summary-${RUN_NAME}-${SLURM_JOB_ID}.out")
exec 2> >(tee -a "$LOG_ROOT/summary-${RUN_NAME}-${SLURM_JOB_ID}.err" >&2)

CONFIG_RUN_ROOT="$(
    python - "$CONFIG" <<'PY'
import sys
from src.utils import load_yaml_with_overrides

print(load_yaml_with_overrides(sys.argv[1], [])["output_dirs"]["run_root"])
PY
)"
RUN_ROOT="$CONFIG_RUN_ROOT/$RUN_NAME"

echo "Summarizing Turkish CV run"
echo "  config: $CONFIG"
echo "  run_name: $RUN_NAME"
echo "  run_root: $RUN_ROOT"
python "$PROJECT_ROOT/src/summarize_runs.py" --run_root "$RUN_ROOT"
echo "Wrote summary: $RUN_ROOT/final_summary.json"
