#!/bin/bash
#SBATCH -J tr-os-screen
#SBATCH -A etur92
#SBATCH -q acc_ehpc
#SBATCH -t 01:00:00
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
source "${ENV_ACTIVATE:-/gpfs/projects/etur92/ozu647717/venvs/qwen_mn5_rebuilt/bin/activate}"

PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
CACHE_DIR="${CACHE_DIR:?Set CACHE_DIR}"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR}"
EXPERIMENT_ID="${EXPERIMENT_ID:-hidden_os_screen}"
INNER_FOLDS="${INNER_FOLDS:-3}"
INNER_SEED="${INNER_SEED:-1337}"
QWEN_HIDDEN_DEPS="${QWEN_HIDDEN_DEPS:-$PROJECT_ROOT/.deps/qwen_hidden}"
LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/logs/slurm_turkish_oversampling/$EXPERIMENT_ID}"
export PYTHONPATH="$QWEN_HIDDEN_DEPS:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT_ROOT"
mkdir -p "$LOG_ROOT"
exec > >(tee -a "$LOG_ROOT/screen-${SLURM_JOB_ID}.out")
exec 2> >(tee -a "$LOG_ROOT/screen-${SLURM_JOB_ID}.err" >&2)

echo "Timestamp: $(date +%Y-%m-%d_%H:%M:%S)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "Hostname: $(hostname)"
echo "Cache Dir: $CACHE_DIR"
echo "Output Dir: $OUTPUT_DIR"
python -V
python -c "import numpy, sklearn, xgboost; print(numpy.__version__, sklearn.__version__, xgboost.__version__)"
python "$PROJECT_ROOT/baselines/qwen_hidden_oversampling_screen.py" \
  --cache-dir "$CACHE_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --experiment-id "$EXPERIMENT_ID" \
  --inner-folds "$INNER_FOLDS" \
  --inner-seed "$INNER_SEED"
