#!/bin/bash
#SBATCH -J p30-heads
#SBATCH -A etur92
#SBATCH -q acc_ehpc
#SBATCH -t 02:00:00
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

CACHE_DIR="${CACHE_DIR:?Set CACHE_DIR}"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR}"
SEED="${SEED:-1337}"
QWEN_HIDDEN_DEPS="${QWEN_HIDDEN_DEPS:-$PROJECT_ROOT/.deps/qwen_hidden}"
export PYTHONPATH="$QWEN_HIDDEN_DEPS:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/logs/slurm_daic_participant_packed30}"
mkdir -p "$LOG_ROOT"
exec > >(tee -a "$LOG_ROOT/heads-${SLURM_JOB_ID}.out")
exec 2> >(tee -a "$LOG_ROOT/heads-${SLURM_JOB_ID}.err" >&2)

python - <<'PY'
import platform, importlib.metadata as md
required = {"python": "3.10.14", "scikit-learn": "1.7.0", "xgboost": "2.1.4"}
actual = {
    "python": platform.python_version(),
    "scikit-learn": md.version("scikit-learn"),
    "xgboost": md.version("xgboost"),
}
print("versions", actual)
if actual != required:
    raise SystemExit(f"STOP: locked hidden/head dependency versions required {required}, got {actual}")
PY

# Exactly the two locked raw-feature heads; no PCA/Optuna/calibration variants.
CMD=(
  python "$PROJECT_ROOT/baselines/qwen_hidden_classifier.py"
  --cache-dir "$CACHE_DIR"
  --output-dir "$OUTPUT_DIR"
  --variants logreg_raw xgb_raw
  --seed "$SEED"
)
echo "== packed30 fixed heads =="
echo "cache=$CACHE_DIR output=$OUTPUT_DIR seed=$SEED"
printf 'Launch: '; printf '%q ' "${CMD[@]}"; printf '\n'
"${CMD[@]}"
echo "== packed30 fixed heads finished =="
