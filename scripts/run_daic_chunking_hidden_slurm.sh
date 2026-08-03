#!/usr/bin/env bash
#SBATCH -J daic-k-hidden
#SBATCH -A etur92
#SBATCH -q acc_ehpc
#SBATCH -t 24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#SBATCH --chdir=/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression
set -euo pipefail
module purge
module load bsc/1.0
module load miniforge/24.3.0-0
PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
source "${ENV_ACTIVATE:-/gpfs/projects/etur92/ozu647717/venvs/qwen_mn5_rebuilt/bin/activate}"
export PYTHONPATH="$PROJECT_ROOT/.deps/qwen_hidden:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT_ROOT"

CHECKPOINT_DIR="${CHECKPOINT_DIR:?Set CHECKPOINT_DIR}"
CACHE_ROOT="${CACHE_ROOT:?Set CACHE_ROOT}"
STRATEGY="${STRATEGY:?Set STRATEGY to joint, rotary, or all}"
PROTOCOL_ID="${PROTOCOL_ID:-}"
EVALUATION_VIEWS="${EVALUATION_VIEWS:-}"
LOG_ROOT="${LOG_ROOT:?Set unique LOG_ROOT}"
mkdir -p "$LOG_ROOT"
exec > >(tee -a "$LOG_ROOT/hidden-${SLURM_JOB_ID}.out")
exec 2> >(tee -a "$LOG_ROOT/hidden-${SLURM_JOB_ID}.err" >&2)

extract() {
  local view="$1"
  shift
  python "$PROJECT_ROOT/src/features/extract_qwen_hidden.py" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --output-dir "$CACHE_ROOT/$view" \
    --condition "daic_chunking_${STRATEGY}_${view}" \
    "$@"
}

if [ -n "$PROTOCOL_ID" ]; then
  IFS=',' read -r -a views <<< "$EVALUATION_VIEWS"
  for view in "${views[@]}"; do
    case "$view" in
      fixed4) extract fixed4 --eval-chunk-policy fixed_k --eval-chunks-per-subject 4 ;;
      mincover4) extract mincover4 --eval-chunk-policy balanced_joint_cover --eval-chunks-per-subject 4 ;;
      fixed15) extract fixed15 --eval-chunk-policy fixed_count_balanced_joint_cover --eval-chunks-per-subject 4 --eval-bundles-per-subject 15 ;;
      all) extract all --eval-chunk-policy all --eval-chunks-per-subject all ;;
      matched10_even) extract matched10_even --eval-chunk-policy matched_k --eval-chunks-per-subject 10 ;;
      matched10_resampled) ;;
      *) echo "Unsupported comprehensive hidden view: $view" >&2; exit 2 ;;
    esac
  done
elif [ "$STRATEGY" = "joint" ]; then
  extract c1_fixed --eval-chunk-policy fixed_k --eval-chunks-per-subject 4
  extract c2_balanced --eval-chunk-policy balanced_joint_cover --eval-chunks-per-subject 4
else
  extract "$STRATEGY"
fi
