#!/usr/bin/env bash
#SBATCH -J daic-k-head
#SBATCH -A etur92
#SBATCH -q acc_ehpc
#SBATCH -t 01:00:00
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
source "${ENV_ACTIVATE:-/gpfs/projects/etur92/ozu647717/venvs/qwen_mn5_rebuilt/bin/activate}"
export PYTHONPATH="$PROJECT_ROOT/.deps/qwen_hidden:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT_ROOT"

CACHE_ROOT="${CACHE_ROOT:?Set CACHE_ROOT}"
OUTPUT_ROOT="${OUTPUT_ROOT:?Set OUTPUT_ROOT}"
STRATEGY="${STRATEGY:?Set STRATEGY}"
PROTOCOL_ID="${PROTOCOL_ID:-}"
EVALUATION_VIEWS="${EVALUATION_VIEWS:-}"
VARIANT="${VARIANT:?Set VARIANT}"
LOG_ROOT="${LOG_ROOT:?Set unique LOG_ROOT}"
mkdir -p "$LOG_ROOT"
exec > >(tee -a "$LOG_ROOT/classical-${STRATEGY}-${VARIANT}-${SLURM_JOB_ID}.out")
exec 2> >(tee -a "$LOG_ROOT/classical-${STRATEGY}-${VARIANT}-${SLURM_JOB_ID}.err" >&2)

run_head() {
  python "$PROJECT_ROOT/baselines/daic_chunking_heads.py" "$@"
}

if [ -n "$PROTOCOL_ID" ]; then
  IFS=',' read -r -a views <<< "$EVALUATION_VIEWS"
  fit_view="${views[0]}"
  for view in "${views[@]}"; do
    if [ "$view" = fixed4 ] || [ "$view" = all ]; then fit_view="$view"; break; fi
  done
  FIT_OUT="$OUTPUT_ROOT/$fit_view/$VARIANT"
  run_head --fit-cache "$CACHE_ROOT/$fit_view" --eval-cache "$CACHE_ROOT/$fit_view" \
    --output-dir "$FIT_OUT" --variant "$VARIANT" --seed 1337
  for view in "${views[@]}"; do
    [ "$view" = "$fit_view" ] && continue
    [ "$view" = matched10_resampled ] && continue
    run_head --fit-cache "$CACHE_ROOT/$fit_view" --eval-cache "$CACHE_ROOT/$view" \
      --output-dir "$OUTPUT_ROOT/$view/$VARIANT" --variant "$VARIANT" --seed 1337 \
      --fitted-model-dir "$FIT_OUT"
  done
elif [ "$STRATEGY" = "joint" ]; then
  C1_OUT="$OUTPUT_ROOT/c1/$VARIANT"
  run_head --fit-cache "$CACHE_ROOT/c1_fixed" --eval-cache "$CACHE_ROOT/c1_fixed" \
    --output-dir "$C1_OUT" --variant "$VARIANT" --seed 1337
  run_head --fit-cache "$CACHE_ROOT/c1_fixed" --eval-cache "$CACHE_ROOT/c2_balanced" \
    --output-dir "$OUTPUT_ROOT/c2/$VARIANT" --variant "$VARIANT" --seed 1337 \
    --fitted-model-dir "$C1_OUT"
else
  condition=c3
  [ "$STRATEGY" = "all" ] && condition=c4
  run_head --fit-cache "$CACHE_ROOT/$STRATEGY" --eval-cache "$CACHE_ROOT/$STRATEGY" \
    --output-dir "$OUTPUT_ROOT/$condition/$VARIANT" --variant "$VARIANT" --seed 1337
fi
