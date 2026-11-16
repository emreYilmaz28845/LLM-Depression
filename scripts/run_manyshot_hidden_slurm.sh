#!/usr/bin/env bash
#SBATCH -J manyshot-hidden
#SBATCH -A etur92
#SBATCH -q acc_ehpc
#SBATCH -p acc
#SBATCH -t 04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:2
# Many-shot-context Qwen3.8-27B text features (src/features/extract_manyshot_hidden.py).
# The model is split layer-wise over 2 H100 (device_map=auto): one GPU computes at a time,
# so more GPUs would only add cost. Prefix caches of up to ~170k tokens fit in the rest.
#
# Required: PROJECT_ROOT (deployed code), ROUTES (json, relative to PROJECT_ROOT or absolute), OUT_ROOT.
# Optional: MODE (manyshot|zeroshot), FOLDS ("0 1 ..."), EXTRA_ARGS, SMOKE=1 (androids fold-0 checks).
set -euo pipefail
module purge
module load bsc/1.0
BASE=/gpfs/projects/etur92/ozu647717
PROJECT_ROOT="${PROJECT_ROOT:?Set PROJECT_ROOT}"
OUT_ROOT="${OUT_ROOT:?Set OUT_ROOT}"
MODEL_DIR="${MODEL_DIR:-$BASE/models/Qwen3.8-27B}"
source "${ENV_ACTIVATE:-$BASE/venvs/gemma4_12b_tf5_14_1/bin/activate}"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
cd "$PROJECT_ROOT"
mkdir -p "$OUT_ROOT"
python -c 'import torch, transformers; print("versions", torch.__version__, transformers.__version__)'
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

run() { python src/features/extract_manyshot_hidden.py --model-dir "$MODEL_DIR" --output-root "$OUT_ROOT" "$@"; }

if [ "${SMOKE:-0}" = "1" ]; then
  # 1) prompt/token plan of every native route (CPU), 2) zero-shot arm reproduces the frozen
  # Qwen3.8 cache, 3) many-shot fold 0 with cache-free and determinism checks.
  run --routes experiments/manyshot/routes_native.json --plan-only
  run --routes experiments/manyshot/routes_smoke_androids.json --mode zeroshot --folds 0 \
      --output-root "$OUT_ROOT/zeroshot" \
      --compare-cache "$BASE/AudioLLM/experiment_runtime/exp-frozen-base-standalone-heads-20261024/caches/androids_interview/text_only_native/main"
  run --routes experiments/manyshot/routes_smoke_androids.json --mode manyshot --folds 0 \
      --output-root "$OUT_ROOT/manyshot" --check-uncached 3
  exit 0
fi

ARGS=(--routes "${ROUTES:?Set ROUTES}" --mode "${MODE:-manyshot}")
[ -n "${FOLDS:-}" ] && ARGS+=(--folds $FOLDS)
# shellcheck disable=SC2206
[ -n "${EXTRA_ARGS:-}" ] && ARGS+=($EXTRA_ARGS)
run "${ARGS[@]}" --output-root "$OUT_ROOT/${MODE:-manyshot}"
