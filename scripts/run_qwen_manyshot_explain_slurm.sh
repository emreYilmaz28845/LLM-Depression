#!/usr/bin/env bash
#SBATCH -J qwen-ms-explain
#SBATCH -A etur92
#SBATCH -q acc_ehpc
#SBATCH -p acc
#SBATCH -t 06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:2
# Qwen3.8-27B many-shot classification with a generated explanation
# (experiments/manyshot_qwen_explain/run_qwen_manyshot_explain.py).
# The model is split layer-wise over 2 H100 (device_map=auto); one GPU computes at a time, so more GPUs
# per job would only add cost. Prompts reach ~255k tokens (DAIC, full transcripts); the KV cache lives
# only in the 16 full-attention layers (~64 KB/token), so prefix + working copies fit in the rest.
#
# Required: PROJECT_ROOT (deployed code), DATASET (daic|turkish), MANIFEST, OUT_DIR.
# Optional: SHARD ("i/n"), EXTRA_ARGS (e.g. "--limit 3 --check-uncached 2 --check-determinism 1").
set -euo pipefail
module purge
module load bsc/1.0
BASE=/gpfs/projects/etur92/ozu647717
PROJECT_ROOT="${PROJECT_ROOT:?Set PROJECT_ROOT}"
OUT_DIR="${OUT_DIR:?Set OUT_DIR}"
MODEL_DIR="${MODEL_DIR:-$BASE/models/Qwen3.8-27B}"
source "${ENV_ACTIVATE:-$BASE/venvs/gemma4_12b_tf5_14_1/bin/activate}"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
cd "$PROJECT_ROOT"
mkdir -p "$OUT_DIR"
if [ -n "${REQUIRE_SMOKE_DIR:-}" ]; then
  # Gate for jobs queued behind the smoke (afterok only sees the exit code): stop unless every smoke
  # answer has a valid label, finished before max_new_tokens, and the determinism check held.
  python experiments/manyshot_qwen_explain/check_smoke.py "$REQUIRE_SMOKE_DIR" --min-targets "${SMOKE_MIN_TARGETS:-3}"
fi
python -c 'import torch, transformers; print("versions", torch.__version__, transformers.__version__)'
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

ARGS=(--dataset "${DATASET:?Set DATASET}" --manifest "${MANIFEST:?Set MANIFEST}" --model-dir "$MODEL_DIR"
      --out-dir "$OUT_DIR")
[ -n "${SHARD:-}" ] && ARGS+=(--shard "$SHARD")
# shellcheck disable=SC2206
[ -n "${EXTRA_ARGS:-}" ] && ARGS+=($EXTRA_ARGS)
python experiments/manyshot_qwen_explain/run_qwen_manyshot_explain.py "${ARGS[@]}"
nvidia-smi --query-gpu=memory.used --format=csv,noheader || true
