#!/bin/bash
#SBATCH -J emotion-qwen2audio
#SBATCH -A etur92
#SBATCH -q acc_ehpc
#SBATCH -t 12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#SBATCH --chdir=/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression

# Offline Qwen2-Audio en-caption extraction (one shard per array task). Runs in the
# main Qwen env (qwen_mn5_rebuilt). Deterministic decode, but still done ONCE
# offline and frozen; training/eval never generate emotion captions dynamically.
# Unlike SECap, no translation pass is needed: emotion_en is populated directly.

set -e
set -o pipefail

module purge
module load bsc/1.0
module load miniforge/24.3.0-0

PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
export PROJECT_ROOT
cd "$PROJECT_ROOT"

# --- Activate the Qwen env (same env used for training/eval) ------------------
ENV_ACTIVATE="${ENV_ACTIVATE:-/gpfs/projects/etur92/ozu647717/venvs/qwen_mn5_rebuilt/bin/activate}"
if [ -f "$ENV_ACTIVATE" ]; then
    # shellcheck disable=SC1090
    source "$ENV_ACTIVATE"
else
    echo "Environment activate script not found: $ENV_ACTIVATE"
    exit 1
fi

DATASET="${DATASET:-daic}"
MANIFEST="${MANIFEST:-$PROJECT_ROOT/outputs/manifests/${DATASET}_manifest.jsonl}"
QWEN_MODEL="${QWEN_MODEL:-/gpfs/projects/etur92/ozu647717/models/Qwen2-Audio-7B-Instruct}"
OUT_EN="${OUT_EN:-$PROJECT_ROOT/outputs/emotion/${DATASET}_qwen2audio_en.jsonl}"
SHARDS="${SHARDS:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-48}"
LIMIT="${LIMIT:-0}"

# Shard index from the SLURM array task id (defaults to 0 for a non-array run).
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-0}"

LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/logs/slurm_emotion_qwen2audio/$DATASET}"
mkdir -p "$LOG_ROOT" "$(dirname "$OUT_EN")"
SLURM_TAG="${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SHARD_INDEX}"
exec > >(tee -a "$LOG_ROOT/extract-${SLURM_TAG}.out")
exec 2> >(tee -a "$LOG_ROOT/extract-${SLURM_TAG}.err" >&2)

echo "========================================"
echo "Qwen2-Audio extraction  dataset=$DATASET  shard=$SHARD_INDEX/$SHARDS"
echo "manifest=$MANIFEST"
echo "model=$QWEN_MODEL"
echo "out_en=$OUT_EN  max_new_tokens=$MAX_NEW_TOKENS  limit=$LIMIT"
echo "hostname=$(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "========================================"
nvidia-smi || true
python -V

CMD=(
    python -m src.emotion.extract_qwen2audio
    --manifest "$MANIFEST"
    --model "$QWEN_MODEL"
    --out "$OUT_EN"
    --shard "${SHARD_INDEX}/${SHARDS}"
    --max-new-tokens "$MAX_NEW_TOKENS"
)
if [ "$LIMIT" != "0" ]; then CMD+=(--limit "$LIMIT"); fi

printf 'Launch: '; printf '%q ' "${CMD[@]}"; printf '\n'
PYTHONPATH="$PROJECT_ROOT" "${CMD[@]}"

# Merge shards (if any) into the canonical cache once all array tasks finish.
# A single non-sharded run already writes OUT_EN directly, so only merge when
# SHARDS>1 and we are the last array task to complete is non-trivial in pure
# bash; instead, run the merge as a separate dependent step (see submit wrapper)
# or invoke build_emotion_cache merge manually:
#   python -m src.emotion.build_emotion_cache merge \
#     --pattern "$PROJECT_ROOT/outputs/emotion/${DATASET}_qwen2audio_en.shard*of*.jsonl" \
#     --out "$OUT_EN"
