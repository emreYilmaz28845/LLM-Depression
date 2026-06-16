#!/bin/bash
#SBATCH -J emotion-extract
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

# Offline SECap zh-caption extraction (one shard per array task). Runs in the
# SECap conda env (Chinese HuBERT + Chinese-LLaMA stack). Non-deterministic, so
# this is done ONCE offline and frozen; training/eval never call SECap.

set -e
set -o pipefail

module purge
module load bsc/1.0
module load miniforge/24.3.0-0

PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
export PROJECT_ROOT
cd "$PROJECT_ROOT"

# --- Activate the SECap env (conda name or a venv activate script) -----------
SECAP_ENV_ACTIVATE="${SECAP_ENV_ACTIVATE:-}"
SECAP_CONDA_ENV="${SECAP_CONDA_ENV:-secap}"
if [ -n "$SECAP_ENV_ACTIVATE" ] && [ -f "$SECAP_ENV_ACTIVATE" ]; then
    # shellcheck disable=SC1090
    source "$SECAP_ENV_ACTIVATE"
else
    CONDA_BASE="$(conda info --base 2>/dev/null || true)"
    if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
        # shellcheck disable=SC1091
        source "$CONDA_BASE/etc/profile.d/conda.sh"
        conda activate "$SECAP_CONDA_ENV"
    else
        echo "Could not activate SECap env (set SECAP_ENV_ACTIVATE or SECAP_CONDA_ENV)."
        exit 1
    fi
fi

DATASET="${DATASET:-daic}"
MANIFEST="${MANIFEST:-$PROJECT_ROOT/outputs/manifests/${DATASET}_manifest.jsonl}"
SECAP_ROOT="${SECAP_ROOT:-/gpfs/projects/etur92/SECap}"
CKPT="${CKPT:-$SECAP_ROOT/model.ckpt}"
OUT_ZH="${OUT_ZH:-$PROJECT_ROOT/outputs/emotion/${DATASET}_secap_zh.jsonl}"
SHARDS="${SHARDS:-1}"
FAST="${FAST:-0}"
LIMIT="${LIMIT:-0}"

# Shard index from the SLURM array task id (defaults to 0 for a non-array run).
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-0}"

LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/logs/slurm_emotion/$DATASET}"
mkdir -p "$LOG_ROOT" "$(dirname "$OUT_ZH")"
SLURM_TAG="${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SHARD_INDEX}"
exec > >(tee -a "$LOG_ROOT/extract-${SLURM_TAG}.out")
exec 2> >(tee -a "$LOG_ROOT/extract-${SLURM_TAG}.err" >&2)

echo "========================================"
echo "SECap extraction  dataset=$DATASET  shard=$SHARD_INDEX/$SHARDS"
echo "manifest=$MANIFEST"
echo "secap_root=$SECAP_ROOT  ckpt=$CKPT"
echo "out_zh=$OUT_ZH  fast=$FAST  limit=$LIMIT"
echo "hostname=$(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "========================================"
nvidia-smi || true
python -V

CMD=(
    python -m src.emotion.extract_secap
    --manifest "$MANIFEST"
    --secap-root "$SECAP_ROOT"
    --ckpt "$CKPT"
    --out "$OUT_ZH"
    --shard "${SHARD_INDEX}/${SHARDS}"
)
if [ "$FAST" = "1" ]; then CMD+=(--fast); fi
if [ "$LIMIT" != "0" ]; then CMD+=(--limit "$LIMIT"); fi

printf 'Launch: '; printf '%q ' "${CMD[@]}"; printf '\n'
PYTHONPATH="$PROJECT_ROOT" "${CMD[@]}"
