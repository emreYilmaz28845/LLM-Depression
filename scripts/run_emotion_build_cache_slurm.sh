#!/bin/bash
#SBATCH -J emotion-build-cache
#SBATCH -A etur92
#SBATCH -q acc_ehpc
#SBATCH -t 00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#SBATCH --chdir=/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression

# Pure JSONL/text orchestration: merge SECap shard files (MODE=merge) or validate
# the final translated cache against the manifest (MODE=validate). No GPU, no model.

set -e
set -o pipefail

module purge
module load bsc/1.0
module load miniforge/24.3.0-0

PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
export PROJECT_ROOT
cd "$PROJECT_ROOT"

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
    fi
fi

DATASET="${DATASET:-daic}"
MODE="${MODE:-validate}"
MANIFEST="${MANIFEST:-$PROJECT_ROOT/outputs/manifests/${DATASET}_manifest.jsonl}"
OUT_ZH="${OUT_ZH:-$PROJECT_ROOT/outputs/emotion/${DATASET}_secap_zh.jsonl}"
OUT_EN="${OUT_EN:-$PROJECT_ROOT/outputs/emotion/${DATASET}_secap_en.jsonl}"
MERGE_PATTERN="${MERGE_PATTERN:-$PROJECT_ROOT/outputs/emotion/${DATASET}_secap_zh.shard*of*.jsonl}"
STRICT="${STRICT:-0}"

LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/logs/slurm_emotion/$DATASET}"
mkdir -p "$LOG_ROOT" "$(dirname "$OUT_EN")"
exec > >(tee -a "$LOG_ROOT/build-${MODE}-${SLURM_JOB_ID}.out")
exec 2> >(tee -a "$LOG_ROOT/build-${MODE}-${SLURM_JOB_ID}.err" >&2)

echo "build_emotion_cache  dataset=$DATASET  mode=$MODE"
python -V

if [ "$MODE" = "merge" ]; then
    CMD=(python -m src.emotion.build_emotion_cache merge --pattern "$MERGE_PATTERN" --out "$OUT_ZH")
elif [ "$MODE" = "validate" ]; then
    CMD=(python -m src.emotion.build_emotion_cache validate --cache "$OUT_EN" --manifest "$MANIFEST")
    if [ "$STRICT" = "1" ]; then CMD+=(--strict); fi
else
    echo "Unknown MODE=$MODE (expected merge|validate)."
    exit 1
fi

printf 'Launch: '; printf '%q ' "${CMD[@]}"; printf '\n'
PYTHONPATH="$PROJECT_ROOT" "${CMD[@]}"
