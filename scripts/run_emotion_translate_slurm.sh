#!/bin/bash
#SBATCH -J emotion-translate
#SBATCH -A etur92
#SBATCH -q acc_ehpc
#SBATCH -t 02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#SBATCH --chdir=/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression

# Offline zh->en translation pass over the merged SECap captions. Separate stage
# so a translation failure never forces re-running the expensive SECap extraction.

set -e
set -o pipefail

module purge
module load bsc/1.0
module load miniforge/24.3.0-0

PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
export PROJECT_ROOT
cd "$PROJECT_ROOT"

SECAP_ENV_ACTIVATE="${SECAP_ENV_ACTIVATE:-/gpfs/projects/etur92/ozu647717/venvs/secap_rebuilt/bin/activate}"
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
IN_ZH="${IN_ZH:-$PROJECT_ROOT/outputs/emotion/${DATASET}_secap_zh.jsonl}"
OUT_EN="${OUT_EN:-$PROJECT_ROOT/outputs/emotion/${DATASET}_secap_en.jsonl}"
TRANSLATE_MODEL="${TRANSLATE_MODEL:-facebook/nllb-200-distilled-600M}"
SRC_LANG="${SRC_LANG:-zho_Hans}"
TGT_LANG="${TGT_LANG:-eng_Latn}"
NUM_BEAMS="${NUM_BEAMS:-4}"
BATCH_SIZE="${BATCH_SIZE:-16}"

LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/logs/slurm_emotion/$DATASET}"
mkdir -p "$LOG_ROOT" "$(dirname "$OUT_EN")"
exec > >(tee -a "$LOG_ROOT/translate-${SLURM_JOB_ID}.out")
exec 2> >(tee -a "$LOG_ROOT/translate-${SLURM_JOB_ID}.err" >&2)

echo "========================================"
echo "Translate  dataset=$DATASET  model=$TRANSLATE_MODEL"
echo "in_zh=$IN_ZH  out_en=$OUT_EN"
echo "========================================"
python -V

CMD=(
    python -m src.emotion.translate
    --in "$IN_ZH"
    --out "$OUT_EN"
    --model "$TRANSLATE_MODEL"
    --src-lang "$SRC_LANG"
    --tgt-lang "$TGT_LANG"
    --num-beams "$NUM_BEAMS"
    --batch-size "$BATCH_SIZE"
)
printf 'Launch: '; printf '%q ' "${CMD[@]}"; printf '\n'
PYTHONPATH="$PROJECT_ROOT" "${CMD[@]}"
