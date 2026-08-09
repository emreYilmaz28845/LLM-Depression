#!/usr/bin/env bash
#SBATCH -J androids-qctx
#SBATCH -A etur92
#SBATCH -q acc_ehpc
#SBATCH -t 00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
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
BASE_ENV_ACTIVATE="${BASE_ENV_ACTIVATE:-/gpfs/projects/etur92/ozu647717/venvs/qwen_mn5_rebuilt/bin/activate}"
ASR_OVERLAY="${ASR_OVERLAY:-/gpfs/projects/etur92/ozu647717/AudioLLM/venvs/qwen3asr_overlay}"
DATASET_ROOT="${ANDROID_DATASET_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/Datasets/Androids-Corpus/Androids-Corpus}"
MODEL_PATH="${QWEN3_ASR_MODEL_PATH:-/gpfs/projects/etur92/ozu647717/models/Qwen3-ASR-1.7B}"
RUN_ID="${RUN_ID:?Set RUN_ID}"
LIMIT="${LIMIT:-}"
RESUME="${RESUME:-0}"
OVERWRITE="${OVERWRITE:-0}"
BATCH_SIZE="${BATCH_SIZE:-16}"

source "$BASE_ENV_ACTIVATE"
export PYTHONPATH="$ASR_OVERLAY:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT_ROOT"

OUTPUT_DIR="$PROJECT_ROOT/outputs/androids_question_recovery/$RUN_ID"
LOG_ROOT="$PROJECT_ROOT/logs/slurm_androids_question_recovery/$RUN_ID"
mkdir -p "$OUTPUT_DIR" "$LOG_ROOT"

STDOUT_FILE="$LOG_ROOT/recover-${SLURM_JOB_ID}.out"
STDERR_FILE="$LOG_ROOT/recover-${SLURM_JOB_ID}.err"
exec > >(tee -a "$STDOUT_FILE")
exec 2> >(tee -a "$STDERR_FILE" >&2)

echo "Androids interviewer-context recovery"
echo "job_id=${SLURM_JOB_ID:-}"
echo "run_id=$RUN_ID"
echo "host=$(hostname)"
echo "dataset_root=$DATASET_ROOT"
echo "model_path=$MODEL_PATH"
echo "asr_overlay=$ASR_OVERLAY"
echo "limit=${LIMIT:-<all>}"
echo "resume=$RESUME"
echo "overwrite=$OVERWRITE"
echo "batch_size=$BATCH_SIZE"
nvidia-smi
python -V
python - <<'PY'
import torch, transformers, soundfile
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("soundfile", soundfile.__version__)
PY

COMMAND=(
    python
    "$PROJECT_ROOT/scripts/recover_androids_interviewer_context.py"
    --dataset-root "$DATASET_ROOT"
    --model "$MODEL_PATH"
    --batch-size "$BATCH_SIZE"
    --skip-forced-aligner-dependency
    --span-manifest "$OUTPUT_DIR/androids_interviewer_context_spans.jsonl"
    --out "$OUTPUT_DIR/androids_interviewer_context_qwen3_asr_italian.jsonl"
    --report "$OUTPUT_DIR/androids_interviewer_context_report.json"
)

if [[ -n "$LIMIT" ]]; then
    COMMAND+=(--limit "$LIMIT")
fi
if [[ "$RESUME" == "1" ]]; then
    COMMAND+=(--resume)
fi
if [[ "$OVERWRITE" == "1" ]]; then
    COMMAND+=(--overwrite)
fi

printf 'command:'
printf ' %q' "${COMMAND[@]}"
printf '\n'
"${COMMAND[@]}"

python - <<PY
import hashlib, json
from pathlib import Path
root = Path("$OUTPUT_DIR")
report = json.loads((root / "androids_interviewer_context_report.json").read_text())
for path in sorted(root.iterdir()):
    if path.is_file():
        print("artifact", path.name, path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest())
print("contexts", report.get("num_context_spans"))
print("asr_rows", report.get("asr_rows"))
print("asr_nonempty_rows", report.get("asr_nonempty_rows"))
PY
