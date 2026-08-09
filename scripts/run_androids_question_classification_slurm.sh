#!/usr/bin/env bash
#SBATCH -J androids-qclass
#SBATCH -A etur92
#SBATCH -q acc_ehpc
#SBATCH -t 00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:2
#SBATCH -o /dev/null
#SBATCH -e /dev/null

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
RUN_ID="${RUN_ID:?Set RUN_ID}"
SOURCE_RUN_ID="${SOURCE_RUN_ID:-androids_qctx_full_20260809T124731Z}"
LIMIT="${LIMIT:-}"
RESUME="${RESUME:-0}"
MODEL_DIR="${MODEL_DIR:-/gpfs/projects/etur92/ozu647717/models/Qwen3.6-27B}"
VENV_QWEN36="${VENV_QWEN36:-/gpfs/projects/etur92/ozu647717/venvs/qwen36_translation/bin/activate}"
MODEL_REVISION="${MODEL_REVISION:-6a9e13bd6fc8f0983b9b99948120bc37f49c13e9}"

SOURCE_DIR="$PROJECT_ROOT/outputs/androids_question_recovery/$SOURCE_RUN_ID"
SOURCE="$SOURCE_DIR/androids_interviewer_context_qwen3_asr_italian.jsonl"
OUT_DIR="$PROJECT_ROOT/outputs/androids_question_classification/$RUN_ID"
LOG_DIR="$PROJECT_ROOT/logs/slurm_androids_question_classification/$RUN_ID"
PREDICTIONS="$OUT_DIR/model_predictions.jsonl"
mkdir -p "$OUT_DIR" "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/classify-${SLURM_JOB_ID}.out") 2> >(tee -a "$LOG_DIR/classify-${SLURM_JOB_ID}.err" >&2)

test -f "$SOURCE"
test -d "$MODEL_DIR"

module purge
module load bsc/1.0
module load miniforge/24.3.0-0
source "$VENV_QWEN36"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_TORCH_COMPILE_OVERRIDE=0

echo "job_id=$SLURM_JOB_ID"
echo "run_id=$RUN_ID"
echo "source=$SOURCE"
echo "model=$MODEL_DIR"
python -VV
python -c "import torch,vllm; print('torch',torch.__version__,'vllm',vllm.__version__)"

python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_DIR" \
  --served-model-name qwen3.6-27b \
  --host 127.0.0.1 \
  --port 8000 \
  --tensor-parallel-size 2 \
  --dtype bfloat16 \
  --language-model-only \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --generation-config vllm \
  --enforce-eager \
  > "$LOG_DIR/vllm-${SLURM_JOB_ID}.log" 2>&1 &
SERVER_PID=$!
cleanup() {
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

READY=0
for _ in $(seq 1 300); do
  if curl -sf -o /dev/null http://127.0.0.1:8000/v1/models; then
    READY=1
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    tail -n 80 "$LOG_DIR/vllm-${SLURM_JOB_ID}.log"
    exit 1
  fi
  sleep 2
done
test "$READY" = "1"

CLASSIFY_ARGS=(
  classify
  --input "$SOURCE"
  --output "$PREDICTIONS"
  --base-url http://127.0.0.1:8000/v1
  --model qwen3.6-27b
  --workers 4
  --seed 42
)
if [ -n "$LIMIT" ]; then CLASSIFY_ARGS+=(--limit "$LIMIT"); fi
if [ "$RESUME" = "1" ]; then CLASSIFY_ARGS+=(--resume); fi
python "$PROJECT_ROOT/scripts/classify_androids_interviewer_context.py" "${CLASSIFY_ARGS[@]}"

if [ -z "$LIMIT" ]; then
  python "$PROJECT_ROOT/scripts/classify_androids_interviewer_context.py" finalize \
    --input "$SOURCE" \
    --predictions "$PREDICTIONS" \
    --turn-map "$OUT_DIR/turn_question_map.jsonl" \
    --question-inventory "$OUT_DIR/question_inventory.jsonl" \
    --report "$OUT_DIR/report.json"
fi

sha256sum "$OUT_DIR"/*
