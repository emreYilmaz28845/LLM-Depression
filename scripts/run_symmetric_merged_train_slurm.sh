#!/bin/bash
#SBATCH -J sym-merged-train
#SBATCH -A etur92
#SBATCH -q acc_ehpc
#SBATCH -t 72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
# MN5 allocates 20 CPUs per requested GPU.
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#SBATCH --chdir=/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression

set -euo pipefail
module purge
module load bsc/1.0
module load miniforge/24.3.0-0

PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
ENV_ACTIVATE="${ENV_ACTIVATE:-/gpfs/projects/etur92/ozu647717/venvs/qwen_mn5_rebuilt/bin/activate}"
DATASET_BASE_ROOT="${DATASET_BASE_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/Datasets}"
export DAIC_DATASET_ROOT="${DAIC_DATASET_ROOT:-$DATASET_BASE_ROOT/DAIC-WOZ/preprocessed}"
export DAIC_UNPROCESSED_ROOT="${DAIC_UNPROCESSED_ROOT:-$DATASET_BASE_ROOT/DAIC-WOZ/unprocessed}"
export DAIC_LABEL_ROOT="${DAIC_LABEL_ROOT:-$DATASET_BASE_ROOT/DAIC-WOZ/minimal_zips}"
export CMDC_DATASET_ROOT="${CMDC_DATASET_ROOT:-$DATASET_BASE_ROOT/CMDC}"
export TURKISH_DATASET_ROOT="${TURKISH_DATASET_ROOT:-$DATASET_BASE_ROOT/Turkish}"
export D3TEC_DATASET_ROOT="${D3TEC_DATASET_ROOT:-$DATASET_BASE_ROOT/D3TEC DATASET/D3TEC DATASET}"
export D3TEC_FULL_TRANSCRIPTS="${D3TEC_FULL_TRANSCRIPTS:-$D3TEC_DATASET_ROOT/transcripts_qwen3_asr_spanish.jsonl}"
export D3TEC_SEGMENT_TRANSCRIPTS="${D3TEC_SEGMENT_TRANSCRIPTS:-$D3TEC_DATASET_ROOT/transcripts_qwen3_asr_spanish_segments.jsonl}"
export ANDROIDS_DATASET_ROOT="${ANDROIDS_DATASET_ROOT:-$DATASET_BASE_ROOT/Androids-Corpus/Androids-Corpus}"
export ANDROIDS_INTERVIEW_FULL_TRANSCRIPTS="${ANDROIDS_INTERVIEW_FULL_TRANSCRIPTS:-$ANDROIDS_DATASET_ROOT/interview_transcripts_qwen3_asr_italian.jsonl}"
export ANDROIDS_INTERVIEW_SEGMENT_TRANSCRIPTS="${ANDROIDS_INTERVIEW_SEGMENT_TRANSCRIPTS:-$ANDROIDS_DATASET_ROOT/interview_transcripts_qwen3_asr_italian_segments.jsonl}"
CONFIG="${CONFIG:?CONFIG is required}"
STAGE="${STAGE:?STAGE is required}"
FOLD="${FOLD:-0}"
RUN_ID="${RUN_ID:?RUN_ID is required}"
EPOCHS="${EPOCHS:-}"
SUBJECTS_PER_CLASS="${SUBJECTS_PER_CLASS:-}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/logs/symmetric_merged}"
OVERRIDES_JSON_B64="${OVERRIDES_JSON_B64:-}"

if [ ! -f "$ENV_ACTIVATE" ]; then
    echo "Environment activate script not found: $ENV_ACTIVATE" >&2
    exit 1
fi
# shellcheck disable=SC1090
source "$ENV_ACTIVATE"
cd "$PROJECT_ROOT"
if [ -n "$OVERRIDES_JSON_B64" ]; then
    mapfile -t OVERRIDE_ARGS < <(python - "$OVERRIDES_JSON_B64" <<'PY'
import base64, json, sys
for token in json.loads(base64.b64decode(sys.argv[1]).decode("utf-8")):
    print(token)
PY
)
else
    OVERRIDE_ARGS=()
fi
mkdir -p "$LOG_ROOT"
exec > >(tee -a "$LOG_ROOT/train-${SLURM_JOB_ID}.out")
exec 2> >(tee -a "$LOG_ROOT/train-${SLURM_JOB_ID}.err" >&2)
export PROJECT_ROOT CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}" PYTHONHASHSEED="${PYTHONHASHSEED:-0}" PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="$PROJECT_ROOT/.deps/qwen_hidden:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [ "$NPROC_PER_NODE" -ne 4 ]; then
    echo "The symmetric merged Qwen worker requires exactly four local processes; got NPROC_PER_NODE=$NPROC_PER_NODE" >&2
    exit 1
fi

# The job requests four GPUs and the worker uses Accelerate/DDP.  A plain
# `python` invocation would initialize one process on only one of the four
# allocated GPUs, so launch the local process group explicitly.
CMD=(torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" -m src.merged.train
    --config "$CONFIG" --stage "$STAGE" --fold "$FOLD" --run-id "$RUN_ID")
if [ -n "$EPOCHS" ]; then CMD+=(--epochs "$EPOCHS"); fi
if [ -n "$SUBJECTS_PER_CLASS" ]; then CMD+=(--subjects-per-class "$SUBJECTS_PER_CLASS"); fi
for token in "${OVERRIDE_ARGS[@]}"; do CMD+=(--override "$token"); done
"${CMD[@]}"
