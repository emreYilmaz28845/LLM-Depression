#!/usr/bin/env bash
#SBATCH -J daic-k4-coverage
#SBATCH -A etur92
#SBATCH -q acc_ehpc
#SBATCH -t 24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#SBATCH --chdir=/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression

set -euo pipefail

module purge
module load bsc/1.0 miniforge/24.3.0-0
source "${ENV_ACTIVATE:-/gpfs/projects/etur92/ozu647717/venvs/qwen_mn5_rebuilt/bin/activate}"

PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
CONFIG="${CONFIG:-$PROJECT_ROOT/configs/main/daic_audio_text_selposf1_tf.yaml}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:?Set CHECKPOINT_DIR}"
RUN_ID="${RUN_ID:?Set a unique RUN_ID}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/daic_k4_coverage_audit}"
FOLD="${FOLD:-0}"
RESUME="${RESUME:-0}"
LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/logs/daic_k4_coverage_audit/$RUN_ID}"
export DAIC_DATASET_ROOT="${DAIC_DATASET_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/Datasets/DAIC-WOZ/preprocessed}"

cd "$PROJECT_ROOT"
mkdir -p "$LOG_ROOT"
exec > >(tee -a "$LOG_ROOT/job-${SLURM_JOB_ID}.out")
exec 2> >(tee -a "$LOG_ROOT/job-${SLURM_JOB_ID}.err" >&2)

args=(
  python "$PROJECT_ROOT/scripts/run_daic_k4_coverage_audit.py"
  --config "$CONFIG"
  --checkpoint-dir "$CHECKPOINT_DIR"
  --fold "$FOLD"
  --run-id "$RUN_ID"
  --output-root "$OUTPUT_ROOT"
)
if [ "$RESUME" = "1" ]; then args+=(--resume); fi
printf 'Launch command: '; printf '%q ' "${args[@]}"; printf '\n'
nvidia-smi || true
"${args[@]}"
