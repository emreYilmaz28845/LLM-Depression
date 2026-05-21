#!/usr/bin/env bash
#SBATCH --job-name=llm-depression-eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${CONFIG:-$PROJECT_ROOT/configs/cmdc_audio_text.yaml}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:?Set CHECKPOINT_DIR}"
FOLD="${FOLD:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-$CHECKPOINT_DIR/standalone_eval}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
hostname
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi || true
python -V
python -c "import torch, transformers, accelerate, peft; print('torch', torch.__version__, 'cuda', torch.version.cuda); print('transformers', transformers.__version__); print('accelerate', accelerate.__version__); print('peft', peft.__version__)"
echo "CONFIG=$CONFIG"
echo "FOLD=$FOLD"
echo "CHECKPOINT_DIR=$CHECKPOINT_DIR"

python "$PROJECT_ROOT/src/evaluate.py" \
  --config "$CONFIG" \
  --fold "$FOLD" \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --output_dir "$OUTPUT_DIR"
