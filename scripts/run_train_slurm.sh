#!/usr/bin/env bash
#SBATCH --job-name=llm-depression-train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${CONFIG:-$PROJECT_ROOT/configs/cmdc_audio_text.yaml}"
FOLD="${FOLD:-0}"
RUN_NAME="${RUN_NAME:-mn5_reproduction}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
hostname
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi || true
python -V
python -c "import torch, transformers, accelerate, peft; print('torch', torch.__version__, 'cuda', torch.version.cuda); print('transformers', transformers.__version__); print('accelerate', accelerate.__version__); print('peft', peft.__version__)"
echo "CONFIG=$CONFIG"
echo "FOLD=$FOLD"
echo "RUN_NAME=$RUN_NAME"
python - <<PY
import yaml
from pathlib import Path
config = yaml.safe_load(Path("$CONFIG").read_text())
per_device = int(config["training"]["per_device_train_batch_size"])
grad_acc = int(config["training"]["gradient_accumulation_steps"])
world_size = int("$NPROC_PER_NODE")
print("dataset", config["dataset"])
print("effective_batch_size", per_device * grad_acc * world_size)
PY

torchrun --nproc_per_node="$NPROC_PER_NODE" \
  "$PROJECT_ROOT/src/train.py" \
  --config "$CONFIG" \
  --fold "$FOLD" \
  --run_name "$RUN_NAME"
