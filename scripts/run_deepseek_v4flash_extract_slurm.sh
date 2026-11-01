#!/usr/bin/env bash
#SBATCH -J dsv4-extract
#SBATCH -A etur92
#SBATCH -q acc_ehpc
#SBATCH -t 02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --exclusive
# 4 x H100, one node: DeepSeek-V4-Flash frozen text features (torchrun, MP=4).
# Required: ROUTES (json). Optional: MAX_EXAMPLES (smoke), PLAN_ONLY=1.
set -euo pipefail
BASE=/gpfs/projects/etur92/ozu647717
PROJECT_ROOT="${PROJECT_ROOT:?Set PROJECT_ROOT (deployment code root)}"
ROUTES="${ROUTES:?Set ROUTES}"
MODEL_DIR=${MODEL_DIR:-$BASE/models/DeepSeek-V4-Flash}
MP_DIR=${MP_DIR:-$BASE/models/DeepSeek-V4-Flash-mp4}
source "${DSV4_ENV:-$PROJECT_ROOT/scripts/deepseek_v4flash_env.sh}"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export TOKENIZERS_PARALLELISM=false
cd "$PROJECT_ROOT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
CMD=(src/features/extract_deepseek_hidden.py --routes "$ROUTES" --model-dir "$MODEL_DIR" --mp-ckpt-dir "$MP_DIR")
[ -n "${MAX_EXAMPLES:-}" ] && CMD+=(--max-examples "$MAX_EXAMPLES")
[ "${PLAN_ONLY:-0}" = "1" ] && CMD+=(--plan-only)
torchrun --nproc-per-node 4 "${CMD[@]}"
