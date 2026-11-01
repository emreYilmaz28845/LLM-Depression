#!/usr/bin/env bash
#SBATCH -J dsv4-convert
#SBATCH -A etur92
#SBATCH -q acc_ehpc
#SBATCH -t 02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --exclusive
#SBATCH --gres=none
# CPU-only: reference convert.py, HF shards -> model-parallel (MP=4) shards, FP4 experts kept.
# Output is a new directory; an existing one is refused.
set -euo pipefail
BASE=/gpfs/projects/etur92/ozu647717
SRC=${SRC:-$BASE/models/DeepSeek-V4-Flash}
OUT=${OUT:-$BASE/models/DeepSeek-V4-Flash-mp4}
[ -e "$OUT" ] && { echo "FAILED: $OUT exists" >&2; exit 1; }
# sbatch copies this script to a spool dir, so locate the helper from the submit dir.
source "${DSV4_ENV:-$SLURM_SUBMIT_DIR/scripts/deepseek_v4flash_env.sh}"
cd "$SRC/inference"
free -g | head -2
python convert.py --hf-ckpt-path "$SRC" --save-path "$OUT" --n-experts 256 --model-parallel 4
ls -la "$OUT"; du -sh "$OUT"
