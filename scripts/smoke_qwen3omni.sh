#!/usr/bin/env bash
# Local Qwen3-Omni smoke gate (see QWEN3_OMNI_IMPLEMENTATION.md §6).
# Runs Tier A (real processor + collator) and Tier B (tiny random-config end-to-end).
# NEVER downloads the 30B weights. Safe for a disk-constrained box.
#
#   ./scripts/smoke_qwen3omni.sh                 # both tiers, CPU
#   DEVICE=cuda ./scripts/smoke_qwen3omni.sh     # Tier B tiny model on the 4090
#   TIER=A ./scripts/smoke_qwen3omni.sh          # single tier
#
# Optional: point HF caches off the main SSD if space is tight:
#   HF_HOME=/path/on/big/disk ./scripts/smoke_qwen3omni.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
TIER="${TIER:-all}"
DEVICE="${DEVICE:-cpu}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-Omni-30B-A3B-Instruct}"

run_python() {
  if [[ -n "${CONDA_ENV:-}" ]]; then
    conda run -n "$CONDA_ENV" "$PYTHON_BIN" "$@"
  else
    "$PYTHON_BIN" "$@"
  fi
}

cd "$PROJECT_ROOT"
echo "[smoke] tier=$TIER device=$DEVICE model_id=$MODEL_ID"
run_python scripts/smoke_qwen3omni.py --tier "$TIER" --device "$DEVICE" --model-id "$MODEL_ID"
