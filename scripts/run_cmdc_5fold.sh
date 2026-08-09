#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_NAME="${RUN_NAME:-cmdc_reproduction}"

for FOLD in 0 1 2 3 4; do
  torchrun --nproc_per_node="${NPROC_PER_NODE:-4}" \
    "$PROJECT_ROOT/src/train.py" \
    --config "$PROJECT_ROOT/configs/main/cmdc_audio_text_harmonized_selmacrof1_tf.yaml" \
    --fold "$FOLD" \
    --run_name "$RUN_NAME"
done

python "$PROJECT_ROOT/src/summarize_runs.py" \
  --run_root "$PROJECT_ROOT/output_model/audio_text/cmdc/$RUN_NAME"
