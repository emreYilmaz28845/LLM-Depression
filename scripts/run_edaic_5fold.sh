#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_NAME="${RUN_NAME:-edaic_cv_reproduction}"

for FOLD in 0 1 2 3 4; do
  torchrun --nproc_per_node="${NPROC_PER_NODE:-4}" \
    "$PROJECT_ROOT/src/train.py" \
    --config "$PROJECT_ROOT/configs/main/edaic_audio_text_selposf1_tf.yaml" \
    --fold "$FOLD" \
    --run_name "$RUN_NAME" \
    --set split.mode=cv
done

python "$PROJECT_ROOT/src/summarize_runs.py" \
  --run_root "$PROJECT_ROOT/output_model/audio_text/edaic/$RUN_NAME"
