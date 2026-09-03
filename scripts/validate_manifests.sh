#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

run_python() {
  if [[ -n "${CONDA_ENV:-}" ]]; then
    conda run -n "$CONDA_ENV" "$PYTHON_BIN" "$@"
  else
    "$PYTHON_BIN" "$@"
  fi
}

run_python "$PROJECT_ROOT/src/data/build_manifest.py" \
  --config \
  "$PROJECT_ROOT/configs/main/daic_audio_text_harmonized_selmacrof1_tf.yaml" \
  "$PROJECT_ROOT/configs/main/edaic_audio_text_selposf1_tf.yaml" \
  "$PROJECT_ROOT/configs/main/cmdc_audio_text_harmonized_selmacrof1_tf.yaml" \
  "$PROJECT_ROOT/configs/main/d3tec_audio_text_harmonized_selmacrof1_tf.yaml" \
  "$PROJECT_ROOT/configs/main/androids_audio_text_harmonized_selmacrof1_tf.yaml" \
  "$PROJECT_ROOT/configs/main/turkish_pos_only_t17_audio_text_harmonized_selmacrof1_tf_qwen3asr.yaml" \
  "$PROJECT_ROOT/configs/archive/eatd/eatd_audio_text.yaml"
