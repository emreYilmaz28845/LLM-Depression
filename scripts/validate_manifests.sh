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
  "$PROJECT_ROOT/configs/daic_audio_text.yaml" \
  "$PROJECT_ROOT/configs/edaic_audio_text.yaml" \
  "$PROJECT_ROOT/configs/cmdc_audio_text.yaml" \
  "$PROJECT_ROOT/configs/eatd_audio_text.yaml"
