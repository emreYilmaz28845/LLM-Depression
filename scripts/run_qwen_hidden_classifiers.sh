#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CACHE_DIR="${CACHE_DIR:?Set CACHE_DIR}"
OUTPUT_DIR="${OUTPUT_DIR:-${CACHE_DIR/hidden_features/hidden_classifiers}}"
QWEN_HIDDEN_DEPS="${QWEN_HIDDEN_DEPS:-$PROJECT_ROOT/.deps/qwen_hidden}"
export PYTHONPATH="$QWEN_HIDDEN_DEPS:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
python "$PROJECT_ROOT/baselines/qwen_hidden_classifier.py" --cache-dir "$CACHE_DIR" --output-dir "$OUTPUT_DIR" "$@"
