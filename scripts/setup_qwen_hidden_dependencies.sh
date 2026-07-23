#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
TARGET="${QWEN_HIDDEN_DEPS:-$PROJECT_ROOT/.deps/qwen_hidden}"
mkdir -p "$TARGET"
python -m pip install --only-binary=:all: --no-deps --target "$TARGET" -r "$PROJECT_ROOT/requirements_hidden_features.txt"
PYTHONPATH="$TARGET${PYTHONPATH:+:$PYTHONPATH}" python -c '
import optuna
import xgboost

assert optuna.__version__ == "4.4.0", optuna.__version__
assert xgboost.__version__ == "2.1.4", xgboost.__version__
print("optuna", optuna.__version__)
print("xgboost", xgboost.__version__)
'
