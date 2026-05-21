#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python -V | tee "$PROJECT_ROOT/ENV_PYTHON_VERSION.txt"
pip freeze | tee "$PROJECT_ROOT/requirements_mn5_freeze.txt"
conda env export --no-builds | tee "$PROJECT_ROOT/environment_mn5_no_builds.yml"
python -c "import torch, transformers, accelerate, peft; print('torch', torch.__version__, 'cuda', torch.version.cuda); print('transformers', transformers.__version__); print('accelerate', accelerate.__version__); print('peft', peft.__version__)" | tee "$PROJECT_ROOT/ENVIRONMENT_PACKAGE_SUMMARY.txt"
