#!/bin/bash
# Dedicated Gemma 4 MN5 environment setup (OFFLINE only).
#
# MN5 has no outbound internet. This script installs exclusively from a
# locally built wheelhouse that was transferred to GPFS through transfer1.
# Never use pip/conda network installation on MN5.
#
# Usage:
#   GEMMA_ENV_TARGET=/gpfs/projects/etur92/ozu647717/venvs/gemma4_12b_tf5_14_1 \
#   GEMMA_WHEELHOUSE=/gpfs/projects/etur92/ozu647717/wheelhouses/gemma4_tf5_14_1 \
#   bash scripts/setup_gemma4_mn5_env.sh
#
# Safe to rerun when the environment already matches exactly (idempotent
# install; `pip check` and freeze/audit are re-emitted).

set -euo pipefail

QENV="/gpfs/projects/etur92/ozu647717/venvs/qwen_mn5_rebuilt"
GEMMA_ENV_TARGET="${GEMMA_ENV_TARGET:-/gpfs/projects/etur92/ozu647717/venvs/gemma4_12b_tf5_14_1}"
GEMMA_WHEELHOUSE="${GEMMA_WHEELHOUSE:-/gpfs/projects/etur92/ozu647717/wheelhouses/gemma4_tf5_14_1}"
# Staged standalone CPython 3.11 (MN5 has no outbound internet and no conda
# package cache, so the base interpreter itself is delivered offline).
GEMMA_PYTHON_BASE="${GEMMA_PYTHON_BASE:-/gpfs/projects/etur92/ozu647717/venvs/gemma4_python311_stage/python}"
GEMMA_AUDIT_DIR="${GEMMA_AUDIT_DIR:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression/outputs/gemma4_env_audits}"
REQUIREMENTS="${REQUIREMENTS:-$PWD/requirements_mn5_gemma4.txt}"

if [ -z "$GEMMA_ENV_TARGET" ]; then
    echo "ERROR: GEMMA_ENV_TARGET is required." >&2
    exit 1
fi
if [ "$GEMMA_ENV_TARGET" = "$QENV" ]; then
    echo "ERROR: refusing to use the Qwen environment path: $QENV" >&2
    exit 1
fi
case "$GEMMA_ENV_TARGET" in
    *gemma4*) ;;
    *)
        echo "ERROR: target path must contain 'gemma4': $GEMMA_ENV_TARGET" >&2
        exit 1
        ;;
esac
if [ ! -f "$REQUIREMENTS" ]; then
    echo "ERROR: requirements file not found: $REQUIREMENTS" >&2
    exit 1
fi
if [ ! -d "$GEMMA_WHEELHOUSE" ] || [ -z "$(ls -A "$GEMMA_WHEELHOUSE" 2>/dev/null)" ]; then
    echo "ERROR: wheelhouse missing or empty: $GEMMA_WHEELHOUSE" >&2
    exit 1
fi
if [ ! -x "$GEMMA_PYTHON_BASE/bin/python3.11" ]; then
    echo "ERROR: staged standalone Python 3.11 not found: $GEMMA_PYTHON_BASE" >&2
    exit 1
fi

mkdir -p "$GEMMA_AUDIT_DIR"

if [ ! -f "$GEMMA_ENV_TARGET/bin/activate" ]; then
    echo "Creating dedicated Python 3.11 environment at $GEMMA_ENV_TARGET"
    "$GEMMA_PYTHON_BASE/bin/python3.11" -m venv --copies "$GEMMA_ENV_TARGET"
else
    echo "Environment exists: $GEMMA_ENV_TARGET (revalidating instead of recreating)"
fi

# shellcheck disable=SC1091
source "$GEMMA_ENV_TARGET/bin/activate"

PYTHON_VERSION="$(python -V 2>&1 | sed 's/Python //')"
case "$PYTHON_VERSION" in
    3.11.*) ;;
    *)
        echo "ERROR: environment Python is not 3.11: $PYTHON_VERSION" >&2
        exit 1
        ;;
esac

echo "Installing pinned Gemma requirements OFFLINE from $GEMMA_WHEELHOUSE"
python -m pip install --no-index --find-links "$GEMMA_WHEELHOUSE" -r "$REQUIREMENTS"

echo "Running pip check"
python -m pip check

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

TIMESTAMP="$(date +%Y-%m-%d_%H:%M:%S)"
FREEZE_FILE="$GEMMA_AUDIT_DIR/pip_freeze_${TIMESTAMP}.txt"
AUDIT_FILE="$GEMMA_AUDIT_DIR/env_audit_${TIMESTAMP}.json"
pip freeze > "$FREEZE_FILE"
python - <<PY
import json, os, sys
import torch, torchvision, transformers, peft, accelerate, numpy, librosa, soundfile, PIL
from transformers import Gemma4UnifiedProcessor, Gemma4UnifiedForConditionalGeneration
import torch.cuda as tc
audit = {
    "environment": "$GEMMA_ENV_TARGET",
    "python": sys.version.split()[0],
    "torch": torch.__version__,
    "torch.cuda": torch.version.cuda,
    "cuda_available": tc.is_available(),
    "cuda_device_count": tc.device_count(),
    "cuda_device_name": (tc.get_device_name(0) if tc.is_available() and tc.device_count() > 0 else None),
    "torchvision": torchvision.__version__,
    "transformers": transformers.__version__,
    "peft": peft.__version__,
    "accelerate": accelerate.__version__,
    "numpy": numpy.__version__,
    "librosa": librosa.__version__,
    "soundfile": soundfile.__version__,
    "pillow": PIL.__version__,
    "gemma4_processor_class": Gemma4UnifiedProcessor.__name__,
    "gemma4_model_class": Gemma4UnifiedForConditionalGeneration.__name__,
    "freeze_file": "$FREEZE_FILE",
}
with open("$AUDIT_FILE", "w", encoding="utf-8") as handle:
    json.dump(audit, handle, indent=2)
print(json.dumps(audit, indent=2))
PY

echo "Environment setup complete."
echo "Package freeze: $FREEZE_FILE"
echo "Environment audit: $AUDIT_FILE"
