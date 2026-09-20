#!/bin/bash
# Dedicated Qwen3-Omni (Thinker-only) MN5 environment setup (OFFLINE only).
#
# MN5 has no outbound internet. This script installs exclusively from the
# locally built wheelhouse that already sits on GPFS. Never use pip/conda
# network installation on MN5.
#
# The base interpreter is miniforge 24.3.0-0 Python 3.10.14, which is what the
# existing /gpfs/projects/etur92/ozu647717/venvs/qwen3omni environment was
# created from. Load the module before running:
#
#   module purge
#   module load bsc/1.0 miniforge/24.3.0-0
#   cd /gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression
#   bash scripts/setup_qwen3omni_mn5_env.sh
#
# Safe to rerun when the environment already matches exactly (idempotent
# install; `pip check` and freeze/audit are re-emitted).

set -euo pipefail

QENV="/gpfs/projects/etur92/ozu647717/venvs/qwen_mn5_rebuilt"
QWEN3OMNI_ENV_TARGET="${QWEN3OMNI_ENV_TARGET:-/gpfs/projects/etur92/ozu647717/venvs/qwen3omni}"
QWEN3OMNI_WHEELHOUSE="${QWEN3OMNI_WHEELHOUSE:-/gpfs/projects/etur92/ozu647717/wheelhouses/wheelhouse_qwen3omni}"
QWEN3OMNI_MODEL_DIR="${QWEN3OMNI_MODEL_DIR:-/gpfs/projects/etur92/ozu647717/models/Qwen3-Omni-30B-A3B-Instruct}"
QWEN3OMNI_AUDIT_DIR="${QWEN3OMNI_AUDIT_DIR:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression/outputs/qwen3omni_env_audits}"
REQUIREMENTS="${REQUIREMENTS:-$PWD/requirements_mn5_qwen3omni.txt}"
DEEPSPEED_SDIST_VERSION="${DEEPSPEED_SDIST_VERSION:-0.19.2}"

if [ -z "$QWEN3OMNI_ENV_TARGET" ]; then
    echo "ERROR: QWEN3OMNI_ENV_TARGET is required." >&2
    exit 1
fi
if [ "$QWEN3OMNI_ENV_TARGET" = "$QENV" ]; then
    echo "ERROR: refusing to use the Qwen environment path: $QENV" >&2
    exit 1
fi
case "$QWEN3OMNI_ENV_TARGET" in
    *qwen3omni*) ;;
    *)
        echo "ERROR: target path must contain 'qwen3omni': $QWEN3OMNI_ENV_TARGET" >&2
        exit 1
        ;;
esac
if [ ! -f "$REQUIREMENTS" ]; then
    echo "ERROR: requirements file not found: $REQUIREMENTS" >&2
    exit 1
fi
if [ ! -d "$QWEN3OMNI_WHEELHOUSE" ] || [ -z "$(ls -A "$QWEN3OMNI_WHEELHOUSE" 2>/dev/null)" ]; then
    echo "ERROR: wheelhouse missing or empty: $QWEN3OMNI_WHEELHOUSE" >&2
    exit 1
fi
if [ ! -x "$QWEN3OMNI_ENV_TARGET/bin/python" ]; then
    echo "ERROR: environment python not found: $QWEN3OMNI_ENV_TARGET/bin/python" >&2
    exit 1
fi
if [ ! -f "$QWEN3OMNI_MODEL_DIR/config.json" ]; then
    echo "WARNING: Qwen3-Omni model snapshot not found at $QWEN3OMNI_MODEL_DIR" >&2
    echo "WARNING: install proceeds, but no run can load the model until it is staged." >&2
fi

mkdir -p "$QWEN3OMNI_AUDIT_DIR"

# shellcheck disable=SC1091
source "$QWEN3OMNI_ENV_TARGET/bin/activate"

PYTHON_VERSION="$(python -V 2>&1 | sed 's/Python //')"
case "$PYTHON_VERSION" in
    3.10.*) ;;
    *)
        echo "ERROR: environment Python is not 3.10: $PYTHON_VERSION" >&2
        exit 1
        ;;
esac

echo "Installing pinned Qwen3-Omni requirements OFFLINE from $QWEN3OMNI_WHEELHOUSE"
python -m pip install --no-index --find-links "$QWEN3OMNI_WHEELHOUSE" -r "$REQUIREMENTS"

# deepspeed ships as an sdist and is installed without build isolation: offline
# build isolation cannot fetch the extra build requirements, and setuptools is
# already upgraded by the step above.
echo "Installing deepspeed==$DEEPSPEED_SDIST_VERSION from the local sdist (no build isolation)"
python -m pip install --no-index --find-links "$QWEN3OMNI_WHEELHOUSE" --no-build-isolation \
    "deepspeed==$DEEPSPEED_SDIST_VERSION"

echo "Running pip check"
python -m pip check

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

TIMESTAMP="$(date +%Y-%m-%d_%H:%M:%S)"
FREEZE_FILE="$QWEN3OMNI_AUDIT_DIR/pip_freeze_${TIMESTAMP}.txt"
AUDIT_FILE="$QWEN3OMNI_AUDIT_DIR/env_audit_${TIMESTAMP}.json"
pip freeze > "$FREEZE_FILE"
python - <<PY
import json, os, sys
import torch, torchvision, transformers, peft, accelerate, numpy, librosa, soundfile, PIL
from transformers import (
    Qwen3OmniMoeForConditionalGeneration,
    Qwen3OmniMoeProcessor,
    Qwen3OmniMoeThinkerForConditionalGeneration,
)
import torch.cuda as tc
audit = {
    "environment": "$QWEN3OMNI_ENV_TARGET",
    "wheelhouse": "$QWEN3OMNI_WHEELHOUSE",
    "model_dir": "$QWEN3OMNI_MODEL_DIR",
    "model_dir_config_exists": os.path.isfile(os.path.join("$QWEN3OMNI_MODEL_DIR", "config.json")),
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
    "qwen3omni_processor_class": Qwen3OmniMoeProcessor.__name__,
    "qwen3omni_model_class": Qwen3OmniMoeForConditionalGeneration.__name__,
    "qwen3omni_thinker_class": Qwen3OmniMoeThinkerForConditionalGeneration.__name__,
    "freeze_file": "$FREEZE_FILE",
}
with open("$AUDIT_FILE", "w", encoding="utf-8") as handle:
    json.dump(audit, handle, indent=2)
print(json.dumps(audit, indent=2))
PY

echo "Environment setup complete."
echo "Package freeze: $FREEZE_FILE"
echo "Environment audit: $AUDIT_FILE"
