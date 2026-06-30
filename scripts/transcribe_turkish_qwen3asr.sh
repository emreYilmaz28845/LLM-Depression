#!/usr/bin/env bash
# Run the Qwen3-ASR Turkish re-transcription in a self-contained env on the local 4090.
#
# See docs/qwen3_asr_turkish_retranscription_plan.md (§5 Steps 1-2, §7). This wrapper
# keeps the heavy bits OFF the system defaults so nothing else breaks:
#   * a SEPARATE conda env `qwen3asr` (the secap env is too old and must stay intact),
#   * HF_HOME redirected to the Backup drive (the `/` partition is nearly full),
#   * everything else delegated to scripts/transcribe_turkish_qwen3asr.py.
#
# One-time setup (idempotent; pass --setup, or run once by hand):
#   bash scripts/transcribe_turkish_qwen3asr.sh --setup
#
# Then transcribe (args after `--` are forwarded verbatim to the Python script):
#   bash scripts/transcribe_turkish_qwen3asr.sh                 # full 1186-clip run
#   bash scripts/transcribe_turkish_qwen3asr.sh -- --limit 1    # single-clip smoke test
#   bash scripts/transcribe_turkish_qwen3asr.sh -- --resume     # continue after a stop
#
# Override knobs via env: ENV_NAME, HF_HOME, CONDA_BASE, PY_VER.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${ENV_NAME:-qwen3asr}"
PY_VER="${PY_VER:-3.11}"

# Keep the (~4 GB) model + future HF downloads off the full root partition.
export HF_HOME="${HF_HOME:-/media/emre/Backup/AudioLLM/hf_cache}"
mkdir -p "$HF_HOME"

# Locate conda and enable `conda activate` inside a non-interactive shell.
CONDA_BASE="${CONDA_BASE:-$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")}"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

setup_env() {
    if conda env list | grep -qE "^\s*${ENV_NAME}\s"; then
        echo "[setup] conda env '${ENV_NAME}' already exists; skipping create."
    else
        echo "[setup] creating conda env '${ENV_NAME}' (python ${PY_VER}) ..."
        conda create -n "$ENV_NAME" "python=${PY_VER}" -y
    fi
    conda activate "$ENV_NAME"
    echo "[setup] installing qwen-asr + transformers-from-source + soundfile ..."
    # qwen-asr pulls a recent torch; install transformers from source until Qwen3-ASR
    # ships in an official release (per the model card).
    pip install -U pip
    pip install -U "git+https://github.com/huggingface/transformers"
    pip install -U qwen-asr soundfile
    # qwen-asr pulls the LATEST torch, which is a CUDA 13 wheel (2.12+cu130). The local
    # driver is 535 (CUDA 12.2): CUDA 13 is a new MAJOR version needing driver >=580, so it
    # cannot initialize here. Pin a CUDA 12.x torch instead — minor-version compatibility
    # lets a 12.x runtime run on the 12.2 driver. cu128 tops out at torch 2.11.0.
    pip install --force-reinstall "torch==${TORCH_VER:-2.11.0}" \
        --index-url "https://download.pytorch.org/whl/${TORCH_CUDA:-cu128}"
    echo "[setup] verifying CUDA + imports ..."
    python - <<'PY'
import torch
from qwen_asr import Qwen3ASRModel  # noqa: F401
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
assert torch.cuda.is_available(), "CUDA not available; check the driver/torch build before the bulk run."
print("Qwen3ASRModel import OK")
PY
    echo "[setup] done. HF_HOME=$HF_HOME"
}

if [[ "${1:-}" == "--setup" ]]; then
    setup_env
    exit 0
fi

# Drop a leading `--` separator so callers can do: ... -- --limit 1
if [[ "${1:-}" == "--" ]]; then
    shift
fi

if ! conda env list | grep -qE "^\s*${ENV_NAME}\s"; then
    echo "conda env '${ENV_NAME}' not found. Run: bash $0 --setup" >&2
    exit 1
fi
conda activate "$ENV_NAME"

echo "[run] HF_HOME=$HF_HOME env=$ENV_NAME"
exec python "$PROJECT_ROOT/scripts/transcribe_turkish_qwen3asr.py" "$@"
