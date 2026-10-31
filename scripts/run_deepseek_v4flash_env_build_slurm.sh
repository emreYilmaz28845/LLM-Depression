#!/usr/bin/env bash
#SBATCH -J dsv4-env-build
#SBATCH -A etur92
#SBATCH -q acc_ehpc
#SBATCH -t 01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=none
# CPU-only: build venvs/deepseek_v4flash offline. Same lock as qwen38_inference,
# then tilelang==0.1.8 (reference pin) and fast_hadamard_transform built from source.
# The shared qwen38_inference venv is not touched.
set -euo pipefail
BASE=/gpfs/projects/etur92/ozu647717
WH=$BASE/wheelhouses/qwen38_vllm_0.25.1_tf5.8.0_cu130_py310
EXTRA=$BASE/wheelhouses/deepseek_v4flash_extra
VENV=$BASE/venvs/deepseek_v4flash
module purge; module load bsc/1.0 miniforge/24.3.0-0
PY="$(command -v python)"; "$PY" -c 'import sys; assert sys.version_info[:2]==(3,10)'
[ -e "$VENV" ] && { echo "FAILED: $VENV exists" >&2; exit 1; }
"$PY" -m venv --copies "$VENV"
export PIP_NO_INDEX=1
"$VENV/bin/python" -m pip install --no-index --find-links "$WH/wheels" --require-hashes --requirement "$WH/requirements.lock"
"$VENV/bin/python" -m pip install --no-index --find-links "$WH/wheels" --find-links "$EXTRA" --no-deps "$EXTRA"/tilelang-0.1.8-*.whl
SP="$("$VENV/bin/python" -c 'import site;print(site.getsitepackages()[0])')"
export CUDA_HOME="$SP/nvidia/cu13"; [ -x "$CUDA_HOME/bin/nvcc" ] || CUDA_HOME="$(dirname "$(dirname "$(find "$SP" -name nvcc -type f | head -1)")")"
echo "CUDA_HOME=$CUDA_HOME"; "$CUDA_HOME/bin/nvcc" -V
export PATH="$CUDA_HOME/bin:$PATH" TORCH_CUDA_ARCH_LIST="9.0" FAST_HADAMARD_TRANSFORM_FORCE_BUILD=TRUE MAX_JOBS=8
T="$(mktemp -d)"; tar xzf "$EXTRA/fht.tar.gz" -C "$T"
"$VENV/bin/python" -m pip install --no-index --no-deps --no-build-isolation "$T"/fast-hadamard-transform-master
"$VENV/bin/python" -c 'import tilelang, fast_hadamard_transform, torch; print("ok", tilelang.__version__, torch.__version__)'
"$VENV/bin/python" -m pip freeze > "$VENV/pip_freeze.txt"
