# Sourced by the DeepSeek-V4-Flash Slurm scripts (not run directly).
# Python env = shared qwen38_inference venv (read-only) + an overlay with
# tilelang==0.1.8 and fast_hadamard_transform (built by
# run_deepseek_v4flash_overlay_build_slurm.sh). The overlay tar is unpacked to
# node-local disk so GPFS never sees thousands of small files at run time.
BASE=${BASE:-/gpfs/projects/etur92/ozu647717}
SHARED_VENV=${SHARED_VENV:-$BASE/venvs/qwen38_inference}
OVERLAY_TAR=${OVERLAY_TAR:-$BASE/venvs/deepseek_v4flash_overlay_v2.tar}
module purge; module load bsc/1.0 miniforge/24.3.0-0
source "$SHARED_VENV/bin/activate"
[ -f "$OVERLAY_TAR" ] || { echo "FAILED: missing $OVERLAY_TAR" >&2; exit 1; }
( cd "$(dirname "$OVERLAY_TAR")" && sha256sum -c "$(basename "$OVERLAY_TAR").sha256" ) \
  || { echo "FAILED: overlay sha256 mismatch" >&2; exit 1; }
DSV4_LOCAL="$(mktemp -d -p "${TMPDIR:-/tmp}" dsv4env.XXXXXX)"
tar xf "$OVERLAY_TAR" -C "$DSV4_LOCAL"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$DSV4_LOCAL/overlay${PYTHONPATH:+:$PYTHONPATH}"
export TILELANG_CACHE_DIR="$DSV4_LOCAL/tilelang_cache" TILELANG_TMP_DIR="$DSV4_LOCAL/tilelang_tmp"
mkdir -p "$TILELANG_CACHE_DIR" "$TILELANG_TMP_DIR"
SP="$(python -c 'import site;print(site.getsitepackages()[0])')"
export CUDA_HOME="$SP/nvidia/cu13" PATH="$SP/nvidia/cu13/bin:$PATH"
# libtvm.so RUNPATH expects z3/ and tvm_ffi/ next to tilelang/; they are in the shared venv.
export LD_LIBRARY_PATH="$SP/z3/lib:$SP/tvm_ffi/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python -c 'import importlib.metadata as m, tilelang, fast_hadamard_transform, torch; assert m.version("tilelang")=="0.1.8"; print("env ok tilelang", m.version("tilelang"), "torch", torch.__version__)'
trap 'rm -rf "$DSV4_LOCAL"' EXIT
