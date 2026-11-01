#!/usr/bin/env bash
#SBATCH -J dsv4-overlay-build
#SBATCH -A etur92
#SBATCH -q acc_ehpc
#SBATCH -t 00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=none
# CPU-only. Replaces the full-venv build (two 60 min timeouts while pip unpacked
# ~200 packages onto GPFS). The shared qwen38_inference venv already has torch 2.11,
# tilelang deps (apache-tvm-ffi, z3, torch-c-dlpack-ext), nvcc 13.2 and ninja.
# Only tilelang==0.1.8 (reference pin) and fast_hadamard_transform are missing.
# We install those two with --target into node-local disk, check imports there,
# and write ONE tar file to GPFS. Jobs untar it to node-local disk and put it
# first on PYTHONPATH. qwen38_inference is only read, never written.
set -euo pipefail
BASE=/gpfs/projects/etur92/ozu647717
SHARED_VENV=$BASE/venvs/qwen38_inference
EXTRA=$BASE/wheelhouses/deepseek_v4flash_extra
OUT_TAR=${OUT_TAR:-$BASE/venvs/deepseek_v4flash_overlay.tar}
[ -e "$OUT_TAR" ] && { echo "FAILED: $OUT_TAR exists" >&2; exit 1; }

module purge; module load bsc/1.0 miniforge/24.3.0-0
PY="$SHARED_VENV/bin/python"
export PIP_NO_INDEX=1 PYTHONNOUSERSITE=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONDONTWRITEBYTECODE=1

WORK="$(mktemp -d -p "${TMPDIR:-/tmp}" dsv4ov.XXXXXX)"
OV="$WORK/overlay"; mkdir -p "$OV"
echo "node=$(hostname) work=$WORK"; date

# 1) tilelang 0.1.8, no deps (all deps already in the shared venv).
"$PY" -m pip install --no-index --no-deps --target "$OV" "$EXTRA"/tilelang-0.1.8-*.whl
date

# 2) fast_hadamard_transform from source for H100 (sm_90).
SP="$("$PY" -c 'import site;print(site.getsitepackages()[0])')"
export CUDA_HOME="$SP/nvidia/cu13"
"$CUDA_HOME/bin/nvcc" -V
export PATH="$CUDA_HOME/bin:$SHARED_VENV/bin:$PATH" TORCH_CUDA_ARCH_LIST="9.0" \
       FAST_HADAMARD_TRANSFORM_FORCE_BUILD=TRUE MAX_JOBS=16
# pip CUDA ships only libcudart.so.13, so `-lcudart` fails to link (job 48380525).
# Give the linker an unversioned name in a job-local dir; at run time torch loads .so.13.
mkdir -p "$WORK/libfix"; ln -s "$CUDA_HOME/lib/libcudart.so.13" "$WORK/libfix/libcudart.so"
export LIBRARY_PATH="$WORK/libfix${LIBRARY_PATH:+:$LIBRARY_PATH}"
# setup.py hard-codes sm_75..sm_121 gencodes (ignores TORCH_CUDA_ARCH_LIST); sm_90 is included.
tar xzf "$EXTRA/fht.tar.gz" -C "$WORK"
FHT="$WORK/fast-hadamard-transform-master"
# setup.py imports wheel.bdist_wheel; the `wheel` package is not in any wheelhouse.
# setuptools>=70.1 ships the same class.
sed -i 's/^from wheel.bdist_wheel import bdist_wheel as _bdist_wheel/from setuptools.command.bdist_wheel import bdist_wheel as _bdist_wheel/' "$FHT/setup.py"
"$PY" -m pip install --no-index --no-deps --no-build-isolation --target "$OV" "$FHT"
date

# 3) Check from node-local disk: overlay versions must win over the shared venv.
# libtvm.so RUNPATH expects z3/ and tvm_ffi/ next to tilelang/ (same site-packages);
# they live in the shared venv, so expose their lib dirs (job 48380913: libz3.so not found).
export LD_LIBRARY_PATH="$SP/z3/lib:$SP/tvm_ffi/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
DSV4_OV="$OV" PYTHONPATH="$OV" "$PY" - <<'EOF'
import importlib.metadata as md, os, sys
import torch, tilelang, fast_hadamard_transform
# tilelang rewrites PYTHONPATH on import, so use a separate variable.
ov = os.path.realpath(os.environ["DSV4_OV"]) + os.sep
for mod in (tilelang, fast_hadamard_transform):
    assert os.path.realpath(mod.__file__).startswith(ov), (mod.__file__, ov)
assert md.version("tilelang") == "0.1.8", md.version("tilelang")
from fast_hadamard_transform import hadamard_transform  # loads the compiled .so
print("ok tilelang", md.version("tilelang"), "fht", md.version("fast_hadamard_transform"),
      "torch", torch.__version__, "cuda", torch.version.cuda, "py", sys.version.split()[0])
EOF
"$PY" -m pip list --path "$OV" --format=freeze > "$OV/OVERLAY_FREEZE.txt"
cat "$OV/OVERLAY_FREEZE.txt"

# 4) One file to GPFS.
tar cf "$WORK/overlay.tar" -C "$WORK" overlay
cp "$WORK/overlay.tar" "$OUT_TAR.partial"
mv "$OUT_TAR.partial" "$OUT_TAR"
sha256sum "$OUT_TAR" | tee "$OUT_TAR.sha256"
ls -la "$OUT_TAR"; echo "files: $(tar tf "$OUT_TAR" | wc -l)"

# 5) Small-file probe (evidence for the GPFS hypothesis; own probe dir only).
PROBE="$EXTRA/fs_probe_${SLURM_JOB_ID:-x}"; mkdir "$PROBE"
t0=$(date +%s%N); for i in $(seq 1 2000); do head -c 4096 /dev/zero > "$PROBE/f$i"; done; sync
t1=$(date +%s%N); echo "gpfs 2000x4KB files: $(( (t1 - t0) / 1000000 )) ms"
LP="$WORK/probe"; mkdir "$LP"
t0=$(date +%s%N); for i in $(seq 1 2000); do head -c 4096 /dev/zero > "$LP/f$i"; done; sync
t1=$(date +%s%N); echo "local 2000x4KB files: $(( (t1 - t0) / 1000000 )) ms"
rm -rf "$PROBE" "$WORK"
date; echo DONE
