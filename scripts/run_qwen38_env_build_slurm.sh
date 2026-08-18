#!/usr/bin/env bash
#SBATCH -J q38-env-build
#SBATCH -A etur92
#SBATCH -q acc_ehpc
#SBATCH -t 00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=none

# Create the isolated qwen38_inference environment offline (runbook section 14).
# CPU-only job: one node, one task, four CPUs, no GPU, 30 minutes.
#
# Required env: DEPLOYMENT_ID.
# Optional: PROJECT_ROOT, DEPLOY_ROOT, MODEL_DIR, WHEELHOUSE_DIR, VENV_DIR,
# MODEL_ID, MODEL_REVISION, SOURCE_COMMIT.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
DEPLOY_ROOT="${DEPLOY_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/deployments/qwen38_inference}"
MODEL_DIR="${MODEL_DIR:-/gpfs/projects/etur92/ozu647717/models/Qwen3.8-27B}"
WHEELHOUSE_DIR="${WHEELHOUSE_DIR:-/gpfs/projects/etur92/ozu647717/wheelhouses/qwen38_vllm_0.25.1_tf5.8.0_cu130_py310}"
VENV_DIR="${VENV_DIR:-/gpfs/projects/etur92/ozu647717/venvs/qwen38_inference}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3.8-27B}"
MODEL_REVISION="${MODEL_REVISION:-1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0}"
SOURCE_COMMIT="${SOURCE_COMMIT:-unknown}"

DEPLOYMENT_ID="${DEPLOYMENT_ID:?Set DEPLOYMENT_ID}"
export MODEL_ID MODEL_REVISION SOURCE_COMMIT
ENV_DIR="$DEPLOY_ROOT/$DEPLOYMENT_ID/environment"
LOG_DIR="$DEPLOY_ROOT/$DEPLOYMENT_ID/logs"
mkdir -p "$ENV_DIR" "$LOG_DIR"
ART_LOG="$LOG_DIR/env_build_${SLURM_JOB_ID:-local}.log"

exec > >(tee -a "$ART_LOG") 2>&1

echo "========================================"
echo "Qwen3.8 environment build job"
echo "Timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-}"
echo "Hostname: $(hostname)"
echo "Deployment: $DEPLOYMENT_ID"
echo "========================================"

module purge
module load bsc/1.0
module load miniforge/24.3.0-0

BOOTSTRAP_PY="$(command -v python)"
if ! "$BOOTSTRAP_PY" -c 'import sys; raise SystemExit(sys.version_info[:2] != (3, 10))'; then
  BOOTSTRAP_PY=/gpfs/projects/etur92/ozu647717/venvs/qwen36_translation/bin/python
fi
"$BOOTSTRAP_PY" -c 'import sys; assert sys.version_info[:2] == (3, 10)'
echo "Bootstrap python: $BOOTSTRAP_PY"

if [ -e "$VENV_DIR" ]; then
  echo "FAILED: $VENV_DIR already exists; refusing to reinstall" >&2
  exit 1
fi

"$BOOTSTRAP_PY" -m venv --copies "$VENV_DIR"
echo "Created venv: $VENV_DIR"

export PIP_NO_INDEX=1
"$VENV_DIR/bin/python" -m pip install \
  --no-index \
  --find-links "$WHEELHOUSE_DIR/wheels" \
  --require-hashes \
  --requirement "$WHEELHOUSE_DIR/requirements.lock"
echo "Offline install completed"

"$VENV_DIR/bin/python" -VV > "$ENV_DIR/python_version.txt"
"$VENV_DIR/bin/python" -m pip freeze > "$ENV_DIR/pip_freeze.txt"
"$VENV_DIR/bin/python" -m pip check > "$ENV_DIR/pip_check.txt"
if ! grep -q "No broken requirements found" "$ENV_DIR/pip_check.txt"; then
  echo "FAILED: pip check reports broken requirements" >&2
  exit 1
fi

"$VENV_DIR/bin/python" - <<'PY' > "$ENV_DIR/runtime_versions.json"
import json
import sys
import torch
import torchaudio
import torchvision
import transformers
import vllm
import huggingface_hub
import openai

print(json.dumps({
    "python_major": sys.version_info.major,
    "python_minor": sys.version_info.minor,
    "vllm": vllm.__version__,
    "transformers": transformers.__version__,
    "torch": torch.__version__.split("+")[0],
    "torchvision": torchvision.__version__,
    "torchaudio": torchaudio.__version__,
    "openai": openai.__version__,
    "huggingface_hub": huggingface_hub.__version__,
    "cuda_runtime": torch.version.cuda,
    "host": __import__("socket").gethostname(),
    "slurm_job_id": __import__("os").environ.get("SLURM_JOB_ID", ""),
    "model_id": __import__("os").environ["MODEL_ID"],
    "model_revision": __import__("os").environ["MODEL_REVISION"],
    "source_commit": __import__("os").environ["SOURCE_COMMIT"],
}, indent=2))
PY

export MODEL_DIR WHEELHOUSE_DIR ENV_DIR
"$VENV_DIR/bin/python" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def manifest_hash(manifest_path):
    lines = []
    with open(manifest_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if line.strip():
                lines.append(line)
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()

env_dir = Path(os.environ["ENV_DIR"])
model_manifest = Path(os.environ["MODEL_DIR"]) / "SHA256SUMS"
wheelhouse_manifest = Path(os.environ["WHEELHOUSE_DIR"]) / "SHA256SUMS"
assert model_manifest.is_file(), "model SHA256SUMS missing"
assert wheelhouse_manifest.is_file(), "wheelhouse SHA256SUMS missing"
(env_dir / "wheelhouse_sha256.txt").write_text(manifest_hash(wheelhouse_manifest) + "\n", encoding="utf-8")
(env_dir / "model_sha256.txt").write_text(manifest_hash(model_manifest) + "\n", encoding="utf-8")
# CPU nodes have no GPU; the first allocated GPU job writes the driver probe.
(env_dir / "driver_probe_pending.txt").write_text("driver probe pending: GPU job required\n", encoding="utf-8")
print("manifest hashes written")
PY

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
"$VENV_DIR/bin/python" - <<'PY'
import torch, transformers, vllm, openai, huggingface_hub
assert torch.__version__.split("+")[0] == "2.11.0"
assert transformers.__version__ == "5.8.0"
assert vllm.__version__ == "0.25.1"
assert openai.__version__ == "3.2.0"
assert huggingface_hub.__version__ == "1.28.0"
print("offline import check passed")
PY

"$VENV_DIR/bin/python" - <<'PY' > "$ENV_DIR/environment_acceptance.json"
import json
import os
print(json.dumps({
    "passed": True,
    "python_major": 3,
    "python_minor": 10,
    "offline_install": True,
    "pip_check": "No broken requirements found",
    "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
    "host": __import__("socket").gethostname(),
}, indent=2))
PY

echo "OK: environment build finished for $DEPLOYMENT_ID"
echo "Artifacts: $ENV_DIR"
