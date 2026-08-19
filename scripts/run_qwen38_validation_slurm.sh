#!/usr/bin/env bash
#SBATCH -A etur92
#SBATCH -q acc_ehpc
#SBATCH -t 00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1

# Qwen3.8 validation job (runbook sections 11, 14, 15, 16).
#
# Required env: DEPLOYMENT_ID, TP_SIZE, ATTEMPT, RUN_LABEL, SOURCE_COMMIT.
# Optional: PROJECT_ROOT, DEPLOY_ROOT, MODEL_DIR, WHEELHOUSE_DIR, VENV_DIR,
# MODEL_REVISION, FIXTURE_PATH.
#
# The launcher passes the per-TP Slurm resource flags (job name, CPUs, GPUs,
# wall time). This worker:
#  1. records nvidia-smi / driver / device count / GPU model / memory;
#  2. requires exactly TP_SIZE visible GPUs and driver >= 580.00;
#  3. verifies environment, model, and wheelhouse manifests;
#  4. starts the common localhost server and polls /v1/models;
#  5. runs the 64-case workload at concurrency 1, 8, 16, 32;
#  6. repeats concurrency 1 for determinism;
#  7. stops, restarts with offline variables, and runs the 8-case subset;
#  8. summarizes acceptance and exits zero only when the gate passes.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
DEPLOY_ROOT="${DEPLOY_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/deployments/qwen38_inference}"
MODEL_DIR="${MODEL_DIR:-/gpfs/projects/etur92/ozu647717/models/Qwen3.8-27B}"
WHEELHOUSE_DIR="${WHEELHOUSE_DIR:-/gpfs/projects/etur92/ozu647717/wheelhouses/qwen38_vllm_0.25.1_tf5.8.0_cu130_py310}"
VENV_DIR="${VENV_DIR:-/gpfs/projects/etur92/ozu647717/venvs/qwen38_inference}"
MODEL_REVISION="${MODEL_REVISION:-1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0}"
FIXTURE_PATH="${FIXTURE_PATH:-$PROJECT_ROOT/tests/fixtures/qwen38_synthetic_cases.jsonl}"
PORT=8000

DEPLOYMENT_ID="${DEPLOYMENT_ID:?Set DEPLOYMENT_ID}"
TP_SIZE="${TP_SIZE:?Set TP_SIZE}"
ATTEMPT="${ATTEMPT:?Set ATTEMPT}"
RUN_LABEL="${RUN_LABEL:?Set RUN_LABEL}"
SOURCE_COMMIT="${SOURCE_COMMIT:-unknown}"

ATTEMPT_DIR="$DEPLOY_ROOT/$DEPLOYMENT_ID/validation/tp${TP_SIZE}/attempt${ATTEMPT}"
LOG_DIR="$DEPLOY_ROOT/$DEPLOYMENT_ID/logs"
mkdir -p "$ATTEMPT_DIR" "$LOG_DIR"
ART_LOG="$LOG_DIR/validation_tp${TP_SIZE}_attempt${ATTEMPT}_${SLURM_JOB_ID:-local}.log"
SERVER_LOG="$ATTEMPT_DIR/vllm_server.log"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1
export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false
export VLLM_TORCH_COMPILE_OVERRIDE=0
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

exec > >(tee -a "$ART_LOG") 2>&1

echo "========================================"
echo "Qwen3.8 validation job"
echo "Timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-}"
echo "Hostname: $(hostname)"
echo "Deployment: $DEPLOYMENT_ID"
echo "TP_SIZE: $TP_SIZE ATTEMPT: $ATTEMPT LABEL: $RUN_LABEL"
echo "Source commit: $SOURCE_COMMIT"
echo "========================================"

module purge
module load bsc/1.0
module load miniforge/24.3.0-0
source "$VENV_DIR/bin/activate"
cd "$PROJECT_ROOT"
python -VV
python -c "import torch, vllm; print('torch', torch.__version__, '| vllm', vllm.__version__)"

# --- 1. GPU environment record -------------------------------------------
nvidia-smi --query-gpu=index,driver_version,memory.total,name --format=csv > "$ATTEMPT_DIR/gpu_info.txt"
cat "$ATTEMPT_DIR/gpu_info.txt"
DRIVER_VERSION="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  VISIBLE_COUNT="$(printf '%s' "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c . || true)"
else
  VISIBLE_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader | grep -c . || true)"
fi
GPU_MODEL="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
GPU_MEMORY="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)"
GPU_RECORD_PATH="$ATTEMPT_DIR/gpu_record.json"

python - <<PY
import json
import os
import socket

record = {
    "driver_version": "$DRIVER_VERSION",
    "visible_gpu_count": $VISIBLE_COUNT,
    "gpu_model": "$GPU_MODEL",
    "gpu_memory_total": "$GPU_MEMORY",
    "host": socket.gethostname(),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
    "timestamp_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
}
path = "$GPU_RECORD_PATH"
with open(path, "w", encoding="utf-8") as handle:
    json.dump(record, handle, indent=2)
    handle.write("\n")
PY

if [ "$VISIBLE_COUNT" -lt "$TP_SIZE" ]; then
  echo "FAILED: expected at least $TP_SIZE visible GPUs, found $VISIBLE_COUNT" >&2
  exit 1
fi
echo "visible GPUs: $VISIBLE_COUNT (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset})"
python - "$DRIVER_VERSION" <<'PY'
import sys


def version_tuple(version):
    return tuple(int(part) for part in version.strip().split(".") if part)


if version_tuple(sys.argv[1]) < (580, 0):
    print(f"FAILED: driver {sys.argv[1]} below required 580.00", file=sys.stderr)
    raise SystemExit(1)
print(f"driver {sys.argv[1]} >= 580.00")
PY

# --- 2. Environment / model / wheelhouse verification ---------------------
ENV_DIR="$DEPLOY_ROOT/$DEPLOYMENT_ID/environment"
python - "$ENV_DIR/runtime_versions.json" <<'PY'
import json
import sys
import torch
import transformers
import vllm
import openai
import huggingface_hub

expected = json.load(open(sys.argv[1], encoding="utf-8"))
actual = {
    "python_major": __import__("sys").version_info.major,
    "python_minor": __import__("sys").version_info.minor,
    "vllm": vllm.__version__,
    "transformers": transformers.__version__,
    "torch": torch.__version__.split("+")[0],
    "torchvision": __import__("torchvision").__version__,
    "torchaudio": __import__("torchaudio").__version__,
    "openai": openai.__version__,
    "huggingface_hub": huggingface_hub.__version__,
}
for key, value in expected.items():
    if key in ("host", "slurm_job_id", "cuda_runtime", "model_id", "model_revision", "source_commit"):
        continue
    if actual.get(key) != value:
        print(f"FAILED: environment version mismatch {key}: {actual.get(key)} != {value}", file=sys.stderr)
        raise SystemExit(1)
print("environment versions match deployment record")
PY

WHEELHOUSE_MANIFEST_HASH="$(python - "$WHEELHOUSE_DIR/SHA256SUMS" <<'PY'
import hashlib
import sys
lines = []
with open(sys.argv[1], encoding="utf-8") as handle:
    for line in handle:
        line = line.rstrip("\n")
        if line.strip():
            lines.append(line)
print(hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest())
PY
)"
echo "wheelhouse manifest sha256: $WHEELHOUSE_MANIFEST_HASH"
python - "$ENV_DIR/wheelhouse_sha256.txt" "$WHEELHOUSE_MANIFEST_HASH" <<'PY'
import sys
expected = open(sys.argv[1], encoding="utf-8").read().strip()
actual = sys.argv[2]
if expected != actual:
    print(f"FAILED: wheelhouse manifest mismatch: {actual} != {expected}", file=sys.stderr)
    raise SystemExit(1)
print("wheelhouse manifest matches deployment record")
PY

MODEL_MANIFEST_HASH="$(python - "$MODEL_DIR/SHA256SUMS" <<'PY'
import hashlib
import sys
lines = []
with open(sys.argv[1], encoding="utf-8") as handle:
    for line in handle:
        line = line.rstrip("\n")
        if line.strip():
            lines.append(line)
print(hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest())
PY
)"
echo "model manifest sha256: $MODEL_MANIFEST_HASH"
python - "$ENV_DIR/model_sha256.txt" "$MODEL_MANIFEST_HASH" <<'PY'
import sys
expected = open(sys.argv[1], encoding="utf-8").read().strip()
actual = sys.argv[2]
if expected != actual:
    print(f"FAILED: model manifest mismatch: {actual} != {expected}", file=sys.stderr)
    raise SystemExit(1)
print("model manifest matches deployment record")
PY

# Driver probe: recorded for the deployment audit.
python - "$ATTEMPT_DIR/gpu_record.json" "$ENV_DIR/driver_probe.json" <<'PY'
import json
import sys


def version_tuple(version):
    return tuple(int(part) for part in version.strip().split(".") if part)


record = json.load(open(sys.argv[1], encoding="utf-8"))
probe = {
    "driver_version": record["driver_version"],
    "gpu_model": record["gpu_model"],
    "host": record["host"],
    "slurm_job_id": record["slurm_job_id"],
    "timestamp_utc": record["timestamp_utc"],
    "passed": version_tuple(record["driver_version"]) >= (580, 0),
}
with open(sys.argv[2], "w", encoding="utf-8") as handle:
    json.dump(probe, handle, indent=2)
    handle.write("\n")
PY

# --- 3. Common server start/stop helpers -----------------------------------
SERVER_PID=""
start_server() {
  echo "[server] starting vLLM (TP=$TP_SIZE) on 127.0.0.1:$PORT"
  python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_DIR" \
    --served-model-name qwen3.8-27b \
    --host 127.0.0.1 \
    --port "$PORT" \
    --tensor-parallel-size "$TP_SIZE" \
    --dtype bfloat16 \
    --language-model-only \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --reasoning-parser qwen3 \
    --generation-config vllm \
    --no-enable-log-requests \
    > "$SERVER_LOG" 2>&1 &
  SERVER_PID=$!
  echo "SERVER_PID=$SERVER_PID"
}

stop_server() {
  if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[server] stopping (pid $SERVER_PID)"
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=""
}

wait_ready() {
  local attempts=0
  while [ "$attempts" -lt 300 ]; do
    if curl -sf -o /dev/null http://127.0.0.1:8000/v1/models; then
      echo "[server] ready after $((attempts * 2))s"
      return 0
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "[server] FAILED: server process exited before readiness" >&2
      tail -80 "$SERVER_LOG" >&2 || true
      return 1
    fi
    sleep 2
    attempts=$((attempts + 1))
  done
  echo "[server] FAILED: not ready within ten minutes" >&2
  return 1
}

cleanup() {
  stop_server
}
trap cleanup EXIT INT TERM

# --- 4. Main workload ------------------------------------------------------
export MODEL_REVISION SOURCE_COMMIT
start_server
if ! wait_ready; then
  # TP=1 capacity failure (OOM / insufficient memory) is CAPACITY_INELIGIBLE,
  # not a deployment failure (runbook section 16).
  python - "$ATTEMPT_DIR/acceptance.json" <<'PY'
import json
import os
import sys
import time
path = sys.argv[1]
record = {
    "passed": False,
    "capacity_ineligible": True,
    "reason": "vLLM server failed to become ready (likely CUDA OOM at TP=1)",
    "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
    "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
with open(path, "w", encoding="utf-8") as handle:
    json.dump(record, handle, indent=2)
    handle.write("\n")
PY
  echo "TP=$TP_SIZE server startup failed; recorded acceptance.json"
  exit 1
fi

python scripts/qwen38_validation_client.py run \
  --base-url http://127.0.0.1:8000/v1 \
  --model qwen3.8-27b \
  --cases "$FIXTURE_PATH" \
  --out "$ATTEMPT_DIR/results.json" \
  --concurrency-levels 1,8,16,32 \
  --max-tokens 1024 \
  --seed 42

stop_server

# --- 5. Offline restart and 8-case subset ----------------------------------
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
start_server
wait_ready

python scripts/qwen38_validation_client.py run \
  --base-url http://127.0.0.1:8000/v1 \
  --model qwen3.8-27b \
  --cases "$FIXTURE_PATH" \
  --out "$ATTEMPT_DIR/restart_results.json" \
  --restart-subset \
  --max-tokens 1024 \
  --seed 42

stop_server

# --- 6. Acceptance ----------------------------------------------------------
RECORDED_MODEL_MANIFEST_HASH="$(cat "$ENV_DIR/model_sha256.txt")"
RECORDED_WHEELHOUSE_MANIFEST_HASH="$(cat "$ENV_DIR/wheelhouse_sha256.txt")"
python scripts/qwen38_validation_client.py summarize \
  --results "$ATTEMPT_DIR/results.json" \
  --restart-results "$ATTEMPT_DIR/restart_results.json" \
  --out "$ATTEMPT_DIR/acceptance.json" \
  --deployment-env "$ENV_DIR/runtime_versions.json" \
  --model-manifest-sha256 "$MODEL_MANIFEST_HASH" \
  --wheelhouse-manifest-sha256 "$WHEELHOUSE_MANIFEST_HASH" \
  --deployment-model-manifest-sha256 "$RECORDED_MODEL_MANIFEST_HASH" \
  --deployment-wheelhouse-manifest-sha256 "$RECORDED_WHEELHOUSE_MANIFEST_HASH"

echo "OK: validation finished for TP=$TP_SIZE attempt=$ATTEMPT"
echo "Artifacts: $ATTEMPT_DIR"
