#!/usr/bin/env bash
#SBATCH -A etur92
#SBATCH -q acc_ehpc
#SBATCH -t 02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1

# Qwen3.8 private Turkish question-recovery job (runbook sections 18-21).
#
# Required env: DEPLOYMENT_ID, TURKISH_RUN_ID, ANALYSIS_ATTEMPT, SELECTED_TP,
# SOURCE_COMMIT, SELECTION_FILE. The source tree supplies the immutable prompt
# and episode-safety policy identities; they are recorded in every ledger event.
# Optional: PROJECT_ROOT, DEPLOY_ROOT, MODEL_DIR, WHEELHOUSE_DIR, VENV_DIR,
# MODEL_REVISION, TRANSCRIPT_PATH.
#
# The launcher passes the resource flags: CPU = 20 * selected TP, GPUs =
# selected TP, two-hour wall time. The worker serves the model on localhost
# only, prepares the 135 subject sequences, infers question families, runs
# the two-level consolidation, renders the compact tables, runs the remote
# audit, and stops the server. Every scientific attempt has a fresh immutable
# run root.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
DEPLOY_ROOT="${DEPLOY_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/deployments/qwen38_inference}"
MODEL_DIR="${MODEL_DIR:-/gpfs/projects/etur92/ozu647717/models/Qwen3.8-27B}"
WHEELHOUSE_DIR="${WHEELHOUSE_DIR:-/gpfs/projects/etur92/ozu647717/wheelhouses/qwen38_vllm_0.25.1_tf5.8.0_cu130_py310}"
VENV_DIR="${VENV_DIR:-/gpfs/projects/etur92/ozu647717/venvs/qwen38_inference}"
MODEL_REVISION="${MODEL_REVISION:-1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0}"
TRANSCRIPT_PATH="${TRANSCRIPT_PATH:-/gpfs/projects/etur92/ozu647717/AudioLLM/private_inputs/turkish_question_recovery/whisper_transcripts_qwen3_asr.jsonl}"

DEPLOYMENT_ID="${DEPLOYMENT_ID:?Set DEPLOYMENT_ID}"
TURKISH_RUN_ID="${TURKISH_RUN_ID:?Set TURKISH_RUN_ID}"
ANALYSIS_ATTEMPT="${ANALYSIS_ATTEMPT:?Set ANALYSIS_ATTEMPT}"
SELECTED_TP="${SELECTED_TP:?Set SELECTED_TP}"
SOURCE_COMMIT="${SOURCE_COMMIT:?Set SOURCE_COMMIT}"
SELECTION_FILE="${SELECTION_FILE:?Set SELECTION_FILE}"
SUPERSEDES_JOB_IDS="${SUPERSEDES_JOB_IDS:-}"

RUN_ROOT="$PROJECT_ROOT/outputs/turkish_question_recovery/$TURKISH_RUN_ID"
RESTRICTED="$RUN_ROOT/restricted"
LOG_DIR="$DEPLOY_ROOT/$DEPLOYMENT_ID/logs"
if [[ ! "$TURKISH_RUN_ID" =~ ^q38tr_[0-9a-f]{12}_attempt[0-9]+$ ]]; then
  echo "FAILED: invalid TURKISH_RUN_ID=$TURKISH_RUN_ID" >&2
  exit 1
fi
if [[ "$TURKISH_RUN_ID" =~ _attempt([0-9]+)$ ]] && [ "${BASH_REMATCH[1]}" != "$ANALYSIS_ATTEMPT" ]; then
  echo "FAILED: ANALYSIS_ATTEMPT=$ANALYSIS_ATTEMPT does not match TURKISH_RUN_ID=$TURKISH_RUN_ID" >&2
  exit 1
fi
if [ -e "$RUN_ROOT" ]; then
  echo "FAILED: refusing to reuse existing Turkish run root $RUN_ROOT" >&2
  exit 1
fi
if [ ! -f "$SELECTION_FILE" ]; then
  echo "FAILED: selection v2 missing at $SELECTION_FILE" >&2
  exit 1
fi
mkdir -p "$LOG_DIR"
ART_LOG="$LOG_DIR/turkish_${TURKISH_RUN_ID}_${SLURM_JOB_ID:-local}.log"
SERVER_LOG="$LOG_DIR/turkish_${TURKISH_RUN_ID}_${SLURM_JOB_ID:-local}_vllm_server.log"
GPU_INFO_PATH="$LOG_DIR/turkish_${TURKISH_RUN_ID}_${SLURM_JOB_ID:-local}_gpu_info.txt"
GPU_RECORD_PATH="$LOG_DIR/turkish_${TURKISH_RUN_ID}_${SLURM_JOB_ID:-local}_gpu_record.json"
LEDGER_PATH="$DEPLOY_ROOT/$DEPLOYMENT_ID/turkish_jobs.jsonl"

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
echo "Qwen3.8 Turkish question-recovery job"
echo "Timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-}"
echo "Hostname: $(hostname)"
echo "Deployment: $DEPLOYMENT_ID"
echo "Turkish run ID: $TURKISH_RUN_ID attempt=$ANALYSIS_ATTEMPT"
echo "Selected TP: $SELECTED_TP"
echo "Source commit: $SOURCE_COMMIT"
echo "Selection file: $SELECTION_FILE"
echo "Run root: $RUN_ROOT"
echo "========================================"

module purge
module load bsc/1.0
module load miniforge/24.3.0-0
source "$VENV_DIR/bin/activate"
cd "$PROJECT_ROOT"
ORIGIN_MAIN_SHA="${LOCAL_ORIGIN_MAIN_SHA:-$(git rev-parse origin/main 2>/dev/null || true)}"
REMOTE_PROVENANCE_SHA="$(tr -d '[:space:]' < "$PROJECT_ROOT/.provenance/git_commit.txt" 2>/dev/null || true)"
if [ "$SOURCE_COMMIT" != "$ORIGIN_MAIN_SHA" ] || [ "$SOURCE_COMMIT" != "$REMOTE_PROVENANCE_SHA" ]; then
  echo "FAILED: source identity mismatch (SOURCE_COMMIT=$SOURCE_COMMIT origin/main=$ORIGIN_MAIN_SHA provenance=$REMOTE_PROVENANCE_SHA)" >&2
  exit 1
fi
if [[ "$TURKISH_RUN_ID" != "q38tr_${SOURCE_COMMIT:0:12}_attempt${ANALYSIS_ATTEMPT}" ]]; then
  echo "FAILED: TURKISH_RUN_ID is not source/attempt matched" >&2
  exit 1
fi
SELECTION_TP="$(python - "$SELECTION_FILE" <<'PY'
import json
import sys
selection = json.load(open(sys.argv[1], encoding="utf-8"))
if selection.get("selection_version") != 2:
    raise SystemExit("selection file is not serving_selection_v2.json")
selected = selection.get("selected_tp")
if selected not in (1, 2, 4):
    raise SystemExit("selection v2 has invalid selected_tp")
print(selected)
PY
)"
if [ "$SELECTED_TP" != "$SELECTION_TP" ]; then
  echo "FAILED: selected TP $SELECTED_TP does not match selection v2 $SELECTION_TP" >&2
  exit 1
fi
SELECTION_FILE_SHA256="$(sha256sum "$SELECTION_FILE" | awk '{print $1}')"
mapfile -t PROMPT_METADATA_LINES < <(python - "$MODEL_REVISION" <<'PY'
import sys
from src.qwen38.turkish_questions import (
    EPISODE_SAFETY_POLICY_SHA256,
    EPISODE_SAFETY_POLICY_VERSION,
    PROMPT_VERSION,
    prompt_contract_sha256,
)
print(prompt_contract_sha256(model_revision=sys.argv[1]))
print(PROMPT_VERSION)
print(EPISODE_SAFETY_POLICY_VERSION)
print(EPISODE_SAFETY_POLICY_SHA256)
PY
)
PROMPT_CONTRACT_SHA256="${PROMPT_METADATA_LINES[0]}"
PROMPT_VERSION="${PROMPT_METADATA_LINES[1]}"
EPISODE_SAFETY_POLICY_VERSION="${PROMPT_METADATA_LINES[2]}"
EPISODE_SAFETY_POLICY_SHA256="${PROMPT_METADATA_LINES[3]}"
echo "Prompt: $PROMPT_VERSION"
echo "Episode safety policy: $EPISODE_SAFETY_POLICY_VERSION ($EPISODE_SAFETY_POLICY_SHA256)"

append_ledger_event() {
  local stage="$1" state="$2" exit_code="$3" run_manifest_sha256="${4:-}"
  python - "$LEDGER_PATH" "$TURKISH_RUN_ID" "$ANALYSIS_ATTEMPT" "$DEPLOYMENT_ID" "$stage" "${SLURM_JOB_ID:-}" "$state" "$exit_code" "$SOURCE_COMMIT" "$PROMPT_VERSION" "$EPISODE_SAFETY_POLICY_VERSION" "$EPISODE_SAFETY_POLICY_SHA256" "$PROMPT_CONTRACT_SHA256" "$run_manifest_sha256" "$SELECTED_TP" "$SELECTION_FILE" "$SELECTION_FILE_SHA256" "${SUPERSEDES_JOB_IDS:-}" <<'PY'
import json
import os
import sys
import time

(path, run_id, attempt, deployment_id, stage, job_id, state, exit_code,
source_commit, prompt_version, policy_version, policy_sha256, prompt_contract,
 manifest_hash, selected_tp, selection_file, selection_sha256, supersedes) = sys.argv[1:]
record = {
    "turkish_run_id": run_id,
    "analysis_attempt": int(attempt),
    "deployment_id": deployment_id,
    "stage": stage,
    "slurm_job_id": job_id,
    "state": state,
    "exit_code": exit_code,
    "source_commit": source_commit,
    "prompt_version": prompt_version,
    "episode_safety_policy_version": policy_version,
    "episode_safety_policy_sha256": policy_sha256,
    "prompt_contract_sha256": prompt_contract,
    "run_manifest_sha256": manifest_hash or None,
    "selected_tp": int(selected_tp),
    "selection_file": selection_file,
    "selection_file_sha256": selection_sha256,
    "supersedes_job_ids": [item for item in supersedes.split(",") if item],
    "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
with open(path, "a", encoding="utf-8") as handle:
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
PY
}

append_ledger_event "start" "STARTED" "unknown"
python -VV
python -c "import torch, vllm; print('torch', torch.__version__, '| vllm', vllm.__version__)"

# --- GPU environment --------------------------------------------------------
nvidia-smi --query-gpu=index,driver_version,memory.total,name --format=csv > "$GPU_INFO_PATH"
chmod 600 "$GPU_INFO_PATH"
DRIVER_VERSION="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  VISIBLE_COUNT="$(printf '%s' "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c . || true)"
else
  VISIBLE_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader | grep -c . || true)"
fi
GPU_MODEL="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
GPU_MEMORY="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)"
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
chmod 600 "$GPU_RECORD_PATH"

if [ "$VISIBLE_COUNT" -ne "$SELECTED_TP" ]; then
  echo "FAILED: expected exactly $SELECTED_TP visible GPUs, found $VISIBLE_COUNT" >&2
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
        print(f"FAILED: environment version mismatch {key}", file=sys.stderr)
        raise SystemExit(1)
print("environment versions match deployment record")
PY

# --- Server lifecycle --------------------------------------------------------
SERVER_PID=""
start_server() {
  echo "[server] starting vLLM (TP=$SELECTED_TP) on 127.0.0.1:8000"
  python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_DIR" \
    --served-model-name qwen3.8-27b \
    --host 127.0.0.1 \
    --port 8000 \
    --tensor-parallel-size "$SELECTED_TP" \
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

start_server
wait_ready

# --- Pipeline ----------------------------------------------------------------
echo "[stage:prepare]"
python scripts/qwen38_turkish_question_recovery.py prepare \
  --deployment-id "$DEPLOYMENT_ID" \
  --turkish-run-id "$TURKISH_RUN_ID" \
  --run-dir "$RUN_ROOT" \
  --transcript "$TRANSCRIPT_PATH" \
  --source-commit "$SOURCE_COMMIT" \
  --model-revision "$MODEL_REVISION" \
  --selection-file "$SELECTION_FILE" \
  --selected-tp "$SELECTED_TP" \
  --analysis-attempt "$ANALYSIS_ATTEMPT" \
  --supersedes-job-ids "${SUPERSEDES_JOB_IDS:-}"
RUN_MANIFEST_SHA256="$(sha256sum "$RUN_ROOT/run_manifest.json" | awk '{print $1}')"
append_ledger_event "prepare" "COMPLETED" "0:0" "$RUN_MANIFEST_SHA256"

echo "[stage:infer-subjects]"
python scripts/qwen38_turkish_question_recovery.py infer-subjects \
  --deployment-id "$DEPLOYMENT_ID" \
  --turkish-run-id "$TURKISH_RUN_ID" \
  --run-dir "$RUN_ROOT" \
  --source-commit "$SOURCE_COMMIT" \
  --model-revision "$MODEL_REVISION" \
  --prepared "$RESTRICTED/prepared_sequences.jsonl" \
  --inferences-dir "$RESTRICTED/subject_inferences" \
  --base-url http://127.0.0.1:8000/v1 \
  --model qwen3.8-27b \
  --concurrency 8 \
  --seed 42 \
  --max-tokens 2048
append_ledger_event "infer-subjects" "COMPLETED" "0:0" "$RUN_MANIFEST_SHA256"

echo "[stage:consolidate]"
python scripts/qwen38_turkish_question_recovery.py consolidate \
  --deployment-id "$DEPLOYMENT_ID" \
  --turkish-run-id "$TURKISH_RUN_ID" \
  --run-dir "$RUN_ROOT" \
  --source-commit "$SOURCE_COMMIT" \
  --model-revision "$MODEL_REVISION" \
  --inferences-dir "$RESTRICTED/subject_inferences" \
  --consolidation-dir "$RESTRICTED/consolidation_batches" \
  --base-url http://127.0.0.1:8000/v1 \
  --model qwen3.8-27b \
  --seed 42 \
  --max-tokens 2048 \
  --tokenizer-path "$MODEL_DIR"
append_ledger_event "consolidate" "COMPLETED" "0:0" "$RUN_MANIFEST_SHA256"

echo "[stage:render]"
python scripts/qwen38_turkish_question_recovery.py render \
  --deployment-id "$DEPLOYMENT_ID" \
  --turkish-run-id "$TURKISH_RUN_ID" \
  --run-dir "$RUN_ROOT" \
  --source-commit "$SOURCE_COMMIT" \
  --model-revision "$MODEL_REVISION" \
  --inferences-dir "$RESTRICTED/subject_inferences" \
  --consolidation-dir "$RESTRICTED/consolidation_batches" \
  --final-merge "$RESTRICTED/consolidation_batches/final_merge.json"
append_ledger_event "render" "COMPLETED" "0:0" "$RUN_MANIFEST_SHA256"

stop_server

# --- Audit (remote; restricted evidence present) ----------------------------
# Terminal Slurm accounting (state, exit code, timestamps) is unknown inside
# the job; the launcher regenerates audit.json after sacct reports COMPLETED.
python scripts/qwen38_audit.py turkish \
  --run-dir "$RUN_ROOT" \
  --turkish-run-id "$TURKISH_RUN_ID" \
  --transcript "$TRANSCRIPT_PATH" \
  --deploy-dir "$DEPLOY_ROOT" \
  --deployment-id "$DEPLOYMENT_ID" \
  --model-dir "$MODEL_DIR" \
  --wheelhouse-dir "$WHEELHOUSE_DIR" \
  --source-commit "$SOURCE_COMMIT" \
  --selection-file "$SELECTION_FILE" \
  --out "$RUN_ROOT/audit_pre_slurm.json"

chmod 600 "$RUN_ROOT/audit_pre_slurm.json"
append_ledger_event "audit-pre-slurm" "COMPLETED" "0:0" "$RUN_MANIFEST_SHA256"
chmod 644 "$RUN_ROOT/turkish_inferred_questions.csv" "$RUN_ROOT/turkish_inferred_questions.json" "$RUN_ROOT/turkish_inferred_questions.md"

echo "OK: Turkish question recovery finished for $DEPLOYMENT_ID"
echo "Artifacts: $RUN_ROOT"
