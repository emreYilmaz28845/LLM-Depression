#!/usr/bin/env bash
#SBATCH -J translate-dataset
#SBATCH -A etur92
#SBATCH -q acc_ehpc
#SBATCH -t 24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:2
#SBATCH -o /dev/null
#SBATCH -e /dev/null

# Full pipeline for one dataset: build the canonical native manifest (CPU),
# export label-free translation units, start the Qwen3.6-27B vLLM server on two
# H100s, translate deterministically with resumable incremental flush, validate
# (coverage, hashes, English-only, invariants, NLLB comparison, consistency,
# optional Qwen verifier), stop the server. Artifacts live outside the
# repository under $TRANSLATION_ROOT. Resumable: re-running the same job skips
# completed candidates and only validates missing pieces.
#
# Required env: DATASET (cmdc|turkish|d3tec|androids_interview), MANIFEST_CONFIG.
# Optional: TRANSLATION_ROOT, TRANSLATION_RUN_ROOT, MODEL_DIR, VENV_QWEN36,
# VENV_MAIN, QWEN36_REVISION, NLLB_MODEL, REVIEWED, FORCE_EXPORT, FORCE_RESYNC,
# SKIP_MANIFEST, SKIP_SERVER, UNIT_LIMIT, EXPECTED_UNIT_COUNT, REQUIRE_COMPLETE.

set -euo pipefail

DATASET="${DATASET:?Set DATASET=cmdc|turkish|d3tec|androids_interview}"
MANIFEST_CONFIG="${MANIFEST_CONFIG:?Set MANIFEST_CONFIG to the native manifest config}"
PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
TRANSLATION_ROOT="${TRANSLATION_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/translations}"
TRANSLATION_ROOT="${TRANSLATION_ROOT%/}"
MODEL_DIR="${MODEL_DIR:-/gpfs/projects/etur92/ozu647717/models/Qwen3.6-27B}"
VENV_QWEN36="${VENV_QWEN36:-/gpfs/projects/etur92/ozu647717/venvs/qwen36_translation/bin/activate}"
VENV_MAIN="${VENV_MAIN:-/gpfs/projects/etur92/ozu647717/venvs/qwen_mn5_rebuilt/bin/activate}"
QWEN36_REVISION="${QWEN36_REVISION:-6a9e13bd6fc8f0983b9b99948120bc37f49c13e9}"
NLLB_MODEL="${NLLB_MODEL:-/gpfs/projects/etur92/ozu647717/models/nllb-200-distilled-600M}"
REVIEWED="${REVIEWED:-}"
FORCE_EXPORT="${FORCE_EXPORT:-0}"
FORCE_RESYNC="${FORCE_RESYNC:-0}"
SKIP_MANIFEST="${SKIP_MANIFEST:-0}"
SKIP_SERVER="${SKIP_SERVER:-0}"
MAX_SOURCE_TOKENS="${MAX_SOURCE_TOKENS:-12000}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-8}"
UNIT_LIMIT="${UNIT_LIMIT:-0}"
EXPECTED_UNIT_COUNT="${EXPECTED_UNIT_COUNT:-0}"
REQUIRE_COMPLETE="${REQUIRE_COMPLETE:-0}"

RUN_ROOT="${TRANSLATION_RUN_ROOT:-$TRANSLATION_ROOT/qwen36_27b_bf16_clinical_v1/$DATASET}"
if [[ ! "$UNIT_LIMIT" =~ ^[0-9]+$ ]] || [[ ! "$EXPECTED_UNIT_COUNT" =~ ^[0-9]+$ ]]; then
  echo "UNIT_LIMIT and EXPECTED_UNIT_COUNT must be non-negative integers." >&2
  exit 1
fi
if [ "$REQUIRE_COMPLETE" != "0" ] && [ "$REQUIRE_COMPLETE" != "1" ]; then
  echo "REQUIRE_COMPLETE must be 0 or 1." >&2
  exit 1
fi
if [[ "$RUN_ROOT" != /* ]] || [[ "$RUN_ROOT" == *"/../"* ]] || [[ "$RUN_ROOT" == */.. ]] || \
   [[ "$RUN_ROOT" == "$TRANSLATION_ROOT" ]] || [[ "$RUN_ROOT/" != "$TRANSLATION_ROOT/"* ]]; then
  echo "TRANSLATION_RUN_ROOT must be an absolute child of TRANSLATION_ROOT: $RUN_ROOT" >&2
  exit 1
fi
if [ "$UNIT_LIMIT" -gt 0 ] && [ -z "${TRANSLATION_RUN_ROOT:-}" ]; then
  echo "UNIT_LIMIT requires an explicit isolated TRANSLATION_RUN_ROOT." >&2
  exit 1
fi
ART_DIR="$RUN_ROOT/logs/slurm_${SLURM_JOB_ID}"
mkdir -p "$ART_DIR" "$RUN_ROOT"

export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_TORCH_COMPILE_OVERRIDE=0

{
  echo "========================================"
  echo "Translation pipeline job: $DATASET"
  echo "Timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-}"
  echo "Hostname: $(hostname)"
  echo "Project: $PROJECT_ROOT"
  echo "Run root: $RUN_ROOT"
  echo "Unit limit: $UNIT_LIMIT"
  echo "Expected units: $EXPECTED_UNIT_COUNT"
  echo "Require complete: $REQUIRE_COMPLETE"
  echo "Model: $MODEL_DIR (revision $QWEN36_REVISION)"
  echo "========================================"
} | tee -a "$ART_DIR/job_summary.txt"

module purge
module load bsc/1.0
module load miniforge/24.3.0-0

MANIFEST_PATH=""
if [ "$SKIP_MANIFEST" != "1" ]; then
  echo "[stage:manifest] Building native manifest with $MANIFEST_CONFIG" | tee -a "$ART_DIR/job_summary.txt"
  source "$VENV_MAIN"
  python -m src.data.build_manifest --config "$MANIFEST_CONFIG" 2>&1 | tee -a "$ART_DIR/manifest_build.log"
  MANIFEST_PATH="$(source "$VENV_MAIN"; python - "$MANIFEST_CONFIG" "$DATASET" <<'PY'
import sys
from src.utils import load_yaml, resolve_project_path
config = load_yaml(sys.argv[1])
manifest_dir = resolve_project_path(config["output_dirs"]["manifest_dir"])
print(manifest_dir / f"{sys.argv[2]}_manifest.jsonl")
PY
)"
  if [ ! -f "$MANIFEST_PATH" ]; then
    echo "FAILED: manifest not found at $MANIFEST_PATH" | tee -a "$ART_DIR/job_summary.txt"
    exit 1
  fi
  echo "Native manifest: $MANIFEST_PATH" | tee -a "$ART_DIR/job_summary.txt"
else
  MANIFEST_PATH="${MANIFEST_PATH:?Set MANIFEST_PATH when SKIP_MANIFEST=1}"
fi

source "$VENV_QWEN36"
python -VV 2>&1 | tee -a "$ART_DIR/job_summary.txt"
python -c "import torch, vllm; print('torch', torch.__version__, '| vllm', vllm.__version__)" | tee -a "$ART_DIR/job_summary.txt"

if [ "$FORCE_EXPORT" = "1" ] || [ ! -f "$RUN_ROOT/units.jsonl" ]; then
  echo "[stage:export] Exporting label-free units" | tee -a "$ART_DIR/job_summary.txt"
  python -m src.translation.units \
    --dataset "$DATASET" \
    --manifest "$MANIFEST_PATH" \
    --out "$RUN_ROOT/units.jsonl" \
    --profile "$RUN_ROOT/length_profile.json" \
    --tokenizer "$MODEL_DIR" \
    --max-source-tokens "$MAX_SOURCE_TOKENS" 2>&1 | tee -a "$ART_DIR/export.log"
else
  echo "[stage:export] Reusing existing $RUN_ROOT/units.jsonl" | tee -a "$ART_DIR/job_summary.txt"
fi
if [ "$UNIT_LIMIT" -gt 0 ]; then
  CURRENT_UNIT_COUNT="$(wc -l < "$RUN_ROOT/units.jsonl")"
  if [ "$CURRENT_UNIT_COUNT" -eq "$UNIT_LIMIT" ]; then
    echo "[stage:export] Reusing already limited smoke units" | tee -a "$ART_DIR/job_summary.txt"
  else
    if [ -s "$RUN_ROOT/candidates.jsonl" ] || [ -s "$RUN_ROOT/accepted.jsonl" ]; then
      echo "FAILED: refusing to limit units in a run root that already has translation output." | tee -a "$ART_DIR/job_summary.txt"
      exit 1
    fi
    python - "$RUN_ROOT/units.jsonl" "$UNIT_LIMIT" <<'PY'
import sys
from pathlib import Path

from src.utils import read_jsonl, write_jsonl

path = Path(sys.argv[1])
limit = int(sys.argv[2])
rows = read_jsonl(path)
if len(rows) < limit:
    raise SystemExit(f"UNIT_LIMIT={limit} exceeds exported units={len(rows)}")
write_jsonl(rows[:limit], path)
PY
    echo "[stage:export] Limited isolated smoke units to $UNIT_LIMIT" | tee -a "$ART_DIR/job_summary.txt"
  fi
fi
UNIT_COUNT="$(wc -l < "$RUN_ROOT/units.jsonl")"
echo "Units: $UNIT_COUNT" | tee -a "$ART_DIR/job_summary.txt"
if [ "$EXPECTED_UNIT_COUNT" -gt 0 ] && [ "$UNIT_COUNT" -ne "$EXPECTED_UNIT_COUNT" ]; then
  echo "FAILED: units=$UNIT_COUNT expected=$EXPECTED_UNIT_COUNT" | tee -a "$ART_DIR/job_summary.txt"
  exit 1
fi

if [ "$SKIP_SERVER" = "1" ]; then
  echo "[stage:server] SKIPPED (SKIP_SERVER=1); assuming an external endpoint" | tee -a "$ART_DIR/job_summary.txt"
  BASE_URL="${BASE_URL:-http://127.0.0.1:8000/v1}"
else
  echo "[stage:server] Starting vLLM server (BF16, tensor-parallel=2)" | tee -a "$ART_DIR/job_summary.txt"
  python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_DIR" \
    --served-model-name qwen3.6-27b \
    --host 127.0.0.1 \
    --port 8000 \
    --tensor-parallel-size 2 \
    --dtype bfloat16 \
    --language-model-only \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.90 \
    > "$ART_DIR/vllm_server.log" 2>&1 &
  SERVER_PID=$!
  READY=0
  for _ in $(seq 1 300); do
    if curl -sf -o /dev/null http://127.0.0.1:8000/v1/models; then
      READY=1
      break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "FAILED: vLLM server exited early" | tee -a "$ART_DIR/job_summary.txt"
      tail -50 "$ART_DIR/vllm_server.log" | tee -a "$ART_DIR/job_summary.txt"
      exit 1
    fi
    sleep 2
  done
  if [ "$READY" != "1" ]; then
    echo "FAILED: vLLM server not ready in time" | tee -a "$ART_DIR/job_summary.txt"
    kill "$SERVER_PID" 2>/dev/null || true
    exit 1
  fi
  BASE_URL="http://127.0.0.1:8000/v1"
  echo "Server ready" | tee -a "$ART_DIR/job_summary.txt"
fi

echo "[stage:translate] Translating units (resumable)" | tee -a "$ART_DIR/job_summary.txt"
rm -f "$RUN_ROOT/failed.jsonl"
TRANSLATE_ARGS=(
  --units "$RUN_ROOT/units.jsonl"
  --out "$RUN_ROOT/candidates.jsonl"
  --failed "$RUN_ROOT/failed.jsonl"
  --base-url "$BASE_URL"
  --model qwen3.6-27b
  --model-revision "$QWEN36_REVISION"
  --batch-size "$BATCH_SIZE"
  --seed "$SEED"
  --max-retries 2
)
if [ "$FORCE_RESYNC" = "1" ]; then
  TRANSLATE_ARGS+=(--force-resync)
fi
if [ -f "$RUN_ROOT/validation_retries.jsonl" ]; then
  TRANSLATE_ARGS+=(--validation-retries "$RUN_ROOT/validation_retries.jsonl")
fi
python -m src.translation.translate "${TRANSLATE_ARGS[@]}" 2>&1 | tee -a "$ART_DIR/translate.log"

echo "[stage:validate] Validating candidates" | tee -a "$ART_DIR/job_summary.txt"
VALIDATE_ARGS=(
  --units "$RUN_ROOT/units.jsonl"
  --candidates "$RUN_ROOT/candidates.jsonl"
  --accepted "$RUN_ROOT/accepted.jsonl"
  --rejected "$RUN_ROOT/rejected.jsonl"
  --audit "$RUN_ROOT/audit.json"
  --nllb-model "$NLLB_MODEL"
  --seed "$SEED"
)
if [ "$SKIP_SERVER" != "1" ]; then
  VALIDATE_ARGS+=(--verifier-base-url "$BASE_URL" --verifier-model qwen3.6-27b)
fi
if [ -n "$REVIEWED" ]; then
  VALIDATE_ARGS+=(--reviewed "$REVIEWED")
fi
python -m src.translation.validate "${VALIDATE_ARGS[@]}" 2>&1 | tee -a "$ART_DIR/validate.log"

if [ "$REQUIRE_COMPLETE" = "1" ]; then
  python - "$RUN_ROOT/audit.json" "$RUN_ROOT/rejected.jsonl" "$UNIT_COUNT" <<'PY'
import json
import sys
from pathlib import Path

audit_path = Path(sys.argv[1])
rejected_path = Path(sys.argv[2])
expected = int(sys.argv[3])
audit = json.loads(audit_path.read_text(encoding="utf-8"))
rejected = sum(1 for line in rejected_path.open("r", encoding="utf-8") if line.strip())
accepted = int(audit.get("accepted_cache_record_count", -1))
candidates = int(audit.get("candidate_count", -1))
if accepted != expected or candidates != expected or rejected != 0:
    raise SystemExit(
        f"Translation completeness failed: expected={expected} "
        f"candidates={candidates} accepted={accepted} rejected={rejected}"
    )
print(f"Translation completeness passed: accepted={accepted} rejected={rejected}")
PY
fi

if [ "$SKIP_SERVER" != "1" ]; then
  echo "[stage:server] Stopping vLLM server" | tee -a "$ART_DIR/job_summary.txt"
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
fi

echo "OK: pipeline finished for $DATASET" | tee -a "$ART_DIR/job_summary.txt"
echo "Artifacts: $RUN_ROOT" | tee -a "$ART_DIR/job_summary.txt"
