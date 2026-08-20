#!/usr/bin/env bash
#SBATCH -A etur92
#SBATCH -q acc_ehpc
#SBATCH -t 01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1

# Consolidation-only recovery from an immutable, completed Turkish inference run.
# This job never reads the source transcript and never runs subject inference.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
DEPLOY_ROOT="${DEPLOY_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/deployments/qwen38_inference}"
MODEL_DIR="${MODEL_DIR:-/gpfs/projects/etur92/ozu647717/models/Qwen3.8-27B}"
VENV_DIR="${VENV_DIR:-/gpfs/projects/etur92/ozu647717/venvs/qwen38_inference}"
DEPLOYMENT_ID="${DEPLOYMENT_ID:?Set DEPLOYMENT_ID}"
SOURCE_COMMIT="${SOURCE_COMMIT:?Set the clean consolidation implementation commit}"
SELECTION_FILE="${SELECTION_FILE:?Set the source-matched serving_selection_v2.json}"
PARENT_RUN_ROOT="${PARENT_RUN_ROOT:?Set the immutable parent run root}"
PARENT_RUN_ID="${PARENT_RUN_ID:?Set the immutable parent run ID}"
PARENT_SOURCE_COMMIT="${PARENT_SOURCE_COMMIT:?Set the parent inference source commit}"
PARENT_SELECTION_FILE="${PARENT_SELECTION_FILE:?Set the parent source-matched selection v2}"
DERIVED_RUN_ID="${DERIVED_RUN_ID:?Set the new derived consolidation run ID}"
ANALYSIS_ATTEMPT="${ANALYSIS_ATTEMPT:?Set the derived attempt number}"

if [[ ! "$DERIVED_RUN_ID" =~ ^q38tc_[0-9a-f]{12}_from_[0-9a-f]{12}_attempt[0-9]+$ ]]; then
  echo "FAILED: invalid DERIVED_RUN_ID=$DERIVED_RUN_ID" >&2
  exit 1
fi
EXPECTED_ID="q38tc_${SOURCE_COMMIT:0:12}_from_${PARENT_SOURCE_COMMIT:0:12}_attempt${ANALYSIS_ATTEMPT}"
if [ "$DERIVED_RUN_ID" != "$EXPECTED_ID" ]; then
  echo "FAILED: derived run identity mismatch: $DERIVED_RUN_ID != $EXPECTED_ID" >&2
  exit 1
fi

RUN_ROOT="$PROJECT_ROOT/outputs/turkish_question_recovery_derived/$DERIVED_RUN_ID"
RESTRICTED="$RUN_ROOT/restricted"
CONSOLIDATION_DIR="$RESTRICTED/consolidation_batches"
LOG_DIR="$DEPLOY_ROOT/$DEPLOYMENT_ID/logs"
SERVER_LOG="$LOG_DIR/${DERIVED_RUN_ID}_${SLURM_JOB_ID:-local}_vllm.log"
if [ -e "$RUN_ROOT" ]; then
  echo "FAILED: refusing to reuse derived run root $RUN_ROOT" >&2
  exit 1
fi

module purge
module load bsc/1.0
module load miniforge/24.3.0-0
source "$VENV_DIR/bin/activate"
cd "$PROJECT_ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1 WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false VLLM_TORCH_COMPILE_OVERRIDE=0
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

REMOTE_PROVENANCE_SHA="$(tr -d '[:space:]' < .provenance/git_commit.txt)"
if [ "$SOURCE_COMMIT" != "$REMOTE_PROVENANCE_SHA" ]; then
  echo "FAILED: consolidation source mismatch" >&2
  exit 1
fi
if [ -s .provenance/uncommitted.patch ]; then
  echo "FAILED: deployed provenance contains tracked source changes" >&2
  exit 1
fi
python - .provenance/source_manifest.json <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path.cwd()
manifest = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
files = manifest.get("files")
if manifest.get("schema_version") != "audiollm.source_manifest.v1" or not isinstance(files, list):
    raise SystemExit("invalid deployed source manifest")
records = {record["path"]: record for record in files}
required = (
    "scripts/qwen38_turkish_question_recovery.py",
    "scripts/run_qwen38_turkish_consolidation_slurm.sh",
    "scripts/submit_qwen38_turkish_consolidation.sh",
    "src/qwen38/__init__.py",
    "src/qwen38/contracts.py",
    "src/qwen38/turkish_questions.py",
)
errors = []
for relative in required:
    record = records.get(relative)
    if record is None:
        errors.append(f"unmanifested:{relative}")
        continue
    path = root / relative
    if not path.is_file():
        errors.append(f"missing:{relative}")
        continue
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != record.get("sha256") or path.stat().st_size != record.get("size_bytes"):
        errors.append(f"mismatch:{relative}")
if len(files) != manifest.get("file_count"):
    errors.append("file_count")
if errors:
    raise SystemExit("deployed source manifest verification failed: " + ", ".join(errors[:10]))
print(f"verified {len(required)} execution-closure files against deployed manifest")
PY

mapfile -t VERIFIED < <(python - "$PARENT_RUN_ROOT" "$PARENT_RUN_ID" "$PARENT_SOURCE_COMMIT" "$PARENT_SELECTION_FILE" "$SELECTION_FILE" "$SOURCE_COMMIT" <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve()
parent_id, parent_commit = sys.argv[2:4]
parent_selection_path, selection_path = map(pathlib.Path, sys.argv[4:6])
source_commit = sys.argv[6]
manifest_path = root / "run_manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if manifest.get("turkish_run_id") != parent_id:
    raise SystemExit("parent run ID mismatch")
if manifest.get("source_commit") != parent_commit:
    raise SystemExit("parent source commit mismatch")
inferences = sorted((root / "restricted" / "subject_inferences").glob("S*.json"))
if len(inferences) != 135:
    raise SystemExit(f"parent must have 135 completed inference files, found {len(inferences)}")
for path in inferences:
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("status") != "completed" or record.get("source_commit") != parent_commit:
        raise SystemExit(f"invalid immutable parent inference: {path.name}")
parent_selection = json.loads(parent_selection_path.read_text(encoding="utf-8"))
selection = json.loads(selection_path.read_text(encoding="utf-8"))
if parent_selection.get("selection_version") != 2 or parent_selection.get("source_commit") != parent_commit:
    raise SystemExit("parent selection is not source matched")
if selection.get("selection_version") != 2 or selection.get("source_commit") != source_commit:
    raise SystemExit("derived selection is not source matched")
if selection.get("selected_tp") != 2:
    raise SystemExit("derived consolidation requires selected TP=2")
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
entries = [{"name": path.name, "sha256": sha(path)} for path in inferences]
aggregate = hashlib.sha256(json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
print(sha(manifest_path))
print(aggregate)
print(sha(parent_selection_path))
print(sha(selection_path))
PY
)
PARENT_MANIFEST_SHA256="${VERIFIED[0]}"
PARENT_INFERENCES_SHA256="${VERIFIED[1]}"
PARENT_SELECTION_SHA256="${VERIFIED[2]}"
SELECTION_SHA256="${VERIFIED[3]}"

mkdir -p "$RESTRICTED" "$CONSOLIDATION_DIR" "$LOG_DIR"
chmod 700 "$RUN_ROOT" "$RESTRICTED" "$CONSOLIDATION_DIR"
cp "$PARENT_RUN_ROOT/run_manifest.json" "$RUN_ROOT/run_manifest.json"
ln -s "$PARENT_RUN_ROOT/restricted/prepared_sequences.jsonl" "$RESTRICTED/prepared_sequences.jsonl"
ln -s "$PARENT_RUN_ROOT/restricted/subject_inferences" "$RESTRICTED/subject_inferences"

python - "$RUN_ROOT/derived_consolidation_provenance.json" <<PY
import json, os
payload = {
    "derived_run_id": "$DERIVED_RUN_ID",
    "analysis_attempt": int("$ANALYSIS_ATTEMPT"),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
    "deployment_id": "$DEPLOYMENT_ID",
    "consolidation_source_commit": "$SOURCE_COMMIT",
    "selection_file": "$SELECTION_FILE",
    "selection_file_sha256": "$SELECTION_SHA256",
    "selected_tp": 2,
    "parent_run_id": "$PARENT_RUN_ID",
    "parent_run_root": "$PARENT_RUN_ROOT",
    "parent_source_commit": "$PARENT_SOURCE_COMMIT",
    "parent_run_manifest_sha256": "$PARENT_MANIFEST_SHA256",
    "parent_inferences_manifest_sha256": "$PARENT_INFERENCES_SHA256",
    "parent_selection_file_sha256": "$PARENT_SELECTION_SHA256",
    "subject_inference_reused": True,
    "subject_inference_requests": 0,
}
path = "$RUN_ROOT/derived_consolidation_provenance.json"
with open(path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
chmod 600 "$RUN_ROOT/run_manifest.json" "$RUN_ROOT/derived_consolidation_provenance.json"

SERVER_PID=""
cleanup() {
  if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM
python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_DIR" --served-model-name qwen3.8-27b \
  --host 127.0.0.1 --port 8000 --tensor-parallel-size 2 \
  --dtype bfloat16 --language-model-only --max-model-len 8192 \
  --gpu-memory-utilization 0.90 --reasoning-parser qwen3 \
  --generation-config vllm --no-enable-log-requests > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
for _ in $(seq 1 300); do
  curl -sf -o /dev/null http://127.0.0.1:8000/v1/models && break
  kill -0 "$SERVER_PID" 2>/dev/null || { tail -80 "$SERVER_LOG" >&2; exit 1; }
  sleep 2
done
curl -sf -o /dev/null http://127.0.0.1:8000/v1/models || { echo "FAILED: vLLM not ready" >&2; exit 1; }

python scripts/qwen38_turkish_question_recovery.py consolidate \
  --deployment-id "$DEPLOYMENT_ID" --turkish-run-id "$PARENT_RUN_ID" \
  --run-dir "$RUN_ROOT" --source-commit "$PARENT_SOURCE_COMMIT" \
  --inferences-dir "$RESTRICTED/subject_inferences" \
  --consolidation-dir "$CONSOLIDATION_DIR" --base-url http://127.0.0.1:8000/v1 \
  --model qwen3.8-27b --seed 42 --max-tokens 2048 --tokenizer-path "$MODEL_DIR"

python scripts/qwen38_turkish_question_recovery.py render \
  --deployment-id "$DEPLOYMENT_ID" --turkish-run-id "$PARENT_RUN_ID" \
  --run-dir "$RUN_ROOT" --source-commit "$PARENT_SOURCE_COMMIT" \
  --inferences-dir "$RESTRICTED/subject_inferences" \
  --consolidation-dir "$CONSOLIDATION_DIR" \
  --final-merge "$CONSOLIDATION_DIR/final_merge.json"

cleanup
trap - EXIT INT TERM

python - "$RUN_ROOT" "$PARENT_RUN_ROOT" "$PARENT_MANIFEST_SHA256" "$PARENT_INFERENCES_SHA256" <<'PY'
import hashlib, json, pathlib, sys
run, parent = map(pathlib.Path, sys.argv[1:3])
expected_manifest, expected_inferences = sys.argv[3:5]
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
if sha(parent / "run_manifest.json") != expected_manifest:
    raise SystemExit("parent manifest changed during derived job")
paths = sorted((parent / "restricted" / "subject_inferences").glob("S*.json"))
entries = [{"name": path.name, "sha256": sha(path)} for path in paths]
aggregate = hashlib.sha256(json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
if aggregate != expected_inferences:
    raise SystemExit("parent inference evidence changed during derived job")
final = json.loads((run / "restricted" / "consolidation_batches" / "final_merge.json").read_text(encoding="utf-8"))
for level in final.get("hierarchy", []):
    for group in level.get("groups", []):
        if group["input_tokens_with_correction"] + group["max_output_tokens"] + group["safety_tokens"] > group["context_tokens"]:
            raise SystemExit("recorded hierarchy request exceeds context budget")
compact = {}
for name in ("turkish_inferred_questions.csv", "turkish_inferred_questions.json", "turkish_inferred_questions.md"):
    compact[name] = sha(run / name)
summary = {
    "derived_run_id": json.loads((run / "derived_consolidation_provenance.json").read_text())["derived_run_id"],
    "parent_run_id": json.loads((run / "derived_consolidation_provenance.json").read_text())["parent_run_id"],
    "parent_manifest_sha256": expected_manifest,
    "parent_inferences_manifest_sha256": expected_inferences,
    "families": len(final["families"]),
    "clusters": final["cluster_count"],
    "candidates": final["candidate_count"],
    "hierarchy_levels": len(final.get("hierarchy", [])),
    "compact_sha256": compact,
}
with (run / "derived_consolidation_summary.json").open("w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
chmod 644 "$RUN_ROOT"/turkish_inferred_questions.{csv,json,md} "$RUN_ROOT/derived_consolidation_summary.json"
echo "OK: derived consolidation completed at $RUN_ROOT"
