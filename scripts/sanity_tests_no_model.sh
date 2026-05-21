#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

run_python() {
  if [[ -n "${CONDA_ENV:-}" ]]; then
    conda run -n "$CONDA_ENV" "$PYTHON_BIN" "$@"
  else
    "$PYTHON_BIN" "$@"
  fi
}

echo "[1/3] Building manifests and split metadata"
"$PROJECT_ROOT/scripts/validate_manifests.sh"

echo "[2/3] Verifying expected audit/split files"
run_python - <<'PY'
from pathlib import Path
root = Path("/home/emre/Projects/AudioLLM/LLM-Depression")
required = [
    root / "outputs/manifests/daic_manifest.jsonl",
    root / "outputs/manifests/cmdc_manifest.jsonl",
    root / "outputs/manifests/eatd_manifest.jsonl",
    root / "outputs/splits/daic_join_audit.csv",
    root / "outputs/splits/cmdc_fold_report.json",
    root / "outputs/splits/eatd_folds.json",
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit(f"Missing expected outputs: {missing}")
print("All expected manifest/split outputs are present.")
PY

echo "[3/3] Printing split source summary"
run_python - <<'PY'
import json
from pathlib import Path
root = Path("/home/emre/Projects/AudioLLM/LLM-Depression/outputs/splits")
for name in ["daic", "cmdc", "eatd"]:
    path = root / f"{name}_manifest_metadata.json"
    data = json.loads(path.read_text())
    print(name, json.dumps({k: data[k] for k in data if k in {"dataset", "split_source", "split_source_notes", "manifest_hash"}}, ensure_ascii=False))
PY
