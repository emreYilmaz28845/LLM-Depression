#!/usr/bin/env bash
# Submit one harmonized symmetric-merged stage with Optuna disabled.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
RUN_ID="${RUN_ID:?Set the shared merged RUN_ID}"
STAGE="${STAGE:-cv}"
DRY_RUN="${DRY_RUN:-1}"
PREFLIGHT_AUDIT="${PREFLIGHT_AUDIT:-$PROJECT_ROOT/outputs/harmonized_mn5_preflight/$RUN_ID/audit.json}"
MAX_CONCURRENT_TRAINS="${MAX_CONCURRENT_TRAINS:-7}"
MAX_CONCURRENT_POSTPROCESS="${MAX_CONCURRENT_POSTPROCESS:-4}"

case "$DRY_RUN" in 0|1) ;; *) echo "DRY_RUN must be 0 or 1" >&2; exit 2;; esac
case "$STAGE" in smoke|cv|final) ;; *) echo "STAGE must be smoke, cv, or final" >&2; exit 2;; esac
if [ $((MAX_CONCURRENT_TRAINS * 4 + MAX_CONCURRENT_POSTPROCESS)) -gt 32 ]; then
    echo "Merged concurrency can exceed 32 GPUs." >&2
    exit 2
fi
if [ "$DRY_RUN" = 0 ]; then
    python - "$PREFLIGHT_AUDIT" "$RUN_ID" <<'PY'
import json, sys
p = json.load(open(sys.argv[1], encoding="utf-8"))
if p.get("status") != "passed" or p.get("run_id") != sys.argv[2] or len(p.get("merged", [])) != 3:
    raise SystemExit(f"Incompatible merged preflight audit: {sys.argv[1]}")
if p.get("optuna_enabled") is not False:
    raise SystemExit("Preflight does not prove Optuna is disabled")
PY
fi

args=(
    python "$PROJECT_ROOT/scripts/submit_symmetric_merged.py"
    --stage "$STAGE"
    --run-id "$RUN_ID"
    --config "$PROJECT_ROOT/configs/experiments/merged/symmetric_merged_harmonized_audio_text.yaml"
    --config "$PROJECT_ROOT/configs/experiments/merged/symmetric_merged_harmonized_audio_only.yaml"
    --config "$PROJECT_ROOT/configs/experiments/merged/symmetric_merged_harmonized_text_only.yaml"
    --smoke-trials 0
    --max-concurrent-trains "$MAX_CONCURRENT_TRAINS"
    --max-concurrent-postprocess "$MAX_CONCURRENT_POSTPROCESS"
)
[ "$DRY_RUN" = 1 ] && args+=(--dry-run)
"${args[@]}"
