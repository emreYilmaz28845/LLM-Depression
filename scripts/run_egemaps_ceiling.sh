#!/usr/bin/env bash
# Reproducible local DAIC eGeMAPSv02 acoustic ceiling (primary fixed-K=4 protocol).
#
# Usage from any directory:
#   PYTHON_BIN=/home/emre/miniconda3/envs/llmdep4090/bin/python \
#     bash scripts/run_egemaps_ceiling.sh
#
# Extra CLI options are forwarded, for example:
#   PYTHON_BIN=/path/to/python bash scripts/run_egemaps_ceiling.sh --jobs 4

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/baselines/daic_egemaps_v02_fixedk4}"

cd "$PROJECT_ROOT"
exec "$PYTHON_BIN" -m src.baselines.egemaps_ceiling \
  --manifest "$PROJECT_ROOT/outputs/manifests/daic_manifest.jsonl" \
  --partitions "$PROJECT_ROOT/outputs/splits/daic_subject_partitions.json" \
  --output-dir "$OUTPUT_DIR" \
  --chunks-per-subject 4 \
  --provision-opensmile \
  "$@"
