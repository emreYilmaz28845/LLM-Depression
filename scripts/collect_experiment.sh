#!/usr/bin/env bash
# Compact-evidence collection from MN5. Thin wrapper over the Python
# implementation in src/experiment_tracking/collect.py (single source of
# truth). Dry-run first; execute transfers through transfer1 without --delete,
# preserves best_model/standalone_eval evidence, and refuses incompatible
# local overwrites.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_ROOT"

MODE="--dry-run"
ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) MODE="--dry-run"; shift ;;
        --execute) MODE="--execute"; shift ;;
        *) ARGS+=("$1"); shift ;;
    esac
done

exec python -m src.experiment_tracking.collect "$MODE" "${ARGS[@]+"${ARGS[@]}"}"
