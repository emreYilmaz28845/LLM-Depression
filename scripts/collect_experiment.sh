#!/usr/bin/env bash
# Compact-evidence collection: dry-run command generation only. Never transfers.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_ROOT"

usage() {
    cat <<'EOF'
usage: collect_experiment.sh --group <group-id> --output <local-dir> [--dry-run]
       collect_experiment.sh --attempt <attempt-id> --fold <n> --output <local-dir> [--dry-run]
Authorization is required for a real transfer (Task 9).
EOF
    exit 1
}

DRY_RUN=0
AUTHORIZED=0
GROUP=""
ATTEMPT=""
FOLD=""
OUTPUT=""

while [ $# -gt 0 ]; do
    case "$1" in
        --group) GROUP="$2"; shift 2 ;;
        --attempt) ATTEMPT="$2"; shift 2 ;;
        --fold) FOLD="$2"; shift 2 ;;
        --output) OUTPUT="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --authorized) AUTHORIZED=1; shift ;;
        *) echo "unknown argument: $1"; usage ;;
    esac
done

if [ -z "$OUTPUT" ]; then
    echo "--output is required" >&2
    exit 1
fi
if [ -z "$GROUP" ] && [ -z "$ATTEMPT" ]; then
    usage
fi

REMOTE_HOST="ozu647717@transfer1.bsc.es"
REMOTE_PROJECT="/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression"

if [ -n "$ATTEMPT" ]; then
    REMOTE_PATH="$REMOTE_PROJECT/output_model/<modality>/<dataset>/<run_name>/fold_$FOLD"
else
    REMOTE_PATH="$REMOTE_PROJECT/output_model/<group-pattern>"
fi

if [ "$DRY_RUN" = "1" ]; then
    echo "rsync -avzn --exclude='best_model/' --exclude='last_model/' \\"
    echo "  --include='*/' \\"
    echo "  '$REMOTE_HOST:$REMOTE_PATH/' \\"
    echo "  '$OUTPUT/'"
    echo "collection includes: run_config.yaml metadata.json status.json jobs.jsonl artifacts.json evaluations.json final_summary.* logs/*.json(l) best_model/standalone_eval/*"
    echo "collection excludes: best_model/ last_model/ (retrieve specific checkpoints only on request)"
    exit 0
fi

if [ "$AUTHORIZED" != "1" ]; then
    echo "refusing real collection without --authorized (Task 9 gates remote mutation)" >&2
    exit 2
fi

echo "authorized collection would run here; Task 9 gates the real transfer" >&2
exit 0
