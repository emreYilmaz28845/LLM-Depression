#!/usr/bin/env bash
# Capture provenance, then rsync the project to the cluster.
#
# Wraps the manual rsync so the .provenance/ snapshot is always refreshed and
# force-included even though it is gitignored (the --include rule is evaluated
# before the .gitignore filter, so it wins). .git/ is still excluded.
#
# Override the destination with REMOTE=... if needed.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE="${REMOTE:-ozu647717@transfer1.bsc.es:/gpfs/projects/etur92/ozu647717/AudioLLM}"

bash "$PROJECT_ROOT/scripts/capture_provenance.sh"

rsync -avhP \
    --include='.provenance/***' \
    --filter=":- .gitignore" \
    --exclude=".git/" \
    "$PROJECT_ROOT" \
    "$REMOTE"
