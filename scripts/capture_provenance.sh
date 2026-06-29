#!/usr/bin/env bash
# Capture the exact local code state into .provenance/ so it can be rsynced to a
# cluster that has no .git (the rsync excludes .git/) and no internet.
#
# Run this LOCALLY right before syncing. Because the dev workflow rsyncs the
# working tree (often with uncommitted edits), the commit hash alone is not
# enough -- we also snapshot `git diff HEAD` so the transferred state can be
# reconstructed exactly: commit + uncommitted.patch == what actually ran.
#
# Everything here is offline: git reads the local repo, no network is touched.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROV_DIR="$PROJECT_ROOT/.provenance"
mkdir -p "$PROV_DIR"

if ! git -C "$PROJECT_ROOT" rev-parse --git-dir >/dev/null 2>&1; then
    echo "Not a git repo: $PROJECT_ROOT" >&2
    exit 1
fi

git -C "$PROJECT_ROOT" rev-parse HEAD                 > "$PROV_DIR/git_commit.txt"
git -C "$PROJECT_ROOT" rev-parse --abbrev-ref HEAD    > "$PROV_DIR/git_branch.txt"
git -C "$PROJECT_ROOT" status --porcelain             > "$PROV_DIR/git_dirty.txt"
git -C "$PROJECT_ROOT" diff HEAD                       > "$PROV_DIR/uncommitted.patch"
date -Iseconds                                        > "$PROV_DIR/captured_at.txt"

if [ -s "$PROV_DIR/git_dirty.txt" ]; then
    DIRTY="dirty (see uncommitted.patch)"
else
    DIRTY="clean"
fi

echo "Provenance captured -> $PROV_DIR"
echo "  commit: $(cat "$PROV_DIR/git_commit.txt")"
echo "  branch: $(cat "$PROV_DIR/git_branch.txt")"
echo "  tree:   $DIRTY"
