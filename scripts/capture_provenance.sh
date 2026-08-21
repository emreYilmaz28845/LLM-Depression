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

# Honor an explicit target so lane worktrees can capture their own provenance
# (tools/exp.py deploy sets PROJECT_ROOT to the pinned worktree).
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
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

# Deterministic source manifest (path, sha256, size) for every tracked file.
# The cluster has no .git; the campaign's source_manifest.json is rebuilt from
# this file after deployment instead of running git there.
python3 - "$PROJECT_ROOT" "$PROV_DIR/source_manifest.json" <<'PY'
import hashlib
import json
import subprocess
import sys
from pathlib import Path

root = Path(sys.argv[1])
output = Path(sys.argv[2])
result = subprocess.run(
    ["git", "-C", str(root), "ls-files"],
    capture_output=True,
    text=True,
    check=True,
)
records = []
for relative in sorted(line for line in result.stdout.splitlines() if line.strip()):
    path = root / relative
    if not path.is_file():
        continue
    records.append(
        {
            "path": relative,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
        }
    )
payload = {
    "schema_version": "audiollm.source_manifest.v1",
    "file_count": len(records),
    "files": records,
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

if [ -s "$PROV_DIR/git_dirty.txt" ]; then
    DIRTY="dirty (see uncommitted.patch)"
else
    DIRTY="clean"
fi

echo "Provenance captured -> $PROV_DIR"
echo "  commit: $(cat "$PROV_DIR/git_commit.txt")"
echo "  branch: $(cat "$PROV_DIR/git_branch.txt")"
echo "  tree:   $DIRTY"
