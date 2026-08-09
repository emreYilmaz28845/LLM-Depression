#!/usr/bin/env bash
# Submit the CPU-only MN5 manifest/protocol rebuild required before harmonized jobs.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
RUN_ID="${RUN_ID:?Set the shared harmonized RUN_ID}"
DRY_RUN="${DRY_RUN:-1}"
WORKER="${WORKER:-$PROJECT_ROOT/scripts/run_harmonized_manifest_preflight_slurm.sh}"
SOURCE_COMMIT="${HARMONIZED_SOURCE_COMMIT:-$(tr -d '\n' < "$PROJECT_ROOT/.provenance/git_commit.txt")}"
SOURCE_BRANCH="${HARMONIZED_SOURCE_BRANCH:-$(tr -d '\n' < "$PROJECT_ROOT/.provenance/git_branch.txt")}"

case "$DRY_RUN" in 0|1) ;; *) echo "DRY_RUN must be 0 or 1" >&2; exit 2;; esac
[ -f "$WORKER" ] || { echo "Missing preflight worker: $WORKER" >&2; exit 3; }
command=(sbatch --parsable --job-name="harm-preflight" --export="ALL,PROJECT_ROOT=$PROJECT_ROOT,RUN_ID=$RUN_ID,HARMONIZED_SOURCE_COMMIT=$SOURCE_COMMIT,HARMONIZED_SOURCE_BRANCH=$SOURCE_BRANCH" "$WORKER")
if [ "$DRY_RUN" = 1 ]; then
    printf 'DRY_RUN '; printf '%q ' "${command[@]}"; printf '\n'
else
    raw="$("${command[@]}")"
    echo "Submitted harmonized preflight job: ${raw%%;*}"
    echo "Wait for a passed audit before any GPU submission: $PROJECT_ROOT/outputs/harmonized_mn5_preflight/$RUN_ID/audit.json"
fi
