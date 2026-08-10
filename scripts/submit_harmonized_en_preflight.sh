#!/usr/bin/env bash
# Submit the CPU-only MN5 English manifest/equivalence preflight required
# before any harmonized English GPU job.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
RUN_ID="${RUN_ID:?Set the shared harmonized English RUN_ID}"
DRY_RUN="${DRY_RUN:-1}"
WORKER="${WORKER:-$PROJECT_ROOT/scripts/run_harmonized_en_preflight_slurm.sh}"
GITHUB_ISSUE="${GITHUB_ISSUE:?Set the harmonized English campaign GITHUB_ISSUE}"
GITHUB_PR="${GITHUB_PR:?Set the harmonized English implementation GITHUB_PR}"
SOURCE_COMMIT="${HARMONIZED_EN_SOURCE_COMMIT:-$(tr -d '\n' < "$PROJECT_ROOT/.provenance/git_commit.txt")}"
SOURCE_BRANCH="${HARMONIZED_EN_SOURCE_BRANCH:-$(tr -d '\n' < "$PROJECT_ROOT/.provenance/git_branch.txt")}"

case "$DRY_RUN" in 0|1) ;; *) echo "DRY_RUN must be 0 or 1" >&2; exit 2;; esac
case "$GITHUB_ISSUE" in ''|*[!0-9]*|0) echo "GITHUB_ISSUE must be a positive integer." >&2; exit 2;; esac
case "$GITHUB_PR" in ''|*[!0-9]*|0) echo "GITHUB_PR must be a positive integer." >&2; exit 2;; esac
[ -f "$WORKER" ] || { echo "Missing preflight worker: $WORKER" >&2; exit 3; }
command=(sbatch --parsable --job-name="harm-en-preflight" --export="ALL,PROJECT_ROOT=$PROJECT_ROOT,RUN_ID=$RUN_ID,GITHUB_ISSUE=$GITHUB_ISSUE,GITHUB_PR=$GITHUB_PR,HARMONIZED_EN_SOURCE_COMMIT=$SOURCE_COMMIT,HARMONIZED_EN_SOURCE_BRANCH=$SOURCE_BRANCH" "$WORKER")
if [ "$DRY_RUN" = 1 ]; then
    printf 'DRY_RUN '; printf '%q ' "${command[@]}"; printf '\n'
else
    raw="$("${command[@]}")"
    echo "Submitted harmonized English preflight job: ${raw%%;*}"
    echo "Wait for a passed audit before any GPU submission: $PROJECT_ROOT/outputs/harmonized_en_mn5_preflight/$RUN_ID/audit.json"
fi
