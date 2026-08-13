#!/usr/bin/env bash
# Submit the model-free DAIC official-development preflight job.
#
# The preflight rebuilds the canonical DAIC manifest on GPFS, proves the
# locked 86/21/35 split contract and expected row counts, validates MN5
# dataset paths, and writes the audit consumed by the production launcher.
# Dry-run by default; zero mutation in dry-run mode.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
RUN_ID="${RUN_ID:?Set the campaign RUN_ID}"
DRY_RUN="${DRY_RUN:-1}"
WORKER="${WORKER:-$PROJECT_ROOT/scripts/run_daic_officialdev_preflight_slurm.sh}"
BUILD_MANIFEST="${BUILD_MANIFEST:-1}"

case "$DRY_RUN" in 0|1) ;; *) echo "DRY_RUN must be 0 or 1" >&2; exit 2;; esac
[ -f "$WORKER" ] || { echo "Missing worker: $WORKER" >&2; exit 3; }

SOURCE_COMMIT="$(tr -d '\n' < "$PROJECT_ROOT/.provenance/git_commit.txt" 2>/dev/null || true)"
SOURCE_BRANCH="$(tr -d '\n' < "$PROJECT_ROOT/.provenance/git_branch.txt" 2>/dev/null || true)"

CMD=(sbatch --parsable --job-name="daic-odv-preflight"
    --export=ALL,PROJECT_ROOT="$PROJECT_ROOT",RUN_ID="$RUN_ID",BUILD_MANIFEST="$BUILD_MANIFEST"
    "$WORKER")

if [ "$DRY_RUN" = 1 ]; then
    printf 'DRY_RUN '; printf '%q ' "${CMD[@]}"; printf '\n'
    exit 0
fi

echo "Preflight source commit: $SOURCE_COMMIT"
echo "Preflight source branch: $SOURCE_BRANCH"
JOB_ID="$("${CMD[@]}" | tail -n 1 | tr -d ' ')"
echo "Submitted preflight job: $JOB_ID"
mkdir -p "$PROJECT_ROOT/outputs/daic_officialdev_mn5_preflight/$RUN_ID"
echo "$JOB_ID" > "$PROJECT_ROOT/outputs/daic_officialdev_mn5_preflight/$RUN_ID/preflight_job_id.txt"
