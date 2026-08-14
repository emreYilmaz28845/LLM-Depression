#!/usr/bin/env bash
# Submit the CPU-only model-free Gemma harmonized preflight (native or
# English) and record the Slurm job ID. Dry-run by default.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
RUN_ID="${RUN_ID:?Set a unique RUN_ID}"
DRY_RUN="${DRY_RUN:-1}"
ENGLISH="${ENGLISH:-0}"
PREFLIGHT_WORKER="${PREFLIGHT_WORKER:-$PROJECT_ROOT/scripts/run_gemma4_harmonized_preflight_slurm.sh}"
GEMMA_ENV="${GEMMA_ENV:-/gpfs/projects/etur92/ozu647717/venvs/gemma4_12b_tf5_14_1}"
MODEL_PATH="${MODEL_PATH:-/gpfs/projects/etur92/ozu647717/models/gemma-4-12B-it/707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7}"
REQUIRED_PATH_PREFIX="${REQUIRED_PATH_PREFIX:-/gpfs/projects/etur92/ozu647717}"

case "$DRY_RUN" in 0|1) ;; *) echo "DRY_RUN must be 0 or 1" >&2; exit 2;; esac
[ -f "$PREFLIGHT_WORKER" ] || { echo "Missing preflight worker: $PREFLIGHT_WORKER" >&2; exit 3; }

export_spec="ALL,PROJECT_ROOT=$PROJECT_ROOT,RUN_ID=$RUN_ID,ENGLISH=$ENGLISH,ENV_ACTIVATE=$GEMMA_ENV/bin/activate,MODEL_PATH=$MODEL_PATH,REQUIRED_PATH_PREFIX=$REQUIRED_PATH_PREFIX"
CMD=(sbatch --parsable --job-name="g4harm-pre" --export="$export_spec" "$PREFLIGHT_WORKER")
if [ "$DRY_RUN" = 1 ]; then
    printf 'DRY_RUN ' >&2; printf '%q ' "${CMD[@]}" >&2; printf '\n' >&2
    echo "dry_gemma4_preflight"
else
    "${CMD[@]}"
fi
