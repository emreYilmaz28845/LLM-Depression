#!/bin/bash
# Submit the Qwen3.8 standalone prompt-context matrix through the managed
# workflow: five native text-only cells, 21 training fits with one standalone
# likelihood evaluation each.
#
# Usage (from the pinned lane worktree):
#   DRY_RUN=1 bash scripts/submit_promptcontext_qwen38_matrix.sh     # contracts only
#   bash scripts/submit_promptcontext_qwen38_matrix.sh               # submit
#
# Env:
#   SLUG          managed lane slug (default: feat-qwen38-standalone-promptcontext)
#   CAMPAIGN      run-root campaign (default: promptcontext_v1_qwen38_likelihood)
#   RUN_PREFIX    run-name prefix (default: qwen38_pc)
#   RUN_SUFFIX    run-name suffix (default: 20260922)
#   ENV_ACTIVATE  venv activate script the workers source (default: the Qwen3.8 env)
#   ONLY          optional dataset filter (daic|d3tec|androids|cmdc|turkish)

set -euo pipefail

mkdir -p outputs/exp_submit

SLUG="${SLUG:-feat-qwen38-standalone-promptcontext}"
CAMPAIGN="${CAMPAIGN:-promptcontext_v1_qwen38_likelihood}"
RUN_PREFIX="${RUN_PREFIX:-qwen38_pc}"
RUN_SUFFIX="${RUN_SUFFIX:-20260922}"
ENV_ACTIVATE="${ENV_ACTIVATE:-/gpfs/projects/etur92/ozu647717/venvs/qwen38_fsdp_fastpath_20260921/bin/activate}"
ONLY="${ONLY:-}"
DRY_RUN="${DRY_RUN:-0}"

# dataset qualifier | config | folds | manifest policy
CELLS=(
    "daic|daic_text_only_harmonized_selmacrof1_likelihood_v1_promptcontext_v1_qwen38_27b.yaml|0|build"
    "d3tec|d3tec_text_only_harmonized_selmacrof1_likelihood_v1_promptcontext_v1_qwen38_27b.yaml|0 1 2 3 4|build"
    "androids_interview|androids_text_only_harmonized_selmacrof1_likelihood_v1_promptcontext_v1_qwen38_27b.yaml|0 1 2 3 4|build"
    "cmdc|cmdc_text_only_harmonized_selmacrof1_likelihood_v1_promptcontext_v1_qwen38_27b.yaml|0 1 2 3 4|build"
    "turkish|turkish_pooled_t17_text_only_harmonized_selmacrof1_likelihood_v1_promptcontext_v1_qwen38_27b.yaml|0 1 2 3 4|prebuilt"
)

submitted=0
for cell in "${CELLS[@]}"; do
    IFS='|' read -r dataset config folds policy <<< "$cell"
    cell_key="$dataset"
    case "$dataset" in
        androids_interview) cell_key="androids" ;;
    esac
    if [ -n "$ONLY" ] && [ "$ONLY" != "$cell_key" ] && [ "$ONLY" != "$dataset" ]; then
        continue
    fi
    for fold in $folds; do
        run_name="${RUN_PREFIX}_${cell_key}_f${fold}_${RUN_SUFFIX}"
        args=(
            "$SLUG"
            --config "configs/main/$config"
            --fold "$fold"
            --run-name "$run_name"
            --campaign "$CAMPAIGN"
            --modality text_only
            --dataset "$dataset"
            --env-activate "$ENV_ACTIVATE"
            --manifest-policy "$policy"
        )
        if [ "$DRY_RUN" = "1" ]; then
            args+=(--dry-run)
        else
            args+=(--execute)
        fi
        echo "=== $cell_key fold $fold policy=$policy run=$run_name ==="
        python tools/exp.py submit "${args[@]}" > "outputs/exp_submit/_submit_${cell_key}_f${fold}.txt" 2>&1 || {
            echo "ERROR: submission failed for $cell_key fold $fold" >&2
            tail -5 "outputs/exp_submit/_submit_${cell_key}_f${fold}.txt" >&2
            exit 1
        }
        tail -3 "outputs/exp_submit/_submit_${cell_key}_f${fold}.txt"
        submitted=$((submitted + 1))
    done
done
echo "matrix submissions: $submitted (expected 21)"
