#!/usr/bin/env bash
# Submit one CPU-only Qwen3 pooled-defaults preflight stage.
#
#   STAGE=pooled    (default env: qwen_mn5_rebuilt)  build + audit pooled inputs
#   STAGE=qwen38    (default env: qwen38_fsdp_fastpath_20260921) prompt rendering
#   STAGE=qwen3omni (default env: qwen3omni) processor/audio input audit
#
# The worker runs on a CPU compute node (1 node, 1 task, 20 CPUs, 4 h) and never
# loads model weights. DRY_RUN=1 prints the full contract (stage, backend
# environment, resources, paths, audited configs, job count) without calling
# sbatch.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
STAGE="${STAGE:?Set STAGE to pooled, qwen38 or qwen3omni}"
RUNTIME_ROOT="${RUNTIME_ROOT:?Set RUNTIME_ROOT to the task runtime root}"
SOURCE_INPUT_ROOT="${SOURCE_INPUT_ROOT:-}"
DRY_RUN="${DRY_RUN:-1}"
WORKER="${WORKER:-$PROJECT_ROOT/scripts/run_qwen3_pooled_preflight_slurm.sh}"
QWEN38_MODEL_PATH="${QWEN38_MODEL_PATH:-/gpfs/projects/etur92/ozu647717/models/Qwen3.8-27B}"
QWEN3OMNI_MODEL_PATH="${QWEN3OMNI_MODEL_PATH:-/gpfs/projects/etur92/ozu647717/models/Qwen3-Omni-30B-A3B-Instruct}"
SOURCE_COMMIT="${HARMONIZED_SOURCE_COMMIT:-$(cat "$PROJECT_ROOT/.provenance/git_commit.txt" 2>/dev/null | tr -d '\n' || true)}"
SOURCE_BRANCH="${HARMONIZED_SOURCE_BRANCH:-$(cat "$PROJECT_ROOT/.provenance/git_branch.txt" 2>/dev/null | tr -d '\n' || true)}"

case "$STAGE" in
    pooled|qwen38|qwen3omni) ;;
    *) echo "STAGE must be pooled, qwen38 or qwen3omni" >&2; exit 2 ;;
esac
case "$DRY_RUN" in 0|1) ;; *) echo "DRY_RUN must be 0 or 1" >&2; exit 2;; esac
[ -f "$WORKER" ] || { echo "Missing preflight worker: $WORKER" >&2; exit 3; }
if [ "$STAGE" = "pooled" ] && [ -z "$SOURCE_INPUT_ROOT" ]; then
    echo "SOURCE_INPUT_ROOT is required for STAGE=pooled" >&2
    exit 3
fi

export_spec="ALL,PROJECT_ROOT=$PROJECT_ROOT,STAGE=$STAGE,RUNTIME_ROOT=$RUNTIME_ROOT"
if [ -n "$SOURCE_INPUT_ROOT" ]; then
    export_spec="$export_spec,SOURCE_INPUT_ROOT=$SOURCE_INPUT_ROOT"
fi

command=(sbatch --parsable --job-name="q3pool-${STAGE:0:4}" --export="$export_spec" "$WORKER")

if [ "$DRY_RUN" = 1 ]; then
    echo "Qwen3 pooled preflight dry-run contract:"
    echo "  stage: $STAGE"
    echo "  worker: $WORKER"
    echo "  project_root: $PROJECT_ROOT"
    echo "  runtime_root: $RUNTIME_ROOT"
    echo "  source_input_root: ${SOURCE_INPUT_ROOT:-<none>}"
    echo "  source: branch=$SOURCE_BRANCH commit=$SOURCE_COMMIT"
    echo "  resources: 1 CPU node, 1 task, 20 CPUs, 4:00:00, no GPUs, no model weights"
    case "$STAGE" in
        pooled) echo "  env_activate: ${ENV_ACTIVATE:-/gpfs/projects/etur92/ozu647717/venvs/qwen_mn5_rebuilt/bin/activate}" ;;
        qwen38) echo "  env_activate: ${ENV_ACTIVATE:-/gpfs/projects/etur92/ozu647717/venvs/qwen38_fsdp_fastpath_20260921/bin/activate}" ;;
        qwen3omni) echo "  env_activate: ${ENV_ACTIVATE:-/gpfs/projects/etur92/ozu647717/venvs/qwen3omni/bin/activate}" ;;
    esac
    case "$STAGE" in
        pooled)
            echo "  audits: builder + tools/audit_qwen3_pooled_inputs.py (recorded hashes) + tools/qwen3_pooled_defaults.py"
            ;;
        qwen38)
            echo "  audits: tools/audit_qwen3_prompt_rendering.py against $QWEN38_MODEL_PATH (tokenizer only)"
            ;;
        qwen3omni)
            echo "  audits: tools/audit_qwen3omni_processor.py against $QWEN3OMNI_MODEL_PATH (processor only)"
            ;;
    esac
    echo "  jobs: 1"
    printf '  command: '; printf '%q ' "${command[@]}"; printf '\n'
    exit 0
fi

raw="$("${command[@]}")"
echo "Submitted Qwen3 pooled preflight ($STAGE): ${raw%%;*}"
echo "Audit outputs land under $RUNTIME_ROOT/preflight/"
