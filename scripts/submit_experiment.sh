#!/usr/bin/env bash
# Local, read-only experiment planning and authorization-gated deploy/submit/collect.
# Only the `plan` action and dry-run command generation are implemented; every
# mutating action requires --authorized and is refused otherwise.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ACTION="${1:-}"

usage() {
    cat <<'EOF'
usage: submit_experiment.sh plan \
         --group <group-definition.yaml> --config <config.yaml> \
         --seeds 7 1337 2024 --folds 0 --issue 86 --pr 91 [--workspace <id>]
       submit_experiment.sh deploy --plan <plan.json> [--authorized]
       submit_experiment.sh submit --plan <plan.json> [--authorized]
       submit_experiment.sh collect --group <group-id> [--authorized]
EOF
    exit 1
}

if [ "$ACTION" != "plan" ] && [ "$ACTION" != "deploy" ] && [ "$ACTION" != "submit" ] && [ "$ACTION" != "collect" ]; then
    usage
fi

shift || true
AUTHORIZED=0
GROUP_DEF=""
CONFIG=""
SEEDS=""
FOLDS=""
ISSUE=""
PR=""
WORKSPACE=""

while [ $# -gt 0 ]; do
    case "$1" in
        --group) GROUP_DEF="$2"; shift 2 ;;
        --config) CONFIG="$2"; shift 2 ;;
        --seeds) SEEDS="$2"; shift 2 ;;
        --folds) FOLDS="$2"; shift 2 ;;
        --issue) ISSUE="$2"; shift 2 ;;
        --pr) PR="$2"; shift 2 ;;
        --workspace) WORKSPACE="$2"; shift 2 ;;
        --plan) PLAN_FILE="$2"; shift 2 ;;
        --authorized) AUTHORIZED=1; shift ;;
        *) echo "unknown argument: $1"; usage ;;
    esac
done

if [ "$ACTION" = "plan" ]; then
    if [ -z "$GROUP_DEF" ] || [ -z "$CONFIG" ] || [ -z "$SEEDS" ] || [ -z "$FOLDS" ]; then
        echo "plan requires --group --config --seeds --folds" >&2
        exit 1
    fi
    if [ ! -f "$GROUP_DEF" ]; then
        echo "group definition not found: $GROUP_DEF" >&2
        exit 1
    fi
    if [ ! -f "$CONFIG" ]; then
        echo "config not found: $CONFIG" >&2
        exit 1
    fi
    cd "$PROJECT_ROOT"
    GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
    GIT_DIRTY="$(git status --porcelain | wc -l | tr -d ' ')"
    RUN_ROOT_REL="$(awk '
      /^output_dirs:/ {in_block=1; next}
      in_block && /^[^[:space:]]/ {in_block=0}
      in_block && /^[[:space:]]+run_root:/ {
        sub(/^[[:space:]]+run_root:[[:space:]]*/, "", $0)
        print $0
        exit
      }
    ' "$CONFIG" | tr -d '"' | tr -d "'")"
    echo "source:"
    echo "  git_commit: $GIT_COMMIT"
    echo "  git_dirty_entries: $GIT_DIRTY"
    echo "config:"
    echo "  path: $CONFIG"
    echo "  run_root: ${RUN_ROOT_REL:-<unparsed>}"
    echo "group:"
    echo "  definition: $GROUP_DEF"
    echo "  issue: ${ISSUE:-<none>}"
    echo "  pr: ${PR:-<none>}"
    echo "matrix:"
    for seed in $SEEDS; do
        for fold in $FOLDS; do
            echo "  seed=$seed fold=$fold logical_run=<group-run-name>-s$seed jobs=train+evaluation job_count=2"
        done
    done
    TOTAL_JOBS=$(( $(echo "$SEEDS" | wc -w) * $(echo "$FOLDS" | wc -w) * 2 ))
    echo "total_jobs: $TOTAL_JOBS"
    echo "resources_per_train_job: 1 node, 4 GPUs, 20 CPUs/task, 72h (run_train_slurm.sh)"
    echo "resources_per_eval_job: 1 node, 1 GPU, 20 CPUs/task, 24h (run_eval_slurm.sh)"
    echo "workspace: ${WORKSPACE:-<default shared workspace>}"
    echo "endpoint_split: transfer=ozu647717@transfer1.bsc.es scheduler=ozu647717@alogin2.bsc.es"
    echo "checkpoint_policy: best_model selected by inner_val_positive_f1; last_model never substituted"
    echo "rsync_policy: no --delete; dry-run first; review every destination change"
    echo "attempt_ids: minted at deploy time (<UTC>-<logical-run>-<git8>-<8hex>); collision is an error"
    exit 0
fi

if [ "$ACTION" = "deploy" ] || [ "$ACTION" = "submit" ] || [ "$ACTION" = "collect" ]; then
    if [ -z "${PLAN_FILE:-}" ] && [ "$ACTION" != "collect" ]; then
        echo "$ACTION requires --plan <plan.json>" >&2
        exit 1
    fi
    if [ "$AUTHORIZED" != "1" ]; then
        echo "refusing $ACTION without explicit --authorized (Tasks 0-8 are local-only; MN5 mutation requires user authorization)" >&2
        exit 2
    fi
    echo "authorized $ACTION would run here; Task 9 gates the real remote mutation" >&2
    exit 0
fi
