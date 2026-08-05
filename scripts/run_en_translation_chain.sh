#!/bin/bash
#SBATCH -J en-translation-chain
#SBATCH -A etur92
#SBATCH -q acc_ehpc
#SBATCH -t 00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -o /gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression/logs/slurm_en_chain-%j.out
#SBATCH -e /gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression/logs/slurm_en_chain-%j.err
#SBATCH --chdir=/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression

# Sequential English-translation CV chain: one fold at a time per config.
# Each invocation submits the fold at INDEX, then chains a continuation job
# with afterok on the fold's terminal jobs (eval when present, else train).
# CONFIG_LIST and RUN_NAMES are '|'-separated, aligned; INDEX walks
# config x fold in row-major order. A per-config CV summary is written when a
# config's folds complete.
#
# Required env: CONFIG_LIST, RUN_NAMES, FOLDS, INDEX.
# Optional: PROJECT_ROOT, ENV_ACTIVATE, SKIP_MANIFEST_BUILD, JOB_PREFIX.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
ENV_ACTIVATE="${ENV_ACTIVATE:-/gpfs/projects/etur92/ozu647717/venvs/qwen_mn5_rebuilt/bin/activate}"
SKIP_MANIFEST_BUILD="${SKIP_MANIFEST_BUILD:-1}"
JOB_PREFIX="${JOB_PREFIX:-en-seq-}"
CHAIN_SCRIPT="${CHAIN_SCRIPT:-$PROJECT_ROOT/scripts/run_en_translation_chain.sh}"
SUBMIT_SCRIPT="${SUBMIT_SCRIPT:-$PROJECT_ROOT/scripts/submit_train_and_eval.sh}"
SUMMARIZE_SCRIPT="${SUMMARIZE_SCRIPT:-$PROJECT_ROOT/src/summarize_runs.py}"

CONFIG_LIST="${CONFIG_LIST:?Set CONFIG_LIST (| separated)}"
RUN_NAMES="${RUN_NAMES:?Set RUN_NAMES (| separated)}"
FOLDS="${FOLDS:?Set FOLDS}"
INDEX="${INDEX:?Set INDEX}"

if [ -f "$ENV_ACTIVATE" ]; then
    # shellcheck disable=SC1090
    source "$ENV_ACTIVATE"
fi

IFS='|' read -r -a CONFIGS <<< "$CONFIG_LIST"
IFS='|' read -r -a RUNS <<< "$RUN_NAMES"
read -r -a FOLD_ARRAY <<< "$FOLDS"
FOLD_COUNT="${#FOLD_ARRAY[@]}"
TOTAL="$(( ${#CONFIGS[@]} * FOLD_COUNT ))"

if [ "${#CONFIGS[@]}" -ne "${#RUNS[@]}" ]; then
    echo "CONFIG_LIST and RUN_NAMES must have the same length." >&2
    exit 1
fi

if [ "$INDEX" -ge "$TOTAL" ]; then
    echo "Chain complete: $TOTAL folds done."
    exit 0
fi

CONFIG_INDEX=$(( INDEX / FOLD_COUNT ))
FOLD_INDEX=$(( INDEX % FOLD_COUNT ))
CONFIG="${CONFIGS[$CONFIG_INDEX]}"
RUN_NAME="${RUNS[$CONFIG_INDEX]}"
FOLD="${FOLD_ARRAY[$FOLD_INDEX]}"

echo "Chain stage $(( INDEX + 1 ))/$TOTAL | config=$CONFIG run=$RUN_NAME fold=$FOLD"
if [ ! -f "$CONFIG" ]; then
    echo "Config not found: $CONFIG" >&2
    exit 1
fi

OUTPUT="$(
    env \
        PROJECT_ROOT="$PROJECT_ROOT" \
        CONFIG="$CONFIG" \
        RUN_NAME="$RUN_NAME" \
        FOLD="$FOLD" \
        SKIP_MANIFEST_BUILD="$SKIP_MANIFEST_BUILD" \
        SBATCH_JOB_NAME="${JOB_PREFIX}${RUN_NAME}-f${FOLD}" \
        bash "$SUBMIT_SCRIPT"
)"
printf '%s\n' "$OUTPUT"

extract_job_id() {
    local pattern="$1"
    printf '%s\n' "$OUTPUT" | awk -v pattern="$pattern" '$0 ~ pattern {print $NF; exit}'
}
TRAIN_JOB_ID="$(extract_job_id "Submitted training job:")"
BEST_EVAL_JOB_ID="$(extract_job_id "Submitted best-checkpoint eval job:")"

TERMINAL_ID="$TRAIN_JOB_ID"
if [ -n "$BEST_EVAL_JOB_ID" ]; then
    TERMINAL_ID="$BEST_EVAL_JOB_ID"
fi
if [ -z "$TERMINAL_ID" ]; then
    echo "Could not parse submitted job ids." >&2
    exit 1
fi
echo "Fold terminal job: $TERMINAL_ID"

NEXT_INDEX=$(( INDEX + 1 ))
if [ $(( NEXT_INDEX % FOLD_COUNT )) -eq 0 ] && [ "$NEXT_INDEX" -le "$TOTAL" ]; then
    RUN_ROOT_REL="$(awk '
      /^output_dirs:/ {in_block=1; next}
      in_block && /^[^[:space:]]/ {in_block=0}
      in_block && /^[[:space:]]+run_root:/ {
        sub(/^[[:space:]]+run_root:[[:space:]]*/, "", $0)
        print $0
        exit
      }
    ' "$CONFIG" | tr -d '"' | tr -d "'")"
    RUN_ROOT="${RUN_ROOT_REL//\$\{PROJECT_ROOT\}/$PROJECT_ROOT}/$RUN_NAME"
    SUMMARY_LOG="$RUN_ROOT/cv_summary-$(date +%Y-%m-%d_%H:%M:%S).log"
    mkdir -p "$RUN_ROOT"
    {
        echo "Summarizing CV results for $RUN_NAME"
        python "$SUMMARIZE_SCRIPT" --run_root "$RUN_ROOT"
    } 2>&1 | tee "$SUMMARY_LOG" || true
fi

if [ "$NEXT_INDEX" -lt "$TOTAL" ]; then
    EXPORT_ARGS="ALL,PROJECT_ROOT=$PROJECT_ROOT,CONFIG_LIST=$CONFIG_LIST,RUN_NAMES=$RUN_NAMES,FOLDS=$FOLDS,INDEX=$NEXT_INDEX,SKIP_MANIFEST_BUILD=$SKIP_MANIFEST_BUILD,JOB_PREFIX=$JOB_PREFIX,CHAIN_SCRIPT=$CHAIN_SCRIPT"
    NEXT_RAW="$(sbatch --parsable --dependency="afterok:$TERMINAL_ID" --export="$EXPORT_ARGS" "$CHAIN_SCRIPT")"
    echo "Chained continuation job: ${NEXT_RAW%%;*}"
fi
