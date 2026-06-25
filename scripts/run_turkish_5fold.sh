#!/bin/bash
#SBATCH -J turkish-3mod-cv
#SBATCH -A etur92
#SBATCH -q acc_ehpc
#SBATCH -t 00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#SBATCH --chdir=/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression

# Submit three Turkish modality experiments in parallel:
#
#   audio-only: fold 0 -> 1 -> 2 -> 3 -> 4
#   text-only:  fold 0 -> 1 -> 2 -> 3 -> 4
#   audio+text: fold 0 -> 1 -> 2 -> 3 -> 4
#
# The three chains are independent and may run concurrently. Folds inside each
# chain use afterok dependencies and therefore run sequentially.

set -euo pipefail

module purge
module load bsc/1.0
module load miniforge/24.3.0-0

PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
CONFIGS="${CONFIGS:-$PROJECT_ROOT/configs/turkish_audio_only.yaml $PROJECT_ROOT/configs/turkish_text_only.yaml $PROJECT_ROOT/configs/turkish_audio_text.yaml}"
FOLDS="${FOLDS:-0 1 2 3 4}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
MODEL_PATH="${MODEL_PATH:-}"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"
SUBMIT_BEST_EVAL="${SUBMIT_BEST_EVAL:-0}"
SUBMIT_LAST_EVAL="${SUBMIT_LAST_EVAL:-0}"
ENV_ACTIVATE="${ENV_ACTIVATE:-/gpfs/projects/etur92/ozu647717/venvs/qwen_mn5_rebuilt/bin/activate}"
DATASET_BASE_ROOT="${DATASET_BASE_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/Datasets}"
TURKISH_DATASET_ROOT="${TURKISH_DATASET_ROOT:-$DATASET_BASE_ROOT/Turkish}"
SUBMIT_SCRIPT="${SUBMIT_SCRIPT:-$PROJECT_ROOT/scripts/submit_train_and_eval.sh}"
SUMMARY_SCRIPT="${SUMMARY_SCRIPT:-$PROJECT_ROOT/scripts/run_turkish_summary_slurm.sh}"
SBATCH_BIN="${SBATCH_BIN:-sbatch}"

export PROJECT_ROOT
export NPROC_PER_NODE
export MODEL_PATH
export ENV_ACTIVATE
export DATASET_BASE_ROOT
export TURKISH_DATASET_ROOT

cd "$PROJECT_ROOT"

if [ ! -f "$ENV_ACTIVATE" ]; then
    echo "Environment activate script not found: $ENV_ACTIVATE"
    exit 1
fi
# shellcheck disable=SC1090
source "$ENV_ACTIVATE"

if [ ! -f "$SUBMIT_SCRIPT" ]; then
    echo "Training submit helper not found: $SUBMIT_SCRIPT"
    exit 1
fi
if [ ! -f "$SUMMARY_SCRIPT" ]; then
    echo "Turkish summary Slurm script not found: $SUMMARY_SCRIPT"
    exit 1
fi
if ! command -v "$SBATCH_BIN" >/dev/null 2>&1; then
    echo "Slurm submission command not found: $SBATCH_BIN"
    exit 1
fi

CONFIGS="${CONFIGS//,/ }"
FOLDS="${FOLDS//,/ }"
read -r -a CONFIG_VALUES <<< "$CONFIGS"
read -r -a FOLD_VALUES <<< "$FOLDS"
if [ ${#CONFIG_VALUES[@]} -eq 0 ] || [ ${#FOLD_VALUES[@]} -eq 0 ]; then
    echo "CONFIGS and FOLDS must both be non-empty."
    exit 1
fi
for CONFIG in "${CONFIG_VALUES[@]}"; do
    if [ ! -f "$CONFIG" ]; then
        echo "Turkish config not found: $CONFIG"
        exit 1
    fi
done

LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/logs/slurm_turkish}"
mkdir -p "$LOG_ROOT"
exec > >(tee -a "$LOG_ROOT/submit-${SLURM_JOB_ID}.out")
exec 2> >(tee -a "$LOG_ROOT/submit-${SLURM_JOB_ID}.err" >&2)

extract_job_id() {
    local output="$1"
    local pattern="$2"
    printf '%s\n' "$output" | awk -v pattern="$pattern" '$0 ~ pattern {print $NF; exit}'
}

terminal_job_ids_from_submission() {
    local output="$1"
    local train_job_id=""
    local best_eval_job_id=""
    local last_eval_job_id=""
    local terminal_ids=()

    train_job_id="$(extract_job_id "$output" "Submitted training job:")"
    best_eval_job_id="$(extract_job_id "$output" "Submitted best-checkpoint eval job:")"
    last_eval_job_id="$(extract_job_id "$output" "Submitted last-checkpoint eval job:")"
    if [ -n "$best_eval_job_id" ]; then
        terminal_ids+=("$best_eval_job_id")
    fi
    if [ -n "$last_eval_job_id" ]; then
        terminal_ids+=("$last_eval_job_id")
    fi
    if [ ${#terminal_ids[@]} -eq 0 ]; then
        if [ -z "$train_job_id" ]; then
            echo "Could not parse a terminal Slurm job id from submission output." >&2
            return 1
        fi
        terminal_ids+=("$train_job_id")
    fi
    IFS=:; printf '%s\n' "${terminal_ids[*]}"
}

run_name_for_config() {
    local config="$1"
    local config_stem=""
    config_stem="$(basename "$config" .yaml)"
    if [ -n "$RUN_NAME_PREFIX" ]; then
        printf '%s_%s\n' "$RUN_NAME_PREFIX" "$config_stem"
    else
        printf '%s\n' "$config_stem"
    fi
}

job_tag_for_config() {
    local config="$1"
    case "$(basename "$config")" in
        turkish_audio_only.yaml) printf 'tr-audio\n' ;;
        turkish_text_only.yaml) printf 'tr-text\n' ;;
        turkish_audio_text.yaml) printf 'tr-audtxt\n' ;;
        *) printf 'tr-cv\n' ;;
    esac
}

echo "Building the shared Turkish manifest once before parallel submission."
python "$PROJECT_ROOT/src/data/build_manifest.py" --config "${CONFIG_VALUES[0]}"

echo "========================================"
echo "Turkish three-modality CV submission"
echo "========================================"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-}"
echo "CONFIGS: ${CONFIG_VALUES[*]}"
echo "FOLDS: ${FOLD_VALUES[*]}"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"
echo "TURKISH_DATASET_ROOT: $TURKISH_DATASET_ROOT"
echo "Each config starts immediately; folds within it are chained with afterok."
echo "========================================"

SUMMARY_JOB_IDS=()
for CONFIG in "${CONFIG_VALUES[@]}"; do
    RUN_NAME="$(run_name_for_config "$CONFIG")"
    JOB_TAG="$(job_tag_for_config "$CONFIG")"
    PREVIOUS_DEPENDENCY=""

    echo "Submitting config=$CONFIG run_name=$RUN_NAME"
    for FOLD in "${FOLD_VALUES[@]}"; do
        echo "  fold=$FOLD dependency=${PREVIOUS_DEPENDENCY:-<none>}"
        OUTPUT="$(
            env \
                PROJECT_ROOT="$PROJECT_ROOT" \
                CONFIG="$CONFIG" \
                FOLD="$FOLD" \
                RUN_NAME="$RUN_NAME" \
                NPROC_PER_NODE="$NPROC_PER_NODE" \
                MODEL_PATH="$MODEL_PATH" \
                ENV_ACTIVATE="$ENV_ACTIVATE" \
                DATASET_BASE_ROOT="$DATASET_BASE_ROOT" \
                TURKISH_DATASET_ROOT="$TURKISH_DATASET_ROOT" \
                SUBMIT_BEST_EVAL="$SUBMIT_BEST_EVAL" \
                SUBMIT_LAST_EVAL="$SUBMIT_LAST_EVAL" \
                EXTRA_TRAIN_ARGS="$EXTRA_TRAIN_ARGS" \
                EXTRA_EVAL_ARGS="" \
                SKIP_MANIFEST_BUILD=1 \
                SBATCH_JOB_NAME="${JOB_TAG}-f${FOLD}" \
                SBATCH_DEPENDENCY="$PREVIOUS_DEPENDENCY" \
                bash "$SUBMIT_SCRIPT"
        )"
        printf '%s\n' "$OUTPUT"
        TERMINAL_IDS="$(terminal_job_ids_from_submission "$OUTPUT")"
        PREVIOUS_DEPENDENCY="afterok:$TERMINAL_IDS"
    done

    SUMMARY_EXPORT="ALL,PROJECT_ROOT=$PROJECT_ROOT,CONFIG=$CONFIG,RUN_NAME=$RUN_NAME,ENV_ACTIVATE=$ENV_ACTIVATE,DATASET_BASE_ROOT=$DATASET_BASE_ROOT,TURKISH_DATASET_ROOT=$TURKISH_DATASET_ROOT"
    SUMMARY_JOB_RAW="$(
        "$SBATCH_BIN" --parsable \
            --job-name="${JOB_TAG}-sum" \
            --dependency="$PREVIOUS_DEPENDENCY" \
            --export="$SUMMARY_EXPORT" \
            "$SUMMARY_SCRIPT"
    )"
    SUMMARY_JOB_ID="${SUMMARY_JOB_RAW%%;*}"
    SUMMARY_JOB_IDS+=("$SUMMARY_JOB_ID")
    echo "Submitted summary job for $RUN_NAME: $SUMMARY_JOB_ID"
done

echo "========================================"
echo "Submission complete"
echo "GPU topology: 3 parallel chains × ${#FOLD_VALUES[@]} sequential folds"
echo "Total GPU training jobs: $((${#CONFIG_VALUES[@]} * ${#FOLD_VALUES[@]}))"
echo "Summary jobs: ${SUMMARY_JOB_IDS[*]}"
echo "========================================"
