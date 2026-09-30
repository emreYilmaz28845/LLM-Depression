#!/bin/bash
#SBATCH -J qwen3-pooled-preflight
#SBATCH -A etur92
#SBATCH -q acc_ehpc
#SBATCH -t 04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#SBATCH --chdir=/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression

# CPU-only preflight for the Qwen3 Turkish pooled defaults. One stage per job:
#
#   STAGE=pooled   build the pooled native+English manifests into the task
#                  runtime and audit them against the recorded contract
#                  (needs SOURCE_INPUT_ROOT with the staged source pairs);
#   STAGE=qwen38   render the five Turkish cells' prompts with the real Qwen3.8
#                  tokenizer/chat template (no weights) and audit leakage,
#                  single-token answer labels and context fit;
#   STAGE=qwen3omni load the Qwen3-Omni processor (no weights) and run the
#                  audio feature extraction on real pooled windows.
#
# No model weights are loaded and no network access is used. Every stage reads
# and writes the task runtime (RUNTIME_ROOT), never the shared checkout.

set -euo pipefail
module purge
module load bsc/1.0
module load miniforge/24.3.0-0

PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
STAGE="${STAGE:?STAGE must be pooled, qwen38 or qwen3omni}"
RUNTIME_ROOT="${RUNTIME_ROOT:?RUNTIME_ROOT must be the task runtime root}"
SOURCE_INPUT_ROOT="${SOURCE_INPUT_ROOT:-}"
QWEN38_MODEL_PATH="${QWEN38_MODEL_PATH:-/gpfs/projects/etur92/ozu647717/models/Qwen3.8-27B}"
QWEN3OMNI_MODEL_PATH="${QWEN3OMNI_MODEL_PATH:-/gpfs/projects/etur92/ozu647717/models/Qwen3-Omni-30B-A3B-Instruct}"

case "$STAGE" in
    pooled)
        ENV_ACTIVATE="${ENV_ACTIVATE:-/gpfs/projects/etur92/ozu647717/venvs/qwen_mn5_rebuilt/bin/activate}"
        ;;
    qwen38)
        ENV_ACTIVATE="${ENV_ACTIVATE:-/gpfs/projects/etur92/ozu647717/venvs/qwen38_fsdp_fastpath_20260921/bin/activate}"
        ;;
    qwen3omni)
        ENV_ACTIVATE="${ENV_ACTIVATE:-/gpfs/projects/etur92/ozu647717/venvs/qwen3omni/bin/activate}"
        ;;
    *)
        echo "Unsupported STAGE: $STAGE" >&2
        exit 2
        ;;
esac

if [ ! -f "$ENV_ACTIVATE" ]; then
    echo "Environment activate script not found: $ENV_ACTIVATE" >&2
    exit 1
fi
# shellcheck disable=SC1090
source "$ENV_ACTIVATE"
export PROJECT_ROOT

LOG_ROOT="${LOG_ROOT:-$RUNTIME_ROOT/logs/$STAGE}"
mkdir -p "$LOG_ROOT"
exec > >(tee -a "$LOG_ROOT/preflight-${SLURM_JOB_ID}.out")
exec 2> >(tee -a "$LOG_ROOT/preflight-${SLURM_JOB_ID}.err" >&2)

cd "$PROJECT_ROOT"
echo "stage: $STAGE"
echo "project_root: $PROJECT_ROOT"
echo "runtime_root: $RUNTIME_ROOT"
echo "env_activate: $ENV_ACTIVATE"

case "$STAGE" in
    pooled)
        if [ -z "$SOURCE_INPUT_ROOT" ]; then
            echo "SOURCE_INPUT_ROOT is required for STAGE=pooled" >&2
            exit 2
        fi
        python "$PROJECT_ROOT/scripts/build_turkish_pooled_manifest.py" \
            --positive-native-manifest "$SOURCE_INPUT_ROOT/manifests/pos_native/turkish_manifest.jsonl" \
            --positive-native-split "$SOURCE_INPUT_ROOT/splits/pos_native/turkish_folds.json" \
            --negative-native-manifest "$SOURCE_INPUT_ROOT/manifests/neg_native/turkish_manifest.jsonl" \
            --negative-native-split "$SOURCE_INPUT_ROOT/splits/neg_native/turkish_folds.json" \
            --positive-english-manifest "$SOURCE_INPUT_ROOT/manifests/pos_english/turkish_manifest.jsonl" \
            --positive-english-split "$SOURCE_INPUT_ROOT/splits/pos_english/turkish_folds.json" \
            --negative-english-manifest "$SOURCE_INPUT_ROOT/manifests/neg_english/turkish_manifest.jsonl" \
            --negative-english-split "$SOURCE_INPUT_ROOT/splits/neg_english/turkish_folds.json" \
            --native-output-dir "$RUNTIME_ROOT/manifests/turkish" \
            --english-output-dir "$RUNTIME_ROOT/manifests_en/turkish" \
            --native-split-output-dir "$RUNTIME_ROOT/splits/turkish" \
            --english-split-output-dir "$RUNTIME_ROOT/splits_en/turkish" \
            --audit-output "$RUNTIME_ROOT/preflight/pooled_manifest_audit.json"
        python "$PROJECT_ROOT/tools/audit_qwen3_pooled_inputs.py" \
            --runtime-root "$RUNTIME_ROOT" \
            --expect-recorded-hashes \
            --output "$RUNTIME_ROOT/preflight/pooled_inputs_audit.json"
        python "$PROJECT_ROOT/tools/qwen3_pooled_defaults.py" \
            --check \
            --emit "$RUNTIME_ROOT/preflight/selection_map.json"
        ;;
    qwen38)
        python "$PROJECT_ROOT/tools/audit_qwen3_prompt_rendering.py" \
            --runtime-root "$RUNTIME_ROOT" \
            --model-path "$QWEN38_MODEL_PATH" \
            --output "$RUNTIME_ROOT/preflight/qwen38_prompt_audit.json"
        ;;
    qwen3omni)
        python "$PROJECT_ROOT/tools/audit_qwen3omni_processor.py" \
            --runtime-root "$RUNTIME_ROOT" \
            --model-path "$QWEN3OMNI_MODEL_PATH" \
            --output "$RUNTIME_ROOT/preflight/qwen3omni_processor_audit.json"
        ;;
esac
