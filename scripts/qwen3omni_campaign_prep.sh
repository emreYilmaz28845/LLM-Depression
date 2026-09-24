#!/bin/bash
#SBATCH -J qwen3omni-campaign-prep
#SBATCH -A etur92
#SBATCH -q acc_ehpc
#SBATCH -t 04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:1
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#SBATCH --chdir=/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression

# Qwen3-Omni standalone prompt-context campaign preflight worker.
#
# One cell per job: environment audit, module-tree audit and the risk inventory
# (model-free plus the real processor) for that cell's config. One GPU so CUDA
# visibility is recorded; no model weights are loaded and nothing is written
# outside PREP_OUTPUT.
#
#   PROJECT_ROOT=<deployed code path>       (required)
#   CELL=<cell id>                          (required; names the output subdirectory)
#   CONFIG=<config path>                    (required; relative to PROJECT_ROOT)
#   PREP_OUTPUT=<directory>                 (required; probe runtime directory)
#   PREP_MODES="<mode> [<mode> ...]"        (default: env)
#       env        environment audit and pip freeze
#       tree       module-tree audit (config only, no weights)
#       inventory  risk inventory over the resolved manifest, with processor
#   OVERRIDES_JSON_B64 / EXTRA_PREP_ARGS: manifest/split dir overrides

set -e
set -o pipefail

module purge
module load bsc/1.0
module load miniforge/24.3.0-0

ENV_ACTIVATE="${ENV_ACTIVATE:-/gpfs/projects/etur92/ozu647717/venvs/qwen3omni/bin/activate}"
if [ -f "$ENV_ACTIVATE" ]; then
    # shellcheck disable=SC1090
    source "$ENV_ACTIVATE"
else
    echo "Environment activate script not found: $ENV_ACTIVATE"
    exit 1
fi

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PROJECT_ROOT="${PROJECT_ROOT:?Set PROJECT_ROOT to the deployed code path}"
export PROJECT_ROOT
CELL="${CELL:?Set CELL to the cell id}"
CONFIG="${CONFIG:?Set CONFIG}"
PREP_OUTPUT="${PREP_OUTPUT:?Set PREP_OUTPUT}"
PREP_MODES="${PREP_MODES:-env}"
EXTRA_PREP_ARGS="${EXTRA_PREP_ARGS:-}"
OVERRIDES_JSON_B64="${OVERRIDES_JSON_B64:-}"

cd "$PROJECT_ROOT"
CELL_OUTPUT="$PREP_OUTPUT/$CELL"
mkdir -p "$CELL_OUTPUT"

DATASET_BASE_ROOT="${DATASET_BASE_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/Datasets}"
export DAIC_DATASET_ROOT="${DAIC_DATASET_ROOT:-$DATASET_BASE_ROOT/DAIC-WOZ/preprocessed}"
export DAIC_UNPROCESSED_ROOT="${DAIC_UNPROCESSED_ROOT:-$DATASET_BASE_ROOT/DAIC-WOZ/unprocessed}"
export DAIC_LABEL_ROOT="${DAIC_LABEL_ROOT:-$DATASET_BASE_ROOT/DAIC-WOZ/minimal_zips}"
export D3TEC_DATASET_ROOT="${D3TEC_DATASET_ROOT:-$DATASET_BASE_ROOT/D3TEC DATASET/D3TEC DATASET}"
export D3TEC_FULL_TRANSCRIPTS="${D3TEC_FULL_TRANSCRIPTS:-$D3TEC_DATASET_ROOT/transcripts_qwen3_asr_spanish.jsonl}"
export D3TEC_SEGMENT_TRANSCRIPTS="${D3TEC_SEGMENT_TRANSCRIPTS:-$D3TEC_DATASET_ROOT/transcripts_qwen3_asr_spanish_segments.jsonl}"
export ANDROIDS_DATASET_ROOT="${ANDROIDS_DATASET_ROOT:-$DATASET_BASE_ROOT/Androids-Corpus/Androids-Corpus}"
export ANDROIDS_INTERVIEW_FULL_TRANSCRIPTS="${ANDROIDS_INTERVIEW_FULL_TRANSCRIPTS:-$ANDROIDS_DATASET_ROOT/interview_transcripts_qwen3_asr_italian.jsonl}"
export ANDROIDS_INTERVIEW_SEGMENT_TRANSCRIPTS="${ANDROIDS_INTERVIEW_SEGMENT_TRANSCRIPTS:-$ANDROIDS_DATASET_ROOT/interview_transcripts_qwen3_asr_italian_segments.jsonl}"
export CMDC_DATASET_ROOT="${CMDC_DATASET_ROOT:-$DATASET_BASE_ROOT/CMDC}"
export TURKISH_DATASET_ROOT="${TURKISH_DATASET_ROOT:-$DATASET_BASE_ROOT/Turkish}"

if [ -n "$OVERRIDES_JSON_B64" ]; then
    mapfile -t -d '' OVERRIDE_ARGS < <(python - "$OVERRIDES_JSON_B64" <<'PY'
import base64, json, sys
tokens = json.loads(base64.b64decode(sys.argv[1]).decode("utf-8"))
sys.stdout.write("\0".join(tokens))
PY
)
else
    read -r -a OVERRIDE_ARGS <<< "$EXTRA_PREP_ARGS"
fi

TIMESTAMP="$(date +%Y-%m-%d_%H:%M:%S)"
RUN_LOG="$CELL_OUTPUT/prep-${SLURM_JOB_ID:-local}-${TIMESTAMP}.log"
exec > >(tee -a "$RUN_LOG") 2>&1

echo "========================================"
echo "Qwen3-Omni campaign preflight | job=${SLURM_JOB_ID:-} | host=$(hostname)"
echo "cell=$CELL"
echo "project_root=$PROJECT_ROOT"
echo "config=$CONFIG"
echo "prep_output=$CELL_OUTPUT"
echo "modes=$PREP_MODES"
echo "overrides=${OVERRIDE_ARGS[*]:-<none>}"
echo "========================================"
nvidia-smi
python -V

for mode in $PREP_MODES; do
    case "$mode" in
        env)
            echo "=== pip check ==="
            python -m pip check
            echo "=== environment freeze ==="
            python -m pip freeze > "$CELL_OUTPUT/pip_freeze.txt"
            wc -l "$CELL_OUTPUT/pip_freeze.txt"
            echo "=== environment audit ==="
            python scripts/qwen3omni_env_audit.py --output "$CELL_OUTPUT" --config "$CONFIG"
            ;;
        tree)
            echo "=== module-tree audit ==="
            python scripts/qwen3omni_backend_probe.py tree \
                --config "$CONFIG" \
                --output "$CELL_OUTPUT/module_tree.json" \
                "${OVERRIDE_ARGS[@]}"
            ;;
        inventory)
            echo "=== risk inventory (model-free plus processor) ==="
            python scripts/qwen3omni_risk_inventory.py \
                --config "$CONFIG" \
                --output "$CELL_OUTPUT/risk_inventory.json" \
                --with-processor \
                "${OVERRIDE_ARGS[@]}"
            ;;
        *)
            echo "unknown PREP_MODE: $mode" >&2
            exit 2
            ;;
    esac
done

echo "preflight complete: $CELL_OUTPUT"
