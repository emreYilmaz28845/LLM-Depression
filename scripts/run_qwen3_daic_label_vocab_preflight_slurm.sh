#!/bin/bash
#SBATCH -J q3dlv-preflight
#SBATCH -A etur92
#SBATCH -q acc_ehpc
#SBATCH -t 04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#SBATCH --chdir=/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression

# CPU preflight for the Qwen3 DAIC label-vocabulary campaign.
#
# Modes:
#   manifest : build the shared DAIC manifest and splits into the lane runtime
#              root with the same resolved overrides the training jobs will use,
#              then record the sha256 of the four prebuilt input files.
#   audit    : run the backend-aware answer-label token audit against the real
#              DAIC train/val rows of the prebuilt manifest. The environment must
#              match the backend (qwen38 environment for the text-only cell,
#              qwen3omni environment for the two audio cells).
#
# Inputs:
#   PROJECT_ROOT   deployed code root (required)
#   CONFIG         config path relative to PROJECT_ROOT (required)
#   MODE           manifest | audit (required)
#   RUNTIME_ROOT   lane runtime root (required; the writable runtime contract)
#   ENV_ACTIVATE   environment activate script (required)
#   PREP_OUTPUT    output directory for audit JSONs and hashes (required)
#   MODALITIES     audit mode only: space separated modalities (default text_only)
#   TAG            optional run/job label used in output file names
#
# The audit never loads model weights and never writes transcripts or subject ids
# into its report.

set -e
set -o pipefail

module purge
module load bsc/1.0
module load miniforge/24.3.0-0

ENV_ACTIVATE="${ENV_ACTIVATE:?Set ENV_ACTIVATE to the backend environment}"
if [ ! -f "$ENV_ACTIVATE" ]; then
    echo "Environment activate script not found: $ENV_ACTIVATE" >&2
    exit 1
fi
# shellcheck disable=SC1090
source "$ENV_ACTIVATE"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PROJECT_ROOT="${PROJECT_ROOT:?Set PROJECT_ROOT to the deployed code path}"
CONFIG="${CONFIG:?Set CONFIG}"
MODE="${MODE:?Set MODE to manifest or audit}"
RUNTIME_ROOT="${RUNTIME_ROOT:?Set RUNTIME_ROOT}"
PREP_OUTPUT="${PREP_OUTPUT:?Set PREP_OUTPUT}"
MODALITIES="${MODALITIES:-text_only}"
TAG="${TAG:-preflight}"

cd "$PROJECT_ROOT"
export PROJECT_ROOT

MANIFEST_DIR="$RUNTIME_ROOT/manifests/daic"
SPLIT_DIR="$RUNTIME_ROOT/splits/daic"
mkdir -p "$PREP_OUTPUT" "$MANIFEST_DIR" "$SPLIT_DIR"

DATASET_BASE_ROOT="${DATASET_BASE_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/Datasets}"
export DAIC_DATASET_ROOT="${DAIC_DATASET_ROOT:-$DATASET_BASE_ROOT/DAIC-WOZ/preprocessed}"
export DAIC_UNPROCESSED_ROOT="${DAIC_UNPROCESSED_ROOT:-$DATASET_BASE_ROOT/DAIC-WOZ/unprocessed}"
export DAIC_LABEL_ROOT="${DAIC_LABEL_ROOT:-$DATASET_BASE_ROOT/DAIC-WOZ/minimal_zips}"

RUN_LOG="$PREP_OUTPUT/${TAG}-${MODE}-${SLURM_JOB_ID:-local}-$(date +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$RUN_LOG") 2>&1

echo "mode=$MODE config=$CONFIG runtime_root=$RUNTIME_ROOT env=$ENV_ACTIVATE"
python -V
python -c "import transformers; print('transformers', transformers.__version__)"

OVERRIDE_ARGS=(
  "--set" "output_dirs.manifest_dir=$MANIFEST_DIR"
  "--set" "output_dirs.split_dir=$SPLIT_DIR"
)

case "$MODE" in
  manifest)
    python -m src.data.build_manifest --config "$CONFIG" "${OVERRIDE_ARGS[@]}"
    python - "$MANIFEST_DIR" "$SPLIT_DIR" "$PREP_OUTPUT/prebuilt_inputs_sha256.json" <<'PY'
import hashlib, json, sys
from pathlib import Path
manifest_dir, split_dir, output = (Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
files = {
    "manifest": manifest_dir / "daic_manifest.jsonl",
    "manifest_csv": manifest_dir / "daic_manifest.csv",
    "folds": split_dir / "daic_folds.json",
    "split_metadata": split_dir / "daic_manifest_metadata.json",
    "subject_partitions": split_dir / "daic_subject_partitions.json",
}
report = {}
for name, path in files.items():
    if not path.is_file():
        raise SystemExit(f"missing prebuilt input: {path}")
    report[name] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}
output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps({name: item["sha256"][:16] for name, item in report.items()}, indent=2))
PY
    ;;
  audit)
    for modality in $MODALITIES; do
      python tools/verify_label_vocab_tokens.py \
        --modality "$modality" \
        --manifest "$MANIFEST_DIR/daic_manifest.jsonl" \
        --split-metadata "$SPLIT_DIR/daic_subject_partitions.json" \
        --partition train --partition val \
        --real-limit 3 --require-real-examples \
        --output "$PREP_OUTPUT/token_audit_${modality}_${TAG}.json"
    done
    ;;
  *)
    echo "unknown MODE: $MODE" >&2
    exit 2
    ;;
esac

echo "preflight complete: mode=$MODE output=$PREP_OUTPUT"
