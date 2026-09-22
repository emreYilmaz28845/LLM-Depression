#!/bin/bash
#SBATCH -J pooled-prebuilt-manifest
#SBATCH -A etur92
#SBATCH -q acc_ehpc
#SBATCH -t 02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#SBATCH --chdir=/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression

# Build the pooled Turkish manifest from the staged source inputs, in a compute
# job: the builder verifies that every source audio file exists, and the Turkish
# dataset is mounted on the cluster only.
#
# Inputs (all under the task runtime, never the shared checkout):
#   SOURCE_INPUT_ROOT : staged MN5-path source inputs (see
#                       scripts/stage_promptcontext_pooled_sources.py)
#   OUTPUT_ROOT       : runtime root the pooled manifest and splits are written to
#
# The audit printed at the end records the manifest hashes the submission later
# verifies (manifest_policy=prebuilt).

set -euo pipefail
module purge
module load bsc/1.0
module load miniforge/24.3.0-0

PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
ENV_ACTIVATE="${ENV_ACTIVATE:-/gpfs/projects/etur92/ozu647717/venvs/qwen_mn5_rebuilt/bin/activate}"
SOURCE_INPUT_ROOT="${SOURCE_INPUT_ROOT:?SOURCE_INPUT_ROOT is required}"
OUTPUT_ROOT="${OUTPUT_ROOT:?OUTPUT_ROOT is required}"

if [ ! -f "$ENV_ACTIVATE" ]; then
    echo "Environment activate script not found: $ENV_ACTIVATE" >&2
    exit 1
fi
# shellcheck disable=SC1090
source "$ENV_ACTIVATE"
export PROJECT_ROOT
cd "$PROJECT_ROOT"

LOG_ROOT="${LOG_ROOT:-$OUTPUT_ROOT/logs}"
mkdir -p "$LOG_ROOT"
exec > >(tee -a "$LOG_ROOT/pooled_prebuilt_manifest-${SLURM_JOB_ID}.out")
exec 2> >(tee -a "$LOG_ROOT/pooled_prebuilt_manifest-${SLURM_JOB_ID}.err" >&2)

echo "source_input_root: $SOURCE_INPUT_ROOT"
echo "output_root: $OUTPUT_ROOT"

python "$PROJECT_ROOT/scripts/build_turkish_pooled_manifest.py" \
    --positive-native-manifest "$SOURCE_INPUT_ROOT/manifests/pos_native/turkish_manifest.jsonl" \
    --positive-native-split "$SOURCE_INPUT_ROOT/splits/pos_native/turkish_folds.json" \
    --negative-native-manifest "$SOURCE_INPUT_ROOT/manifests/neg_native/turkish_manifest.jsonl" \
    --negative-native-split "$SOURCE_INPUT_ROOT/splits/neg_native/turkish_folds.json" \
    --positive-english-manifest "$SOURCE_INPUT_ROOT/manifests/pos_english/turkish_manifest.jsonl" \
    --positive-english-split "$SOURCE_INPUT_ROOT/splits/pos_english/turkish_folds.json" \
    --negative-english-manifest "$SOURCE_INPUT_ROOT/manifests/neg_english/turkish_manifest.jsonl" \
    --negative-english-split "$SOURCE_INPUT_ROOT/splits/neg_english/turkish_folds.json" \
    --native-output-dir "$OUTPUT_ROOT/manifests/turkish" \
    --english-output-dir "$OUTPUT_ROOT/manifests_en/turkish" \
    --native-split-output-dir "$OUTPUT_ROOT/splits/turkish" \
    --english-split-output-dir "$OUTPUT_ROOT/splits_en/turkish" \
    --audit-output "$OUTPUT_ROOT/preflight/pooled_manifest_audit.json"
