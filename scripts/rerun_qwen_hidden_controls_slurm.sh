#!/bin/bash
#SBATCH -J hidden-controls
#SBATCH -A etur92
#SBATCH -q acc_ehpc
#SBATCH -t 00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH -o /gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression/logs/slurm_qwen_hidden/control-%j.out
#SBATCH -e /gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression/logs/slurm_qwen_hidden/control-%j.err
#SBATCH --chdir=/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression

set -euo pipefail
module purge
module load bsc/1.0
module load miniforge/24.3.0-0
source "${ENV_ACTIVATE:-/gpfs/projects/etur92/ozu647717/venvs/qwen_mn5_rebuilt/bin/activate}"

PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
VARIANTS="${VARIANTS:-xgb_raw_shuffled_labels}"
export PYTHONPATH="$PROJECT_ROOT/.deps/qwen_hidden:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT_ROOT"

python -m unittest tests.test_qwen_hidden_pipeline -q
while IFS= read -r metadata; do
  cache="$(dirname "$metadata")"
  output="${cache/hidden_features/hidden_classifiers}"
  # VARIANTS is an intentional whitespace-separated CLI list.
  # shellcheck disable=SC2086
  python baselines/qwen_hidden_classifier.py \
    --cache-dir "$cache" \
    --output-dir "$output" \
    --variants $VARIANTS
done < <(find outputs/hidden_features -name extraction_metadata.json -print | sort)

python baselines/summarize_qwen_hidden.py
