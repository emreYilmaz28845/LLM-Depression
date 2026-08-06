#!/bin/bash
#SBATCH -J jk4-eval-det
#SBATCH -A etur92
#SBATCH -q acc_ehpc
#SBATCH -t 24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --exclude=as01r2b12
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#SBATCH --chdir=/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression

set -euo pipefail
module purge
module load bsc/1.0
module load miniforge/24.3.0-0

ENV_ACTIVATE="${ENV_ACTIVATE:-/gpfs/projects/etur92/ozu647717/venvs/qwen_mn5_rebuilt/bin/activate}"
source "$ENV_ACTIVATE"
PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression}"
export PROJECT_ROOT
cd "$PROJECT_ROOT"

CONFIG="${CONFIG:?Set CONFIG}"
FOLD="${FOLD:-0}"
RUN_NAME="${RUN_NAME:?Set RUN_NAME}"
SEED="${SEED:-1337}"
SMOKE_SUBJECT_LIMIT="${SMOKE_SUBJECT_LIMIT:-6}"
MODEL_PATH="${MODEL_PATH:-}"
TEXT_MODEL_PATH="${TEXT_MODEL_PATH:-}"
DAIC_UNPROCESSED_ROOT="${DAIC_UNPROCESSED_ROOT:?Set DAIC_UNPROCESSED_ROOT}"
DAIC_LABEL_ROOT="${DAIC_LABEL_ROOT:?Set DAIC_LABEL_ROOT}"
export DAIC_UNPROCESSED_ROOT DAIC_LABEL_ROOT

LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/logs/slurm_daic_participant_packed30_jointk4}"
mkdir -p "$LOG_ROOT"
exec > >(tee -a "$LOG_ROOT/eval-determinism-${SLURM_JOB_ID}.out")
exec 2> >(tee -a "$LOG_ROOT/eval-determinism-${SLURM_JOB_ID}.err" >&2)

# Deterministic evaluation gate (smoke): run the teacher-forced evaluation
# twice in bf16 into pass1/pass2 and require byte-identical normalized sample
# rows, subject rows, and metrics. The outputs double as the smoke Qwen
# evaluation (one final row per smoke test subject).

RUN_ROOT="$(python - "$CONFIG" <<PY
import sys
from pathlib import Path
sys.path.insert(0, "$PROJECT_ROOT")
from src.utils import load_yaml_with_overrides
config = load_yaml_with_overrides(Path("$CONFIG"), [])
print(Path(config["output_dirs"]["run_root"]))
PY
)"
BEST_DIR="$RUN_ROOT/$RUN_NAME/fold_$FOLD/best_model"
if [ ! -f "$BEST_DIR/adapter_model.safetensors" ] || [ ! -f "$BEST_DIR/adapter_config.json" ]; then
    echo "Refusing evaluation: best_model is missing at $BEST_DIR" >&2
    exit 1
fi

run_pass() {
  local pass="$1"
  echo "== packed30 jointk4 deterministic evaluation pass $pass =="
  python "$PROJECT_ROOT/src/evaluate.py" \
    --config "$CONFIG" --fold "$FOLD" --checkpoint_dir "$BEST_DIR" \
    --set "seed=$SEED" --set "split.seed=1337" \
    --set "evaluation.inference_dtype=bf16" \
    --set "split.smoke_subject_limit=$SMOKE_SUBJECT_LIMIT" \
    --output_dir "$BEST_DIR/standalone_eval_pass$pass"
}

run_pass 1
run_pass 2

echo "== comparing deterministic evaluation passes =="
python - "$BEST_DIR" <<PY
import json
import sys
from pathlib import Path

best_dir = Path(sys.argv[1])

def normalized_lines(path: Path):
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    return sorted(json.dumps(row, sort_keys=True, separators=(",", ":")) for row in rows)

def compare(name):
    first = best_dir / f"standalone_eval_pass1/{name}"
    second = best_dir / f"standalone_eval_pass2/{name}"
    if not first.is_file() or not second.is_file():
        raise SystemExit(f"Determinism gate: missing {name} in a pass.")
    a = normalized_lines(first)
    b = normalized_lines(second)
    if a != b:
        raise SystemExit(f"Determinism gate FAILED: {name} differs between passes.")
    print(f"pass1 == pass2 ({name}): {len(a)} rows")

compare("predictions_sample_level.jsonl")
compare("predictions_subject_level.csv")
compare("metrics_original_teacher_forced.json")
compare("final_and_best_validation_metrics.json")
print("Determinism gate PASSED: both evaluation passes are byte-identical.")
PY

echo "== packed30 jointk4 deterministic evaluation finished =="
