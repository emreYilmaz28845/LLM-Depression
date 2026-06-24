#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${CONFIG:-$PROJECT_ROOT/configs/turkish_audio_text.yaml}"
RUN_NAME="${RUN_NAME:-turkish_audio_text}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
SEEDS="${SEEDS:-1337 7 2024}"

CONFIG_RUN_ROOT="$(
  cd "$PROJECT_ROOT"
  python - "$CONFIG" <<'PY'
import sys
from src.utils import load_yaml_with_overrides

print(load_yaml_with_overrides(sys.argv[1], [])["output_dirs"]["run_root"])
PY
)"

read -r -a SEED_VALUES <<< "${SEEDS//,/ }"
RUN_ROOTS=()
for SEED in "${SEED_VALUES[@]}"; do
  SEED_RUN_NAME="${RUN_NAME}_s${SEED}"
  python "$PROJECT_ROOT/src/data/build_manifest.py" \
    --config "$CONFIG" \
    --set "seed=$SEED" \
    --set "split.seed=$SEED"

  for FOLD in 0 1 2 3 4; do
    torchrun --nproc_per_node="$NPROC_PER_NODE" \
      "$PROJECT_ROOT/src/train.py" \
      --config "$CONFIG" \
      --fold "$FOLD" \
      --run_name "$SEED_RUN_NAME" \
      --set "seed=$SEED" \
      --set "split.seed=$SEED"
  done

  SEED_RUN_ROOT="$CONFIG_RUN_ROOT/$SEED_RUN_NAME"
  RUN_ROOTS+=("$SEED_RUN_ROOT")
  python "$PROJECT_ROOT/src/summarize_runs.py" --run_root "$SEED_RUN_ROOT"
done

python "$PROJECT_ROOT/scripts/summarize_seed_sweep.py" \
  --run-roots "${RUN_ROOTS[@]}" \
  --output "$CONFIG_RUN_ROOT/${RUN_NAME}_seed_sweep_summary.json"
