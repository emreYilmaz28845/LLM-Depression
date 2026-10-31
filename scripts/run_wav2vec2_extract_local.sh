#!/bin/bash
# Local (single-GPU workstation) run of the frozen wav2vec2 extraction.
#
# Same Python entry point and arguments as scripts/run_wav2vec2_extract_slurm.sh,
# but reference runs, manifests and audio are local copies. ROUTES_FILE lists one
# route per line: name|reference_fold_dir|final_eval_source|manifest_path.
# AUDIO_PATH_MAP is the JSON {mn5_path: local_path} written by the content-hash
# check; every manifest audio path must be in it.
#
# Required: ROUTES_FILE, AUDIO_PATH_MAP, OUTPUT_ROOT, MODEL_PATH.

set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

ROUTES_FILE="${ROUTES_FILE:?Set ROUTES_FILE}"
AUDIO_PATH_MAP="${AUDIO_PATH_MAP:?Set AUDIO_PATH_MAP}"
OUTPUT_ROOT="${OUTPUT_ROOT:?Set OUTPUT_ROOT}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH}"
MODEL_REVISION="${MODEL_REVISION:-1a640f32ac3e39899438a2931f9924c02f080a54}"
SEGMENT_SECONDS="${SEGMENT_SECONDS:-20}"
SMOKE_SUBJECTS="${SMOKE_SUBJECTS:-0}"
ONLY="${ONLY:-}"
ALL_LAYERS="${ALL_LAYERS:-0}"
EXTRA_ARGS=()
if [ "$ALL_LAYERS" = "1" ]; then EXTRA_ARGS+=(--all-layers); fi

mkdir -p "$OUTPUT_ROOT/logs"
LOG="$OUTPUT_ROOT/logs/w2v2-local-$(date -u +%Y%m%dT%H%M%SZ)"
exec > >(tee -a "$LOG.out") 2> >(tee -a "$LOG.err" >&2)
cp "$ROUTES_FILE" "$OUTPUT_ROOT/routes.txt.$(date -u +%Y%m%dT%H%M%SZ)"

python -c 'import torch, transformers, soundfile; print("versions", torch.__version__, transformers.__version__, soundfile.__version__, torch.cuda.get_device_name(0))'
echo "project_git=$(git -C "$PROJECT_ROOT" rev-parse HEAD) dirty=$(git -C "$PROJECT_ROOT" status --porcelain --untracked-files=no | wc -l)"
echo "output_root=$OUTPUT_ROOT segment_seconds=$SEGMENT_SECONDS smoke_subjects=$SMOKE_SUBJECTS all_layers=$ALL_LAYERS"

while IFS='|' read -r name ref source manifest; do
  [ -z "$name" ] && continue
  case "$name" in \#*) continue;; esac
  if [ -n "$ONLY" ] && [[ ",$ONLY," != *",$name,"* ]]; then continue; fi
  echo "=== $name $(date -u +%FT%TZ)"
  python "$PROJECT_ROOT/src/features/extract_wav2vec2.py" \
    --reference-fold-dir "$ref" \
    --manifest-path "$manifest" \
    --audio-path-map "$AUDIO_PATH_MAP" \
    --model-path "$MODEL_PATH" --model-revision "$MODEL_REVISION" \
    --segment-seconds "$SEGMENT_SECONDS" \
    --final-eval-source "$source" \
    --smoke-subjects "$SMOKE_SUBJECTS" \
    --output-root "$OUTPUT_ROOT/$name" "${EXTRA_ARGS[@]}"
done < "$ROUTES_FILE"
echo "=== done $(date -u +%FT%TZ)"
