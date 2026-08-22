#!/usr/bin/env bash
# Transcribe only the scored Turkish negative-question recordings on the local RTX 4090.
#
# Fresh full run:
#   bash scripts/transcribe_turkish_negative_only_qwen3asr.sh
# Resume an interrupted or failed run:
#   bash scripts/transcribe_turkish_negative_only_qwen3asr.sh --resume
# Four-file smoke to a disposable output:
#   bash scripts/transcribe_turkish_negative_only_qwen3asr.sh \
#     --limit 4 --out /tmp/turkish_negative_only_qwen3asr_smoke.jsonl
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_ROOT="${TURKISH_NEGATIVE_ONLY_ROOT:-/media/emre/Backup/AudioLLM/Datasets/Turkish_Negative_Only}"
METADATA_CSV="${TURKISH_NEGATIVE_ONLY_METADATA:-$DATASET_ROOT/metadata_turkish_negative_only_t17.csv}"
OUT_PATH="${TURKISH_NEGATIVE_ONLY_TRANSCRIPTS:-$DATASET_ROOT/whisper_transcripts_qwen3_asr.jsonl}"

test -d "$DATASET_ROOT" || { echo "Dataset root not found: $DATASET_ROOT" >&2; exit 1; }
test -f "$METADATA_CSV" || { echo "Metadata CSV not found: $METADATA_CSV" >&2; exit 1; }

exec bash "$PROJECT_ROOT/scripts/transcribe_turkish_qwen3asr.sh" -- \
    --audio-dir "$DATASET_ROOT" \
    --metadata-csv "$METADATA_CSV" \
    --out "$OUT_PATH" \
    "$@"
