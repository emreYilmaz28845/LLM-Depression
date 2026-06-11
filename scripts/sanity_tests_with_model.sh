#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-/home/emre/models/Qwen2-Audio-7B-Instruct}"

run_python() {
  if [[ -n "${CONDA_ENV:-}" ]]; then
    conda run -n "$CONDA_ENV" "$PYTHON_BIN" "$@"
  else
    "$PYTHON_BIN" "$@"
  fi
}

run_python - <<'PY'
import os
import sys
from pathlib import Path

project_root = Path("/home/emre/Projects/AudioLLM/LLM-Depression")
sys.path.insert(0, str(project_root))

import torch

from src.data.build_manifest import build_for_config
from src.data.runtime import AudioTextDataset, build_examples, filter_rows_by_subjects, load_manifest_rows
from src.evaluate import generate_label_text, score_candidate_label
from src.model.collator import Qwen2AudioSFTCollator
from src.model.qwen2audio_lora import load_model_for_training, load_processor
from src.utils import load_yaml

config_path = project_root / "configs/daic_audio_text.yaml"
config = load_yaml(config_path)
config["model_name_or_path"] = os.environ.get("MODEL_PATH", "/home/emre/models/Qwen2-Audio-7B-Instruct")
audio_only_config_path = project_root / "configs/daic_audio_only.yaml"
audio_only_config = load_yaml(audio_only_config_path)
audio_only_config["model_name_or_path"] = config["model_name_or_path"]
metadata_path = project_root / "outputs/splits/daic_manifest_metadata.json"
if not metadata_path.exists():
    build_for_config(config_path)

import json
metadata = json.loads(metadata_path.read_text())
rows = load_manifest_rows(metadata["manifest_path"])
train_rows = [row for row in rows if row["split_original"] == "train"]
subject_id = train_rows[0]["subject_id"]
examples = build_examples(filter_rows_by_subjects(train_rows, [subject_id]), config, partition_name="smoke")
example = examples[0]
audio_only_examples = build_examples(
    filter_rows_by_subjects(train_rows, [subject_id]),
    audio_only_config,
    partition_name="smoke_audio_only",
)
audio_only_example = audio_only_examples[0]

processor = load_processor(config["model_name_or_path"])
dataset = AudioTextDataset([example], processor_sampling_rate=processor.feature_extractor.sampling_rate, silence_audio=False)
collator = Qwen2AudioSFTCollator(processor=processor, debug=True)
batch = collator([dataset[0]])
print("Loaded processor and collated one batch.")
print("Unmasked token ids:", collator.last_debug_example["unmasked_token_ids"])

audio_only_dataset = AudioTextDataset(
    [audio_only_example],
    processor_sampling_rate=processor.feature_extractor.sampling_rate,
    silence_audio=False,
)
audio_only_collator = Qwen2AudioSFTCollator(processor=processor, debug=True)
audio_only_batch = audio_only_collator([audio_only_dataset[0]])
print("Loaded audio-only processor inputs and collated one batch.")
print("Audio-only unmasked token ids:", audio_only_collator.last_debug_example["unmasked_token_ids"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model_for_training(config["model_name_or_path"], config).to(device)
batch = {key: value.to(device) for key, value in batch.items()}
with torch.no_grad():
    outputs = model(**batch)
print("Forward pass loss:", float(outputs.loss))

audio_only_batch = {key: value.to(device) for key, value in audio_only_batch.items()}
with torch.no_grad():
    audio_only_outputs = model(**audio_only_batch)
print("Audio-only forward pass loss:", float(audio_only_outputs.loss))

model.eval()
dep_score = score_candidate_label(model, processor, example, "Depressed", device, silence_audio=False)
non_score = score_candidate_label(model, processor, example, "Non-depressed", device, silence_audio=False)
generation = generate_label_text(model, processor, example, config, device, silence_audio=False)
print("Depressed score:", dep_score)
print("Non-depressed score:", non_score)
print("Generation:", generation)

audio_only_dep_score = score_candidate_label(model, processor, audio_only_example, "Depressed", device, silence_audio=False)
audio_only_non_score = score_candidate_label(
    model,
    processor,
    audio_only_example,
    "Non-depressed",
    device,
    silence_audio=False,
)
audio_only_generation = generate_label_text(
    model,
    processor,
    audio_only_example,
    audio_only_config,
    device,
    silence_audio=False,
)
print("Audio-only Depressed score:", audio_only_dep_score)
print("Audio-only Non-depressed score:", audio_only_non_score)
print("Audio-only generation:", audio_only_generation)
PY
