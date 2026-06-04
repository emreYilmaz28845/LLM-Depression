#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

run_python() {
  if [[ -n "${CONDA_ENV:-}" ]]; then
    conda run -n "$CONDA_ENV" "$PYTHON_BIN" "$@"
  else
    "$PYTHON_BIN" "$@"
  fi
}

echo "[1/4] Building manifests and split metadata"
"$PROJECT_ROOT/scripts/validate_manifests.sh"

echo "[2/4] Verifying expected audit/split files and EDAIC invariants"
run_python - <<'PY'
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from src.train import _resolve_outer_partitions
from src.utils import load_yaml_with_overrides, resolve_metadata_paths

root = Path("/home/emre/Projects/AudioLLM/LLM-Depression")
required = [
    root / "outputs/manifests/daic_manifest.jsonl",
    root / "outputs/manifests/edaic_manifest.jsonl",
    root / "outputs/manifests/cmdc_manifest.jsonl",
    root / "outputs/manifests/eatd_manifest.jsonl",
    root / "outputs/splits/daic_join_audit.csv",
    root / "outputs/splits/edaic_join_audit.csv",
    root / "outputs/splits/edaic_subject_partitions.json",
    root / "outputs/splits/edaic_manifest_metadata.json",
    root / "outputs/splits/cmdc_fold_report.json",
    root / "outputs/splits/eatd_folds.json",
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit(f"Missing expected outputs: {missing}")

edaic_manifest_rows = [
    json.loads(line)
    for line in (root / "outputs/manifests/edaic_manifest.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
if len(edaic_manifest_rows) != 3080:
    raise SystemExit(f"Unexpected EDAIC sample count: {len(edaic_manifest_rows)}")

edaic_subject_ids = {row["subject_id"] for row in edaic_manifest_rows}
if len(edaic_subject_ids) != 275:
    raise SystemExit(f"Unexpected EDAIC subject count: {len(edaic_subject_ids)}")

sample_partition_counts = Counter(row["split_original"] for row in edaic_manifest_rows)
expected_sample_partition_counts = {"train": 1815, "val": 620, "test": 645}
if dict(sample_partition_counts) != expected_sample_partition_counts:
    raise SystemExit(
        f"Unexpected EDAIC sample partition counts: {dict(sample_partition_counts)} "
        f"expected={expected_sample_partition_counts}"
    )

subject_partition_rows = json.loads((root / "outputs/splits/edaic_subject_partitions.json").read_text(encoding="utf-8"))
subject_partition_counts = Counter(row["partition"] for row in subject_partition_rows)
expected_subject_partition_counts = {"train": 163, "val": 56, "test": 56}
if dict(subject_partition_counts) != expected_subject_partition_counts:
    raise SystemExit(
        f"Unexpected EDAIC subject partition counts: {dict(subject_partition_counts)} "
        f"expected={expected_subject_partition_counts}"
    )

partitions_by_subject = defaultdict(set)
for row in subject_partition_rows:
    partitions_by_subject[row["subject_id"]].add(row["partition"])
overlaps = {subject_id: sorted(parts) for subject_id, parts in partitions_by_subject.items() if len(parts) > 1}
if overlaps:
    raise SystemExit(f"EDAIC subject partition overlap detected: {list(overlaps.items())[:5]}")

with (root / "outputs/splits/edaic_join_audit.csv").open("r", encoding="utf-8", newline="") as handle:
    join_audit_rows = list(csv.DictReader(handle))
if len(join_audit_rows) != 3080:
    raise SystemExit(f"Unexpected EDAIC join audit row count: {len(join_audit_rows)}")
missing_join_rows = [
    row["sample_id"]
    for row in join_audit_rows
    if row["audio_found"] != "True" or row["transcript_found"] != "True"
]
if missing_join_rows:
    raise SystemExit(f"EDAIC join audit has missing pairs: {missing_join_rows[:10]}")

if not any("_random_segment_" in row["sample_id"] for row in edaic_manifest_rows):
    raise SystemExit("EDAIC manifest is missing random_segment samples.")
if not any("_segment_" in row["sample_id"] and "_random_segment_" not in row["sample_id"] for row in edaic_manifest_rows):
    raise SystemExit("EDAIC manifest is missing segment samples.")

transcripts_by_subject = defaultdict(set)
for row in edaic_manifest_rows:
    transcripts_by_subject[row["subject_id"]].add(row["transcript"])
bad_transcript_subjects = [
    subject_id for subject_id, transcripts in transcripts_by_subject.items() if len(transcripts) != 1
]
if bad_transcript_subjects:
    raise SystemExit(f"EDAIC subjects have inconsistent repeated full transcripts: {bad_transcript_subjects[:10]}")

edaic_config_path = root / "configs/edaic_audio_text.yaml"
edaic_config = load_yaml_with_overrides(edaic_config_path, [])
edaic_metadata = resolve_metadata_paths(
    json.loads((root / "outputs/splits/edaic_manifest_metadata.json").read_text(encoding="utf-8"))
)
outer_partitions = _resolve_outer_partitions(edaic_config, edaic_metadata, 0)
if len(outer_partitions["outer_train_subject_ids"]) != 219:
    raise SystemExit(
        f"Unexpected EDAIC outer-train subject count from train.py logic: "
        f"{len(outer_partitions['outer_train_subject_ids'])}"
    )
if len(outer_partitions["final_eval_subject_ids"]) != 56:
    raise SystemExit(
        f"Unexpected EDAIC final-eval subject count from train.py logic: "
        f"{len(outer_partitions['final_eval_subject_ids'])}"
    )

print("All expected manifest/split outputs are present.")
print("EDAIC manifest invariants passed.")
PY

echo "[3/4] Validating DepAdapter helper round-trip"
run_python - <<'PY'
import tempfile
import types

import torch
from transformers.modeling_outputs import BaseModelOutput

from src.model.qwen2audio_lora import (
    attach_dep_adapter,
    configure_trainable_audio_modules,
    load_additional_audio_modules,
    resolve_audio_adapter_config,
    save_additional_audio_modules,
    summarize_audio_module_state,
)


class DummyEncoder(torch.nn.Module):
    def __init__(self, d_model: int = 8):
        super().__init__()
        self.config = types.SimpleNamespace(d_model=d_model)

    def forward(
        self,
        input_features,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        hidden = input_features + 1.0
        if return_dict is False:
            return (hidden,)
        return BaseModelOutput(last_hidden_state=hidden, hidden_states=None, attentions=None)


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.audio_tower = DummyEncoder()
        self.multi_modal_projector = torch.nn.Linear(8, 8)
        self.config = types.SimpleNamespace(use_cache=True)


config = {
    "audio_adapter": {
        "enabled": True,
        "adapter_dim": 4,
        "dropout": 0.0,
        "train_projector": True,
    }
}
audio_cfg = resolve_audio_adapter_config(config)
model = DummyModel()
attach_dep_adapter(model, audio_cfg)
configure_trainable_audio_modules(model, audio_cfg)
state = summarize_audio_module_state(model)
assert state["adapter_attached"], state
assert state["adapter_trainable_params"] > 0, state
assert state["projector_trainable_params"] > 0, state

forward_output = model.audio_tower(input_features=torch.zeros(2, 3, 8))
assert forward_output.last_hidden_state.shape == (2, 3, 8)

for parameter in model.audio_tower.audio_adapter.parameters():
    torch.nn.init.constant_(parameter, 0.25)
for parameter in model.multi_modal_projector.parameters():
    torch.nn.init.constant_(parameter, 0.5)

with tempfile.TemporaryDirectory() as tmpdir:
    metadata = save_additional_audio_modules(model, tmpdir, config=config)
    restored = DummyModel()
    loaded = load_additional_audio_modules(restored, tmpdir)
    assert metadata["enabled"] is True, metadata
    assert metadata["train_projector"] is True, metadata
    assert loaded["adapter_state_loaded"] is True, loaded
    assert loaded["projector_state_loaded"] is True, loaded
    for key, value in model.audio_tower.audio_adapter.state_dict().items():
        assert torch.equal(value, restored.audio_tower.audio_adapter.state_dict()[key]), key
    for key, value in model.multi_modal_projector.state_dict().items():
        assert torch.equal(value, restored.multi_modal_projector.state_dict()[key]), key

print("DepAdapter helper round-trip passed.")
PY

echo "[4/4] Printing split source summary"
run_python - <<'PY'
import json
from pathlib import Path
root = Path("/home/emre/Projects/AudioLLM/LLM-Depression/outputs/splits")
for name in ["daic", "edaic", "cmdc", "eatd"]:
    path = root / f"{name}_manifest_metadata.json"
    data = json.loads(path.read_text())
    print(name, json.dumps({k: data[k] for k in data if k in {"dataset", "split_source", "split_source_notes", "manifest_hash"}}, ensure_ascii=False))
PY
