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

echo "[2/4] Verifying expected audit/split files"
run_python - <<'PY'
from pathlib import Path
root = Path("/home/emre/Projects/AudioLLM/LLM-Depression")
required = [
    root / "outputs/manifests/daic_manifest.jsonl",
    root / "outputs/manifests/cmdc_manifest.jsonl",
    root / "outputs/manifests/eatd_manifest.jsonl",
    root / "outputs/splits/daic_join_audit.csv",
    root / "outputs/splits/cmdc_fold_report.json",
    root / "outputs/splits/eatd_folds.json",
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit(f"Missing expected outputs: {missing}")
print("All expected manifest/split outputs are present.")
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
for name in ["daic", "cmdc", "eatd"]:
    path = root / f"{name}_manifest_metadata.json"
    data = json.loads(path.read_text())
    print(name, json.dumps({k: data[k] for k in data if k in {"dataset", "split_source", "split_source_notes", "manifest_hash"}}, ensure_ascii=False))
PY
