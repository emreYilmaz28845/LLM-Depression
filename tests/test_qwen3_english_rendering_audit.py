"""Regression checks for the English render/notice audit and seed independence.

The audit itself is weight-free and CPU only: it verifies the eight generated
English cells against their native counterparts (allowlisted derivation, exact
translation-notice sentences, single notice before the transcript block, native
counterpart without a notice) and records current config/prompt hashes.

The second check documents the fixed-split invariant: overriding the top-level
training seed must not change ``split.seed`` or the manifest build signature,
because the split lives on its own seed and the manifest builder does not read
the top-level seed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_qwen3_english_configs as generator  # noqa: E402
from src.data.build_manifest import manifest_build_signature  # noqa: E402
from src.utils import load_yaml_with_overrides  # noqa: E402
from tools import audit_qwen3_english_rendering as audit_tool  # noqa: E402

MATRIX = ROOT / "configs/experiments/harmonized/english_translation_matrix.yaml"


def test_english_render_audit_passes(tmp_path: Path) -> None:
    payload = audit_tool.run_audit(tmp_dir=tmp_path / "audio")
    assert payload["status"] == "passed", payload["failures"]
    assert len(payload["cells"]) == 8
    for cell in payload["cells"]:
        assert cell["notice"]["version"] == "translation_notice_v1"
        assert cell["notice"]["occurrences_in_rendered_prompt"] == 1
        assert cell["notice"]["offset_before_transcript_block"]
        assert cell["native"]["notice"] is None
        assert not cell["native"]["rendered_carries_notice"]
        assert cell["train_equals_eval_prompt"]
        assert not cell["failures"], cell["failures"]


def test_notice_sentences_are_modality_exact(tmp_path: Path) -> None:
    payload = audit_tool.run_audit(tmp_dir=tmp_path / "audio")
    by_id = {cell["cell_id"]: cell for cell in payload["cells"]}
    text_only = by_id["d3tec_text_only"]["notice"]["text"]
    audio_text = by_id["d3tec_audio_text"]["notice"]["text"]
    assert "English translation" in text_only
    assert "audio" not in text_only.lower()
    assert "audio remains in the original language" in audio_text


def test_audit_cells_match_the_english_matrix() -> None:
    matrix = yaml.safe_load(MATRIX.read_text(encoding="utf-8"))
    selected = {item["config"] for item in matrix["experiments"]}
    generated = {f"configs/main/{cell[2]}" for cell in generator.CELLS}
    assert selected == generated


def test_top_level_seed_does_not_change_the_split_contract() -> None:
    for cell in generator.CELLS:
        config_path = ROOT / "configs/main" / cell[2]
        base = load_yaml_with_overrides(config_path, [])
        base_signature = manifest_build_signature(base)
        assert base["split"]["seed"] == 1337
        for seed in (7, 2024):
            config = load_yaml_with_overrides(config_path, ["--set", f"seed={seed}"])
            assert config["seed"] == seed
            assert config["split"]["seed"] == 1337
            assert manifest_build_signature(config) == base_signature


def test_main_writes_a_passing_audit(tmp_path: Path) -> None:
    output = tmp_path / "audit.json"
    rc = audit_tool.main(
        [
            "--output",
            str(output),
            "--tmp-dir",
            str(tmp_path / "audio"),
        ]
    )
    assert rc == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "passed"
