"""--base-only extraction loads no adapter and records that in the cache identity."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from src.features import extract_qwen_hidden as ex


def _setup(tmp_path: Path, monkeypatch, *, base_only: bool) -> tuple[Path, dict]:
    fold_dir = tmp_path / "fold_0"
    checkpoint = fold_dir / "best_model"
    checkpoint.mkdir(parents=True)
    (checkpoint / "adapter_config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "adapter_model.safetensors").write_bytes(b"adapter")
    run_config = fold_dir / "run_config.yaml"
    run_config.write_text("fold: 0\n", encoding="utf-8")
    split = fold_dir / "split_used.json"
    split.write_text("{}", encoding="utf-8")
    split_metadata = tmp_path / "partitions.json"
    split_metadata.write_text("{}", encoding="utf-8")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("{}\n", encoding="utf-8")
    config = {"dataset": "daic", "model_name_or_path": "base", "data": {}, "evaluation": {}}
    saved = {"fold": 0, "config": config, "manifest_path": str(manifest), "input_modality": "audio_only"}
    calls: dict = {}

    def load_model(name, adapter_path, cfg):
        calls["adapter_path"] = adapter_path
        return type("M", (), {"eval": lambda self: None})()

    patches = {
        "_load_saved_run": lambda c: (saved, config, run_config, split),
        "ensure_hidden_extraction_supported": lambda c, p: None,
        "_resolve_subject_partitions": lambda s, c, p, **k: ({"outer_train": ["1"], "final_eval": ["2"]}, {}),
        "_validate_saved_split": lambda *a, **k: split_metadata,
        "load_manifest_rows": lambda p: [{}],
        "sha256_jsonl_rows": lambda rows: "m",
        "resolve_model_backend": lambda c: ex.MODEL_BACKEND_QWEN3OMNI,
        "_base_snapshot_identity": lambda name: {"snapshot": "s"},
        "_parent_attempt_id": lambda c: "parent",
        "_git_commit": lambda: "g",
        "_emotion_provenance": lambda *a, **k: None,
        "_partition_examples": lambda *a, **k: ({"outer_train": [], "final_eval": []}, None),
        "load_processor": lambda c, cfg: None,
        "prepare_backend_examples": lambda e, c, p: e,
        "load_model_for_inference": load_model,
        "_place_model_for_extraction": lambda m, c: None,
        "_decoder_hidden_size": lambda m, c: 2048,
        "_extract_partition": lambda **k: {"rows": 0},
    }
    for name, value in patches.items():
        monkeypatch.setattr(ex, name, value)
    out = tmp_path / "out"
    argv = ["x", "--checkpoint-dir", str(checkpoint), "--output-dir", str(out)]
    if base_only:
        argv.append("--base-only")
    monkeypatch.setattr(sys, "argv", argv)
    ex.main()
    return checkpoint, {"calls": calls, "metadata": json.loads((out / "extraction_metadata.json").read_text())}


def test_base_only_skips_adapter_and_records_identity(tmp_path, monkeypatch):
    _, result = _setup(tmp_path, monkeypatch, base_only=True)
    meta = result["metadata"]
    assert result["calls"]["adapter_path"] is None
    assert meta["checkpoint_type"] == "base_model"
    assert meta["weights_source"] == "base_model_no_adapter"
    assert meta["adapter_sha256"] is None
    cache = meta["cache_config"]
    assert cache["weights_source"] == "base_model_no_adapter"
    assert cache["parent_checkpoint_role"] == "reference_config_only"
    assert cache["reference_checkpoint_adapter"]["adapter_sha256"]


def test_default_still_loads_adapter(tmp_path, monkeypatch):
    checkpoint, result = _setup(tmp_path, monkeypatch, base_only=False)
    meta = result["metadata"]
    assert result["calls"]["adapter_path"] == checkpoint
    assert meta["checkpoint_type"] == "best_model"
    assert meta["adapter_sha256"]
    assert "weights_source" not in meta["cache_config"]
    assert meta["cache_config"]["parent_checkpoint_role"] == "best_model"


def test_adapter_run_refuses_existing_base_only_cache(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch, base_only=True)
    with pytest.raises(ValueError, match="incompatible"):
        _setup_existing(tmp_path, monkeypatch)


def _setup_existing(tmp_path: Path, monkeypatch) -> None:
    argv = list(sys.argv)
    argv.remove("--base-only")
    monkeypatch.setattr(sys, "argv", argv)
    ex.main()


def _daic_official(tmp_path: Path):
    config = {
        "dataset": "daic",
        "protocol_id": "daic_participant_speech_packed30_v1",
        "split": {"train_partition": "train", "selection_partition": "val", "final_eval_partition": "test"},
    }
    split_payload = {
        "train_subject_ids": ["1", "2"],
        "selection_subject_ids": ["3"],
        "final_eval_subject_ids": ["4"],
    }
    metadata = [
        {"subject_id": "1", "partition": "train"},
        {"subject_id": "2", "partition": "train"},
        {"subject_id": "3", "partition": "val"},
        {"subject_id": "4", "partition": "test"},
    ]
    path = tmp_path / "partitions.json"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    saved = {"split_metadata_path": str(path)}
    return saved, config, split_payload


def test_selection_source_scores_official_dev(tmp_path):
    saved, config, payload = _daic_official(tmp_path)
    parts, prov = ex._resolve_subject_partitions(saved, config, payload, final_eval_source="selection")
    assert parts == {"outer_train": ["1", "2"], "final_eval": ["3"]}
    assert prov["evaluation_protocol"] == "daic_official_train_fit_dev_evaluation"
    assert prov["split_name"] == "val"
    ex._validate_saved_split(saved, config, parts, 0, 1, split_payload=payload, final_eval_source="selection")
    with pytest.raises(ValueError, match="does not match"):
        ex._validate_saved_split(saved, config, parts, 0, 1, split_payload=payload)


def test_saved_source_is_unchanged(tmp_path):
    saved, config, payload = _daic_official(tmp_path)
    parts, prov = ex._resolve_subject_partitions(saved, config, payload)
    assert parts["final_eval"] == ["4"]
    assert prov["evaluation_protocol"] == "daic_official_train_fit_locked_test_evaluation"


def test_selection_source_refused_outside_official_daic(tmp_path):
    saved, config, payload = _daic_official(tmp_path)
    officialdev = json.loads(json.dumps(config))
    officialdev["split"] = {"train_partition": "train", "final_eval_partition": "val"}
    with pytest.raises(ValueError, match="official"):
        ex._resolve_subject_partitions(saved, officialdev, payload, final_eval_source="selection")
    other = dict(config, dataset="cmdc")
    with pytest.raises(ValueError, match="DAIC only"):
        ex._resolve_subject_partitions(saved, other, payload, final_eval_source="selection")
