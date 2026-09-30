"""Qwen3 hidden-extraction and fixed-head support contract tests.

CPU-only and hermetic: the fake checkpoint tree carries a real saved-split
layout, so the extractor's own resolution helpers run against it, but no dataset,
model or cache from a checkout is required.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from src.features.extract_qwen_hidden import (
    CACHE_SCHEMA_VERSION_GEMMA4,
    CACHE_SCHEMA_VERSION_QWEN,
    CACHE_SCHEMA_VERSION_QWEN3,
    _backend_cache_schema,
    _decoder_hidden_size,
    _place_model_for_extraction,
    ensure_hidden_extraction_supported,
)
from src.features.qwen_hidden_collator import Qwen3OmniPromptOnlyExtractionCollator
from src.utils import load_yaml_with_overrides, sha256_file

ROOT = Path(__file__).resolve().parents[1]
QWEN38_POOLED_CONFIG = (
    "configs/main/turkish_pooled_t17_text_only_harmonized_selmacrof1_likelihood_v1"
    "_promptcontext_v1_qwen38_27b.yaml"
)
QWEN3OMNI_POOLED_CONFIG = (
    "configs/main/turkish_pooled_t17_audio_only_harmonized_selmacrof1_likelihood_v1_qwen3asr"
    "_promptcontext_v1_qwen3omni_30b_a3b.yaml"
)
CONDITIONS = ("pos_only_t17", "negative_only_t17")


class _FakeOmniProcessor:
    """Qwen3-Omni processor double returning the audited key set."""

    def __init__(self, *, with_features: bool = True):
        self.feature_extractor = SimpleNamespace(sampling_rate=16_000)
        self.with_features = with_features

    def __call__(self, *, text, return_tensors, padding, audio=None, sampling_rate=None):
        ids = [ord(char) for char in text]
        output = {
            "input_ids": torch.tensor([ids], dtype=torch.long),
            "attention_mask": torch.ones((1, len(ids)), dtype=torch.long),
        }
        if audio is not None and self.with_features:
            output["input_features"] = torch.zeros((1, 128, 2000))
            output["feature_attention_mask"] = torch.ones((1, 2000), dtype=torch.long)
        return output


def _omni_example(*, with_audio: bool = True) -> dict:
    example = {
        "dataset": "turkish",
        "sample_id": "s1",
        "subject_id": "p1",
        "label": 1,
        "partition": "outer_train",
        "fold": 0,
        "prompt_text": "constant prompt",
        "audio_arrays": [torch.zeros(1600)] if with_audio else [],
    }
    return example


def _write_fake_qwen38_checkpoint(tmp_path: Path) -> dict:
    """Write a minimal but contract-shaped pooled Qwen3.8 checkpoint tree."""
    base = tmp_path / "base_model"
    base.mkdir()
    (base / "config.json").write_text("{}", encoding="utf-8")
    fold = tmp_path / "runs" / "fake_run" / "fold_0"
    best = fold / "best_model"
    best.mkdir(parents=True)
    (best / "adapter_config.json").write_text("{}", encoding="utf-8")
    (best / "adapter_model.safetensors").write_text("adapter", encoding="utf-8")
    (fold / "logs").mkdir()
    split = {
        "train_subject_ids": ["t1", "t2", "t3", "t4"],
        "selection_subject_ids": ["v1", "v2", "v3", "v4"],
        "final_eval_subject_ids": [],
    }
    (fold / "logs" / "split_used.json").write_text(json.dumps(split), encoding="utf-8")
    rows = []
    for subject_id, label in (("t1", 1), ("t2", 0), ("t3", 1), ("t4", 0), ("v1", 1), ("v2", 0), ("v3", 1), ("v4", 0)):
        for condition in CONDITIONS:
            rows.append(
                {
                    "dataset": "turkish",
                    "dataset_variant": condition,
                    "sample_id": f"{subject_id}-{condition}",
                    "subject_id": subject_id,
                    "label": label,
                    "score": 17.0,
                    "threshold": 17.0,
                    "transcript": "synthetic",
                    "audio_path": str(tmp_path / "audio.wav"),
                    "audio_paths": [str(tmp_path / "audio.wav")],
                }
            )
    (tmp_path / "audio.wav").write_bytes(b"RIFF0000WAVE")
    manifest = tmp_path / "turkish_manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    saved = {
        "config": {
            "dataset": "turkish",
            "dataset_variant": "pooled_t17",
            "model_backend": "qwen38",
            "split": {"cv_protocol": "train_val"},
            "evaluation": {"evaluation_view": "harmonized_all_windows_full_coverage"},
            "resources": {"eval_gpus_per_node": 1},
            "model_name_or_path": str(base),
        },
        "resolved_model_name_or_path": str(base),
        "fold": 0,
        "manifest_path": str(manifest),
    }
    (fold / "run_config.yaml").write_text(yaml.safe_dump(saved), encoding="utf-8")
    return {"fold": fold, "best": best, "manifest": manifest}


def _env(**overrides) -> dict:
    env = dict(os.environ)
    env["PATH"] = f"{Path(sys.executable).parent}:{env.get('PATH', '')}"
    env.update({key: str(value) for key, value in overrides.items()})
    return env


class TestHiddenSizes:
    def test_qwen3_decoder_hidden_sizes(self):
        qwen38_model = SimpleNamespace(
            config=SimpleNamespace(text_config=SimpleNamespace(hidden_size=5120)),
            base_model=None,
        )
        omni_model = SimpleNamespace(
            config=SimpleNamespace(
                thinker_config=SimpleNamespace(text_config=SimpleNamespace(hidden_size=2048))
            ),
            base_model=None,
        )
        assert _decoder_hidden_size(qwen38_model, {"model_backend": "qwen38"}) == 5120
        assert _decoder_hidden_size(omni_model, {"model_backend": "qwen3omni"}) == 2048
        with pytest.raises(ValueError, match="Unexpected decoder hidden size"):
            _decoder_hidden_size(
                SimpleNamespace(config=SimpleNamespace(hidden_size=4096), base_model=None),
                {"model_backend": "qwen38"},
            )

    def test_backend_cache_schema_isolates_qwen3_caches(self):
        assert _backend_cache_schema({"model_backend": "qwen38"}) == CACHE_SCHEMA_VERSION_QWEN3
        assert _backend_cache_schema({"model_backend": "qwen3omni"}) == CACHE_SCHEMA_VERSION_QWEN3
        assert _backend_cache_schema({"model_backend": "qwen2audio"}) == CACHE_SCHEMA_VERSION_QWEN
        assert _backend_cache_schema({}) == CACHE_SCHEMA_VERSION_QWEN
        assert _backend_cache_schema({"model_backend": "gemma4"}) == CACHE_SCHEMA_VERSION_GEMMA4


class TestGuard:
    def test_guard_accepts_canonical_pooled_configs(self, tmp_path: Path):
        for rel in (QWEN38_POOLED_CONFIG, QWEN3OMNI_POOLED_CONFIG):
            ensure_hidden_extraction_supported(
                load_yaml_with_overrides(ROOT / rel, []), tmp_path / "run_config.yaml"
            )

    def test_guard_refuses_broken_qwen3_contracts(self, tmp_path: Path):
        with pytest.raises(ValueError, match="Qwen3.8 hidden extraction refused"):
            ensure_hidden_extraction_supported(
                {"model_backend": "qwen38"}, tmp_path / "run_config.yaml"
            )
        with pytest.raises(ValueError, match="Qwen3-Omni hidden extraction refused"):
            ensure_hidden_extraction_supported(
                {"model_backend": "qwen3omni"}, tmp_path / "run_config.yaml"
            )


class TestDevicePlacement:
    def test_declared_device_map_is_kept(self):
        model = SimpleNamespace(
            hf_device_map={"a": "cuda:0", "b": "cuda:1"},
            calls=[],
        )
        model.to = lambda **kwargs: model.calls.append(kwargs)  # type: ignore[attr-defined]
        _place_model_for_extraction(model, {"training": {"bf16": True}})
        assert model.calls == []

    def test_single_device_model_moves_with_training_dtype(self):
        model = SimpleNamespace(calls=[])
        model.to = lambda **kwargs: model.calls.append(kwargs)  # type: ignore[attr-defined]
        _place_model_for_extraction(model, {"training": {"bf16": True}})
        assert len(model.calls) == 1
        assert "device" in model.calls[0]


class TestOmniCollator:
    def test_audio_is_mandatory(self):
        collator = Qwen3OmniPromptOnlyExtractionCollator(_FakeOmniProcessor())
        with pytest.raises(ValueError, match="requires audio input"):
            collator([_omni_example(with_audio=False)])

    def test_audited_key_set_is_enforced(self):
        collator = Qwen3OmniPromptOnlyExtractionCollator(
            _FakeOmniProcessor(with_features=False)
        )
        with pytest.raises(AssertionError, match="missing required keys"):
            collator([_omni_example()])

    def test_full_processor_output_keeps_labels_external(self):
        collator = Qwen3OmniPromptOnlyExtractionCollator(_FakeOmniProcessor())
        model_inputs, metadata = collator([_omni_example()])
        assert set(model_inputs) == {
            "input_ids",
            "attention_mask",
            "input_features",
            "feature_attention_mask",
        }
        assert "labels" not in model_inputs
        assert metadata[0]["sample_id"] == "s1"
        assert metadata[0]["prompt_text"] == "constant prompt"


class TestSelectionBuilder:
    def test_preview_write_and_verify(self, tmp_path: Path):
        fixture = _write_fake_qwen38_checkpoint(tmp_path)
        output = tmp_path / "selection.json"
        base = [
            sys.executable,
            str(ROOT / "scripts/build_qwen3_smoke_subject_selection.py"),
            "--checkpoint-dir", str(fixture["best"]),
            "--train-per-label", "2",
            "--eval-per-label", "2",
        ]
        preview = subprocess.run(
            [*base, "--preview", "--print-json"], cwd=ROOT, text=True, capture_output=True
        )
        assert preview.returncode == 0, preview.stderr
        payload = json.loads(preview.stdout)
        assert payload["outer_train"] == ["t1", "t2", "t3", "t4"]
        assert payload["final_eval"] == ["v1", "v2", "v3", "v4"]
        assert not output.exists()
        written = subprocess.run(
            [*base, "--output", str(output)], cwd=ROOT, text=True, capture_output=True
        )
        assert written.returncode == 0, written.stderr
        assert json.loads(output.read_text(encoding="utf-8")) == payload
        verified = subprocess.run(
            [*base, "--output", str(output), "--verify"], cwd=ROOT, text=True, capture_output=True
        )
        assert verified.returncode == 0, verified.stderr

    def test_per_label_shortfall_fails_closed(self, tmp_path: Path):
        fixture = _write_fake_qwen38_checkpoint(tmp_path)
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/build_qwen3_smoke_subject_selection.py"),
                "--checkpoint-dir", str(fixture["best"]),
                "--train-per-label", "3",
                "--preview",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        assert result.returncode != 0
        assert "required" in (result.stderr + result.stdout)


class TestSmokeSubmitter:
    def _matrix(self, tmp_path: Path, fixture: dict, *, gpus: int = 1) -> Path:
        matrix = {
            "schema_version": "audiollm.qwen3_hidden_smoke.v1",
            "jobs": [
                {
                    "name": "fake_text_smoke",
                    "backend": "qwen38",
                    "condition": "text_only",
                    "config": QWEN38_POOLED_CONFIG,
                    "checkpoint_dir": str(fixture["best"]),
                    "gpus": gpus,
                    "train_per_label": 2,
                    "eval_per_label": 2,
                }
            ],
        }
        path = tmp_path / "smoke_matrix.yaml"
        path.write_text(yaml.safe_dump(matrix, sort_keys=False), encoding="utf-8")
        return path

    def test_dry_run_prints_the_contract(self, tmp_path: Path):
        fixture = _write_fake_qwen38_checkpoint(tmp_path)
        matrix = self._matrix(tmp_path, fixture)
        result = subprocess.run(
            ["bash", str(ROOT / "scripts/submit_qwen3_hidden_smoke.sh")],
            cwd=ROOT,
            env=_env(
                PROJECT_ROOT=ROOT,
                MATRIX=matrix,
                EVIDENCE_ROOT=tmp_path / "evidence",
                DRY_RUN="1",
            ),
            text=True,
            capture_output=True,
        )
        assert result.returncode == 0, result.stderr
        for expected in (
            "--- job fake_text_smoke ---",
            "backend: qwen38",
            "condition: text_only",
            "gpus: 1 (recorded eval_gpus_per_node)",
            "worker env:",
            "cache:",
            "fit output:",
            "subject selection:",
            "jobs: 2 (1 extraction + 1 classifier)",
        ):
            assert expected in result.stdout, expected
        assert sum(line.startswith("DRY_RUN sbatch") for line in result.stderr.splitlines()) == 2
        assert not (tmp_path / "evidence").exists()

    def test_submit_requires_explicit_opt_in(self, tmp_path: Path):
        fixture = _write_fake_qwen38_checkpoint(tmp_path)
        matrix = self._matrix(tmp_path, fixture)
        result = subprocess.run(
            ["bash", str(ROOT / "scripts/submit_qwen3_hidden_smoke.sh")],
            cwd=ROOT,
            env=_env(
                PROJECT_ROOT=ROOT,
                MATRIX=matrix,
                EVIDENCE_ROOT=tmp_path / "evidence",
                DRY_RUN="0",
            ),
            text=True,
            capture_output=True,
        )
        assert result.returncode == 2
        assert "explicit-only" in result.stderr

    def test_gpu_shape_mismatch_is_refused(self, tmp_path: Path):
        fixture = _write_fake_qwen38_checkpoint(tmp_path)
        matrix = self._matrix(tmp_path, fixture, gpus=2)
        result = subprocess.run(
            ["bash", str(ROOT / "scripts/submit_qwen3_hidden_smoke.sh")],
            cwd=ROOT,
            env=_env(
                PROJECT_ROOT=ROOT,
                MATRIX=matrix,
                EVIDENCE_ROOT=tmp_path / "evidence",
                DRY_RUN="1",
            ),
            text=True,
            capture_output=True,
        )
        assert result.returncode == 5
        assert "recorded eval_gpus_per_node" in result.stderr


class TestRetryHelperGate:
    def _cells(self, tmp_path: Path) -> Path:
        path = tmp_path / "cells.tsv"
        path.write_text("turkish\ttext_only\t0\t0\t\t\tFAILED\n", encoding="utf-8")
        return path

    def _matrix(self, tmp_path: Path, *, fixed_heads: list[str]) -> Path:
        matrix = {
            "name": "fake_retry_matrix",
            "fixed_heads": fixed_heads,
            "experiments": [
                {"config": QWEN38_POOLED_CONFIG, "folds": [0], "separate_eval": False}
            ],
        }
        path = tmp_path / "matrix.yaml"
        path.write_text(yaml.safe_dump(matrix, sort_keys=False), encoding="utf-8")
        return path

    def _run(self, tmp_path: Path, matrix: Path) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["bash", str(ROOT / "scripts/submit_harmonized_standalone_retry.sh")],
            cwd=ROOT,
            env=_env(
                PROJECT_ROOT=ROOT,
                MATRIX=matrix,
                CELLS=self._cells(tmp_path),
                RUN_ID="fake_retry",
                GITHUB_ISSUE="1",
                GITHUB_PR="1",
                DRY_RUN="1",
                SUBMISSIONS_ROOT=tmp_path / "submissions",
                CONTEXTS_ROOT=tmp_path / "contexts",
            ),
            text=True,
            capture_output=True,
        )

    def test_declared_heads_with_qwen3_cell_are_refused(self, tmp_path: Path):
        result = self._run(tmp_path, self._matrix(tmp_path, fixed_heads=["logreg_raw", "xgb_raw"]))
        assert result.returncode == 5
        assert "Qwen3 fixed heads are explicit-only" in result.stderr
        assert "refusing to submit" in result.stderr

    def test_undeclared_heads_skip_head_dispatch(self, tmp_path: Path):
        result = self._run(tmp_path, self._matrix(tmp_path, fixed_heads=[]))
        assert result.returncode == 0, result.stderr
        assert "Skipping fixed-head dispatch" in result.stderr
        assert "hrh-" not in result.stderr
