from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest
import yaml

from src.model.gemma4_io import (
    GEMMA4_EVALUATION_VIEW,
    GEMMA4_LORA_TARGET_REGEX,
    GEMMA4_MODEL_REVISION,
    validate_gemma4_config,
)
from src.utils import load_yaml

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "configs/main"
HARMONIZED = ROOT / "configs/experiments/harmonized"
GEMMA_NATIVE_MATRIX = HARMONIZED / "gemma4_standalone_matrix.yaml"
GEMMA_EN_MATRIX = HARMONIZED / "gemma4_english_translation_matrix.yaml"
QWEN_MATRIX = HARMONIZED / "standalone_matrix.yaml"
QWEN_EN_MATRIX = HARMONIZED / "english_translation_matrix.yaml"

GEMMA_NATIVE_CONFIGS = [
    f"{dataset}_{modality}_harmonized_selmacrof1_tf_gemma4_12b.yaml"
    for dataset in ("d3tec", "androids", "cmdc")
    for modality in ("audio_text", "audio_only", "text_only")
] + [
    f"turkish_pos_only_t17_{modality}_harmonized_selmacrof1_tf_qwen3asr_gemma4_12b.yaml"
    for modality in ("audio_text", "audio_only", "text_only")
]
GEMMA_EN_CONFIGS = [
    f"{dataset}_{modality}_harmonized_selmacrof1_tf_en_gemma4_12b.yaml"
    for dataset in ("d3tec", "androids", "cmdc")
    for modality in ("audio_text", "text_only")
] + [
    f"turkish_pos_only_t17_{modality}_harmonized_selmacrof1_tf_qwen3asr_en_gemma4_12b.yaml"
    for modality in ("audio_text", "text_only")
]
GEMMA_NEGATIVE_ONLY_NATIVE_CONFIGS = [
    f"turkish_negative_only_t17_{modality}_harmonized_selmacrof1_tf_qwen3asr_gemma4_12b.yaml"
    for modality in ("audio_text", "audio_only", "text_only")
]
GEMMA_NEGATIVE_ONLY_EN_CONFIGS = [
    f"turkish_negative_only_t17_{modality}_harmonized_selmacrof1_tf_qwen3asr_en_gemma4_12b.yaml"
    for modality in ("audio_text", "text_only")
]
GEMMA_CAMPAIGN_CONFIGS = (
    GEMMA_NATIVE_CONFIGS
    + GEMMA_EN_CONFIGS
    + GEMMA_NEGATIVE_ONLY_NATIVE_CONFIGS
    + GEMMA_NEGATIVE_ONLY_EN_CONFIGS
)

ALLOWED_NEW_KEYS = ("model_backend", "model_revision")
ALLOWED_DIFFERING_KEYS = {
    "model_backend",
    "model_name_or_path",
    "model_revision",
    "output_dirs",
    "lora",
    "evaluation",
}
ALLOWED_OUTPUT_DIRS_DIFF = ("run_root",)
ALLOWED_LORA_DIFF = ("target_modules",)
ALLOWED_EVALUATION_DIFF = ("evaluation_view",)


def _deep_diff(source: dict, target: dict, path: str = "") -> list[str]:
    differences: list[str] = []
    for key in sorted(set(source) | set(target)):
        key_path = f"{path}.{key}" if path else key
        if key not in source:
            differences.append(f"{key_path}: added in Gemma config")
            continue
        if key not in target:
            differences.append(f"{key_path}: missing in Gemma config")
            continue
        if isinstance(source[key], dict) and isinstance(target[key], dict):
            differences.extend(_deep_diff(source[key], target[key], key_path))
        elif source[key] != target[key]:
            differences.append(f"{key_path}: {source[key]!r} != {target[key]!r}")
    return differences


def qwen_base_name(gemma_name: str) -> str:
    return gemma_name.replace("_gemma4_12b.yaml", ".yaml")


def test_exact_gemma_config_sets_exist() -> None:
    all_names = sorted(path.name for path in MAIN.glob("*gemma4_12b.yaml"))
    legacy = sorted(name for name in all_names if name.startswith("turkish_t17_"))
    # Legacy pre-rename canonical Turkish files stay as history; see the rename map.
    assert legacy == sorted(
        f"turkish_t17_{modality}_harmonized_selmacrof1_tf_qwen3asr{variant}_gemma4_12b.yaml"
        for modality, variant in (
            ("audio_text", ""),
            ("audio_only", ""),
            ("text_only", ""),
            ("audio_text", "_en"),
            ("text_only", "_en"),
        )
    )
    current = [name for name in all_names if name not in legacy]
    native = sorted(
        name for name in current if "_en_" not in name and not name.startswith("daic_")
    )
    english = sorted(name for name in current if "_en_" in name)
    daic = sorted(name for name in current if name.startswith("daic_"))
    assert native == sorted(GEMMA_NATIVE_CONFIGS + GEMMA_NEGATIVE_ONLY_NATIVE_CONFIGS)
    assert english == sorted(GEMMA_EN_CONFIGS + GEMMA_NEGATIVE_ONLY_EN_CONFIGS)
    assert len(daic) == 3
    assert len(all_names) == 33


def test_gemma_configs_validate_and_differ_only_by_backend_allowlist() -> None:
    for name in GEMMA_CAMPAIGN_CONFIGS:
        gemma = load_yaml(MAIN / name)
        validate_gemma4_config(gemma)
        assert gemma["model_backend"] == "gemma4"
        assert gemma["model_revision"] == GEMMA4_MODEL_REVISION
        assert gemma["lora"]["target_modules"] == GEMMA4_LORA_TARGET_REGEX
        assert gemma["evaluation"]["evaluation_view"] == GEMMA4_EVALUATION_VIEW
        if "negative_only" in name:
            assert "turkish_qcond_v1_gemma4_negonly" in gemma["output_dirs"]["run_root"]
        else:
            assert "harmonized_v1" in gemma["output_dirs"]["run_root"]
        assert "gemma4" in gemma["output_dirs"]["run_root"]
        base = load_yaml(MAIN / qwen_base_name(name))
        differences = []
        for key in sorted(set(base) | set(gemma)):
            if key not in gemma:
                differences.append(f"{key}: missing in Gemma config")
                continue
            if key not in base:
                if key in ALLOWED_NEW_KEYS:
                    continue
                differences.append(f"{key}: unexpected new key")
                continue
            if key in ALLOWED_DIFFERING_KEYS:
                if key == "output_dirs":
                    assert gemma["output_dirs"]["manifest_dir"] == base["output_dirs"]["manifest_dir"]
                    assert gemma["output_dirs"]["split_dir"] == base["output_dirs"]["split_dir"]
                elif key == "lora":
                    for sub in gemma["lora"]:
                        assert sub in base["lora"] or sub in ALLOWED_LORA_DIFF
                elif key == "evaluation":
                    for sub in gemma["evaluation"]:
                        assert sub in base["evaluation"] or sub in ALLOWED_EVALUATION_DIFF
                continue
            if base[key] != gemma[key]:
                differences.append(f"{key}: {base[key]!r} != {gemma[key]!r}")
        assert not differences, f"{name}: {differences}"


def test_gemma_configs_preserve_scientific_fields() -> None:
    for name in GEMMA_CAMPAIGN_CONFIGS:
        gemma = load_yaml(MAIN / name)
        base = load_yaml(MAIN / qwen_base_name(name))
        for key in ("dataset", "seed", "recipe_id", "labels", "prompt", "split", "data", "training"):
            assert gemma[key] == base[key], f"{name}: {key} changed"
        assert gemma.get("audio_adapter") == base.get("audio_adapter")
        assert gemma["output_dirs"]["manifest_dir"] == base["output_dirs"]["manifest_dir"]
        assert gemma["output_dirs"]["split_dir"] == base["output_dirs"]["split_dir"]


def test_gemma_en_configs_keep_transcript_overlay_and_no_audio_only() -> None:
    for name in GEMMA_EN_CONFIGS + GEMMA_NEGATIVE_ONLY_EN_CONFIGS:
        gemma = load_yaml(MAIN / name)
        assert (gemma.get("transcripts") or {}).get("variant") == "english"
        assert not gemma["data"]["use_audio"] or gemma["data"]["use_text"]
    assert not [p for p in MAIN.glob("*_en_gemma4_12b.yaml") if "audio_only" in p.name]


def test_native_matrix_counts() -> None:
    matrix = yaml.safe_load(GEMMA_NATIVE_MATRIX.read_text(encoding="utf-8"))
    assert matrix["model_backend"] == "gemma4"
    assert matrix["fixed_heads"] == ["logreg_raw"]
    assert matrix["max_epochs"] == 20
    assert matrix["checkpoint_selection"] == "inner_val_macro_f1"
    assert len(matrix["experiments"]) == 12
    folds = sum(len(item["folds"]) for item in matrix["experiments"])
    evals = sum(len(item["folds"]) for item in matrix["experiments"] if item["separate_eval"])
    assert folds == 60
    assert evals == 30
    assert {str(item["config"]).split("/")[-1] for item in matrix["experiments"]} == set(GEMMA_NATIVE_CONFIGS)
    assert all(len(item["folds"]) == 5 for item in matrix["experiments"])


def test_en_matrix_counts_and_no_audio_only() -> None:
    matrix = yaml.safe_load(GEMMA_EN_MATRIX.read_text(encoding="utf-8"))
    assert matrix["model_backend"] == "gemma4"
    assert matrix["fixed_heads"] == ["logreg_raw"]
    assert matrix["optuna"] is False
    assert len(matrix["experiments"]) == 8
    folds = sum(len(item["folds"]) for item in matrix["experiments"])
    evals = sum(len(item["folds"]) for item in matrix["experiments"] if item["separate_eval"])
    assert folds == 40
    assert evals == 20
    assert {str(item["config"]).split("/")[-1] for item in matrix["experiments"]} == set(GEMMA_EN_CONFIGS)


def _run_launcher(matrix: Path, *, extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    import os

    env = {
        **os.environ,
        "MATRIX": str(matrix),
        "RUN_ID": "t2_test",
        "DRY_RUN": "1",
        "GITHUB_ISSUE": "60",
        "GITHUB_PR": "52",
        "PROJECT_ROOT": str(ROOT),
        "PYTHONPATH": str(ROOT),
    }
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        ["bash", str(ROOT / "scripts/submit_harmonized_standalone.sh")],
        capture_output=True,
        text=True,
        env=env,
    )


def test_native_gemma_launcher_dry_run_counts_and_backend_routing() -> None:
    result = _run_launcher(GEMMA_NATIVE_MATRIX)
    assert result.returncode == 0, result.stderr
    output = result.stderr
    # 60 trains + 30 evals + 60 hidden = 150 submitted commands.
    assert output.count("DRY_RUN ") == 150
    assert "tasks=60" in result.stdout
    assert "gemma4_12b_tf5_14_1" in output
    assert "gemma-4-12B-it" in output
    assert "run_gemma4_harmonized_hidden_slurm.sh" in output
    assert "CLASSIFIER_VARIANTS=logreg_raw" in output
    assert "run_qwen_hidden_extract_slurm.sh" not in output


def test_native_gemma_launcher_allows_unlimited_parallelism() -> None:
    ok = _run_launcher(GEMMA_NATIVE_MATRIX, extra_env={"MAX_CONCURRENT_TRAINS": "15", "MAX_CONCURRENT_AUX": "4"})
    assert ok.returncode == 0
    wide = _run_launcher(GEMMA_NATIVE_MATRIX, extra_env={"MAX_CONCURRENT_TRAINS": "60", "MAX_CONCURRENT_AUX": "60"})
    assert wide.returncode == 0
    assert "max_gpus=300" in wide.stdout


def test_en_gemma_launcher_dry_run_counts() -> None:
    import os

    env = {
        **os.environ,
        "MATRIX": str(GEMMA_EN_MATRIX),
        "RUN_ID": "t2_en_test",
        "DRY_RUN": "1",
        "GITHUB_ISSUE": "20",
        "GITHUB_PR": "21",
        "PROJECT_ROOT": str(ROOT),
        "PYTHONPATH": str(ROOT),
    }
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/submit_harmonized_en_standalone.sh")],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    assert result.stderr.count("DRY_RUN ") == 100
    assert "tasks=40" in result.stdout
    assert "gemma4_12b_tf5_14_1" in result.stderr
    assert "gemma-4-12B-it" in result.stderr
    assert "CLASSIFIER_VARIANTS=logreg_raw" in result.stderr


def test_qwen_launchers_keep_their_behavior() -> None:
    import os

    native = _run_launcher(QWEN_MATRIX)
    assert native.returncode == 0, native.stderr
    assert native.stderr.count("DRY_RUN ") == 159
    assert "tasks=63" in native.stdout
    assert "run_qwen_hidden_extract_slurm.sh" in native.stderr
    assert "CLASSIFIER_VARIANTS=logreg_raw:xgb_raw" in native.stderr
    env = {
        **{key: value for key, value in os.environ.items()},
        "MATRIX": str(QWEN_EN_MATRIX),
        "RUN_ID": "t2_qwen_en",
        "DRY_RUN": "1",
        "GITHUB_ISSUE": "20",
        "GITHUB_PR": "21",
        "PROJECT_ROOT": str(ROOT),
        "PYTHONPATH": str(ROOT),
    }
    en = subprocess.run(
        ["bash", str(ROOT / "scripts/submit_harmonized_en_standalone.sh")],
        capture_output=True,
        text=True,
        env=env,
    )
    assert en.returncode == 0, en.stderr
    assert en.stderr.count("DRY_RUN ") == 100
    assert "run_qwen_hidden_extract_slurm.sh" in en.stderr


def test_backend_env_helper_routing() -> None:
    gemma_config = MAIN / "d3tec_audio_text_harmonized_selmacrof1_tf_gemma4_12b.yaml"
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/harmonized_backend_env.sh"), str(gemma_config), str(ROOT)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "MODEL_BACKEND=gemma4" in result.stdout
    assert "gemma4_12b_tf5_14_1" in result.stdout
    assert "gemma-4-12B-it" in result.stdout
    assert "CLASSIFIER_VARIANTS=logreg_raw" in result.stdout
    qwen_config = MAIN / "d3tec_audio_text_harmonized_selmacrof1_tf.yaml"
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/harmonized_backend_env.sh"), str(qwen_config), str(ROOT)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "MODEL_BACKEND=qwen" in result.stdout
    assert "qwen_mn5_rebuilt" in result.stdout
    assert "CLASSIFIER_VARIANTS=logreg_raw:xgb_raw" in result.stdout


def test_config_generator_is_idempotent_and_consistent() -> None:
    import os

    result = subprocess.run(
        ["python", str(ROOT / "scripts/build_gemma4_harmonized_configs.py")],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
    )
    assert result.returncode == 0, result.stderr
    assert "0 written" in result.stdout


class FakeTokenizer:
    def encode(self, text: str) -> list[int]:
        return [ord(char) for char in text]


class FakeGemmaProcessor:
    def __init__(self) -> None:
        self.tokenizer = FakeTokenizer()

    def apply_chat_template(self, messages, **kwargs):
        system = messages[0]["content"]
        user_content = messages[-1]["content"]
        if isinstance(user_content, list):
            text = "".join(item["text"] for item in user_content if item.get("type") == "text")
        else:
            text = str(user_content)
        return f"<bos>system|{system}|user|{text}|"


def test_gemma_preflight_audit_rejects_when_processor_unavailable(
    monkeypatch, tmp_path: Path
) -> None:
    import scripts.prepare_gemma4_harmonized_mn5 as preflight

    def fake_validate(config_path, required_path_prefix=None):
        return {
            "dataset": "d3tec",
            "config": str(config_path),
            "config_sha256": "0" * 64,
            "metadata_path": str(tmp_path / "meta.json"),
            "split_metadata_sha256": "0" * 64,
            "manifest_path": str(tmp_path / "manifest.jsonl"),
            "manifest_file_sha256": "0" * 64,
            "manifest_hash": "0" * 64,
            "rows": 1,
            "subjects": 1,
            "verified_source_paths": 1,
        }

    monkeypatch.setattr(preflight, "validate_manifest", fake_validate)
    audit = preflight.prepare(
        run_id="t2_local",
        build=False,
        required_path_prefix=tmp_path,
        english=False,
        model_path="/nonexistent/gemma-snapshot",
    )
    assert audit["status"] == "failed"
    assert any("Gemma processor" in failure for failure in audit["failures"])
    assert audit["job_scope"] == {
        "train_jobs": 60,
        "separate_eval_jobs": 30,
        "hidden_jobs": 60,
        "total_jobs": 150,
    }
    assert audit["optuna_enabled"] is False
