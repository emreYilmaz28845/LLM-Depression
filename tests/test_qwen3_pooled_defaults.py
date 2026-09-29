"""Qwen3 Turkish pooled-defaults contract tests.

These tests pin the preparation task's public contract: the default matrices,
the five Turkish standalone cells, the four merged contracts, the deterministic
generators, the pooled manifest source contract, the aggregation rules and the
readiness guards. They are hermetic: no dataset, model, manifest or cache from a
checkout is required.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from scripts import build_canonical_backend_configs as canonical
from scripts import build_qwen3_english_configs as english_configs
from scripts import build_qwen3_pooled_merged_configs as merged_configs
from scripts import build_turkish_pooled_manifest as pooled_builder
from scripts.prepare_harmonized_en_mn5 import (
    BUILD_CONFIGS as EN_BUILD_CONFIGS,
    PREBUILT_CONFIGS as EN_PREBUILT_CONFIGS,
)
from scripts.prepare_harmonized_mn5 import (
    COMPONENT_CONFIGS as NATIVE_COMPONENT_CONFIGS,
    MERGED_CONFIGS as NATIVE_MERGED_CONFIGS,
)
from src.aggregate import INVALID_PREDICTION, TURKISH_POOLED_TEXT_PAIR_POLICY, _pair_prediction
from src.data.prompt_context import resolve_question_context_sentences
from src.features.extract_qwen_hidden import ensure_hidden_extraction_supported
from src.utils import sha256_jsonl_rows
from tools import audit_qwen3_pooled_inputs as inputs_audit
from tools import qwen3_pooled_defaults as selection


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "configs/main"

# The three pooled native Qwen3 configs are the migration's native source
# contract: this task must not rewrite them.
POOLED_NATIVE_CONFIG_SHA256 = {
    "turkish_pooled_t17_text_only_harmonized_selmacrof1_likelihood_v1_promptcontext_v1_qwen38_27b.yaml":
        "bd945a3ceb3d8e70f8c5894a4702aa87722dc3074774f6a18faa9f4d0b96c8ab",
    "turkish_pooled_t17_audio_only_harmonized_selmacrof1_likelihood_v1_qwen3asr_promptcontext_v1_qwen3omni_30b_a3b.yaml":
        "e22d52df3f58408d5591f07c86a7ebccd5bb39821c8d17d2f39af9b29f4d1200",
    "turkish_pooled_t17_audio_text_harmonized_selmacrof1_likelihood_v1_qwen3asr_promptcontext_v1_qwen3omni_30b_a3b.yaml":
        "d3b89e740ff28dfd15fbad2021cc00859cf25fa5ea91a8d08e358e204c1e62a4",
}


def load_config(rel: str) -> dict:
    return yaml.safe_load((ROOT / rel).read_text(encoding="utf-8"))


def test_selection_validator_passes() -> None:
    failures = selection.validate_all()
    assert set(failures) == {
        "native_matrix",
        "english_matrix",
        "turkish_cells",
        "merged_contracts",
        "head_deferral",
    }
    assert failures == {section: [] for section in failures}


def test_default_matrices_select_qwen3_and_pooled_turkish() -> None:
    native = yaml.safe_load(selection.NATIVE_MATRIX.read_text(encoding="utf-8"))
    turkish = [item["config"] for item in native["experiments"] if "turkish" in item["config"]]
    assert sorted(turkish) == sorted(selection.POOLED_TURKISH_NATIVE.values())
    assert not any("pos_only" in item["config"] for item in native["experiments"])
    english = yaml.safe_load(selection.ENGLISH_MATRIX.read_text(encoding="utf-8"))
    assert {item["config"] for item in english["experiments"]} == {
        f"configs/main/{cell[2]}" for cell in english_configs.CELLS
    }
    assert english["fixed_heads"] == []
    assert native["fixed_heads"] == []


def test_pooled_qwen3_migration_contract_is_pinned() -> None:
    for name, digest in POOLED_NATIVE_CONFIG_SHA256.items():
        payload = (MAIN / name).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == digest, name


def test_generators_are_deterministic_and_idempotent() -> None:
    generated = sorted(
        [f"configs/main/{cell[2]}" for cell in english_configs.CELLS]
        + [f"configs/experiments/merged/{cell[2]}" for cell in merged_configs.CELLS]
        + ["configs/experiments/harmonized/english_translation_matrix.yaml"]
    )
    before = {rel: hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() for rel in generated}
    for script in (
        "scripts/build_canonical_backend_configs.py",
        "scripts/build_qwen3_english_configs.py",
        "scripts/build_qwen3_pooled_merged_configs.py",
    ):
        check = subprocess.run(
            [sys.executable, script, "--check"], cwd=ROOT, capture_output=True, text=True
        )
        assert check.returncode == 0, (script, check.stderr)
        write = subprocess.run(
            [sys.executable, script], cwd=ROOT, capture_output=True, text=True
        )
        assert write.returncode == 0, (script, write.stderr)
    after = {rel: hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() for rel in generated}
    assert before == after


def test_english_generator_fails_closed_on_invalid_cells() -> None:
    source = load_config(f"configs/main/{english_configs.CELLS[0][1]}")
    pooled_source = load_config(selection.POOLED_TURKISH_NATIVE["text_only"])
    _, source_name, _, _, _, modality, folds, separate_eval, _ = english_configs.CELLS[0]
    # A pooled source with a translation cache and a plain source without one.
    pooled_cell = ("bad_pooled", selection.POOLED_TURKISH_NATIVE["text_only"].split("/")[-1],
                   "bad_pooled.yaml", "d3tec", "d3tec", modality, folds, separate_eval, "d3tec")
    with pytest.raises(english_configs.GenerationError):
        english_configs.derive(pooled_source, pooled_cell)
    plain_cell = ("bad_plain", source_name, "bad_plain.yaml", "d3tec", "d3tec", modality,
                  folds, separate_eval, None)
    with pytest.raises(english_configs.GenerationError):
        english_configs.derive(source, plain_cell)


def test_merged_generator_fails_closed_on_mixed_backends(monkeypatch) -> None:
    cell = merged_configs.CELLS[0]
    source = yaml.safe_load(
        (ROOT / "configs/experiments/merged" / cell[1]).read_text(encoding="utf-8")
    )
    mixed = merged_configs.native_components(cell[3])
    mixed[0]["config"] = "configs/main/daic_audio_text_harmonized_selmacrof1_likelihood_v1.yaml"
    monkeypatch.setattr(merged_configs, "native_components", lambda modality: mixed)
    with pytest.raises(merged_configs.GenerationError, match="mixed merged backends"):
        merged_configs.derive(source, cell)


def test_canonical_generator_verifies_pooled_defaults_fail_closed(tmp_path: Path, monkeypatch) -> None:
    broken_main = tmp_path / "main"
    broken_main.mkdir()
    for name in canonical.POOLED_TURKISH_DEFAULTS.values():
        (broken_main / name).write_text("dataset: turkish\n", encoding="utf-8")
    monkeypatch.setattr(canonical, "MAIN", broken_main)
    failures = canonical.verify_pooled_turkish_defaults()
    assert failures
    assert any("prompt.version" in failure for failure in failures)
    assert any("manifest_policy" in failure for failure in failures)


def test_prepare_routes_are_pooled_aware() -> None:
    assert "configs/main/turkish_pooled_t17_audio_text_harmonized_selmacrof1_likelihood_v1_qwen3asr_promptcontext_v1_qwen3omni_30b_a3b.yaml" in NATIVE_COMPONENT_CONFIGS
    assert not any("pos_only" in rel for rel in NATIVE_COMPONENT_CONFIGS)
    assert set(NATIVE_MERGED_CONFIGS) == {
        f"configs/experiments/merged/{cell[2]}" for cell in merged_configs.CELLS
    }
    assert len(EN_BUILD_CONFIGS) == 3
    assert EN_PREBUILT_CONFIGS == (
        "configs/main/turkish_pooled_t17_audio_text_harmonized_selmacrof1_likelihood_v1_qwen3asr_promptcontext_v1_en_qwen3omni_30b_a3b.yaml",
    )


def test_hidden_extraction_guard_refuses_qwen3_backends(tmp_path: Path) -> None:
    for backend in ("qwen38", "qwen3omni"):
        with pytest.raises(ValueError, match="Qwen3 hidden-extraction prerequisite incomplete"):
            ensure_hidden_extraction_supported({"model_backend": backend}, tmp_path / "run_config.yaml")
    for backend in ("text", "qwen2audio", "gemma4"):
        ensure_hidden_extraction_supported({"model_backend": backend}, tmp_path / "run_config.yaml")
    ensure_hidden_extraction_supported({}, tmp_path / "run_config.yaml")


def test_text_pair_rule_and_question_context() -> None:
    assert TURKISH_POOLED_TEXT_PAIR_POLICY == "turkish_pooled_text_pair_mean_margin_strict_v1"
    assert _pair_prediction(0.25) == 1
    assert _pair_prediction(-0.25) == 0
    assert _pair_prediction(0.0) == INVALID_PREDICTION
    assert _pair_prediction(0.25, decoded_valid=False) == INVALID_PREDICTION
    sentences_by_config = []
    for rel in sorted(selection.POOLED_TURKISH_NATIVE.values()) + sorted(selection.POOLED_TURKISH_ENGLISH.values()):
        sentences = resolve_question_context_sentences(load_config(rel))
        sentences_by_config.append(sentences)
        assert set(sentences) == {"pos_only_t17", "negative_only_t17"}
        assert sentences["pos_only_t17"] != sentences["negative_only_t17"]
        for text in sentences.values():
            # Condition drives the sentence, never the gold label or the score.
            # ("depression label" appears only inside the clarifying negation.)
            for leaked in ("Depressed", "Non-depressed", "BDI", "17", "score"):
                assert leaked not in text, (rel, leaked)
    assert all(item == sentences_by_config[0] for item in sentences_by_config)


def test_selection_map_records_cells_contracts_and_readiness(tmp_path: Path) -> None:
    target = tmp_path / "selection_map.json"
    exit_code = selection.main(["--emit", str(target)])
    assert exit_code == 0
    payload = json.loads(target.read_text(encoding="utf-8"))
    turkish = [cell for cell in payload["cells"] if cell["family"] == "turkish_pooled"]
    assert len(turkish) == 5
    assert len(payload["merged"]) == 4
    assert payload["readiness"]
    assert any("deferred" in item["state"] for item in payload["readiness"])
    assert all(cell["backend"] in {"qwen38", "qwen3omni"} for cell in payload["cells"])


# --- pooled manifest source contract -----------------------------------------


def _write_pooled_sources(root: Path, *, mutate=None) -> dict[str, Path]:
    """Write a complete synthetic four-source pooled input set.

    Counts, subjects, labels, threshold and folds follow the recorded contract
    (1051 positive rows, 1170 negative rows, 120 subjects, 37/83 labels,
    threshold 17, five folds at seed 1337); every row points at one dummy audio
    file so the builder's existence check passes.
    """
    audio = root / "audio.wav"
    root.mkdir(parents=True, exist_ok=True)
    audio.write_bytes(b"RIFF0000WAVE")
    subjects = [f"s{index:03d}" for index in range(120)]
    labels = {subject: (1 if index < 83 else 0) for index, subject in enumerate(subjects)}
    scores = {subject: (17.0 if labels[subject] == 1 else 5.0) for subject in subjects}
    counts = {
        "pos_only_t17": {subject: (9 if index < 91 else 8) for index, subject in enumerate(subjects)},
        "negative_only_t17": {subject: (10 if index < 90 else 9) for index, subject in enumerate(subjects)},
    }

    def rows(condition: str, language: str) -> list[dict]:
        result = []
        for subject in subjects:
            for index in range(counts[condition][subject]):
                text = f"{language} {subject} {condition} {index}"
                row = {
                    "dataset": "turkish",
                    "dataset_variant": condition,
                    "sample_id": f"{subject}-{condition}-{index}",
                    "subject_id": subject,
                    "label": labels[subject],
                    "score": scores[subject],
                    "threshold": 17.0,
                    "transcript": text,
                    "audio_path": str(audio),
                    "audio_paths": [str(audio)],
                }
                if language == "english":
                    row["language"] = "en"
                    row["transcript_variant"] = "english"
                    row["translation_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
                result.append(row)
        return result

    folds = {
        str(fold): {
            "final_eval_subject_ids": subjects[fold * 24:(fold + 1) * 24],
            "outer_train_subject_ids": subjects[: fold * 24] + subjects[(fold + 1) * 24:],
        }
        for fold in range(5)
    }
    sources: dict[str, Path] = {}
    for condition in ("pos", "neg"):
        full = "pos_only_t17" if condition == "pos" else "negative_only_t17"
        for language in ("native", "english"):
            manifest = rows(full, language)
            split_payload = copy.deepcopy(folds)
            if mutate is not None:
                mutate(manifest, split_payload, condition=condition, language=language)
            manifest_dir = root / "manifests" / f"{condition}_{language}"
            split_dir = root / "splits" / f"{condition}_{language}"
            manifest_dir.mkdir(parents=True, exist_ok=True)
            split_dir.mkdir(parents=True, exist_ok=True)
            manifest_path = manifest_dir / "turkish_manifest.jsonl"
            manifest_path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest),
                encoding="utf-8",
            )
            folds_path = split_dir / "turkish_folds.json"
            folds_path.write_text(json.dumps(split_payload), encoding="utf-8")
            sources[f"{condition}_{language}_manifest"] = manifest_path
            sources[f"{condition}_{language}_split"] = folds_path
    return sources


def _builder_argv(sources: dict[str, Path], output_root: Path) -> list[str]:
    return [
        "--positive-native-manifest", str(sources["pos_native_manifest"]),
        "--positive-native-split", str(sources["pos_native_split"]),
        "--negative-native-manifest", str(sources["neg_native_manifest"]),
        "--negative-native-split", str(sources["neg_native_split"]),
        "--positive-english-manifest", str(sources["pos_english_manifest"]),
        "--positive-english-split", str(sources["pos_english_split"]),
        "--negative-english-manifest", str(sources["neg_english_manifest"]),
        "--negative-english-split", str(sources["neg_english_split"]),
        "--native-output-dir", str(output_root / "manifests" / "turkish"),
        "--english-output-dir", str(output_root / "manifests_en" / "turkish"),
        "--native-split-output-dir", str(output_root / "splits" / "turkish"),
        "--english-split-output-dir", str(output_root / "splits_en" / "turkish"),
        "--audit-output", str(output_root / "preflight" / "pooled_manifest_audit.json"),
    ]


def test_pooled_manifest_builder_accepts_the_recorded_contract(tmp_path: Path) -> None:
    sources = _write_pooled_sources(tmp_path / "sources")
    output_root = tmp_path / "runtime"
    argv = _builder_argv(sources, output_root)
    assert pooled_builder.main(argv) == 0
    native_rows = [
        json.loads(line)
        for line in (output_root / "manifests/turkish/turkish_manifest.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    english_rows = [
        json.loads(line)
        for line in (output_root / "manifests_en/turkish/turkish_manifest.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert len(native_rows) == 2221
    assert len(english_rows) == 2221
    assert len({row["subject_id"] for row in native_rows}) == 120
    assert {row["dataset_variant"] for row in native_rows} == {"pos_only_t17", "negative_only_t17"}
    assert native_rows != english_rows
    metadata = json.loads(
        (output_root / "splits/turkish/turkish_manifest_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["manifest_hash"] == sha256_jsonl_rows(native_rows)
    assert metadata["split_source"] == "reused_canonical_source_folds"
    assert metadata["translation_pairing"] is None
    english_metadata = json.loads(
        (output_root / "splits_en/turkish/turkish_manifest_metadata.json").read_text(encoding="utf-8")
    )
    assert english_metadata["transcript_variant"] == "english"
    assert english_metadata["translation_pairing"]["paired_rows"] == 2221
    audit = json.loads(
        (output_root / "preflight/pooled_manifest_audit.json").read_text(encoding="utf-8")
    )
    assert audit["status"] == "passed"
    assert audit["checks"]["native_english_sample_pairing"] is True
    assert audit["checks"]["four_split_mappings_identical"] is True


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda m, s, *, condition, language: _drop_translation(m, condition, language), "translation hash missing"),
        (lambda m, s, *, condition, language: _break_translation(m, condition, language), "translation hash mismatch"),
        (lambda m, s, *, condition, language: _duplicate_sample_id(m, condition, language), "duplicated across pooled source manifests"),
        (lambda m, s, *, condition, language: _move_subject_across_folds(m, s), "subjects occur in multiple folds"),
        (lambda m, s, *, condition, language: _non_turkish_dataset(m, condition, language), "non-Turkish row"),
        (lambda m, s, *, condition, language: _wrong_condition_tag(m, condition, language), "has condition"),
        (lambda m, s, *, condition, language: _wrong_threshold(m, condition, language), "threshold is not 17"),
        (lambda m, s, *, condition, language: _empty_transcript(m, condition, language), "empty transcript"),
        (lambda m, s, *, condition, language: _missing_audio(m, condition, language), "missing audio"),
        (lambda m, s, *, condition, language: _cross_condition_score_mismatch(m, condition, language), "differ from the positive source"),
    ],
)
def test_pooled_manifest_builder_fails_closed(tmp_path: Path, mutate, message: str) -> None:
    sources = _write_pooled_sources(tmp_path / "sources", mutate=mutate)
    with pytest.raises(pooled_builder.ManifestError, match=message):
        pooled_builder.main(_builder_argv(sources, tmp_path / "runtime"))


def _drop_translation(manifest: list[dict], condition: str, language: str) -> None:
    if language == "english":
        manifest[0].pop("translation_sha256", None)


def _break_translation(manifest: list[dict], condition: str, language: str) -> None:
    if language == "english":
        manifest[0]["translation_sha256"] = "0" * 64


def _duplicate_sample_id(manifest: list[dict], condition: str, language: str) -> None:
    if condition == "neg":
        manifest[0]["sample_id"] = "s000-pos_only_t17-0"


def _move_subject_across_folds(manifest: list[dict], split: dict) -> None:
    split["1"]["final_eval_subject_ids"].append("s000")


def _non_turkish_dataset(manifest: list[dict], condition: str, language: str) -> None:
    if condition == "pos":
        manifest[0]["dataset"] = "turkish_geriatri"


def _wrong_condition_tag(manifest: list[dict], condition: str, language: str) -> None:
    if condition == "pos":
        manifest[0]["dataset_variant"] = "negative_only_t17"


def _wrong_threshold(manifest: list[dict], condition: str, language: str) -> None:
    if condition == "neg":
        manifest[0]["threshold"] = 13.0


def _empty_transcript(manifest: list[dict], condition: str, language: str) -> None:
    if condition == "pos" and language == "native":
        manifest[0]["transcript"] = "   "


def _missing_audio(manifest: list[dict], condition: str, language: str) -> None:
    if condition == "neg" and language == "native":
        manifest[0]["audio_path"] = "/nonexistent/missing.wav"
        manifest[0]["audio_paths"] = ["/nonexistent/missing.wav"]


def _cross_condition_score_mismatch(manifest: list[dict], condition: str, language: str) -> None:
    # Keep the subject consistent inside its condition but disagree with the
    # positive source, so the cross-condition identity check is the one that fires.
    if condition == "neg" and language == "native":
        subject = manifest[0]["subject_id"]
        for row in manifest:
            if row["subject_id"] == subject:
                row["score"] = 3.0


# --- preflight tooling --------------------------------------------------------


def _built_runtime(tmp_path: Path) -> Path:
    sources = _write_pooled_sources(tmp_path / "sources")
    runtime = tmp_path / "runtime"
    assert pooled_builder.main(_builder_argv(sources, runtime)) == 0
    return runtime


def test_pooled_inputs_audit_accepts_the_built_runtime(tmp_path: Path) -> None:
    runtime = _built_runtime(tmp_path)
    payload = inputs_audit.audit(runtime)
    assert payload["native"]["rows"] == 2221
    assert payload["native"]["subjects"] == 120
    assert payload["native"]["label_counts"] == {"0": 37, "1": 83}
    assert payload["native"]["fold_counts"] == {"0": 24, "1": 24, "2": 24, "3": 24, "4": 24}
    assert payload["pairing"]["identity_projection_equal"] is True
    assert payload["pairing"]["paired_rows"] == 2221
    # The synthetic build cannot reproduce the recorded production identities.
    assert inputs_audit.main(["--runtime-root", str(runtime), "--expect-recorded-hashes"]) == 2
    assert inputs_audit.main(["--runtime-root", str(runtime)]) == 0


def test_pooled_inputs_audit_fails_closed(tmp_path: Path) -> None:
    runtime = _built_runtime(tmp_path)
    manifest = runtime / "manifests/turkish/turkish_manifest.jsonl"
    lines = manifest.read_text(encoding="utf-8").splitlines()
    manifest.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(inputs_audit.AuditError, match="rows"):
        inputs_audit.audit(runtime)

    runtime = _built_runtime(tmp_path / "second")
    folds = runtime / "splits/turkish/turkish_folds.json"
    payload = json.loads(folds.read_text(encoding="utf-8"))
    payload["1"]["final_eval_subject_ids"].append("s000")
    folds.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(inputs_audit.AuditError):
        inputs_audit.audit(runtime)


def test_prompt_rendering_audit_passes_without_a_tokenizer(tmp_path: Path) -> None:
    from tools import audit_qwen3_prompt_rendering as rendering_audit

    runtime = _built_runtime(tmp_path)
    output = tmp_path / "prompt_audit.json"
    assert rendering_audit.main(["--runtime-root", str(runtime), "--output", str(output)]) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "passed"
    assert {f"{cell['language']}/{cell['modality']}" for cell in payload["cells"]} == {
        "native/text_only",
        "native/audio_only",
        "native/audio_text",
        "english/text_only",
        "english/audio_text",
    }
    assert payload["qwen38_rendering"] is None
    for cell in payload["cells"]:
        for record in cell["records"]:
            assert record["question_context"]
            assert record["prompt_text_sha256"]
            assert record["subject_id"] not in cell["config"]


def test_pooled_preflight_submitter_dry_run_contract(tmp_path: Path) -> None:
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/submit_qwen3_pooled_preflight.sh")],
        cwd=ROOT,
        env={
            **os.environ,
            "PROJECT_ROOT": str(ROOT),
            "STAGE": "qwen38",
            "RUNTIME_ROOT": str(tmp_path / "runtime"),
            "DRY_RUN": "1",
        },
        text=True,
        capture_output=True,
        check=True,
    )
    assert "dry-run contract" in result.stdout
    assert "stage: qwen38" in result.stdout
    assert "qwen38_fsdp_fastpath_20260921" in result.stdout
    assert "resources: 1 CPU node" in result.stdout
    assert "no model weights" in result.stdout
    assert "jobs: 1" in result.stdout
    assert "sbatch" in result.stdout

    bad = subprocess.run(
        ["bash", str(ROOT / "scripts/submit_qwen3_pooled_preflight.sh")],
        cwd=ROOT,
        env={**os.environ, "PROJECT_ROOT": str(ROOT), "STAGE": "bogus",
             "RUNTIME_ROOT": str(tmp_path), "DRY_RUN": "1"},
        text=True,
        capture_output=True,
    )
    assert bad.returncode == 2
    assert "STAGE must be" in bad.stderr

    missing_sources = subprocess.run(
        ["bash", str(ROOT / "scripts/submit_qwen3_pooled_preflight.sh")],
        cwd=ROOT,
        env={**os.environ, "PROJECT_ROOT": str(ROOT), "STAGE": "pooled",
             "RUNTIME_ROOT": str(tmp_path), "DRY_RUN": "1"},
        text=True,
        capture_output=True,
    )
    assert missing_sources.returncode == 3
    assert "SOURCE_INPUT_ROOT" in missing_sources.stderr
