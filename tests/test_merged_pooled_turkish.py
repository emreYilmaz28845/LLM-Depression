"""Tests for the pooled question-conditioned Turkish input in the merged family."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.data.runtime import (
    QUESTION_CONTEXT_SENTENCES,
    build_examples,
    render_user_prompt_text,
)
from src.merged.protocol import namespace_row
from src.utils import load_yaml

import scripts.prepare_harmonized_mn5 as preflight


ROOT = Path(__file__).parents[1]
MERGED = ROOT / "configs/experiments/merged"
HEX64 = re.compile(r"^[0-9a-f]{64}$")

# parent (harmonized_v1) -> pooled variant
PAIRS = (
    ("symmetric_merged_harmonized_audio_text.yaml", "symmetric_merged_harmonized_pooled_t17_audio_text.yaml"),
    ("symmetric_merged_harmonized_audio_only.yaml", "symmetric_merged_harmonized_pooled_t17_audio_only.yaml"),
    ("symmetric_merged_harmonized_text_only.yaml", "symmetric_merged_harmonized_pooled_t17_text_only.yaml"),
    ("symmetric_merged_harmonized_gemma4_audio_text.yaml", "symmetric_merged_harmonized_gemma4_pooled_t17_audio_text.yaml"),
    ("symmetric_merged_harmonized_gemma4_audio_only.yaml", "symmetric_merged_harmonized_gemma4_pooled_t17_audio_only.yaml"),
    ("symmetric_merged_harmonized_gemma4_text_only.yaml", "symmetric_merged_harmonized_gemma4_pooled_t17_text_only.yaml"),
)


def _config(name: str) -> dict:
    return load_yaml(MERGED / name)


def _turkish_component(config: dict) -> dict:
    components = [item for item in config["components"] if item["name"] == "turkish"]
    assert len(components) == 1
    return components[0]


@pytest.mark.parametrize(("parent_name", "pooled_name"), PAIRS)
def test_pooled_merged_variant_changes_only_the_turkish_component(
    parent_name: str, pooled_name: str
) -> None:
    parent = _config(parent_name)
    pooled = _config(pooled_name)

    assert pooled["name"] == pooled_name[: -len(".yaml")]
    assert pooled["protocol"] == parent["protocol"]
    assert pooled["modality"] == parent["modality"]
    assert pooled["seed"] == parent["seed"]
    assert pooled["model_name_or_path"] == parent["model_name_or_path"]
    assert pooled.get("model_backend") == parent.get("model_backend")
    assert (
        pooled["recipe_id"]
        == "harmonized_full_transcript_single30_allwindows_selmacrof1_tf_qcond_v1"
    )
    assert pooled["protocol_settings"] == parent["protocol_settings"]
    assert pooled["training"] == parent["training"]
    assert pooled["heads"] == parent["heads"]
    assert pooled["execution"] == parent["execution"]

    # every non-Turkish component is untouched: only Turkish may change
    assert [item for item in pooled["components"] if item["name"] != "turkish"] == [
        item for item in parent["components"] if item["name"] != "turkish"
    ]
    assert [item["name"] for item in pooled["components"]] == [
        item["name"] for item in parent["components"]
    ]


@pytest.mark.parametrize(("parent_name", "pooled_name"), PAIRS)
def test_pooled_merged_variant_binds_the_pooled_turkish_input(
    parent_name: str, pooled_name: str
) -> None:
    parent = _config(parent_name)
    pooled = _config(pooled_name)
    turkish = _turkish_component(pooled)

    assert "turkish_pooled_t17" in turkish["config"]
    assert "turkish_pos_only_t17" not in turkish["config"]
    assert turkish["config"] != _turkish_component(parent)["config"]
    assert (ROOT / turkish["config"]).is_file()

    modality = pooled["modality"]
    assert turkish["config"].endswith(
        f"turkish_pooled_t17_{modality}_harmonized_selmacrof1_tf_qwen3asr.yaml"
        if pooled.get("model_backend") != "gemma4"
        else f"turkish_pooled_t17_{modality}_harmonized_selmacrof1_tf_qwen3asr_gemma4_12b.yaml"
    )
    assert turkish["manifest_path"].endswith(
        "manifests_harmonized/turkish_pooled_t17_qwen3asr/turkish_manifest.jsonl"
    )
    assert turkish["metadata_path"].endswith(
        "splits_harmonized/turkish_pooled_t17_qwen3asr/turkish_manifest_metadata.json"
    )


@pytest.mark.parametrize(("parent_name", "pooled_name"), PAIRS)
def test_pooled_merged_variant_writes_to_a_new_campaign(
    parent_name: str, pooled_name: str
) -> None:
    parent = _config(parent_name)
    pooled = _config(pooled_name)

    for key in ("merged_root", "run_root"):
        assert "harmonized_v2_pooled_t17" in pooled["output_dirs"][key]
        assert "harmonized_v1" not in pooled["output_dirs"][key]
        assert pooled["output_dirs"][key] != parent["output_dirs"][key]


def test_pooled_component_set_swaps_only_turkish() -> None:
    components = preflight.pooled_component_configs()
    non_turkish = tuple(path for path in preflight.COMPONENT_CONFIGS if "/turkish_" not in path)

    assert len(components) == len(preflight.COMPONENT_CONFIGS)
    assert components[: len(non_turkish)] == non_turkish
    assert components[-1] == preflight.POOLED_TURKISH_COMPONENT_CONFIG
    assert preflight.POOLED_TURKISH_COMPONENT_CONFIG not in preflight.COMPONENT_CONFIGS
    assert all("/turkish_pos_only" not in path for path in components)
    for path in components:
        assert (ROOT / path).is_file()


def test_preflight_registers_the_pooled_merged_family() -> None:
    expected = {name for _, name in PAIRS}
    registered = {
        Path(path).name
        for path in (*preflight.POOLED_MERGED_CONFIGS, *preflight.POOLED_GEMMA_MERGED_CONFIGS)
    }
    assert registered == expected
    assert len(preflight.POOLED_MERGED_CONFIGS) == 3
    assert len(preflight.POOLED_GEMMA_MERGED_CONFIGS) == 3
    for path in (*preflight.POOLED_MERGED_CONFIGS, *preflight.POOLED_GEMMA_MERGED_CONFIGS):
        assert (ROOT / path).is_file()
    # the harmonized_v1 family stays registered and untouched
    assert "configs/experiments/merged/symmetric_merged_harmonized_text_only.yaml" in (
        preflight.MERGED_CONFIGS
    )


def test_published_pooled_contract_is_internally_consistent() -> None:
    sources = preflight.POOLED_TURKISH_SOURCE_FILES
    assert set(sources) == {
        ("pos_only_t17", "native"),
        ("pos_only_t17", "english"),
        ("negative_only_t17", "native"),
        ("negative_only_t17", "english"),
    }
    rows = {key: value[2] for key, value in sources.items()}
    assert rows[("pos_only_t17", "native")] == rows[("pos_only_t17", "english")] == 1051
    assert (
        rows[("negative_only_t17", "native")]
        == rows[("negative_only_t17", "english")]
        == 1170
    )
    assert sum(rows.values()) == 2 * preflight.POOLED_TURKISH_EXPECTED_ROWS
    for _, sha256, _ in sources.values():
        assert HEX64.match(sha256)
    assert HEX64.match(preflight.POOLED_TURKISH_SOURCE_SPLIT_SHA256)

    expected = preflight.POOLED_TURKISH_EXPECTED
    for language in ("native", "english"):
        assert HEX64.match(expected[language]["manifest_sha256"])
        assert HEX64.match(expected[language]["manifest_hash"])
        assert HEX64.match(expected[language]["folds_sha256"])
        assert HEX64.match(expected[language]["fold_hash"])
        assert (
            expected[language]["folds_sha256"]
            == preflight.POOLED_TURKISH_EXPECTED[language]["folds_sha256"]
        )
    assert (
        expected["native"]["manifest_sha256"] != expected["english"]["manifest_sha256"]
    )
    assert sum(preflight.POOLED_TURKISH_EXPECTED_CONDITIONS.values()) == (
        preflight.POOLED_TURKISH_EXPECTED_ROWS
    )


def test_pooled_source_verification_rejects_missing_root(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        preflight.verify_pooled_turkish_source(tmp_path / "absent")


def test_pooled_source_verification_rejects_incomplete_root(tmp_path: Path) -> None:
    (tmp_path / "manifests/pos_native").mkdir(parents=True)
    (tmp_path / "manifests/pos_native/turkish_manifest.jsonl").write_text("{}\n")
    with pytest.raises(FileNotFoundError):
        preflight.verify_pooled_turkish_source(tmp_path)


def test_pooled_source_verification_rejects_tampered_inputs(tmp_path: Path) -> None:
    for _, (relative, _, _) in preflight.POOLED_TURKISH_SOURCE_FILES.items():
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{}\n")
    for relative in preflight.POOLED_TURKISH_SOURCE_SPLIT_FILES.values():
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{}\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        preflight.verify_pooled_turkish_source(tmp_path)


def test_pooled_output_gate_requires_the_built_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_dir = tmp_path / "manifests"
    split_dir = tmp_path / "splits"
    manifest_dir.mkdir()
    split_dir.mkdir()
    monkeypatch.setattr(
        preflight, "_pooled_output_dirs", lambda config_path: (manifest_dir, split_dir)
    )
    with pytest.raises(FileNotFoundError, match="metadata"):
        preflight.verify_pooled_turkish_outputs()


def test_pooled_output_gate_rejects_a_non_pooled_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_dir = tmp_path / "manifests"
    split_dir = tmp_path / "splits"
    manifest_dir.mkdir()
    split_dir.mkdir()
    (split_dir / "turkish_manifest_metadata.json").write_text(
        '{"dataset_variant": "pos_only_t17"}\n'
    )
    monkeypatch.setattr(
        preflight, "_pooled_output_dirs", lambda config_path: (manifest_dir, split_dir)
    )
    with pytest.raises(ValueError, match="pooled_t17"):
        preflight.verify_pooled_turkish_outputs()


def test_pooled_build_requires_an_explicit_source_root() -> None:
    with pytest.raises(ValueError, match="source root"):
        preflight.prepare(
            run_id="test-pooled-source-root",
            build=True,
            required_path_prefix=None,
            build_merged=False,
            pooled_turkish=True,
        )


def test_pooled_build_wires_sources_to_the_canonical_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pin the rebuild wiring: the verified sources go to the pooled config's own directories."""
    captured: list[list[str]] = []
    monkeypatch.setattr(preflight, "verify_pooled_turkish_source", lambda root: {"root": str(root)})
    monkeypatch.setattr(
        preflight, "verify_pooled_turkish_outputs", lambda: {"native": {}, "english": {}}
    )
    builder = pytest.importorskip("scripts.build_turkish_pooled_manifest")
    monkeypatch.setattr(builder, "main", lambda argv: captured.append(list(argv)))

    source_root = tmp_path / "pooled"
    payload = preflight.build_pooled_turkish(source_root, audit_path=tmp_path / "audit.json")

    assert payload == {"source": {"root": str(source_root)}, "outputs": {"native": {}, "english": {}}}
    argv = captured[0]

    def value(flag: str) -> str:
        return argv[argv.index(flag) + 1]

    manifest_flags = {
        ("pos_only_t17", "native"): "--positive-native-manifest",
        ("negative_only_t17", "native"): "--negative-native-manifest",
        ("pos_only_t17", "english"): "--positive-english-manifest",
        ("negative_only_t17", "english"): "--negative-english-manifest",
    }
    for key, flag in manifest_flags.items():
        assert value(flag) == str(source_root / preflight.POOLED_TURKISH_SOURCE_FILES[key][0])
        split_flag = flag.replace("-manifest", "-split")
        assert value(split_flag) == str(
            source_root / preflight.POOLED_TURKISH_SOURCE_SPLIT_FILES[key]
        )

    assert value("--native-output-dir").endswith(
        "outputs/manifests_harmonized/turkish_pooled_t17_qwen3asr"
    )
    assert value("--native-split-output-dir").endswith(
        "outputs/splits_harmonized/turkish_pooled_t17_qwen3asr"
    )
    assert value("--english-output-dir").endswith(
        "outputs/manifests_harmonized_en/turkish_pooled_t17_qwen3asr"
    )
    assert value("--english-split-output-dir").endswith(
        "outputs/splits_harmonized_en/turkish_pooled_t17_qwen3asr"
    )
    assert value("--native-config").endswith(preflight.POOLED_TURKISH_COMPONENT_CONFIG)
    assert value("--english-config").endswith(preflight.POOLED_TURKISH_COMPONENT_CONFIG_EN)
    assert value("--audit-output") == str(tmp_path / "audit.json")


def test_namespacing_preserves_the_question_condition() -> None:
    row = {
        "dataset": "turkish",
        "dataset_variant": "negative_only_t17",
        "sample_id": "s1::negative_only_t17",
        "subject_id": "s1",
        "label": 1,
        "transcript": "text",
        "audio_path": "audio.wav",
        "response_id": "r1",
    }
    namespaced = namespace_row(row, "turkish")

    assert namespaced["dataset"] == "turkish"
    assert namespaced["dataset_variant"] == "negative_only_t17"
    assert namespaced["subject_id"] == "turkish::s1"
    assert namespaced["sample_id"] == "turkish::s1::negative_only_t17"
    assert namespaced["response_id"] == "turkish::r1"
    assert namespaced["component_sample_id"] == "s1::negative_only_t17"


@pytest.mark.parametrize(
    "config_name",
    (
        "turkish_pooled_t17_audio_text_harmonized_selmacrof1_tf_qwen3asr.yaml",
        "turkish_pooled_t17_audio_only_harmonized_selmacrof1_tf_qwen3asr.yaml",
        "turkish_pooled_t17_text_only_harmonized_selmacrof1_tf_qwen3asr.yaml",
    ),
)
def test_pooled_component_prompts_switch_on_the_condition(config_name: str) -> None:
    """Merged renders each component with its own config, so the pooled switch carries over."""
    config = load_yaml(ROOT / "configs/main" / config_name)
    assert "{question_context}" in config["prompt"]["user_template"]

    for condition, sentence in QUESTION_CONTEXT_SENTENCES.items():
        rendered = render_user_prompt_text(
            config, "the transcript", question_condition=condition
        )
        assert sentence in rendered
        if config["data"]["use_text"]:
            assert rendered.index(sentence) < rendered.index("the transcript")

    with pytest.raises(ValueError, match="question_condition"):
        render_user_prompt_text(config, "x", question_condition=None)


def test_pooled_text_only_examples_keep_one_pair_per_subject() -> None:
    config = load_yaml(
        ROOT / "configs/main/turkish_pooled_t17_text_only_harmonized_selmacrof1_tf_qwen3asr.yaml"
    )
    rows = [
        {
            "dataset": "turkish",
            "dataset_variant": condition,
            "sample_id": f"{condition}-s{subject}",
            "subject_id": f"s{subject}",
            "label": 1,
            "label_text": "Depressed",
            "transcript": f"transcript-{condition}-{subject}",
            "audio_path": "",
        }
        for subject in ("1", "2")
        for condition in ("pos_only_t17", "negative_only_t17")
    ]
    examples = build_examples(rows, config, "train")
    assert len(examples) == 4
    for example in examples:
        sentence = QUESTION_CONTEXT_SENTENCES[str(example["question_condition"])]
        assert sentence in str(example["prompt_user_text"])
