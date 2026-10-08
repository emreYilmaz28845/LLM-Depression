"""Focused tests for the window15 treatment recipes.

Covers the config generator allowlist, the versioned Androids opt-in guard, and
the equal_duration ceil semantics used by the response datasets.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

LANE = Path(__file__).resolve().parents[1]

import sys

if str(LANE) not in sys.path:
    sys.path.insert(0, str(LANE))

from src.data.androids import (  # noqa: E402
    ANDROIDS_ALLOWED_SEGMENT_SECONDS,
    ANDROIDS_DEFAULT_SEGMENT_SECONDS,
    build_androids_interview_manifest,
    equal_duration_windows as androids_equal_duration_windows,
)
from src.data.d3tec import equal_duration_windows as d3tec_equal_duration_windows  # noqa: E402
from tools.qwen3_window15_build_configs import generate, parsed_diff  # noqa: E402

CONTRACT = LANE / "outputs/qwen3_window15_20261008/contracts/treatment_contract.json"


def test_generator_allowlist_and_files() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert len(contract["routes"]) == 10
    allowed = {
        "recipe_id",
        "manifest_variant",
        "data.segment_seconds",
        "data.participant_chunk_samples",
        "data.processor_min_audio_samples",
    }
    import yaml

    for route_id, route in contract["routes"].items():
        control_path = LANE / route["control_config"]
        target_path = LANE / route["treatment_config"]
        control_text = control_path.read_text(encoding="utf-8")
        treatment_text, _ = generate(control_text)
        assert target_path.is_file(), f"missing treatment config for {route_id}"
        assert target_path.read_text(encoding="utf-8") == treatment_text
        diff = parsed_diff(yaml.safe_load(control_text), yaml.safe_load(treatment_text))
        assert diff <= allowed, f"{route_id}: undeclared diff {sorted(diff - allowed)}"
        assert diff, f"{route_id}: no declared diff applied"


def test_recipe_marker_and_window_values() -> None:
    import yaml

    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    for route_id, route in contract["routes"].items():
        parsed = yaml.safe_load((LANE / route["treatment_config"]).read_text(encoding="utf-8"))
        assert "single15" in str(parsed["recipe_id"])
        if str(parsed["data"].get("sample_mode")) == "participant_speech_packed30":
            assert parsed["manifest_variant"] == "unprocessed_participant_speech_packed30_15s_v1"
            assert int(parsed["data"]["participant_chunk_samples"]) == 240000
        else:
            assert math.isclose(float(parsed["data"]["segment_seconds"]), 15.0)


def test_androids_guard_allows_15_and_30_only(tmp_path: Path) -> None:
    assert set(ANDROIDS_ALLOWED_SEGMENT_SECONDS) == {30.0, 15.0}
    base = {
        "dataset_root": str(tmp_path / "missing-root"),
        "full_transcript_path": str(tmp_path / "missing.jsonl"),
        "data": {"segment_seconds": 15.0, "segment_partition": "equal_duration"},
        "split": {"outer_folds": 5},
    }
    # 15 seconds passes the guard (the build then fails on the missing dataset).
    with pytest.raises(Exception) as allowed_exc:
        build_androids_interview_manifest(dict(base), {})
    assert "segment_seconds in" not in str(allowed_exc.value)
    # Other durations are refused by the guard.
    for value in (20.0, 45.0, 0.0):
        bad = json.loads(json.dumps(base))
        bad["data"]["segment_seconds"] = value
        with pytest.raises(ValueError, match="segment_seconds in"):
            build_androids_interview_manifest(bad, {})


def test_equal_duration_ceil_semantics_at_15_seconds() -> None:
    for equal_duration_windows in (d3tec_equal_duration_windows, androids_equal_duration_windows):
        windows_30 = equal_duration_windows(44.0, 30.0)
        windows_15 = equal_duration_windows(44.0, 15.0)
        assert len(windows_30) == 2
        assert len(windows_15) == 3
        for start, end in windows_15:
            assert end > start
            assert math.isclose(end - start, 44.0 / 3, rel_tol=1e-9)
        # Full coverage and no overlap.
        assert math.isclose(windows_15[0][0], 0.0)
        assert math.isclose(windows_15[-1][1], 44.0)
        for (_, end), (start, _) in zip(windows_15, windows_15[1:]):
            assert math.isclose(end, start)


def test_source_reference_index_is_a_reference_not_containment() -> None:
    from src.data.window_mapping import reference_fields, source_reference_index

    # duration 70: canonical widths are 70/3; the 15s child [14,28] references
    # canonical window 0 but extends past its end 70/3 -- a reference, not a
    # containment or alignment claim.
    assert source_reference_index(14.0, 70.0) == 0
    assert 28.0 > 70.0 / 3
    starts = [index * (70.0 / 5) for index in range(5)]
    assert [source_reference_index(start, 70.0) for start in starts] == [0, 0, 1, 1, 2]
    # duration 90: first child [0,15] references canonical [0,30]; the treatment
    # path must not apply exact interval equality.
    assert source_reference_index(0.0, 90.0) == 0
    # Unchanged 30-second identity: no reference fields are added.
    assert reference_fields(30.0, 30.0, 12.0, 70.0) == {}
    # Subdivided path adds the reference index.
    assert reference_fields(15.0, 30.0, 14.0, 70.0) == {"source_reference_index": 0}
    # Boundaries and invalid inputs.
    assert source_reference_index(29.999, 60.0) == 0
    assert source_reference_index(30.0, 60.0) == 1
    for bad in ((0.0, 0.0), (0.0, -1.0)):
        with pytest.raises(ValueError):
            source_reference_index(*bad)


def test_androids_discovery_schema_identity_and_references(tmp_path: Path) -> None:
    import numpy as np
    import soundfile as sf

    from src.data.androids import discover_androids_interview_windows

    clip_dir = tmp_path / "Interview-Task" / "audio_clip" / "001_CF20_5"
    clip_dir.mkdir(parents=True)
    for turn, seconds in ((1, 45.0), (2, 20.0)):
        sf.write(
            clip_dir / f"001_CF20_5_{turn}.wav",
            np.zeros(int(seconds * 1000), dtype="float32"),
            1000,
        )
    rows_30 = discover_androids_interview_windows(
        tmp_path, segment_seconds=30.0, enforce_corpus_contract=False
    )
    # Canonical 30-second schema is unchanged: no reference fields anywhere.
    assert all("source_reference_index" not in row for row in rows_30)
    assert len(rows_30) == 3  # ceil(45/30) + ceil(20/30)
    rows_15 = discover_androids_interview_windows(
        tmp_path, segment_seconds=15.0, enforce_corpus_contract=False
    )
    assert len(rows_15) == 5  # ceil(45/15) + ceil(20/15)
    refs: dict[int, list[int]] = {}
    for row in rows_15:
        assert "source_reference_index" in row
        refs.setdefault(int(row["turn_id"]), []).append(int(row["source_reference_index"]))
    assert refs[1] == [0, 0, 1]
    assert refs[2] == [0, 0]


class _FakeQwenProcessor:
    """Minimal stand-in that records the waveform lengths it receives."""

    class _Tokenizer:
        pad_token_id = 0

    class _FeatureExtractor:
        sampling_rate = 16000

    tokenizer = _Tokenizer()
    feature_extractor = _FeatureExtractor()

    def __init__(self) -> None:
        self.received_audio_lengths: list[int] = []

    def __call__(self, *, text, audio=None, sampling_rate=None, return_tensors=None, padding=False):
        import numpy as np

        if audio is not None:
            self.received_audio_lengths.append(int(len(audio[0])))
            features = np.zeros((1, 128, 2), dtype=np.float32)
            mask = np.ones((1, 2), dtype=np.int64)
        else:
            features = None
            mask = None
        return {
            "input_ids": [1, 2, 3],
            "attention_mask": [1, 1, 1],
            "input_features": features,
            "feature_attention_mask": mask,
        }


def test_audio_padding_helper_and_resolver() -> None:
    import numpy as np

    from src.model.audio_padding import (
        MIN_PROCESSOR_SAMPLES,
        pad_audio_array,
        pad_audio_arrays,
        resolve_min_audio_samples,
    )

    assert MIN_PROCESSOR_SAMPLES == 201
    padded, missing = pad_audio_array(np.zeros(160, dtype=np.float32), 201)
    assert padded.shape == (201,)
    assert missing == 41
    signal = np.arange(160, dtype=np.float32)
    padded_signal, _ = pad_audio_array(signal, 201)
    assert np.array_equal(padded_signal[:160], signal)
    assert float(np.abs(padded_signal[160:]).sum()) == 0.0
    same, missing = pad_audio_array(np.zeros(201, dtype=np.float32), 201)
    assert missing == 0
    arrays, total = pad_audio_arrays([np.zeros(160), np.zeros(400)], 201)
    assert total == 41
    assert [len(array) for array in arrays] == [201, 400]
    assert resolve_min_audio_samples({}) == 0
    assert resolve_min_audio_samples({"data": {}}) == 0
    assert resolve_min_audio_samples({"data": {"processor_min_audio_samples": 201}}) == 201
    for bad in (200, -1):
        with pytest.raises(ValueError):
            resolve_min_audio_samples({"data": {"processor_min_audio_samples": bad}})
    with pytest.raises(ValueError):
        resolve_min_audio_samples({"data": {"processor_min_audio_samples": "abc"}})


def test_training_collator_pads_short_audio_and_records_audit() -> None:
    import numpy as np

    from src.model.collator import Qwen2AudioSFTCollator

    example = {
        "audio_arrays": [np.zeros(160, dtype=np.float32)],
        "training_text": "t",
        "prompt_text": "p",
        "sample_id": "s1",
        "subject_id": "sub1",
        "label": 1,
    }
    processor = _FakeQwenProcessor()
    collator = Qwen2AudioSFTCollator(processor, min_audio_samples=201)
    collator([dict(example)])
    assert processor.received_audio_lengths == [201, 201]
    audit = collator.padding_audit()
    assert audit["padded_examples"] == 1
    assert audit["padded_samples_total"] == 41
    assert audit["min_real_samples"] == 160
    sample = audit["padded_examples_sample"][0]
    assert sample["sample_id"] == "s1"
    assert sample["waveforms"] == 1
    assert sample["min_real_samples"] == 160
    assert sample["padded_samples_total"] == 41
    assert "not corpus totals" in audit["count_semantics"]

    processor2 = _FakeQwenProcessor()
    collator2 = Qwen2AudioSFTCollator(processor2, min_audio_samples=201)
    collator2([dict(example, audio_arrays=[np.zeros(201, dtype=np.float32)], sample_id="s2")])
    assert processor2.received_audio_lengths == [201, 201]
    assert collator2.padding_audit()["padded_examples"] == 0

    processor3 = _FakeQwenProcessor()
    collator3 = Qwen2AudioSFTCollator(processor3)
    collator3([dict(example)])
    assert processor3.received_audio_lengths == [160, 160]
    assert collator3.padding_audit()["padded_examples"] == 0


def test_extraction_collator_pads_short_audio_and_records_audit() -> None:
    import numpy as np

    from src.features.qwen_hidden_collator import PromptOnlyExtractionCollator

    example = {
        "audio_arrays": [np.zeros(160, dtype=np.float32)],
        "prompt_text": "p",
        "dataset": "daic",
        "sample_id": "s1",
        "subject_id": "sub1",
        "label": 1,
        "partition": "test",
        "fold": 0,
    }
    processor = _FakeQwenProcessor()
    collator = PromptOnlyExtractionCollator(processor, min_audio_samples=201)
    collator([dict(example)])
    assert processor.received_audio_lengths == [201]
    audit = collator.padding_audit()
    assert audit["padded_examples"] == 1
    assert audit["padded_samples_total"] == 41
    assert audit["padded_examples_sample"][0]["min_real_samples"] == 160
    assert "not corpus totals" in audit["count_semantics"]


def test_treatment_configs_declare_processor_minimum() -> None:
    import yaml

    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    for route_id, route in contract["routes"].items():
        parsed = yaml.safe_load((LANE / route["treatment_config"]).read_text(encoding="utf-8"))
        assert int(parsed["data"]["processor_min_audio_samples"]) == 201, route_id
        assert "data.processor_min_audio_samples" in route["declared_diff"], route_id


def test_child_window_transcript_fields_keep_full_text_and_empty_segment() -> None:
    from src.data.androids import child_window_transcript_fields as androids_fields
    from src.data.d3tec import child_window_transcript_fields as d3tec_fields

    for fields in (d3tec_fields, androids_fields):
        transcript, segment = fields("full canonical text")
        assert transcript == "full canonical text"
        assert segment == ""


def test_harmonized_unit_transcript_prefers_full_fields() -> None:
    from src.data.runtime import _harmonized_unit_transcript

    assert (
        _harmonized_unit_transcript(
            {"full_response_transcript": "full", "transcript": "manifest"}, "d3tec"
        )
        == "full"
    )
    assert (
        _harmonized_unit_transcript(
            {"full_turn_transcript": "turn", "transcript": "manifest"},
            "androids_interview",
        )
        == "turn"
    )
