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
    allowed = {"recipe_id", "manifest_variant", "data.segment_seconds", "data.participant_chunk_samples"}
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
