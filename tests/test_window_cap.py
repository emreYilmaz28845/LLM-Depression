"""Unit tests for the deterministic training-window cap."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.window_cap import (  # noqa: E402
    ALGORITHM_VERSION,
    WindowCapError,
    apply_training_window_cap,
    build_mask,
    permute_subject_samples,
    selections_for_fraction,
)

SEED = 1337


def _examples(subject_sizes: dict[str, int], *, prefix: str = "s") -> list[dict]:
    rows: list[dict] = []
    for subject_id, count in subject_sizes.items():
        for index in range(count):
            rows.append(
                {
                    "subject_id": subject_id,
                    "sample_id": f"{prefix}{subject_id}_w{index:03d}",
                    "label": 1,
                    "raw_loss_weight": 1.0 / count,
                    "loss_weight": 1.0 / count,
                }
            )
    return rows


def test_permutation_is_deterministic_across_processes() -> None:
    script = (
        "import sys; sys.path.insert(0, r'%s');"
        "from src.data.window_cap import permute_subject_samples;"
        "print(permute_subject_samples('301', sys.argv[1:], %d))"
        % (ROOT, SEED)
    )
    ids = [f"w{i:03d}" for i in range(12)]
    outputs = []
    for hash_seed in ("0", "42"):
        proc = subprocess.run(
            [sys.executable, "-c", script, *ids],
            capture_output=True,
            text=True,
            env={"PYTHONHASHSEED": hash_seed, "PATH": "/usr/bin:/bin"},
            cwd=ROOT,
            check=True,
        )
        outputs.append(proc.stdout.strip())
    assert outputs[0] == outputs[1]
    assert outputs[0] == str(permute_subject_samples("301", ids, SEED))


def test_nested_prefixes_for_all_fractions() -> None:
    rows = _examples({"a": 9, "b": 5, "c": 2})
    sel25 = selections_for_fraction(rows, 0.25, SEED)
    sel50 = selections_for_fraction(rows, 0.50, SEED)
    sel75 = selections_for_fraction(rows, 0.75, SEED)
    for subject_id in sel25:
        assert set(sel25[subject_id]) <= set(sel50[subject_id]) <= set(sel75[subject_id])


def test_rounding_max_one_ceil() -> None:
    for count in range(1, 11):
        rows = _examples({"s": count})
        for fraction in (0.25, 0.5, 0.75):
            selections = selections_for_fraction(rows, fraction, SEED)
            expected = max(1, math.ceil(fraction * count))
            assert len(selections["s"]) == expected


def test_input_order_and_labels_do_not_change_selection() -> None:
    rows = _examples({"a": 7, "b": 4})
    shuffled = list(reversed(rows))
    for row in shuffled:
        row["label"] = 1 - row["label"]
    assert selections_for_fraction(rows, 0.5, SEED) == selections_for_fraction(
        shuffled, 0.5, SEED
    )


def test_fraction_one_selects_everything_and_keeps_baseline_weights() -> None:
    baseline = _examples({"a": 6, "b": 3})
    scale = len(baseline) / sum(row["raw_loss_weight"] for row in baseline)
    for row in baseline:
        row["loss_weight"] = row["raw_loss_weight"] * scale
    selected, audit, mask = apply_training_window_cap(
        baseline, fraction=1.0, sampling_seed=SEED
    )
    assert len(selected) == len(baseline)
    by_id = {row["sample_id"]: row for row in selected}
    for row in baseline:
        assert math.isclose(by_id[row["sample_id"]]["loss_weight"], row["loss_weight"])
        assert math.isclose(by_id[row["sample_id"]]["raw_loss_weight"], row["raw_loss_weight"])
    assert audit["omitted_example_count"] == 0
    assert mask["selection_sha256"]


def test_weight_mass_and_relative_contribution_preserved() -> None:
    baseline = _examples({"a": 8, "b": 5, "c": 3})
    baseline_totals = {}
    for row in baseline:
        baseline_totals[row["subject_id"]] = (
            baseline_totals.get(row["subject_id"], 0.0) + row["raw_loss_weight"]
        )
    selected, audit, mask = apply_training_window_cap(
        baseline, fraction=0.5, sampling_seed=SEED
    )
    assert len(selected) == sum(
        max(1, math.ceil(0.5 * count)) for count in (8, 5, 3)
    )
    for subject_id, total in audit["selected_subject_raw_totals_before_rescale"].items():
        assert math.isclose(total, baseline_totals[subject_id], abs_tol=1e-9)
    assert math.isclose(audit["mean_loss_weight"], 1.0, abs_tol=1e-9)
    relative_total = sum(audit["relative_subject_contribution"].values())
    assert math.isclose(relative_total, 1.0, abs_tol=1e-9)
    for subject_id, contribution in audit["relative_subject_contribution"].items():
        assert math.isclose(
            contribution,
            baseline_totals[subject_id] / sum(baseline_totals.values()),
            abs_tol=1e-9,
        )
    assert audit["selection_sha256"] == mask["selection_sha256"]
    assert audit["algorithm_version"] == ALGORITHM_VERSION


def test_mask_artifact_fields_and_hash_stability() -> None:
    rows = _examples({"a": 4, "b": 4})
    mask_a = build_mask(rows, fraction=0.25, sampling_seed=SEED)
    mask_b = build_mask(list(reversed(rows)), fraction=0.25, sampling_seed=SEED)
    assert mask_a["selection_sha256"] == mask_b["selection_sha256"]
    assert mask_a["schema_version"] == "audiollm.window_cap_mask.v1"
    assert mask_a["algorithm_version"] == ALGORITHM_VERSION
    assert mask_a["total_available"] == 8
    assert mask_a["total_selected"] == 2
    assert set(mask_a["subjects"]) == {"a", "b"}
    assert json.dumps(mask_a, sort_keys=True)


def test_duplicate_sample_ids_rejected() -> None:
    rows = _examples({"a": 2})
    rows.append(dict(rows[0]))
    with pytest.raises(WindowCapError, match="unique sample ids"):
        selections_for_fraction(rows, 0.5, SEED)


def test_missing_raw_weight_rejected() -> None:
    rows = _examples({"a": 2})
    for row in rows:
        row.pop("raw_loss_weight")
    with pytest.raises(WindowCapError, match="raw_loss_weight"):
        apply_training_window_cap(rows, fraction=0.5, sampling_seed=SEED)


def test_invalid_fraction_rejected() -> None:
    rows = _examples({"a": 2})
    for fraction in (0.0, 1.5, -0.2):
        with pytest.raises(WindowCapError, match="fraction"):
            selections_for_fraction(rows, fraction, SEED)


def test_omitted_responses_recorded() -> None:
    rows = []
    for index in range(8):
        rows.append(
            {
                "subject_id": "a",
                "sample_id": f"a_w{index:03d}",
                "response_id": f"a_r{index}",
                "label": 0,
                "raw_loss_weight": 1.0 / 8,
                "loss_weight": 1.0 / 8,
            }
        )
    _, audit, _ = apply_training_window_cap(rows, fraction=0.25, sampling_seed=SEED)
    assert audit["omitted_example_count"] == 6
    assert audit["omitted_response_count"] == 6
    assert audit["omitted_examples_per_subject"] == {"a": 6}
