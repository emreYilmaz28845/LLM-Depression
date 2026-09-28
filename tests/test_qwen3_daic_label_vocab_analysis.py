"""Tests for the prespecified Qwen3 DAIC label-vocabulary analysis.

Synthetic evidence is written in the exact layout the campaign produces
(``output_model/<campaign>/<modality>/daic/<run_name>/fold_0``) so the resolver,
the REPORTABLE gate and the fail-closed paths are exercised without any real
result.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import yaml

from tools import qwen3_daic_label_vocab_analysis as analysis
from tools.qwen3_daic_label_vocab_matrix import run_name

ROOT = Path(__file__).resolve().parents[1]
FAMILY = analysis.DEFAULT_FAMILY
SUBJECTS = [str(300 + index) for index in range(8)]


def _rows_for(arm: str, seed: int) -> list[dict[str, str]]:
    """Deterministic subject predictions that differ by arm and seed."""
    offsets = {"ab": 0, "01": 1, "truefalse": 2, "yesno": 3, "en": 4}
    shift = offsets[arm] + int(seed) % 3
    rows = []
    for index, subject in enumerate(SUBJECTS):
        label = index % 2  # alternating labels
        correct = label
        wrong = 1 - label
        right = (index + shift) % 4 != 0
        prediction = correct if right else wrong
        rows.append(
            {
                "subject_id": subject,
                "label": str(label),
                "label_text": "Depressed" if label == 1 else "Non-depressed",
                "prediction": str(prediction),
                "prediction_text": "Depressed" if prediction == 1 else "Non-depressed",
            }
        )
    return rows


def _write_evidence(root: Path, family: dict, *, reportable_skip: str | None = None) -> None:
    for cell in family["cells"]:
        for arm in family["arm_order"]:
            for seed in family["seeds"]:
                fold = (
                    root
                    / family["campaign"]
                    / cell["modality"]
                    / family["dataset"]
                    / run_name(smoke=False, modality=cell["modality"], tag=str(arm), seed=int(seed))
                    / f"fold_{int(family['fold'])}"
                )
                standalone = fold / "best_model" / "standalone_eval"
                standalone.mkdir(parents=True, exist_ok=True)
                state = "LOCALLY_VALIDATED" if reportable_skip == f"{cell['id']}|{arm}|{seed}" else "REPORTABLE"
                (fold / "status.json").write_text(
                    json.dumps({"schema_version": "audiollm.status.v1", "state": state, "attempt_id": f"att-{cell['id']}-{arm}-{seed}"}),
                    encoding="utf-8",
                )
                with (standalone / "predictions_subject_level.csv").open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(
                        handle,
                        fieldnames=["subject_id", "label", "label_text", "prediction", "prediction_text"],
                    )
                    writer.writeheader()
                    writer.writerows(_rows_for(str(arm), int(seed)))


def _run(tmp_path: Path, *, permutations: int = 400, bootstrap: int = 300, reportable_skip: str | None = None) -> Path:
    family = analysis.load_family(FAMILY)
    root = tmp_path / "output_model"
    _write_evidence(root, family, reportable_skip=reportable_skip)
    output = tmp_path / "analysis"
    code = analysis.main([
        "--evidence-root", str(root),
        "--output-dir", str(output),
        "--permutations", str(permutations),
        "--bootstrap-iterations", str(bootstrap),
    ])
    assert code == 0
    return output


def test_family_is_prespecified_and_derives_thirty_contrasts() -> None:
    family = analysis.load_family(FAMILY)
    analysis.validate_family(family)
    members = analysis.contrasts(family)
    assert len(members) == 30
    assert len(analysis.mcnemar_members(family)) == 90
    assert [member["arm_earlier"] for member in members[:10]] == ["ab"] * 4 + ["01"] * 3 + ["truefalse"] * 2 + ["yesno"]
    assert members[0]["arm_later"] == "01"
    assert family["tests"]["iteration_counts"]["permutation"] >= 1_000_000
    assert family["tests"]["iteration_counts"]["bootstrap"] >= 100_000
    assert family["tests"]["analysis_seed"] == 1337
    assert family["alpha"] == 0.05

    broken = dict(family)
    broken["contrast_count"] = 29
    with pytest.raises(analysis.AnalysisError):
        analysis.contrasts(broken)


def test_end_to_end_tables_are_written_and_seed_aware(tmp_path: Path) -> None:
    output = _run(tmp_path)
    seed_rows = list(csv.DictReader((output / "seed_rows.csv").open(encoding="utf-8")))
    assert len(seed_rows) == 45  # 3 cells x 5 arms x 3 seeds
    summary = list(csv.DictReader((output / "summary_long.csv").open(encoding="utf-8")))
    assert len(summary) == 45  # 3 cells x 5 arms x 3 metrics
    contrasts = list(csv.DictReader((output / "contrasts.csv").open(encoding="utf-8")))
    assert len(contrasts) == 90  # 30 contrasts x 3 metrics
    mcnemar = list(csv.DictReader((output / "mcnemar.csv").open(encoding="utf-8")))
    assert len(mcnemar) == 90

    payload = json.loads((output / "analysis.json").read_text(encoding="utf-8"))
    assert payload["family"]["iterations"]["iterations_overridden"] is True
    assert payload["family"]["sha256"] == analysis.hashlib.sha256(FAMILY.read_bytes()).hexdigest()
    for row in payload["contrasts"]:
        assert row["subjects"] == len(SUBJECTS)  # subject clustering
        assert row["keys"] == len(SUBJECTS) * 3  # all three seed records of a subject move together
    markdown = (output / "report.md").read_text(encoding="utf-8")
    for seed in (7, 1337, 2024):
        assert f"| text_only_qwen38_27b | {seed} |" in markdown  # one reader-facing row per cell and seed
    for arm in ("ab", "01", "truefalse", "yesno", "en"):
        assert f"| text_only_qwen38_27b | {arm} |" in markdown  # mean/std summary row
    assert "unadjusted" in markdown


def test_summary_averages_the_seed_rows(tmp_path: Path) -> None:
    output = _run(tmp_path)
    payload = json.loads((output / "analysis.json").read_text(encoding="utf-8"))
    seed_index = {(row["cell"], row["arm"], row["seed"]): row for row in payload["seed_rows"]}
    for row in payload["summary_long"]:
        values = [
            seed_index[(row["cell"], row["arm"], seed)][row["metric"]]
            for seed in (7, 1337, 2024)
        ]
        assert row["n_seeds"] == 3
        assert row["mean"] == pytest.approx(sum(values) / 3)
        # ddof=1 (sample) standard deviation over the three seed values.
        variance = sum((value - sum(values) / 3) ** 2 for value in values) / 2
        assert row["std_ddof1"] == pytest.approx(variance ** 0.5, abs=1e-9)


def test_non_reportable_evidence_is_refused(tmp_path: Path) -> None:
    family = analysis.load_family(FAMILY)
    root = tmp_path / "output_model"
    _write_evidence(root, family, reportable_skip="text_only_qwen38_27b|ab|1337")
    with pytest.raises(analysis.AnalysisError):
        analysis.load_all_rows(root, family)


def test_missing_run_is_refused(tmp_path: Path) -> None:
    family = analysis.load_family(FAMILY)
    root = tmp_path / "output_model"
    _write_evidence(root, family)
    fold = (
        root
        / family["campaign"]
        / "audio_text"
        / "daic"
        / run_name(smoke=False, modality="audio_text", tag="en", seed=2024)
        / "fold_0"
    )
    for child in sorted(fold.rglob("*"), reverse=True):
        child.unlink() if child.is_file() else child.rmdir()
    fold.rmdir()
    with pytest.raises(analysis.AnalysisError):
        analysis.load_all_rows(root, family)


def test_missing_seed_rows_are_refused(tmp_path: Path) -> None:
    family = analysis.load_family(FAMILY)
    root = tmp_path / "output_model"
    _write_evidence(root, family)
    fold = (
        root
        / family["campaign"]
        / "text_only"
        / "daic"
        / run_name(smoke=False, modality="text_only", tag="yesno", seed=7)
        / "fold_0"
        / analysis.PREDICTIONS_RELATIVE
    )
    rows = list(csv.DictReader(fold.open(encoding="utf-8")))
    with fold.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows[:-1])
    rows_by_key, _ = analysis.load_all_rows(root, family)
    with pytest.raises(analysis.AnalysisError):
        analysis.analyse(
            family=family,
            rows_by_key=rows_by_key,
            provenance={},
            permutations=200,
            bootstrap_iterations=200,
            iterations_overridden=True,
        )


def test_family_file_round_trips_through_yaml() -> None:
    family = yaml.safe_load(FAMILY.read_text(encoding="utf-8"))
    assert family["schema_version"] == analysis.EXPECTED_SCHEMA
    assert family["arm_order"] == ["ab", "01", "truefalse", "yesno", "en"]
    assert family["seeds"] == [7, 1337, 2024]
    assert family["cells"][0]["modality"] == "text_only"
