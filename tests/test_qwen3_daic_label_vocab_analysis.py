"""Tests for the prespecified Qwen3 DAIC label-vocabulary analysis.

Synthetic evidence is written in the exact layout the campaign produces
(``output_model/<campaign>/<modality>/daic/<run_name>/fold_0``) including the
tracking sidecars, so the resolver, the REPORTABLE gate, the run-record
verification and the fail-closed paths are exercised without any real result.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from scripts.build_qwen3_daic_label_configs import ARMS
from tools import qwen3_daic_label_vocab_analysis as analysis
from tools.qwen3_daic_label_vocab_matrix import SOURCES, run_name

ROOT = Path(__file__).resolve().parents[1]
FAMILY = analysis.DEFAULT_FAMILY
SUBJECTS = [str(300 + index) for index in range(8)]
CAMPAIGN = "qwen3_daic_label_vocab_v1"


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


def _fold_dir(root: Path, family: dict, modality: str, arm: str, seed: int) -> Path:
    run = run_name(smoke=False, modality=modality, tag=arm, seed=int(seed))
    return root / family["campaign"] / modality / family["dataset"] / run / f"fold_{int(family['fold'])}"


def _write_evidence(root: Path, family: dict, *, faults: dict[str, str] | None = None) -> None:
    """Write the fixtures; ``faults`` injects one mismatch per run key.

    Supported fault names: ``labels``, ``seed``, ``predictions_hash``,
    ``missing_predictions_role``, ``backend``, ``state``.
    """
    faults = faults or {}
    for cell in family["cells"]:
        for arm in family["arm_order"]:
            for seed in family["seeds"]:
                fault = faults.get(f"{cell['id']}|{arm}|{seed}")
                run = run_name(smoke=False, modality=cell["modality"], tag=str(arm), seed=int(seed))
                fold = _fold_dir(root, family, cell["modality"], str(arm), int(seed))
                standalone = fold / "best_model" / "standalone_eval"
                standalone.mkdir(parents=True, exist_ok=True)
                attempt_id = f"att-{cell['id']}-{arm}-{seed}"

                vocab, positive, negative = ARMS[str(arm)]
                labels = {
                    "label_vocab_version": vocab,
                    "internal_positive_label": positive,
                    "internal_negative_label": negative,
                    "external_positive_label": "Depressed",
                    "external_negative_label": "Non-depressed",
                }
                if fault == "labels":
                    labels["label_vocab_version"] = "yesno_labels"

                backend = SOURCES[cell["modality"]][2]
                if fault == "backend":
                    backend = "qwen2audio"

                config_seed = int(seed) + 1 if fault == "seed" else int(seed)
                model_revision = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0" if backend == "qwen38" else None
                run_config = {
                    "config": {
                        "dataset": family["dataset"],
                        "seed": config_seed,
                        "model_backend": backend,
                        "input_modality": cell["modality"],
                        "model_name_or_path": f"/models/{backend}",
                        "labels": labels,
                        "output_dirs": {
                            "run_root": f"/permanent/output_model/{family['campaign']}/{cell['modality']}/{family['dataset']}"
                        },
                        "split": {"seed": 1337},
                        "evaluation": {
                            "sample_prediction_mode": "likelihood",
                            "evaluation_view": "harmonized_all_windows_full_coverage",
                            "aggregation_level": "subject",
                        },
                    },
                    "manifest_hash": "7" * 64,
                    "split_metadata_hash": "8" * 64,
                    "manifest_path": "/runtime/manifests/daic/daic_manifest.jsonl",
                    "split_metadata_path": "/runtime/splits/daic/daic_manifest_metadata.json",
                    "resolved_model_name_or_path": f"/models/{backend}",
                    "tracking": {"attempt_id": attempt_id, "logical_run_name": run, "group_id": "g"},
                }
                if model_revision:
                    run_config["config"]["model_revision"] = model_revision
                (fold / "run_config.yaml").write_text(yaml.safe_dump(run_config), encoding="utf-8")

                predictions = standalone / "predictions_subject_level.csv"
                with predictions.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(
                        handle,
                        fieldnames=["subject_id", "label", "label_text", "prediction", "prediction_text"],
                    )
                    writer.writeheader()
                    writer.writerows(_rows_for(str(arm), int(seed)))
                (standalone / "metrics_likelihood.json").write_text(
                    json.dumps({"aggregation_level": "subject", "num_subjects": len(SUBJECTS)}), encoding="utf-8"
                )

                recorded_predictions = hashlib.sha256(predictions.read_bytes()).hexdigest()
                if fault == "predictions_hash":
                    recorded_predictions = "0" * 64

                artifacts = [
                    {"role": "run_config", "path": "run_config.yaml", "sha256": hashlib.sha256((fold / "run_config.yaml").read_bytes()).hexdigest()},
                    {"role": "standalone_eval_predictions", "path": str(analysis.PREDICTIONS_RELATIVE), "sha256": recorded_predictions},
                ]
                if fault == "missing_predictions_role":
                    artifacts = [artifacts[0]]
                (fold / "artifacts.json").write_text(
                    json.dumps({"schema_version": "audiollm.artifacts.v1", "attempt_id": attempt_id, "fold": 0, "artifacts": artifacts}),
                    encoding="utf-8",
                )
                state = "LOCALLY_VALIDATED" if fault == "state" else "REPORTABLE"
                (fold / "status.json").write_text(
                    json.dumps({"schema_version": "audiollm.status.v1", "state": state, "attempt_id": attempt_id}),
                    encoding="utf-8",
                )
                (fold / "metadata.json").write_text(
                    json.dumps({
                        "schema_version": "audiollm.metadata.v1",
                        "attempt_id": attempt_id,
                        "source": {
                            "git_commit": "d5" * 20,
                            "git_branch": "agent/feat-qwen3-daic-label-vocab",
                            "git_dirty": False,
                            "deployment_id": "deployment-1",
                            "deployed_source_sha256": "9" * 64,
                        },
                    }),
                    encoding="utf-8",
                )
                events = [
                    {"job_key": "train", "event_type": "SUBMITTED", "slurm_job_id": f"1{seed}"},
                    {"job_key": "best_eval", "event_type": "SUBMITTED", "slurm_job_id": f"2{seed}"},
                    {"job_key": "train", "event_type": "COMPLETED", "exit_code": "0:0", "slurm_job_id": f"1{seed}"},
                    {"job_key": "best_eval", "event_type": "COMPLETED", "exit_code": "0:0", "slurm_job_id": f"2{seed}"},
                ]
                (fold / "jobs.jsonl").write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")
                (fold / "evaluations.json").write_text("{}\n", encoding="utf-8")


def _run(tmp_path: Path, *, permutations: int = 400, bootstrap: int = 300, faults: dict[str, str] | None = None) -> Path:
    family = analysis.load_family(FAMILY)
    root = tmp_path / "output_model"
    _write_evidence(root, family, faults=faults)
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
    assert payload["family"]["sha256"] == hashlib.sha256(FAMILY.read_bytes()).hexdigest()
    for row in payload["contrasts"]:
        assert row["subjects"] == len(SUBJECTS)  # subject clustering
        assert row["keys"] == len(SUBJECTS) * 3  # all three seed records of a subject move together
    markdown = (output / "report.md").read_text(encoding="utf-8")
    for seed in (7, 1337, 2024):
        assert f"| text_only_qwen38_27b | {seed} |" in markdown  # one reader-facing row per cell and seed
    for arm in ("ab", "01", "truefalse", "yesno", "en"):
        assert f"| text_only_qwen38_27b | {arm} |" in markdown  # mean/std summary row
    assert "unadjusted" in markdown


def test_verified_provenance_carries_the_recorded_identity(tmp_path: Path) -> None:
    output = _run(tmp_path)
    payload = json.loads((output / "analysis.json").read_text(encoding="utf-8"))
    record = payload["provenance"]["text_only_qwen38_27b|ab|7"]
    assert record["state"] == "REPORTABLE"
    assert record["label_vocab_version"] == "short_internal_ab_labels"
    assert record["model_backend"] == "qwen38"
    assert record["model_revision"].startswith("1d4bf0f2")
    assert record["manifest_hash"] == "7" * 64
    assert record["split_metadata_hash"] == "8" * 64
    assert record["predictions_sha256"] == record["recorded_predictions_sha256"]
    assert record["run_config_sha256"] == record["recorded_run_config_sha256"]
    assert record["git_commit"] == "d5" * 20
    assert record["job_ids"]["train"] and record["job_ids"]["best_eval"]


def test_verify_records_writes_a_new_payload_without_touching_the_analysis(tmp_path: Path) -> None:
    output = _run(tmp_path)
    before = (output / "analysis.json").read_bytes()
    provenance_path = tmp_path / "provenance" / "run_provenance.json"
    code = analysis.main([
        "--evidence-root", str(tmp_path / "output_model"),
        "--verify-records",
        "--provenance-output", str(provenance_path),
    ])
    assert code == 0
    payload = json.loads(provenance_path.read_text(encoding="utf-8"))
    assert payload["run_count"] == 45
    assert payload["passed"] is True
    assert set(payload["runs"]) == {key for key in payload["runs"]}
    first = payload["runs"]["text_only_qwen38_27b|ab|7"]
    assert first["subjects"] == len(SUBJECTS)
    assert first["strict_headline"]["binary_strict_macro_f1"] is not None
    assert (output / "analysis.json").read_bytes() == before  # analysis artifact untouched
    with pytest.raises(analysis.AnalysisError):
        analysis.main([
            "--evidence-root", str(tmp_path / "output_model"),
            "--verify-records",
            "--provenance-output", str(provenance_path),
        ])


@pytest.mark.parametrize(
    "fault",
    ["labels", "seed", "predictions_hash", "missing_predictions_role", "backend"],
)
def test_run_record_mismatches_are_rejected(tmp_path: Path, fault: str) -> None:
    family = analysis.load_family(FAMILY)
    root = tmp_path / "output_model"
    _write_evidence(root, family, faults={f"text_only_qwen38_27b|ab|7": fault})
    with pytest.raises(analysis.AnalysisError):
        analysis.verify_all_records(root, family)


def test_non_reportable_record_is_rejected_by_verification(tmp_path: Path) -> None:
    family = analysis.load_family(FAMILY)
    root = tmp_path / "output_model"
    _write_evidence(root, family, faults={"audio_text_qwen3omni_30b_a3b|en|2024": "state"})
    with pytest.raises(analysis.AnalysisError):
        analysis.verify_all_records(root, family)


def test_missing_run_config_is_rejected(tmp_path: Path) -> None:
    family = analysis.load_family(FAMILY)
    root = tmp_path / "output_model"
    _write_evidence(root, family)
    fold = _fold_dir(root, family, "text_only", "yesno", 7)
    (fold / "run_config.yaml").unlink()
    with pytest.raises(analysis.AnalysisError):
        analysis.verify_all_records(root, family)


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
    _write_evidence(root, family, faults={"text_only_qwen38_27b|ab|1337": "state"})
    with pytest.raises(analysis.AnalysisError):
        analysis.load_all_rows(root, family)


def test_missing_run_is_refused(tmp_path: Path) -> None:
    family = analysis.load_family(FAMILY)
    root = tmp_path / "output_model"
    _write_evidence(root, family)
    fold = _fold_dir(root, family, "audio_text", "en", 2024)
    for child in sorted(fold.rglob("*"), reverse=True):
        child.unlink() if child.is_file() else child.rmdir()
    fold.rmdir()
    with pytest.raises(analysis.AnalysisError):
        analysis.load_all_rows(root, family)


def test_missing_seed_rows_are_refused(tmp_path: Path) -> None:
    family = analysis.load_family(FAMILY)
    root = tmp_path / "output_model"
    _write_evidence(root, family)
    fold_dir = _fold_dir(root, family, "text_only", "yesno", 7)
    predictions = fold_dir / analysis.PREDICTIONS_RELATIVE
    rows = list(csv.DictReader(predictions.open(encoding="utf-8")))
    with predictions.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows[:-1])
    # Re-record the hash so the file stays consistent with its own evidence and the
    # missing-seed check is the one that has to fail.
    artifacts_path = fold_dir / "artifacts.json"
    recorded = json.loads(artifacts_path.read_text(encoding="utf-8"))
    for artifact in recorded["artifacts"]:
        if artifact["role"] == analysis.PREDICTIONS_ROLE:
            artifact["sha256"] = hashlib.sha256(predictions.read_bytes()).hexdigest()
    artifacts_path.write_text(json.dumps(recorded), encoding="utf-8")

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
