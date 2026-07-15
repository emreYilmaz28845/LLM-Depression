from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts import summarize_e0_views as aggregation


class EightViewAggregationTests(unittest.TestCase):
    SUBJECT_LABELS = {"100": 0, "200": 1, "300": 0, "400": 1}
    CONDITION_BASE = {
        "real": (-1.5, 1.25, -0.75, 2.0),
        "silence": (-0.5, 0.25, 0.25, -0.5),
        "audio_shuffle": (-0.25, 0.5, 0.75, -0.25),
        "audio_shuffle_same_class": (-1.0, 0.75, -0.5, 1.25),
        "transcript_shuffle": (0.75, -0.75, 0.5, -0.5),
        "audio_only_real": (-0.75, 0.75, -0.25, 0.5),
        "audio_only_silence": (-0.25, 0.25, 0.25, -0.25),
        "audio_only_shuffle": (0.5, -0.5, 0.75, -0.75),
    }

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.mapping = self._make_views()

    def _make_views(self) -> dict[str, Path]:
        mapping = {}
        for view_index in range(aggregation.EXPECTED_VIEW_COUNT):
            view_key = f"view_{view_index}"
            root = self.base / view_key
            mapping[view_key] = root
            view_offset = (view_index - 3.5) * 0.1
            for condition in aggregation.CONDITIONS:
                path = root / condition / "predictions_subject_level.jsonl"
                path.parent.mkdir(parents=True)
                rows = []
                for subject_index, (subject_id, label) in enumerate(
                    self.SUBJECT_LABELS.items()
                ):
                    first_margin = float(
                        self.CONDITION_BASE[condition][subject_index] + view_offset
                    )
                    candidate_margin = float(first_margin * 0.5 + view_offset * 0.25)
                    rows.append(
                        {
                            "schema_version": 1,
                            "condition": condition,
                            "view_id": f"deterministic_{view_index}",
                            "view_index": view_index,
                            "subject_id": subject_id,
                            "label": label,
                            "first_token_margin": first_margin,
                            "first_token_prediction": int(first_margin > 0.0),
                            "candidate_likelihood_margin": candidate_margin,
                            "candidate_likelihood_prediction": int(
                                candidate_margin > 0.0
                            ),
                        }
                    )
                path.write_text(
                    "".join(json.dumps(row) + "\n" for row in rows),
                    encoding="utf-8",
                )
        return mapping

    def _prediction_path(self, view: str, condition: str) -> Path:
        return (
            self.mapping[view]
            / condition
            / "predictions_subject_level.jsonl"
        )

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    @staticmethod
    def _write_jsonl(path: Path, rows: list[dict]) -> None:
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )

    def test_emits_metrics_mean_predictions_comparisons_and_hashes(self) -> None:
        output_dir = self.base / "aggregate"
        result = aggregation.aggregate_views(
            self.mapping,
            output_dir,
            seed=29,
            invocation=["synthetic-test"],
        )

        self.assertEqual(result["n_views"], 8)
        self.assertEqual(result["n_subjects"], 4)
        self.assertEqual(result["n_conditions"], 8)
        self.assertEqual(result["n_paired_reports"], 12)
        self.assertEqual(result["bootstrap_repetitions"], 10_000)

        per_view = json.loads(
            (output_dir / "per_view_metrics.json").read_text(encoding="utf-8")
        )
        self.assertEqual(set(per_view["views"]), set(self.mapping))
        self.assertEqual(
            set(per_view["views"]["view_0"]["conditions"]["real"]),
            {"first_token", "candidate_likelihood"},
        )

        real_mean_path = (
            output_dir
            / "mean_predictions"
            / "real"
            / "predictions_subject_level.jsonl"
        )
        real_rows = self._read_jsonl(real_mean_path)
        self.assertEqual(len(real_rows), 4)
        self.assertAlmostEqual(real_rows[0]["first_token_margin"], -1.5)
        self.assertEqual(real_rows[0]["first_token_prediction"], 0)
        self.assertEqual(len(real_rows[0]["first_token_view_margins"]), 8)

        comparison_path = (
            output_dir
            / "comparisons"
            / "paired_audio_only_real_vs_audio_only_silence_first_token.json"
        )
        report = json.loads(comparison_path.read_text(encoding="utf-8"))
        self.assertEqual(report["comparison"], "audio_only_real_minus_audio_only_silence")
        self.assertEqual(report["reference_condition"], "audio_only_real")
        self.assertEqual(report["bootstrap"]["repetitions"], 10_000)
        self.assertEqual(report["bootstrap"]["seed"], 29)
        self.assertEqual(
            report["paired"]["correct_class_margin_delta"]["valid_replicates"],
            10_000,
        )
        self.assertEqual(len(list((output_dir / "comparisons").glob("*.json"))), 12)
        self.assertEqual(len(list((output_dir / "comparisons").glob("*.csv"))), 12)

        provenance = json.loads(
            (output_dir / "provenance.json").read_text(encoding="utf-8")
        )
        source_path = self._prediction_path("view_0", "real")
        expected_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
        self.assertEqual(
            provenance["inputs"]["view_0"]["conditions"]["real"]["sha256"],
            expected_hash,
        )
        recorded_mean = provenance["outputs"]["mean_predictions"]["real"]
        self.assertEqual(
            recorded_mean["sha256"],
            hashlib.sha256(real_mean_path.read_bytes()).hexdigest(),
        )

        with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
            aggregation.aggregate_views(self.mapping, output_dir)

    def test_requires_exactly_eight_views(self) -> None:
        seven = dict(list(self.mapping.items())[:7])
        with self.assertRaisesRegex(ValueError, "Exactly 8 views"):
            aggregation.aggregate_views(seven, self.base / "aggregate")

    def test_rejects_subject_and_label_mismatches(self) -> None:
        subject_path = self._prediction_path("view_7", "silence")
        rows = self._read_jsonl(subject_path)
        rows[-1]["subject_id"] = "unexpected"
        self._write_jsonl(subject_path, rows)
        with self.assertRaisesRegex(ValueError, "Subject IDs differ"):
            aggregation.aggregate_views(self.mapping, self.base / "subject_failure")

        # Restore the subject universe, then introduce a label mismatch.
        rows[-1]["subject_id"] = "400"
        rows[-1]["label"] = 0
        self._write_jsonl(subject_path, rows)
        with self.assertRaisesRegex(ValueError, "Labels differ"):
            aggregation.aggregate_views(self.mapping, self.base / "label_failure")

    def test_rejects_missing_or_inconsistent_score_schema(self) -> None:
        path = self._prediction_path("view_3", "audio_shuffle")
        rows = self._read_jsonl(path)
        del rows[0]["candidate_likelihood_margin"]
        self._write_jsonl(path, rows)
        with self.assertRaisesRegex(ValueError, "Canonical score schema is incomplete"):
            aggregation.aggregate_views(self.mapping, self.base / "aggregate")

    def test_rejects_duplicate_reported_views(self) -> None:
        for condition in aggregation.CONDITIONS:
            path = self._prediction_path("view_7", condition)
            rows = self._read_jsonl(path)
            for row in rows:
                row["view_id"] = "deterministic_6"
                row["view_index"] = 6
            self._write_jsonl(path, rows)
        with self.assertRaisesRegex(ValueError, "eight distinct row-level"):
            aggregation.aggregate_views(self.mapping, self.base / "aggregate")


class ViewMappingTests(unittest.TestCase):
    def test_json_mapping_resolves_relative_to_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = {f"v{i}": f"views/{i}" for i in range(8)}
            manifest = root / "views.json"
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            mapping = aggregation.load_view_mapping(view_map_path=manifest)
            self.assertEqual(mapping["v0"], (root / "views" / "0").resolve())


if __name__ == "__main__":
    unittest.main()
