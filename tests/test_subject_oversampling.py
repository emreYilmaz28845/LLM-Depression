from __future__ import annotations

import unittest
import tempfile
import importlib.util
import json
from pathlib import Path
from unittest import mock

import numpy as np

from baselines import qwen_hidden_oversampling_screen as screen
from baselines import qwen_hidden_xgb_optuna as optuna_xgb
from scripts import build_turkish_oversampling_optuna_matrix as optuna_matrix
from scripts import build_turkish_oversampling_qwen_matrix as qwen_matrix
from scripts.summarize_turkish_oversampling_qwen_pilot import _metrics
from src.sampling import build_subject_oversampling, validate_oversampling_ratio


def _rows(negative_subjects: int = 4, positive_subjects: int = 8):
    rows = []
    for label, count in ((0, negative_subjects), (1, positive_subjects)):
        for subject_index in range(count):
            subject_id = f"y{label}-s{subject_index}"
            response_count = subject_index % 3 + 1
            for response_index in range(response_count):
                rows.append(
                    {
                        "sample_id": f"{subject_id}-r{response_index}",
                        "subject_id": subject_id,
                        "label": label,
                    }
                )
    return rows


class SubjectOversamplingTests(unittest.TestCase):
    def test_pilot_metrics_are_json_serializable(self):
        metrics = _metrics(
            [
                {"label": 0, "prediction": 0},
                {"label": 1, "prediction": 1},
            ]
        )

        json.dumps(metrics)
        self.assertTrue(all(type(value) is float for value in metrics.values()))

    def test_exact_ratio_retains_all_subjects_and_duplicates_only_minority_groups(self):
        rows = _rows()
        result = build_subject_oversampling(rows, ratio=0.75, seed=1337)
        audit = result.audit
        self.assertEqual(audit["final_subject_occurrence_counts_by_class"], {"0": 6, "1": 8})
        self.assertEqual(result.indices[: len(rows)], tuple(range(len(rows))))
        self.assertTrue(
            all(
                subject_id.startswith("y0-") or multiplicity == 1
                for subject_id, multiplicity in audit[
                    "duplicate_multiplicity_by_subject"
                ].items()
            )
        )
        for subject_id, multiplicity in audit["duplicate_multiplicity_by_subject"].items():
            source_count = sum(row["subject_id"] == subject_id for row in rows)
            expanded_count = sum(rows[index]["subject_id"] == subject_id for index in result.indices)
            self.assertEqual(expanded_count, source_count * multiplicity)

    def test_same_seed_is_identical_and_different_seed_varies(self):
        rows = _rows(negative_subjects=3, positive_subjects=10)
        first = build_subject_oversampling(rows, ratio=1.0, seed=7)
        repeated = build_subject_oversampling(rows, ratio=1.0, seed=7)
        alternate = build_subject_oversampling(rows, ratio=1.0, seed=2024)
        self.assertEqual(first.indices, repeated.indices)
        self.assertEqual(first.audit, repeated.audit)
        self.assertNotEqual(
            first.audit["sampled_additional_subject_ids"],
            alternate.audit["sampled_additional_subject_ids"],
        )

    def test_validation_and_evaluation_are_only_fingerprinted(self):
        rows = _rows()
        validation = [{"sample_id": "v", "subject_id": "v", "label": 0}]
        evaluation = [{"sample_id": "e", "subject_id": "e", "label": 1}]
        original_validation = list(validation)
        original_evaluation = list(evaluation)
        result = build_subject_oversampling(
            rows,
            ratio=1.0,
            seed=1337,
            validation_rows=validation,
            evaluation_rows=evaluation,
        )
        self.assertEqual(validation, original_validation)
        self.assertEqual(evaluation, original_evaluation)
        self.assertTrue(result.audit["validation_indices_untouched"])
        self.assertTrue(result.audit["evaluation_indices_untouched"])

    def test_rejects_inconsistent_subject_labels(self):
        rows = _rows()
        rows.append({"sample_id": "bad", "subject_id": "y0-s0", "label": 1})
        with self.assertRaisesRegex(ValueError, "Inconsistent labels"):
            build_subject_oversampling(rows, ratio=1.0, seed=1337)

    def test_rejects_one_class_and_wrong_minority(self):
        with self.assertRaisesRegex(ValueError, "two-class"):
            build_subject_oversampling(_rows(negative_subjects=0), ratio=1.0, seed=1337)
        with self.assertRaisesRegex(ValueError, "Expected minority label 0"):
            build_subject_oversampling(
                _rows(negative_subjects=8, positive_subjects=4),
                ratio=1.0,
                seed=1337,
            )

    def test_rejects_invalid_or_missing_ratio(self):
        for ratio in (None, 0.5, 1.1):
            with self.assertRaises(ValueError):
                validate_oversampling_ratio(ratio)

    @unittest.skipIf(importlib.util.find_spec("torch") is None, "PyTorch is not installed")
    def test_qwen_weighted_sampler_behavior_is_unchanged(self):
        from src.train import _build_weighted_train_sampler

        rows = _rows()
        config = {"seed": 1337, "training": {"class_balance": "weighted_sampler"}}
        sampler = _build_weighted_train_sampler(rows, config)
        self.assertEqual(sampler.num_samples, len(rows))
        self.assertTrue(sampler.replacement)

    @unittest.skipIf(importlib.util.find_spec("torch") is None, "PyTorch is not installed")
    def test_qwen_output_collision_is_rejected(self):
        from src.train import _apply_subject_oversampling

        rows = _rows()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            config = {
                "seed": 1337,
                "training": {
                    "class_balance": "minority_subject_oversample",
                    "oversampling_ratio": 0.75,
                    "oversampling_seed": 7,
                },
            }
            _apply_subject_oversampling(rows, [], [], config, output)
            config["training"]["oversampling_seed"] = 2024
            with self.assertRaisesRegex(ValueError, "Incompatible sampling"):
                _apply_subject_oversampling(rows, [], [], config, output)

    @unittest.skipIf(importlib.util.find_spec("torch") is None, "PyTorch is not installed")
    def test_qwen_smoke_limit_preserves_minority_status_for_oversampling(self):
        from src.train import _limit_subject_ids_for_smoke

        labels = {f"n{i}": 0 for i in range(4)}
        labels.update({f"p{i}": 1 for i in range(8)})
        selected = _limit_subject_ids_for_smoke(
            list(labels),
            labels,
            limit=6,
            seed=1337,
            preserve_class_ratio=True,
        )
        counts = {label: sum(labels[item] == label for item in selected) for label in (0, 1)}
        self.assertEqual(counts, {0: 2, 1: 4})

    def test_optuna_sampling_search_fixes_scale_pos_weight(self):
        search_space = optuna_xgb.resolved_oversampling_search_space(
            "standard_d6", "minority_subject_oversample"
        )
        fixed = optuna_xgb.fixed_xgb_params(
            1337, 1, sampling_mode="minority_subject_oversample"
        )
        self.assertNotIn("scale_pos_weight", search_space)
        self.assertEqual(fixed["scale_pos_weight"], 1.0)
        self.assertIn(
            "scale_pos_weight",
            optuna_xgb.resolved_oversampling_search_space("standard_d6", "legacy"),
        )


class _FakeClassifier:
    def fit(self, x, y):
        self.fit_rows = len(x)
        return self

    def predict_proba(self, x):
        probability = 1.0 / (1.0 + np.exp(-np.asarray(x)[:, 0]))
        return np.column_stack([1.0 - probability, probability])


class HiddenScreenTests(unittest.TestCase):
    @unittest.skipIf(importlib.util.find_spec("sklearn") is None, "scikit-learn is not installed")
    def test_screen_uses_outer_train_only_and_writes_all_fit_audits(self):
        rows = _rows(negative_subjects=6, positive_subjects=12)
        vectors = np.asarray(
            [
                [-2.0 if row["label"] == 0 else 2.0, float(index % 3)]
                for index, row in enumerate(rows)
            ],
            dtype=np.float32,
        )
        metadata = {
            "dataset": "turkish",
            "input_modality": "audio_text",
            "condition": "audio_text",
            "fold": 0,
        }

        def load_partition(_cache, name):
            self.assertEqual(name, "outer_train")
            return vectors, rows

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "hidden_os_screen"
            with (
                mock.patch.object(screen, "_load_partition", side_effect=load_partition),
                mock.patch.object(screen, "_classifier", return_value=_FakeClassifier()),
                mock.patch.object(screen, "read_json", return_value=metadata),
            ):
                summaries = screen.run_screen(
                    cache_dir=Path("unused"),
                    output_dir=output,
                    experiment_id="hidden_os_screen",
                )
            self.assertEqual(len(summaries), 14)
            completion = screen.read_json(output / "completion.json")
            self.assertEqual(completion["observed_sampling_audits"], 42)
            self.assertFalse(completion["final_eval_loaded"])


class MatrixBuilderTests(unittest.TestCase):
    def test_stage3_matrix_has_exact_control_and_sampling_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = root / "screen.json"
            summary.write_text(
                json.dumps(
                    {
                        "gate_passed": True,
                        "outer_evaluation_metrics_inspected": False,
                        "selected_ratio": 0.75,
                    }
                ),
                encoding="utf-8",
            )
            payload = optuna_matrix.build(
                Path("configs/features/turkish_oversampling_hidden_matrix.yaml"),
                summary,
            )
        self.assertEqual(len(payload["jobs"]), 60)
        self.assertEqual(
            sum(job["sampling_mode"] == "none" for job in payload["jobs"]), 15
        )
        self.assertEqual(
            sum(
                job["sampling_mode"] == "minority_subject_oversample"
                for job in payload["jobs"]
            ),
            45,
        )

    def test_qwen_pilot_selection_obeys_declared_tie_order(self):
        with tempfile.TemporaryDirectory() as directory:
            summary = Path(directory) / "optuna.json"
            summary.write_text(
                json.dumps(
                    {
                        "proceed_to_qwen": True,
                        "selected_ratio": 1.0,
                        "qualifying_modalities": ["audio_only", "text_only"],
                        "decisions": [
                            {"condition": "audio_only", "mean_macro_f1_gain": 0.03},
                            {"condition": "text_only", "mean_macro_f1_gain": 0.03},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            payload = qwen_matrix.build("pilot", summary, None)
        self.assertEqual(payload["selected_modality"], "audio_only")
        self.assertEqual(len(payload["jobs"]), 4)


if __name__ == "__main__":
    unittest.main()
