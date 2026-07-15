from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.baselines.egemaps_ceiling import (
    EXPECTED_EGEMAPS_DIMENSION,
    aggregate_subject_features,
    evaluate_binary,
    fit_final_linear_model,
    parse_opensmile_csv,
    run_shuffled_audio_control,
    select_equal_chunks,
    validate_inputs,
)


class InputValidationTests(unittest.TestCase):
    def _fixture(self):
        partitions = []
        manifest = []
        subject_number = 1
        for partition in ("train", "val", "test"):
            for label in (0, 1):
                subject_id = str(subject_number)
                subject_number += 1
                partitions.append(
                    {"subject_id": subject_id, "partition": partition, "label": label}
                )
                for chunk in (1, 2):
                    manifest.append(
                        {
                            "subject_id": subject_id,
                            "sample_id": f"{subject_id}_segment_{chunk}",
                            "audio_path": f"/not/needed/{subject_id}_{chunk}.wav",
                            "split_original": partition,
                            "label": label,
                        }
                    )
        return manifest, partitions

    def test_validates_disjoint_subject_membership(self):
        manifest, partitions = self._fixture()
        rows, metadata = validate_inputs(manifest, partitions, require_audio=False)
        self.assertEqual(len(rows), 12)
        self.assertEqual(len(metadata), 6)

    def test_rejects_duplicate_partition_subject(self):
        manifest, partitions = self._fixture()
        partitions.append(dict(partitions[0]))
        with self.assertRaisesRegex(ValueError, "more than once"):
            validate_inputs(manifest, partitions, require_audio=False)


class ChunkSelectionTests(unittest.TestCase):
    def test_fixed_four_are_evenly_spaced_in_numeric_order(self):
        rows = []
        for subject_id, label, count in (("10", 0, 10), ("11", 1, 15)):
            for chunk in range(1, count + 1):
                kind = "random_segment" if label == 0 else "segment"
                rows.append(
                    {
                        "subject_id": subject_id,
                        "sample_id": f"{subject_id}_{kind}_{chunk}",
                        "audio_path": f"/{subject_id}_{chunk}.wav",
                        "split_original": "train",
                        "label": label,
                    }
                )
        selected, audit = select_equal_chunks(list(reversed(rows)), 4)
        by_subject = {
            subject_id: [row["sample_id"] for row in selected if row["subject_id"] == subject_id]
            for subject_id in ("10", "11")
        }
        self.assertEqual(
            by_subject["10"],
            ["10_random_segment_1", "10_random_segment_4", "10_random_segment_7", "10_random_segment_10"],
        )
        self.assertEqual(
            by_subject["11"],
            ["11_segment_1", "11_segment_6", "11_segment_10", "11_segment_15"],
        )
        self.assertEqual(audit["resolved_chunks_per_subject"], 4)
        self.assertEqual(audit["selected_zero_based_positions_by_original_chunk_count"]["10"], [0, 3, 6, 9])
        self.assertEqual(audit["selected_zero_based_positions_by_original_chunk_count"]["15"], [0, 5, 9, 14])


class FeatureTests(unittest.TestCase):
    def test_parse_expected_88_dimensional_csv(self):
        names = [f"feature_{index}" for index in range(EXPECTED_EGEMAPS_DIMENSION)]
        values = [str(index / 10) for index in range(EXPECTED_EGEMAPS_DIMENSION)]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "features.csv"
            path.write_text(
                ";".join(["name", "frameTime", *names])
                + "\n"
                + ";".join(["'unknown'", "0.000000", *values])
                + "\n",
                encoding="utf-8",
            )
            observed_names, vector = parse_opensmile_csv(path)
        self.assertEqual(observed_names, names)
        np.testing.assert_allclose(vector, np.asarray(values, dtype=float))

    def test_subject_pooling_produces_one_mean_std_row(self):
        rows = [
            {"subject_id": "1", "split_original": "train", "label": 0},
            {"subject_id": "1", "split_original": "train", "label": 0},
            {"subject_id": "2", "split_original": "test", "label": 1},
            {"subject_id": "2", "split_original": "test", "label": 1},
        ]
        chunks = np.asarray([[1.0, 2.0], [3.0, 6.0], [10.0, 20.0], [14.0, 28.0]])
        subject_rows, pooled, names = aggregate_subject_features(rows, chunks, ["a", "b"])
        self.assertEqual([row["subject_id"] for row in subject_rows], ["1", "2"])
        np.testing.assert_allclose(pooled[0], [2.0, 4.0, 1.0, 2.0])
        np.testing.assert_allclose(pooled[1], [12.0, 24.0, 2.0, 4.0])
        self.assertEqual(names, ["chunk_mean__a", "chunk_mean__b", "chunk_std__a", "chunk_std__b"])


class ModelingTests(unittest.TestCase):
    def test_linear_fit_and_shuffle_are_reproducible(self):
        rng = np.random.default_rng(7)

        def partition(size):
            labels = np.tile([0, 1], size // 2)
            features = rng.normal(scale=0.5, size=(size, 6))
            features[:, 0] += 2.5 * labels
            return features, labels

        x_train, y_train = partition(60)
        x_val, y_val = partition(30)
        x_test, y_test = partition(30)
        model, selected_c, records = fit_final_linear_model(
            x_train,
            y_train,
            x_val,
            y_val,
            c_grid=[0.01, 0.1, 1.0],
            seed=13,
        )
        metrics = evaluate_binary(y_test, model.predict_proba(x_test)[:, 1])
        self.assertGreater(metrics["auroc"], 0.95)
        self.assertIn(selected_c, {0.01, 0.1, 1.0})
        self.assertEqual(len(records), 3)

        first_records, first_summary = run_shuffled_audio_control(
            x_train,
            y_train,
            x_val,
            y_val,
            x_test,
            y_test,
            c_grid=[0.1, 1.0],
            repeats=4,
            seed=99,
            real_metrics=metrics,
        )
        second_records, second_summary = run_shuffled_audio_control(
            x_train,
            y_train,
            x_val,
            y_val,
            x_test,
            y_test,
            c_grid=[0.1, 1.0],
            repeats=4,
            seed=99,
            real_metrics=metrics,
        )
        self.assertEqual(first_records, second_records)
        self.assertEqual(first_summary, second_summary)


if __name__ == "__main__":
    unittest.main()
