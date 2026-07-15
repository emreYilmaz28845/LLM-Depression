from __future__ import annotations

import unittest

import numpy as np

from src.baselines.acoustic_crossfold import (
    FeatureBundle,
    _assert_exact_ids,
    build_fold_assignments,
    derangement,
    numeric_chunk_number,
    paired_bootstrap_differences,
    subject_features,
    validate_derangement,
    validate_fold_assignments,
    validate_oof_coverage,
)


def _metadata() -> dict[str, dict]:
    metadata: dict[str, dict] = {}
    # Fifty development subjects provide ten subjects per stratified outer
    # holdout and enough examples for every four-fold inner split.
    for index in range(50):
        subject_id = str(300 + index)
        metadata[subject_id] = {
            "subject_id": subject_id,
            "original_partition": "train" if index < 38 else "val",
            "label": index % 2,
            "samples": [],
        }
    for index in range(10):
        subject_id = str(500 + index)
        metadata[subject_id] = {
            "subject_id": subject_id,
            "original_partition": "test",
            "label": index % 2,
            "samples": [],
        }
    return metadata


class FoldTests(unittest.TestCase):
    def test_folds_are_reproducible_stratified_and_subject_isolated(self):
        metadata = _metadata()
        first = build_fold_assignments(metadata)
        second = build_fold_assignments(metadata)
        self.assertEqual(first, second)
        validate_fold_assignments(first, metadata)

        test_ids = set(first["locked_test_subject_ids"])
        holdout_ids: list[str] = []
        for outer in first["folds"]:
            train = set(outer["train_subject_ids"])
            holdout = set(outer["holdout_subject_ids"])
            self.assertFalse(train & holdout)
            self.assertFalse((train | holdout) & test_ids)
            self.assertEqual(
                {metadata[subject_id]["label"] for subject_id in holdout},
                {0, 1},
            )
            holdout_ids.extend(outer["holdout_subject_ids"])
            for inner in outer["inner_folds"]:
                inner_train = set(inner["train_subject_ids"])
                inner_val = set(inner["validation_subject_ids"])
                self.assertFalse(inner_train & inner_val)
                self.assertFalse((inner_train | inner_val) & (holdout | test_ids))
        self.assertEqual(len(holdout_ids), len(set(holdout_ids)))
        self.assertEqual(set(holdout_ids), set(first["development_subject_ids"]))


class SelectionAndShuffleTests(unittest.TestCase):
    def test_trailing_chunk_numbers_are_numeric_not_lexical(self):
        sample_ids = [f"308_segment_{index}" for index in range(1, 16)]
        observed = sorted(sample_ids, key=numeric_chunk_number)
        self.assertEqual([numeric_chunk_number(value) for value in observed], list(range(1, 16)))
        self.assertLess(observed.index("308_segment_2"), observed.index("308_segment_10"))

    def test_derangement_is_deterministic_bijective_and_has_no_fixed_points(self):
        subject_ids = [str(index) for index in range(20)]
        first = derangement(subject_ids, seed=20260717)
        second = derangement(subject_ids, seed=20260717)
        self.assertEqual(first, second)
        validate_derangement(subject_ids, first)
        self.assertTrue(all(target != source for target, source in first.items()))


class FailClosedFeatureTests(unittest.TestCase):
    def _selection_and_bundle(self):
        samples = [
            {
                "sample_id": f"300_random_segment_{index}",
                "selected_position": position,
            }
            for position, index in enumerate((1, 4, 7, 10))
        ]
        selection = {
            "subjects": {
                "300": {
                    "subject_id": "300",
                    "original_partition": "train",
                    "label": 0,
                    "samples": samples,
                }
            }
        }
        vectors = {
            sample["sample_id"]: np.asarray([position, position + 1], dtype=np.float64)
            for position, sample in enumerate(samples)
        }
        bundle = FeatureBundle(
            family="fixture",
            chunk_dimension=2,
            feature_names=["a", "b"],
            vectors=vectors,
            validation={},
        )
        return selection, bundle

    def test_subject_pooling_requires_complete_finite_consistent_features(self):
        selection, bundle = self._selection_and_bundle()
        pooled = subject_features(bundle, selection)
        self.assertEqual(pooled["300"].shape, (4,))

        missing = dict(bundle.vectors)
        missing.pop("300_random_segment_10")
        with self.assertRaisesRegex(ValueError, "Missing fixture feature"):
            subject_features(
                FeatureBundle("fixture", 2, ["a", "b"], missing, {}),
                selection,
            )

        nonfinite = {key: value.copy() for key, value in bundle.vectors.items()}
        nonfinite["300_random_segment_10"][0] = np.nan
        with self.assertRaisesRegex(ValueError, "Invalid fixture subject feature"):
            subject_features(
                FeatureBundle("fixture", 2, ["a", "b"], nonfinite, {}),
                selection,
            )

        wrong_dimension = {key: value.copy() for key, value in bundle.vectors.items()}
        wrong_dimension["300_random_segment_10"] = np.ones(3)
        with self.assertRaises(ValueError):
            subject_features(
                FeatureBundle("fixture", 2, ["a", "b"], wrong_dimension, {}),
                selection,
            )

    def test_duplicate_missing_and_extra_ids_fail_closed(self):
        expected = {"a", "b"}
        with self.assertRaisesRegex(ValueError, "duplicated"):
            _assert_exact_ids(["a", "a", "b"], expected, "fixture")
        with self.assertRaisesRegex(ValueError, "differ"):
            _assert_exact_ids(["a", "c"], expected, "fixture")


class OofAndStatisticsTests(unittest.TestCase):
    def test_oof_coverage_requires_exactly_one_prediction(self):
        rows = [{"subject_id": "a"}, {"subject_id": "b"}]
        validate_oof_coverage(rows, expected_subject_ids={"a", "b"}, context="fixture")
        with self.assertRaisesRegex(ValueError, "exactly one"):
            validate_oof_coverage(
                rows + [{"subject_id": "a"}],
                expected_subject_ids={"a", "b"},
                context="fixture",
            )

    def test_paired_subject_bootstrap_is_reproducible(self):
        y = np.tile(np.asarray([0, 1], dtype=np.int64), 20)
        real = np.where(y == 1, 0.8, 0.2).astype(np.float64)
        shuffled = np.vstack(
            [np.roll(real, shift) for shift in (1, 3, 5, 7)]
        )
        first = paired_bootstrap_differences(y, real, shuffled, repeats=50, seed=13)
        second = paired_bootstrap_differences(y, real, shuffled, repeats=50, seed=13)
        self.assertEqual(first, second)
        self.assertGreater(
            first["point_difference_vs_mean_shuffled_metric"]["auroc"],
            0,
        )


if __name__ == "__main__":
    unittest.main()
