from __future__ import annotations

import unittest
import warnings

import numpy as np

from scripts import compare_e0_conditions as comparison


class ScoreModeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.subject_ids = ["100", "200"]
        self.rows = {
            "100": {
                "subject_id": "100",
                "label": 0,
                "first_token_margin": 0.0,
                "first_token_prediction": 0,
                "candidate_likelihood_margin": 0.75,
                "candidate_likelihood_prediction": 1,
                "dep_score": -1.0,
                "non_score": 0.0,
                "likelihood_prediction": 0,
            },
            "200": {
                "subject_id": "200",
                "label": 1,
                "first_token_margin": 2.0,
                "first_token_prediction": 1,
                "candidate_likelihood_margin": -0.25,
                "candidate_likelihood_prediction": 0,
                "dep_score": 3.0,
                "non_score": 1.0,
                "likelihood_prediction": 1,
            },
        }

    def test_canonical_modes_do_not_read_compatibility_aliases(self) -> None:
        _, first_scores, first_predictions = comparison._condition_arrays(
            self.rows,
            self.subject_ids,
            score_mode="first-token",
        )
        _, candidate_scores, candidate_predictions = comparison._condition_arrays(
            self.rows,
            self.subject_ids,
            score_mode="candidate",
        )
        np.testing.assert_array_equal(first_scores, [0.0, 2.0])
        np.testing.assert_array_equal(first_predictions, [0, 1])
        np.testing.assert_array_equal(candidate_scores, [0.75, -0.25])
        np.testing.assert_array_equal(candidate_predictions, [1, 0])

    def test_legacy_mode_retains_dep_non_alias_contract(self) -> None:
        _, scores, predictions = comparison._condition_arrays(
            self.rows,
            self.subject_ids,
            score_mode="legacy_alias",
        )
        np.testing.assert_array_equal(scores, [-1.0, 2.0])
        np.testing.assert_array_equal(predictions, [0, 1])

    def test_candidate_mode_accepts_pre_e0_candidate_alias_rows(self) -> None:
        legacy_rows = {
            subject_id: {
                key: value
                for key, value in row.items()
                if key
                not in {
                    "first_token_margin",
                    "first_token_prediction",
                    "candidate_likelihood_margin",
                    "candidate_likelihood_prediction",
                }
            }
            for subject_id, row in self.rows.items()
        }
        _, scores, predictions = comparison._condition_arrays(
            legacy_rows,
            self.subject_ids,
            score_mode="candidate_likelihood",
        )
        np.testing.assert_array_equal(scores, [-1.0, 2.0])
        np.testing.assert_array_equal(predictions, [0, 1])

    def test_candidate_mode_does_not_misread_first_token_aliases(self) -> None:
        rows_without_secondary = {
            subject_id: {
                key: value
                for key, value in row.items()
                if not key.startswith("candidate_likelihood_")
            }
            for subject_id, row in self.rows.items()
        }
        with self.assertRaisesRegex(ValueError, "canonical candidate_likelihood_margin"):
            comparison._condition_arrays(
                rows_without_secondary,
                self.subject_ids,
                score_mode="candidate_likelihood",
            )


class FastBootstrapTests(unittest.TestCase):
    def test_fast_bootstrap_matches_simple_sklearn_reference_with_ties(self) -> None:
        labels = np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
        # Repeated and zero-valued scores exercise AUROC/AP ties and the
        # preregistered non-positive prediction tie break.
        real_scores = np.asarray([-1.0, 0.0, 0.0, 2.0, -1.0, 2.0, 0.5, 0.5])
        perturbed_scores = np.asarray([0.0, 0.0, -1.0, 0.5, 0.5, 0.5, -1.0, 2.0])
        real_predictions = (real_scores > 0.0).astype(np.int64)
        perturbed_predictions = (perturbed_scores > 0.0).astype(np.int64)
        signed_margin_delta = (labels * 2 - 1) * (real_scores - perturbed_scores)
        repetitions = 257
        seed = 19

        fast_deltas, fast_margins = comparison._paired_bootstrap_fast(
            labels,
            real_scores,
            perturbed_scores,
            real_predictions,
            perturbed_predictions,
            signed_margin_delta,
            bootstrap_reps=repetitions,
            seed=seed,
            batch_size=17,
        )

        reference_deltas = {
            name: np.empty(repetitions, dtype=np.float64) for name in comparison.METRICS
        }
        reference_margins = np.empty(repetitions, dtype=np.float64)
        rng = np.random.default_rng(seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for repetition in range(repetitions):
                indices = rng.integers(0, labels.size, size=labels.size)
                boot_labels = labels[indices]
                real_metrics = comparison._metrics(
                    boot_labels,
                    real_scores[indices],
                    real_predictions[indices],
                )
                perturbed_metrics = comparison._metrics(
                    boot_labels,
                    perturbed_scores[indices],
                    perturbed_predictions[indices],
                )
                for name in comparison.METRICS:
                    reference_deltas[name][repetition] = (
                        real_metrics[name] - perturbed_metrics[name]
                    )
                reference_margins[repetition] = np.mean(signed_margin_delta[indices])

        for name in comparison.METRICS:
            np.testing.assert_allclose(
                fast_deltas[name],
                reference_deltas[name],
                rtol=0.0,
                atol=1e-15,
                equal_nan=True,
                err_msg=name,
            )
            fast_interval = comparison._percentile_interval(fast_deltas[name])
            reference_interval = comparison._percentile_interval(reference_deltas[name])
            self.assertEqual(
                fast_interval["valid_replicates"],
                reference_interval["valid_replicates"],
            )
            for field in ("estimate", "ci_95_low", "ci_95_high"):
                self.assertAlmostEqual(fast_interval[field], reference_interval[field], places=15)
        np.testing.assert_allclose(fast_margins, reference_margins, rtol=0.0, atol=0.0)


if __name__ == "__main__":
    unittest.main()
