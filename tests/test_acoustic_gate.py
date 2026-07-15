from __future__ import annotations

import unittest

from src.baselines.acoustic_gate import evaluate_development_candidate


class DevelopmentGateTests(unittest.TestCase):
    def test_candidate_passes_only_when_every_necessary_condition_passes(self):
        passed = evaluate_development_candidate(
            name="fixture",
            pooled_auroc=0.6,
            auroc_delta_interval={"lower_2.5pct": 0.01, "upper_97.5pct": 0.2},
            balanced_delta_interval={"lower_2.5pct": 0.02, "upper_97.5pct": 0.2},
            auroc_positive_folds=4,
            balanced_positive_folds=4,
            auroc_positive_seeds=4,
            balanced_positive_seeds=4,
        )
        self.assertTrue(passed["development_gate_pass"])
        self.assertEqual(passed["failed_criteria"], [])

        failed = evaluate_development_candidate(
            name="fixture",
            pooled_auroc=0.6,
            auroc_delta_interval={"lower_2.5pct": -0.01, "upper_97.5pct": 0.2},
            balanced_delta_interval={"lower_2.5pct": 0.0, "upper_97.5pct": 0.2},
            auroc_positive_folds=4,
            balanced_positive_folds=4,
            auroc_positive_seeds=5,
            balanced_positive_seeds=5,
        )
        self.assertFalse(failed["development_gate_pass"])
        self.assertIn("auroc_delta_ci_excludes_zero_positive", failed["failed_criteria"])
        self.assertIn(
            "balanced_accuracy_delta_ci_excludes_zero_positive",
            failed["failed_criteria"],
        )

    def test_linear_candidate_omits_seed_criterion(self):
        result = evaluate_development_candidate(
            name="linear",
            pooled_auroc=0.6,
            auroc_delta_interval={"lower_2.5pct": 0.01, "upper_97.5pct": 0.2},
            balanced_delta_interval={"lower_2.5pct": 0.01, "upper_97.5pct": 0.2},
            auroc_positive_folds=4,
            balanced_positive_folds=4,
            auroc_positive_seeds=None,
            balanced_positive_seeds=None,
        )
        self.assertTrue(result["development_gate_pass"])
        self.assertFalse(any("seeds" in key for key in result["criteria"]))


if __name__ == "__main__":
    unittest.main()
