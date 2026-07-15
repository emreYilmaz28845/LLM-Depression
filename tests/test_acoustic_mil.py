from __future__ import annotations

import unittest

import numpy as np
import torch

from src.baselines.acoustic_mil import (
    ARCHITECTURES,
    INSTANCE_DIMENSION,
    MeanPoolingMLP,
    GatedAttentionMIL,
    architecture_model,
    fit_standardizer,
    load_bags,
    paired_seed_subject_bootstrap,
    train_early_stopping,
    trainable_parameter_counts,
)


class ArchitectureTests(unittest.TestCase):
    def test_models_emit_one_subject_logit_and_normalized_chunk_weights(self):
        bags = torch.randn(3, 4, INSTANCE_DIMENSION)
        for model_class in (MeanPoolingMLP, GatedAttentionMIL):
            model = model_class().eval()
            with torch.inference_mode():
                logits, attention = model(bags)
            self.assertEqual(logits.shape, (3,))
            self.assertEqual(attention.shape, (3, 4))
            torch.testing.assert_close(attention.sum(dim=1), torch.ones(3))
            self.assertTrue(torch.isfinite(logits).all())

    def test_both_models_are_substantially_below_one_million_parameters(self):
        counts = trainable_parameter_counts()
        self.assertEqual(set(counts), set(ARCHITECTURES))
        self.assertGreater(counts["mean_pooling"], 200_000)
        self.assertGreater(counts["gated_attention"], counts["mean_pooling"])
        self.assertTrue(all(count < 1_000_000 for count in counts.values()))

    def test_unknown_architecture_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "Unknown N2 architecture"):
            architecture_model("not_a_model")


class BagAndStandardizationTests(unittest.TestCase):
    def test_load_bags_preserves_frozen_selected_position_order(self):
        samples = [
            {"sample_id": f"s_{index}", "selected_position": index}
            for index in range(4)
        ]
        selection = {
            "subjects": {
                "s": {
                    "label": 0,
                    "original_partition": "train",
                    "samples": list(reversed(samples)),
                }
            }
        }
        vectors = {
            f"s_{index}": np.full(INSTANCE_DIMENSION, index, dtype=np.float32)
            for index in range(4)
        }
        bag = load_bags(selection, vectors)["s"]
        self.assertEqual(bag.shape, (4, INSTANCE_DIMENSION))
        np.testing.assert_array_equal(bag[:, 0], np.arange(4))

    def test_missing_nonfinite_and_wrong_dimension_bags_fail_closed(self):
        selection = {
            "subjects": {
                "s": {
                    "label": 0,
                    "original_partition": "train",
                    "samples": [
                        {"sample_id": f"s_{index}", "selected_position": index}
                        for index in range(4)
                    ],
                }
            }
        }
        vectors = {
            f"s_{index}": np.zeros(INSTANCE_DIMENSION, dtype=np.float32)
            for index in range(4)
        }
        missing = dict(vectors)
        missing.pop("s_3")
        with self.assertRaisesRegex(ValueError, "Missing frozen WavLM"):
            load_bags(selection, missing)
        nonfinite = {key: value.copy() for key, value in vectors.items()}
        nonfinite["s_3"][0] = np.nan
        with self.assertRaisesRegex(ValueError, "Invalid WavLM bag"):
            load_bags(selection, nonfinite)
        wrong = {key: value.copy() for key, value in vectors.items()}
        wrong["s_3"] = np.zeros(INSTANCE_DIMENSION + 1, dtype=np.float32)
        with self.assertRaises(ValueError):
            load_bags(selection, wrong)

    def test_standardizer_uses_only_passed_training_bags_and_handles_constants(self):
        train = np.zeros((2, 4, INSTANCE_DIMENSION), dtype=np.float32)
        train[1] = 2.0
        standardizer = fit_standardizer(train)
        np.testing.assert_allclose(standardizer.mean, 1.0)
        np.testing.assert_allclose(standardizer.scale, 1.0)
        holdout = np.full((1, 4, INSTANCE_DIMENSION), 100.0, dtype=np.float32)
        transformed = standardizer.transform(holdout)
        np.testing.assert_allclose(transformed, 99.0)


class TrainingAndStatisticsTests(unittest.TestCase):
    def test_cpu_training_is_reproducible_and_subject_level(self):
        rng = np.random.default_rng(8)
        train_y = np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
        val_y = np.asarray([0, 1, 0, 1], dtype=np.int64)
        train = rng.normal(size=(8, 4, INSTANCE_DIMENSION)).astype(np.float32)
        validation = rng.normal(size=(4, 4, INSTANCE_DIMENSION)).astype(np.float32)
        train[:, :, 0] += train_y[:, None] * 2
        validation[:, :, 0] += val_y[:, None] * 2
        kwargs = dict(
            architecture="mean_pooling",
            train_bags=train,
            train_labels=train_y,
            validation_bags=validation,
            validation_labels=val_y,
            learning_rate=3e-4,
            weight_decay=0.1,
            seed=17,
            device=torch.device("cpu"),
        )
        first = train_early_stopping(**kwargs)
        second = train_early_stopping(**kwargs)
        np.testing.assert_array_equal(first["probabilities"], second["probabilities"])
        np.testing.assert_array_equal(first["attention"], second["attention"])
        self.assertEqual(first["best_epoch"], second["best_epoch"])
        self.assertEqual(first["probabilities"].shape, (4,))

    def test_paired_seed_subject_bootstrap_is_reproducible(self):
        y = np.tile([0, 1], 20)
        real = np.vstack(
            [np.where(y == 1, 0.75 + offset, 0.25 - offset) for offset in (0, .01, .02)]
        )
        shuffled = np.vstack([np.roll(row, 1) for row in real])
        first = paired_seed_subject_bootstrap(y, real, shuffled, repeats=50, seed=11)
        second = paired_seed_subject_bootstrap(y, real, shuffled, repeats=50, seed=11)
        self.assertEqual(first, second)
        self.assertGreater(first["point_mean_seedwise_real_minus_shuffled"]["auroc"], 0)


if __name__ == "__main__":
    unittest.main()
