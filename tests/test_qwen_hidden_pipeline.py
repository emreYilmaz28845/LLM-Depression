from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from src.aggregate import aggregate_binary_classifier_predictions, aggregate_original_teacher_forced_predictions
from src.features.pooling import aligned_attention_mask, last_valid_token
from src.features.qwen_hidden_collator import PromptOnlyExtractionCollator
from src.features.extract_qwen_hidden import _decoder_hidden_size


class _FakeFeatureExtractor:
    sampling_rate = 16_000


class _FakeProcessor:
    feature_extractor = _FakeFeatureExtractor()

    def __call__(self, *, text, return_tensors, padding, **kwargs):
        ids = [ord(char) for char in text]
        return {
            "input_ids": torch.tensor([ids], dtype=torch.long),
            "attention_mask": torch.ones((1, len(ids)), dtype=torch.long),
        }


class PoolingTests(unittest.TestCase):
    def test_backend_specific_decoder_dimensions(self):
        text_model = SimpleNamespace(config=SimpleNamespace(hidden_size=3584), base_model=None)
        audio_model = SimpleNamespace(
            config=SimpleNamespace(text_config=SimpleNamespace(hidden_size=4096)),
            base_model=None,
        )
        self.assertEqual(_decoder_hidden_size(text_model), 3584)
        self.assertEqual(_decoder_hidden_size(audio_model), 4096)

    def test_last_valid_token_right_padding_and_batches(self):
        hidden = torch.arange(2 * 4 * 3).reshape(2, 4, 3).to(torch.bfloat16)
        mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])
        result = last_valid_token(hidden, mask)
        self.assertEqual(result.dtype, torch.float32)
        torch.testing.assert_close(result[0], hidden[0, 1].float())
        torch.testing.assert_close(result[1], hidden[1, 2].float())

    def test_one_token_and_empty_mask(self):
        hidden = torch.tensor([[[2.0, 3.0]]])
        torch.testing.assert_close(last_valid_token(hidden, torch.ones((1, 1))), hidden[:, 0])
        with self.assertRaisesRegex(ValueError, "at least one valid"):
            last_valid_token(hidden, torch.zeros((1, 1)))

    def test_audio_expansion_uses_aligned_all_valid_mask(self):
        hidden = torch.zeros((1, 5, 2))
        input_mask = torch.ones((1, 2), dtype=torch.long)
        mask, source = aligned_attention_mask(hidden, input_mask)
        self.assertEqual(source, "batch1_all_valid")
        self.assertEqual(tuple(mask.shape), (1, 5))
        self.assertTrue(bool(torch.all(mask == 1)))


class CollatorTests(unittest.TestCase):
    def test_prompt_only_never_adds_answer_or_metadata_to_model_inputs(self):
        example = {
            "dataset": "cmdc",
            "sample_id": "s1",
            "subject_id": "p1",
            "label": 1,
            "partition": "outer_train",
            "fold": 0,
            "prompt_text": "constant prompt",
            "training_text": "constant promptDepressed",
            "audio_arrays": [],
        }
        model_inputs, metadata = PromptOnlyExtractionCollator(_FakeProcessor())([example])
        decoded = "".join(chr(value) for value in model_inputs["input_ids"][0].tolist())
        self.assertEqual(decoded, example["prompt_text"])
        self.assertNotIn("Depressed", decoded)
        self.assertNotIn("labels", model_inputs)
        self.assertNotIn("sample_id", model_inputs)
        self.assertEqual(metadata[0]["sample_id"], "s1")


class AggregationTests(unittest.TestCase):
    def test_classifier_vote_matches_existing_tie_behavior(self):
        classifier_rows = [
            {"subject_id": "a", "label": 1, "predicted_class": 1, "probability": 0.8},
            {"subject_id": "a", "label": 1, "predicted_class": 0, "probability": 0.4},
        ]
        baseline_rows = [
            {"subject_id": "a", "label": 1, "teacher_forced_prediction": 1, "dep_score": 0.8, "non_score": 0.2},
            {"subject_id": "a", "label": 1, "teacher_forced_prediction": 0, "dep_score": 0.4, "non_score": 0.6},
        ]
        classifier_subjects, _ = aggregate_binary_classifier_predictions(classifier_rows)
        baseline_subjects, _ = aggregate_original_teacher_forced_predictions(baseline_rows)
        self.assertEqual(classifier_subjects[0]["prediction"], baseline_subjects[0]["prediction"])
        self.assertEqual(classifier_subjects[0]["prediction"], 1)


class ClassifierTests(unittest.TestCase):
    def test_shuffled_labels_preserve_subject_groups(self):
        from baselines.qwen_hidden_classifier import shuffled_subject_labels

        rows = [
            {"subject_id": "a", "label": 0},
            {"subject_id": "a", "label": 0},
            {"subject_id": "b", "label": 1},
            {"subject_id": "b", "label": 1},
            {"subject_id": "c", "label": 0},
            {"subject_id": "c", "label": 0},
            {"subject_id": "d", "label": 1},
            {"subject_id": "d", "label": 1},
        ]
        shuffled = shuffled_subject_labels(rows, seed=1337)
        self.assertEqual(shuffled[0], shuffled[1])
        self.assertEqual(shuffled[2], shuffled[3])
        self.assertEqual(shuffled[4], shuffled[5])
        self.assertEqual(shuffled[6], shuffled[7])
        self.assertEqual(sorted(shuffled[::2].tolist()), [0, 0, 1, 1])

    def test_pca_artifact_records_only_training_rows(self):
        from baselines.qwen_hidden_classifier import run_variant
        from src.utils import save_json, write_jsonl

        rng = np.random.default_rng(7)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "cache"
            output = root / "output"
            cache.mkdir()
            train_x = rng.normal(size=(40, 36)).astype(np.float32)
            test_x = rng.normal(size=(8, 36)).astype(np.float32) + 100.0
            np.savez_compressed(cache / "outer_train.npz", vectors=train_x)
            np.savez_compressed(cache / "final_eval.npz", vectors=test_x)
            train_rows = [
                {"sample_id": f"tr{i}", "subject_id": f"tr{i}", "label": i % 2}
                for i in range(40)
            ]
            test_rows = [
                {"sample_id": f"te{i}", "subject_id": f"te{i}", "label": i % 2}
                for i in range(8)
            ]
            write_jsonl(train_rows, cache / "outer_train_rows.jsonl")
            write_jsonl(test_rows, cache / "final_eval_rows.jsonl")
            save_json(
                {
                    "dataset": "synthetic",
                    "input_modality": "text_only",
                    "fold": 0,
                    "checkpoint_dir": "synthetic/best_model",
                },
                cache / "extraction_metadata.json",
            )
            run_variant(cache, output, "logreg_pca32", seed=1337)
            metadata = __import__("json").loads(
                (output / "logreg_pca32" / "classifier_metadata.json").read_text()
            )
            self.assertEqual(metadata["training_row_ids"], [f"tr{i}" for i in range(40)])
            self.assertTrue(set(metadata["training_subject_ids"]).isdisjoint(metadata["heldout_subject_ids"]))
            import joblib

            pipeline = joblib.load(output / "logreg_pca32" / "pipeline.joblib")
            np.testing.assert_allclose(pipeline.named_steps["pca"].mean_, train_x.mean(axis=0), rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
