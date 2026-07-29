from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from src.aggregate import aggregate_binary_classifier_predictions, aggregate_original_teacher_forced_predictions
from src.features.pooling import aligned_attention_mask, last_valid_token
from src.features.hidden_classifier_policy import response_normalized_sample_weights
from src.features.qwen_hidden_collator import PromptOnlyExtractionCollator
from src.features.extract_qwen_hidden import (
    _decoder_hidden_size,
    _emotion_provenance,
    _resolve_subject_partitions,
    resolve_condition,
)


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

    def test_emotion_conditions_are_explicit_and_path_safe(self):
        self.assertEqual(resolve_condition(None, "audio_text", False), "audio_text")
        self.assertEqual(
            resolve_condition("audio_text_emotion_en", "audio_text", True),
            "audio_text_emotion_en",
        )
        with self.assertRaisesRegex(ValueError, "distinct"):
            resolve_condition(None, "audio_text", True)
        with self.assertRaisesRegex(ValueError, "Invalid"):
            resolve_condition("../collision", "audio_text", True)

    def test_emotion_provenance_records_hash_coverage_and_fallbacks(self):
        from src.utils import write_jsonl

        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "emotion.jsonl"
            write_jsonl(
                [
                    {"sample_id": "train-covered", "emotion_en": "calm"},
                    {"sample_id": "test-null", "emotion_en": None},
                ],
                cache_path,
            )
            config = {
                "data": {
                    "use_audio": True,
                    "use_emotion": True,
                    "emotion_cache_path": str(cache_path),
                    "emotion_caption_field": "emotion_en",
                    "emotion_on_missing": "neutral_fallback",
                }
            }
            rows = [
                {"sample_id": "train-covered", "subject_id": "train"},
                {"sample_id": "val-missing", "subject_id": "val"},
                {"sample_id": "test-null", "subject_id": "test"},
            ]
            split = {
                "outer_train": ["train", "val"],
                "final_eval": ["test"],
            }
            provenance = _emotion_provenance(
                config,
                rows,
                split,
                source="secap_local",
                language="en",
            )
            self.assertIsNotNone(provenance)
            self.assertEqual(provenance["partition_coverage"]["outer_train"]["covered"], 1)
            self.assertEqual(provenance["partition_coverage"]["outer_train"]["missing_row"], 1)
            self.assertEqual(provenance["partition_coverage"]["final_eval"]["null_caption"], 1)
            self.assertEqual(provenance["fallback_caption_count"], 2)
            self.assertEqual(len(provenance["cache_sha256"]), 64)

    def test_standard_partition_resolution_preserves_train_selection_union(self):
        split = {
            "train_subject_ids": ["train"],
            "selection_subject_ids": ["selection"],
            "final_eval_subject_ids": ["test"],
        }
        partitions, provenance = _resolve_subject_partitions({}, {"split": {}}, split)
        self.assertEqual(partitions["outer_train"], ["selection", "train"])
        self.assertEqual(partitions["final_eval"], ["test"])
        self.assertEqual(provenance["evaluation_protocol"], "saved_final_evaluation")

    def test_train_val_partition_resolution_uses_selection_as_heldout(self):
        split = {
            "train_subject_ids": ["train"],
            "selection_subject_ids": ["outer-validation"],
            "final_eval_subject_ids": [],
        }
        partitions, provenance = _resolve_subject_partitions(
            {"cv_protocol": "train_val"},
            {"split": {"cv_protocol": "train_val"}},
            split,
        )
        self.assertEqual(partitions["outer_train"], ["train"])
        self.assertEqual(partitions["final_eval"], ["outer-validation"])
        self.assertEqual(provenance["evaluation_protocol"], "table_aligned_outer_validation")
        self.assertEqual(provenance["partition_sources"]["final_eval"], ["selection_subject_ids"])

    def test_partition_resolution_rejects_empty_or_overlapping_heldout(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            _resolve_subject_partitions(
                {"cv_protocol": "train_val"},
                {"split": {}},
                {"train_subject_ids": ["a"], "selection_subject_ids": []},
            )
        with self.assertRaisesRegex(ValueError, "overlap"):
            _resolve_subject_partitions(
                {},
                {"split": {}},
                {
                    "train_subject_ids": ["a"],
                    "selection_subject_ids": [],
                    "final_eval_subject_ids": ["a"],
                },
            )

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
            "dataset": "d3tec",
            "sample_id": "s1",
            "subject_id": "p1",
            "label": 1,
            "partition": "outer_train",
            "fold": 0,
            "prompt_text": "constant prompt",
            "training_text": "constant promptDepressed",
            "audio_arrays": [],
            "response_id": "p1_p3",
            "prompt_id": 3,
            "segment_index": 1,
            "num_segments": 2,
        }
        model_inputs, metadata = PromptOnlyExtractionCollator(_FakeProcessor())([example])
        decoded = "".join(chr(value) for value in model_inputs["input_ids"][0].tolist())
        self.assertEqual(decoded, example["prompt_text"])
        self.assertNotIn("Depressed", decoded)
        self.assertNotIn("labels", model_inputs)
        self.assertNotIn("sample_id", model_inputs)
        self.assertEqual(metadata[0]["sample_id"], "s1")
        self.assertEqual(metadata[0]["response_id"], "p1_p3")
        self.assertEqual(metadata[0]["prompt_id"], 3)
        self.assertEqual(metadata[0]["segment_index"], 1)
        self.assertEqual(metadata[0]["num_segments"], 2)
        self.assertNotIn("p1_p3", decoded)


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

    def test_d3tec_hierarchy_gives_each_response_one_vote_and_uses_subject_auroc(self):
        rows = []
        # Subject a: one 5-segment positive response and two one-segment
        # negative responses. A flat vote would be positive; equal response
        # votes must be negative.
        for index in range(5):
            rows.append(
                {
                    "sample_id": f"a0-{index}",
                    "subject_id": "a",
                    "response_id": "a_p0",
                    "prompt_id": 0,
                    "segment_index": index,
                    "num_segments": 5,
                    "label": 0,
                    "predicted_class": 1,
                    "probability": 0.9,
                }
            )
        for prompt in (1, 2):
            rows.append(
                {
                    "sample_id": f"a{prompt}-0",
                    "subject_id": "a",
                    "response_id": f"a_p{prompt}",
                    "prompt_id": prompt,
                    "segment_index": 0,
                    "num_segments": 1,
                    "label": 0,
                    "predicted_class": 0,
                    "probability": 0.1,
                }
            )
        for prompt in range(3):
            rows.append(
                {
                    "sample_id": f"b{prompt}-0",
                    "subject_id": "b",
                    "response_id": f"b_p{prompt}",
                    "prompt_id": prompt,
                    "segment_index": 0,
                    "num_segments": 1,
                    "label": 1,
                    "predicted_class": 1,
                    "probability": 0.8,
                }
            )
        subjects, metrics = aggregate_binary_classifier_predictions(rows)
        by_subject = {row["subject_id"]: row for row in subjects}
        self.assertEqual(by_subject["a"]["prediction"], 0)
        self.assertEqual(by_subject["a"]["response_count"], 3)
        self.assertEqual(metrics["auroc"], 1.0)
        self.assertEqual(metrics["aggregation_level"], "response_subject")

    def test_d3tec_hierarchical_ties_use_probability_margin_at_each_level(self):
        rows = [
            {
                "sample_id": sample_id,
                "subject_id": "a",
                "response_id": response_id,
                "prompt_id": prompt_id,
                "segment_index": segment_index,
                "num_segments": 2,
                "label": 1,
                "predicted_class": prediction,
                "probability": probability,
            }
            for response_id, prompt_id, pairs in (
                ("a_p0", 0, ((1, 0.95), (0, 0.45))),
                ("a_p1", 1, ((0, 0.1), (1, 0.6))),
            )
            for segment_index, (prediction, probability) in enumerate(pairs)
            for sample_id in (f"{response_id}-{segment_index}",)
        ]
        subjects, _ = aggregate_binary_classifier_predictions(rows)
        # Response p0 resolves positive and p1 negative. Their subject vote is
        # tied, so p0's larger probability margin resolves the subject positive.
        self.assertEqual(subjects[0]["prediction"], 1)
        self.assertEqual(subjects[0]["num_responses"], 2)


class ClassifierTests(unittest.TestCase):
    def test_majority_control_uses_subject_not_segment_prevalence(self):
        from baselines.qwen_hidden_classifier import majority_subject_control

        rows = [
            {"subject_id": "negative-a", "label": 0},
            {"subject_id": "negative-b", "label": 0},
        ] + [
            {"subject_id": "positive", "label": 1}
            for _ in range(20)
        ]
        prediction, prevalence = majority_subject_control(rows)
        self.assertEqual(prediction, 0)
        self.assertAlmostEqual(prevalence, 1 / 3)

    def test_d3tec_response_weights_have_mean_one_and_equal_response_subject_totals(self):
        rows = []
        for subject_id in ("a", "b"):
            for prompt_id in range(27):
                count = 1 + (prompt_id % 3)
                for segment_index in range(count):
                    rows.append(
                        {
                            "sample_id": f"{subject_id}-{prompt_id}-{segment_index}",
                            "subject_id": subject_id,
                            "response_id": f"{subject_id}_p{prompt_id}",
                            "prompt_id": prompt_id,
                            "segment_index": segment_index,
                            "num_segments": count,
                            "label": int(subject_id == "b"),
                        }
                    )
        weights, audit = response_normalized_sample_weights(
            rows,
            {"dataset": "d3tec", "input_modality": "audio_text"},
        )
        self.assertAlmostEqual(float(weights.mean()), 1.0)
        self.assertTrue(audit["equal_response_totals"])
        self.assertTrue(audit["equal_subject_totals"])
        self.assertEqual(set(audit["responses_per_subject"].values()), {27})
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
                    "condition": "text_only_control",
                    "fold": 0,
                    "checkpoint_dir": "synthetic/best_model",
                },
                cache / "extraction_metadata.json",
            )
            run_variant(cache, output, "logreg_pca32", seed=1337)
            first_metadata_mtime = (
                output / "logreg_pca32" / "classifier_metadata.json"
            ).stat().st_mtime_ns
            run_variant(cache, output, "logreg_pca32", seed=1337)
            self.assertEqual(
                (output / "logreg_pca32" / "classifier_metadata.json").stat().st_mtime_ns,
                first_metadata_mtime,
            )
            metadata = __import__("json").loads(
                (output / "logreg_pca32" / "classifier_metadata.json").read_text()
            )
            self.assertEqual(metadata["training_row_ids"], [f"tr{i}" for i in range(40)])
            self.assertEqual(metadata["condition"], "text_only_control")
            self.assertTrue(set(metadata["training_subject_ids"]).isdisjoint(metadata["heldout_subject_ids"]))
            import joblib

            pipeline = joblib.load(output / "logreg_pca32" / "pipeline.joblib")
            np.testing.assert_allclose(pipeline.named_steps["pca"].mean_, train_x.mean(axis=0), rtol=1e-5)
            np.savez_compressed(cache / "outer_train.npz", vectors=train_x + 1.0)
            with self.assertRaisesRegex(ValueError, "incompatible"):
                run_variant(cache, output, "logreg_pca32", seed=1337)


if __name__ == "__main__":
    unittest.main()
