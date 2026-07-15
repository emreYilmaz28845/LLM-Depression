from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from src import evaluate
from src.model import qwen2audio_lora, text_lora


class _DropoutModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.ones(()))
        self.dropout = torch.nn.Dropout(p=0.75)
        self.config = SimpleNamespace(use_cache=False)


class _InferenceModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.ones(()))
        self.config = SimpleNamespace(use_cache=False)
        self.gradient_checkpointing_disable_calls = 0

    def gradient_checkpointing_disable(self) -> None:
        self.gradient_checkpointing_disable_calls += 1


class DeterministicEvaluationStateTests(unittest.TestCase):
    def test_repeated_evaluation_is_identical_and_restores_training_state(self) -> None:
        model = _DropoutModel()
        model.train()
        outputs: list[torch.Tensor] = []

        def prepare(current_model, _config) -> None:
            current_model.config.use_cache = True

        def restore(current_model, _config) -> None:
            current_model.config.use_cache = False

        def evaluate_in_current_state(**kwargs):
            current_model = kwargs["model"]
            self.assertFalse(current_model.training)
            value = current_model.dropout(torch.ones(32))
            outputs.append(value.detach().clone())
            return {"value": value}

        with (
            patch.object(evaluate, "prepare_model_for_evaluation", side_effect=prepare) as prepare_mock,
            patch.object(evaluate, "restore_model_for_training", side_effect=restore) as restore_mock,
            patch.object(
                evaluate,
                "_evaluate_examples_in_current_state",
                side_effect=evaluate_in_current_state,
            ),
        ):
            for _ in range(2):
                evaluate.evaluate_examples(
                    model=model,
                    processor=None,
                    examples=[],
                    config={"training": {"gradient_checkpointing": True}},
                    output_dir=Path("unused"),
                    checkpoint_name="test",
                )
                self.assertTrue(model.training)
                self.assertFalse(model.config.use_cache)

        self.assertTrue(torch.equal(outputs[0], outputs[1]))
        self.assertEqual(prepare_mock.call_count, 2)
        self.assertEqual(restore_mock.call_count, 2)

    def test_training_state_is_restored_when_evaluation_raises(self) -> None:
        model = _DropoutModel()
        model.train()

        with (
            patch.object(evaluate, "prepare_model_for_evaluation"),
            patch.object(evaluate, "restore_model_for_training") as restore_mock,
            patch.object(
                evaluate,
                "_evaluate_examples_in_current_state",
                side_effect=RuntimeError("evaluation failed"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "evaluation failed"):
                evaluate.evaluate_examples(
                    model=model,
                    processor=None,
                    examples=[],
                    config={"training": {}},
                    output_dir=Path("unused"),
                    checkpoint_name="test",
                )

        self.assertTrue(model.training)
        restore_mock.assert_called_once_with(model, {"training": {}})

    def test_likelihood_prediction_matches_score_difference_sign(self) -> None:
        example = {
            "config": {},
            "subject_id": "subject-1",
            "sample_id": "sample-1",
            "label": 1,
            "label_text": "Depressed",
            "internal_label_text": "A",
        }
        score_pairs = ((-0.1, -0.4), (-0.8, -0.2), (-0.5, -0.5))
        sign_mismatches = 0

        for dep_score, non_score in score_pairs:
            with patch.object(evaluate, "score_candidate_label", side_effect=(dep_score, non_score)):
                row = evaluate._predict_sample_likelihood(
                    model=None,
                    processor=None,
                    example=example,
                    device=torch.device("cpu"),
                    silence_audio=False,
                    checkpoint_name="test",
                )
            expected = int(dep_score > non_score)
            sign_mismatches += int(row["likelihood_prediction"] != expected)

        self.assertEqual(sign_mismatches, 0)


class DirectScoringTests(unittest.TestCase):
    def test_candidate_scoring_uses_inference_mode_without_kv_cache(self) -> None:
        calls: list[dict[str, bool]] = []

        class Processor:
            feature_extractor = None

            def __call__(self, *, text, **_kwargs):
                token_ids = [0] if text == "prompt" else [0, 1]
                return {"input_ids": torch.tensor([token_ids], dtype=torch.long)}

        class Model:
            def __call__(self, *, input_ids, use_cache):
                calls.append(
                    {
                        "use_cache": bool(use_cache),
                        "inference_mode": torch.is_inference_mode_enabled(),
                    }
                )
                logits = torch.zeros((1, input_ids.shape[1], 3), dtype=torch.float32)
                logits[0, 0, 1] = 2.0
                return SimpleNamespace(logits=logits)

        score = evaluate.score_candidate_label(
            model=Model(),
            processor=Processor(),
            example={"audio_paths": [], "prompt_text": "prompt"},
            candidate_label="A",
            device=torch.device("cpu"),
            silence_audio=False,
        )

        self.assertTrue(torch.isfinite(torch.tensor(score)))
        self.assertEqual(calls, [{"use_cache": False, "inference_mode": True}])

    def test_single_token_teacher_forced_ab_decision_matches_margin_sign(self) -> None:
        class Processor:
            feature_extractor = None

            def __call__(self, *, text, **_kwargs):
                token_by_text = {"prompt": [0], "promptA": [0, 1], "promptB": [0, 2]}
                return {"input_ids": torch.tensor([token_by_text[text]], dtype=torch.long)}

            def decode(self, token_ids, **_kwargs):
                labels = {1: "A", 2: "B"}
                return "".join(labels.get(int(token_id), "") for token_id in token_ids.reshape(-1))

        class Model(torch.nn.Module):
            def __init__(self, winner_token_id: int) -> None:
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.ones(()))
                self.winner_token_id = winner_token_id

            def forward(self, *, input_ids, use_cache):
                self.assert_protocol(use_cache)
                logits = torch.zeros((1, input_ids.shape[1], 4), dtype=torch.float32)
                logits[..., 1] = 1.0
                logits[..., 2] = 1.0
                logits[..., self.winner_token_id] = 3.0
                return SimpleNamespace(logits=logits)

            @staticmethod
            def assert_protocol(use_cache) -> None:
                if use_cache or not torch.is_inference_mode_enabled():
                    raise AssertionError("Direct A/B scoring must use inference mode without KV cache.")

        sign_mismatches = 0
        for winner_token_id, expected_prediction in ((1, 1), (2, 0)):
            internal_label = "A" if expected_prediction == 1 else "B"
            row = evaluate._predict_sample_original_teacher_forced(
                model=Model(winner_token_id),
                processor=Processor(),
                example={
                    "config": {},
                    "audio_paths": [],
                    "prompt_text": "prompt",
                    "internal_label_text": internal_label,
                    "subject_id": "subject-1",
                    "sample_id": f"sample-{internal_label}",
                    "label": expected_prediction,
                    "label_text": "Depressed" if expected_prediction == 1 else "Non-depressed",
                },
                device=torch.device("cpu"),
                silence_audio=False,
                checkpoint_name="test",
            )
            margin_prediction = int(row["teacher_forced_margin"] > 0)
            sign_mismatches += int(row["teacher_forced_prediction"] != margin_prediction)

        self.assertEqual(sign_mismatches, 0)


class InferenceDtypeTests(unittest.TestCase):
    def test_standalone_loaders_match_bf16_training_protocol_on_cuda(self) -> None:
        cases = (
            (qwen2audio_lora, qwen2audio_lora.Qwen2AudioForConditionalGeneration),
            (text_lora, text_lora.AutoModelForCausalLM),
        )
        for module, model_class in cases:
            with self.subTest(module=module.__name__):
                model = _InferenceModel()
                with (
                    patch.object(module.torch.cuda, "is_available", return_value=True),
                    patch.object(model_class, "from_pretrained", return_value=model) as load_mock,
                ):
                    loaded = module.load_model_for_inference(
                        "base-model",
                        config={"training": {"bf16": True}},
                    )

                load_mock.assert_called_once_with("base-model", torch_dtype=torch.bfloat16)
                self.assertIs(loaded, model)
                self.assertFalse(loaded.training)
                self.assertTrue(loaded.config.use_cache)

    def test_standalone_loaders_do_not_force_bf16_without_cuda(self) -> None:
        cases = (
            (qwen2audio_lora, qwen2audio_lora.Qwen2AudioForConditionalGeneration),
            (text_lora, text_lora.AutoModelForCausalLM),
        )
        for module, model_class in cases:
            with self.subTest(module=module.__name__):
                model = _InferenceModel()
                with (
                    patch.object(module.torch.cuda, "is_available", return_value=False),
                    patch.object(model_class, "from_pretrained", return_value=model) as load_mock,
                ):
                    module.load_model_for_inference(
                        "base-model",
                        config={"training": {"bf16": True}},
                    )

                load_mock.assert_called_once_with("base-model", torch_dtype=None)


class StandaloneRuntimeSummaryTests(unittest.TestCase):
    def test_cpu_summary_persists_dtype_and_explicitly_empty_cuda_peaks(self) -> None:
        summary = evaluate._standalone_runtime_summary(_InferenceModel(), torch.device("cpu"))

        self.assertEqual(summary["model_dtype"], "float32")
        self.assertEqual(summary["model_parameter_dtypes"], ["float32"])
        self.assertIsNone(summary["cuda_max_memory_allocated_bytes"])
        self.assertIsNone(summary["cuda_max_memory_reserved_bytes"])

    def test_cuda_summary_persists_allocated_and_reserved_peaks(self) -> None:
        properties = SimpleNamespace(name="Test GPU", total_memory=24_000)
        with (
            patch.object(evaluate.torch.cuda, "get_device_properties", return_value=properties),
            patch.object(evaluate.torch.cuda, "max_memory_allocated", return_value=11_000),
            patch.object(evaluate.torch.cuda, "max_memory_reserved", return_value=13_000),
            patch.object(evaluate.torch.cuda, "memory_allocated", return_value=9_000),
            patch.object(evaluate.torch.cuda, "memory_reserved", return_value=10_000),
        ):
            summary = evaluate._standalone_runtime_summary(
                _InferenceModel(),
                torch.device("cuda"),
            )

        self.assertEqual(summary["cuda_device_name"], "Test GPU")
        self.assertEqual(summary["cuda_max_memory_allocated_bytes"], 11_000)
        self.assertEqual(summary["cuda_max_memory_reserved_bytes"], 13_000)
        self.assertEqual(summary["cuda_total_memory_bytes"], 24_000)


if __name__ == "__main__":
    unittest.main()
