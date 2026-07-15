from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from src import e0_perturbations as e0


def _config() -> dict:
    return {
        "dataset": "daic",
        "prompt": {
            "system": "Analyze speech and transcript information.",
            "user_template": (
                "{audio_context_block}\n{transcript_block}"
                "Based on the {decision_basis}, determine whether the subject is "
                "{label_descriptor}.\n{label_instruction}"
            ),
        },
        "labels": {"label_vocab_version": "legacy_english_labels"},
        "data": {
            "use_audio": True,
            "use_text": True,
            "sample_mode": "subject_audio",
            "chunks_per_subject": 4,
        },
        "training": {"bf16": True},
    }


def _examples(count: int = 6) -> list[dict]:
    config = _config()
    rows = []
    for index in range(count):
        subject_id = f"s{index}"
        label = index % 2
        example = {
            "dataset": "daic",
            "subject_id": subject_id,
            "sample_id": subject_id,
            "label": label,
            "label_text": "Depressed" if label else "Non-depressed",
            "internal_label_text": "Depressed" if label else "Non-depressed",
            "transcript": f"unique transcript {subject_id}",
            "audio_paths": [f"/{subject_id}/chunk_{chunk}.wav" for chunk in range(4)],
            "audio_clip_seconds": [10.0 + chunk for chunk in range(4)],
            "subject_chunk_paths": [f"/{subject_id}/pool_{chunk}.wav" for chunk in range(8)],
            "chunks_per_subject": 4,
            "input_modality": "audio_text",
            "question_id": "subject_audio_bundle",
        }
        prompt, _ = e0._render_condition_prompt(example, config, use_text=True)
        example["prompt_text"] = prompt
        example["training_text"] = prompt + example["internal_label_text"]
        rows.append(example)
    return rows


class PerturbationTransformTests(unittest.TestCase):
    def test_derangements_are_deterministic_bijective_and_class_safe(self) -> None:
        examples = _examples()
        first = e0.build_perturbation_plan(examples, seed=1337)
        second = e0.build_perturbation_plan(examples, seed=1337)
        self.assertEqual(first, second)
        labels = {row["subject_id"]: row["label"] for row in examples}

        for mapping_name in ("across_subject_audio", "same_class_audio", "transcript"):
            mapping = first[mapping_name]
            self.assertEqual(set(mapping), set(mapping.values()))
            self.assertTrue(all(recipient != donor for recipient, donor in mapping.items()))
        self.assertTrue(
            all(labels[recipient] == labels[donor] for recipient, donor in first["same_class_audio"].items())
        )

    def test_impossible_same_class_derangement_fails_closed(self) -> None:
        examples = _examples(3)
        with self.assertRaisesRegex(ValueError, "at least two subjects"):
            e0.build_perturbation_plan(examples, seed=1337)

    def test_bundle_transcript_and_audio_only_transforms_do_not_mutate_sources(self) -> None:
        config = _config()
        examples = _examples()
        originals = copy.deepcopy(examples)
        by_subject = e0._examples_by_subject(examples)
        plan = e0.build_perturbation_plan(examples, seed=1337)
        recipient = examples[0]

        shuffled = e0.build_condition_example(
            recipient, by_subject, plan, "audio_shuffle", config
        )
        audio_donor = by_subject[plan["across_subject_audio"][recipient["subject_id"]]]
        self.assertEqual(shuffled["audio_paths"], audio_donor["audio_paths"])
        self.assertEqual(shuffled["audio_clip_seconds"], audio_donor["audio_clip_seconds"])
        self.assertEqual(shuffled["prompt_text"], recipient["prompt_text"])
        self.assertEqual(shuffled["transcript"], recipient["transcript"])
        self.assertIsNot(shuffled["audio_paths"], audio_donor["audio_paths"])

        transcript_shuffled = e0.build_condition_example(
            recipient, by_subject, plan, "transcript_shuffle", config
        )
        transcript_donor = by_subject[plan["transcript"][recipient["subject_id"]]]
        self.assertEqual(transcript_shuffled["audio_paths"], recipient["audio_paths"])
        self.assertEqual(transcript_shuffled["transcript"], transcript_donor["transcript"])
        self.assertNotEqual(transcript_shuffled["prompt_text"], recipient["prompt_text"])
        self.assertIn(transcript_donor["transcript"], transcript_shuffled["prompt_text"])

        audio_only = e0.build_condition_example(
            recipient, by_subject, plan, "audio_only_real", config
        )
        self.assertEqual(audio_only["transcript"], "")
        self.assertNotIn(recipient["transcript"], audio_only["prompt_text"])
        self.assertEqual(audio_only["prompt_text"].count("<|AUDIO|>"), 4)
        self.assertIn("Based on the audio,", audio_only["prompt_text"])
        self.assertEqual(examples, originals)

    def test_emotion_prompt_schema_fails_closed(self) -> None:
        examples = _examples()
        examples[0]["chunk_caption_by_path"] = {}
        with self.assertRaisesRegex(ValueError, "Unsupported emotion prompt fields"):
            e0._assert_supported_examples(examples, _config())

    def test_legacy_view_rejects_unmaterialized_extra_view(self) -> None:
        examples = _examples()
        with self.assertRaisesRegex(ValueError, "view_index=0"):
            e0.validate_legacy_view(examples, _config(), expected_k=4, view_index=1)


class _FakeTokenizer:
    prompt = "PROMPT"
    token_ids = {
        prompt: [10, 11],
        prompt + "Depressed": [10, 11, 1, 9],
        prompt + "Non-depressed": [10, 11, 2, 8, 7],
    }

    def __call__(self, text, **_kwargs):
        return {"input_ids": torch.tensor([self.token_ids[text]], dtype=torch.long)}

    def decode(self, token_ids):
        return {1: "Dep", 2: "Non"}.get(int(token_ids[0]), "?")


class _FakeProcessor:
    def __init__(self, *, break_prefix: bool = False) -> None:
        self.tokenizer = _FakeTokenizer()
        self.feature_extractor = SimpleNamespace(sampling_rate=16_000)
        self.break_prefix = break_prefix
        self.text_calls: list[str] = []

    def __call__(self, *, text, audio, **_kwargs):
        self.text_calls.append(text)
        self.last_audio_count = len(audio)
        expanded = {
            "PROMPT": [30, 31, 32, 33],
            "PROMPTDepressed": [30, 31, 32, 33, 1, 9],
            "PROMPTNon-depressed": [30, 31, 32, 33, 2, 8, 7],
        }[text]
        if self.break_prefix and text == "PROMPTDepressed":
            expanded = [99, *expanded[1:]]
        return {"input_ids": torch.tensor([expanded], dtype=torch.long)}


class _FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.ones(()))
        self.forward_calls = 0
        self.input_lengths: list[int] = []

    def forward(self, *, input_ids, use_cache):
        self.forward_calls += 1
        self.input_lengths.append(int(input_ids.shape[1]))
        if use_cache or not torch.is_inference_mode_enabled():
            raise AssertionError("E0 scorer must use inference mode without KV cache.")
        logits = torch.zeros((1, input_ids.shape[1], 16), dtype=torch.float32)
        logits[..., 1] = 5.0
        logits[..., 2] = 1.0
        logits[..., 9] = 2.0
        logits[..., 8] = 0.5
        logits[..., 7] = 0.25
        return SimpleNamespace(logits=logits)


def _scoring_example() -> dict:
    return {
        "prompt_text": "PROMPT",
        "audio_paths": [f"chunk-{index}" for index in range(4)],
        "audio_clip_seconds": [30.0] * 4,
        "e0_silence_audio": False,
    }


class DirectFirstTokenScorerTests(unittest.TestCase):
    def test_one_prompt_forward_uses_processor_validated_first_tokens(self) -> None:
        model = _FakeModel()
        processor = _FakeProcessor()
        audio_calls = []

        def audio_loader(path, sampling_rate, max_seconds, silence_audio):
            audio_calls.append((path, sampling_rate, max_seconds, silence_audio))
            return torch.zeros(8).numpy()

        scorer = e0.DirectFirstTokenScorer(
            model,
            processor,
            _config(),
            torch.device("cpu"),
            audio_loader=audio_loader,
        )
        row = scorer.score(_scoring_example())

        self.assertEqual(model.forward_calls, 1)
        self.assertEqual(model.input_lengths, [4])
        self.assertEqual(processor.text_calls, ["PROMPT", "PROMPTDepressed", "PROMPTNon-depressed"])
        self.assertEqual(processor.last_audio_count, 4)
        self.assertEqual(len(audio_calls), 4)
        self.assertEqual(row["positive_first_token_id"], 1)
        self.assertEqual(row["negative_first_token_id"], 2)
        self.assertEqual(row["expanded_prompt_input_tokens"], 4)
        self.assertEqual(row["positive_label_continuation_tokens"], 2)
        self.assertEqual(row["negative_label_continuation_tokens"], 3)
        self.assertEqual(row["first_token_margin"], 4.0)
        self.assertEqual(row["first_token_prediction"], 1)

    def test_processor_prefix_mismatch_fails_before_model_forward(self) -> None:
        model = _FakeModel()
        scorer = e0.DirectFirstTokenScorer(
            model,
            _FakeProcessor(break_prefix=True),
            _config(),
            torch.device("cpu"),
            audio_loader=lambda *_args: torch.zeros(8).numpy(),
        )
        with self.assertRaisesRegex(ValueError, "does not preserve the expanded prompt prefix"):
            scorer.score(_scoring_example())
        self.assertEqual(model.forward_calls, 0)

    def test_optional_candidate_scores_are_explicitly_secondary(self) -> None:
        model = _FakeModel()
        scorer = e0.DirectFirstTokenScorer(
            model,
            _FakeProcessor(),
            _config(),
            torch.device("cpu"),
            include_candidate_likelihood=True,
            audio_loader=lambda *_args: torch.zeros(8).numpy(),
        )
        row = scorer.score(_scoring_example())
        self.assertEqual(model.forward_calls, 3)
        self.assertIn("candidate_likelihood_margin", row)
        self.assertEqual(row["scorer_protocol"], e0.SCORER_PROTOCOL)


class ArtifactTests(unittest.TestCase):
    def test_condition_summary_reports_preregistered_primary_and_secondary_metrics(self) -> None:
        rows = [
            {
                "label": 0,
                "first_token_prediction": 0,
                "first_token_margin": -2.0,
                "candidate_likelihood_prediction": 1,
                "candidate_likelihood_margin": 0.2,
            },
            {
                "label": 1,
                "first_token_prediction": 1,
                "first_token_margin": 3.0,
                "candidate_likelihood_prediction": 1,
                "candidate_likelihood_margin": 0.1,
            },
        ]
        summary = e0._condition_summary(rows)
        for metric in ("auroc", "auprc", "balanced_accuracy", "macro_f1", "positive_f1"):
            self.assertIn(metric, summary)
        self.assertEqual(summary["auroc"], 1.0)
        self.assertEqual(summary["auprc"], 1.0)
        self.assertEqual(summary["balanced_accuracy"], 1.0)
        secondary = summary["candidate_likelihood_secondary"]
        for metric in ("auroc", "auprc", "balanced_accuracy", "macro_f1", "positive_f1"):
            self.assertIn(metric, secondary)

    def test_condition_output_is_canonical_subject_jsonl_and_immutable(self) -> None:
        examples = _examples()
        plan = e0.build_perturbation_plan(examples, seed=1337)

        class FakeScorer:
            device = torch.device("cpu")
            include_candidate_likelihood = False

            def score(self, example):
                label = int(example["label"])
                margin = 1.0 if label else -1.0
                return {
                    "scorer_protocol": e0.SCORER_PROTOCOL,
                    "positive_first_token_id": 1,
                    "negative_first_token_id": 2,
                    "positive_first_token_logit": margin,
                    "negative_first_token_logit": 0.0,
                    "first_token_margin": margin,
                    "first_token_prediction": int(margin > 0),
                    "dep_score": margin,
                    "non_score": 0.0,
                    "likelihood_prediction": int(margin > 0),
                }

        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary)
            result = e0.run_condition(
                condition="real",
                recipient_examples=examples,
                examples_by_subject=e0._examples_by_subject(examples),
                plan=plan,
                scorer=FakeScorer(),
                model=_FakeModel(),
                config=_config(),
                view_spec={"view_id": e0.LEGACY_VIEW_ID, "view_index": 0, "k": 4},
                seed=1337,
                output_root=output_root,
                input_provenance={"inputs": {}},
                progress_every=100,
            )
            predictions = Path(result["condition_dir"]) / "predictions_subject_level.jsonl"
            rows = [json.loads(line) for line in predictions.read_text().splitlines()]
            self.assertEqual(len(rows), len(examples))
            self.assertEqual(len({row["subject_id"] for row in rows}), len(examples))
            self.assertTrue((predictions.parent / "condition_config.json").is_file())
            self.assertTrue((predictions.parent / "provenance.json").is_file())
            summary = json.loads((predictions.parent / "condition_summary.json").read_text())
            self.assertIn("auprc", summary)
            self.assertIn("balanced_accuracy", summary)
            with self.assertRaises(FileExistsError):
                e0.run_condition(
                    condition="real",
                    recipient_examples=examples,
                    examples_by_subject=e0._examples_by_subject(examples),
                    plan=plan,
                    scorer=FakeScorer(),
                    model=_FakeModel(),
                    config=_config(),
                    view_spec={"view_id": e0.LEGACY_VIEW_ID, "view_index": 0, "k": 4},
                    seed=1337,
                    output_root=output_root,
                    input_provenance={"inputs": {}},
                    progress_every=100,
                )


if __name__ == "__main__":
    unittest.main()
