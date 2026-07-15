from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from scripts import e0_gradient_audit as audit


class _Tokenizer:
    def convert_ids_to_tokens(self, values):
        return [f"tok-{int(value)}" for value in values]

    def decode(self, values, skip_special_tokens=False):
        del skip_special_tokens
        if [int(value) for value in values] == [41, 42]:
            return "Non-depressed<|im_end|>\n"
        return " ".join(f"tok-{int(value)}" for value in values)


class _Processor:
    tokenizer = _Tokenizer()


class _LoRALayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base_weight = torch.nn.Parameter(torch.tensor([3.0]), requires_grad=False)
        self.lora_A = torch.nn.Parameter(torch.tensor([2.0]))
        self.lora_B = torch.nn.Parameter(torch.tensor([4.0]))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value * self.lora_A * self.lora_B


class _AuditModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.audio_tower = torch.nn.Linear(2, 2, bias=False)
        self.multi_modal_projector = torch.nn.Linear(2, 2, bias=False)
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([_LoRALayer(), _LoRALayer()])
        self.other = torch.nn.Parameter(torch.tensor([5.0]), requires_grad=False)
        self.audio_tower.requires_grad_(False)
        self.multi_modal_projector.requires_grad_(False)
        self.forward_calls = 0

    def forward(self, *, input_features, labels, use_cache, **_kwargs):
        self.forward_calls += 1
        if use_cache:
            raise AssertionError("Audit must keep KV cache disabled.")
        value = input_features.mean()
        for layer in self.model.layers:
            value = layer(value)
        return SimpleNamespace(loss=value + labels.float().mean() * 0.0)


def _example(subject: str = "300", k: int = 4) -> dict:
    return {
        "subject_id": subject,
        "sample_id": subject,
        "label": 0,
        "internal_label_text": "Non-depressed",
        "audio_paths": [f"/{subject}/audio_{index}.wav" for index in range(k)],
        "audio_clip_seconds": [30.0] * k,
        "chunks_per_subject": k,
        "prompt_text": "private prompt",
        "training_text": "private promptNon-depressed",
    }


class SelectionAndArtifactTests(unittest.TestCase):
    def test_single_example_selection_is_stable_and_requires_exact_k4(self) -> None:
        examples = [_example("301"), _example("300")]
        selected = audit.select_single_k4_example(examples)
        self.assertEqual(selected["subject_id"], "300")
        self.assertIs(audit.select_single_k4_example(examples, "301"), examples[0])
        with self.assertRaisesRegex(ValueError, "baked K=4"):
            audit.select_single_k4_example([_example(k=3)])

    def test_checkpoint_inventory_is_canonical_and_content_sensitive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "nested").mkdir()
            (root / "b.txt").write_text("b", encoding="utf-8")
            (root / "nested" / "a.txt").write_text("a", encoding="utf-8")
            first = audit.checkpoint_inventory(root)
            second = audit.checkpoint_inventory(root)
            self.assertEqual(first, second)
            self.assertEqual(
                [item["relative_path"] for item in first["files"]],
                ["b.txt", "nested/a.txt"],
            )
            (root / "nested" / "a.txt").write_text("changed", encoding="utf-8")
            self.assertNotEqual(
                first["checkpoint_inventory_sha256"],
                audit.checkpoint_inventory(root)["checkpoint_inventory_sha256"],
            )

    def test_json_artifacts_are_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "audit.json"
            audit._atomic_write_json_new(path, {"status": "completed"})
            self.assertEqual(json.loads(path.read_text()), {"status": "completed"})
            with self.assertRaises(FileExistsError):
                audit._atomic_write_json_new(path, {"status": "overwritten"})

    def test_oom_detection_and_failure_payload_forbid_fallback(self) -> None:
        exception = torch.cuda.OutOfMemoryError("CUDA out of memory")
        self.assertTrue(audit._is_cuda_oom(exception))
        payload = audit._failure_payload(
            exception=exception,
            stage="single_forward_backward",
            request={"expected_k": 4},
        )
        self.assertEqual(payload["status"], "failed_cuda_oom")
        self.assertEqual(payload["exit_code"], 2)
        self.assertFalse(any(payload["fallback_policy"].values()))


class MainOrchestrationTests(unittest.TestCase):
    def _invoke_main(self, root: Path, *, oom: bool) -> tuple[int, object]:
        checkpoint = root / "checkpoint"
        metadata = root / "manifest_metadata.json"
        model_path = root / "base_model"
        output = root / "output"
        pinned = _example(audit.EXPECTED_SUBJECT_ID)
        pinned["audio_paths"] = [
            f"/audio/{name}" for name in audit.EXPECTED_AUDIO_SHA256
        ]
        config = {
            "training": {
                "bf16": True,
                "gradient_checkpointing": True,
                "learning_rate": 2e-4,
                "weight_decay": 0.0,
            }
        }

        def pinned_file(path, expected_sha256, description):
            del description
            return {
                "path": str(Path(path)),
                "size_bytes": 1,
                "sha256": expected_sha256,
            }

        determinism = {
            "seed": None,
            "cublas_workspace_config": ":4096:8",
            "python_hash_seed": "0",
            "deterministic_algorithms_enabled": True,
            "deterministic_algorithms_warn_only": True,
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
        }
        run_effect = (
            torch.cuda.OutOfMemoryError("CUDA out of memory in controlled backward")
            if oom
            else None
        )
        with (
            patch.object(audit, "_pinned_file", side_effect=pinned_file),
            patch.object(audit, "_load_checkpoint_config", return_value=(config, {})),
            patch.object(
                audit,
                "_resolve_input_examples",
                return_value=(
                    [pinned],
                    {"view_id": "legacy_deterministic_k4"},
                    {"manifest_hash": "logical-hash"},
                    root / "manifest.jsonl",
                    root / "partitions.json",
                ),
            ) as resolve_examples,
            patch.object(audit, "_canonical_example_sha256", return_value=audit.EXPECTED_EXAMPLE_SHA256),
            patch.object(audit, "_resolve_model_path", return_value=model_path),
            patch.object(
                audit,
                "checkpoint_inventory",
                return_value={
                    "checkpoint_inventory_sha256": audit.EXPECTED_CHECKPOINT_INVENTORY_SHA256,
                    "checkpoint_file_count": 1,
                    "files": [],
                },
            ),
            patch.object(
                audit,
                "_pinned_base_model_inventory",
                return_value={"verified_file_count": 7},
            ),
            patch.object(audit.torch.cuda, "is_available", return_value=True),
            patch.object(audit.torch.cuda, "empty_cache"),
            patch.object(audit, "set_seed"),
            patch.object(audit, "_determinism_metadata", return_value=determinism),
            patch.object(audit, "_safe_cuda_failure_memory", return_value={"peak": 123}),
            patch.object(audit, "load_processor", return_value=object()),
            patch("builtins.print"),
            patch.object(
                audit,
                "run_gpu_audit",
                side_effect=run_effect,
                return_value={"status": "completed"},
            ),
        ):
            code = audit.main(
                [
                    "--checkpoint-dir",
                    str(checkpoint),
                    "--manifest-metadata",
                    str(metadata),
                    "--model-name-or-path",
                    str(model_path),
                    "--output-dir",
                    str(output),
                ]
            )
        return code, resolve_examples

    def test_main_success_pins_legacy_resolver_and_writes_one_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            code, resolve_examples = self._invoke_main(root, oom=False)
            self.assertEqual(code, 0)
            self.assertTrue((root / "output" / "gradient_audit.json").is_file())
            self.assertFalse((root / "output" / "gradient_audit_failure.json").exists())
            self.assertEqual(resolve_examples.call_args.kwargs["partition"], "test")
            self.assertEqual(
                resolve_examples.call_args.kwargs["view_family"],
                audit.LEGACY_VIEW_FAMILY,
            )
            self.assertEqual(resolve_examples.call_args.kwargs["view_index"], 0)

    def test_main_cuda_oom_returns_nonzero_and_writes_only_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            code, _ = self._invoke_main(root, oom=True)
            self.assertEqual(code, 2)
            failure_path = root / "output" / "gradient_audit_failure.json"
            self.assertTrue(failure_path.is_file())
            self.assertFalse((root / "output" / "gradient_audit.json").exists())
            failure = json.loads(failure_path.read_text())
            self.assertEqual(failure["status"], "failed_cuda_oom")
            self.assertFalse(failure["fallback_policy"]["fallback_attempted"])


class BackwardAndParameterAuditTests(unittest.TestCase):
    def test_exactly_one_forward_backward_has_input_and_lora_gradients_without_step(self) -> None:
        model = _AuditModel()
        batch = {
            "input_features": torch.tensor([[1.0, 2.0]], dtype=torch.float32),
            "labels": torch.tensor([[-100, 7]], dtype=torch.long),
        }
        before = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        }
        loss, moved = audit.one_forward_backward(
            model,
            batch,
            device=torch.device("cpu"),
            use_bf16_autocast=False,
        )
        self.assertGreater(loss, 0.0)
        self.assertEqual(model.forward_calls, 1)
        self.assertTrue(moved["input_features"].requires_grad)
        self.assertIsNotNone(moved["input_features"].grad)
        self.assertGreater(float(moved["input_features"].grad.norm()), 0.0)
        self.assertTrue(all(layer.lora_A.grad is not None for layer in model.model.layers))
        for name, parameter in model.named_parameters():
            self.assertTrue(torch.equal(parameter.detach(), before[name]))

    def test_parameter_groups_report_freeze_optimizer_and_per_layer_gradients(self) -> None:
        model = _AuditModel()
        batch = {
            "input_features": torch.tensor([[1.0, 2.0]], dtype=torch.float32),
            "labels": torch.tensor([[-100, 7]], dtype=torch.long),
        }
        audit.one_forward_backward(
            model,
            batch,
            device=torch.device("cpu"),
            use_bf16_autocast=False,
        )
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad]
        )
        report = audit.parameter_audit(model, optimizer, decoder_layer_count=2)
        groups = report["primary_groups"]
        self.assertEqual(groups["frozen_audio_tower"]["requires_grad_status"], "none")
        self.assertEqual(groups["frozen_projector"]["optimizer_membership_status"], "none")
        self.assertEqual(groups["lora_overall"]["requires_grad_status"], "all")
        self.assertEqual(groups["lora_overall"]["gradient_status"], "all")
        self.assertGreater(groups["lora_overall"]["gradient_l2_norm"], 0.0)
        self.assertEqual(report["lora_per_decoder_layer"]["0"]["gradient_tensor_count"], 2)
        self.assertEqual(report["lora_per_decoder_layer"]["1"]["gradient_tensor_count"], 2)
        self.assertTrue(
            report["membership_invariants"][
                "trainable_parameter_names_equal_optimizer_parameter_names"
            ]
        )

    def test_target_metadata_reports_full_unmasked_span(self) -> None:
        batch = {
            "input_ids": torch.tensor([[10, 11, 41, 42]], dtype=torch.long),
            "labels": torch.tensor([[-100, -100, 41, 42]], dtype=torch.long),
        }
        report = audit.target_metadata(
            batch,
            _Processor(),
            "Non-depressed",
            "Non-depressed<|im_end|>\n",
        )
        self.assertEqual(report["masked_prompt_tokens"], 2)
        self.assertEqual(report["supervised_target_tokens"], 2)
        self.assertEqual(report["target_token_ids"], [41, 42])
        self.assertEqual(
            report["decoded_supervised_target"], "Non-depressed<|im_end|>\n"
        )
        self.assertTrue(report["decoded_target_equals_expected_full_suffix"])

    def test_target_metadata_rejects_gapped_mask_wrong_ids_and_wrong_label(self) -> None:
        processor = _Processor()
        gapped = {
            "input_ids": torch.tensor([[10, 41, 11, 42]], dtype=torch.long),
            "labels": torch.tensor([[-100, 41, -100, 42]], dtype=torch.long),
        }
        with self.assertRaisesRegex(ValueError, "contiguous"):
            audit.target_metadata(
                gapped,
                processor,
                "Non-depressed",
                "Non-depressed<|im_end|>\n",
            )

        mismatched = {
            "input_ids": torch.tensor([[10, 11, 41, 99]], dtype=torch.long),
            "labels": torch.tensor([[-100, -100, 41, 42]], dtype=torch.long),
        }
        with self.assertRaisesRegex(ValueError, "do not equal input_ids"):
            audit.target_metadata(
                mismatched,
                processor,
                "Non-depressed",
                "Non-depressed<|im_end|>\n",
            )

        valid = {
            "input_ids": torch.tensor([[10, 11, 41, 42]], dtype=torch.long),
            "labels": torch.tensor([[-100, -100, 41, 42]], dtype=torch.long),
        }
        with self.assertRaisesRegex(ValueError, "exact full training suffix"):
            audit.target_metadata(
                valid,
                processor,
                "Depressed",
                "Depressed<|im_end|>\n",
            )

    def test_training_state_validation_fails_before_a_noncompliant_backward(self) -> None:
        valid = {
            "model_training": True,
            "configured_bf16": True,
            "configured_gradient_checkpointing": True,
            "model_is_gradient_checkpointing": True,
            "model_use_cache": False,
            "base_model_use_cache": False,
        }
        audit._validate_training_state(valid)
        for key in (
            "model_training",
            "configured_bf16",
            "configured_gradient_checkpointing",
            "model_is_gradient_checkpointing",
        ):
            invalid = dict(valid)
            invalid[key] = False
            with self.assertRaises(RuntimeError, msg=key):
                audit._validate_training_state(invalid)
        invalid_cache = dict(valid)
        invalid_cache["model_use_cache"] = True
        with self.assertRaisesRegex(RuntimeError, "KV cache"):
            audit._validate_training_state(invalid_cache)


if __name__ == "__main__":
    unittest.main()
