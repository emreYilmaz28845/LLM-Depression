"""Validation harness and audit tests: fixture, gates, determinism, wheels."""
from __future__ import annotations

import json
import math

import pytest

from src.qwen38.audit import (
    audit_deployment,
    audit_wheelhouse,
    parse_wheel_tags,
    wheel_tag_errors,
)
from src.qwen38.validation import (
    compare_determinism,
    load_synthetic_cases,
    summarize_acceptance,
)

FIXTURE = "tests/fixtures/qwen38_synthetic_cases.jsonl"


class TestSyntheticFixture:
    def test_system_prompt_is_plain_string(self):
        from src.qwen38.validation import VALIDATION_SYSTEM_PROMPT

        assert isinstance(VALIDATION_SYSTEM_PROMPT, str)
        assert VALIDATION_SYSTEM_PROMPT.strip()

    def test_fixture_distribution(self):
        cases = load_synthetic_cases(FIXTURE)
        assert len(cases) == 64
        tr = [c for c in cases if c.language == "tr"]
        en = [c for c in cases if c.language == "en"]
        assert len(tr) == 32 and len(en) == 32
        for label in ("POSITIVE", "NEGATIVE", "NEUTRAL", "MIXED"):
            assert sum(1 for c in tr if c.expected_label == label) == 8
            assert sum(1 for c in en if c.expected_label == label) == 8

    def test_fixture_echoes_questions(self):
        cases = load_synthetic_cases(FIXTURE)
        for case in cases:
            assert case.answer_text.strip()
            assert case.required_question_concepts

    def test_fixture_rejects_bad_distribution(self, tmp_path):
        path = tmp_path / "bad.jsonl"
        rows = []
        for case in load_synthetic_cases(FIXTURE)[:63]:
            rows.append(case.to_dict())
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="expected 64"):
            load_synthetic_cases(path)


class TestDeterminism:
    def test_identical_passes_match(self):
        cases = load_synthetic_cases(FIXTURE)[:8]
        pass_a = [
            {
                "case_id": c.case_id,
                "parsed": {"label": "POSITIVE", "inferred_question": "What Makes You Happy?"},
            }
            for c in cases
        ]
        pass_b = [
            {
                "case_id": c.case_id,
                "parsed": {"label": "POSITIVE", "inferred_question": "what makes you happy"},
            }
            for c in cases
        ]
        ok, mismatches = compare_determinism(pass_a, pass_b)
        assert ok and not mismatches

    def test_label_mismatch_detected(self):
        cases = load_synthetic_cases(FIXTURE)[:4]
        pass_a = [
            {"case_id": c.case_id, "parsed": {"label": "POSITIVE", "inferred_question": "q"}}
            for c in cases
        ]
        pass_b = [
            {"case_id": c.case_id, "parsed": {"label": "NEGATIVE", "inferred_question": "q"}}
            for c in cases
        ]
        ok, mismatches = compare_determinism(pass_a, pass_b)
        assert not ok and mismatches

    def test_question_mismatch_detected(self):
        pass_a = [{"case_id": "syn-001", "parsed": {"label": "P", "inferred_question": "a b"}}]
        pass_b = [{"case_id": "syn-001", "parsed": {"label": "P", "inferred_question": "a c"}}]
        ok, mismatches = compare_determinism(pass_a, pass_b)
        assert not ok and mismatches


def _good_results():
    def make_case(case_id, label, question):
        return {
            "case_id": case_id,
            "json_valid": True,
            "schema_valid": True,
            "parsed": {"label": label, "inferred_question": question},
            "label_correct": True,
            "concepts_covered": True,
            "empty_output": False,
            "request_error": None,
            "repair_used": False,
            "ttft_seconds": 0.5,
            "e2e_seconds": 1.0,
            "completion_tokens": 20,
        }

    cases = [make_case(f"syn-{i:03d}", "POSITIVE", "q") for i in range(1, 65)]
    levels = {
        "c1_pass_a": list(cases),
        "c1_pass_b": list(cases),
        "c8": list(cases),
        "c16": list(cases),
        "c32": list(cases),
    }
    results = {
        "ready": True,
        "startup_seconds": 120.0,
        "levels": levels,
        "gpu_memory": {"sampled": True, "peak_mib": 48000},
        "model_revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    }
    restart = {
        "ready": True,
        "startup_seconds": 90.0,
        "levels": {"restart_subset": list(cases[:8])},
        "gpu_memory": {"sampled": True, "peak_mib": 48000},
    }
    return results, restart


def _env(overrides=None):
    env = {
        "python_major": 3,
        "python_minor": 10,
        "vllm": "0.25.1",
        "transformers": "5.8.0",
        "torch": "2.11.0",
        "torchvision": "0.26.0",
        "torchaudio": "2.11.0",
        "openai": "3.2.0",
        "huggingface_hub": "1.28.0",
    }
    if overrides:
        env.update(overrides)
    return env


class TestAcceptance:
    def test_full_gate_passes(self):
        results, restart = _good_results()
        acceptance = summarize_acceptance(
            results,
            restart,
            model_revision="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
            deployment_model_revision="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
            environment_versions=_env(),
            deployment_environment_versions=_env(),
            model_manifest_sha256="abc",
            deployment_model_manifest_sha256="abc",
            wheelhouse_manifest_sha256="def",
            deployment_wheelhouse_manifest_sha256="def",
        )
        assert acceptance["passed"]
        assert len(acceptance["checks"]) >= 20

    def test_not_ready_fails(self):
        results = {"ready": False, "error": "nope", "startup_seconds": 601, "levels": {}}
        acceptance = summarize_acceptance(
            results,
            None,
            model_revision="r",
            deployment_model_revision="r",
            environment_versions=_env(),
            deployment_environment_versions=_env(),
            model_manifest_sha256=None,
            deployment_model_manifest_sha256=None,
            wheelhouse_manifest_sha256=None,
            deployment_wheelhouse_manifest_sha256=None,
        )
        assert not acceptance["passed"]

    def test_label_accuracy_below_threshold_fails(self):
        results, restart = _good_results()
        for case in results["levels"]["c1_pass_a"][:10]:
            case["label_correct"] = False
        acceptance = summarize_acceptance(
            results,
            restart,
            model_revision="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
            deployment_model_revision="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
            environment_versions=_env(),
            deployment_environment_versions=_env(),
            model_manifest_sha256="abc",
            deployment_model_manifest_sha256="abc",
            wheelhouse_manifest_sha256="def",
            deployment_wheelhouse_manifest_sha256="def",
        )
        assert not acceptance["passed"]

    def test_manifest_mismatch_fails(self):
        results, restart = _good_results()
        acceptance = summarize_acceptance(
            results,
            restart,
            model_revision="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
            deployment_model_revision="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
            environment_versions=_env(),
            deployment_environment_versions=_env(),
            model_manifest_sha256="abc",
            deployment_model_manifest_sha256="XXX",
            wheelhouse_manifest_sha256="def",
            deployment_wheelhouse_manifest_sha256="def",
        )
        assert not acceptance["passed"]

    def test_nan_metrics_fail(self):
        results, restart = _good_results()
        results["levels"]["c8"][0]["e2e_seconds"] = math.nan
        acceptance = summarize_acceptance(
            results,
            restart,
            model_revision="r",
            deployment_model_revision="r",
            environment_versions=_env(),
            deployment_environment_versions=_env(),
            model_manifest_sha256="a",
            deployment_model_manifest_sha256="a",
            wheelhouse_manifest_sha256="b",
            deployment_wheelhouse_manifest_sha256="b",
        )
        assert not acceptance["passed"]


class TestWheelTags:
    @pytest.mark.parametrize(
        "filename",
        [
            "torch-2.11.0+cu130-cp310-cp310-manylinux_2_28_x86_64.whl",
            "torch-2.11.0+cu130-cp310-cp310-manylinux_2_34_x86_64.whl",
            "vllm-0.25.1-cp310-cp310-manylinux1_x86_64.whl",
            "greenlet-3.2.4-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
            "typing_extensions-4.15.0-py3-none-any.whl",
            "filelock-3.20.0-py3-none-any.whl",
            "jsonschema-4.27.0-cp39-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
            "packaging-25.0-cp38-abi3-manylinux_2_28_x86_64.whl",
            "soundfile-0.13.1-cp310-cp310-musllinux_2_17_x86_64.whl",
            "psutil-7.0.0-cp310-cp310-linux_x86_64.whl",
            "interegular-0.3.3-py37-none-any.whl",
            "attrs-25.0.0-py39-none-any.whl",
        ],
    )
    def test_acceptable_tags(self, filename):
        assert wheel_tag_errors(filename) == []

    @pytest.mark.parametrize(
        "filename",
        [
            "torch-2.11.0+cu130-cp310-cp310-manylinux_2_35_x86_64.whl",
            "torch-2.11.0+cu130-cp310-cp310-manylinux_2_36_x86_64.whl",
            "numpy-2.4.0-cp310-cp310-manylinux_2_28_aarch64.whl",
            "torch-2.11.0+cu130-cp311-cp311-manylinux_2_28_x86_64.whl",
            "torch-2.11.0+cu130-cp313-cp313-manylinux_2_28_x86_64.whl",
            "torch-2.11.0+cu130-cp310-cp310-win_amd64.whl",
            "torch-2.11.0+cu130-cp310-cp310-macosx_11_0_arm64.whl",
            "foo-1.0.tar.gz",
            "torch-2.11.0+cu130-cp310-cp310-manylinux_2_28_i686.whl",
        ],
    )
    def test_rejected_tags(self, filename):
        assert wheel_tag_errors(filename) != []

    def test_parse_wheel_tags(self):
        parsed = parse_wheel_tags("vllm-0.25.1-cp310-cp310-manylinux_2_28_x86_64.whl")
        assert parsed is not None
        py_tags, abi_tags, platform_tags = parsed
        assert py_tags == ["cp310"]
        assert platform_tags == ["manylinux_2_28_x86_64"]
        assert parse_wheel_tags("not-a-wheel") is None

    def test_audit_wheelhouse_accepts_binary_only(self, tmp_path):
        wheels = tmp_path / "wheels"
        wheels.mkdir()
        (wheels / "vllm-0.25.1-cp310-cp310-manylinux_2_28_x86_64.whl").write_bytes(b"x")
        audit = audit_wheelhouse(tmp_path)
        assert not audit["passed"]  # SHA256SUMS manifest is missing
        assert audit["wheel_count"] == 1

    def test_audit_wheelhouse_rejects_sdist(self, tmp_path):
        wheels = tmp_path / "wheels"
        wheels.mkdir()
        (wheels / "foo-1.0.tar.gz").write_bytes(b"x")
        audit = audit_wheelhouse(tmp_path)
        assert not audit["passed"]
        assert any("non-wheel" in error for error in audit["errors"])


class TestDeploymentAudit:
    def _build_deployment(self, tmp_path):
        deploy = tmp_path / "deploy"
        deployment_id = "qwen38_test"
        env_dir = deploy / deployment_id / "environment"
        env_dir.mkdir(parents=True)
        (env_dir / "runtime_versions.json").write_text(
            json.dumps(
                {
                    "python_major": 3,
                    "python_minor": 10,
                    "vllm": "0.25.1",
                    "transformers": "5.8.0",
                    "torch": "2.11.0",
                    "torchvision": "0.26.0",
                    "torchaudio": "2.11.0",
                    "openai": "3.2.0",
                    "huggingface_hub": "1.28.0",
                    "model_id": "Qwen/Qwen3.8-27B",
                    "model_revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (env_dir / "driver_probe.json").write_text(
            json.dumps({"driver_version": "595.71.05", "passed": True}) + "\n", encoding="utf-8"
        )
        import hashlib

        model_hash = hashlib.sha256(b"x").hexdigest()
        (env_dir / "model_sha256.txt").write_text(model_hash + "\n", encoding="utf-8")
        wheelhouse_hash = hashlib.sha256(b"x").hexdigest()
        (env_dir / "wheelhouse_sha256.txt").write_text(wheelhouse_hash + "\n", encoding="utf-8")
        model = tmp_path / "model"
        model.mkdir()
        (model / "config.json").write_bytes(b"x")
        (model / "SHA256SUMS").write_text(
            f"{model_hash}  config.json\n", encoding="utf-8"
        )
        wheelhouse = tmp_path / "wheelhouse"
        wheels = wheelhouse / "wheels"
        wheels.mkdir(parents=True)
        (wheels / "vllm-0.25.1-cp310-cp310-manylinux_2_28_x86_64.whl").write_bytes(b"x")
        (wheelhouse / "SHA256SUMS").write_text(
            f"{wheelhouse_hash}  wheels/vllm-0.25.1-cp310-cp310-manylinux_2_28_x86_64.whl\n",
            encoding="utf-8",
        )
        for tp in (1, 2, 4):
            attempt = deploy / deployment_id / "validation" / f"tp{tp}" / "attempt1"
            attempt.mkdir(parents=True)
            (attempt / "acceptance.json").write_text(
                json.dumps({"passed": True}) + "\n", encoding="utf-8"
            )
        selection = {
            "deployment_id": deployment_id,
            "model_id": "Qwen/Qwen3.8-27B",
            "model_revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
            "selected_tp": 1,
            "decision_rule": "rule3_select_tp1_eligible_within_2h",
            "candidate_results": {},
            "projected_requests": 142,
            "projected_wall_seconds": {},
            "measured_metrics_paths": {},
            "created_utc": "2026-08-18T00:00:00Z",
            "source_commit": "abc123",
        }
        (deploy / deployment_id / "serving_selection.json").write_text(
            json.dumps(selection) + "\n", encoding="utf-8"
        )
        return deploy, deployment_id, model, wheelhouse, env_dir

    def test_deployment_audit_passes(self, tmp_path):
        deploy, deployment_id, model, wheelhouse, env_dir = self._build_deployment(tmp_path)
        result = audit_deployment(
            deploy,
            deployment_id,
            model_dir=model,
            wheelhouse_dir=wheelhouse,
            environment_dir=env_dir,
            source_commit="abc123",
        )
        assert result["passed"], [c for c in result["checks"] if not c["passed"]]

    def test_deployment_audit_driver_fails(self, tmp_path):
        deploy, deployment_id, model, wheelhouse, env_dir = self._build_deployment(tmp_path)
        (env_dir / "driver_probe.json").write_text(
            json.dumps({"driver_version": "575.00", "passed": False}) + "\n", encoding="utf-8"
        )
        result = audit_deployment(
            deploy,
            deployment_id,
            model_dir=model,
            wheelhouse_dir=wheelhouse,
            environment_dir=env_dir,
        )
        assert not result["passed"]

    def test_deployment_audit_selection_mismatch_fails(self, tmp_path):
        deploy, deployment_id, model, wheelhouse, env_dir = self._build_deployment(tmp_path)
        selection_path = deploy / deployment_id / "serving_selection.json"
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        selection["selected_tp"] = 9
        selection_path.write_text(json.dumps(selection) + "\n", encoding="utf-8")
        result = audit_deployment(
            deploy,
            deployment_id,
            model_dir=model,
            wheelhouse_dir=wheelhouse,
            environment_dir=env_dir,
        )
        assert not result["passed"]
