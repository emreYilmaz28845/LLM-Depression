"""Qwen3 merged contract tests: evaluation contract, resources, readiness, overrides.

The decision rule (``evaluation.sample_prediction_mode``) and the evidence view
(``evaluation.evaluation_view``) are different fields and are validated
separately per component. The four Qwen3 pooled contracts must declare both and
use likelihood; legacy merged contracts keep the historical teacher-forced
decision rule and are not re-interpreted.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from scripts.submit_symmetric_merged import (
    QWEN3_CONTRACT_READINESS,
    _normalized_extra_overrides,
    _runtime_override_tokens,
    merged_route_decision,
)
from src.merged.configuration import (
    head_support_ready,
    is_qwen3_backend,
    model_config,
    validate_evaluation_contract,
    validate_merged_resources,
)

ROOT = Path(__file__).resolve().parents[1]
MERGED = ROOT / "configs/experiments/merged"
POOLED = {
    "native_text_only": MERGED / "symmetric_merged_qwen3_pooled_native_text_only.yaml",
    "native_audio_only": MERGED / "symmetric_merged_qwen3_pooled_native_audio_only.yaml",
    "native_audio_text": MERGED / "symmetric_merged_qwen3_pooled_native_audio_text.yaml",
    "english_text_only": MERGED / "symmetric_merged_qwen3_pooled_english_text_only.yaml",
}


def _merged(name: str) -> dict:
    return yaml.safe_load(POOLED[name].read_text(encoding="utf-8"))


def _records(config: dict) -> list[dict]:
    return [
        {
            "dataset": str(component["name"]),
            "config": yaml.safe_load((ROOT / component["config"]).read_text(encoding="utf-8")),
        }
        for component in config["components"]
    ]


def test_pooled_contracts_pass_the_strict_evaluation_contract() -> None:
    for name, path in POOLED.items():
        config = _merged(name)
        records = _records(config)
        contract = validate_evaluation_contract(config, records)
        assert contract["qwen3_family"] is True, name
        assert contract["declared"] is True, name
        assert contract["sample_prediction_mode"] == "likelihood", name
        assert contract["evaluation_view"] == "harmonized_all_windows_full_coverage", name
        resources = validate_merged_resources(config, records)
        expected_gpus = 1 if config["modality"] == "text_only" else 4
        assert resources["eval_gpus_per_node"] == expected_gpus, name
        assert resources["sharded"] is (expected_gpus > 1), name
        resolved = model_config(config, records)
        assert resolved["evaluation"]["sample_prediction_mode"] == "likelihood", name
        assert resolved["resources"]["eval_gpus_per_node"] == expected_gpus, name
        assert resolved["merged_evaluation_contract"]["evaluation_view"], name
        assert path.is_file()


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda evaluation: evaluation.pop("sample_prediction_mode"), "must declare evaluation.sample_prediction_mode"),
        (lambda evaluation: evaluation.pop("evaluation_view"), "must declare evaluation.evaluation_view"),
        (lambda evaluation: evaluation.pop("headline_mode"), "must declare evaluation.headline_mode"),
        (
            lambda evaluation: evaluation.update({"evaluation_view": "some_other_view"}),
            "must agree on one evaluation.evaluation_view",
        ),
        (
            lambda evaluation: evaluation.update({"sample_prediction_mode": "original_teacher_forced"}),
            "must use evaluation.sample_prediction_mode='likelihood'",
        ),
        (
            lambda evaluation: evaluation.update({"headline_mode": "original_teacher_forced"}),
            "must keep evaluation.headline_mode equal to",
        ),
    ],
)
def test_invalid_qwen3_component_declarations_fail_closed(mutation, message) -> None:
    config = _merged("native_text_only")
    records = _records(config)
    mutation(records[2]["config"]["evaluation"])
    with pytest.raises(ValueError, match=message.replace("'", "'")):
        validate_evaluation_contract(config, records)


def test_legacy_contracts_keep_the_historical_decision_rule() -> None:
    records = [{"dataset": name, "config": {"dataset": name}} for name in ("daic", "cmdc", "turkish", "d3tec", "androids_interview")]
    contract = validate_evaluation_contract({}, records)
    assert contract["declared"] is False
    assert contract["sample_prediction_mode"] == ""
    resolved = model_config({"modality": "text_only"}, records)
    assert resolved["evaluation"]["sample_prediction_mode"] == "original_teacher_forced"
    assert resolved["evaluation"]["headline_mode"] == "original_teacher_forced"


def test_fsdp_resource_mismatch_fails_closed() -> None:
    config = _merged("native_audio_text")
    records = _records(config)
    config = copy.deepcopy(config)
    config["execution"]["postprocess_gpus"] = 1
    with pytest.raises(ValueError, match="disagrees with the component"):
        validate_merged_resources(config, records)


def test_head_support_is_deferred_for_qwen3_backends() -> None:
    assert is_qwen3_backend("qwen38") is True
    assert is_qwen3_backend("qwen3omni") is True
    assert is_qwen3_backend("qwen2audio") is False
    assert head_support_ready("qwen38") is False
    assert head_support_ready("qwen3omni") is False
    assert head_support_ready("qwen2audio") is True


def test_route_readiness_covers_exactly_the_four_pooled_contracts() -> None:
    expected_names = {f"symmetric_merged_qwen3_pooled_{name}" for name in (
        "native_text_only",
        "native_audio_only",
        "native_audio_text",
        "english_text_only",
    )}
    assert set(QWEN3_CONTRACT_READINESS) == expected_names
    for name, path in POOLED.items():
        config = _merged(name)
        smoke = merged_route_decision(config, stage="smoke")
        assert smoke["declared"] is True
        assert smoke["allowed"] is True
        assert smoke["head_ready"] is False
        assert smoke["head_deferred_reason"] == "Qwen3 merged head support prerequisite incomplete"
        for stage in ("cv", "final"):
            decision = merged_route_decision(config, stage=stage)
            production_ready = QWEN3_CONTRACT_READINESS[config["name"]]["production_ready"]
            assert decision["allowed"] is production_ready
            if not production_ready:
                assert decision["reason"].startswith("Qwen3 merged FSDP/postprocess prerequisite incomplete")
        assert path.is_file()


def test_runtime_override_tokens_repoint_component_inputs() -> None:
    config = _merged("native_audio_text")
    tokens = _runtime_override_tokens(
        config,
        input_root="/prebuilt/checkout",
        pooled_runtime_root="/prebuilt/pooled_runtime",
    )
    assert "--set=components.0.manifest_path=/prebuilt/checkout/outputs/manifests_harmonized/daic/daic_manifest.jsonl" in tokens
    turkish_index = next(
        index for index, component in enumerate(config["components"]) if component["name"] == "turkish"
    )
    assert (
        f"--set=components.{turkish_index}.manifest_path="
        "/prebuilt/pooled_runtime/manifests/turkish/turkish_manifest.jsonl" in tokens
    )
    assert (
        f"--set=components.{turkish_index}.metadata_path="
        "/prebuilt/pooled_runtime/splits/turkish/turkish_manifest_metadata.json" in tokens
    )
    assert _normalized_extra_overrides(["a.b=1", "--set=x=2", " "]) == ["--set=a.b=1", "--set=x=2"]


def test_audio_and_text_routes_resolve_their_declared_gpu_shapes() -> None:
    text = _merged("native_text_only")
    audio = _merged("native_audio_text")
    assert text["execution"]["postprocess_gpus"] == 1
    assert audio["execution"]["postprocess_gpus"] == 4
    for config in (text, audio):
        assert config["training"]["strategy"] == "fsdp"
        assert config["training"]["activation_offload"] == "cpu"
        assert config["status"] in {"smoke_only", "execute_verified"}


def test_worker_exports_the_job_level_resolved_overrides(monkeypatch) -> None:
    """A deployed worker must receive the per-config resolved token array.

    The registry-level ``overrides`` list only holds the explicit extra --set
    tokens; the component input redirection lives on the job's own resolved
    tokens. Exporting the wrong list silently falls back to PROJECT_ROOT-relative
    component manifests inside the deployment.
    """

    import base64
    import json

    import scripts.submit_symmetric_merged as planner

    captured: dict[str, list[str]] = {}

    def fake_check_output(arguments, cwd=None, text=None):
        captured["argv"] = list(arguments)
        return "12345\n"

    monkeypatch.setattr(planner.subprocess, "check_output", fake_check_output)
    job = {
        "kind": "train",
        "config": "/deployed/config.yaml",
        "stage": "smoke",
        "fold": 0,
        "run_id": "r",
        "modality": "text_only",
        "model_backend": "qwen38",
        "resource": {"gpus": 4, "cpus": 80, "time": 10},
        "overrides": ["--set=components.0.manifest_path=/prebuilt/x.jsonl"],
    }
    job_id = planner._submit_job(
        job,
        worker=Path("scripts/run_symmetric_merged_train_slurm.sh"),
        dependency_id=None,
        throttle_dependency_id=None,
        overrides=job["overrides"],
    )
    assert job_id == "12345"
    export = next(argument for argument in captured["argv"] if str(argument).startswith("--export="))
    encoded = export.split("OVERRIDES_JSON_B64=", 1)[1].split(",", 1)[0]
    assert json.loads(base64.b64decode(encoded).decode("utf-8")) == job["overrides"]
    assert "NPROC_PER_NODE=4" in export
    assert "--gres=gpu:4" in captured["argv"]
    # The call site must prefer the job's resolved tokens over the registry list.
    source = (ROOT / "scripts/submit_symmetric_merged.py").read_text(encoding="utf-8")
    assert 'overrides=job.get("overrides") or registry.get("overrides") or []' in source
