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
from src.merged.runtime import load_merged_config

ROOT = Path(__file__).resolve().parents[1]
MERGED = ROOT / "configs/experiments/merged"
POOLED = {
    "native_text_only": MERGED / "symmetric_merged_qwen3_pooled_native_text_only.yaml",
    "native_audio_only": MERGED / "symmetric_merged_qwen3_pooled_native_audio_only.yaml",
    "native_audio_text": MERGED / "symmetric_merged_qwen3_pooled_native_audio_text.yaml",
    "english_text_only": MERGED / "symmetric_merged_qwen3_pooled_english_text_only.yaml",
    "english_audio_text": MERGED / "symmetric_merged_qwen3_pooled_english_audio_text.yaml",
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


def test_head_support_covers_every_backend_family() -> None:
    """The code path is ready everywhere; per-route production readiness is the guard's."""
    assert is_qwen3_backend("qwen38") is True
    assert is_qwen3_backend("qwen3omni") is True
    assert is_qwen3_backend("qwen2audio") is False
    assert head_support_ready("qwen38") is True
    assert head_support_ready("qwen3omni") is True
    assert head_support_ready("qwen2audio") is True
    assert head_support_ready(None) is True


def test_merged_feature_dimensions_must_match_the_backend() -> None:
    from src.merged.postprocess import validate_feature_dimensions

    validate_feature_dimensions({5120}, "qwen38")
    validate_feature_dimensions({2048}, "qwen3omni")
    validate_feature_dimensions(set(), "qwen38")
    validate_feature_dimensions({1234}, "unlisted_backend")
    with pytest.raises(ValueError, match="does not match the recorded hidden size"):
        validate_feature_dimensions({2048}, "qwen38")


def test_english_audio_text_contract_mirrors_the_english_text_contract() -> None:
    """DAIC keeps its native English input; the other four cells carry the notice."""
    text = _merged("english_text_only")
    audio = _merged("english_audio_text")
    assert audio["name"] == "symmetric_merged_qwen3_pooled_english_audio_text"
    assert audio["modality"] == "audio_text"
    assert audio["model_backend"] == "qwen3omni"
    assert audio["recipe_id"].endswith("_en")
    assert (audio["execution"] or {}).get("postprocess_gpus") == 4
    assert (audio["training"] or {}).get("strategy") == "fsdp"
    assert (audio["protocol_settings"] or {}).get("selection_metric") == "mean_dataset_macro_f1"
    assert audio["seed"] == text["seed"]
    assert [component["name"] for component in audio["components"]] == [
        component["name"] for component in text["components"]
    ]
    assert audio["components"][0]["config"] == (
        "configs/main/daic_audio_text_harmonized_selmacrof1_likelihood_v1.yaml"
    )
    assert "manifests_harmonized/daic" in audio["components"][0]["manifest_path"]
    for component in audio["components"][1:]:
        config = yaml.safe_load((ROOT / component["config"]).read_text(encoding="utf-8"))
        assert component["config"].endswith("_en_qwen3omni_30b_a3b.yaml"), component["name"]
        assert config["transcripts"]["variant"] == "english", component["name"]
        assert (
            config["prompt"]["translation_notice_version"]
            == "translation_notice_v1"
        ), component["name"]
        assert "manifests_harmonized_en" in component["manifest_path"], component["name"]
    assert audio["output_dirs"]["merged_root"].endswith("qwen3_pooled_english/audio_text")
    assert audio["output_dirs"]["run_root"].endswith("qwen3_pooled_english_likelihood/audio_text")


def test_route_readiness_covers_exactly_the_five_pooled_contracts() -> None:
    expected_names = {f"symmetric_merged_qwen3_pooled_{name}" for name in (
        "native_text_only",
        "native_audio_only",
        "native_audio_text",
        "english_text_only",
        "english_audio_text",
    )}
    assert set(QWEN3_CONTRACT_READINESS) == expected_names
    for name, path in POOLED.items():
        config = _merged(name)
        smoke = merged_route_decision(config, stage="smoke")
        assert smoke["declared"] is True
        assert smoke["allowed"] is True
        # Every route passed its bounded hidden-feature audit, so its head kind is open.
        assert smoke["head_ready"] is True
        assert smoke["head_deferred_reason"] is None
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


def test_merged_roots_honor_absolute_overrides(tmp_path) -> None:
    """An absolute merged root override must not be relocated onto the checkout.

    ``resolve_project_path`` moves an absolute path that does not exist yet onto
    the current project root; a runtime override can point outside the checkout
    and be created later by the job, so the runtime layout helpers must keep it
    literal while ${PROJECT_ROOT}-relative values still resolve normally.
    """

    from src.merged.runtime import merged_aux_root, merged_fold_root, protocol_artifact_path

    absolute = tmp_path / "elsewhere" / "run_root"
    config = {"output_dirs": {"run_root": str(absolute), "merged_root": str(absolute / "merged")}}
    assert merged_fold_root(config, run_id="r", stage="smoke", fold=0) == absolute / "r" / "smoke" / "fold_0"
    assert merged_aux_root(config, run_id="r", stage="smoke", fold=0) == absolute / "merged" / "r" / "smoke" / "fold_0"
    assert protocol_artifact_path(config) == absolute / "merged" / "merged_protocol.json"


def test_audio_and_text_routes_resolve_their_declared_gpu_shapes() -> None:
    text = _merged("native_text_only")
    audio = _merged("native_audio_text")
    assert text["execution"]["postprocess_gpus"] == 1
    assert audio["execution"]["postprocess_gpus"] == 4
    for config in (text, audio):
        assert config["training"]["strategy"] == "fsdp"
        assert config["training"]["activation_offload"] == "cpu"
        assert config["status"] in {"smoke_only", "execute_verified"}


def test_every_qwen3_omni_merged_route_runs_the_two_node_lane() -> None:
    """Any Qwen3-Omni route declares two four-GPU nodes with accumulation 16.

    The shape matches the standalone audio lane, whose baselines were submitted
    that way; the text contracts keep one node with accumulation 32. Both keep
    the FSDP effective global batch of 128.
    """
    for slug in ("native_audio_only", "native_audio_text", "english_audio_text"):
        config = _merged(slug)
        execution = config["execution"]
        training = config["training"]
        assert execution["train_nodes"] == 2, slug
        assert execution["qwen_gpus"] == 4, slug
        assert training["gradient_accumulation_steps"] == 16, slug
        assert (
            execution["train_nodes"]
            * execution["qwen_gpus"]
            * training["gradient_accumulation_steps"]
            == 128
        ), slug
    for slug in ("native_text_only", "english_text_only"):
        config = _merged(slug)
        assert int((config["execution"] or {}).get("train_nodes", 1)) == 1, slug
        assert config["training"]["gradient_accumulation_steps"] == 32, slug


def test_readiness_table_records_the_shape_every_route_was_verified_in() -> None:
    """The guard's verified shape must match what the shipped contract declares.

    Otherwise a route could be opened on evidence recorded for another lane, and
    the drift would only show up after production jobs were submitted.
    """
    for name, entry in QWEN3_CONTRACT_READINESS.items():
        config = _merged(name.removeprefix("symmetric_merged_qwen3_pooled_"))
        declared = {
            "train_nodes": int((config["execution"] or {}).get("train_nodes", 1)),
            "gpus_per_node": int((config["execution"] or {}).get("qwen_gpus", 4)),
            "gradient_accumulation_steps": int(
                (config["training"] or {}).get("gradient_accumulation_steps", 1)
            ),
        }
        assert entry["verified_shape"] == declared, name
        assert declared["train_nodes"] * declared["gpus_per_node"] * declared[
            "gradient_accumulation_steps"
        ] == 128, name


def test_guard_refuses_a_production_run_in_an_unverified_shape() -> None:
    """An override that keeps the effective batch but changes the lane is refused.

    One node with accumulation 32 gives the same effective global batch as the
    verified two-node lane with 16, so only an explicit shape check can stop a
    production run from using the other shape's evidence. The bounded smoke stage
    stays allowed: that is how a route is verified in the first place.
    """
    config = _merged("native_audio_text")
    verified = merged_route_decision(config, stage="cv")
    assert verified["shape_verified"] is True
    assert verified["allowed"] is True
    assert verified["declared_shape"] == {
        "train_nodes": 2,
        "gpus_per_node": 4,
        "gradient_accumulation_steps": 16,
    }

    overridden = copy.deepcopy(config)
    overridden["execution"]["train_nodes"] = 1
    overridden["training"]["gradient_accumulation_steps"] = 32
    decision = merged_route_decision(overridden, stage="cv")
    assert decision["allowed"] is False
    assert decision["shape_verified"] is False
    assert "was verified as" in decision["reason"]
    for stage in ("cv", "final"):
        assert merged_route_decision(overridden, stage=stage)["allowed"] is False
    smoke = merged_route_decision(overridden, stage="smoke")
    assert smoke["allowed"] is True
    assert smoke["shape_verified"] is False


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
    assert "NNODES=1" in export
    assert "--gres=gpu:4" in captured["argv"]
    # The call site must prefer the job's resolved tokens over the registry list.
    source = (ROOT / "scripts/submit_symmetric_merged.py").read_text(encoding="utf-8")
    assert 'overrides=job.get("overrides") or registry.get("overrides") or []' in source


def test_two_node_train_job_requests_both_nodes_and_exports_the_rendezvous(monkeypatch) -> None:
    import scripts.submit_symmetric_merged as planner

    captured: dict[str, list[str]] = {}

    def fake_check_output(arguments, cwd=None, text=None):
        captured["argv"] = list(arguments)
        return "23456\n"

    monkeypatch.setattr(planner.subprocess, "check_output", fake_check_output)
    job = {
        "kind": "train",
        "config": "/deployed/config.yaml",
        "stage": "smoke",
        "fold": 0,
        "run_id": "r",
        "modality": "audio_text",
        "model_backend": "qwen3omni",
        "resource": {"nodes": 2, "gpus": 4, "cpus": 80, "time": 10},
    }
    job_id = planner._submit_job(
        job,
        worker=Path("scripts/run_symmetric_merged_train_slurm.sh"),
        dependency_id=None,
        throttle_dependency_id=None,
    )
    assert job_id == "23456"
    argv = [str(argument) for argument in captured["argv"]]
    assert "--nodes=2" in argv
    # One Slurm task per node holds the whole node's CPUs and GPUs, matching the
    # worker's per-node srun step.
    assert "--ntasks=2" in argv
    assert "--ntasks-per-node=1" in argv
    assert "--cpus-per-task=80" in argv
    assert "--gres=gpu:4" in argv
    export = next(argument for argument in argv if argument.startswith("--export="))
    assert "NNODES=2" in export
    assert "NPROC_PER_NODE=4" in export


def test_pooled_contracts_declare_explicit_split_and_head_seeds() -> None:
    """The approved three-seed contract fixes both seeds in every contract.

    The split seed must not follow the top-level training seed (that would
    change fold/inner-validation membership between seeds), and the classifier
    seed must stay 1337 for every parent checkpoint. Both are declared in the
    generated contracts so no resolver falls back.
    """

    from src.merged.heads import resolve_fixed_head_seed
    from src.merged.protocol import resolve_protocol_split_seed

    for name, path in POOLED.items():
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert raw["protocol_settings"]["split_seed"] == 1337, name
        assert raw["heads"]["fixed_seed"] == 1337, name
        resolved = load_merged_config(
            path,
            [
                "--set=seed=7",
                "--set=heads.fixed_seed=1337",
                "--set=protocol_settings.split_seed=1337",
            ],
        )
        assert resolve_protocol_split_seed(resolved) == 1337, name
        assert resolve_fixed_head_seed(resolved) == 1337, name


def test_pooled_contract_generator_reproduces_the_disk_contracts(tmp_path) -> None:
    """The supported generator workflow must stay the source of truth.

    ``--check`` compares the derived contracts with the checked-in files and
    fails on any drift outside the allowed diff, so this catches an edit that
    bypasses the generator (including a future accidental recipe change).
    """

    from scripts import build_qwen3_pooled_merged_configs as merged_configs

    assert merged_configs.main(["--check", "--audit-output", str(tmp_path / "audit.json")]) == 0


def test_submit_job_exports_the_project_local_hidden_dependencies(monkeypatch) -> None:
    """Head jobs must carry the explicit QWEN_HIDDEN_DEPS path through sbatch.

    The tracked deployment does not include ``.deps/``; an unexported shell
    variable would leave the head worker without xgboost/scikit-learn.
    """

    import scripts.submit_symmetric_merged as planner

    captured: dict[str, list[str]] = {}

    def fake_check_output(arguments, cwd=None, text=None):
        captured["argv"] = list(arguments)
        return "12345\n"

    monkeypatch.setattr(planner.subprocess, "check_output", fake_check_output)
    monkeypatch.setenv(
        "QWEN_HIDDEN_DEPS",
        "/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression/.deps/qwen_hidden",
    )
    job = {
        "kind": "head",
        "config": "/deployed/config.yaml",
        "stage": "cv",
        "fold": 0,
        "run_id": "r",
        "modality": "text_only",
        "model_backend": "qwen38",
        "resource": {"gpus": 0, "cpus": 20, "time": 10},
    }
    job_id = planner._submit_job(
        job,
        worker=Path("scripts/run_symmetric_merged_head_slurm.sh"),
        dependency_id=None,
        throttle_dependency_id=None,
    )
    assert job_id == "12345"
    export = next(argument for argument in captured["argv"] if str(argument).startswith("--export="))
    assert (
        "QWEN_HIDDEN_DEPS=/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression/.deps/qwen_hidden"
        in export
    )
