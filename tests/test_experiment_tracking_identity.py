from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.experiment_tracking.constants import LEGACY_ATTEMPT_ID_ALGORITHM_VERSION
from src.experiment_tracking.identity import (
    artifact_id,
    deployed_source_sha256,
    evaluation_id,
    legacy_attempt_id,
    logical_run_id,
    new_attempt_id,
    reserve_attempt_dir,
    sanitize_logical_run_name,
    validate_attempt_id,
    wandb_run_id,
)

GIT_COMMIT = "1c2344f1d33e301978549748c5bf936319a43db6"

_LEGACY_PAYLOAD = {
    "relative_run_dir": "output_model/audio_text/daic/daic_run/fold_0",
    "fold": 0,
    "resolved_config_sha256": "a" * 64,
    "manifest_sha256": "b" * 64,
    "split_sha256": "c" * 64,
    "checkpoint_role": "best_model",
    "checkpoint_path": "best_model",
    "evaluation_artifact_sha256": "d" * 64,
}


def test_new_attempt_ids_validate_and_differ_across_calls() -> None:
    first = new_attempt_id("daic_rotary_k4_seed1337", GIT_COMMIT)
    second = new_attempt_id("daic_rotary_k4_seed1337", GIT_COMMIT)
    assert validate_attempt_id(first)
    assert validate_attempt_id(second)
    assert first != second
    assert validate_attempt_id("not-an-attempt-id") is False


def test_attempt_id_contains_sanitized_run_and_sha_prefix() -> None:
    value = new_attempt_id("Daic Rotary K4!", GIT_COMMIT)
    assert f"-{GIT_COMMIT[:8]}-" in value
    tail = value.split(f"-{GIT_COMMIT[:8]}-")[1]
    assert len(tail) == 8


def test_attempt_id_timestamp_is_compact_utc() -> None:
    fixed = datetime(2026, 8, 7, 11, 35, 22, tzinfo=timezone.utc)
    value = new_attempt_id("daic_rotary_k4_seed1337", GIT_COMMIT, at_utc=fixed)
    assert value.startswith("20260807T113522Z-daic_rotary_k4_seed1337-")


def test_attempt_id_rejects_invalid_git_sha() -> None:
    with pytest.raises(ValueError):
        new_attempt_id("daic_rotary_k4_seed1337", "short")
    with pytest.raises(ValueError):
        new_attempt_id("daic_rotary_k4_seed1337", "X" * 40)


def test_reserve_attempt_dir_collision_raises(tmp_path) -> None:
    target = tmp_path / "attempt_dir"
    reserve_attempt_dir(target)
    assert target.is_dir()
    with pytest.raises(FileExistsError):
        reserve_attempt_dir(target)


def test_legacy_attempt_ids_are_deterministic() -> None:
    assert legacy_attempt_id(_LEGACY_PAYLOAD) == legacy_attempt_id(dict(_LEGACY_PAYLOAD))


def test_legacy_attempt_id_changes_when_any_component_changes() -> None:
    expected = legacy_attempt_id(_LEGACY_PAYLOAD)
    variants = {
        "relative_run_dir": "output_model/text_only/cmdc/other_run/fold_0",
        "fold": 1,
        "resolved_config_sha256": "e" * 64,
        "manifest_sha256": "f" * 64,
        "split_sha256": "g" * 64,
        "checkpoint_role": "last_model",
        "checkpoint_path": "last_model",
        "evaluation_artifact_sha256": "h" * 64,
    }
    for key, value in variants.items():
        altered = dict(_LEGACY_PAYLOAD)
        altered[key] = value
        assert legacy_attempt_id(altered) != expected, key
    assert legacy_attempt_id({**_LEGACY_PAYLOAD, "resolved_config_sha256": None}) != expected


def test_legacy_attempt_id_has_versioned_prefix() -> None:
    value = legacy_attempt_id({"fold": 0})
    assert value.startswith(f"legacy-{LEGACY_ATTEMPT_ID_ALGORITHM_VERSION}-")
    suffix = value.rsplit("-", 1)[1]
    assert len(suffix) == 24
    assert set(suffix) <= set("0123456789abcdef")


def test_stable_ids_are_deterministic_and_input_sensitive() -> None:
    run_kwargs = dict(
        group_id="daic_rotary_k4_vs_joint_k4",
        logical_run_name="daic_rotary_k4_seed1337",
        dataset="daic",
        modality="audio_text",
        method="lora",
        seed=1337,
    )
    assert logical_run_id(**run_kwargs) == logical_run_id(**run_kwargs)
    assert logical_run_id(**run_kwargs) != logical_run_id(**{**run_kwargs, "seed": 2024})
    assert logical_run_id(**run_kwargs).startswith("lr-")

    artifact_kwargs = dict(
        attempt_id="20260807T113522Z-daic_rotary_k4_seed1337-a83f17c9-7f31a92b",
        fold=0,
        role="metrics",
        relative_path="best_model/standalone_eval/metrics_original_teacher_forced.json",
        artifact_sha256="d" * 64,
    )
    assert artifact_id(**artifact_kwargs) == artifact_id(**artifact_kwargs)
    assert artifact_id(**artifact_kwargs) != artifact_id(**{**artifact_kwargs, "fold": 1})
    assert artifact_id(**artifact_kwargs).startswith("art-")

    eval_kwargs = dict(
        attempt_id="20260807T113522Z-daic_rotary_k4_seed1337-a83f17c9-7f31a92b",
        fold=0,
        dataset="daic",
        split_name="test",
        split_protocol="fixed_train_val_test",
        checkpoint_role="best_model",
        checkpoint_path="best_model",
        backend="original_teacher_forced",
        evaluation_view="full_coverage_k4",
        aggregation="subject_level",
        metric_namespace="headline/binary_strict",
        metrics_artifact_sha256="d" * 64,
    )
    assert evaluation_id(**eval_kwargs) == evaluation_id(**eval_kwargs)
    assert evaluation_id(**eval_kwargs) != evaluation_id(
        **{**eval_kwargs, "evaluation_view": "fixed_k4"}
    )
    assert evaluation_id(**eval_kwargs).startswith("eval-")


def test_wandb_run_id() -> None:
    attempt_id = "20260807T113522Z-daic_rotary_k4_seed1337-a83f17c9-7f31a92b"
    assert wandb_run_id(attempt_id, 0) == f"{attempt_id}-fold0"


def test_deployed_source_sha256_is_deterministic_and_order_independent() -> None:
    records = [
        {"path": "a.py", "sha256": "b" * 64, "size_bytes": 1},
        {"path": "b.py", "sha256": "c" * 64, "size_bytes": 2},
    ]
    assert deployed_source_sha256(records) == deployed_source_sha256(list(reversed(records)))
    assert deployed_source_sha256(records) != deployed_source_sha256([records[0]])


def test_sanitize_logical_run_name() -> None:
    assert sanitize_logical_run_name("DAIC Rotary K4_v2") == "daic-rotary-k4_v2"
    assert sanitize_logical_run_name("daic_rotary_k4") == "daic_rotary_k4"
