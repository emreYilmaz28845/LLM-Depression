"""The standalone head matrix: parent resolution and fail-closed planning.

A head job may only exist when its parent training attempt is identified by
recorded provenance: the reduced resolved config must match the current cell
config (with the inventory's declaration-only ``resources.train_nodes`` rule),
the recorded training seed must match, the prompt hash and translation notice
version must match, the split seed must stay fixed, the adapter files must be
present, and the attempt must carry successful lifecycle and job evidence.
Anything else is ``waiting_for_checkpoint``, ``blocked_failed_parent`` or a
refusal.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from src.data.prompt_context import resolve_system_prompt
from src.utils import load_yaml_with_overrides
from tools import qwen3_heads_matrix as heads

ROOT = Path(__file__).resolve().parents[1]
CELL_CONFIG = "configs/main/d3tec_text_only_harmonized_selmacrof1_likelihood_v1.yaml"
AUDIO_CELL_CONFIG = "configs/main/daic_audio_only_harmonized_selmacrof1_likelihood_v1.yaml"
EN_CELL_CONFIG = (
    "configs/main/d3tec_text_only_harmonized_selmacrof1_likelihood_v1"
    "_promptcontext_v1_en_qwen38_27b.yaml"
)


@pytest.fixture()
def synthetic_cell(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(heads, "PROJECT_ROOT", tmp_path)
    for rel in (CELL_CONFIG, AUDIO_CELL_CONFIG, EN_CELL_CONFIG):
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / rel).read_bytes())
    return tmp_path


def _cell_config(tmp_path: Path, rel: str = CELL_CONFIG) -> dict:
    return load_yaml_with_overrides(tmp_path / rel, [])


def _write_run(
    tmp_path: Path,
    *,
    cell_config: dict,
    seed: int = 1337,
    run_name: str = "run_a",
    fold: int = 0,
    adapters: bool = True,
    prompt_hash: str | None = None,
    notice_version: str | None = None,
    notice_explicit: bool = False,
    seed_override: int | None = None,
    state: str = "REPORTABLE",
    failed_event: bool = False,
    attempt_id: str = "attempt-1",
    supersedes: str | None = None,
    drop_train_nodes: bool = False,
    world_size: int | None = None,
    write_sidecars: bool = True,
) -> Path:
    fold_dir = tmp_path / "run_root" / run_name / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    recorded = copy.deepcopy(cell_config)
    recorded["seed"] = seed if seed_override is None else seed_override
    if drop_train_nodes:
        resources = recorded.get("resources") or {}
        resources.pop("train_nodes", None)
        if resources:
            recorded["resources"] = resources
        else:
            recorded.pop("resources", None)
    if notice_explicit:
        recorded_notice = notice_version
    else:
        recorded_notice = (cell_config.get("prompt") or {}).get("translation_notice_version")
    prompt_context = {
        "version": "promptcontext_v1",
        "dataset_context": "d3tec",
        "input_modality": "text_only",
        "system_prompt_sha256": prompt_hash
        or hashlib.sha256(resolve_system_prompt(cell_config).encode("utf-8")).hexdigest(),
        "translation_notice_version": recorded_notice,
    }
    recorded_world = world_size
    if recorded_world is None:
        declared_nodes = int((cell_config.get("resources") or {}).get("train_nodes", 1) or 1)
        recorded_world = declared_nodes * 4
    payload = {
        "config": recorded,
        "prompt_context": prompt_context,
        "tracking": {"attempt_id": attempt_id},
        "training_strategy": {"world_size": recorded_world},
        "manifest_hash": "m" * 64,
    }
    (fold_dir / "run_config.yaml").write_text(
        yaml.safe_dump(payload), encoding="utf-8"
    )
    if write_sidecars:
        (fold_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "attempt_id": attempt_id,
                    "supersedes_attempt_id": supersedes,
                }
            ),
            encoding="utf-8",
        )
        (fold_dir / "status.json").write_text(
            json.dumps({"state": state}), encoding="utf-8"
        )
        events = [
            {
                "job_key": "train",
                "job_type": "train",
                "event_type": "COMPLETED",
                "status": "COMPLETED",
                "exit_code": "0:0",
            }
        ]
        if failed_event:
            events.append(
                {
                    "job_key": "train",
                    "job_type": "train",
                    "event_type": "FAILED",
                    "status": "FAILED",
                    "exit_code": "1:0",
                }
            )
        (fold_dir / "jobs.jsonl").write_text(
            "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
        )
    if adapters:
        checkpoint = fold_dir / "best_model"
        checkpoint.mkdir(parents=True, exist_ok=True)
        (checkpoint / "adapter_config.json").write_text("{}", encoding="utf-8")
        (checkpoint / "adapter_model.safetensors").write_bytes(b"weights")
    return fold_dir


def _resolve(tmp_path: Path, cell_config: dict, **kwargs) -> dict:
    return heads.resolve_parent(
        cell={"config": CELL_CONFIG},
        cell_config=cell_config,
        run_root=tmp_path / "run_root",
        fold=kwargs.pop("fold", 0),
        seed=kwargs.pop("seed", 1337),
        notice_version=kwargs.pop(
            "notice_version", (cell_config.get("prompt") or {}).get("translation_notice_version")
        ),
    )


def test_resolves_a_matching_parent_with_its_recorded_shape(synthetic_cell: Path) -> None:
    config = _cell_config(synthetic_cell)
    _write_run(synthetic_cell, cell_config=config)
    parent = _resolve(synthetic_cell, config)
    assert parent["status"] == "resolved"
    assert parent["run_name"] == "run_a"
    assert parent["attempt_id"] == "attempt-1"
    assert parent["extract_gpus"] == 1
    assert parent["checkpoint_adapter_model_sha256"]
    assert parent["translation_notice_version"] is None
    assert parent["selection"] == "unique_automatic"
    assert parent["parent_training_seed"] == 1337
    assert parent["head_seed"] == 1337


def test_missing_adapter_files_wait_for_the_checkpoint(synthetic_cell: Path) -> None:
    config = _cell_config(synthetic_cell)
    _write_run(synthetic_cell, cell_config=config, adapters=False)
    parent = _resolve(synthetic_cell, config)
    assert parent["status"] == "waiting_for_checkpoint"
    assert "adapter" in parent["reason"]


def test_other_seed_is_not_a_parent(synthetic_cell: Path) -> None:
    config = _cell_config(synthetic_cell)
    _write_run(synthetic_cell, cell_config=config, seed_override=2024)
    parent = _resolve(synthetic_cell, config, seed=1337)
    assert parent["status"] == "waiting_for_checkpoint"


def test_english_cell_refuses_a_native_checkpoint(synthetic_cell: Path) -> None:
    native = _cell_config(synthetic_cell)
    _write_run(synthetic_cell, cell_config=native, notice_version=None, notice_explicit=True)
    english = _cell_config(synthetic_cell, EN_CELL_CONFIG)
    # A run recorded from the English cell config but without the notice version
    # is exactly the native-checkpoint-under-an-English-cell case.
    _write_run(
        synthetic_cell,
        cell_config=english,
        run_name="run_en",
        notice_version=None,
        notice_explicit=True,
    )
    with pytest.raises(heads.HeadsMatrixError, match="translation notice version"):
        _resolve(synthetic_cell, english, notice_version="translation_notice_v1")


def test_ambiguous_parents_are_refused(synthetic_cell: Path) -> None:
    config = _cell_config(synthetic_cell)
    _write_run(synthetic_cell, cell_config=config, run_name="run_a")
    _write_run(synthetic_cell, cell_config=config, run_name="run_b")
    with pytest.raises(heads.HeadsMatrixError, match="ambiguous parent"):
        _resolve(synthetic_cell, config)


def test_prompt_hash_drift_is_refused(synthetic_cell: Path) -> None:
    config = _cell_config(synthetic_cell)
    _write_run(synthetic_cell, cell_config=config, prompt_hash="0" * 64)
    with pytest.raises(heads.HeadsMatrixError, match="prompt hash"):
        _resolve(synthetic_cell, config)


def test_failed_candidate_is_excluded_and_successful_retry_selected(synthetic_cell: Path) -> None:
    config = _cell_config(synthetic_cell)
    _write_run(
        synthetic_cell,
        cell_config=config,
        run_name="run_failed",
        state="FAILED",
        failed_event=True,
        attempt_id="attempt-failed",
    )
    _write_run(
        synthetic_cell,
        cell_config=config,
        run_name="run_retry",
        attempt_id="attempt-retry",
        supersedes="attempt-failed",
    )
    parent = _resolve(synthetic_cell, config)
    assert parent["status"] == "resolved"
    assert parent["attempt_id"] == "attempt-retry"
    assert parent["supersedes_attempt_id"] == "attempt-failed"
    assert any(item["run_name"] == "run_failed" for item in parent["excluded_attempts"])


def test_only_failed_candidate_blocks_the_key(synthetic_cell: Path) -> None:
    config = _cell_config(synthetic_cell)
    _write_run(
        synthetic_cell,
        cell_config=config,
        state="FAILED",
        failed_event=True,
        attempt_id="attempt-failed",
    )
    parent = _resolve(synthetic_cell, config)
    assert parent["status"] == "blocked_failed_parent"
    assert parent["excluded_attempts"]


def test_superseded_candidate_is_excluded(synthetic_cell: Path) -> None:
    config = _cell_config(synthetic_cell)
    _write_run(synthetic_cell, cell_config=config, run_name="run_old", attempt_id="attempt-old")
    _write_run(
        synthetic_cell,
        cell_config=config,
        run_name="run_new",
        attempt_id="attempt-new",
        supersedes="attempt-old",
    )
    parent = _resolve(synthetic_cell, config)
    assert parent["status"] == "resolved"
    assert parent["attempt_id"] == "attempt-new"
    assert any(item["attempt_id"] == "attempt-old" for item in parent["excluded_attempts"])


def test_declaration_only_train_nodes_is_accepted(synthetic_cell: Path) -> None:
    config = _cell_config(synthetic_cell, AUDIO_CELL_CONFIG)
    _write_run(
        synthetic_cell,
        cell_config=config,
        drop_train_nodes=True,
        world_size=8,
    )
    parent = heads.resolve_parent(
        cell={"config": AUDIO_CELL_CONFIG},
        cell_config=config,
        run_root=synthetic_cell / "run_root",
        fold=0,
        seed=1337,
        notice_version=None,
    )
    assert parent["status"] == "resolved"
    assert parent["declaration_only_shape_difference"] is True


def test_genuine_shape_mismatch_waits(synthetic_cell: Path) -> None:
    config = _cell_config(synthetic_cell, AUDIO_CELL_CONFIG)
    _write_run(
        synthetic_cell,
        cell_config=config,
        drop_train_nodes=True,
        world_size=4,
    )
    parent = heads.resolve_parent(
        cell={"config": AUDIO_CELL_CONFIG},
        cell_config=config,
        run_root=synthetic_cell / "run_root",
        fold=0,
        seed=1337,
        notice_version=None,
    )
    assert parent["status"] == "waiting_for_checkpoint"
    assert any(
        "resource shape" in item["reason"] for item in parent["excluded_attempts"]
    )


def _build_matrix(tmp_path: Path, **kwargs) -> dict:
    return heads.build_matrix(seeds=[1337], scan_roots=[], **kwargs)


def test_explicit_parent_map_resolves_with_reason(tmp_path: Path) -> None:
    config = load_yaml_with_overrides(ROOT / CELL_CONFIG, [])
    fold_dir = _write_run(tmp_path, cell_config=config, attempt_id="attempt-explicit")
    map_path = tmp_path / "parent_map.json"
    map_path.write_text(
        json.dumps(
            {
                "schema_version": heads.PARENT_MAP_SCHEMA,
                "entries": [
                    {
                        "route_id": "d3tec_text_only_native",
                        "config": CELL_CONFIG,
                        "parent_training_seed": 1337,
                        "fold": 0,
                        "fold_dir": str(fold_dir),
                        "attempt_id": "attempt-explicit",
                        "selection_reason": "canonical production run",
                        "excluded_duplicates": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    matrix = _build_matrix(tmp_path, parent_map_path=map_path)
    route = next(r for r in matrix["routes"] if r["route_id"] == "d3tec_text_only_native")
    job = next(j for j in route["jobs"] if j["fold"] == 0)
    assert job["parent_status"] == "resolved"
    assert job["parent"]["selection"] == "explicit_parent_map"
    assert job["parent"]["selection_reason"] == "canonical production run"
    assert job["parent"]["attempt_id"] == "attempt-explicit"


def test_explicit_parent_map_attempt_id_mismatch_fails(tmp_path: Path) -> None:
    config = load_yaml_with_overrides(ROOT / CELL_CONFIG, [])
    fold_dir = _write_run(tmp_path, cell_config=config, attempt_id="attempt-real")
    map_path = tmp_path / "parent_map.json"
    map_path.write_text(
        json.dumps(
            {
                "schema_version": heads.PARENT_MAP_SCHEMA,
                "entries": [
                    {
                        "route_id": "d3tec_text_only_native",
                        "config": CELL_CONFIG,
                        "parent_training_seed": 1337,
                        "fold": 0,
                        "fold_dir": str(fold_dir),
                        "attempt_id": "attempt-wrong",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(heads.HeadsMatrixError, match="does not match the recorded attempt id"):
        _build_matrix(tmp_path, parent_map_path=map_path)


def test_explicit_parent_map_unknown_key_fails(tmp_path: Path) -> None:
    map_path = tmp_path / "parent_map.json"
    map_path.write_text(
        json.dumps(
            {
                "schema_version": heads.PARENT_MAP_SCHEMA,
                "entries": [
                    {
                        "route_id": "not_a_route",
                        "config": CELL_CONFIG,
                        "parent_training_seed": 1337,
                        "fold": 0,
                        "fold_dir": str(tmp_path / "run_root" / "run_a" / "fold_0"),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(heads.HeadsMatrixError, match="outside the requested matrix"):
        _build_matrix(tmp_path, parent_map_path=map_path)


def test_campaign_root_discovers_managed_outputs(tmp_path: Path) -> None:
    config = load_yaml_with_overrides(ROOT / CELL_CONFIG, [])
    campaign_root = tmp_path / "campaign"
    cell_root = campaign_root / "text_only" / "d3tec"
    _write_run(cell_root, cell_config=config, run_name="run_managed")
    import shutil

    shutil.move(str(cell_root / "run_root" / "run_managed"), str(cell_root / "run_managed"))
    matrix = _build_matrix(tmp_path, campaign_root=campaign_root)
    route = next(r for r in matrix["routes"] if r["route_id"] == "d3tec_text_only_native")
    job = next(j for j in route["jobs"] if j["fold"] == 0)
    assert job["parent_status"] == "resolved"
    assert job["parent"]["run_name"] == "run_managed"


def test_cache_root_is_lane_owned(tmp_path: Path) -> None:
    config = load_yaml_with_overrides(ROOT / CELL_CONFIG, [])
    campaign_root = tmp_path / "campaign"
    cell_root = campaign_root / "text_only" / "d3tec"
    _write_run(cell_root, cell_config=config, run_name="run_managed")
    import shutil

    shutil.move(str(cell_root / "run_root" / "run_managed"), str(cell_root / "run_managed"))
    cache_root = tmp_path / "lane_cache"
    matrix = _build_matrix(tmp_path, campaign_root=campaign_root, cache_root=cache_root)
    route = next(r for r in matrix["routes"] if r["route_id"] == "d3tec_text_only_native")
    job = next(j for j in route["jobs"] if j["fold"] == 0)
    cache_dir = Path(job["extract"]["cache_dir"])
    assert cache_dir.is_relative_to(cache_root)
    assert "run_managed_fold_0_pseed1337_hseed1337" in str(cache_dir)


def test_matrix_covers_every_route_and_waits_without_checkpoints() -> None:
    matrix = heads.build_matrix(seeds=[1337], scan_roots=[])
    assert matrix["summary"]["routes"] == 23
    assert matrix["summary"]["jobs"] == 103
    assert matrix["summary"]["resolved"] == 0
    assert matrix["summary"]["waiting_for_checkpoint"] == 103
    assert heads.check_matrix(matrix) == []


def test_check_matrix_fails_closed_on_tampering() -> None:
    matrix = heads.build_matrix(seeds=[1337], scan_roots=[])
    text_route = next(route for route in matrix["routes"] if route["backend"] == "qwen38")
    text_route["jobs"][0] = {
        "route_id": text_route["route_id"],
        "seed": 1337,
        "fold": 0,
        "parent_status": "resolved",
        "parent": {
            "attempt_id": None,
            "prompt_sha256": None,
            "translation_notice_version": "translation_notice_v1",
            "parent_training_seed": 2024,
            "head_seed": 2024,
            "selection": None,
        },
        "extract": {"gpus": 4},
        "heads": {"depends_on": None, "seed": 2024},
    }
    failures = heads.check_matrix(matrix)
    assert any("attempt id" in failure for failure in failures)
    assert any("prompt hash" in failure for failure in failures)
    assert any("native route bound" in failure for failure in failures)
    assert any("extraction shape" in failure for failure in failures)
    assert any("extract dependency" in failure for failure in failures)
    assert any("parent training seed" in failure for failure in failures)
    assert any("head seed" in failure for failure in failures)
