"""The standalone head matrix: parent resolution and fail-closed planning.

A head job may only exist when its parent training attempt is identified by
recorded provenance: the reduced resolved config must match the current cell
config, the recorded training seed must match, the prompt hash and translation
notice version must match, and the adapter files must be present. Anything else
is either ``waiting_for_checkpoint`` or a refusal.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from src.data.prompt_context import resolve_system_prompt
from src.utils import load_yaml_with_overrides
from tools import qwen3_heads_matrix as heads

ROOT = Path(__file__).resolve().parents[1]
CELL_CONFIG = "configs/main/d3tec_text_only_harmonized_selmacrof1_likelihood_v1.yaml"
EN_CELL_CONFIG = (
    "configs/main/d3tec_text_only_harmonized_selmacrof1_likelihood_v1"
    "_promptcontext_v1_en_qwen38_27b.yaml"
)


@pytest.fixture()
def synthetic_cell(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(heads, "PROJECT_ROOT", tmp_path)
    for rel in (CELL_CONFIG, EN_CELL_CONFIG):
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
) -> Path:
    fold_dir = tmp_path / "run_root" / run_name / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    recorded = dict(cell_config)
    recorded["seed"] = seed if seed_override is None else seed_override
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
    (fold_dir / "run_config.yaml").write_text(
        yaml.safe_dump(
            {
                "config": recorded,
                "prompt_context": prompt_context,
                "tracking": {"attempt_id": "attempt-1"},
            }
        ),
        encoding="utf-8",
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
        },
        "extract": {"gpus": 4},
        "heads": {"depends_on": None},
    }
    failures = heads.check_matrix(matrix)
    assert any("attempt id" in failure for failure in failures)
    assert any("prompt hash" in failure for failure in failures)
    assert any("native route bound" in failure for failure in failures)
    assert any("extraction shape" in failure for failure in failures)
    assert any("extract dependency" in failure for failure in failures)
