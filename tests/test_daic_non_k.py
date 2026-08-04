from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_builder():
    path = ROOT / "scripts/build_daic_non_k_matrix.py"
    spec = importlib.util.spec_from_file_location("build_daic_non_k_matrix", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_non_k_matrix_has_declared_twelve_cell_contract() -> None:
    module = load_builder()
    spec = yaml.safe_load((ROOT / "configs/experiments/daic_non_k/matrix.yaml").read_text())
    payload = module.expand(spec, "unit_non_k", "production")
    assert payload["kind_counts"] == {"train": 12, "evaluation": 12}
    train = [task for task in payload["tasks"] if task["kind"] == "train"]
    assert sum(task["group"] == "joint" for task in train) == 4
    assert sum(task["group"] == "independent" for task in train) == 8
    assert len({task["output_root"] for task in train}) == 12


def test_non_k_smoke_covers_both_groups_and_matched_policies() -> None:
    module = load_builder()
    spec = yaml.safe_load((ROOT / "configs/experiments/daic_non_k/matrix.yaml").read_text())
    payload = module.expand(spec, "unit_non_k_smoke", "smoke")
    train = [task for task in payload["tasks"] if task["kind"] == "train"]
    assert payload["kind_counts"] == {"train": 4, "evaluation": 4}
    assert {task["group"] for task in train} == {"joint", "independent"}
    assert any(task["overrides"]["data.train_chunk_policy"] == "fixed_k" for task in train)
    assert all(task["overrides"]["training.num_train_epochs"] == 1 for task in train)
