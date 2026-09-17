"""Atomic-write regression tests for the shared JSON/JSONL writers.

Concurrent manifest builds into the same experiment-runtime path corrupted the
manifest when the writers used a plain open("w"); these writers must replace
the target atomically and leave no temporary files behind.
"""

from pathlib import Path

from src.utils import atomic_write_path, save_json, write_jsonl


def _temp_leftovers(directory: Path) -> list[Path]:
    return [path for path in directory.iterdir() if ".tmp." in path.name]


def test_write_jsonl_replaces_content_atomically(tmp_path: Path) -> None:
    target = tmp_path / "rows.jsonl"
    target.write_text("stale\n", encoding="utf-8")
    write_jsonl([{"a": 1}, {"a": 2}], target)
    assert target.read_text(encoding="utf-8") == '{"a": 1}\n{"a": 2}\n'
    assert _temp_leftovers(tmp_path) == []


def test_save_json_replaces_content_atomically(tmp_path: Path) -> None:
    target = tmp_path / "meta.json"
    target.write_text("stale", encoding="utf-8")
    save_json({"k": "v"}, target)
    assert target.read_text(encoding="utf-8") == '{\n  "k": "v"\n}\n'
    assert _temp_leftovers(tmp_path) == []


def test_atomic_write_path_is_unique_per_call(tmp_path: Path) -> None:
    target = tmp_path / "shared.json"
    first = atomic_write_path(target)
    second = atomic_write_path(target)
    assert first != second
    assert first.parent == target.parent
