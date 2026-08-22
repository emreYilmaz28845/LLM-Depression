from __future__ import annotations

from pathlib import Path

from src.data import cmdc


def test_cmdc_fallback_uses_declared_split_seed(monkeypatch, tmp_path: Path) -> None:
    for subject_id in ("MDD01", "MDD02", "MDD03", "MDD04", "MDD05", "HC01", "HC02", "HC03", "HC04", "HC05"):
        subject_dir = tmp_path / subject_id
        subject_dir.mkdir()
        (subject_dir / "Q1.wav").write_bytes(b"RIFF")
        (subject_dir / "Q1.txt").write_text("text", encoding="utf-8")

    # Force the same deterministic fallback used when the workbook cannot
    # prove complete, non-overlapping coverage.
    monkeypatch.setattr(cmdc, "build_cmdc_official_folds", lambda *args, **kwargs: {})
    observed: list[int] = []
    original = cmdc.assign_stratified_group_folds

    def capture(labels, *, n_splits, seed):
        observed.append(seed)
        return original(labels, n_splits=n_splits, seed=seed)

    monkeypatch.setattr(cmdc, "assign_stratified_group_folds", capture)
    result = cmdc.build_cmdc_manifest(
        {
            "dataset_root": str(tmp_path),
            "seed": 7,
            "split": {"seed": 1337},
        },
        {},
    )

    assert observed == [1337]
    assert len(result["folds"]) == 5
