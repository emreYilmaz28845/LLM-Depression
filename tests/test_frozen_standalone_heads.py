import json
from pathlib import Path

import numpy as np
import pytest

from baselines.frozen_standalone_heads import (
    PooledCache,
    inner_cv_folds,
    run_route,
    unit_key,
)


def _cache(path: Path, subjects, *, units=3, windows=2, seed=0, base=True) -> None:
    rng = np.random.default_rng(seed)
    path.mkdir(parents=True)
    rows, vectors = [], []
    for subject, label in subjects:
        for unit in range(units):
            for window in range(windows):
                rows.append({"sample_id": f"{subject}_{unit}_{window}", "subject_id": subject,
                             "label": label, "response_id": f"{subject}::q{unit}"})
                vectors.append(rng.normal(size=6) + 1.5 * label)
    half = len(rows) // 2
    for name, part in (("outer_train", slice(0, half)), ("final_eval", slice(half, None))):
        np.savez(path / f"{name}.npz", vectors=np.asarray(vectors[part], dtype=np.float32))
        (path / f"{name}_rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows[part]))
    meta = {"weights_source": "base_model_no_adapter" if base else "adapter", "base_model": "m",
            "manifest_sha256": "x", "condition": "c", "input_modality": "audio_only", "pooling": "p"}
    (path / "extraction_metadata.json").write_text(json.dumps(meta))


def _subjects(n=40):
    return [(f"s{i:02d}", int(i % 3 == 0)) for i in range(n)]


def _split(tmp_path: Path, name: str, payload: dict) -> Path:
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(payload))
    return path


def test_unit_key_prefers_response_then_condition_then_sample():
    assert unit_key({"response_id": "r", "question_condition": "q", "sample_id": "s"}) == "r"
    assert unit_key({"response_id": "", "question_condition": "q", "sample_id": "s"}) == "q"
    assert unit_key({"sample_id": "s"}) == "s"


def test_train_val_test_and_train_val_protocols(tmp_path):
    subjects = _subjects()
    _cache(tmp_path / "cache", subjects)
    ids = [s for s, _ in subjects]
    splits = []
    for fold in range(2):
        test = ids[fold * 8:(fold + 1) * 8]
        rest = [s for s in ids if s not in test]
        splits.append(_split(tmp_path, f"tvt{fold}", {
            "train_subject_ids": rest[:24], "selection_subject_ids": rest[24:], "final_eval_subject_ids": test}))
    summary = run_route(cache_dirs=[tmp_path / "cache"], dev_caches=None, split_files=splits,
                        protocol="train_val_test", output_dir=tmp_path / "out_tvt", route={})
    assert len(summary["folds"]) == 2 and summary["folds"][0]["report"] == 8
    assert summary["fold_mean"]["macro_f1"] > 0.7

    tv = _split(tmp_path, "tv0", {"train_subject_ids": ids[:30], "selection_subject_ids": ids[30:],
                                  "final_eval_subject_ids": []})
    summary = run_route(cache_dirs=[tmp_path / "cache"], dev_caches=None, split_files=[tv],
                        protocol="train_val", output_dir=tmp_path / "out_tv", route={})
    selection = json.loads((tmp_path / "out_tv" / "fold_0" / "selection.json").read_text())
    assert selection["selection_source"].startswith("inner_5fold")
    assert sorted(sum(selection["inner_folds"], [])) == sorted(ids[:30])
    assert summary["folds"][0]["report"] == 10


def test_inner_folds_are_stratified():
    subjects = [f"s{i}" for i in range(23)]
    y = np.asarray([int(i < 8) for i in range(23)])
    folds = inner_cv_folds(subjects, y)
    positives = [sum(int(s[1:]) < 8 for s in fold) for fold in folds]
    assert max(positives) - min(positives) <= 1
    assert sorted(sum(folds, [])) == sorted(subjects)


def test_refuses_adapter_cache_and_overlap(tmp_path):
    _cache(tmp_path / "lora", _subjects(), base=False)
    split = _split(tmp_path, "s", {"train_subject_ids": ["s00"], "selection_subject_ids": ["s01"],
                                   "final_eval_subject_ids": ["s02"]})
    with pytest.raises(ValueError, match="frozen base"):
        run_route(cache_dirs=[tmp_path / "lora"], dev_caches=None, split_files=[split],
                  protocol="train_val_test", output_dir=tmp_path / "o1", route={})
    _cache(tmp_path / "base", _subjects())
    bad = _split(tmp_path, "bad", {"train_subject_ids": ["s00", "s01"], "selection_subject_ids": ["s01"],
                                   "final_eval_subject_ids": ["s02"]})
    with pytest.raises(ValueError, match="overlap"):
        run_route(cache_dirs=[tmp_path / "base"], dev_caches=None, split_files=[bad],
                  protocol="train_val_test", output_dir=tmp_path / "o2", route={})


def test_hierarchical_subject_mean_weights_units_equally(tmp_path):
    _cache(tmp_path / "c", [("a", 0), ("b", 1)], units=2, windows=1)
    cache = PooledCache([(tmp_path / "c", "outer_train"), (tmp_path / "c", "final_eval")])
    index, weight = cache.row_weights(["a", "b"])
    assert np.isclose(weight.sum(), 2.0)


def _fold_cache(path: Path, train, held, *, seed=0) -> None:
    """A per-fold LoRA-style cache: outer_train = train subjects, final_eval = held subjects."""
    rng = np.random.default_rng(seed)
    path.mkdir(parents=True)
    for name, subjects in (("outer_train", train), ("final_eval", held)):
        rows, vectors = [], []
        for subject, label in subjects:
            for unit in range(2):
                rows.append({"sample_id": f"{subject}_{unit}", "subject_id": subject, "label": label,
                             "response_id": f"{subject}::q{unit}"})
                vectors.append(rng.normal(size=6) + 1.5 * label)
        np.savez(path / f"{name}.npz", vectors=np.asarray(vectors, dtype=np.float32))
        (path / f"{name}_rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    meta = {"base_model": "m", "manifest_sha256": "x", "condition": "c", "input_modality": "audio_only",
            "pooling": "p", "checkpoint_dir": str(path), "adapter_sha256": "a"}
    (path / "extraction_metadata.json").write_text(json.dumps(meta))


def test_adapter_mode_uses_one_cache_per_fold(tmp_path):
    subjects = _subjects()
    splits, caches = [], []
    for fold in range(2):
        val = subjects[fold * 8:(fold + 1) * 8]
        train = [s for s in subjects if s not in val]
        splits.append(_split(tmp_path, f"f{fold}", {"train_subject_ids": [s for s, _ in train],
                                                     "selection_subject_ids": [s for s, _ in val],
                                                     "final_eval_subject_ids": []}))
        _fold_cache(tmp_path / f"lora{fold}", train, val, seed=fold)
        caches.append(tmp_path / f"lora{fold}")
    with pytest.raises(ValueError, match="frozen base"):
        run_route(cache_dirs=caches, dev_caches=None, split_files=splits, protocol="train_val",
                  output_dir=tmp_path / "o0", route={})
    with pytest.raises(ValueError, match="one cache per split"):
        run_route(cache_dirs=caches[:1], dev_caches=None, split_files=splits, protocol="train_val",
                  output_dir=tmp_path / "o1", route={}, allow_adapter=True)
    with pytest.raises(ValueError, match="do not match"):
        run_route(cache_dirs=caches[::-1], dev_caches=None, split_files=splits, protocol="train_val",
                  output_dir=tmp_path / "o2", route={}, allow_adapter=True)
    summary = run_route(cache_dirs=caches, dev_caches=None, split_files=splits, protocol="train_val",
                        output_dir=tmp_path / "o3", route={}, allow_adapter=True)
    assert summary["feature_source"] == "lora_adapter"
    assert [f["report"] for f in summary["folds"]] == [8, 8]


def test_train_cv_test_ignores_selection_subjects(tmp_path):
    subjects = _subjects()
    _cache(tmp_path / "cache", subjects)
    ids = [s for s, _ in subjects]
    split = _split(tmp_path, "d", {"train_subject_ids": ids[:24], "selection_subject_ids": ids[24:32],
                                   "final_eval_subject_ids": ids[32:]})
    summary = run_route(cache_dirs=[tmp_path / "cache"], dev_caches=None, split_files=[split],
                        protocol="train_cv_test", output_dir=tmp_path / "o", route={})
    selection = json.loads((tmp_path / "o" / "fold_0" / "selection.json").read_text())
    assert sorted(sum(selection["inner_folds"], [])) == sorted(ids[:24])
    assert summary["folds"][0]["report"] == 8 and summary["folds"][0]["selection"] == 0


def test_accepts_handcrafted_mfcc_cache_with_same_head(tmp_path):
    subjects = _subjects()
    _cache(tmp_path / "base", subjects)
    _cache(tmp_path / "mfcc", subjects)
    meta_path = tmp_path / "mfcc" / "extraction_metadata.json"
    meta = json.loads(meta_path.read_text())
    meta.pop("weights_source")
    meta["model_backend"] = "opensmile_mfcc"
    meta_path.write_text(json.dumps(meta))
    ids = [s for s, _ in subjects]
    split = _split(tmp_path, "tv", {"train_subject_ids": ids[:30], "selection_subject_ids": ids[30:]})
    base = run_route(cache_dirs=[tmp_path / "base"], dev_caches=None, split_files=[split],
                     protocol="train_val", output_dir=tmp_path / "o_base", route={})
    mfcc = run_route(cache_dirs=[tmp_path / "mfcc"], dev_caches=None, split_files=[split],
                     protocol="train_val", output_dir=tmp_path / "o_mfcc", route={})
    assert mfcc["fold_mean"] == base["fold_mean"] and mfcc["feature_source"] == "handcrafted"
    with pytest.raises(ValueError, match="handcrafted"):
        run_route(cache_dirs=[tmp_path / "mfcc"], dev_caches=None, split_files=[split], protocol="train_val",
                  output_dir=tmp_path / "o_adapter", route={}, allow_adapter=True)
