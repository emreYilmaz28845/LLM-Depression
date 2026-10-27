import json
from pathlib import Path

import numpy as np
import pytest

from baselines.merged_subject_mean_heads import SubjectUnits, dataset_class_weights, run
from src.merged.protocol import DATASETS


def _rows(subjects_per_dataset=9, units=2, windows=2, seed=0):
    rng = np.random.default_rng(seed)
    rows, vectors = [], []
    for dataset in DATASETS:
        for i in range(subjects_per_dataset):
            subject, label = f"{dataset}_s{i}", int(i % 2 == 0)
            for unit in range(units):
                for window in range(windows):
                    component = f"{subject}_{unit}_{window}"
                    rows.append({"dataset": dataset, "sample_id": f"{dataset}::{component}",
                                 "component_sample_id": component, "subject_id": f"{dataset}::{subject}",
                                 "label": label, "response_id": f"{dataset}::{subject}::{unit}",
                                 "prompt_sha256": "p", "checkpoint": "ckpt"})
                    vectors.append(rng.normal(size=6) + 1.5 * label)
    return rows, np.asarray(vectors, dtype=np.float32)


def _features(path: Path, rows, vectors, holdout_every=3) -> Path:
    path.mkdir(parents=True)
    subjects = sorted({r["subject_id"] for r in rows})
    holdout = set(subjects[::holdout_every])
    for name, keep in (("outer_train", lambda r: r["subject_id"] not in holdout),
                       ("outer_holdout", lambda r: r["subject_id"] in holdout)):
        index = [i for i, r in enumerate(rows) if keep(r)]
        np.savez(path / f"{name}.npz", vectors=vectors[index])
        (path / f"{name}_rows.jsonl").write_text("".join(json.dumps(rows[i]) + "\n" for i in index))
    (path / "feature_metadata.json").write_text(json.dumps({"stage": "cv", "fold": 0, "modality": "audio_only"}))
    return path


def _frozen_cache(path: Path, rows, vectors, prompt="p") -> Path:
    path.mkdir(parents=True)
    np.savez(path / "outer_train.npz", vectors=vectors)
    (path / "outer_train_rows.jsonl").write_text("".join(
        json.dumps({"sample_id": r["component_sample_id"], "prompt_sha256": prompt}) + "\n" for r in rows))
    (path / "extraction_metadata.json").write_text(json.dumps({"weights_source": "base_model_no_adapter",
                                                               "base_model": "m"}))
    return path


def test_dataset_class_weights_give_equal_dataset_mass_and_balanced_classes():
    rows, _ = _rows()
    units = SubjectUnits(rows)
    subjects = sorted(units.units)
    weights = dict(zip(subjects, dataset_class_weights(subjects, units)))
    for dataset in DATASETS:
        mine = [s for s in subjects if units.dataset[s] == dataset]
        positive = sum(weights[s] for s in mine if units.labels[s] == 1)
        negative = sum(weights[s] for s in mine if units.labels[s] == 0)
        assert positive == pytest.approx(negative)
        assert positive + negative == pytest.approx(1.0 / len(DATASETS))


def test_lora_run_fits_both_heads_per_dataset(tmp_path):
    rows, vectors = _rows()
    features = _features(tmp_path / "features", rows, vectors)
    summary = run(features, tmp_path / "out", "lora")
    for head in ("paper_logreg", "subjmean_l2"):
        metrics = summary["heads"][head]["metrics_by_dataset"]
        assert set(metrics) == set(DATASETS)
        assert np.mean([m["macro_f1"] for m in metrics.values()]) > 0.7
        assert (tmp_path / "out" / head / "predictions_subject_level.csv").is_file()
    selection = summary["heads"]["subjmean_l2"]["selection"]
    assert selection["inner_folds"] == 3 and len(selection["candidates"]) == 8


def test_frozen_source_with_identical_vectors_matches_lora(tmp_path):
    rows, vectors = _rows()
    features = _features(tmp_path / "features", rows, vectors)
    half = len(rows) // 2
    caches = [_frozen_cache(tmp_path / "c0", rows[:half], vectors[:half]),
              _frozen_cache(tmp_path / "c1", rows[half - 4:], vectors[half - 4:])]
    lora = run(features, tmp_path / "lora", "lora")
    frozen = run(features, tmp_path / "frozen", "frozen", caches)
    assert frozen["frozen_audit"]["duplicate_samples"] == 4
    assert frozen["frozen_audit"]["duplicate_max_abs_diff"] == 0.0
    for head in ("paper_logreg", "subjmean_l2"):
        assert frozen["heads"][head]["metrics_by_dataset"] == lora["heads"][head]["metrics_by_dataset"]


def test_frozen_source_fails_on_missing_sample_or_prompt_mismatch(tmp_path):
    rows, vectors = _rows()
    features = _features(tmp_path / "features", rows, vectors)
    with pytest.raises(ValueError, match="1 missing"):
        run(features, tmp_path / "a", "frozen", [_frozen_cache(tmp_path / "c0", rows[1:], vectors[1:])])
    with pytest.raises(ValueError, match="prompt mismatches"):
        run(features, tmp_path / "b", "frozen", [_frozen_cache(tmp_path / "c1", rows, vectors, prompt="q")])


def test_refuses_to_overwrite_and_checks_source_arguments(tmp_path):
    rows, vectors = _rows()
    features = _features(tmp_path / "features", rows, vectors)
    run(features, tmp_path / "out", "lora", heads=("paper_logreg",))
    with pytest.raises(ValueError, match="overwrite"):
        run(features, tmp_path / "out", "lora")
    with pytest.raises(ValueError, match="required"):
        run(features, tmp_path / "x", "frozen")
    with pytest.raises(ValueError, match="only to the frozen"):
        run(features, tmp_path / "y", "lora", [tmp_path])
