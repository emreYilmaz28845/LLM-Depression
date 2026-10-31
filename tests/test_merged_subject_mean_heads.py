import json
from pathlib import Path

import numpy as np
import pytest

from baselines.merged_subject_mean_heads import (
    SubjectUnits,
    _fit_subject_head,
    balanced_training_set,
    class_balanced_hierarchical_weights,
    dataset_class_weights,
    oversample_dataset_class,
    smote_dataset_class,
    run,
)
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


def _imbalanced_rows(seed=0):
    """Dataset x class cells of different sizes (positives are every third subject)."""
    rows, vectors = _rows(subjects_per_dataset=12, seed=seed)
    for r in rows:
        r["label"] = int(int(r["subject_id"].rsplit("_s", 1)[1]) % 3 == 0)
    return rows, vectors


def test_oversampling_fills_every_cell_to_the_largest_and_is_seeded():
    rows, _ = _imbalanced_rows()
    units = SubjectUnits(rows)
    subjects = sorted(units.units)[:-5]  # uneven cells
    expanded = oversample_dataset_class(subjects, units, np.random.default_rng(3))
    cells = {}
    for s in expanded:
        cells[(units.dataset[s], units.labels[s])] = cells.get((units.dataset[s], units.labels[s]), 0) + 1
    assert len(set(cells.values())) == 1 and set(expanded) == set(subjects)
    for s in subjects:  # whole copies first, so copy counts differ by at most one inside a cell
        mates = [t for t in subjects if (units.dataset[t], units.labels[t]) == (units.dataset[s], units.labels[s])]
        counts = {expanded.count(t) for t in mates}
        assert max(counts) - min(counts) <= 1
    assert expanded == oversample_dataset_class(subjects, units, np.random.default_rng(3))


def test_oversampling_with_whole_copies_matches_dataset_class_weights():
    # 4 positives / 8 negatives per dataset: every cell reaches 8 by whole copies only,
    # so the copied fit and the weighted fit solve the same normalized objective.
    rows, vectors = _imbalanced_rows()
    units = SubjectUnits(rows)
    subjects = sorted(units.units)
    weighted = _fit_subject_head(vectors, units, subjects, 0.1, "dataset_class_weight")
    copied = _fit_subject_head(vectors, units, subjects, 0.1, "oversample_dataset_class", np.random.default_rng(0))
    assert copied[4]["fit_rows"] == 2 * 8 * len({units.dataset[s] for s in subjects})
    np.testing.assert_allclose(copied[2], weighted[2], atol=1e-6)
    assert copied[3] == pytest.approx(weighted[3], abs=1e-6)


def test_other_balancing_policies_and_holdout_untouched(tmp_path):
    rows, vectors = _imbalanced_rows()
    units = SubjectUnits(rows)
    subjects = sorted(units.units)
    _, none = balanced_training_set(subjects, units, "none", np.random.default_rng(0))
    assert set(none.tolist()) == {1.0}
    _, by_dataset = balanced_training_set(subjects, units, "dataset_weight", np.random.default_rng(0))
    for dataset in {units.dataset[s] for s in subjects}:
        assert sum(w for s, w in zip(subjects, by_dataset) if units.dataset[s] == dataset) == pytest.approx(0.2)
    with pytest.raises(ValueError, match="Unknown balancing"):
        balanced_training_set(subjects, units, "smote", np.random.default_rng(0))
    features = _features(tmp_path / "features", rows, vectors)
    summary = run(features, tmp_path / "out", "lora", heads=("subjmean_l2",),
                  balancing="oversample_dataset_class", oversample_seed=5)
    head = summary["heads"]["subjmean_l2"]
    assert head["balancing"] == {"policy": "oversample_dataset_class", "oversample_seed": 5}
    holdout = {json.loads(line)["subject_id"] for line in (features / "outer_holdout_rows.jsonl").read_text().splitlines()}
    assert sum(m["num_subjects"] for m in head["metrics_by_dataset"].values()) == len(holdout)


def test_smote_fills_cells_with_points_between_cell_neighbours():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(14, 3))
    y = np.asarray([1] * 4 + [0] * 6 + [1] * 2 + [0] * 2)
    groups = ["a"] * 10 + ["b"] * 4
    xs, ys, gs = smote_dataset_class(x, y, groups, np.random.default_rng(1))
    cells = {}
    for g, label in zip(gs, ys.tolist()):
        cells[(g, label)] = cells.get((g, label), 0) + 1
    assert set(cells.values()) == {6}
    np.testing.assert_array_equal(xs[:14], x)
    # A synthetic point of cell ("b", 1) lies on the segment between its two members.
    a, b = x[10], x[11]
    for point, g, label in zip(xs[14:], gs[14:], ys[14:].tolist()):
        if (g, label) == ("b", 1):
            t = np.dot(point - a, b - a) / np.dot(b - a, b - a)
            np.testing.assert_allclose(point, a + t * (b - a), atol=1e-12)
            assert -1e-12 <= t <= 1 + 1e-12
    again = smote_dataset_class(x, y, groups, np.random.default_rng(1))
    np.testing.assert_array_equal(again[0], xs)


def test_class_balanced_paper_weights_split_each_dataset_evenly():
    rows, _ = _imbalanced_rows()
    weights = class_balanced_hierarchical_weights(rows)
    assert weights.mean() == pytest.approx(1.0)
    for dataset in DATASETS:
        pos = sum(w for r, w in zip(rows, weights) if r["dataset"] == dataset and r["label"] == 1)
        neg = sum(w for r, w in zip(rows, weights) if r["dataset"] == dataset and r["label"] == 0)
        assert pos == pytest.approx(neg)
        assert pos + neg == pytest.approx(len(rows) / len(DATASETS))


def test_smote_and_class_balanced_paper_run_end_to_end(tmp_path):
    rows, vectors = _imbalanced_rows()
    features = _features(tmp_path / "features", rows, vectors)
    summary = run(features, tmp_path / "out", "lora", heads=("subjmean_l2", "paper_logreg_class_balanced"),
                  balancing="smote_dataset_class", oversample_seed=2)
    head = summary["heads"]["subjmean_l2"]
    assert head["balancing"] == {"policy": "smote_dataset_class", "oversample_seed": 2}
    assert len(set(head["fit_info"]["fit_cell_counts"].values())) == 1
    assert set(summary["heads"]["paper_logreg_class_balanced"]["metrics_by_dataset"]) == set(DATASETS)
