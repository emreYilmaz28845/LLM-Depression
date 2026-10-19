from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from tools import qwen3_late_fusion as lf


def write_scores(path: Path, rows: list[tuple[str, int, float]], *, score_margin: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if score_margin:
            writer.writerow(["subject_id", "label", "prediction", "score_margin"])
            for subject, label, margin in rows:
                writer.writerow([subject, label, int(margin > 0), margin])
        else:
            writer.writerow(["subject_id", "label", "prediction", "dep_score", "non_score"])
            for subject, label, margin in rows:
                writer.writerow([subject, label, int(margin > 0), margin - 1.0, -1.0])


def build_tree(tmp_path: Path, *, drop: str | None = None) -> dict[str, Path]:
    roots = {"native": tmp_path / "native", "androids_mergedfolds": tmp_path / "amf", "merged": tmp_path / "merged"}
    rng = np.random.default_rng(0)
    for dataset in lf.CV_DATASETS + lf.TEST_DATASETS:
        folds = [0] if dataset in lf.TEST_DATASETS else range(5)
        for fold in folds:
            subjects = [(f"{dataset}{fold}s{i}", i % 2) for i in range(6)]
            for regime in lf.REGIMES:
                for modality in lf.MODALITIES:
                    rows = [(s, y, float((1 if y else -1) + rng.normal(0, 1.5)) or 0.1) for s, y in subjects]
                    if drop == f"{dataset}{fold}{regime}{modality}":
                        rows = rows[1:]
                    path = lf.source_path(roots, 7, dataset, fold, regime, modality)
                    write_scores(path, rows, score_margin=(dataset == "turkish" and modality == "text_only"))
    return roots


def test_subject_margin_prefers_score_margin_and_falls_back_to_difference():
    assert lf.subject_margin({"score_margin": "1.5", "dep_score": "-9", "non_score": "0"}) == 1.5
    assert lf.subject_margin({"score_margin": "", "dep_score": "-1", "non_score": "-3"}) == 2.0


def test_read_scores_rejects_margin_that_disagrees_with_prediction(tmp_path: Path):
    path = tmp_path / "p.csv"
    path.write_text("subject_id,label,prediction,dep_score,non_score\nA,1,0,-1.0,-3.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="disagrees"):
        lf.read_scores(path)


def test_load_matched_joins_six_models_and_strips_dataset_prefix(tmp_path: Path):
    rows, files = lf.load_matched(build_tree(tmp_path), 7)
    assert len(rows) == 4 * 5 * 6 + 6
    assert len(files) == (4 * 5 + 1) * 6
    assert all(set(lf.MODELS) <= set(row) for row in rows)


def test_load_matched_fails_closed_on_subject_mismatch(tmp_path: Path):
    roots = build_tree(tmp_path, drop="cmdc2multi_enaudio_text")
    with pytest.raises(ValueError, match="different subject set"):
        lf.load_matched(roots, 7)


def test_source_path_uses_mergedfold_rerun_for_androids_and_final_stage_for_daic(tmp_path: Path):
    roots = {"native": tmp_path / "n", "androids_mergedfolds": tmp_path / "a", "merged": tmp_path / "m"}
    assert "q3amf_text_only_native_s7_f3_r1" in str(lf.source_path(roots, 7, "androids_interview", 3, "uni", "text_only"))
    assert "/final/fold_0/qwen/daic/" in str(lf.source_path(roots, 7, "daic", 0, "multi_en", "audio_text"))
    assert "qwen3_pooled_english" in str(lf.source_path(roots, 7, "cmdc", 1, "multi_en", "text_only"))


def row(label: int, margins: list[float], fold: int = 0) -> dict:
    return {"dataset": "cmdc", "fold": fold, "subject_id": "x", "label": label, **dict(zip(lf.MODELS, margins))}


def test_training_free_oracle_vote_and_mean():
    rows = [
        row(1, [2, -1, -1, -1, -1, -1]),   # one model right -> oracle right, vote wrong
        row(0, [1, 1, 1, -1, -1, -1]),     # 3-3 tie, mean decides
        row(1, [-1, -1, -1, -1, -1, -1]),  # all wrong -> oracle wrong
    ]
    out = lf.training_free(rows)
    assert out["oracle_any6"] == [1, 0, 0]
    assert out["vote_all6"][0] == 0 and out["vote_all6"][2] == 0
    mean = 1 / (1 + np.exp(-np.asarray([1, 1, 1, -1, -1, -1.0])))
    assert out["vote_all6"][1] == int(mean.mean() >= 0.5)
    agree = lf.agreement(rows)
    assert agree == {"subjects": 3, "all_six_agree": 1, "all_six_correct": 0, "all_six_wrong": 1}


def test_leave_fold_out_never_trains_on_the_test_fold(monkeypatch):
    rows = [row(i % 2, [float(i % 2 * 2 - 1)] * 6, fold=i % 5) for i in range(30)]
    for i, r in enumerate(rows):
        r["subject_id"] = f"s{i}"
    seen = []

    def fake(kind, seed, x_train, y_train, x_test):
        seen.append((len(x_train), len(x_test)))
        assert len(x_train) + len(x_test) == len(rows)
        return np.full(len(x_test), 0.9)

    monkeypatch.setattr(lf, "fit_predict", fake)
    out = lf.leave_fold_out(rows, "logreg", 1)
    assert out == [1] * 30
    assert seen == [(24, 6)] * 5


def test_run_is_deterministic(tmp_path: Path):
    rows, _ = lf.load_matched(build_tree(tmp_path), 7)
    first = lf.run(rows, 1337)
    second = lf.run(rows, 1337)
    assert first == second
    assert {"logreg_per_dataset", "xgb_pooled"} <= set(first["cmdc"])
    assert {"logreg_loo", "xgb_transfer"} <= set(first["daic"])
