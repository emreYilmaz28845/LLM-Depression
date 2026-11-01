from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from tools import qwen3_late_fusion as lf


def write_scores(path: Path, rows: list[tuple[str, int, float, float]], *, margin_only: bool = False) -> None:
    """Subject file; ``margin_only`` mimics Turkish text (score_margin + sample-level dep/non)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if margin_only:
            writer.writerow(["subject_id", "label", "prediction", "score_margin"])
            for subject, label, dep, non in rows:
                writer.writerow([subject, label, int(dep > non), dep - non])
        else:
            writer.writerow(["subject_id", "label", "prediction", "dep_score", "non_score"])
            for subject, label, dep, non in rows:
                writer.writerow([subject, label, int(dep > non), dep, non])
    if margin_only:
        with path.with_name("predictions_sample_level.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["subject_id", "dep_score", "non_score"])
            for subject, _, dep, non in rows:
                writer.writerow([subject, dep - 0.5, non - 0.5])
                writer.writerow([subject, dep + 0.5, non + 0.5])


def build_tree(tmp_path: Path, *, drop: str | None = None) -> dict[str, Path]:
    roots = {"native": tmp_path / "native", "androids_mergedfolds": tmp_path / "amf", "merged": tmp_path / "merged"}
    rng = np.random.default_rng(0)
    for dataset in lf.CV_DATASETS + lf.TEST_DATASETS:
        folds = [0] if dataset in lf.TEST_DATASETS else range(5)
        for fold in folds:
            subjects = [(f"{dataset}{fold}s{i}", i % 2) for i in range(6)]
            for regime in lf.REGIMES:
                for modality in lf.MODALITIES:
                    rows = []
                    for s, y in subjects:
                        margin = float((1 if y else -1) + rng.normal(0, 1.5)) or 0.1
                        non = -float(rng.uniform(0.1, 3.0))
                        rows.append((s, y, non + margin, non))
                    if drop == f"{dataset}{fold}{regime}{modality}":
                        rows = rows[1:]
                    path = lf.source_path(roots, 7, dataset, fold, regime, modality)
                    write_scores(path, rows, margin_only=(dataset == "turkish" and modality == "text_only"))
    return roots


def test_subject_margin_prefers_score_margin_and_falls_back_to_difference():
    assert lf.subject_margin({"score_margin": "1.5", "dep_score": "-9", "non_score": "0"}) == 1.5
    assert lf.subject_margin({"score_margin": "", "dep_score": "-1", "non_score": "-3"}) == 2.0


def test_read_scores_rejects_margin_that_disagrees_with_prediction(tmp_path: Path):
    path = tmp_path / "p.csv"
    path.write_text("subject_id,label,prediction,dep_score,non_score\nA,1,0,-1.0,-3.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="disagrees"):
        lf.read_scores(path)


def test_read_scores_rejects_label_scores_that_do_not_reproduce_margin(tmp_path: Path):
    path = tmp_path / "p.csv"
    path.write_text(
        "subject_id,label,prediction,score_margin,dep_score,non_score\nA,1,1,0.5,-1.0,-3.0\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="reproduce"):
        lf.read_scores(path)


def test_read_scores_takes_label_scores_from_sample_file_when_subject_file_has_only_a_margin(tmp_path: Path):
    path = tmp_path / "predictions_subject_level.csv"
    write_scores(path, [("tk::A", 1, -0.2, -0.9)], margin_only=True)
    label, dep, non = lf.read_scores(path)["A"]
    assert label == 1 and dep == pytest.approx(-0.2) and non == pytest.approx(-0.9)


def test_load_matched_joins_six_models_and_strips_dataset_prefix(tmp_path: Path):
    rows, files = lf.load_matched(build_tree(tmp_path), 7)
    assert len(rows) == 4 * 5 * 6 + 6
    assert len(files) == (4 * 5 + 1) * 6
    assert all(f"{m}:dep" in row and f"{m}:non" in row for row in rows for m in lf.MODELS)


def test_load_matched_fails_closed_on_subject_mismatch(tmp_path: Path):
    roots = build_tree(tmp_path, drop="cmdc2multi_enaudio_text")
    with pytest.raises(ValueError, match="different subject set"):
        lf.load_matched(roots, 7)


def test_source_path_uses_mergedfold_rerun_for_androids_and_final_stage_for_daic(tmp_path: Path):
    roots = {"native": tmp_path / "n", "androids_mergedfolds": tmp_path / "a", "merged": tmp_path / "m"}
    assert "q3amf_text_only_native_s7_f3_r1" in str(lf.source_path(roots, 7, "androids_interview", 3, "uni", "text_only"))
    assert "/final/fold_0/qwen/daic/" in str(lf.source_path(roots, 7, "daic", 0, "multi_en", "audio_text"))
    assert "qwen3_pooled_english" in str(lf.source_path(roots, 7, "cmdc", 1, "multi_en", "text_only"))


def row(label: int, dep: list[float], non: list[float] | None = None, fold: int = 0) -> dict:
    non = non if non is not None else [0.0] * 6
    out = {"dataset": "cmdc", "fold": fold, "subject_id": "x", "label": label}
    for model, d, n in zip(lf.MODELS, dep, non):
        out[f"{model}:dep"], out[f"{model}:non"] = d, n
    return out


def test_views_normalize_over_two_labels_and_rescale_by_token_counts():
    r = [row(1, [-0.2] * 6, [-0.4] * 6)]
    assert lf.features(r, "mean_logit")[0, 0] == pytest.approx(0.2)
    assert lf.features(r, "seq_logit")[0, 0] == pytest.approx(2 * -0.2 - 3 * -0.4)
    p = lf.features(r, "mean_prob")[0, 0]
    assert p == pytest.approx(np.exp(-0.2) / (np.exp(-0.2) + np.exp(-0.4)))
    p = lf.features(r, "seq_prob")[0, 0]
    assert p == pytest.approx(np.exp(-0.4) / (np.exp(-0.4) + np.exp(-1.2)))
    raw = lf.features(r, "mean_raw")
    assert raw.shape == (1, 12) and raw[0, 0] == pytest.approx(np.exp(-0.2))
    assert lf.features(r, "seq_raw")[0, 6] == pytest.approx(np.exp(-1.2))


def test_training_free_oracle_vote_and_mean():
    rows = [
        row(1, [2, -1, -1, -1, -1, -1]),   # one model right -> oracle right, vote wrong
        row(0, [1, 1, 1, -1, -1, -1]),     # 3-3 tie, mean decides
        row(1, [-1, -1, -1, -1, -1, -1]),  # all wrong -> oracle wrong
    ]
    out = lf.training_free(rows, "mean_logit")
    assert out["oracle_any6"] == [1, 0, 0]
    assert out["vote_all6"][0] == 0 and out["vote_all6"][2] == 0
    mean = 1 / (1 + np.exp(-np.asarray([1, 1, 1, -1, -1, -1.0])))
    assert out["vote_all6"][1] == int(mean.mean() >= 0.5)
    agree = lf.agreement(rows)
    assert agree == {"subjects": 3, "all_six_agree": 1, "all_six_correct": 0, "all_six_wrong": 1}


def test_stacking_and_calibration_never_train_on_the_test_fold(monkeypatch):
    rows = [row(i % 2, [float(i % 2 * 2 - 1)] * 6, fold=i % 5) for i in range(30)]
    seen = []

    def fake(kind, seed, x_train, y_train, x_test):
        seen.append((len(x_train), len(x_test)))
        assert len(x_train) + len(x_test) == len(rows)
        return np.full(len(x_test), 0.9)

    monkeypatch.setattr(lf, "fit_predict", fake)
    assert lf.stacked(rows, "logreg", 1, "mean_logit") == [1] * 30
    assert seen == [(24, 6)] * 5
    seen.clear()
    out = lf.calibrated(rows, "mean_logit", "fold", 1)
    assert out["mean_all6_platt"] == [1] * 30
    assert seen == [(24, 6)] * 30  # 5 folds x 6 models
    seen.clear()
    lf.stacked(rows, "xgb", 1, "seq_raw", scheme="loo")
    assert seen == [(29, 1)] * 30


def test_xgb_is_invariant_to_the_sigmoid_normalization(tmp_path: Path):
    rows, _ = lf.load_matched(build_tree(tmp_path), 7)
    group = [r for r in rows if r["dataset"] == "d3tec"]
    assert lf.stacked(group, "xgb", 1337, "mean_logit") == lf.stacked(group, "xgb", 1337, "mean_prob")


def test_run_is_deterministic(tmp_path: Path):
    rows, _ = lf.load_matched(build_tree(tmp_path), 7)
    first = lf.run(rows, 1337)
    second = lf.run(rows, 1337)
    assert first == second
    assert {"logreg_per_dataset", "xgb_pooled", "mean_all6_platt"} <= set(first["cmdc"]["seq_logit"])
    assert {"logreg_loo", "xgb_transfer"} <= set(first["daic"]["mean_raw"])
    assert "mean_all6" not in first["cmdc"]["mean_raw"]
