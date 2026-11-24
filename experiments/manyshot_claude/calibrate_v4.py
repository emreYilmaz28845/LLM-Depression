"""Linear calibration of zero-shot v4 PHQ-8 item scores (plan in RESULTS.md, written before the run).

Fits one OLS regression, PHQ-8 total ~ intercept + 8 item scores, on the official train+dev subjects scored by
zero-shot v4, then applies it unchanged to each test run. Label = predicted score >= 10. Reports the raw item
total next to the calibrated score. INVALID train subjects are dropped from the fit; INVALID test subjects count
as wrong for the label metrics and are left out of the score metrics (their number is reported).

Usage:
  python experiments/manyshot_claude/calibrate_v4.py --train-run TRAINDEV_DIR --test-run R1 [--test-run R2 ...] \
      --phq-csv TRAIN.csv --phq-csv DEV.csv --phq-csv TEST.csv [--out OUT.json]
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import f1_score, recall_score


def load_phq(paths: list[Path]) -> dict[str, int]:
    phq: dict[str, int] = {}
    for path in paths:
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                phq[str(int(row["Participant_ID"]))] = int(row["PHQ8_Score"])
    return phq


def load_run(run_dir: Path) -> dict[str, dict]:
    return {r["subject_id"]: r for r in map(json.loads, (run_dir / "responses.jsonl").read_text().splitlines())}


def fit_ols(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return coef


def evaluate(scores: dict[str, float | None], labels: dict[str, int], phq: dict[str, int]) -> dict:
    ids = sorted(labels, key=int)
    ok = [s for s in ids if scores[s] is not None]
    pred_score = np.array([scores[s] for s in ok], float)
    true_score = np.array([phq[s] for s in ok], float)
    y = [labels[s] for s in ids]
    pred = [int(scores[s] >= 10) if scores[s] is not None else 1 - labels[s] for s in ids]
    return {"n": len(ids), "n_scored": len(ok),
            "rmse": round(float(np.sqrt(np.mean((pred_score - true_score) ** 2))), 3),
            "mae": round(float(np.mean(np.abs(pred_score - true_score))), 3),
            "spearman": round(float(spearmanr(pred_score, true_score)[0]), 3),
            "bias": round(float(np.mean(pred_score - true_score)), 3),
            "macro_f1": round(f1_score(y, pred, average="macro"), 4),
            "positive_f1": round(f1_score(y, pred, pos_label=1, zero_division=0), 4),
            "uar": round(recall_score(y, pred, average="macro"), 4)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--train-run", required=True, type=Path)
    p.add_argument("--test-run", required=True, type=Path, action="append")
    p.add_argument("--phq-csv", required=True, type=Path, action="append")
    p.add_argument("--out", type=Path)
    args = p.parse_args()

    phq = load_phq(args.phq_csv)
    train = load_run(args.train_run)
    fit_ids = sorted((s for s, r in train.items() if r.get("phq8_items")), key=int)
    x = np.array([train[s]["phq8_items"] for s in fit_ids], float)
    y = np.array([phq[s] for s in fit_ids], float)
    coef = fit_ols(x, y)
    report = {"train_run": str(args.train_run), "n_train": len(train), "n_fit": len(fit_ids),
              "coef": {"intercept": round(float(coef[0]), 4), **{f"item{i}": round(float(c), 4)
                                                                  for i, c in enumerate(coef[1:], 1)}},
              "train_fit_raw_total": evaluate({s: float(sum(train[s]["phq8_items"])) for s in fit_ids},
                                              {s: train[s]["label"] for s in fit_ids}, phq),
              "test": {}}
    for run_dir in args.test_run:
        test = load_run(run_dir)
        labels = {s: r["label"] for s, r in test.items()}
        raw = {s: (float(sum(r["phq8_items"])) if r.get("phq8_items") else None) for s, r in test.items()}
        cal = {s: (float(coef[0] + np.dot(coef[1:], r["phq8_items"])) if r.get("phq8_items") else None)
               for s, r in test.items()}
        report["test"][run_dir.name] = {"raw_total": evaluate(raw, labels, phq),
                                        "calibrated": evaluate(cal, labels, phq)}
    for kind in ("raw_total", "calibrated"):
        vals = {k: [v[kind][k] for v in report["test"].values()] for k in
                ("rmse", "mae", "spearman", "bias", "macro_f1", "positive_f1", "uar")}
        report[f"test_mean_{kind}"] = {k: [round(float(np.mean(v)), 3), round(float(np.std(v, ddof=1)), 3)
                                           if len(v) > 1 else 0.0] for k, v in vals.items()}
    text = json.dumps(report, indent=1)
    if args.out:
        args.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
