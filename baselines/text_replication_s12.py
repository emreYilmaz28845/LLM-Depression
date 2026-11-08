#!/usr/bin/env python3
"""Text replication of Kaan's protocol, Section 12 source-control arms on the combined setup.

Every arm uses core A3's setup: the five TRAIN portions of outer fold f, class
oversampling with Kaan's reference sampler (seeds 0/1/2), and evaluation on each
language's TEST and the pooled TEST. Logistic arms use the fixed Combined settings
(C=0.01, thresholds 0.45/0.45/0.60).

Logistic arms (``LogisticRegression(C, max_iter=10000, tol=1e-8, lbfgs)``):
- ``base``: global TRAIN z-norm (= core A3, refit here).
- ``onehot_lr``: global z-norm features + a raw 5-way language one-hot.
- ``perlang_z_lr``: each language z-normed with its own TRAIN speakers.
- ``global_leace_lr``: global z-norm, then LEACE of the language concept fit on the
  unique TRAIN speakers.
- ``perlang_leace_lr``: per-language z-norm, then LEACE as above.

XGBoost arms (fixed before evaluation, no tuning: max_depth 3, 300 trees,
learning_rate 0.05, colsample_bytree 0.5, subsample 1.0, hist, random_state = seed,
threshold 0.5 on the probability, because the logistic thresholds were chosen for
the audio logistic head):
- ``xgb_all``: all D features.
- ``xgb_onehot``: all D features + the language one-hot.
- ``xgb_top50`` / ``xgb_top100``: the K features with the largest ANOVA F between
  classes on the unique TRAIN speakers of the fold (ranking nested inside each fold).

Not run here: the trainable 32D encoder, GRL, MMD and cross-source supervised
contrastive arms. Their losses, warm-up and coefficients are in Kaan's source, which
we do not have yet.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from sklearn.feature_selection import f_classif

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.concept_erasure import fit_leace  # noqa: E402
from baselines.dataset_probe import load_merged  # noqa: E402
from baselines.text_replication import (FOLDS, LANGUAGES, SEEDS, SETTINGS, Cohort, fit_head, fit_scaler,  # noqa: E402
                                        metrics, oversample_draw, sha256_file, summarize)

LR_ARMS = ("base", "onehot_lr", "perlang_z_lr", "global_leace_lr", "perlang_leace_lr")
XGB_ARMS = ("xgb_all", "xgb_onehot", "xgb_top50", "xgb_top100")
XGB_PARAMS = {"max_depth": 3, "n_estimators": 300, "learning_rate": 0.05, "colsample_bytree": 0.5,
              "subsample": 1.0, "tree_method": "hist", "eval_metric": "logloss"}
XGB_THRESHOLD = 0.5
_COHORT: Cohort | None = None


def _onehot(languages: np.ndarray) -> np.ndarray:
    return (languages[:, None] == np.asarray(LANGUAGES)[None, :]).astype(np.float64)


def _perlang_z(x: np.ndarray, languages: np.ndarray, train: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    for lang in LANGUAGES:
        rows = np.flatnonzero(languages == lang)
        mean, std = fit_scaler(x[np.intersect1d(rows, train)])
        out[rows] = (x[rows] - mean) / std
    return out


def transform(arm: str, cohort: Cohort, train: np.ndarray) -> np.ndarray:
    """Features for every speaker; all statistics come from the unique TRAIN speakers only."""
    x, lang = cohort.x, cohort.language
    if arm in ("base", "onehot_lr", "global_leace_lr"):
        mean, std = fit_scaler(x[train])
        z = (x - mean) / std
    elif arm in ("perlang_z_lr", "perlang_leace_lr"):
        z = _perlang_z(x, lang, train)
    else:
        z = x
    if arm.endswith("leace_lr"):
        eraser = fit_leace(z[train], lang[train])
        z = eraser(z)
    if arm in ("onehot_lr", "xgb_onehot"):
        z = np.hstack([z, _onehot(lang)])
    if arm in ("xgb_top50", "xgb_top100"):
        k = int(arm.removeprefix("xgb_top"))
        f, _ = f_classif(x[train], cohort.label[train])
        top = np.argsort(-np.nan_to_num(f, nan=-np.inf), kind="stable")[:k]
        z = x[:, np.sort(top)]
    return z


def _init(cohort: Cohort) -> None:
    from threadpoolctl import threadpool_limits

    global _COHORT
    _COHORT = cohort
    threadpool_limits(1)


def _fit(task: tuple[str, int, int]) -> list[dict]:
    arm, fold, seed = task
    cohort = _COHORT
    y = cohort.label
    train = cohort.idx(None, fold, "train")
    draw = train[oversample_draw(y[train], seed)]
    z = transform(arm, cohort, train)
    if arm in LR_ARMS:
        c, thresholds = SETTINGS["Combined"]
        threshold = thresholds[seed]
        identity = (np.zeros(z.shape[1]), np.ones(z.shape[1]))  # features are already normalized
        model, conv = fit_head(z, y, draw, c, identity)
        prob_all = model.predict_proba(z)[:, 1]
        logit_all = model.decision_function(z)
    else:
        from xgboost import XGBClassifier

        c, threshold = None, XGB_THRESHOLD
        model = XGBClassifier(**XGB_PARAMS, random_state=seed, n_jobs=1)
        model.fit(z[draw], y[draw])
        prob_all = model.predict_proba(z)[:, 1]
        logit_all = model.predict(z, output_margin=True)
        conv = {"n_iter": XGB_PARAMS["n_estimators"], "converged": True}
    pred_all = (prob_all >= threshold).astype(np.int64)
    evals = {lang: cohort.idx(lang, fold, "test") for lang in LANGUAGES}
    evals["Pooled"] = cohort.idx(None, fold, "test")
    out = []
    for name, ev in evals.items():
        out.append({"experiment": "s12", "source": arm, "arm": arm, "eval": name, "fold": fold, "seed": seed,
                    "C": c, "threshold": threshold, "feature_dim": int(z.shape[1]), **conv,
                    "train_unique": int(len(train)), "train_rows": int(len(draw)),
                    **metrics(y[ev], pred_all[ev]),
                    "predictions": [{"subject": cohort.subjects[i], "label": int(y[i]), "logit": float(logit_all[i]),
                                     "prob": float(prob_all[i]), "pred": int(pred_all[i])} for i in ev]})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--language", choices=("native", "english"), required=True)
    parser.add_argument("--merged-features-root", type=Path, required=True)
    parser.add_argument("--frozen-caches-root", type=Path, required=True)
    parser.add_argument("--fold-assignments", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    data = load_merged(args.merged_features_root, args.frozen_caches_root, Path("/nonexistent"),
                       args.language, "text_only")
    cohort = Cohort(data, args.fold_assignments)
    tasks = [(arm, fold, seed) for arm in (*LR_ARMS, *XGB_ARMS) for fold in FOLDS for seed in SEEDS]
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init, initargs=(cohort,)) as pool:
        results = [r for batch in pool.map(_fit, tasks) for r in batch]
    fits = Counter(r["arm"] for r in results if r["eval"] == "Pooled")
    assert set(fits.values()) == {15} and len(fits) == len(LR_ARMS) + len(XGB_ARMS), fits

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "fits.jsonl").open("w") as fh:
        for r in results:
            fh.write(json.dumps({k: v for k, v in r.items() if k != "predictions"}) + "\n")
    with (args.output_dir / "predictions.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "eval", "fold", "seed", "threshold", "subject", "label", "logit", "prob", "pred"])
        for r in results:
            for p in r["predictions"]:
                w.writerow([r["arm"], r["eval"], r["fold"], r["seed"], r["threshold"], p["subject"], p["label"],
                            p["logit"], p["prob"], p["pred"]])
    git_head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=PROJECT_ROOT).stdout.strip()
    result = {"git_head": git_head, "language": args.language,
              "fold_assignments_sha256": sha256_file(args.fold_assignments),
              "feature_provenance": data["provenance"], "xgb_params": XGB_PARAMS, "xgb_threshold": XGB_THRESHOLD,
              "combined_settings": SETTINGS["Combined"], "fits_per_arm": dict(fits),
              "all_converged": all(r["converged"] for r in results), "summary": summarize(results)}
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({k: result[k] for k in ("fits_per_arm", "all_converged")}))


if __name__ == "__main__":
    main()
