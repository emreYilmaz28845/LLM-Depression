#!/usr/bin/env python3
"""Text replication of Kaan's multilingual logistic protocol (handoff 2026-11-08), core A.

Frozen Qwen3.8-27B text features (base model, no adapter), one equal-weight mean
vector per speaker (``dataset_probe.load_merged``), five corpora with DAIC-WOZ as
English. Folds come from ``fold_assignments.csv`` built from our merged protocol
(English fixed 142 train / 47 official test in every fold).

Head: global z-norm fit on the unique training speakers (population std, std < 1e-6
-> 1), then class oversampling by duplicating minority speakers (Kaan's reference
sampler), then ``LogisticRegression(C, max_iter=10000, tol=1e-8, lbfgs)``. C and the
per-seed probability thresholds are the fixed audio-derived settings from the
handoff, not tuned on text. A speaker is predicted Depressed when P >= threshold.

- A1 ``same``: X TRAIN -> X TEST, five folds x seeds 0/1/2 (English: one fixed split).
- A2 ``cross``: all X speakers -> all Y speakers, X's settings, scaler on X only.
- A3 ``combined``: the five TRAIN portions -> each TEST and the pooled TEST, Combined settings.

Metric: macro-F1 over labels [0, 1] with zero_division=0; positive-F1 and UAR are
also stored. Aggregation: seed mean inside each fold, then the equal mean of fold
means; SD is the sample SD (ddof=1) of the fold means.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, recall_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.dataset_probe import load_merged  # noqa: E402

# Language order fixes the seeded sampling (handoff section 7).
LANGUAGES = ("English", "Turkish", "Italian", "Spanish", "Mandarin")
DATASET_LANGUAGE = {"daic": "English", "turkish": "Turkish", "androids_interview": "Italian",
                    "d3tec": "Spanish", "cmdc": "Mandarin"}
SEEDS = (0, 1, 2)
FOLDS = (0, 1, 2, 3, 4)
SETTINGS = {  # C, thresholds for seeds 0, 1, 2 (handoff section 5, audio-derived)
    "English": (0.01, (0.50, 0.50, 0.45)),
    "Turkish": (1000.0, (0.50, 0.50, 0.35)),
    "Italian": (1000.0, (0.60, 0.60, 0.60)),
    "Spanish": (0.0001, (0.45, 0.45, 0.45)),
    "Mandarin": (1.0, (0.25, 0.30, 0.30)),
    "Combined": (0.01, (0.45, 0.45, 0.60)),
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def oversample_draw(labels: np.ndarray, seed) -> np.ndarray:
    """Kaan's reference sampler: indices into the unique training array, classes equalized.

    ``seed`` is an int or a ``SeedSequence``; ``default_rng(seed)`` is used either way.
    """
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(labels == 0), np.flatnonzero(labels == 1)]
    n = max(len(g) for g in groups)
    parts = [np.concatenate([g, rng.choice(g, n - len(g), replace=True)]) for g in groups]
    draw = np.concatenate(parts)
    rng.shuffle(draw)
    return draw


def fit_scaler(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    std = x.std(axis=0, ddof=0)
    std[std < 1e-6] = 1.0
    return mean, std


def fit_head(x_unique: np.ndarray, y_unique: np.ndarray, draw: np.ndarray, c: float,
             scaler: tuple[np.ndarray, np.ndarray]) -> tuple[LogisticRegression, dict]:
    mean, std = scaler
    xz = (x_unique[draw] - mean) / std
    model = LogisticRegression(C=c, max_iter=10000, tol=1e-8, solver="lbfgs")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(xz, y_unique[draw])
    n_iter = int(np.max(model.n_iter_))
    converged = n_iter < 10000 and not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    return model, {"n_iter": n_iter, "converged": converged}


def fit_rows(x: np.ndarray, y: np.ndarray, rows: np.ndarray, scaler_keys: np.ndarray, c: float):
    """Fit on global row indices ``rows`` (duplicates allowed); scaler on unique ``scaler_keys``."""
    scaler = fit_scaler(x[np.unique(scaler_keys)])
    model, conv = fit_head(x, y, rows, c, scaler)
    return model, scaler, conv


def predict(model: LogisticRegression, scaler, x: np.ndarray, threshold: float):
    mean, std = scaler
    xz = (x - mean) / std
    logit = model.decision_function(xz)
    prob = model.predict_proba(xz)[:, 1]
    return logit, prob, (prob >= threshold).astype(np.int64)


def metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    cm = confusion_matrix(y, pred, labels=[0, 1])
    return {"macro_f1": float(f1_score(y, pred, labels=[0, 1], average="macro", zero_division=0)),
            "positive_f1": float(f1_score(y, pred, pos_label=1, zero_division=0)),
            "uar": float(recall_score(y, pred, labels=[0, 1], average="macro", zero_division=0)),
            "confusion_matrix": cm.tolist(), "n": int(len(y)), "n_pos": int(y.sum()), "n_neg": int((1 - y).sum())}


class Cohort:
    """Speaker vectors aligned with the fold assignment."""

    def __init__(self, data: dict, assignments: Path):
        self.subjects = [str(s) for s in data["subjects"]]
        self.x = np.asarray(data["x"], dtype=np.float64)
        self.label = np.asarray(data["label"], dtype=np.int64)
        self.language = np.asarray([DATASET_LANGUAGE[d] for d in data["dataset"]])
        rows = list(csv.DictReader(assignments.open()))
        key = {s: i for i, s in enumerate(self.subjects)}
        self.role: dict[int, np.ndarray] = {}
        for fold in FOLDS:
            role = np.full(len(self.subjects), "", dtype=object)
            for r in rows:
                if int(r["fold"]) != fold:
                    continue
                k = f"{r['dataset']}::{r['subject_id']}"
                if k not in key:
                    raise ValueError(f"Speaker {k} in fold_assignments.csv has no feature vector.")
                i = key[k]
                if int(r["label"]) != self.label[i]:
                    raise ValueError(f"Label mismatch for {k}.")
                role[i] = r["role"]
            if (role == "").any():
                missing = [self.subjects[i] for i in np.flatnonzero(role == "")][:5]
                raise ValueError(f"Fold {fold}: speakers without an assignment, e.g. {missing}.")
            self.role[fold] = role
        if not np.isfinite(self.x).all() or (np.linalg.norm(self.x, axis=1) == 0).any():
            raise ValueError("Non-finite or zero speaker vectors.")

    def idx(self, language: str | None = None, fold: int | None = None, role: str | None = None) -> np.ndarray:
        mask = np.ones(len(self.subjects), dtype=bool)
        if language is not None:
            mask &= self.language == language
        if fold is not None and role is not None:
            mask &= self.role[fold] == role
        return np.flatnonzero(mask)  # sorted by speaker key: fixed ordering for seeded sampling


def _fit_eval(cohort: Cohort, train: np.ndarray, evals: dict[str, np.ndarray], setting: str, seed: int,
              record: dict) -> list[dict]:
    c, thresholds = SETTINGS[setting]
    threshold = thresholds[seed]
    x_u, y_u = cohort.x[train], cohort.label[train]
    draw = oversample_draw(y_u, seed)
    counts = Counter(y_u[draw].tolist())
    assert counts[0] == counts[1] and set(draw.tolist()) == set(range(len(train)))
    scaler = fit_scaler(x_u)
    model, conv = fit_head(x_u, y_u, draw, c, scaler)
    fit = {**record, "seed": seed, "setting": setting, "C": c, "threshold": threshold,
           "train_unique": int(len(train)), "train_rows": int(len(draw)),
           "train_unique_pos": int(y_u.sum()), "train_rows_per_class": int(counts[0]),
           "train_by_language": dict(Counter(cohort.language[train].tolist())),
           "scaler_fit_keys_sha256": hashlib.sha256("\n".join(cohort.subjects[i] for i in train).encode()).hexdigest(),
           **conv}
    out = []
    for name, ev in evals.items():
        assert not set(ev.tolist()) & set(train.tolist()) or record["experiment"] == "cross", name
        logit, prob, pred = predict(model, scaler, cohort.x[ev], threshold)
        out.append({**fit, "eval": name, **metrics(cohort.label[ev], pred),
                    "predictions": [{"subject": cohort.subjects[i], "label": int(cohort.label[i]),
                                     "logit": float(lg), "prob": float(p), "pred": int(pr)}
                                    for i, lg, p, pr in zip(ev, logit, prob, pred)]})
    return out


def run_same(cohort: Cohort) -> list[dict]:
    rows = []
    for lang in LANGUAGES:
        folds = (0,) if lang == "English" else FOLDS
        for fold in folds:
            train, test = cohort.idx(lang, fold, "train"), cohort.idx(lang, fold, "test")
            for seed in SEEDS:
                rows += _fit_eval(cohort, train, {lang: test}, lang, seed,
                                  {"experiment": "same", "source": lang, "fold": fold})
    return rows


def run_cross(cohort: Cohort) -> list[dict]:
    rows = []
    for src in LANGUAGES:
        train = cohort.idx(src)
        evals = {tgt: cohort.idx(tgt) for tgt in LANGUAGES if tgt != src}
        for seed in SEEDS:
            rows += _fit_eval(cohort, train, evals, src, seed, {"experiment": "cross", "source": src, "fold": None})
    return rows


def run_combined(cohort: Cohort) -> list[dict]:
    rows = []
    for fold in FOLDS:
        train = cohort.idx(None, fold, "train")
        evals = {lang: cohort.idx(lang, fold, "test") for lang in LANGUAGES}
        evals["Pooled"] = cohort.idx(None, fold, "test")
        for seed in SEEDS:
            rows += _fit_eval(cohort, train, evals, "Combined", seed,
                              {"experiment": "combined", "source": "Combined", "fold": fold})
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    """Seed mean per fold, then equal mean over folds; SD (ddof=1) of the fold means."""
    groups: dict[tuple, dict[int, list[dict]]] = {}
    for r in rows:
        groups.setdefault((r["experiment"], r["source"], r["eval"]), {}).setdefault(r["fold"], []).append(r)
    out = []
    for (exp, src, ev), by_fold in groups.items():
        entry = {"experiment": exp, "source": src, "eval": ev, "n_folds": len(by_fold),
                 "seeds_per_fold": sorted({len(v) for v in by_fold.values()})}
        for m in ("macro_f1", "positive_f1", "uar"):
            fold_means = [float(np.mean([r[m] for r in v])) for _, v in sorted(by_fold.items(), key=lambda t: str(t[0]))]
            entry[m] = float(np.mean(fold_means))
            entry[f"{m}_fold_sd"] = float(np.std(fold_means, ddof=1)) if len(fold_means) > 1 else None
            entry[f"{m}_fold_means"] = fold_means
        entry["all_converged"] = all(r["converged"] for v in by_fold.values() for r in v)
        out.append(entry)
    return out


def render(result: dict) -> str:
    s = {(e["experiment"], e["source"], e["eval"]): e for e in result["summary"]}
    lines = [f"# Text replication of Kaan's protocol, core A ({result['language']} text)", "",
             f"Commit `{result['git_head']}`. Features: {result['feature_note']}. "
             f"Cohort: {result['cohort']}. Fits: {result['fit_counts']}. "
             f"All converged: {result['all_converged']}.", "",
             "Macro-F1 / positive-F1 / UAR, subject level. Fold-mean of seed means (SD over the 5 fold means). "
             "English same-language is one fixed split (142 -> 47), mean of 3 seeds.", "",
             "## A1 same-language (held-out TEST) and A3 combined (per-language TEST)", "",
             "| Target | A1 X -> X | A3 Combined -> X |", "|---|---|---|"]

    def cell(e):
        if e is None:
            return "—"
        sd = "" if e["macro_f1_fold_sd"] is None else f" (±{e['macro_f1_fold_sd']:.3f})"
        return f"{e['macro_f1']:.3f}{sd} / {e['positive_f1']:.3f} / {e['uar']:.3f}"
    for lang in (*LANGUAGES, "Pooled"):
        lines.append(f"| {lang} | {cell(s.get(('same', lang, lang)))} | {cell(s.get(('combined', 'Combined', lang)))} |")
    lines += ["", "## A2 cross-language (train on all X, evaluate on all Y; 3-seed mean macro-F1)",
              "", "Not comparable with held-out scores: different cohort and aggregation.", "",
              "| Source \\ Target | " + " | ".join(LANGUAGES) + " |", "|---" * (len(LANGUAGES) + 1) + "|"]
    for src in LANGUAGES:
        lines.append(f"| {src} | " + " | ".join(
            "—" if src == tgt else f"{s[('cross', src, tgt)]['macro_f1']:.3f}" for tgt in LANGUAGES) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--language", choices=("native", "english"), required=True)
    parser.add_argument("--merged-features-root", type=Path, required=True)
    parser.add_argument("--frozen-caches-root", type=Path, required=True)
    parser.add_argument("--fold-assignments", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    data = load_merged(args.merged_features_root, args.frozen_caches_root, Path("/nonexistent"),
                       args.language, "text_only")
    cohort = Cohort(data, args.fold_assignments)
    rows = run_same(cohort) + run_cross(cohort) + run_combined(cohort)
    fits = {(r["experiment"], r["source"], r["fold"], r["seed"]) for r in rows}
    fit_counts = dict(Counter(k[0] for k in fits))
    assert fit_counts == {"same": 63, "cross": 15, "combined": 15}, fit_counts
    # Each non-English speaker is held out exactly once per seed in A1.
    for lang in LANGUAGES[1:]:
        for seed in SEEDS:
            seen = Counter(p["subject"] for r in rows if r["experiment"] == "same" and r["source"] == lang
                           and r["seed"] == seed for p in r["predictions"])
            assert set(seen.values()) == {1} and len(seen) == len(cohort.idx(lang)), (lang, seed)

    summary = summarize(rows)
    git_head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=PROJECT_ROOT).stdout.strip()
    cohort_counts = {lang: f"{len(cohort.idx(lang))} ({int(cohort.label[cohort.idx(lang)].sum())}+)"
                     for lang in LANGUAGES}
    result = {"git_head": git_head, "language": args.language, "settings": SETTINGS, "seeds": SEEDS,
              "fold_assignments": str(args.fold_assignments),
              "fold_assignments_sha256": sha256_file(args.fold_assignments),
              "feature_provenance": data["provenance"], "dimension": int(cohort.x.shape[1]),
              "feature_note": f"frozen Qwen3.8-27B base, final layer, last prompt token, D={cohort.x.shape[1]}, "
                              "equal-weight speaker mean",
              "cohort": cohort_counts, "fit_counts": fit_counts,
              "all_converged": all(r["converged"] for r in rows), "summary": summary}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=1))
    with (args.output_dir / "fits.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps({k: v for k, v in r.items() if k != "predictions"}) + "\n")
    with (args.output_dir / "predictions.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["experiment", "source", "eval", "fold", "seed", "threshold", "subject", "label", "logit", "prob", "pred"])
        for r in rows:
            for p in r["predictions"]:
                w.writerow([r["experiment"], r["source"], r["eval"], r["fold"], r["seed"], r["threshold"],
                            p["subject"], p["label"], p["logit"], p["prob"], p["pred"]])
    (args.output_dir / "REPORT.md").write_text(render(result))
    print(render(result))


if __name__ == "__main__":
    main()
