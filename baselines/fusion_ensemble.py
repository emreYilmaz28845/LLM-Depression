#!/usr/bin/env python3
"""Calibrated text + audio ensemble of standalone frozen-feature heads.

Follow-up to ``baselines/band_score_experiments.py``: single arms gave no
Holm-significant change, but they disagreed by modality (the score target
helped audio, dropping the band helped text). This script averages members
instead of choosing one.

Members, fitted per corpus on its own training fold (standalone, global z),
for both frozen sources (Qwen3.8-27B native text, Qwen3-Omni audio):

- ``logit_base`` for every corpus;
- ``logit_drop`` and ``score_reg`` (see ``band_score_experiments``) for the
  questionnaire corpora (DAIC, D3TEC, Turkish) only.

Each member's penalty is chosen on inner stratified folds (3) over a wider L2
grid (1e-4 ... 1e5; the text heads chose the old top value 1000 most often).
The inner out-of-fold predictions at the chosen penalty fit a class-balanced
Platt scaling (one slope and intercept), which maps the member's outer test
prediction to a calibrated logit. An ensemble is the mean calibrated logit of
its members, classified at 0. Text and audio subjects, folds and labels are
identical (checked at load).

Pre-specified primary comparison (one per corpus, Holm over the five):
``fuse_all`` (every member of both sources) vs ``text_base_raw`` (the text
``logit_base`` member at threshold 0, i.e. the usual standalone text head).
Everything else is reported as exploratory with its own Holm family.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.band_score_experiments import SCORED, _fit  # noqa: E402
from baselines.merge_diag_common import (  # noqa: E402
    DATASETS, _inner_splits, fit_logistic, load_source, macro_f1, metrics,
)
from baselines.merge_diagnostics import boot_diff, fold_mean, git_state, holm, load_scores  # noqa: E402

SEED = 20261113
GRID_WIDE = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 1e4, 1e5)
INNER_FOLDS = 3
SOURCES = ("text_native", "audio_native")
MOD = {"text_native": "text", "audio_native": "audio"}


def members_for(dataset: str) -> list[tuple[str, str]]:
    arms = ("logit_base", "logit_drop", "score_reg") if dataset in SCORED else ("logit_base",)
    return [(MOD[s], a) for s in SOURCES for a in arms]


def balanced_weights(y: np.ndarray) -> np.ndarray:
    w = np.where(y == 1, 0.5 / max((y == 1).sum(), 1), 0.5 / max((y == 0).sum(), 1))
    return w / w.sum() * len(w)


def platt(pred: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Class-balanced 1-D logistic calibration; returns (slope, intercept)."""
    sd = float(np.std(pred)) or 1.0
    w, b, _ = fit_logistic((pred / sd)[:, None], y, balanced_weights(y), 1e-6)
    return float(w[0] / sd), float(b)


def select_with_oof(x, g, y, s, arm, seed, grid=GRID_WIDE):
    """Penalty by max inner macro-F1 (ties -> larger penalty) and the inner OOF at that penalty."""
    splits = _inner_splits(g, y, INNER_FOLDS, seed)
    best = None
    for penalty in grid:
        oof = np.zeros(len(y))
        for a, b in splits:
            oof[b] = _fit(x[a], g[a], y[a], s[a], "global", arm, penalty)(x[b], g[b])
        cand = (macro_f1(y, oof), penalty)
        if best is None or cand > best[0]:
            best = (cand, oof)
    return best[0][1], best[1]


def run(srcs: dict) -> dict:
    t, a = srcs["text_native"], srcs["audio_native"]
    if not (np.array_equal(t["subjects"], a["subjects"]) and np.array_equal(t["fold"], a["fold"])
            and np.array_equal(t["y"], a["y"]) and np.array_equal(t["g"], a["g"])):
        raise ValueError("text and audio subjects, folds, labels or datasets differ")
    g, y, fold, subjects = t["g"], t["y"], t["fold"], t["subjects"]
    scores = load_scores()
    s = np.asarray([scores.get(str(sub), np.nan) if d in SCORED else np.nan for sub, d in zip(subjects, g)])
    xs = {"text": t["x"], "audio": a["x"]}
    cvm = fold >= 0
    keys = [f"{m}/{arm}" for m in ("text", "audio") for arm in ("logit_base", "logit_drop", "score_reg")]
    raw = {k: np.full(len(y), np.nan) for k in keys}
    cal = {k: np.full(len(y), np.nan) for k in keys}
    log = []
    for k in range(5):
        for d in DATASETS:
            i, j = cvm & (g == d) & (fold != k), fold == k
            j = j & (g == d)
            for mod, arm in members_for(d):
                x = xs[mod]
                pen, inner = select_with_oof(x[i], g[i], y[i], s[i], arm, seed=k)
                pred = _fit(x[i], g[i], y[i], s[i], "global", arm, pen)(x[j], g[j])
                slope, icpt = platt(inner, y[i])
                raw[f"{mod}/{arm}"][j] = pred
                cal[f"{mod}/{arm}"][j] = slope * pred + icpt
                log.append({"fold": k, "dataset": d, "member": f"{mod}/{arm}", "penalty": pen,
                            "platt_slope": slope, "platt_intercept": icpt,
                            "inner_macro_f1": macro_f1(y[i], inner)})
        print(f"  fold {k} done", flush=True)

    def mean_of(names, m):
        return np.mean(np.stack([cal[n][m] for n in names]), axis=0)

    variants = {}
    for d in DATASETS:
        m = cvm & (g == d)
        mem = [f"{mod}/{arm}" for mod, arm in members_for(d)]
        v = {"text_base_raw": raw["text/logit_base"][m], "audio_base_raw": raw["audio/logit_base"][m],
             "text_base_cal": cal["text/logit_base"][m], "audio_base_cal": cal["audio/logit_base"][m],
             "fuse_base": mean_of(["text/logit_base", "audio/logit_base"], m),
             "text_all": mean_of([n for n in mem if n.startswith("text")], m),
             "audio_all": mean_of([n for n in mem if n.startswith("audio")], m),
             "fuse_all": mean_of(mem, m)}
        variants[d] = v
    rng = np.random.default_rng(SEED)
    table, primary, explore = {}, {}, {}
    for d in DATASETS:
        m = cvm & (g == d)
        yd = y[m]
        table[d] = {}
        for name, p in variants[d].items():
            full = np.full(len(y), np.nan)
            full[m] = p
            table[d][name] = {**fold_mean(y, full, g, fold, d), "pooled": metrics(yd, p)}
        primary[f"{d}|fuse_all_vs_text_base_raw"] = boot_diff(yd, variants[d]["fuse_all"],
                                                              variants[d]["text_base_raw"], rng)
        for name in ("fuse_base", "text_all", "audio_all", "text_base_cal"):
            explore[f"{d}|{name}_vs_text_base_raw"] = boot_diff(yd, variants[d][name],
                                                                variants[d]["text_base_raw"], rng)
        for name in ("fuse_all", "fuse_base"):
            explore[f"{d}|{name}_vs_audio_base_raw"] = boot_diff(yd, variants[d][name],
                                                                 variants[d]["audio_base_raw"], rng)
    for fam in (primary, explore):
        adj = holm({kk: vv["p"] for kk, vv in fam.items()})
        for kk in fam:
            fam[kk]["p_holm"] = adj[kk]
    mean_row = {n: float(np.mean([table[d][n]["macro_f1"] for d in DATASETS])) for n in variants[DATASETS[0]]}
    return {"table": table, "mean_macro_f1": mean_row, "primary_tests": primary, "exploratory_tests": explore,
            "members": {d: [f"{a}/{b}" for a, b in members_for(d)] for d in DATASETS}, "fit_log": log,
            "grid": list(GRID_WIDE), "oof_raw": {k: v.tolist() for k, v in raw.items()},
            "oof_cal": {k: v.tolist() for k, v in cal.items()}, "subjects": subjects.tolist()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    path = args.output_dir / "fusion_ensemble.json"
    if path.exists():
        raise SystemExit(f"Refusing to overwrite {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    state = git_state()
    srcs = {s: load_source(s) for s in SOURCES}
    t0 = time.time()
    res = run(srcs)
    payload = {**state, "analysis": "fusion_ensemble", "seed": SEED,
               "provenance": {s: srcs[s]["provenance"] for s in SOURCES}, "seconds": round(time.time() - t0, 1),
               "cv_subjects_per_dataset": {d: int(np.sum((srcs["text_native"]["g"] == d)
                                                         & (srcs["text_native"]["fold"] >= 0))) for d in DATASETS},
               "result": res}
    path.write_text(json.dumps(payload, indent=1))
    print(f"written {path} in {payload['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
