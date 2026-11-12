#!/usr/bin/env python3
"""Band-aware training and a continuous-score target for the questionnaire corpora.

Follow-up to ``baselines/merge_diagnostics.py``. That analysis found that
subjects whose questionnaire score lies near the label cut-off are classified
at chance. This script tests two ways to use that finding, on the same frozen
subject-mean vectors and merged CV folds (``merge_diag_common.load_source``):

1. Band-aware training (binary L2 logistic head):
   - ``logit_base``: every training subject, equal dataset mass, balanced classes.
   - ``logit_drop``: training subjects inside the band around the cut are removed
     (Turkish BDI 14-19, cut 17; D3TEC PHQ-9 and DAIC PHQ-8 8-11, cut 10).
   - ``logit_soft``: every training subject of a scored corpus is weighted by
     ``clip(|score - (cut - 0.5)| / ramp, 0.2, 1)`` (ramp 3 BDI points, 2 PHQ
     points), so weight grows with distance from the cut.
2. Continuous-score target (``score_reg``): weighted L2 ridge regression on a
   cut-centred score in corpus standard-deviation units,
   ``(score - (cut - 0.5)) / sd_train(score)``; the prediction is classified as
   depressed when it is above 0, i.e. above the corpus cut. In the merged
   setting the clinical-diagnosis corpora (CMDC, Androids), which have no
   score, get the target +1 (patient) / -1 (control). This is an approximate
   common scale, not a validated PHQ/BDI crosswalk.

Each arm runs standalone (one corpus, global z) on the three scored corpora
and merged (all five corpora, per-corpus z). Evaluation always uses every test
subject, including band subjects. The penalty is chosen on inner stratified
folds (3) by mean per-corpus inner macro-F1 on all inner validation subjects,
ties to the larger penalty; the same rule for every arm.

Output: ``<output>/<source>/band_score.json`` with per-corpus fold-mean and
pooled metrics, metrics inside/outside the band, Spearman of the prediction
with the score, and paired subject-bootstrap tests against the base arm of the
same setting with Holm correction over every test of the source.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.merge_diag_common import (  # noqa: E402
    DATASETS, GRID, Prep, _inner_splits, fit_logistic, head_weights, load_source, macro_f1, metrics,
)
from baselines.merge_diagnostics import BANDS, boot_diff, fold_mean, git_state, holm, load_scores  # noqa: E402

SEED = 20261112
SCORED = ("d3tec", "daic", "turkish")
RAMP = {"turkish": 3.0, "d3tec": 2.0, "daic": 2.0}
SOFT_FLOOR = 0.2
ARMS = ("logit_base", "logit_drop", "logit_soft", "score_reg")
INNER_FOLDS = 3


# ------------------------------------------------------------------ training weights / targets
def band_mask(g: np.ndarray, s: np.ndarray) -> np.ndarray:
    out = np.zeros(len(g), dtype=bool)
    for d in SCORED:
        _, lo, hi = BANDS[d]
        out |= (g == d) & (s >= lo) & (s <= hi)
    return out


def train_multiplier(g: np.ndarray, s: np.ndarray, mode: str) -> np.ndarray:
    """Per-subject training weight multiplier; 0 removes the subject."""
    mult = np.ones(len(g))
    if mode == "drop":
        mult[band_mask(g, s)] = 0.0
    elif mode == "soft":
        for d in SCORED:
            m = g == d
            mid = BANDS[d][0] - 0.5
            mult[m] = np.clip(np.abs(s[m] - mid) / RAMP[d], SOFT_FLOOR, 1.0)
    elif mode != "base":
        raise ValueError(mode)
    return mult


def score_target(g: np.ndarray, y: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Cut-centred score in corpus-sd units (scored corpora), +1/-1 for clinical corpora.

    The sd is computed on the rows passed in (training rows only at fit time).
    """
    t = np.where(y == 1, 1.0, -1.0)
    for d in SCORED:
        m = g == d
        if m.any():
            sd = float(np.std(s[m]))
            t[m] = (s[m] - (BANDS[d][0] - 0.5)) / (sd if sd > 1e-8 else 1.0)
    return t


# ------------------------------------------------------------------ linear model
class Linear:
    """Weighted L2 logistic (binary) or L2 ridge (continuous target) in the row span."""

    def __init__(self, z, g, y, penalty, mult=None, target=None):
        _, sing, vt = np.linalg.svd(z, full_matrices=False)
        self.basis = vt[sing > np.finfo(float).eps * max(z.shape) * sing[0]].T
        w = head_weights(g, y) * (1.0 if mult is None else mult)
        w = w / w.sum() * len(w)
        a = z @ self.basis
        if target is None:
            self.w, self.b, _ = fit_logistic(a, y, w, penalty)
            return
        sw = w.sum()
        mu_a, mu_t = w @ a / sw, float(w @ target / sw)
        ac = a - mu_a
        lhs = ac.T @ (w[:, None] * ac) / sw + penalty * np.eye(a.shape[1])
        self.w = np.linalg.solve(lhs, ac.T @ (w * (target - mu_t)) / sw)
        self.b = mu_t - float(mu_a @ self.w)

    def predict(self, z):
        return (z @ self.basis) @ self.w + self.b


def _fit(x, g, y, s, prep_kind, arm, penalty):
    mode = arm.split("_")[1] if arm.startswith("logit") else "base"
    mult = train_multiplier(g, s, mode)
    keep = mult > 0
    prep = Prep(prep_kind).fit(x, g, y)  # unsupervised statistics on every training subject
    z = prep.transform(x[keep], g[keep])
    target = score_target(g[keep], y[keep], s[keep]) if arm == "score_reg" else None
    model = Linear(z, g[keep], y[keep], penalty, mult[keep], target)
    return lambda xx, gg: model.predict(prep.transform(xx, gg))


def select_penalty(x, g, y, s, prep_kind, arm, seed):
    splits = _inner_splits(g, y, INNER_FOLDS, seed)
    if splits is None:
        return 1.0
    cands = []
    for penalty in GRID:
        oof = np.zeros(len(y))
        for a, b in splits:
            oof[b] = _fit(x[a], g[a], y[a], s[a], prep_kind, arm, penalty)(x[b], g[b])
        f1 = np.mean([macro_f1(y[g == d], oof[g == d]) for d in sorted(set(g.tolist()))
                      if len(set(y[g == d].tolist())) == 2])
        cands.append((float(f1), penalty))
    return max(cands)[1]


# ------------------------------------------------------------------ experiment
def run(src: dict) -> dict:
    x, g, y, fold, subjects = src["x"], src["g"], src["y"], src["fold"], src["subjects"]
    scores = load_scores()
    s = np.asarray([scores.get(str(sub), np.nan) if d in SCORED else np.nan for sub, d in zip(subjects, g)])
    cvm = fold >= 0
    for d in SCORED:
        miss = int(np.isnan(s[cvm & (g == d)]).sum())
        if miss:
            raise ValueError(f"{d}: {miss} CV subjects without a score")
    agree = {d: float(np.mean((s[cvm & (g == d)] >= BANDS[d][0]).astype(int) == y[cvm & (g == d)]))
             for d in SCORED}
    settings = {"standalone": SCORED, "merged": DATASETS}
    oof = {f"{st}/{a}": np.full(len(y), np.nan) for st in settings for a in ARMS}
    chosen = {k: [] for k in oof}
    for k in range(5):
        tr, te = cvm & (fold != k), fold == k
        for a in ARMS:
            for d in SCORED:
                i, j = tr & (g == d), te & (g == d)
                pen = select_penalty(x[i], g[i], y[i], s[i], "global", a, seed=k)
                oof[f"standalone/{a}"][j] = _fit(x[i], g[i], y[i], s[i], "global", a, pen)(x[j], g[j])
                chosen[f"standalone/{a}"].append({"fold": k, "dataset": d, "penalty": pen})
            pen = select_penalty(x[tr], g[tr], y[tr], s[tr], "per_dataset", a, seed=k)
            oof[f"merged/{a}"][te] = _fit(x[tr], g[tr], y[tr], s[tr], "per_dataset", a, pen)(x[te], g[te])
            chosen[f"merged/{a}"].append({"fold": k, "penalty": pen})
        print(f"  fold {k} done", flush=True)

    band = band_mask(g, s)
    rng = np.random.default_rng(SEED)
    table, tests = {}, {}
    for st, ds in settings.items():
        for d in ds:
            m = cvm & (g == d)
            row = {}
            for a in ARMS:
                p = oof[f"{st}/{a}"]
                cell = {**fold_mean(y, p, g, fold, d), "pooled": metrics(y[m], p[m])}
                if d in SCORED:
                    cell["inside_band"] = metrics(y[m & band], p[m & band])
                    cell["outside_band"] = metrics(y[m & ~band], p[m & ~band])
                    cell["spearman_score"] = float(stats.spearmanr(p[m], s[m]).statistic)
                row[a] = cell
            table[f"{st}/{d}"] = row
            for a in ARMS[1:]:
                tests[f"{st}/{d}|{a}_vs_logit_base"] = boot_diff(y[m], oof[f"{st}/{a}"][m],
                                                                 oof[f"{st}/logit_base"][m], rng)
    adj = holm({kk: v["p"] for kk, v in tests.items()})
    for kk in tests:
        tests[kk]["p_holm"] = adj[kk]
    n_band = {d: int((cvm & (g == d) & band).sum()) for d in SCORED}
    return {"arms": list(ARMS), "settings": {k: list(v) for k, v in settings.items()}, "table": table,
            "tests": tests, "chosen_penalty": chosen, "n_band": n_band, "label_agrees_with_cut": agree,
            "bands": {d: list(BANDS[d]) for d in SCORED}, "ramp": RAMP, "soft_floor": SOFT_FLOOR,
            "oof": {kk: v.tolist() for kk, v in oof.items()}, "subjects": subjects.tolist()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--source", action="append", required=True, choices=("text_native", "audio_native"))
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    state = git_state()
    for name in args.source:
        path = args.output_dir / name / "band_score.json"
        if path.exists():
            raise SystemExit(f"Refusing to overwrite {path}.")
        path.parent.mkdir(parents=True, exist_ok=True)
        src = load_source(name)
        t0 = time.time()
        print(f"[{name}] band_score ...", flush=True)
        res = run(src)
        payload = {**state, "source": name, "analysis": "band_score", "seed": SEED, "l2_grid": list(GRID),
                   "provenance": src["provenance"], "seconds": round(time.time() - t0, 1),
                   "cv_subjects_per_dataset": {d: int(np.sum((src["g"] == d) & (src["fold"] >= 0)))
                                               for d in DATASETS},
                   "result": res}
        path.write_text(json.dumps(payload, indent=1))
        print(f"[{name}] written in {payload['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
