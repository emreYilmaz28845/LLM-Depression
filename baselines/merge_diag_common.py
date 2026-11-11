#!/usr/bin/env python3
"""Shared pieces for the merge diagnostics (``baselines/merge_diagnostics.py``).

Data: one frozen vector per subject (mean over response units of the mean
window vector), loaded with ``dataset_probe.load_merged`` on the merged
final-fold row index, plus the subject's outer CV fold from the merged CV fold
directories (``cv/fold_<k>/features/outer_holdout_rows.jsonl``). The merged CV
folds equal the standalone folds for every corpus; DAIC CV covers the 142
train+dev subjects and the 47 official test subjects get fold -1.

Head: L2 logistic (``subject_mean_logreg_head.fit_logistic``) fitted in the
row span of the training matrix (exact for L2), sample weights with equal
dataset mass and balanced classes inside each dataset, threshold 0 on the
logit. The penalty is chosen on inner stratified folds by mean per-dataset
macro-F1, then lower BCE, then larger penalty.

Normalizations (``Prep``), all fit on training subjects only; ``unlabeled``
optionally adds feature-only subjects of a dataset for its statistics:

- ``global``: one mean/sd over all training subjects.
- ``per_dataset``: each dataset standardized with its own mean/sd.
- ``cb_center``: each dataset centred on the mean of its two class means
  (needs labels of both classes in that dataset; otherwise falls back to the
  plain mean) and scaled by its own sd. Unlike ``per_dataset`` the centre does
  not move with the dataset's prevalence.
- ``combat``: ComBat-style location/scale harmonization with the label as a
  protected covariate and empirical-Bayes shrinkage; the estimated dataset
  shift and scale are applied to every row the same way (labels are used only
  for estimation), then a global standardization.
- ``consistent``: ``per_dataset`` followed by keeping the K features whose
  class-mean difference (Cohen's d) has the same sign in every labeled
  training dataset, ranked by the smallest |d| across datasets.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.dataset_probe import load_merged  # noqa: E402
from baselines.run_merged_head_matrix import features_dir  # noqa: E402
from baselines.subject_mean_logreg_head import DEFAULT_L2_GRID, fit_logistic, score_metrics  # noqa: E402

DATASETS = ("androids_interview", "cmdc", "d3tec", "daic", "turkish")
SHORT = {"androids_interview": "Italian (Androids)", "cmdc": "Mandarin (CMDC)", "d3tec": "Spanish (D3TEC)",
         "daic": "English (DAIC-WOZ)", "turkish": "Turkish"}
GRID = tuple(DEFAULT_L2_GRID)
INNER_FOLDS = 3
PREPS = ("global", "per_dataset", "cb_center", "combat", "consistent")

MAIN = Path("/home/emre/Projects/AudioLLM/LLM-Depression/output_model")
FEATURES_ROOT = MAIN / "frozen_base_standalone_heads/merged_lora_features"
CACHES_ROOT = MAIN / "frozen_base_standalone_heads/caches"
DAIC_AUDIO = MAIN / "frozen_base_omni_heads/audio_only/daic/daic_base_omni_heads_seed1337_20261023/fold_0"
SOURCES = {"text_native": ("native", "text_only"), "audio_native": ("native", "audio_only")}


# ------------------------------------------------------------------ data
def load_source(name: str) -> dict:
    language, modality = SOURCES[name]
    src = load_merged(FEATURES_ROOT, CACHES_ROOT, DAIC_AUDIO, language, modality)
    fold = {}
    for k in range(5):
        fdir = features_dir(FEATURES_ROOT, language, modality, 1337, "cv", k)
        for line in (fdir / "outer_holdout_rows.jsonl").read_text().splitlines():
            if line.strip():
                s = str(json.loads(line)["subject_id"])
                if fold.setdefault(s, k) != k:
                    raise ValueError(f"{s} is held out in two CV folds.")
    subjects = src["subjects"]
    return {"name": name, "subjects": np.asarray(subjects), "x": np.asarray(src["x"], dtype=np.float64),
            "g": np.asarray(src["dataset"]), "y": np.asarray(src["label"], dtype=int),
            "fold": np.asarray([fold.get(s, -1) for s in subjects]), "provenance": src["provenance"]}


# ------------------------------------------------------------------ weights / metrics
def head_weights(g: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Equal mass per dataset, equal mass per class present inside each dataset."""
    out = np.empty(len(y))
    datasets = sorted(set(g.tolist()))
    for d in datasets:
        classes = sorted(set(y[g == d].tolist()))
        for c in classes:
            m = (g == d) & (y == c)
            out[m] = 1.0 / (len(datasets) * len(classes) * m.sum())
    return out / out.sum() * len(out)


def macro_f1(y: np.ndarray, logits: np.ndarray) -> float:
    return float(score_metrics(y, logits)["macro_f1"])


def metrics(y: np.ndarray, logits: np.ndarray) -> dict:
    m = score_metrics(y, logits)
    return {"macro_f1": m["macro_f1"], "positive_f1": m["positive_f1"], "uar": m["binary_strict_uar"],
            "auroc": m["auroc"], "n": int(len(y))}


# ------------------------------------------------------------------ normalizations
def _mean_sd(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean, sd = x.mean(0), x.std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return mean, sd


class Prep:
    def __init__(self, kind: str, k_consistent: int = 256):
        if kind not in PREPS:
            raise ValueError(kind)
        self.kind, self.k = kind, k_consistent

    def fit(self, x, g, y, unlabeled: dict | None = None) -> "Prep":
        unlabeled = unlabeled or {}
        if self.kind == "global":
            self.mean, self.sd = _mean_sd(x)
            return self
        if self.kind == "combat":
            return self._fit_combat(x, g, y)
        self.stats = {}
        for d in sorted(set(g.tolist()) | set(unlabeled)):
            parts = [x[g == d]] + ([unlabeled[d]] if d in unlabeled else [])
            allx = np.vstack([p for p in parts if len(p)])
            mean, sd = _mean_sd(allx)
            if self.kind == "cb_center":
                xd, yd = x[g == d], y[g == d]
                if (yd == 0).any() and (yd == 1).any():
                    mean = 0.5 * (xd[yd == 0].mean(0) + xd[yd == 1].mean(0))
            self.stats[d] = (mean, sd)
        if self.kind == "consistent":
            z = self._per_dataset(x, g)
            ds = []
            for d in sorted(set(g.tolist())):
                zd, yd = z[g == d], y[g == d]
                if (yd == 0).sum() >= 2 and (yd == 1).sum() >= 2:
                    v = 0.5 * (zd[yd == 0].var(0) + zd[yd == 1].var(0))
                    ds.append((zd[yd == 1].mean(0) - zd[yd == 0].mean(0)) / np.sqrt(np.maximum(v, 1e-8)))
            ds = np.stack(ds)
            same = (np.all(ds > 0, axis=0) | np.all(ds < 0, axis=0))
            score = np.where(same, np.min(np.abs(ds), axis=0), -1.0)
            order = np.argsort(-score, kind="stable")
            self.columns = np.sort(order[:min(self.k, int(same.sum())) or 1])
            self.n_consistent = int(same.sum())
        return self

    def _per_dataset(self, x, g):
        z = np.empty_like(x)
        for d in set(g.tolist()):
            if d not in self.stats:
                raise ValueError(f"No statistics for dataset {d}.")
            mean, sd = self.stats[d]
            z[g == d] = (x[g == d] - mean) / sd
        return z

    def _fit_combat(self, x, g, y):
        self.gmean, self.gsd = _mean_sd(x)
        s0 = (x - self.gmean) / self.gsd
        batches = sorted(set(g.tolist()))
        design = np.column_stack([(g == b).astype(float) for b in batches] + [y.astype(float)])
        coef, *_ = np.linalg.lstsq(design, s0, rcond=None)
        n_b = np.asarray([(g == b).sum() for b in batches], dtype=float)
        grand = (n_b / n_b.sum()) @ coef[:-1]
        resid = s0 - design @ coef
        var_pooled = np.maximum(resid.var(0), 1e-8)
        stand = grand + np.outer(y, coef[-1])
        s = (s0 - stand) / np.sqrt(var_pooled)
        self.cb = {}
        for b in batches:
            sb = s[g == b]
            gamma_hat, delta_hat = sb.mean(0), np.maximum(sb.var(0, ddof=1) if len(sb) > 1 else np.ones(sb.shape[1]),
                                                          1e-6)
            gamma_bar, tau2 = gamma_hat.mean(), max(gamma_hat.var(ddof=1), 1e-8)
            m, v = delta_hat.mean(), max(delta_hat.var(ddof=1), 1e-8)
            a_prior, b_prior = (2 * v + m ** 2) / v, (m * v + m ** 3) / v
            n = len(sb)
            gamma, delta = gamma_hat.copy(), delta_hat.copy()
            for _ in range(100):
                gamma_new = (n * tau2 * gamma_hat + delta * gamma_bar) / (n * tau2 + delta)
                ss = ((sb - gamma_new) ** 2).sum(0)
                delta_new = (b_prior + 0.5 * ss) / (n / 2.0 + a_prior - 1.0)
                done = max(np.max(np.abs(gamma_new - gamma)), np.max(np.abs(delta_new - delta))) < 1e-6
                gamma, delta = gamma_new, delta_new
                if done:
                    break
            self.cb[b] = (grand + np.sqrt(var_pooled) * gamma, np.sqrt(delta))
        self.grand = grand
        out = self._apply_combat(x, g)
        self.post_mean, self.post_sd = _mean_sd(out)
        return self

    def _apply_combat(self, x, g):
        s0 = (x - self.gmean) / self.gsd
        out = np.empty_like(s0)
        for b in set(g.tolist()):
            shift, scale = self.cb[b]
            out[g == b] = (s0[g == b] - shift) / scale + self.grand
        return out

    def transform(self, x, g):
        if self.kind == "global":
            return (x - self.mean) / self.sd
        if self.kind == "combat":
            return (self._apply_combat(x, g) - self.post_mean) / self.post_sd
        z = self._per_dataset(x, g)
        return z[:, self.columns] if self.kind == "consistent" else z


# ------------------------------------------------------------------ head
class Head:
    """Weighted L2 logistic in the row span of the (prepped) training matrix."""

    def __init__(self, z, g, y, penalty, weights=None):
        _, sing, vt = np.linalg.svd(z, full_matrices=False)
        self.basis = vt[sing > np.finfo(float).eps * max(z.shape) * sing[0]].T
        w = head_weights(g, y) if weights is None else weights
        self.w, self.b, self.diag = fit_logistic(z @ self.basis, y, w, penalty)

    def logits(self, z):
        return (z @ self.basis) @ self.w + self.b

    def direction(self):
        return self.basis @ self.w


def _inner_splits(g, y, folds, seed):
    key = np.asarray([f"{a}|{b}" for a, b in zip(g, y)])
    _, counts = np.unique(key, return_counts=True)
    n = min(folds, int(counts.min()))
    if n < 2:
        return None
    return list(StratifiedKFold(n, shuffle=True, random_state=seed).split(np.zeros(len(key)), key))


def select_penalty(x, g, y, prep_kind, seed=0, unlabeled=None, grid=GRID, k_consistent=256,
                   return_oof=False):
    """Max mean per-dataset inner macro-F1, then lower BCE, then larger penalty."""
    splits = _inner_splits(g, y, INNER_FOLDS, seed)
    if splits is None:
        return (1.0, None) if return_oof else 1.0
    cands, oofs = [], {}
    prepped = []
    for a, b in splits:
        prep = Prep(prep_kind, k_consistent).fit(x[a], g[a], y[a], unlabeled)
        prepped.append((a, b, prep.transform(x[a], g[a]), prep.transform(x[b], g[b])))
    for penalty in grid:
        f1s, bces, oof = [], [], np.zeros(len(y))
        for a, b, za, zb in prepped:
            logits = Head(za, g[a], y[a], penalty).logits(zb)
            oof[b] = logits
        for d in sorted(set(g.tolist())):
            m = g == d
            if len(set(y[m].tolist())) == 2:
                f1s.append(macro_f1(y[m], oof[m]))
        bces.append(float(np.mean(np.logaddexp(0.0, oof) - y * oof)))
        cands.append((float(np.mean(f1s)), -float(np.mean(bces)), penalty))
        oofs[penalty] = oof
    best = max(cands)[2]
    return (best, oofs[best]) if return_oof else best


def fit_predict(xtr, gtr, ytr, xte, gte, prep_kind, penalty=None, seed=0, unlabeled=None, k_consistent=256):
    if penalty is None:
        penalty = select_penalty(xtr, gtr, ytr, prep_kind, seed, unlabeled, k_consistent=k_consistent)
    prep = Prep(prep_kind, k_consistent).fit(xtr, gtr, ytr, unlabeled)
    head = Head(prep.transform(xtr, gtr), gtr, ytr, penalty)
    return head.logits(prep.transform(xte, gte)), penalty, head, prep


def best_threshold(y, logits):
    """Logit threshold maximizing macro-F1 (midpoints of sorted scores); ties -> closest to 0."""
    s = np.unique(logits)
    cands = np.r_[s[0] - 1.0, (s[1:] + s[:-1]) / 2.0, s[-1] + 1.0, 0.0]
    best = max(((macro_f1(y, logits - t), -abs(t), t) for t in cands))
    return float(best[2])
