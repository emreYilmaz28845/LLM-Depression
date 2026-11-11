#!/usr/bin/env python3
"""Can the five corpora be merged? Diagnostics and local normalization arms.

Frozen subject-mean vectors (``merge_diag_common``), merged CV folds (= the
standalone folds; DAIC CV is the 142 train+dev subjects). Analyses, each
written to ``<output>/<source>/<analysis>.json``:

- ``cv``: per-corpus out-of-fold logits for the standalone head and for merged
  heads with the ``Prep`` normalizations, plus a merged per-dataset-standardized
  head with a per-dataset logit threshold tuned on its inner out-of-fold logits
  (``per_dataset_thr``) and a sweep of K for ``consistent``. Paired subject
  bootstrap of the pooled macro-F1 difference against standalone, Holm.
- ``mde``: from ``cv``: bootstrap SE of a corpus's macro-F1 and of the paired
  merged-minus-standalone difference; minimal detectable effect at 80 % power
  for alpha 0.05 and for alpha 0.05/552 (the replication's comparison count).
- ``curve``: standalone learning curve, 25/50/75/100 % of each training fold
  (stratified, 10 draws below 100 %), slope of macro-F1 per doubling.
- ``direction``: per-corpus depression directions on the full corpus
  (corpus-standardized): transfer AUROC matrix (head of A scores corpus B), cosine
  matrix and split-half cosine ceiling (20 stratified halves).
- ``target_n``: leave-one-corpus-out with k labeled target subjects
  (k = 0/5/10/20/40, 20 draws). Each draw splits the target into a stratified
  test half and a pool; labeled subjects are nested prefixes of a stratified
  ordering of the pool. Arms: target only, merged global, merged per-dataset
  (target statistics from the unlabeled pool, never the test half), merged
  class-balanced centring (target centre from the k labels). Merged penalties
  are chosen once per target on the source corpora only.
- ``band``: label-threshold band. Uses the standalone out-of-fold logits of
  ``cv`` and the raw scores (Turkish BDI, cut 17; D3TEC PHQ-9 and DAIC PHQ-8,
  cut 10): metrics inside vs outside the band, Spearman of logit vs score, and a
  standalone retrain that drops band subjects from training.
- ``tfidf``: per-corpus word 1-2 gram + char 2-5 gram TF-IDF logistic baseline
  on the native transcripts, same folds; top n-grams per class (local only).
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.merge_diag_common import (  # noqa: E402
    DATASETS, GRID, Head, Prep, best_threshold, fit_predict, load_source, macro_f1, metrics, select_penalty,
)

SEED = 20261111
MERGED_ARMS = ("global", "per_dataset", "cb_center", "combat", "consistent")
K_SWEEP = (32, 128, 1024)
BOOT = 2000
N_COMPARISONS_REPLICATION = 552
DATA = Path("/media/emre/Backup/AudioLLM/Datasets")
TRANSLATIONS = Path("/media/emre/Backup/AudioLLM/translations/harmonized_en_complete_v1")


def git_state() -> dict:
    run = lambda *a: subprocess.run(a, capture_output=True, text=True, cwd=PROJECT_ROOT).stdout.strip()  # noqa: E731
    return {"git_head": run("git", "rev-parse", "HEAD"),
            "git_dirty": bool(run("git", "status", "--porcelain", "--untracked-files=no"))}


def strat_order(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Indices ordered so every prefix is close to the class ratio (interleaved shuffled classes)."""
    pos, neg = rng.permutation(np.flatnonzero(y == 1)), rng.permutation(np.flatnonzero(y == 0))
    ratio = len(pos) / len(y)
    out, ip, ineg = [], 0, 0
    for i in range(len(y)):
        want_pos = round((i + 1) * ratio) - ip
        if (want_pos > 0 and ip < len(pos)) or ineg >= len(neg):
            out.append(pos[ip]); ip += 1  # noqa: E702
        else:
            out.append(neg[ineg]); ineg += 1  # noqa: E702
    # make sure both classes appear in the first two
    if len(out) > 2 and y[out[0]] == y[out[1]]:
        j = next(i for i in range(2, len(out)) if y[out[i]] != y[out[0]])
        out[1], out[j] = out[j], out[1]
    return np.asarray(out)


def boot_diff(y, a, b, rng, n=BOOT):
    """Paired subject bootstrap of macro-F1(a) - macro-F1(b); returns diff, se, two-sided p, CI."""
    idx = np.arange(len(y))
    diffs = []
    for _ in range(n):
        s = rng.choice(idx, len(idx), replace=True)
        if len(set(y[s].tolist())) < 2:
            continue
        diffs.append(macro_f1(y[s], a[s]) - macro_f1(y[s], b[s]))
    diffs = np.asarray(diffs)
    obs = macro_f1(y, a) - macro_f1(y, b)
    p = float(min(1.0, 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))))
    return {"diff": obs, "se": float(diffs.std(ddof=1)), "p": p,
            "ci95": [float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))]}


def holm(ps: dict) -> dict:
    keys = sorted(ps, key=lambda k: ps[k])
    out, running = {}, 0.0
    for i, k in enumerate(keys):
        running = max(running, min(1.0, (len(keys) - i) * ps[k]))
        out[k] = running
    return out


def fold_mean(y, logits, g, fold, d):
    m = (g == d) & (fold >= 0)
    vals = [metrics(y[m & (fold == k)], logits[m & (fold == k)]) for k in range(5)]
    return {key: float(np.mean([v[key] for v in vals])) for key in ("macro_f1", "positive_f1", "uar")}


# ------------------------------------------------------------------ cv
def analysis_cv(src: dict) -> dict:
    x, g, y, fold = src["x"], src["g"], src["y"], src["fold"]
    arms = ["standalone", *MERGED_ARMS, "per_dataset_thr", *[f"consistent_k{k}" for k in K_SWEEP]]
    oof = {a: np.full(len(y), np.nan) for a in arms}
    chosen = defaultdict(list)
    extra = defaultdict(list)
    for k in range(5):
        tr, te = (fold >= 0) & (fold != k), fold == k
        for d in DATASETS:
            a, b = tr & (g == d), te & (g == d)
            logits, pen, _, _ = fit_predict(x[a], g[a], y[a], x[b], g[b], "global", seed=k)
            oof["standalone"][b] = logits
            chosen[f"standalone/{d}"].append(pen)
        for arm in MERGED_ARMS:
            logits, pen, _, prep = fit_predict(x[tr], g[tr], y[tr], x[te], g[te], arm, seed=k)
            oof[arm][te] = logits
            chosen[arm].append(pen)
            if arm == "consistent":
                extra["consistent_n_same_sign"].append(prep.n_consistent)
        pen, inner_oof = select_penalty(x[tr], g[tr], y[tr], "per_dataset", seed=k, return_oof=True)
        logits, _, _, _ = fit_predict(x[tr], g[tr], y[tr], x[te], g[te], "per_dataset", penalty=pen)
        for d in DATASETS:
            t = best_threshold(y[tr & (g == d)], inner_oof[g[tr] == d])
            oof["per_dataset_thr"][te & (g == d)] = logits[g[te] == d] - t
            extra[f"threshold/{d}"].append(t)
        for kk in K_SWEEP:
            logits, pen, _, _ = fit_predict(x[tr], g[tr], y[tr], x[te], g[te], "consistent", seed=k,
                                            k_consistent=kk)
            oof[f"consistent_k{kk}"][te] = logits
        print(f"  cv fold {k} done", flush=True)
    cvm = fold >= 0
    rng = np.random.default_rng(SEED)
    table, tests = {}, {}
    for d in DATASETS:
        m = cvm & (g == d)
        table[d] = {a: {**fold_mean(y, oof[a], g, fold, d), "pooled": metrics(y[m], oof[a][m])} for a in arms}
        for a in arms[1:]:
            tests[f"{d}|{a}_vs_standalone"] = boot_diff(y[m], oof[a][m], oof["standalone"][m], rng)
        for a in ("per_dataset", "cb_center", "combat", "consistent", "per_dataset_thr"):
            tests[f"{d}|{a}_vs_global"] = boot_diff(y[m], oof[a][m], oof["global"][m], rng)
    adj = holm({k: v["p"] for k, v in tests.items()})
    for k in tests:
        tests[k]["p_holm"] = adj[k]
    mean_row = {a: float(np.mean([table[d][a]["macro_f1"] for d in DATASETS])) for a in arms}
    return {"arms": arms, "table": table, "mean_macro_f1": mean_row, "tests": tests,
            "chosen_penalty": dict(chosen), "extra": {k: v for k, v in extra.items()},
            "oof": {a: oof[a].tolist() for a in arms}, "subjects": src["subjects"].tolist()}


# ------------------------------------------------------------------ mde
def analysis_mde(src: dict, cv: dict) -> dict:
    g, y, fold = src["g"], src["y"], src["fold"]
    rng = np.random.default_rng(SEED + 1)
    z05, z80 = stats.norm.ppf(0.975), stats.norm.ppf(0.80)
    zh = stats.norm.ppf(1 - 0.025 / N_COMPARISONS_REPLICATION)
    out = {}
    stand = np.asarray(cv["oof"]["standalone"])
    glob = np.asarray(cv["oof"]["global"])
    for d in DATASETS:
        m = (g == d) & (fold >= 0)
        se_single = []
        idx = np.flatnonzero(m)
        for _ in range(BOOT):
            s = rng.choice(idx, len(idx), replace=True)
            se_single.append(macro_f1(y[s], stand[s]))
        bd = boot_diff(y[m], glob[m], stand[m], rng)
        out[d] = {"n": int(m.sum()), "se_macro_f1": float(np.std(se_single, ddof=1)), "se_paired_diff": bd["se"],
                  "mde_alpha05": float((z05 + z80) * bd["se"]),
                  "mde_alpha_holm552": float((zh + z80) * bd["se"]),
                  "observed_merged_global_minus_standalone": bd["diff"]}
    return out


# ------------------------------------------------------------------ curve
def analysis_curve(src: dict, fractions=(0.25, 0.5, 0.75, 1.0), draws=10) -> dict:
    x, g, y, fold = src["x"], src["g"], src["y"], src["fold"]
    out = {}
    for d in DATASETS:
        res = {f: [] for f in fractions}
        ntrain = {f: [] for f in fractions}
        for k in range(5):
            tr = np.flatnonzero((g == d) & (fold >= 0) & (fold != k))
            te = (g == d) & (fold == k)
            for f in fractions:
                for r in range(1 if f == 1.0 else draws):
                    rng = np.random.default_rng([SEED, k, int(f * 100), r])
                    order = strat_order(y[tr], rng)
                    sub = tr[np.sort(order[:max(4, int(round(f * len(tr))))])]
                    logits, _, _, _ = fit_predict(x[sub], g[sub], y[sub], x[te], g[te], "global", seed=r)
                    res[f].append((k, macro_f1(y[te], logits)))
                    ntrain[f].append(len(sub))
        means = {f: float(np.mean([np.mean([v for kk, v in res[f] if kk == k]) for k in range(5)])) for f in fractions}
        ns = {f: float(np.mean(ntrain[f])) for f in fractions}
        xs, ys = np.log2([ns[f] for f in fractions]), [means[f] for f in fractions]
        slope = float(np.polyfit(xs, ys, 1)[0])
        out[d] = {"macro_f1": {str(f): means[f] for f in fractions}, "n_train": {str(f): ns[f] for f in fractions},
                  "slope_per_doubling": slope,
                  "half_to_full_gain": means[1.0] - means[0.5]}
        print(f"  curve {d}: " + ", ".join(f"{f}:{means[f]:.3f}" for f in fractions) + f" slope {slope:+.3f}",
              flush=True)
    return out


# ------------------------------------------------------------------ direction
def _direction(x, y, g, penalty):
    prep = Prep("global").fit(x, g, y)
    return Head(prep.transform(x, g), g, y, penalty).direction(), prep


def analysis_direction(src: dict, halves=20) -> dict:
    x, g, y = src["x"], src["g"], src["y"]
    dirs, preps, pens = {}, {}, {}
    for d in DATASETS:
        m = g == d
        pens[d] = select_penalty(x[m], g[m], y[m], "global", seed=0)
        dirs[d], preps[d] = _direction(x[m], y[m], g[m], pens[d])
    auroc = {}
    for a in DATASETS:
        for b in DATASETS:
            if a == b:
                continue
            mb = g == b
            zb = Prep("global").fit(x[mb], g[mb], y[mb]).transform(x[mb], g[mb])
            auroc[f"{a}->{b}"] = metrics(y[mb], zb @ dirs[a])["auroc"]
    cos = {f"{a}|{b}": float(dirs[a] @ dirs[b] / np.linalg.norm(dirs[a]) / np.linalg.norm(dirs[b]))
           for i, a in enumerate(DATASETS) for b in DATASETS[i + 1:]}
    split = {}
    for d in DATASETS:
        idx = np.flatnonzero(g == d)
        vals = []
        for r in range(halves):
            order = strat_order(y[idx], np.random.default_rng([SEED, r]))
            h1, h2 = idx[np.sort(order[0::2])], idx[np.sort(order[1::2])]
            w1, _ = _direction(x[h1], y[h1], g[h1], pens[d])
            w2, _ = _direction(x[h2], y[h2], g[h2], pens[d])
            vals.append(float(w1 @ w2 / np.linalg.norm(w1) / np.linalg.norm(w2)))
        split[d] = {"mean": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1))}
    # half-sample cross-corpus cosine, comparable to the split-half ceiling
    cross_half = {}
    for i, a in enumerate(DATASETS):
        for b in DATASETS[i + 1:]:
            vals = []
            for r in range(halves):
                ia, ib = np.flatnonzero(g == a), np.flatnonzero(g == b)
                ha = ia[np.sort(strat_order(y[ia], np.random.default_rng([SEED, r]))[0::2])]
                hb = ib[np.sort(strat_order(y[ib], np.random.default_rng([SEED, r]))[1::2])]
                wa, _ = _direction(x[ha], y[ha], g[ha], pens[a])
                wb, _ = _direction(x[hb], y[hb], g[hb], pens[b])
                vals.append(float(wa @ wb / np.linalg.norm(wa) / np.linalg.norm(wb)))
            ceiling = np.sqrt(max(split[a]["mean"], 1e-9) * max(split[b]["mean"], 1e-9))
            cross_half[f"{a}|{b}"] = {"mean": float(np.mean(vals)), "ceiling": float(ceiling),
                                      "ratio": float(np.mean(vals) / ceiling)}
    return {"penalty": pens, "transfer_auroc": auroc, "cosine_full": cos, "split_half": split,
            "cosine_half": cross_half}


# ------------------------------------------------------------------ target_n
def analysis_target_n(src: dict, ks=(0, 5, 10, 20, 40), draws=20) -> dict:
    x, g, y = src["x"], src["g"], src["y"]
    arms = ("target_only", "merged_global", "merged_per_dataset", "merged_cb_center")
    out = {}
    for target in DATASETS:
        t0 = time.time()
        src_m = g != target
        tidx = np.flatnonzero(g == target)
        pens = {a: select_penalty(x[src_m], g[src_m], y[src_m], a.replace("merged_", ""), seed=0)
                for a in arms[1:]}
        res = {a: {k: [] for k in ks} for a in arms}
        for r in range(draws):
            rng = np.random.default_rng([SEED, DATASETS.index(target), r])
            order = strat_order(y[tidx], rng)
            test, pool = tidx[np.sort(order[0::2])], tidx[order[1::2]]
            pool = pool[strat_order(y[pool], rng)]
            for k in ks:
                if k > len(pool):
                    continue
                lab = pool[:k]
                unl = {target: x[pool[k:]]}  # labeled + unlabeled = the pool, never the test half
                if k >= 10:
                    lg, _, _, _ = fit_predict(x[lab], g[lab], y[lab], x[test], g[test], "global", seed=r)
                    res["target_only"][k].append(metrics(y[test], lg))
                tr = np.r_[np.flatnonzero(src_m), lab]
                for a in arms[1:]:
                    kind = a.replace("merged_", "")
                    lg, _, _, _ = fit_predict(x[tr], g[tr], y[tr], x[test], g[test], kind, penalty=pens[a],
                                              unlabeled=unl if kind != "global" else None)
                    res[a][k].append(metrics(y[test], lg))
        summ = {}
        for a in arms:
            summ[a] = {}
            for k in ks:
                v = res[a][k]
                if v:
                    summ[a][str(k)] = {key: float(np.mean([m[key] for m in v])) for key in
                                       ("macro_f1", "positive_f1", "uar", "auroc")}
                    summ[a][str(k)]["sd_macro_f1"] = float(np.std([m["macro_f1"] for m in v], ddof=1))
                    summ[a][str(k)]["draws"] = len(v)
        # paired over draws: merged_* minus target_only at the same k
        paired = {}
        for a in arms[1:]:
            for k in ks:
                if k >= 10 and res["target_only"][k] and res[a][k]:
                    dd = [m["macro_f1"] - t["macro_f1"] for m, t in zip(res[a][k], res["target_only"][k])]
                    paired[f"{a}-target_only@{k}"] = {"mean": float(np.mean(dd)),
                                                      "share_positive": float(np.mean(np.asarray(dd) > 0))}
        out[target] = {"penalty": pens, "summary": summ, "paired": paired, "test_n": int(len(test)),
                       "pool_n": int(len(pool))}
        print(f"  target_n {target} done in {time.time() - t0:.0f}s", flush=True)
    return out


# ------------------------------------------------------------------ band
def load_scores() -> dict:
    scores = {}
    for path, key in ((DATA / "Turkish_Positive_Only/metadata_turkish_t25_binary_merged.csv", "patient_id"),
                      (DATA / "Turkish_Negative_Only/metadata_turkish_negative_only_t17.csv", "patient_id")):
        for r in csv.DictReader(path.open(encoding="utf-8-sig")):
            s = f"turkish::{r[key]}"
            v = float(r["depresyon_skoru"])
            if scores.setdefault(s, v) != v:
                raise ValueError(f"{s}: conflicting BDI scores")
    for r in csv.DictReader((DATA / "D3TEC DATASET/D3TEC DATASET/Dataset.csv").open(encoding="utf-8-sig")):
        scores[f"d3tec::{str(r['Participant_ID']).strip().zfill(3)}"] = float(r["PHQ-9 Score"])
    for name in ("train_split_Depression_AVEC2017 (1).csv", "dev_split_Depression_AVEC2017.csv",
                 "full_test_split (1).csv"):
        for r in csv.DictReader((DATA / "DAIC-WOZ/minimal_zips" / name).open(encoding="utf-8-sig")):
            scores[f"daic::{str(r['Participant_ID']).strip()}"] = float(r["PHQ8_Score"])
    return scores


BANDS = {"turkish": (17.0, 14.0, 19.0), "d3tec": (10.0, 8.0, 11.0), "daic": (10.0, 8.0, 11.0)}


def analysis_band(src: dict, cv: dict) -> dict:
    x, g, y, fold, subjects = src["x"], src["g"], src["y"], src["fold"], src["subjects"]
    scores = load_scores()
    stand = np.asarray(cv["oof"]["standalone"])
    out = {}
    for d, (cut, lo, hi) in BANDS.items():
        m = (g == d) & (fold >= 0)
        sc = np.asarray([scores.get(s, np.nan) for s in subjects])
        if np.isnan(sc[m]).any():
            raise ValueError(f"{d}: {int(np.isnan(sc[m]).sum())} subjects without a score")
        agree = np.mean((sc[m] >= cut).astype(int) == y[m])
        band = m & (sc >= lo) & (sc <= hi)
        outside = m & ~band
        rho = stats.spearmanr(stand[m], sc[m])
        # retrain without band subjects in training
        oof_nb = np.full(len(y), np.nan)
        for k in range(5):
            tr = (g == d) & (fold >= 0) & (fold != k) & ~band
            te = (g == d) & (fold == k)
            oof_nb[te], _, _, _ = fit_predict(x[tr], g[tr], y[tr], x[te], g[te], "global", seed=k)
        hist = np.histogram(sc[m], bins=np.arange(0, max(28, sc[m].max() + 2), 2 if d != "turkish" else 4))
        out[d] = {"cut": cut, "band": [lo, hi], "label_agrees_with_cut": float(agree),
                  "n": int(m.sum()), "n_band": int(band.sum()), "positive_rate": float(y[m].mean()),
                  "score_hist": {"edges": hist[1].tolist(), "counts": hist[0].tolist()},
                  "all": metrics(y[m], stand[m]), "inside_band": metrics(y[band], stand[band]),
                  "outside_band": metrics(y[outside], stand[outside]),
                  "spearman_logit_score": {"rho": float(rho.statistic), "p": float(rho.pvalue)},
                  "spearman_outside_band": float(stats.spearmanr(stand[outside], sc[outside]).statistic),
                  "train_without_band": {"all": metrics(y[m], oof_nb[m]),
                                         "outside_band": metrics(y[outside], oof_nb[outside])}}
    return out


# ------------------------------------------------------------------ tfidf
def _subject_of(dataset: str, unit: str) -> str:
    if dataset == "androids_interview":
        a, b = unit.split("_")[:2]
        return f"{a}_{b[0]}"
    if dataset == "cmdc":
        return unit.split("_")[0]
    if dataset == "d3tec":
        return unit.split("_")[0]
    if dataset == "turkish":
        return unit.split("-")[0]
    raise ValueError(dataset)


def load_texts() -> tuple[dict, dict]:
    parts = defaultdict(list)
    sources = {"androids_interview": ["androids_interview"], "cmdc": ["cmdc"], "d3tec": ["d3tec"],
               "turkish": ["turkish_pos_only_t17", "turkish_negative_only_t17"]}
    for dataset, dirs in sources.items():
        for name in dirs:
            for line in (TRANSLATIONS / name / "units.jsonl").read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                s = f"{dataset}::{_subject_of(dataset, str(r['unit_id']))}"
                parts[s].append((name, str(r["unit_id"]), int(r.get("part_index") or 0), str(r["source_text"])))
    for name in ("preprocessed_whisper_transcripts.jsonl", "preprocessed_test_whisper_transcripts.jsonl"):
        for line in (DATA / "DAIC-WOZ" / name).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                parts[f"daic::{r['participant_id']}"].append(("daic", str(r.get("segment_name", "")), 0,
                                                              str(r["transcript"])))
    texts = {s: " ".join(t for *_, t in sorted(v)) for s, v in parts.items()}
    prov = {"translations_units": str(TRANSLATIONS), "daic": str(DATA / "DAIC-WOZ"),
            "note": "native source_text of the translation units (Androids, CMDC, D3TEC, Turkish t17 pos+neg) and "
                    "DAIC preprocessed Whisper segment transcripts; close to, not byte-identical with, the prompts"}
    return texts, prov


def analysis_tfidf(src: dict, cv: dict, top=15) -> dict:
    from scipy.sparse import hstack
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    texts, prov = load_texts()
    g, y, fold, subjects = src["g"], src["y"], src["fold"], src["subjects"]
    missing = [s for s, d, f in zip(subjects, g, fold) if f >= 0 and s not in texts]
    if missing:
        raise ValueError(f"{len(missing)} subjects without text, e.g. {missing[:5]}")
    cgrid = (0.01, 0.1, 1.0, 10.0, 100.0)

    def vec():
        return (TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                                token_pattern=r"(?u)\b\w+\b"),
                TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True,
                                max_features=50000))

    def fit(docs, labels, c):
        vw, vc = vec()
        xm = hstack([vw.fit_transform(docs), vc.fit_transform(docs)]).tocsr()
        clf = LogisticRegression(C=c, class_weight="balanced", max_iter=5000).fit(xm, labels)
        return lambda dd: clf.decision_function(hstack([vw.transform(dd), vc.transform(dd)]).tocsr()), vw, clf

    oof = np.full(len(y), np.nan)
    out = {}
    for d in DATASETS:
        for k in range(5):
            tr = np.flatnonzero((g == d) & (fold >= 0) & (fold != k))
            te = np.flatnonzero((g == d) & (fold == k))
            docs = [texts[s] for s in subjects[tr]]
            best = (-1, 0, None)
            for c in cgrid:
                f1s = []
                for a, b in StratifiedKFold(3, shuffle=True, random_state=k).split(docs, y[tr]):
                    pred, *_ = fit([docs[i] for i in a], y[tr][a], c)
                    f1s.append(macro_f1(y[tr][b], pred([docs[i] for i in b])))
                best = max(best, (float(np.mean(f1s)), -c, c))
            pred, *_ = fit(docs, y[tr], best[2])
            oof[te] = pred([texts[s] for s in subjects[te]])
        m = (g == d) & (fold >= 0)
        stand = np.asarray(cv["oof"]["standalone"])
        rng = np.random.default_rng(SEED + 7)
        out[d] = {"tfidf": fold_mean(y, oof, g, fold, d), "tfidf_pooled": metrics(y[m], oof[m]),
                  "qwen_standalone": fold_mean(y, stand, g, fold, d),
                  "qwen_minus_tfidf": boot_diff(y[m], stand[m], oof[m], rng)}
        idx = np.flatnonzero(m)
        _, vw, clf = fit([texts[s] for s in subjects[idx]], y[idx], 1.0)
        nw = len(vw.vocabulary_)
        coef = clf.coef_[0][:nw]
        vocab = np.asarray(sorted(vw.vocabulary_, key=vw.vocabulary_.get))
        out[d]["top_word_ngrams"] = {"depressed": vocab[np.argsort(-coef)[:top]].tolist(),
                                     "non_depressed": vocab[np.argsort(coef)[:top]].tolist()}
        print(f"  tfidf {d}: {out[d]['tfidf']['macro_f1']:.3f} vs qwen {out[d]['qwen_standalone']['macro_f1']:.3f}",
              flush=True)
    return {"per_dataset": out, "provenance": prov, "oof": oof.tolist()}


# ------------------------------------------------------------------ main
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--source", action="append", required=True, choices=("text_native", "audio_native"))
    p.add_argument("--analysis", action="append", required=True,
                   choices=("cv", "mde", "curve", "direction", "target_n", "band", "tfidf"))
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    state = git_state()
    for name in args.source:
        src = load_source(name)
        odir = args.output_dir / name
        odir.mkdir(parents=True, exist_ok=True)
        cv_path = odir / "cv.json"
        for an in args.analysis:
            path = odir / f"{an}.json"
            if path.exists():
                raise SystemExit(f"Refusing to overwrite {path}.")
            t0 = time.time()
            print(f"[{name}] {an} ...", flush=True)
            if an == "cv":
                res = analysis_cv(src)
            else:
                cv = json.loads(cv_path.read_text()) if an in ("mde", "band", "tfidf") else None
                res = {"mde": lambda: analysis_mde(src, cv), "curve": lambda: analysis_curve(src),
                       "direction": lambda: analysis_direction(src), "target_n": lambda: analysis_target_n(src),
                       "band": lambda: analysis_band(src, cv), "tfidf": lambda: analysis_tfidf(src, cv)}[an]()
            payload = {**state, "source": name, "analysis": an, "seed": SEED, "l2_grid": list(GRID),
                       "provenance": src["provenance"], "seconds": round(time.time() - t0, 1),
                       "subjects_per_dataset": {d: int(np.sum(src["g"] == d)) for d in DATASETS},
                       "cv_subjects_per_dataset": {d: int(np.sum((src["g"] == d) & (src["fold"] >= 0)))
                                                   for d in DATASETS},
                       "result": res}
            path.write_text(json.dumps(payload, indent=1))
            print(f"[{name}] {an} written in {payload['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
