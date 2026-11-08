#!/usr/bin/env python3
"""Text replication of Kaan's protocol, core E: speaker-pair cosine geometry and nearest neighbours.

The same speaker vectors as the classifiers, natural corpus and label frequencies, no
oversampling. Views: ``raw`` and the five combined-TRAIN global scalers of core A3
(``scaler_f0`` .. ``scaler_f4``); normalized statistics are computed per view and then
averaged over the five views.

- E1: mean cosine over ALL speaker pairs between languages (N_A * N_B pairs) and within a
  language (unordered pairs, self-pairs excluded), using the exact unit-vector mean
  shortcut, plus the pooled, within-language and cross-language pair means (every pair
  weighted equally). Distance = 1 - cosine.
- E2: pooled cosine kNN, K = 30/60/90, other speakers only: same-language counts,
  language mixing matrix (rows sum to K), random expectation (N_lang - 1) / (N - 1) and
  capacity limits.
- E3: within-language cosine kNN, K = 10: fraction of neighbours sharing the query's
  depression label, by query class, against the random expectation.

Also writes the inputs of the UMAP maps (raw matrix + per-map point/scaler indices); the
maps themselves are drawn by ``text_replication_umap.py`` in an env that has umap-learn.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.dataset_probe import load_merged  # noqa: E402
from baselines.text_replication import FOLDS, LANGUAGES, Cohort, fit_scaler, sha256_file  # noqa: E402

KS = (30, 60, 90)
K_WITHIN = 10


def unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def pair_cosines(u: np.ndarray, lang: np.ndarray) -> dict:
    sums = {a: u[lang == a].sum(0) for a in LANGUAGES}
    n = {a: int((lang == a).sum()) for a in LANGUAGES}
    matrix = {}
    for a in LANGUAGES:
        for b in LANGUAGES:
            if a == b:
                matrix[f"{a}|{b}"] = float((sums[a] @ sums[a] - n[a]) / (n[a] * (n[a] - 1)))
            else:
                matrix[f"{a}|{b}"] = float(sums[a] @ sums[b] / (n[a] * n[b]))
    total = u.sum(0)
    big_n = len(u)
    within_pairs = sum(n[a] * (n[a] - 1) // 2 for a in LANGUAGES)
    all_pairs = big_n * (big_n - 1) // 2
    within_sum = sum((sums[a] @ sums[a] - n[a]) / 2 for a in LANGUAGES)
    all_sum = (total @ total - big_n) / 2
    return {"matrix": matrix, "pooled_all": float(all_sum / all_pairs),
            "pooled_within": float(within_sum / within_pairs),
            "pooled_cross": float((all_sum - within_sum) / (all_pairs - within_pairs)),
            "pairs": {"all": all_pairs, "within": within_pairs, "cross": all_pairs - within_pairs}}


def neighbours(u: np.ndarray, candidates: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """For every row of u: top-k other candidates by cosine (ties: lower index first)."""
    sim = u @ u[candidates].T
    sim[candidates[None, :] == np.arange(len(u))[:, None]] = -np.inf
    order = np.argsort(-sim, axis=1, kind="stable")[:, :k]
    return candidates[order], np.take_along_axis(sim, order, axis=1)


def knn_stats(u: np.ndarray, cohort: Cohort, view: str, writer) -> dict:
    lang, y, subj = cohort.language, cohort.label, cohort.subjects
    big_n = len(u)
    nbr, sim = neighbours(u, np.arange(big_n), max(KS))
    for q in range(big_n):
        for r in range(max(KS)):
            j = nbr[q, r]
            writer.writerow([view, "pooled", subj[q], lang[q], int(y[q]), r + 1, subj[j], lang[j], int(y[j]),
                             f"{sim[q, r]:.6f}", f"{1 - sim[q, r]:.6f}"])
    pooled = {}
    for k in KS:
        same = (lang[nbr[:, :k]] == lang[:, None]).sum(1)
        per_lang = {}
        for a in LANGUAGES:
            m = lang == a
            n_a = int(m.sum())
            mix = {b: float((lang[nbr[m, :k]] == b).sum(1).mean()) for b in LANGUAGES}
            per_lang[a] = {"n": n_a, "same_mean": float(same[m].mean()), "same_pct": float(same[m].mean() / k),
                           "random_same_pct": (n_a - 1) / (big_n - 1), "capacity": n_a - 1,
                           "capacity_binding": n_a - 1 < k,
                           "same_quantiles": [float(v) for v in np.quantile(same[m], [0, 0.25, 0.5, 0.75, 1])],
                           "mixing_row": mix}
        pooled[str(k)] = {"overall_same_mean": float(same.mean()), "overall_same_pct": float(same.mean() / k),
                          "per_language": per_lang}
    within = {}
    pos_all, neg_all = [], []
    for a in LANGUAGES:
        idx = cohort.idx(a)
        nb, sm = neighbours(u, idx, K_WITHIN)
        nb, sm = nb[idx], sm[idx]
        for qi, q in enumerate(idx):
            for r in range(K_WITHIN):
                j = nb[qi, r]
                writer.writerow([view, "within", subj[q], lang[q], int(y[q]), r + 1, subj[j], lang[j], int(y[j]),
                                 f"{sm[qi, r]:.6f}", f"{1 - sm[qi, r]:.6f}"])
        frac = (y[nb] == y[idx][:, None]).mean(1)
        n_a, n_pos = len(idx), int(y[idx].sum())
        expect = np.where(y[idx] == 1, (n_pos - 1) / (n_a - 1), (n_a - n_pos - 1) / (n_a - 1))
        pos, neg = y[idx] == 1, y[idx] == 0
        pos_all.append((frac[pos], expect[pos]))
        neg_all.append((frac[neg], expect[neg]))
        within[a] = {"n": n_a, "n_pos": n_pos, "n_neg": n_a - n_pos,
                     "same_label": float(frac.mean()), "random_expectation": float(expect.mean()),
                     "pos_query": float(frac[pos].mean()), "pos_expectation": float(expect[pos].mean()),
                     "neg_query": float(frac[neg].mean()), "neg_expectation": float(expect[neg].mean()),
                     "balanced": float((frac[pos].mean() + frac[neg].mean()) / 2),
                     "balanced_expectation": float((expect[pos].mean() + expect[neg].mean()) / 2)}
    fp = np.concatenate([f for f, _ in pos_all]); fn = np.concatenate([f for f, _ in neg_all])
    ep = np.concatenate([e for _, e in pos_all]); en = np.concatenate([e for _, e in neg_all])
    within["Overall"] = {"same_label": float(np.concatenate([fp, fn]).mean()),
                         "random_expectation": float(np.concatenate([ep, en]).mean()),
                         "pos_query": float(fp.mean()), "neg_query": float(fn.mean()),
                         "balanced": float((fp.mean() + fn.mean()) / 2),
                         "balanced_expectation": float((ep.mean() + en.mean()) / 2)}
    return {"pooled_knn": pooled, "within_knn": within}


def _average(stats: list):
    if isinstance(stats[0], dict):
        return {k: _average([s[k] for s in stats]) for k in stats[0]}
    if isinstance(stats[0], (int, float)) and not isinstance(stats[0], bool):
        return float(np.mean(stats))
    if isinstance(stats[0], list):
        return [_average([s[i] for s in stats]) for i in range(len(stats[0]))]
    return stats[0]


def umap_inputs(cohort: Cohort, plans_path: Path, output: Path) -> int:
    maps = [{"name": "main_scaler_f0", "kind": "main", "points": list(range(len(cohort.subjects))),
             "scaler_keys": cohort.idx(None, 0, "train").tolist(),
             "roles": [str(r) for r in cohort.role[0]]}]
    key = {s: i for i, s in enumerate(cohort.subjects)}
    for line in plans_path.read_text().splitlines():
        p = json.loads(line)
        if p["experiment"] not in ("pair", "triple") or p["fold"] != 0 or p["seed"] != 0:
            continue
        x_all = cohort.idx(p["target"]).tolist()
        aux = [key[s] for a in p["aux"] for s in a["unique"]]
        x_unique = [key[s] for s in p["x_unique"]]
        roles = ["target TEST" if cohort.role[0][i] == "test" else "target TRAIN" for i in x_all]
        roles += [f"aux {cohort.language[i]}" for i in aux]
        for arm, keys in (("mixed_train", x_unique + aux), ("x_train_only", x_unique)):
            maps.append({"name": f"{p['experiment']}_{p['case'].replace('+', '_')}_{arm}_f0_s0",
                         "kind": p["experiment"], "points": x_all + aux, "scaler_keys": keys, "roles": roles})
    np.savez_compressed(output / "umap_speakers.npz", x=cohort.x.astype(np.float32),
                        subjects=np.asarray(cohort.subjects), language=cohort.language, label=cohort.label)
    (output / "umap_maps.json").write_text(json.dumps(maps))
    return len(maps)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--language", choices=("native", "english"), required=True)
    parser.add_argument("--merged-features-root", type=Path, required=True)
    parser.add_argument("--frozen-caches-root", type=Path, required=True)
    parser.add_argument("--fold-assignments", type=Path, required=True)
    parser.add_argument("--sampling-plans", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    data = load_merged(args.merged_features_root, args.frozen_caches_root, Path("/nonexistent"),
                       args.language, "text_only")
    cohort = Cohort(data, args.fold_assignments)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    views = {"raw": cohort.x}
    for f in FOLDS:
        mean, std = fit_scaler(cohort.x[cohort.idx(None, f, "train")])
        views[f"scaler_f{f}"] = (cohort.x - mean) / std
    per_view = {}
    with gzip.open(args.output_dir / "neighbours.csv.gz", "wt", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["view", "scope", "query", "query_language", "query_label", "rank", "neighbour",
                    "neighbour_language", "neighbour_label", "cosine", "distance"])
        for name, x in views.items():
            u = unit(x)
            per_view[name] = {"cosine": pair_cosines(u, cohort.language), **knn_stats(u, cohort, name, w)}
    normalized = _average([per_view[f"scaler_f{f}"] for f in FOLDS])
    n_maps = umap_inputs(cohort, args.sampling_plans, args.output_dir)
    git_head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=PROJECT_ROOT).stdout.strip()
    result = {"git_head": git_head, "language": args.language, "n_speakers": len(cohort.subjects),
              "fold_assignments_sha256": sha256_file(args.fold_assignments), "feature_provenance": data["provenance"],
              "raw": per_view["raw"], "normalized_mean_of_5_views": normalized, "per_view": per_view,
              "umap_maps": n_maps}
    (args.output_dir / "geometry.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({"n_speakers": result["n_speakers"], "pairs": per_view["raw"]["cosine"]["pairs"],
                      "umap_maps": n_maps}))


if __name__ == "__main__":
    main()
