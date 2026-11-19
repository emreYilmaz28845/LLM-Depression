"""Equal-fraction near-cut removal: drop the same share of training subjects in every corpus.

Fixed score bands (BDI 14-19, PHQ 8-11) remove 18-26 % of subjects depending on the corpus.
Here, inside each CV training fold and corpus, the ``frac`` share of training subjects
closest to the label cut (|score - (cut - 0.5)|, seeded random tie-break) is removed before
fitting the standalone L2 logistic head on frozen subject-mean embeddings. Test always uses
every subject of the held-out fold. Penalty is chosen on inner stratified folds of the kept
training subjects. Paired bootstrap vs frac=0, Holm over all (corpus, frac) tests per source.

Output: ``<output>/<source>/band_fraction.json``.
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

from baselines.band_score_experiments import SCORED, select_penalty, _fit  # noqa: E402
from baselines.merge_diag_common import load_source, metrics  # noqa: E402
from baselines.merge_diagnostics import BANDS, boot_diff, fold_mean, git_state, holm, load_scores  # noqa: E402

SEED = 20261119
FRACS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)


def near_cut_drop(s: np.ndarray, d: str, frac: float, rng: np.random.Generator) -> np.ndarray:
    """Boolean mask of the round(frac*n) subjects closest to the cut of corpus ``d``."""
    n = len(s)
    k = int(round(frac * n))
    out = np.zeros(n, dtype=bool)
    if k == 0:
        return out
    dist = np.abs(s - (BANDS[d][0] - 0.5))
    order = np.lexsort((rng.random(n), dist))
    out[order[:k]] = True
    return out


def run(src: dict) -> dict:
    x, g, y, fold, subjects = src["x"], src["g"], src["y"], src["fold"], src["subjects"]
    scores = load_scores()
    s = np.asarray([scores.get(str(sub), np.nan) if d in SCORED else np.nan for sub, d in zip(subjects, g)])
    cvm = fold >= 0
    oof = {f: np.full(len(y), np.nan) for f in FRACS}
    log = {f: [] for f in FRACS}
    for k in range(5):
        tr, te = cvm & (fold != k), fold == k
        for d in SCORED:
            i = np.flatnonzero(tr & (g == d))
            j = te & (g == d)
            if np.isnan(s[i]).any():
                raise ValueError(f"{d}: training subjects without a score")
            for f in FRACS:
                rng = np.random.default_rng([SEED, k, SCORED.index(d)])
                drop = near_cut_drop(s[i], d, f, rng)
                keep = i[~drop]
                yk = y[keep]
                pen = select_penalty(x[keep], g[keep], yk, s[keep], "global", "logit_base", seed=k)
                oof[f][j] = _fit(x[keep], g[keep], yk, s[keep], "global", "logit_base", pen)(x[j], g[j])
                dd = np.abs(s[i][drop] - (BANDS[d][0] - 0.5))
                log[f].append({"fold": k, "dataset": d, "penalty": pen, "n_train": int(len(i)),
                               "n_dropped": int(drop.sum()), "n_pos_dropped": int(y[i][drop].sum()),
                               "max_dist_dropped": float(dd.max()) if len(dd) else None,
                               "pos_rate_kept": float(yk.mean())})
        print(f"  fold {k} done", flush=True)
    rng = np.random.default_rng(SEED)
    table, tests = {}, {}
    for d in SCORED:
        m = cvm & (g == d)
        table[d] = {str(f): {**fold_mean(y, oof[f], g, fold, d), "pooled": metrics(y[m], oof[f][m])}
                    for f in FRACS}
        for f in FRACS[1:]:
            tests[f"{d}|{f}_vs_0"] = boot_diff(y[m], oof[f][m], oof[0.0][m], rng)
    adj = holm({kk: v["p"] for kk, v in tests.items()})
    for kk in tests:
        tests[kk]["p_holm"] = adj[kk]
    return {"fracs": list(FRACS), "table": table, "tests": tests, "train_log": {str(f): v for f, v in log.items()},
            "oof": {str(f): v.tolist() for f, v in oof.items()}, "subjects": subjects.tolist()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--source", action="append", required=True, choices=("text_native", "audio_native"))
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    for name in args.source:
        t0 = time.time()
        src = load_source(name)
        print(f"[{name}] band_fraction ...", flush=True)
        res = run(src)
        out = args.output_dir / name / "band_fraction.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {**git_state(), "source": name, "analysis": "band_fraction", "seed": SEED, "provenance": src["provenance"],
                   "seconds": round(time.time() - t0, 1), "result": res}
        out.write_text(json.dumps(payload, indent=1))
        print(f"[{name}] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
