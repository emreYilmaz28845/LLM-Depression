#!/usr/bin/env python3
"""Text replication of Kaan's protocol, cores B, C and D (limited-auxiliary augmentation).

- ``xonly``: X TRAIN -> X TEST (same fits as core A1), the baseline of every case.
- B ``pair``: X + limited Y -> X TEST, all 20 directional pairs.
- C ``triple``: X + limited Y + limited Z -> X TEST, all 30 (X, {Y, Z}) cases.
- D ``all_italian``: Spanish + all 116 Italian speakers (uncapped) -> Spanish TEST.

Auxiliary speakers come from the auxiliary language's whole pool (TRAIN + TEST); X TEST
never enters a fit, a scaler or a selection. X's draw is the core-A reference draw for
the seed. B budget: per class at most (|X|//2)//2 unique auxiliary speakers, duplicated
to len(X draw)//4 rows; C: (|X|//2)//4 unique and len(X draw)//8 rows per class and per
auxiliary. Samplers are seeded with ``SeedSequence`` exactly as in the handoff, with
the language order English, Turkish, Italian, Spanish, Mandarin and speakers sorted by
key. Two scaler arms share one plan: ``mixed_train`` (unique X TRAIN + selected
auxiliary speakers) and ``x_train_only``. C and threshold are X's fixed settings.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.dataset_probe import load_merged  # noqa: E402
from baselines.text_replication import (FOLDS, LANGUAGES, SEEDS, SETTINGS, Cohort, fit_rows,  # noqa: E402
                                        metrics, oversample_draw, predict, sha256_file, summarize)

ARMS = ("mixed_train", "x_train_only")
BASE_SEED = 20261108
_COHORT: Cohort | None = None


def _select(rng: np.random.Generator, pool: np.ndarray, labels: np.ndarray, unique_per_class: int,
            draw_per_class: int) -> tuple[np.ndarray, np.ndarray]:
    """Labels 0 then 1: sample unique speakers without replacement, duplicate inside the class."""
    selected, draws = [], []
    for lab in (0, 1):
        avail = pool[labels[pool] == lab]
        k = min(len(avail), unique_per_class)
        sel = rng.choice(avail, k, replace=False)
        draws.append(np.concatenate([sel, rng.choice(sel, draw_per_class - k, replace=True)]))
        selected.append(sel)
    return np.sort(np.concatenate(selected)), np.concatenate(draws)


def build_plans(cohort: Cohort) -> list[dict]:
    plans = []
    y = cohort.label
    for xi, x_lang in enumerate(LANGUAGES):
        folds = (0,) if x_lang == "English" else FOLDS
        others = [lang for lang in LANGUAGES if lang != x_lang]
        for fold in folds:
            x_unique = cohort.idx(x_lang, fold, "train")
            x_test = cohort.idx(x_lang, fold, "test")
            for seed in SEEDS:
                xd = x_unique[oversample_draw(y[x_unique], seed)]
                base = {"target": x_lang, "fold": fold, "seed": seed, "x_unique": x_unique, "x_draw": xd,
                        "x_test": x_test}
                plans.append({**base, "experiment": "xonly", "case": x_lang, "aux": []})
                n = len(x_unique)
                for y_lang in others:
                    yi = LANGUAGES.index(y_lang)
                    rng = np.random.default_rng(np.random.SeedSequence([BASE_SEED, xi, yi, int(fold), seed]))
                    sel, yd = _select(rng, cohort.idx(y_lang), y, (n // 2) // 2, len(xd) // 4)
                    rng.shuffle(yd)
                    plans.append({**base, "experiment": "pair", "case": f"{x_lang}+{y_lang}",
                                  "aux": [{"language": y_lang, "unique": sel, "draw": yd}]})
                for ci, combo in enumerate(combinations(others, 2)):
                    aux = []
                    for slot, z_lang in enumerate(combo):
                        rng = np.random.default_rng(
                            np.random.SeedSequence([BASE_SEED, 3, xi, ci, int(fold), seed, slot]))
                        sel, zd = _select(rng, cohort.idx(z_lang), y, (n // 2) // 4, len(xd) // 8)
                        aux.append({"language": z_lang, "unique": sel, "draw": zd})
                    plans.append({**base, "experiment": "triple", "case": f"{x_lang}+{combo[0]}+{combo[1]}",
                                  "aux": aux})
                if x_lang == "Spanish":
                    italian = cohort.idx("Italian")
                    rng_seed = np.random.SeedSequence([BASE_SEED, 4, int(fold), seed])
                    plans.append({**base, "experiment": "all_italian", "case": "Spanish+allItalian",
                                  "aux": [{"language": "Italian", "unique": italian,
                                           "draw": italian[oversample_draw(y[italian], rng_seed)]}]})
    return plans


def check_plan(plan: dict, cohort: Cohort) -> None:
    y = cohort.label
    xd, xu = plan["x_draw"], plan["x_unique"]
    assert np.array_equal(xd, xu[oversample_draw(y[xu], plan["seed"])]), "X draw differs from the X-only draw"
    assert (y[xd] == 0).sum() == (y[xd] == 1).sum()
    test = set(plan["x_test"].tolist())
    fit_keys = set(xu.tolist()) | {int(i) for a in plan["aux"] for i in a["unique"]}
    assert not test & fit_keys, "X TEST in a fitting input"
    for a in plan["aux"]:
        assert set(a["draw"].tolist()) == set(a["unique"].tolist())
        assert (y[a["draw"]] == 0).sum() == (y[a["draw"]] == 1).sum(), "auxiliary rows not class-balanced"
        assert (cohort.language[a["unique"]] == a["language"]).all()
    exp = plan["experiment"]
    if exp == "pair":
        a = plan["aux"][0]
        assert len(a["unique"]) * 2 <= len(xu) and len(a["draw"]) * 2 <= len(xd)
    if exp == "triple":
        u = [len(a["unique"]) for a in plan["aux"]]
        d = [len(a["draw"]) for a in plan["aux"]]
        assert d[0] == d[1] and all(4 * v <= len(xu) for v in u) and all(4 * v <= len(xd) for v in d)
        assert 2 * sum(u) <= len(xu) and 2 * sum(d) <= len(xd)
    if exp == "all_italian":
        a = plan["aux"][0]
        assert len(a["unique"]) == 116 and len(a["draw"]) == 128


def _init(cohort: Cohort) -> None:
    from threadpoolctl import threadpool_limits

    global _COHORT
    _COHORT = cohort
    threadpool_limits(1)


def _fit(task: tuple[dict, str]) -> dict:
    plan, arm = task
    cohort = _COHORT
    c, thresholds = SETTINGS[plan["target"]]
    threshold = thresholds[plan["seed"]]
    rows = np.concatenate([plan["x_draw"], *[a["draw"] for a in plan["aux"]]])
    keys = plan["x_unique"] if arm == "x_train_only" else np.concatenate(
        [plan["x_unique"], *[a["unique"] for a in plan["aux"]]])
    model, scaler, conv = fit_rows(cohort.x, cohort.label, rows, keys, c)
    test = plan["x_test"]
    logit, prob, pred = predict(model, scaler, cohort.x[test], threshold)
    return {"experiment": plan["experiment"], "case": plan["case"], "target": plan["target"], "arm": arm,
            "fold": plan["fold"], "seed": plan["seed"], "C": c, "threshold": threshold,
            "source": f"{plan['case']}|{arm}", "eval": plan["target"],
            "x_unique": int(len(plan["x_unique"])), "x_rows": int(len(plan["x_draw"])),
            "aux_counts": {a["language"]: {"unique": int(len(a["unique"])), "rows": int(len(a["draw"])),
                                           "unique_pos": int(cohort.label[a["unique"]].sum())}
                           for a in plan["aux"]},
            "scaler_unique": int(len(np.unique(keys))), **conv, **metrics(cohort.label[test], pred),
            "predictions": [{"subject": cohort.subjects[i], "label": int(cohort.label[i]), "logit": float(lg),
                             "prob": float(p), "pred": int(pr)} for i, lg, p, pr in zip(test, logit, prob, pred)]}


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
    plans = build_plans(cohort)
    for p in plans:
        check_plan(p, cohort)
    n_plans = Counter(p["experiment"] for p in plans)
    assert n_plans == {"xonly": 63, "pair": 252, "triple": 378, "all_italian": 15}, n_plans
    tasks = [(p, arm) for p in plans for arm in (("x_train_only",) if p["experiment"] == "xonly" else ARMS)]
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init, initargs=(cohort,)) as pool:
        results = list(pool.map(_fit, tasks, chunksize=4))
    fits = Counter(r["experiment"] for r in results)
    assert fits == {"xonly": 63, "pair": 504, "triple": 756, "all_italian": 30}, fits

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "sampling_plans.jsonl").open("w") as fh:
        for p in plans:
            fh.write(json.dumps({"experiment": p["experiment"], "case": p["case"], "target": p["target"],
                                 "fold": p["fold"], "seed": p["seed"],
                                 "x_unique": [cohort.subjects[i] for i in p["x_unique"]],
                                 "x_draw": [cohort.subjects[i] for i in p["x_draw"]],
                                 "aux": [{"language": a["language"],
                                          "unique": [cohort.subjects[i] for i in a["unique"]],
                                          "draw": [cohort.subjects[i] for i in a["draw"]]} for a in p["aux"]]})
                     + "\n")
    with (args.output_dir / "fits.jsonl").open("w") as fh:
        for r in results:
            fh.write(json.dumps({k: v for k, v in r.items() if k != "predictions"}) + "\n")
    with (args.output_dir / "predictions.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["experiment", "case", "arm", "target", "fold", "seed", "threshold", "subject", "label",
                    "logit", "prob", "pred"])
        for r in results:
            for p in r["predictions"]:
                w.writerow([r["experiment"], r["case"], r["arm"], r["target"], r["fold"], r["seed"], r["threshold"],
                            p["subject"], p["label"], p["logit"], p["prob"], p["pred"]])
    summary = summarize(results)
    git_head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=PROJECT_ROOT).stdout.strip()
    result = {"git_head": git_head, "language": args.language, "fold_assignments_sha256": sha256_file(args.fold_assignments),
              "feature_provenance": data["provenance"], "plans": dict(n_plans), "fits": dict(fits),
              "all_converged": all(r["converged"] for r in results),
              "sampling_plans_sha256": hashlib.sha256((args.output_dir / "sampling_plans.jsonl").read_bytes()).hexdigest(),
              "summary": summary}
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({k: result[k] for k in ("plans", "fits", "all_converged")}))


if __name__ == "__main__":
    main()
