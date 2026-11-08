#!/usr/bin/env python3
"""Text replication of Kaan's protocol, core F: frozen comparison registry and paired tests.

``register`` writes the comparison family from the fitted-model structure only (no
scores are read), before any test is run. ``analyze`` refuses to run if the registry
changed. Every comparison is between two models scored on exactly the same
(fold, seed, speaker) cells with the same aggregation: macro-F1 per cell, seed mean per
fold, equal mean over folds.

- Paired speaker-block swap test, two-sided: all predictions of one speaker (every seed
  and every fold it appears in) are swapped together. Exact enumeration for <= 16
  discordant speakers, otherwise 100000 random swaps with the plus-one correction.
- Paired bootstrap, 10000 samples: speakers resampled with replacement inside
  (folds, language, true label) strata, all their predictions kept together; percentile
  95% CI of the difference (B - A).
- Holm correction over the whole registered family; raw and adjusted p are kept.

Registered family (per text language):
- B: X+Y (each scaler arm) vs X-only.
- C: X+Y+Z (each arm) vs X-only, vs X+Y and vs X+Z in the same arm.
- D: Spanish + all Italian (each arm) vs Spanish-only and vs capped Spanish+Italian.
- A3 vs A1: Combined -> X vs X -> X for the four non-English targets (English is
  excluded: A1 English has one fit per seed, A3 English five, so the cells differ).
- Section 12: every arm vs ``base`` on each language TEST and the pooled TEST.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.text_replication import LANGUAGES  # noqa: E402

TEXT_LANGUAGES = ("native", "english")
ARMS = ("mixed_train", "x_train_only")
S12_ARMS = ("onehot_lr", "perlang_z_lr", "global_leace_lr", "perlang_leace_lr",
            "xgb_all", "xgb_onehot", "xgb_top50", "xgb_top100")
EXACT_MAX = 16
N_SWAPS = 100_000
N_BOOT = 10_000
SEED = 20261108


def build_registry() -> list[dict]:
    reg = []
    for tl in TEXT_LANGUAGES:
        def add(family, a, b, label):
            reg.append({"id": len(reg), "text": tl, "family": family, "a": a, "b": b, "label": label})
        for x in LANGUAGES:
            xonly = f"aug|{x}|x_train_only"
            others = [o for o in LANGUAGES if o != x]
            for arm in ARMS:
                for y in others:
                    add("B", xonly, f"aug|{x}+{y}|{arm}", f"{x}+{y} [{arm}] vs {x}-only")
                for y, z in combinations(others, 2):
                    tri = f"aug|{x}+{y}+{z}|{arm}"
                    add("C", xonly, tri, f"{x}+{y}+{z} [{arm}] vs {x}-only")
                    add("C", f"aug|{x}+{y}|{arm}", tri, f"{x}+{y}+{z} [{arm}] vs {x}+{y}")
                    add("C", f"aug|{x}+{z}|{arm}", tri, f"{x}+{y}+{z} [{arm}] vs {x}+{z}")
        for arm in ARMS:
            add("D", "aug|Spanish|x_train_only", f"aug|Spanish+allItalian|{arm}", f"Spanish+allItalian [{arm}] vs Spanish-only")
            add("D", f"aug|Spanish+Italian|{arm}", f"aug|Spanish+allItalian|{arm}",
                f"Spanish+allItalian [{arm}] vs capped Spanish+Italian")
        for x in LANGUAGES[1:]:
            add("A3vsA1", f"A|same|{x}|{x}", f"A|combined|Combined|{x}", f"Combined -> {x} vs {x} -> {x}")
        for arm in S12_ARMS:
            for ev in (*LANGUAGES, "Pooled"):
                add("S12", f"s12|base|{ev}", f"s12|{arm}|{ev}", f"{arm} vs base on {ev}")
    return reg


def load_predictions(roots: dict[str, dict[str, Path]]) -> tuple[dict, dict]:
    """{text: {model_id: {(fold, seed, subject): pred}}}, {subject: label}."""
    preds: dict = {tl: defaultdict(dict) for tl in roots}
    labels: dict = {}
    for tl, paths in roots.items():
        for kind, path in paths.items():
            with (path / "predictions.csv").open() as fh:
                for r in csv.DictReader(fh):
                    if kind == "A":
                        mid = f"A|{r['experiment']}|{r['source']}|{r['eval']}"
                    elif kind == "aug":
                        mid = f"aug|{r['case']}|{r['arm']}"
                    else:
                        mid = f"s12|{r['arm']}|{r['eval']}"
                    fold = 0 if r["fold"] in ("", "None") else int(r["fold"])
                    preds[tl][mid][(fold, int(r["seed"]), r["subject"])] = int(r["pred"])
                    labels[r["subject"]] = int(r["label"])
    return preds, labels


def _f1(tp, fp, npos, nneg):
    fn, tn = npos - tp, nneg - fp
    with np.errstate(invalid="ignore", divide="ignore"):
        fpos = np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), 0.0)
        fneg = np.where(2 * tn + fn + fp > 0, 2 * tn / (2 * tn + fn + fp), 0.0)
    return (fpos + fneg) / 2


class Pair:
    """Two models on identical cells, in (cell x speaker) matrices."""

    def __init__(self, a: dict, b: dict, labels: dict):
        if set(a) != set(b):
            raise ValueError("Models are not scored on the same (fold, seed, speaker) cells.")
        cells = sorted({(f, s) for f, s, _ in a})
        subjects = sorted({u for _, _, u in a})
        ci = {c: i for i, c in enumerate(cells)}
        ui = {u: i for i, u in enumerate(subjects)}
        self.mask = np.zeros((len(cells), len(subjects)))
        self.pa = np.zeros_like(self.mask)
        self.pb = np.zeros_like(self.mask)
        for (f, s, u), v in a.items():
            self.mask[ci[(f, s)], ui[u]] = 1
            self.pa[ci[(f, s)], ui[u]] = v
            self.pb[ci[(f, s)], ui[u]] = b[(f, s, u)]
        self.y = np.asarray([labels[u] for u in subjects], dtype=np.float64)
        folds = sorted({f for f, _ in cells})
        seeds_in = {f: sum(1 for c in cells if c[0] == f) for f in folds}
        self.w = np.asarray([1.0 / (len(folds) * seeds_in[f]) for f, _ in cells])
        self.subjects = subjects
        self.folds_of = [tuple(sorted({cells[c][0] for c in np.flatnonzero(self.mask[:, j])})) for j in range(len(subjects))]

    def _counts(self, pred, weights=None):
        """TP, FP, npos, nneg per cell; ``weights`` (n x speakers) gives bootstrap counts."""
        y, m = self.y, self.mask
        parts = (pred * y * m, pred * (1 - y) * m, y * m, (1 - y) * m)
        if weights is None:
            return [p.sum(1) for p in parts]
        return [weights @ p.T for p in parts]

    def score(self, pred, weights=None):
        tp, fp, npos, nneg = self._counts(pred, weights)
        return _f1(tp, fp, npos, nneg) @ self.w

    def observed(self) -> tuple[float, float, float]:
        sa, sb = float(self.score(self.pa)), float(self.score(self.pb))
        return sa, sb, sb - sa

    def swap_test(self, seed) -> tuple[float, int, str]:
        disc = np.flatnonzero(((self.pa != self.pb) & (self.mask > 0)).any(0))
        k = len(disc)
        _, _, d_obs = self.observed()
        if k == 0:
            return 1.0, 0, "none"
        y, m = self.y[disc], self.mask[:, disc]
        da = (self.pb - self.pa)[:, disc] * m
        base_a = self._counts(self.pa)
        base_b = self._counts(self.pb)
        dtp, dfp = da * y, da * (1 - y)

        def stats(masks):
            tpa = base_a[0] + masks @ dtp.T
            fpa = base_a[1] + masks @ dfp.T
            tpb = base_b[0] - masks @ dtp.T
            fpb = base_b[1] - masks @ dfp.T
            sa = _f1(tpa, fpa, base_a[2], base_a[3]) @ self.w
            sb = _f1(tpb, fpb, base_b[2], base_b[3]) @ self.w
            return sb - sa

        if k <= EXACT_MAX:
            codes = np.arange(2 ** k, dtype=np.int64)
            masks = ((codes[:, None] >> np.arange(k)) & 1).astype(np.float64)
            d = stats(masks)
            return float(np.mean(np.abs(d) >= abs(d_obs) - 1e-12)), k, "exact"
        rng = np.random.default_rng(seed)
        hits = 0
        for start in range(0, N_SWAPS, 20_000):
            masks = rng.integers(0, 2, size=(min(20_000, N_SWAPS - start), k)).astype(np.float64)
            hits += int(np.sum(np.abs(stats(masks)) >= abs(d_obs) - 1e-12))
        return (1 + hits) / (1 + N_SWAPS), k, "random"

    def bootstrap(self, seed) -> tuple[float, float]:
        rng = np.random.default_rng(seed)
        strata = defaultdict(list)
        for j, u in enumerate(self.subjects):
            strata[(self.folds_of[j], u.split("::")[0], int(self.y[j]))].append(j)
        deltas = []
        for start in range(0, N_BOOT, 2_000):
            n = min(2_000, N_BOOT - start)
            w = np.zeros((n, len(self.subjects)))
            for members in strata.values():
                draws = rng.integers(0, len(members), size=(n, len(members)))
                np.add.at(w, (np.repeat(np.arange(n), len(members)), np.asarray(members)[draws.ravel()]), 1)
            deltas.append(self.score(self.pb, w) - self.score(self.pa, w))
        d = np.concatenate(deltas)
        return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def holm(p: list[float]) -> list[float]:
    order = np.argsort(p, kind="stable")
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj.tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("register")
    r.add_argument("--output-dir", type=Path, required=True)
    a = sub.add_parser("analyze")
    a.add_argument("--output-dir", type=Path, required=True)
    for tl in TEXT_LANGUAGES:
        for kind in ("A", "aug", "s12"):
            a.add_argument(f"--{kind.lower()}-{tl}", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reg_path = args.output_dir / "registry.json"
    if args.cmd == "register":
        if reg_path.exists():
            raise SystemExit(f"{reg_path} exists; the registry is frozen.")
        reg = build_registry()
        reg_path.write_text(json.dumps(reg, indent=1))
        (args.output_dir / "registry.sha256").write_text(hashlib.sha256(reg_path.read_bytes()).hexdigest() + "\n")
        print(len(reg), "comparisons registered")
        return
    expected = (args.output_dir / "registry.sha256").read_text().strip()
    if hashlib.sha256(reg_path.read_bytes()).hexdigest() != expected or json.loads(reg_path.read_text()) != build_registry():
        raise SystemExit("Registry changed after it was frozen.")
    reg = json.loads(reg_path.read_text())
    roots = {tl: {k: getattr(args, f"{k.lower()}_{tl}") for k in ("A", "aug", "s12")} for tl in TEXT_LANGUAGES}
    preds, labels = load_predictions(roots)
    # Core A1 and the X-only arm of B/C/D are the same fits: their predictions must agree.
    for tl in TEXT_LANGUAGES:
        for x in LANGUAGES:
            if preds[tl][f"A|same|{x}|{x}"] != preds[tl][f"aug|{x}|x_train_only"]:
                raise SystemExit(f"{tl}: A1 {x} and the X-only augmentation baseline differ.")
    rows = []
    for c in reg:
        pair = Pair(preds[c["text"]][c["a"]], preds[c["text"]][c["b"]], labels)
        sa, sb, d = pair.observed()
        p, k, method = pair.swap_test(np.random.SeedSequence([SEED, 1, c["id"]]))
        lo, hi = pair.bootstrap(np.random.SeedSequence([SEED, 2, c["id"]]))
        rows.append({**c, "score_a": sa, "score_b": sb, "delta": d, "ci_low": lo, "ci_high": hi,
                     "p_raw": p, "discordant_speakers": k, "method": method, "n_speakers": len(pair.subjects),
                     "n_cells": len(pair.w)})
    for row, adj in zip(rows, holm([r["p_raw"] for r in rows])):
        row["p_holm"] = adj
    git_head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=PROJECT_ROOT).stdout.strip()
    with (args.output_dir / "comparisons.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    out = {"git_head": git_head, "registry_sha256": expected, "n_comparisons": len(rows),
           "inputs": {tl: {k: str(v) for k, v in p.items()} for tl, p in roots.items()},
           "settings": {"exact_max_discordant": EXACT_MAX, "n_swaps": N_SWAPS, "n_bootstrap": N_BOOT},
           "n_raw_below_0.05": sum(r["p_raw"] < 0.05 for r in rows),
           "n_holm_below_0.05": sum(r["p_holm"] < 0.05 for r in rows)}
    (args.output_dir / "summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
