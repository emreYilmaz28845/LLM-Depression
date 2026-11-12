"""Seed ensembles and seed-agreement selection for direct text vs audio+text LLMs.

No model inference and no tuning on labels. Rules fixed before looking at outer results:

- text_ens3: soft vote of the three text seeds (sign of the mean scaled likelihood margin).
- at_ens3: the same for the three audio+text seeds (descriptive only, not tested).
- ens6: soft vote of all six models (3 text + 3 audio+text seeds).
- seed_agree: if text_ens3 and at_ens3 disagree, trust the modality whose seeds agree more
  (3/3 beats 2/3); tie -> text_ens3.

Margins are scaled per model, seed and fold by the median absolute margin on that fold's
inner validation (label-free), as in margin_selector_analysis.py.

Ensemble methods use three or six trained models; single-seed references use one. The
secondary family (vs text_ens3) asks whether audio+text adds anything beyond the variance
reduction that seed ensembling alone gives.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from knn_reliability_analysis import CORPORA, aggregate, holm, sha, write  # noqa: E402
from margin_selector_analysis import SEEDS, join, load_cells, scale_of, uncertainty  # noqa: E402

MODS = ("text_only", "audio_text")
METHODS = ("text_ens3", "ens6", "seed_agree")
PRIMARY = METHODS
SECONDARY = ("ens6", "seed_agree")


def soft_vote(z):
    """z: (n_models, n_subjects) scaled margins -> 0/1 by sign of the mean; 0 mean -> 0."""
    return (np.mean(z, 0) > 0).astype(int)


def agreement(z):
    """Number of models agreeing with the majority label (for 3 models: 2 or 3)."""
    pos = (z > 0).sum(0)
    return np.maximum(pos, z.shape[0] - pos)


def seed_agree(zt, za):
    pt, pa = soft_vote(zt), soft_vote(za)
    return np.where((pt != pa) & (agreement(za) > agreement(zt)), pa, pt)


def run(args):
    out = args.output; out.mkdir(parents=True, exist_ok=False)
    t0 = time.time(); evidence = []

    def record(path, role):
        evidence.append({"role": role, "path": str(path), "sha256": sha(path)})

    cells = load_cells(args.cell_provenance, record, mods=MODS)
    rows = []
    for ds in CORPORA:
        folds = sorted({k[2] for k in cells if k[0] == ds})
        for fold in folds:
            z = {"t": [], "a": []}; base = None
            for seed in SEEDS:
                ct, ca = cells[(ds, seed, fold, "text_only")], cells[(ds, seed, fold, "audio_text")]
                where = f"{ds} seed {seed} fold {fold}"
                inner = join(ct["inner"], ca["inner"], "inner " + where)
                o = join(ct["outer"], ca["outer"], "outer " + where).sort_values("subject_id").reset_index(drop=True)
                if base is None:
                    base = o[["subject_id", "label"]]
                elif not (o.subject_id.equals(base.subject_id) and o.label.equals(base.label)):
                    raise ValueError("Seeds do not share outer subjects/labels in " + where)
                z["t"].append(o.margin_t.to_numpy() / scale_of(inner.margin_t))
                z["a"].append(o.margin_at.to_numpy() / scale_of(inner.margin_at))
                if not (((o.margin_t > 0).astype(int) == o.prediction_t).all()
                        and ((o.margin_at > 0).astype(int) == o.prediction_at).all()):
                    raise ValueError("Margin sign disagrees with prediction in " + where)
            zt, za = np.stack(z["t"]), np.stack(z["a"])
            ens = {"text_ens3": soft_vote(zt), "at_ens3": soft_vote(za),
                   "ens6": soft_vote(np.concatenate([zt, za])), "seed_agree": seed_agree(zt, za)}
            for k, seed in enumerate(SEEDS):
                r = base.copy(); r["corpus"], r["seed"], r["fold"] = ds, seed, fold
                r["text"], r["audio_text"] = (zt[k] > 0).astype(int), (za[k] > 0).astype(int)
                for name, p in ens.items():
                    r[name] = p  # seed-independent; repeated so paired tests align per seed
                r["oracle"] = np.where(r.text == r.label, r.text, r.audio_text)
                r["text_seed_agreement"], r["at_seed_agreement"] = agreement(zt), agreement(za)
                rows.append(r)
        print("done", ds, flush=True)
    df = pd.concat(rows, ignore_index=True)

    shown = ("text", "audio_text", "text_ens3", "at_ens3", "ens6", "seed_agree", "oracle")
    perf = [{"corpus": ds, "seed": s, "method": m, **aggregate(h, m)}
            for (ds, s), h in df.groupby(["corpus", "seed"], sort=False) for m in shown]
    # Diagnostic: is seed agreement a confidence signal? Accuracy of the seed-ensemble vote by agreement level.
    diag = []
    first = df[df.seed == SEEDS[0]]
    for ds, g in first.groupby("corpus", sort=False):
        for mod, col, ag in (("text", "text_ens3", "text_seed_agreement"), ("audio_text", "at_ens3", "at_seed_agreement")):
            for level in (2, 3):
                h = g[g[ag] == level]
                diag.append({"corpus": ds, "model": mod, "seeds_agreeing": level, "n": len(h),
                             "share": len(h) / len(g), "ensemble_accuracy": float((h[col] == h.label).mean()) if len(h) else None})
    tests = []
    for ds, g in df.groupby("corpus", sort=False):
        for m in PRIMARY:
            tests.append({"corpus": ds, "method": m, "reference": "text", "family": "primary_vs_single_seed_text",
                          **uncertainty(g, m, "text", args.reps, args.seed)})
        for m in SECONDARY:
            tests.append({"corpus": ds, "method": m, "reference": "text_ens3", "family": "secondary_vs_text_ens3",
                          **uncertainty(g, m, "text_ens3", args.reps, args.seed)})
    td = pd.DataFrame(tests); td["holm_p"] = 1.
    for _, g in td.groupby("family"):
        td.loc[g.index, "holm_p"] = holm(g.permutation_p.to_numpy())

    df.to_csv(out / "private_seed_ensemble_subjects.csv", index=False)
    pd.DataFrame(perf).to_csv(out / "seed_ensemble_performance.csv", index=False)
    pd.DataFrame(diag).to_csv(out / "seed_agreement_diagnostic.csv", index=False)
    td.to_csv(out / "seed_ensemble_tests.csv", index=False)
    write(out / "seed_ensemble_provenance.json", evidence)
    write(out / "seed_ensemble_cells.json", [{k: v for k, v in c.items() if k not in ("outer", "inner")} | {"key": list(map(str, key))}
                                             for key, c in cells.items()])
    write(out / "seed_ensemble_config.json", {
        "source_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                        cwd=Path(__file__).parent).stdout.strip(),
        "source_script_sha256": sha(__file__), "seeds": SEEDS, "reps": args.reps, "rng_seed": args.seed,
        "rules": {"text_ens3": "sign of mean scaled margin over 3 text seeds",
                  "at_ens3": "sign of mean scaled margin over 3 audio+text seeds (descriptive)",
                  "ens6": "sign of mean scaled margin over 3 text + 3 audio+text seeds",
                  "seed_agree": "if text_ens3 != at_ens3, take the modality with more seeds agreeing; tie -> text_ens3"},
        "scale": "z = margin / median |margin| of the same model, seed and fold on inner validation (label-free)",
        "aggregation": "equal-weight fold-mean macro-F1; ensembles are seed-independent, references are the mean over 3 single seeds",
        "tests": "class-stratified within-fold subject bootstrap joint over seeds; prediction-swap permutation; Holm within "
                 "primary family (3 methods x 5 corpora vs single-seed text) and secondary family (2 methods x 5 corpora vs text_ens3)",
        "limitations": ["ensembles use 3 or 6 trained models; single-seed references use 1",
                        "bootstrap conditions on fixed models; seed training variance enters only through the three available seeds",
                        "Turkish and CMDC are outer validation, not locked test; DAIC is one official test fold",
                        "Turkish/CMDC margin scales come from the same subjects (label-free)"],
        "runtime_seconds": round(time.time() - t0, 1)})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cell-provenance", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--reps", type=int, default=5000)
    p.add_argument("--seed", type=int, default=20261113)
    run(p.parse_args())


if __name__ == "__main__":
    main()
