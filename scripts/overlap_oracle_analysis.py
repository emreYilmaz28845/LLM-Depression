"""Error overlap, oracle ceiling and seed-noise check for text vs audio+text models.

Works on direct LLM likelihood cells or LoRA-embedding head cells (--backend). No inference.

- Overlap: per corpus and seed, pooled outer subjects: both wrong, only text wrong,
  only A+T wrong, shared errors as % of the error union, exact McNemar p.
- Oracle gain, text vs A+T: oracle fold-mean Macro-F1 minus the better of the two models.
- Seed noise: the same oracle gain for two seeds of the same model (text, A+T), against the
  better of the two seeds. If seed pairs give similar gains, much of the text vs A+T oracle
  gap is training noise rather than complementary modality information.
"""
from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from knn_reliability_analysis import CORPORA, aggregate, holm, sha, write  # noqa: E402
from margin_selector_analysis import SEEDS, add_backend_args, get_cells  # noqa: E402

MODS = ("text_only", "audio_text")


def frames(cells, ds, seed, mod):
    folds = sorted({k[2] for k in cells if k[0] == ds and k[1] == seed and k[3] == mod})
    parts = []
    for f in folds:
        d = cells[(ds, seed, f, mod)]["outer"].copy(); d["fold"] = f; parts.append(d)
    return pd.concat(parts, ignore_index=True).sort_values(["fold", "subject_id"]).reset_index(drop=True)


def pair(a, b):
    if not (a[["fold", "subject_id"]].equals(b[["fold", "subject_id"]]) and (a.label.values == b.label.values).all()):
        raise ValueError("Unpaired subjects or labels")
    return a.label.to_numpy(), a.prediction.to_numpy(), b.prediction.to_numpy(), a.fold.to_numpy()


def oracle_gain(y, pa, pb, fold):
    d = pd.DataFrame({"fold": fold, "label": y, "a": pa, "b": pb, "o": np.where(pa == y, pa, pb)})
    ma, mb, mo = (aggregate(d, c)["macro_f1"] for c in ("a", "b", "o"))
    return mo - max(ma, mb), float((pa != pb).mean())


def run(args):
    out = args.output; out.mkdir(parents=True, exist_ok=False)
    t0 = time.time(); evidence = []

    def record(path, role):
        evidence.append({"role": role, "path": str(path), "sha256": sha(path)})

    cells = get_cells(args, record, MODS)
    ov, orc = [], []
    for ds in CORPORA:
        fr = {(s, m): frames(cells, ds, s, m) for s in SEEDS for m in MODS}
        for s in SEEDS:
            y, pt, pa, fold = pair(fr[(s, "text_only")], fr[(s, "audio_text")])
            t, a = pt == y, pa == y
            only_t, only_a = int((t & ~a).sum()), int((~t & a).sum())
            union = int((~t | ~a).sum()); both = int((~t & ~a).sum())
            ov.append({"corpus": ds, "seed": s, "n": len(y), "both_wrong": both, "only_text_wrong": only_a,
                       "only_audio_text_wrong": only_t, "error_union": union,
                       "shared_error_pct_of_union": 100 * both / union if union else None,
                       "mcnemar_exact_p": float(binomtest(only_t, only_t + only_a).pvalue) if only_t + only_a else 1.})
            g, dr = oracle_gain(y, pt, pa, fold)
            orc.append({"corpus": ds, "comparison": "text_vs_audio_text", "seeds": str(s), "oracle_gain": g, "disagreement_rate": dr})
        for m in MODS:
            for s1, s2 in itertools.combinations(SEEDS, 2):
                y, p1, p2, fold = pair(fr[(s1, m)], fr[(s2, m)])
                g, dr = oracle_gain(y, p1, p2, fold)
                orc.append({"corpus": ds, "comparison": f"{m}_seed_pair", "seeds": f"{s1}-{s2}", "oracle_gain": g, "disagreement_rate": dr})
    od = pd.DataFrame(ov); od["mcnemar_holm_p_within_seed"] = 1.
    for _, g in od.groupby("seed"):
        od.loc[g.index, "mcnemar_holm_p_within_seed"] = holm(g.mcnemar_exact_p.to_numpy())
    od.to_csv(out / "error_overlap.csv", index=False)
    pd.DataFrame(orc).to_csv(out / "oracle_seed_noise.csv", index=False)
    write(out / "overlap_provenance.json", evidence)
    write(out / "overlap_config.json", {
        "source_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                        cwd=Path(__file__).parent).stdout.strip(),
        "source_script_sha256": sha(__file__), "backend": args.backend, "seeds": SEEDS,
        "oracle_gain": "oracle fold-mean Macro-F1 minus the better of the two models' fold-mean Macro-F1",
        "runtime_seconds": round(time.time() - t0, 1)})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cell-provenance", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    add_backend_args(p)
    run(p.parse_args())


if __name__ == "__main__":
    main()
