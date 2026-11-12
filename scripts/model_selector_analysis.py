"""Which model to trust when direct text and audio+text LLMs disagree?

No model inference. Per subject, choose between direct LoRA Qwen3.8 text and Qwen3-Omni
audio+text likelihood predictions with rules fixed before looking at outer results:

- ola: dynamic classifier selection, overall local accuracy. Find the K nearest
  selection subjects in a frozen Qwen3-Omni audio+text embedding space; trust the model
  that was right more often on them. Tie -> text.
- lca: local class accuracy. For each model, look only at the K nearest selection subjects
  whose true label equals that model's prediction; trust the model right more often there.
  Tie -> text.
- vote3: majority vote of text, audio+text and a third direct LLM (Qwen3-Omni audio-only,
  same seed and fold).
- stack: logistic regression (C=1, standardized) on the three scaled likelihood margins,
  fitted on the selection subjects only.

Selection subjects ("DSEL") are the same as in margin_selector_analysis.py: the fold's own
inner validation (DAIC, Androids, D3TEC), or the other folds' predictions
(leave-one-fold-out) for Turkish and CMDC, whose checkpoints were selected on the outer
validation fold. Outer evaluation labels never choose anything.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from knn_reliability_analysis import CORPORA, aggregate, flat_means, holm, read, sha, write  # noqa: E402
from margin_selector_analysis import SEEDS, join, load_cells, scale_of, seed_fold_mean, uncertainty  # noqa: E402
from baselines import frozen_standalone_heads as fsh  # noqa: E402

MODS = ("text_only", "audio_text", "audio_only")
RULES = ("ola", "lca", "vote3", "stack")
K = 7  # DESlib default region size, fixed a priori


def neighbours(xsel, xq, k, ref=None):
    """Indices of the k nearest xsel rows per query. z-norm statistics come from `ref`
    (default xsel: the whole selection set, never a one-class pool); cosine similarity."""
    ref = xsel if ref is None else ref
    mu, sd = ref.mean(0), ref.std(0) + 1e-6
    a, b = (xsel - mu) / sd, (xq - mu) / sd
    na, nb = np.linalg.norm(a, axis=1, keepdims=True), np.linalg.norm(b, axis=1, keepdims=True)
    if (na == 0).any() or (nb == 0).any():
        raise ValueError("Zero norm after normalization")
    a, b = a / na, b / nb
    return np.argsort(-(b @ a.T), axis=1, kind="stable")[:, :min(k, len(xsel))]


def ola(xsel, ysel, pt_sel, pa_sel, xq, pt, pa, k=K):
    nn = neighbours(xsel, xq, k)
    ct, ca = (pt_sel == ysel)[nn].mean(1), (pa_sel == ysel)[nn].mean(1)
    return np.where(ca > ct, pa, pt), ct, ca


def lca(xsel, ysel, pt_sel, pa_sel, xq, pt, pa, k=K):
    comp = {}
    for name, psel, pq in (("t", pt_sel, pt), ("a", pa_sel, pa)):
        c = np.zeros(len(xq))
        for cls in (0, 1):
            q = pq == cls; pool = np.flatnonzero(ysel == cls)
            if not q.any():
                continue
            if not len(pool):
                c[q] = 0.; continue
            nn = pool[neighbours(xsel[pool], xq[q], k, ref=xsel)]
            c[q] = (psel[nn] == cls).mean(1)
        comp[name] = c
    return np.where(comp["a"] > comp["t"], pa, pt), comp["t"], comp["a"]


def vote3(pt, pa, po):
    return ((pt + pa + po) >= 2).astype(int)


def stack(zsel, ysel, zq):
    if len(set(ysel)) < 2:
        raise ValueError("Stacking selection set has one class")
    mu, sd = zsel.mean(0), zsel.std(0) + 1e-9
    m = LogisticRegression(C=1.0).fit((zsel - mu) / sd, ysel)
    return m.predict((zq - mu) / sd).astype(int), m.coef_[0].tolist()


def embedding_bank(source, ds, record):
    route = source / "output_model/frozen_base_standalone_heads/heads" / ds / "audio_text_native"
    cfg = read(route / "result_config.json"); record(route / "result_config.json", "embedding_head_config")
    ci = cfg.get("cache_identity") or cfg["cache_identities"][0]
    p = Path(ci["outer_train.npz"]["path"]); main = (p if p.is_absolute() else source / p).parent
    di = cfg.get("dev_cache_identity")
    dev = None
    if di:
        q = Path(di["final_eval.npz"]["path"]); dev = (q if q.is_absolute() else source / q).parent
    cache, _ = fsh._load_caches(main, dev, False)
    record(main / "extraction_metadata.json", "embedding_extraction_metadata")
    meta = read(main / "extraction_metadata.json")
    if meta.get("weights_source") != "base_model_no_adapter" or meta.get("adapter_sha256"):
        raise ValueError("Embedding bank is not frozen base: " + ds)
    subjects = sorted(cache.units)
    x = flat_means(cache, subjects)
    return dict(zip(subjects, x)), {"cache": str(main), "dev_cache": str(dev) if dev else None,
                                    "weights_source": meta.get("weights_source"), "dim": int(x.shape[1])}


def stacked(frames):
    return pd.concat(frames, ignore_index=True)


def run(args):
    out = args.output; out.mkdir(parents=True, exist_ok=False)
    t0 = time.time(); evidence = []

    def record(path, role):
        evidence.append({"role": role, "path": str(path), "sha256": sha(path)})

    cells = load_cells(args.cell_provenance, record, mods=MODS)
    rows, regions, coefs, banks = [], [], [], {}
    for ds in CORPORA:
        bank, info = embedding_bank(args.source, ds, record); banks[ds] = info
        X = lambda ids: np.stack([bank[s] for s in ids])  # noqa: E731
        for seed in SEEDS:
            folds = sorted({k[2] for k in cells if k[0] == ds and k[1] == seed})
            prep = {}
            for fold in folds:
                c = {m: cells[(ds, seed, fold, m)] for m in MODS}
                where = f"{ds} seed {seed} fold {fold}"
                fr = {}
                for role in ("inner", "outer"):
                    d = join(c["text_only"][role], c["audio_text"][role], role + " " + where)
                    o = c["audio_only"][role].rename(columns={"prediction": "prediction_ao", "margin": "margin_ao",
                                                              "label": "label_ao"})
                    if set(o.subject_id) != set(d.subject_id):
                        raise ValueError("Audio-only subject set differs in " + role + " " + where)
                    d = d.merge(o, on="subject_id", validate="one_to_one")
                    if not (d.label == d.label_ao).all():
                        raise ValueError("Audio-only labels differ in " + where)
                    missing = set(d.subject_id) - set(bank)
                    if missing:
                        raise ValueError(f"{len(missing)} subjects lack embeddings in {where}")
                    fr[role] = d.drop(columns="label_ao").sort_values("subject_id").reset_index(drop=True)
                si, so = set(fr["inner"].subject_id), set(fr["outer"].subject_id)
                mode = "inner_val" if not si & so else ("leave_one_fold_out" if si == so else None)
                if mode is None:
                    raise ValueError("Inner and outer subjects partially overlap in " + where)
                sc = [scale_of(fr["inner"][col]) for col in ("margin_t", "margin_at", "margin_ao")]
                for role in ("inner", "outer"):
                    fr[role]["z"] = list(np.stack([fr[role][col] / s for col, s in
                                                   zip(("margin_t", "margin_at", "margin_ao"), sc)], 1))
                prep[fold] = (fr, mode)
            if len({m for _, m in prep.values()}) != 1:
                raise ValueError(f"Inconsistent selection mode for {ds} seed {seed}")
            for fold in folds:
                fr, mode = prep[fold]
                src = [fold] if mode == "inner_val" else [f for f in folds if f != fold]
                sel = stacked([prep[f][0]["inner"] for f in src])
                if mode == "leave_one_fold_out" and set(sel.subject_id) & set(fr["outer"].subject_id):
                    raise ValueError("Selection subjects overlap the evaluated fold in " + ds)
                o = fr["outer"]
                ysel = sel.label.to_numpy(); xs, xq = X(sel.subject_id), X(o.subject_id)
                pts, pas = sel.prediction_t.to_numpy(), sel.prediction_at.to_numpy()
                pt, pa, po = o.prediction_t.to_numpy(), o.prediction_at.to_numpy(), o.prediction_ao.to_numpy()
                r = o[["subject_id", "label"]].copy(); r["corpus"], r["seed"], r["fold"] = ds, seed, fold
                r["text"], r["audio_text"], r["audio_only"] = pt, pa, po
                r["ola"], ct_o, ca_o = ola(xs, ysel, pts, pas, xq, pt, pa)
                r["lca"], ct_l, ca_l = lca(xs, ysel, pts, pas, xq, pt, pa)
                r["vote3"] = vote3(pt, pa, po)
                r["stack"], coef = stack(np.stack(sel.z), ysel, np.stack(o.z))
                r["oracle"] = np.where(pt == r.label, pt, pa)
                r["disagree"] = (pt != pa).astype(int)
                r["ola_comp_text"], r["ola_comp_audio_text"] = ct_o, ca_o
                rows.append(r)
                regions.append({"corpus": ds, "seed": seed, "fold": fold, "mode": mode, "selection_folds": src,
                                "n_selection": len(sel), "k": min(K, len(sel)), "n_outer": len(o),
                                "selection_disagreements": int((pts != pas).sum())})
                coefs.append({"corpus": ds, "seed": seed, "fold": fold, "coef_text": coef[0],
                              "coef_audio_text": coef[1], "coef_audio_only": coef[2]})
            print("done", ds, seed, flush=True)
    df = pd.concat(rows, ignore_index=True)

    methods = ("text", "audio_text", "audio_only") + RULES + ("oracle",)
    perf = [{"corpus": ds, "seed": s, "method": m, **aggregate(h, m)}
            for (ds, s), h in df.groupby(["corpus", "seed"], sort=False) for m in methods]
    # Disagreement view: on subjects where text and A+T differ, how often does each rule pick the right one?
    dis = []
    for (ds, s), h in df[df.disagree == 1].groupby(["corpus", "seed"], sort=False):
        for m in ("text", "audio_text") + RULES + ("oracle",):
            dis.append({"corpus": ds, "seed": s, "method": m, "n_disagreements": len(h),
                        "correct_picks": int((h[m] == h.label).sum())})
    tests = []
    for ds, g in df.groupby("corpus", sort=False):
        for r in RULES:
            for ref in ("text", "audio_text"):
                tests.append({"corpus": ds, "rule": r, "reference": ref,
                              "family": "primary_vs_text" if ref == "text" else "secondary_vs_audio_text",
                              **uncertainty(g, r, ref, args.reps, args.seed)})
    td = pd.DataFrame(tests); td["holm_p"] = 1.
    for _, g in td.groupby("family"):
        td.loc[g.index, "holm_p"] = holm(g.permutation_p.to_numpy())

    df.drop(columns=["ola_comp_text", "ola_comp_audio_text"]).to_csv(out / "private_selector_subjects.csv", index=False)
    pd.DataFrame(perf).to_csv(out / "selector_performance.csv", index=False)
    pd.DataFrame(dis).to_csv(out / "selector_disagreements.csv", index=False)
    td.to_csv(out / "selector_tests.csv", index=False)
    pd.DataFrame(coefs).to_csv(out / "stack_coefficients.csv", index=False)
    write(out / "selection_regions.json", regions)
    write(out / "selector_provenance.json", evidence)
    write(out / "embedding_banks.json", banks)
    write(out / "selector_cells.json", [{k: v for k, v in c.items() if k not in ("outer", "inner")} | {"key": list(map(str, key))}
                                        for key, c in cells.items()])
    write(out / "selector_config.json", {
        "source_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                        cwd=Path(__file__).parent).stdout.strip(),
        "source_script_sha256": sha(__file__), "seeds": SEEDS, "k": K, "reps": args.reps, "rng_seed": args.seed,
        "embedding": "frozen Qwen3-Omni audio+text native, flat window mean per subject; z-norm on selection rows; cosine",
        "rules": {"ola": "K nearest selection subjects; higher local accuracy wins; tie -> text",
                  "lca": "per model, K nearest selection subjects with true label = model's prediction; tie -> text",
                  "vote3": "majority of text, audio+text, audio-only direct LLMs",
                  "stack": "logistic regression C=1 on standardized scaled margins (text, A+T, audio-only), fit on selection subjects"},
        "selection": "fold inner validation (DAIC, Androids, D3TEC); leave-one-fold-out (Turkish, CMDC)",
        "aggregation": "equal-weight fold-mean macro-F1 per seed, then mean over 3 seeds",
        "tests": "class-stratified within-fold subject bootstrap joint over seeds; prediction-swap permutation; "
                 "Holm within family (5 corpora x 4 rules)",
        "limitations": ["selection sets are small (inner validation 10-35 subjects per fold), so K covers much of them",
                        "inner validation also selected each LLM checkpoint (mild optimism)",
                        "bootstrap conditions on fixed models and selection sets",
                        "Turkish and CMDC are outer validation, not locked test; DAIC is one official test fold",
                        "audio-only LLM is a third model, so vote3 and stack use more information than the two-model rules"],
        "runtime_seconds": round(time.time() - t0, 1)})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True, help="checkout holding output_model/ frozen caches")
    p.add_argument("--cell-provenance", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--reps", type=int, default=5000)
    p.add_argument("--seed", type=int, default=20261112)
    run(p.parse_args())


if __name__ == "__main__":
    main()
