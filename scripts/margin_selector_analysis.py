"""Direct LoRA LLM selector: text (Qwen3.8) vs audio+text (Qwen3-Omni) by likelihood margin.

No model inference. Confidence is each model's own candidate-label likelihood margin
(dep_score - non_score). Margins are put on a common scale by dividing by the median
absolute margin of the same model on the same fold's inner validation (label-free).
Thresholds are chosen per fold on that fold's saved inner-validation predictions of the
selected checkpoint only. Outer evaluation labels never choose anything.

Caveat: inner validation also chose each model's checkpoint, so it is not fully
independent of model fitting; results may be mildly optimistic.
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
import yaml
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from knn_reliability_analysis import CORPORA, aggregate, holm, read, scores, sha, write  # noqa: E402

SEEDS = (7, 1337, 2024)
MODS = ("text_only", "audio_text")
GRID = [round(x, 2) for x in np.arange(0., 3.01, .25)] + [float("inf")]
RULES = ("max_conf", "advisor_gate")  # margin_sum == max_conf identically (sign of sum = sign of larger |z|)


def margin(d):
    """Positive margin means the model prefers Depressed; must agree with its prediction."""
    m = d["score_margin"] if "score_margin" in d else d["dep_score"] - d["non_score"]
    m = pd.to_numeric(m, errors="coerce").to_numpy(float)
    if not np.isfinite(m).all():
        raise ValueError("Non-finite likelihood margin")
    pred = pd.to_numeric(d["prediction"], errors="coerce").to_numpy()
    if not np.array_equal((m > 0).astype(int), pred):
        raise ValueError("Margin sign disagrees with saved prediction")
    return m


def scale_of(m):
    s = float(np.median(np.abs(m)))
    if not s > 0:
        raise ValueError("Zero margin scale")
    return s


def rule_predict(rule, zt, za, tau=None):
    """Return 0/1 predictions. Ties go to text (the reference model)."""
    pt, pa = (zt > 0).astype(int), (za > 0).astype(int)
    if rule == "text":
        return pt
    if rule == "audio_text":
        return pa
    if rule == "max_conf":
        return np.where(np.abs(za) > np.abs(zt), pa, pt)
    if rule == "margin_sum":
        return ((zt + za) > 0).astype(int)
    if rule == "advisor_gate":
        # Both reliable -> combine; only one reliable -> that one; neither -> text.
        rt, ra = np.abs(zt) >= tau, np.abs(za) >= tau
        both = ((zt + za) > 0).astype(int)
        return np.where(rt & ra, both, np.where(ra & ~rt, pa, pt))
    raise ValueError(rule)


def select_tau(y, zt, za):
    """Inner-val macro-F1; ties prefer the larger tau (closer to text alone)."""
    cands = [{"tau": t, **scores(y, rule_predict("advisor_gate", zt, za, t))} for t in GRID]
    best = max(cands, key=lambda c: (c["macro_f1"], c["tau"]))
    return best["tau"], cands


def fast_mf1(y, p):
    out = 0.
    for c in (0, 1):
        tp = np.sum((y == c) & (p == c)); fp = np.sum((y != c) & (p == c)); fn = np.sum((y == c) & (p != c))
        d = 2 * tp + fp + fn
        out += 2 * tp / d if d else 0.
    return out / 2


def seed_fold_mean(df, col):
    """Mean over seeds of equal-weight fold-mean macro-F1."""
    return float(np.mean([aggregate(df[df.seed == s], col)["macro_f1"] for s in SEEDS]))


def uncertainty(df, cand, ref, reps, seed):
    """Class-stratified within-fold subject bootstrap, resampling the same subjects for
    all seeds jointly; paired prediction-swap permutation. Fixed models and thresholds."""
    rng = np.random.default_rng(seed)
    observed = seed_fold_mean(df, cand) - seed_fold_mean(df, ref)
    blocks = []
    for fold, g in df.groupby("fold"):
        w = {s: g[g.seed == s].set_index("subject_id").sort_index() for s in SEEDS}
        ids = w[SEEDS[0]].index
        for s in SEEDS:
            if not w[s].index.equals(ids) or not (w[s].label.values == w[SEEDS[0]].label.values).all():
                raise ValueError("Seeds do not share subjects/labels in fold %s" % fold)
        y = w[SEEDS[0]].label.to_numpy()
        a = np.stack([w[s][cand].to_numpy() for s in SEEDS]); b = np.stack([w[s][ref].to_numpy() for s in SEEDS])
        blocks.append((y, a, b))
    boot, perm = np.empty(reps), np.empty(reps)
    for r in range(reps):
        bd, pd_ = [], []
        for y, a, b in blocks:
            idx = np.concatenate([rng.choice(np.flatnonzero(y == c), int((y == c).sum()), replace=True)
                                  for c in (0, 1) if (y == c).any()])
            bd.append(np.mean([fast_mf1(y[idx], a[k, idx]) - fast_mf1(y[idx], b[k, idx]) for k in range(len(SEEDS))]))
            diffs = []
            for k in range(len(SEEDS)):
                sw = rng.random(len(y)) < .5
                diffs.append(fast_mf1(y, np.where(sw, b[k], a[k])) - fast_mf1(y, np.where(sw, a[k], b[k])))
            pd_.append(np.mean(diffs))
        boot[r], perm[r] = np.mean(bd), np.mean(pd_)
    return {"delta_macro_f1": observed, "ci95_low": float(np.quantile(boot, .025)),
            "ci95_high": float(np.quantile(boot, .975)),
            "permutation_p": float((1 + np.sum(np.abs(perm) >= abs(observed) - 1e-12)) / (reps + 1))}


def inner_val_path(fd, epoch):
    cands = [fd / f"logs/{d}_epoch_{epoch}/predictions_subject_level.csv"
             for d in ("val", "inner_val", "fold_validation")] + [fd / "eval/best_validation/predictions_subject_level.csv"]
    found = [p for p in cands if p.exists()]
    if not found:
        raise FileNotFoundError("No inner-validation predictions for selected epoch in " + str(fd))
    return found[0]


def load_cells(prov_path, record, mods=MODS):
    cells = read(prov_path)
    out = {}
    for c in cells:
        cid = c["cell_id"]
        if not cid.startswith("S-qwen-") or not cid.endswith("-N"):
            continue
        ds, mod, seed, fold, _ = cid[len("S-qwen-"):].rsplit("-", 4)
        if ds not in CORPORA or mod not in mods:
            continue
        pv = c["provenance"]
        p = pv.get("text_rerun") or pv.get("androids_rerun") or pv.get("fit") or pv.get("row", {})
        fd = Path(p.get("fold_dir") or p.get("local_fold"))
        outer = Path(p.get("subject_file") or fd / "best_model/standalone_eval/predictions_subject_level.csv")
        if p.get("subject_sha256") and sha(outer) != p["subject_sha256"]:
            raise ValueError("Outer prediction hash changed: " + cid)
        cfg = yaml.safe_load((fd / "run_config.yaml").read_text())
        if cfg.get("evaluation", {}).get("sample_prediction_mode") != "likelihood":
            raise ValueError("Non-likelihood cell " + cid)
        sel = read(fd / "logs/selected_checkpoint_selection_metrics.json")
        if sel.get("selection_metric") != "inner_val_macro_f1":
            raise ValueError("Unexpected selection metric " + cid)
        ivp = inner_val_path(fd, sel["selected_epoch"])
        frames = {}
        for role, path in (("outer", outer), ("inner", ivp)):
            d = pd.read_csv(path, dtype={"subject_id": str})
            d["prediction"] = pd.to_numeric(d.prediction, errors="coerce").fillna(-1).astype(int)
            d["margin"] = margin(d)
            frames[role] = d[["subject_id", "label", "prediction", "margin"]]
            record(path, f"llm_{role}_predictions")
        if abs(scores(frames["outer"].label, frames["outer"].prediction)["macro_f1"] - c["macro_f1"]) > 1e-9:
            raise ValueError("Outer macro-F1 does not reproduce " + cid)
        if abs(scores(frames["inner"].label, frames["inner"].prediction)["macro_f1"] - sel["selection_metric_value"]) > 1e-9:
            raise ValueError("Inner-val macro-F1 does not reproduce recorded selection value " + cid)
        record(fd / "run_config.yaml", "llm_run_config")
        record(fd / "logs/selected_checkpoint_selection_metrics.json", "llm_selection")
        out[(ds, int(seed), int(fold), mod)] = {"cell_id": cid, "fold_dir": str(fd), "selected_epoch": sel["selected_epoch"],
                                                "inner_file": str(ivp), **frames}
    return out


def join(a, b, what):
    keys = ["subject_id"]
    if a.subject_id.duplicated().any() or b.subject_id.duplicated().any():
        raise ValueError("Duplicate subjects in " + what)
    if set(a.subject_id) != set(b.subject_id):
        raise ValueError("Text and audio+text subject sets differ in " + what)
    d = a.merge(b, on=keys, suffixes=("_t", "_at"), validate="one_to_one")
    if not (d.label_t == d.label_at).all():
        raise ValueError("Labels differ in " + what)
    return d.rename(columns={"label_t": "label"}).drop(columns="label_at")


def run(args):
    out = args.output; out.mkdir(parents=True, exist_ok=False)
    t0 = time.time(); evidence = []

    def record(path, role):
        evidence.append({"role": role, "path": str(path), "sha256": sha(path)})

    cells = load_cells(args.cell_provenance, record)
    rows, thresholds, diag = [], [], []
    for ds in CORPORA:
        for seed in SEEDS:
            folds = sorted({k[2] for k in cells if k[0] == ds and k[1] == seed})
            prep = {}
            for fold in folds:
                ct, ca = cells[(ds, seed, fold, "text_only")], cells[(ds, seed, fold, "audio_text")]
                where = f"{ds} seed {seed} fold {fold}"
                inner = join(ct["inner"], ca["inner"], "inner " + where)
                outer = join(ct["outer"], ca["outer"], "outer " + where)
                si, so = set(inner.subject_id), set(outer.subject_id)
                if not si & so:
                    mode = "inner_val"
                elif si == so:
                    # Checkpoint was selected on the outer validation fold itself (Turkish, CMDC):
                    # no independent inner set exists, so tau comes from the other folds only.
                    mode = "leave_one_fold_out"
                else:
                    raise ValueError("Inner and outer subjects partially overlap in " + where)
                st, sa = scale_of(inner.margin_t), scale_of(inner.margin_at)
                prep[fold] = (inner, outer, st, sa, mode)
            modes = {v[4] for v in prep.values()}
            if len(modes) != 1 or ("leave_one_fold_out" in modes and len(folds) < 2):
                raise ValueError(f"Inconsistent selection mode for {ds} seed {seed}: {modes}")
            for fold in folds:
                inner, outer, st, sa, mode = prep[fold]
                src = [fold] if mode == "inner_val" else [f for f in folds if f != fold]
                y = np.concatenate([prep[f][0].label.to_numpy() for f in src])
                izt = np.concatenate([prep[f][0].margin_t.to_numpy() / prep[f][2] for f in src])
                iza = np.concatenate([prep[f][0].margin_at.to_numpy() / prep[f][3] for f in src])
                tau, cands = select_tau(y, izt, iza)
                thresholds.append({"corpus": ds, "seed": seed, "fold": fold, "mode": mode, "selection_folds": src,
                                   "tau": tau, "n_selection": len(y),
                                   "text_scale": st, "audio_text_scale": sa, "candidates": cands})
                zt, za = outer.margin_t.to_numpy() / st, outer.margin_at.to_numpy() / sa
                o = outer[["subject_id", "label"]].copy(); o["corpus"], o["seed"], o["fold"] = ds, seed, fold
                for r in ("text", "audio_text", "max_conf", "margin_sum"):
                    o[r] = rule_predict(r, zt, za)
                if not (o.margin_sum == o.max_conf).all():
                    raise AssertionError("margin_sum and max_conf must be identical")
                o = o.drop(columns="margin_sum")
                o["advisor_gate"] = rule_predict("advisor_gate", zt, za, tau)
                if not (o.text == outer.prediction_t).all() or not (o.audio_text == outer.prediction_at).all():
                    raise ValueError("Reconstructed single-model predictions differ " + where)
                o["oracle"] = np.where(o.text == o.label, o.text, o.audio_text)
                o["abs_z_text"], o["abs_z_audio_text"] = np.abs(zt), np.abs(za)
                o["gate_source"] = np.select([(np.abs(zt) >= tau) & (np.abs(za) >= tau), (np.abs(za) >= tau)],
                                             ["combined", "audio_text"], "text")
                rows.append(o)
    df = pd.concat(rows, ignore_index=True)

    # Confidence diagnostic: does |z| predict that model's own correctness on outer eval?
    for ds, g in df.groupby("corpus", sort=False):
        for seed, h in g.groupby("seed"):
            for m in ("text", "audio_text"):
                correct = (h[m] == h.label).astype(int)
                auc = roc_auc_score(correct, h["abs_z_" + m]) if correct.nunique() == 2 else None
                diag.append({"corpus": ds, "seed": seed, "model": m, "n": len(h), "errors": int((1 - correct).sum()),
                             "correctness_auroc_abs_margin": auc})

    perf = []
    for ds, g in df.groupby("corpus", sort=False):
        for seed, h in g.groupby("seed"):
            for r in ("text", "audio_text") + RULES + ("oracle",):
                perf.append({"corpus": ds, "seed": seed, "method": r, **aggregate(h, r)})
            perf.append({"corpus": ds, "seed": seed, "method": "advisor_gate_share",
                         **{f"share_{k}": float((h.gate_source == k).mean()) for k in ("text", "audio_text", "combined")}})

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

    df.to_csv(out / "private_margin_selector_subjects.csv", index=False)
    pd.DataFrame(perf).to_csv(out / "margin_selector_performance.csv", index=False)
    td.to_csv(out / "margin_selector_tests.csv", index=False)
    pd.DataFrame(diag).to_csv(out / "margin_confidence_diagnostic.csv", index=False)
    write(out / "margin_threshold_selection.json", json.loads(json.dumps(thresholds, default=str).replace("Infinity", '"inf"')))
    write(out / "margin_provenance.json", evidence)
    cells_index = [{k: v for k, v in c.items() if k not in ("outer", "inner")} | {"key": list(map(str, key))}
                   for key, c in cells.items()]
    write(out / "margin_cells.json", cells_index)
    write(out / "margin_analysis_config.json", {
        "source_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                        cwd=Path(__file__).parent).stdout.strip(),
        "source_script_sha256": sha(__file__), "seeds": SEEDS, "tau_grid": [str(t) for t in GRID],
        "rules": {"max_conf": "larger |z| wins; tie -> text (identical to sign(z_text + z_audio_text))",
                  "advisor_gate": "both |z|>=tau -> margin_sum; only A+T -> A+T; else text; tau per fold on inner val, tie -> larger tau"},
        "scale": "z = margin / median |margin| of the same model on the same fold's inner validation (label-free)",
        "aggregation": "equal-weight fold-mean macro-F1 per seed, then mean over 3 seeds",
        "tests": "subject bootstrap (class-stratified within fold, joint over seeds) and prediction-swap permutation; "
                 "Holm within family (5 corpora x 2 rules)", "reps": args.reps, "rng_seed": args.seed,
        "limitations": ["inner validation also selected each model's checkpoint (mild optimism)",
                        "inner validation sets are small (10-35 subjects per fold)",
                        "bootstrap conditions on fixed models and thresholds",
                        "Turkish and CMDC are outer validation, not locked test; DAIC is one official test fold"],
        "runtime_seconds": round(time.time() - t0, 1)})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cell-provenance", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--reps", type=int, default=5000)
    p.add_argument("--seed", type=int, default=20261111)
    run(p.parse_args())


if __name__ == "__main__":
    main()
