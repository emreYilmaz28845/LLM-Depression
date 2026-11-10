"""Private-artifact error overlap, kNN reliability and frozen-feature gating.

No model inference is performed. Native-language conditions only. LLM likelihood,
frozen heads and LoRA heads are kept separate; seeds are never pooled. Gate tuning
uses saved inner validation or cross-fitted outer-training predictions only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baselines import frozen_standalone_heads as fsh

CORPORA = {"cmdc": "Mandarin (CMDC)", "androids_interview": "Italian (Androids)",
           "daic": "English (DAIC-WOZ)", "turkish": "Turkish", "d3tec": "Spanish (D3TEC)"}
MODS = ("text_only", "audio_only", "audio_text")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def scores(y, p):
    """Strict binary metrics; invalid predictions remain errors for recall/F1."""
    y, p = np.asarray(y), np.asarray(p)
    f, r = [], []
    for c in (0, 1):
        tp = np.sum((y == c) & (p == c))
        fp = np.sum((y != c) & (p == c))
        fn = np.sum((y == c) & (p != c))
        f.append(float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.)
        r.append(float(tp / (tp + fn)) if tp + fn else 0.)
    return {"macro_f1": sum(f) / 2, "positive_f1": f[1], "uar": sum(r) / 2,
            "accuracy": float(np.mean(y == p))}


def aggregate(df, col):
    """Primary metric is equal-weight fold mean, not pooled F1."""
    ms = [scores(g.label, g[col]) for _, g in df.groupby("fold")]
    return {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}


def paired(a, b):
    """Require exactly the same keys and labels rather than silently intersect."""
    keys = ["fold", "subject_id"]
    if a.duplicated(keys).any() or b.duplicated(keys).any():
        raise ValueError("Duplicate fold/subject keys")
    if set(map(tuple, a[keys].values)) != set(map(tuple, b[keys].values)):
        raise ValueError("Unpaired evaluation subject sets")
    out = a.merge(b, on=keys, suffixes=("_t", "_at"), validate="one_to_one")
    if not (out.label_t == out.label_at).all():
        raise ValueError("Paired labels differ")
    return out


def overlap(a, b):
    d = paired(a, b)
    t, at = d.prediction_t == d.label_t, d.prediction_at == d.label_at
    both = int((~t & ~at).sum()); union = int((~t | ~at).sum())
    only_t = int((t & ~at).sum()); only_at = int((~t & at).sum())
    return {"n": len(d), "both_correct": int((t & at).sum()), "only_text_correct": only_t,
            "only_audio_text_correct": only_at, "both_wrong": both,
            "error_union": union, "shared_error_pct_of_union": 100 * both / union if union else None,
            "text_errors_corrected_pct": 100 * only_at / int((~t).sum()) if (~t).sum() else None,
            "audio_text_errors_corrected_pct": 100 * only_t / int((~at).sum()) if (~at).sum() else None,
            "mcnemar_exact_p": float(binomtest(only_t, only_t + only_at).pvalue) if only_t + only_at else 1.}


def knn_share(xtrain, ytrain, xquery, k=10):
    """Train-subject-mean z-norm then cosine, stable tie order, K fixed a priori."""
    if len(xtrain) < k:
        raise ValueError("Fewer training neighbours than fixed K")
    mu, sd = xtrain.mean(0), xtrain.std(0) + 1e-6
    t, q = (xtrain - mu) / sd, (xquery - mu) / sd
    tn, qn = np.linalg.norm(t, axis=1), np.linalg.norm(q, axis=1)
    if np.any(tn == 0) or np.any(qn == 0):
        raise ValueError("Zero cosine norm")
    t, q = t / tn[:, None], q / qn[:, None]
    nn = np.argsort(-(q @ t.T), axis=1, kind="stable")[:, :k]
    return ytrain[nn].mean(1)


def flat_means(cache, subjects):
    # Preserve the initial kNN diagnostic's flat window mean; heads retain their
    # recorded hierarchical unit/window aggregation. Both choices are explicit.
    return np.stack([cache.x[[i for unit in cache.units[s].values() for i in unit]].mean(0)
                     for s in subjects])


def confidence(share, pred):
    return np.where(pred == 1, share, np.where(pred == 0, 1 - share, 0.))


def gate(ptext, paudio, ctext, caudio, threshold):
    """Neither reliable -> text; only one reliable -> that one; both -> mean."""
    rt, ra = ctext >= threshold, caudio >= threshold
    return np.where(~rt & ra, paudio, np.where(rt & ra, (ptext + paudio) / 2, ptext))


def select_threshold(y, pt, pa, ct, ca):
    # Includes never-combine (1.1); ties prefer the conservative higher threshold.
    grid = np.round(np.arange(0., 1.11, .1), 1)
    candidates = [{"threshold": float(t), **scores(y, gate(pt, pa, ct, ca, t) > .5)} for t in grid]
    best = max(candidates, key=lambda c: (c["macro_f1"], c["threshold"]))
    return best["threshold"], candidates


def holm(ps):
    order = np.argsort(ps); out = np.empty(len(ps)); maximum = 0.
    for rank, i in enumerate(order):
        maximum = max(maximum, min(1., (len(ps) - rank) * ps[i]))
        out[i] = maximum
    return out.tolist()


def paired_uncertainty(df, candidate, reference, reps, seed):
    """Paired, class-stratified within-fold bootstrap and paired permutation.

    Fixed fitted predictions: intervals do not include training/threshold-selection
    uncertainty. Primary statistic is the fold-mean macro-F1 difference.
    """
    rng = np.random.default_rng(seed)
    groups = [g.reset_index(drop=True) for _, g in df.groupby("fold")]
    observed = aggregate(df, candidate)["macro_f1"] - aggregate(df, reference)["macro_f1"]
    boot, perm = [], []
    for _ in range(reps):
        bd, pdiff = [], []
        for g in groups:
            y, a, b = g.label.to_numpy(), g[candidate].to_numpy(), g[reference].to_numpy()
            idx = np.concatenate([rng.choice(np.flatnonzero(y == c), int((y == c).sum()), replace=True)
                                  for c in (0, 1) if (y == c).any()])
            bd.append(scores(y[idx], a[idx])["macro_f1"] - scores(y[idx], b[idx])["macro_f1"])
            swap = rng.random(len(y)) < .5
            pdiff.append(scores(y, np.where(swap, b, a))["macro_f1"] -
                         scores(y, np.where(swap, a, b))["macro_f1"])
        boot.append(np.mean(bd)); perm.append(np.mean(pdiff))
    return {"delta_macro_f1": observed, "ci95_low": float(np.quantile(boot, .025)),
            "ci95_high": float(np.quantile(boot, .975)),
            "permutation_p": float((1 + np.sum(np.abs(perm) >= abs(observed) - 1e-12)) / (reps + 1))}


class Analysis:
    def __init__(self, args):
        self.args = args; self.base = args.source / "output_model/frozen_base_standalone_heads"
        self.out = args.output; self.out.mkdir(parents=True, exist_ok=False)
        self.evidence = []; self.overlaps = []; self.reliability = []; self.gating = []; self.tests = []
        self.selections = []; self.private_rows = []; self.training = []; self.bins = []

    def record(self, path, role):
        path = Path(path)
        item = {"role": role, "path": str(path), "exists": path.is_file()}
        if path.is_file():
            item["sha256"] = sha(path)
        self.evidence.append(item)

    def source_path(self, p):
        p = Path(p)
        return p if p.is_absolute() else self.args.source / p

    def cache_path(self, identity):
        return self.source_path(identity["outer_train.npz"]["path"]).parent

    def route(self, family, ds, mod):
        root = "heads" if family == "frozen_head" else ("heads_lora_textrerun" if mod == "text_only" else "heads_lora")
        path = self.base / root / ds / (mod + "_native")
        self.record(path / "result_config.json", "head_config")
        return path, read(path / "result_config.json")

    def saved(self, family, ds, mod):
        route, cfg = self.route(family, ds, mod)
        frames = []
        for fold in range(len(cfg["split_files"])):
            p = route / f"fold_{fold}/predictions_subject_level.csv"
            self.record(p, "heldout_predictions")
            d = pd.read_csv(p, dtype={"subject_id": str}); d["fold"] = fold
            if not d.prediction.isin([0, 1]).all():
                raise ValueError("Unexpected invalid cached-head prediction")
            frames.append(d)
        return pd.concat(frames, ignore_index=True)

    def reliability_row(self, ds, family, mod, fold, d, share):
        d = d.copy(); d["knn_confidence"] = confidence(share, d.prediction.to_numpy())
        d["head_confidence"] = confidence(d.probability.to_numpy(), d.prediction.to_numpy())
        d["correct"] = (d.prediction == d.label).astype(int)
        d["corpus"] = ds; d["family"] = family; d["modality"] = mod; d["fold"] = fold
        self.private_rows.append(d)
        return d

    def diagnostic(self, d):
        for (ds, family, mod), g in d.groupby(["corpus", "family", "modality"]):
            ok = g.correct.to_numpy(); c = g.knn_confidence.to_numpy()
            auc = lambda x: float(roc_auc_score(ok, x)) if len(set(ok)) == 2 else None
            self.reliability.append({"corpus": ds, "family": family, "modality": mod, "n": len(g),
                "correctness_auroc_knn_diagnostic": auc(c),
                "correctness_auroc_head_diagnostic": auc(g.head_confidence),
                "mean_knn_confidence": float(c.mean()), "accuracy": float(ok.mean()),
                "brier_if_treated_as_probability": float(np.mean((c - ok)**2))})
            for lo, hi in [(0., .5), (.5, .7), (.7, .9), (.9, 1.01)]:
                mask = (c >= lo) & (c < hi)
                if mask.any():
                    self.bins.append({"corpus": ds, "family": family, "modality": mod,
                        "confidence_lo": lo, "confidence_hi_exclusive": hi, "n": int(mask.sum()),
                        "mean_confidence": float(c[mask].mean()), "observed_accuracy": float(ok[mask].mean())})

    def audit_metadata(self, path, ds, family, mod, fold):
        self.record(path, "extraction_metadata")
        meta = read(path)
        # Extraction metadata can inherit an adapter parent even for frozen vectors.
        # Always preserve actual weights_source and base identity explicitly.
        entry = {k: meta.get(k) for k in ["weights_source", "base_model", "base_model_revision", "model_backend",
                 "adapter_sha256", "parent_attempt_id", "saved_run_config_sha256", "saved_split_sha256",
                 "manifest_sha256", "evaluation_provenance", "evaluation_view", "hidden_layer", "pooling"]}
        entry.update(corpus=ds, family=family, modality=mod, fold=fold, metadata=str(path))
        self.evidence.append(entry)
        remote = str(meta.get("saved_run_config", ""))
        if remote:
            suffix = remote.split("/LLM-Depression/", 1)[-1]
            local = self.args.source / suffix
            self.record(local, "parent_run_config")
            if local.is_file():
                import yaml
                cfg = yaml.safe_load(local.read_text())
                self.evidence.append({"parent_config": str(local), "seed": cfg.get("seed"),
                                      "source_hash_matches": sha(local) == meta.get("saved_run_config_sha256")})

    def overlap_heads(self):
        for family in ("frozen_head", "lora_head_s1337"):
            for ds in CORPORA:
                a, b = self.saved(family, ds, "text_only"), self.saved(family, ds, "audio_text")
                self.overlaps.append({"corpus": ds, "family": family, "seed": 1337 if family.startswith("lora") else "frozen",
                    **overlap(a, b), **{"text_" + k: v for k, v in aggregate(a, "prediction").items()},
                    **{"audio_text_" + k: v for k, v in aggregate(b, "prediction").items()}})

    def overlap_llm(self):
        cells = read(self.args.cell_provenance)
        self.record(self.args.cell_provenance, "llm_cell_provenance")
        data = {}
        for c in cells:
            cid = c["cell_id"]
            if not cid.startswith("S-qwen-") or not cid.endswith("-N"):
                continue
            # Parse from suffix because dataset/modality names contain underscores.
            ds, mod, seed, fold, _ = cid[len("S-qwen-"):].rsplit("-", 4)
            if ds not in CORPORA or mod not in ("text_only", "audio_text"):
                continue
            pv = c["provenance"]
            p = pv.get("text_rerun") or pv.get("androids_rerun") or pv.get("fit") or pv.get("row", {})
            fd = p.get("fold_dir") or p.get("local_fold")
            if not fd:
                raise ValueError("LLM cell has no unambiguous fold directory: " + cid)
            fd = Path(fd)
            predpath = Path(p.get("subject_file") or fd / "best_model/standalone_eval/predictions_subject_level.csv")
            self.record(predpath, "llm_heldout_predictions")
            self.record(fd / "run_config.yaml", "llm_run_config")
            self.record(fd / "logs/split_used.json", "llm_split")
            d = pd.read_csv(predpath, dtype={"subject_id": str})
            if p.get("subject_sha256") and sha(predpath) != p["subject_sha256"]:
                raise ValueError("LLM prediction source hash changed: " + cid)
            d["prediction"] = pd.to_numeric(d.prediction, errors="coerce").fillna(-1).astype(int)
            d["fold"] = int(fold)
            if abs(scores(d.label, d.prediction)["macro_f1"] - c["macro_f1"]) > 1e-9:
                raise ValueError("LLM cell macro-F1 does not reproduce: " + cid)
            import yaml
            cfg = yaml.safe_load((fd / "run_config.yaml").read_text())
            backend = cfg.get("evaluation", {}).get("sample_prediction_mode")
            if backend != "likelihood":
                raise ValueError("Non-likelihood LLM cell: " + cid + " " + str(backend))
            data.setdefault((ds, int(seed), mod), []).append(d)
            obs = {"cell_id": cid, "run_config": str(fd / "run_config.yaml"), "fold": int(fold),
                   "backend": backend, "seed": int(seed)}
            for name in ("training_history.json", "selected_checkpoint_selection_metrics.json"):
                hp = fd / "logs" / name
                self.record(hp, "training_observations")
                if hp.exists():
                    j = read(hp)
                    # Preserve paths, not logs carrying potentially sensitive content.
                    obs[name] = str(hp)
                    if isinstance(j, dict):
                        for k in ("selected_epoch", "best_epoch", "total_epochs"):
                            if k in j:
                                obs[k] = j[k]
            self.training.append(obs)
            history_path = fd / "logs/training_history.json"
            selected_path = fd / "logs/selected_checkpoint_selection_metrics.json"
            if history_path.exists() and selected_path.exists():
                history = read(history_path); selected = read(selected_path)
                if isinstance(history, list) and history:
                    epoch = selected.get("selected_epoch")
                    obs["epochs_recorded"] = len(history)
                    obs["last_epoch_recorded"] = history[-1].get("epoch")
                    obs["train_loss_first"] = history[0].get("train_loss")
                    obs["train_loss_last"] = history[-1].get("train_loss")
                    obs["selected_epoch"] = epoch
                    obs["selection_metric"] = selected.get("selection_metric")
                    obs["selection_metric_value"] = selected.get("selection_metric_value")
                    vals = [r.get("selection_macro_f1") for r in history if r.get("selection_macro_f1") is not None]
                    later = [r["selection_macro_f1"] for r in history if r.get("epoch", 0) > (epoch or 0)
                             and r.get("selection_macro_f1") is not None]
                    obs["selection_macro_f1_last"] = vals[-1] if vals else None
                    obs["selection_macro_f1_min_after_selected"] = min(later) if later else None
            logs = sorted(fd.glob("*.out")) + sorted(fd.glob("*.err"))
            obs["slurm_logs"] = [str(p) for p in logs] or "not recorded in local fold directory"
            obs["slurm_anomaly_counts"] = {}
            for lp in logs:
                text = lp.read_text(errors="replace")
                for marker in ("out of memory", "CUDA error", "Traceback", "INVALID"):
                    obs["slurm_anomaly_counts"][marker] = obs["slurm_anomaly_counts"].get(marker, 0) + text.lower().count(marker.lower())
        for ds in CORPORA:
            for seed in (7, 1337, 2024):
                a = pd.concat(data[(ds, seed, "text_only")], ignore_index=True)
                b = pd.concat(data[(ds, seed, "audio_text")], ignore_index=True)
                self.overlaps.append({"corpus": ds, "family": "lora_llm_likelihood", "seed": seed,
                    **overlap(a, b), **{"text_" + k: v for k, v in aggregate(a, "prediction").items()},
                    **{"audio_text_" + k: v for k, v in aggregate(b, "prediction").items()}})

    def frozen(self, ds):
        reports, validations, trains, selections = {}, {}, {}, {}
        for mod in MODS:
            route, cfg = self.route("frozen_head", ds, mod)
            main = self.cache_path(cfg.get("cache_identity") or cfg["cache_identities"][0])
            di = cfg.get("dev_cache_identity")
            dev = self.source_path(di["final_eval.npz"]["path"]).parent if di else None
            cache, _ = fsh._load_caches(main, dev, False)
            self.audit_metadata(main / "extraction_metadata.json", ds, "frozen_head", mod, "all")
            report_parts, val_parts = [], []
            for fold, sf in enumerate(cfg["split_files"]):
                splitpath = self.source_path(sf["path"]); split = read(splitpath)
                self.record(splitpath, "head_split")
                if fsh.canonical_sha256(split) != sf["sha256"]:
                    raise ValueError("Saved split hash mismatch")
                train = sorted(map(str, split["train_subject_ids"]))
                report = sorted(map(str, split["selection_subject_ids"] if cfg["protocol"] == "train_val" else split["final_eval_subject_ids"]))
                if set(train) & set(report):
                    raise ValueError("Train/report leakage")
                selection_path = route / f"fold_{fold}/selection.json"
                self.record(selection_path, "head_selection")
                sel = read(selection_path); penalty = sel["selected_l2_penalty"]
                head = fsh._fit(cache, train, penalty)
                prob = expit(fsh._logits(cache, head, report))
                saved = pd.read_csv(route / f"fold_{fold}/predictions_subject_level.csv", dtype={"subject_id": str}).set_index("subject_id").loc[report]
                if np.max(np.abs(saved.probability.to_numpy() - prob)) > 1e-8:
                    raise ValueError("Frozen head failed saved-probability reproduction")
                xtrain = flat_means(cache, train)
                share = knn_share(xtrain, cache.y(train), flat_means(cache, report), self.args.k)
                rd = saved.reset_index(); rd["fold"] = fold
                rd = self.reliability_row(ds, "frozen_head", mod, fold, rd, share)
                report_parts.append(rd)
                if cfg["protocol"] in fsh.INNER_CV_PROTOCOLS:
                    # Lambda was selected on pooled inner OOF in the original head;
                    # refit each inner head, scaler and neighbour bank excluding queries.
                    inner = sel["inner_folds"]
                    if set(s for h in inner for s in h) != set(train):
                        raise ValueError("Invalid inner-CV partition")
                    for held in inner:
                        fit = [s for s in train if s not in set(held)]
                        ih = fsh._fit(cache, fit, penalty)
                        ip = expit(fsh._logits(cache, ih, held))
                        ishare = knn_share(flat_means(cache, fit), cache.y(fit), flat_means(cache, held), self.args.k)
                        val_parts.append(pd.DataFrame({"subject_id": held, "fold": fold, "label": cache.y(held),
                            "probability": ip, "confidence": confidence(ishare, (ip > .5).astype(int))}))
                else:
                    held = sorted(map(str, split["selection_subject_ids"]))
                    if (set(held) & set(train)) or (set(held) & set(report)):
                        raise ValueError("Validation leakage")
                    ip = expit(fsh._logits(cache, head, held))
                    ishare = knn_share(xtrain, cache.y(train), flat_means(cache, held), self.args.k)
                    val_parts.append(pd.DataFrame({"subject_id": held, "fold": fold, "label": cache.y(held),
                        "probability": ip, "confidence": confidence(ishare, (ip > .5).astype(int))}))
                print("frozen", ds, mod, fold, "verified", flush=True)
            reports[mod] = pd.concat(report_parts, ignore_index=True)
            validations[mod] = pd.concat(val_parts, ignore_index=True)
        ev = paired(reports["text_only"], reports["audio_only"])
        val = paired(validations["text_only"], validations["audio_only"])
        out = pd.DataFrame({"fold": ev.fold, "subject_id": ev.subject_id, "label": ev.label_t,
                            "text": ev.prediction_t, "audio": ev.prediction_at})
        out["mean_probability"] = ((ev.probability_t + ev.probability_at) / 2 > .5).astype(int)
        out["higher_knn"] = np.where(ev.knn_confidence_t >= ev.knn_confidence_at, ev.prediction_t, ev.prediction_at)
        # Audio+text is an independent comparator, not an input to the advisor gate.
        at = reports["audio_text"].set_index(["fold", "subject_id"])
        out["audio_text"] = at.loc[list(zip(out.fold, out.subject_id)), "prediction"].to_numpy()
        out["threshold_gate"] = -1
        for fold, v in val.groupby("fold"):
            threshold, candidates = select_threshold(v.label_t.to_numpy(), v.probability_t.to_numpy(),
                v.probability_at.to_numpy(), v.confidence_t.to_numpy(), v.confidence_at.to_numpy())
            self.selections.append({"corpus": ds, "fold": int(fold), "threshold": threshold,
                "selection_scope": "saved_inner_validation_or_outer_train_crossfit_only", "candidates": candidates})
            mask = ev.fold == fold
            out.loc[mask, "threshold_gate"] = (gate(ev.loc[mask, "probability_t"], ev.loc[mask, "probability_at"],
                ev.loc[mask, "knn_confidence_t"], ev.loc[mask, "knn_confidence_at"], threshold) > .5).astype(int)
        for method in ["text", "audio", "audio_text", "mean_probability", "higher_knn", "threshold_gate"]:
            self.gating.append({"corpus": ds, "method": method, "n": len(out), **aggregate(out, method)})
        # Prespecified family: tuned gate vs each of three single inputs, 5 corpora.
        for ref in ("text", "audio", "audio_text"):
            self.tests.append({"corpus": ds, "candidate": "threshold_gate", "reference": ref,
                **paired_uncertainty(out, "threshold_gate", ref, self.args.reps, self.args.seed)})
        out.to_csv(self.out / f"private_frozen_gate_{ds}.csv", index=False)

    def lora_diagnostics(self):
        # Held-out queries only; no gate tuning on adapter-training subjects.
        for ds in CORPORA:
            route, cfg = self.route("lora_head_s1337", ds, "text_only")
            for fold, identity in enumerate(cfg["cache_identities"]):
                cd = self.cache_path(identity)
                self.audit_metadata(cd / "extraction_metadata.json", ds, "lora_head_s1337", "text_only", fold)
                # No extra dev is needed for held-out confidence.
                cache = fsh.PooledCache([(cd, "outer_train"), (cd, "final_eval")])
                trainrows = [json.loads(x) for x in (cd / "outer_train_rows.jsonl").read_text().splitlines()]
                evalrows = [json.loads(x) for x in (cd / "final_eval_rows.jsonl").read_text().splitlines()]
                train, report = sorted({str(r["subject_id"]) for r in trainrows}), sorted({str(r["subject_id"]) for r in evalrows})
                if set(train) & set(report):
                    raise ValueError("LoRA train/report overlap")
                saved = pd.read_csv(route / f"fold_{fold}/predictions_subject_level.csv", dtype={"subject_id": str})
                if set(saved.subject_id) != set(report):
                    raise ValueError("LoRA evaluation subjects mismatch")
                saved = saved.set_index("subject_id").loc[report].reset_index()
                share = knn_share(flat_means(cache, train), cache.y(train), flat_means(cache, report), self.args.k)
                self.reliability_row(ds, "lora_head_s1337", "text_only", fold, saved, share)
                print("LoRA diagnostic", ds, fold, flush=True)

    def run(self):
        start = time.monotonic()
        self.overlap_heads(); self.overlap_llm()
        od = pd.DataFrame(self.overlaps)
        od["mcnemar_holm_p_within_family_seed"] = 1.
        for _, g in od.groupby(["family", "seed"]):
            od.loc[g.index, "mcnemar_holm_p_within_family_seed"] = holm(g.mcnemar_exact_p.to_numpy())
        self.overlaps = od.to_dict("records")
        pd.DataFrame(self.overlaps).to_csv(self.out / "error_overlap.csv", index=False)
        print("All overlap tables complete", flush=True)
        for ds in CORPORA:
            self.frozen(ds)
            pd.DataFrame(self.gating).to_csv(self.out / "gating.csv", index=False)
        self.lora_diagnostics()
        private = pd.concat(self.private_rows, ignore_index=True)
        private.to_csv(self.out / "private_reliability_subjects.csv", index=False)
        self.diagnostic(private)
        ps = [t["permutation_p"] for t in self.tests]
        for t, p in zip(self.tests, holm(ps)):
            t["holm_p_15_tests"] = p
        pd.DataFrame(self.tests).to_csv(self.out / "paired_tests.csv", index=False)
        pd.DataFrame(self.reliability).to_csv(self.out / "reliability.csv", index=False)
        pd.DataFrame(self.bins).to_csv(self.out / "reliability_bins.csv", index=False)
        write(self.out / "threshold_selection.json", self.selections)
        write(self.out / "training_observation_sources.json", self.training)
        write(self.out / "provenance.json", self.evidence)
        write(self.out / "analysis_config.json", {"schema": "knn_reliability.v1", "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(), "source_root": str(self.args.source),
            "source_script_sha256": sha(__file__), "k": self.args.k, "seed": self.args.seed,
            "bootstrap_and_permutation_reps": self.args.reps, "elapsed_seconds": time.monotonic() - start,
            "aggregation": "equal-weight fold mean metrics; pooled unique-subject overlap counts per seed",
            "gate": "text unreliable and audio reliable: audio; both reliable: mean; otherwise text",
            "knn": "flat window subject mean; train-only z-normalization and cosine",
            "head": "recorded hierarchical unit mean and equal-subject/window scaler; saved L2",
            "limitations": ["LoRA diagnostics only; no adapter-training OOF used for gate selection",
                "Turkish/CMDC outer validation, not independent locked test",
                "Bootstrap conditions on fixed models/thresholds; overlapping CV training sets not resampled",
                "Language is confounded with dataset; one corpus per language",
                "Diagnostic AUROC concerns correctness, not depression classification headline"]})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--cell-provenance", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--reps", type=int, default=5000)
    p.add_argument("--seed", type=int, default=20261110)
    Analysis(p.parse_args()).run()


if __name__ == "__main__":
    main()
