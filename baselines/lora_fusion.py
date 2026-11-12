#!/usr/bin/env python3
"""Calibrated ensemble of direct LoRA LLM predictions and frozen-feature heads.

Inputs (no model inference):

- LoRA predictions: the per-subject candidate-label likelihoods of the
  selected checkpoints of Qwen3.8-27B text LoRA and Qwen3-Omni audio+text LoRA
  (native, seeds 7/1337/2024), listed in a provenance file (role
  ``llm_outer_predictions`` / ``llm_inner_predictions`` / ``llm_run_config``
  per fold directory; the margin-selector list). Every file is hashed. The
  margin is ``dep_score - non_score`` and must agree with the saved prediction.
  The outer folds equal the frozen CV folds (checked); DAIC is evaluated on the
  47 official test subjects (train on train, checkpoint on dev).
- Frozen heads: the calibrated out-of-fold members of
  ``baselines/fusion_ensemble.py`` (``--frozen-json``) for the CV corpora; for
  the DAIC test set the same members are refitted here on the 142 train+dev
  subjects and applied to the 47 test subjects.

Calibration (class-balanced Platt, one slope and intercept per member, seed and
fold): LoRA members are fitted on the selected checkpoint's inner-validation
predictions when those are disjoint from the outer fold (Androids, D3TEC,
DAIC); for Turkish and CMDC the checkpoint was chosen on the outer fold itself,
so the calibration is fitted on the other folds' outer predictions of the same
seed (cross-fit, no subject calibrates its own prediction).

Ensembles are means of calibrated logits, classified at 0:
``lora_both`` (text + audio+text LoRA) and ``lora_all`` (both LoRA members
plus the frozen text ensemble and the frozen audio ensemble, four members).

Pre-specified primary comparison, one per corpus, Holm over five:
``lora_all`` vs ``lora_text`` (the reported text LoRA model, its own
prediction). Metrics: equal-weight fold-mean macro-F1 per seed, then mean over
the three seeds. Test: class-stratified subject bootstrap of the seed-mean
pooled macro-F1 difference (subjects resampled jointly for all seeds).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.band_score_experiments import SCORED, _fit  # noqa: E402
from baselines.fusion_ensemble import members_for, platt, select_with_oof  # noqa: E402
from baselines.merge_diag_common import DATASETS, load_source, macro_f1, metrics  # noqa: E402
from baselines.merge_diagnostics import git_state, holm, load_scores  # noqa: E402

SEED = 20261114
SEEDS = (7, 1337, 2024)
MODS = {"text_only": "lora_text", "audio_text": "lora_at"}
BOOT = 2000
VARIANTS = ("lora_text", "lora_at", "frozen_text", "frozen_audio", "lora_both", "lora_all")


def sha(path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(2 ** 20), b""):
            h.update(block)
    return h.hexdigest()


def read_predictions(path: Path) -> dict:
    out = {}
    for r in csv.DictReader(path.open(encoding="utf-8")):
        if "score_margin" in r and r["score_margin"] != "":
            m = float(r["score_margin"])
        else:
            m = float(r["dep_score"]) - float(r["non_score"])
        pred = int(float(r["prediction"])) if r["prediction"] not in ("", None) else -1
        if not np.isfinite(m) or int(m > 0) != pred:
            raise ValueError(f"{path}: margin and prediction disagree for a subject")
        if r["subject_id"] in out:
            raise ValueError(f"{path}: duplicate subject")
        out[r["subject_id"]] = (int(r["label"]), m, pred)
    return out


def load_lora(prov_path: Path) -> tuple[dict, list]:
    """{(dataset, mod, seed, fold): {"outer": preds, "inner": preds, "fold_dir": str}} + evidence."""
    entries = json.loads(prov_path.read_text())
    by_dir = defaultdict(dict)
    for e in entries:
        fd = re.match(r"(.*/fold_\d+)/", e["path"]).group(1)
        by_dir[fd][e["role"]] = e
    cells, evidence = {}, []
    for fd, roles in sorted(by_dir.items()):
        for role in ("llm_outer_predictions", "llm_inner_predictions", "llm_run_config"):
            if sha(roles[role]["path"]) != roles[role]["sha256"]:
                raise ValueError(f"hash changed: {roles[role]['path']}")
            evidence.append({"role": role, "path": roles[role]["path"], "sha256": roles[role]["sha256"]})
        cfg = yaml.safe_load(Path(roles["llm_run_config"]["path"]).read_text())
        if cfg.get("evaluation", {}).get("sample_prediction_mode") != "likelihood":
            raise ValueError(f"non-likelihood run: {fd}")
        m = re.search(r"/(text_only|audio_text)/([a-z0-9_]+)/[^/]+/fold_(\d+)$", fd)
        mod, ds, fold = m.group(1), m.group(2), int(m.group(3))
        seed = int(cfg["config"]["seed"])
        if cfg["config"].get("dataset") not in (ds, None) and not ds.startswith(str(cfg["config"]["dataset"])):
            raise ValueError(f"dataset mismatch in {fd}: {cfg['config'].get('dataset')}")
        key = (ds, MODS[mod], seed, fold)
        if key in cells:
            raise ValueError(f"duplicate cell {key}")
        cells[key] = {"outer": read_predictions(Path(roles["llm_outer_predictions"]["path"])),
                      "inner": read_predictions(Path(roles["llm_inner_predictions"]["path"])), "fold_dir": fd}
    return cells, evidence


def frozen_daic_test(srcs: dict) -> dict:
    """Frozen members fitted on DAIC train+dev (CV subjects), calibrated by inner OOF, applied to the test set."""
    t = srcs["text_native"]
    g, y, fold, subjects = t["g"], t["y"], t["fold"], t["subjects"]
    scores = load_scores()
    s = np.asarray([scores.get(str(a), np.nan) if d in SCORED else np.nan for a, d in zip(subjects, g)])
    tr, te = (g == "daic") & (fold >= 0), (g == "daic") & (fold < 0)
    out = {}
    for mod, arm in members_for("daic"):
        x = srcs[f"{mod}_native"]["x"]
        pen, inner = select_with_oof(x[tr], g[tr], y[tr], s[tr], arm, seed=0)
        pred = _fit(x[tr], g[tr], y[tr], s[tr], "global", arm, pen)(x[te], g[te])
        a, b = platt(inner, y[tr])
        out[f"{mod}/{arm}"] = dict(zip([str(v).split("::", 1)[1] for v in subjects[te]], a * pred + b))
    return out


def run(lora_prov: Path, frozen_json: Path) -> dict:
    srcs = {s: load_source(s) for s in ("text_native", "audio_native")}
    t = srcs["text_native"]
    g, y, fold, subjects = t["g"], t["y"], t["fold"], t["subjects"]
    local = {(d, str(v).split("::", 1)[1]): i for i, (v, d) in enumerate(zip(subjects, g))}
    fz = json.loads(frozen_json.read_text())
    if fz["result"]["subjects"] != [str(v) for v in subjects]:
        raise ValueError("frozen JSON subject order differs")
    cal = {k: np.asarray(v) for k, v in fz["result"]["oof_cal"].items()}
    daic_test = frozen_daic_test(srcs)
    cells, evidence = load_lora(lora_prov)

    def frozen_mean(ds, idx, modality):
        names = [f"{m}/{a}" for m, a in members_for(ds) if m == modality]
        if ds == "daic":
            sid = [str(subjects[i]).split("::", 1)[1] for i in idx]
            return np.mean([[daic_test[n][s] for s in sid] for n in names], axis=0)
        return np.mean([cal[n][idx] for n in names], axis=0)

    rows = []   # one row per (dataset, seed, fold, subject)
    calib = []
    for ds in DATASETS:
        for seed in SEEDS:
            folds = sorted({k[3] for k in cells if k[0] == ds and k[2] == seed})
            for fold_k in folds:
                outer = {m: cells[(ds, m, seed, fold_k)]["outer"] for m in MODS.values()}
                sids = sorted(outer["lora_text"])
                if sorted(outer["lora_at"]) != sids:
                    raise ValueError(f"{ds} s{seed} f{fold_k}: text and A+T subjects differ")
                idx = np.asarray([local[(ds, s)] for s in sids])
                exp_fold = -1 if ds == "daic" else fold_k
                if not (fold[idx] == exp_fold).all() or not all(outer["lora_text"][s][0] == y[i] for s, i in zip(sids, idx)):
                    raise ValueError(f"{ds} s{seed} f{fold_k}: fold or labels differ from the frozen data")
                calibrated = {}
                for m in MODS.values():
                    inner = cells[(ds, m, seed, fold_k)]["inner"]
                    if set(inner) & set(sids):
                        if set(inner) != set(sids):
                            raise ValueError(f"{ds}: inner and outer partly overlap")
                        mode = "cross_fit_other_folds"
                        ref = [cells[(ds, m, seed, f)]["outer"] for f in folds if f != fold_k]
                        cy = np.asarray([v[0] for r in ref for v in r.values()])
                        cm = np.asarray([v[1] for r in ref for v in r.values()])
                    else:
                        mode = "inner_validation"
                        cy = np.asarray([v[0] for v in inner.values()])
                        cm = np.asarray([v[1] for v in inner.values()])
                    a, b = platt(cm, cy)
                    calib.append({"dataset": ds, "seed": seed, "fold": fold_k, "member": m, "mode": mode,
                                  "slope": a, "intercept": b, "n": int(len(cy))})
                    calibrated[m] = a * np.asarray([outer[m][s][1] for s in sids]) + b
                ft, fa = frozen_mean(ds, idx, "text"), frozen_mean(ds, idx, "audio")
                v = {"lora_text": np.asarray([outer["lora_text"][s][2] for s in sids]) * 2.0 - 1.0,
                     "lora_at": np.asarray([outer["lora_at"][s][2] for s in sids]) * 2.0 - 1.0,
                     "frozen_text": ft, "frozen_audio": fa,
                     "lora_both": (calibrated["lora_text"] + calibrated["lora_at"]) / 2.0,
                     "lora_all": (calibrated["lora_text"] + calibrated["lora_at"] + ft + fa) / 4.0}
                for j, s in enumerate(sids):
                    rows.append({"dataset": ds, "seed": seed, "fold": fold_k, "subject": s, "y": int(y[idx[j]]),
                                 **{n: float(v[n][j]) for n in VARIANTS}})

    table, primary, explore = {}, {}, {}
    rng = np.random.default_rng(SEED)
    for ds in DATASETS:
        r = [x for x in rows if x["dataset"] == ds]
        folds = sorted({x["fold"] for x in r})
        table[ds] = {}
        for n in VARIANTS:
            per_seed = []
            for seed in SEEDS:
                fm = [metrics(np.asarray([x["y"] for x in r if x["seed"] == seed and x["fold"] == k]),
                              np.asarray([x[n] for x in r if x["seed"] == seed and x["fold"] == k])) for k in folds]
                per_seed.append({key: float(np.mean([m[key] for m in fm])) for key in ("macro_f1", "positive_f1", "uar")})
            table[ds][n] = {key: float(np.mean([p[key] for p in per_seed])) for key in ("macro_f1", "positive_f1", "uar")}
            table[ds][n]["seed_sd_macro_f1"] = float(np.std([p["macro_f1"] for p in per_seed], ddof=1))
        # subject-level arrays per seed for the bootstrap (same subject order in every seed)
        subj = sorted({x["subject"] for x in r})
        lookup = {(x["seed"], x["subject"]): x for x in r}
        yy = np.asarray([lookup[(SEEDS[0], s)]["y"] for s in subj])
        arr = {n: np.stack([[lookup[(sd, s)][n] for s in subj] for sd in SEEDS]) for n in VARIANTS}

        def test(a, b):
            obs = float(np.mean([macro_f1(yy, arr[a][k]) - macro_f1(yy, arr[b][k]) for k in range(len(SEEDS))]))
            diffs = []
            for _ in range(BOOT):
                ix = np.concatenate([rng.choice(np.flatnonzero(yy == c), int((yy == c).sum()), replace=True)
                                     for c in (0, 1)])
                diffs.append(np.mean([macro_f1(yy[ix], arr[a][k][ix]) - macro_f1(yy[ix], arr[b][k][ix])
                                      for k in range(len(SEEDS))]))
            diffs = np.asarray(diffs)
            p = float(min(1.0, 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))))
            return {"diff_pooled": obs, "p": p, "ci95": [float(np.quantile(diffs, .025)),
                                                         float(np.quantile(diffs, .975))]}

        primary[f"{ds}|lora_all_vs_lora_text"] = test("lora_all", "lora_text")
        for a, b in (("lora_both", "lora_text"), ("lora_all", "lora_both"), ("lora_all", "lora_at"),
                     ("lora_text", "frozen_text")):
            explore[f"{ds}|{a}_vs_{b}"] = test(a, b)
    for fam in (primary, explore):
        adj = holm({k: v["p"] for k, v in fam.items()})
        for k in fam:
            fam[k]["p_holm"] = adj[k]
    mean_row = {n: float(np.mean([table[d][n]["macro_f1"] for d in DATASETS])) for n in VARIANTS}
    return {"table": table, "mean_macro_f1": mean_row, "primary_tests": primary, "exploratory_tests": explore,
            "calibration": calib, "evidence": evidence,
            "frozen_json": {"path": str(frozen_json), "sha256": sha(frozen_json), "git_head": fz["git_head"]},
            "rows": rows, "provenance": {s: srcs[s]["provenance"] for s in srcs}}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--lora-provenance", type=Path, required=True)
    p.add_argument("--frozen-json", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    path = args.output_dir / "lora_fusion.json"
    if path.exists():
        raise SystemExit(f"Refusing to overwrite {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    res = run(args.lora_provenance, args.frozen_json)
    payload = {**git_state(), "analysis": "lora_fusion", "seed": SEED, "seconds": round(time.time() - t0, 1),
               "lora_provenance": {"path": str(args.lora_provenance), "sha256": sha(args.lora_provenance)},
               "result": res}
    path.write_text(json.dumps(payload, indent=1))
    print(f"written {path} in {payload['seconds']}s", flush=True)
    for d, row in res["table"].items():
        print(d, " ".join(f"{n}:{row[n]['macro_f1']:.3f}" for n in VARIANTS))
    print("mean", " ".join(f"{n}:{v:.3f}" for n, v in res["mean_macro_f1"].items()))
    for k, v in res["primary_tests"].items():
        print(k, f"{v['diff_pooled']:+.3f} p={v['p']:.3f} holm={v['p_holm']:.3f}")


if __name__ == "__main__":
    main()
