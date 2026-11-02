#!/usr/bin/env python3
"""Linear probe: can frozen features tell which dataset (= language here) a subject comes from?

In this project every dataset has one language (D3TEC es, Turkish tr, Androids
it, DAIC en, CMDC zh) and one recording setup, so "dataset", "language" and
"channel" are the same label. The probe therefore measures how readable the
dataset is; the native vs English-transcript text sources and the acoustic
baseline are what separate language from the rest.

Sources (every one reduced to one vector per subject, mean over response units
of the mean window vector, like the subject-mean head):

- ``merged:<name>:<language>:<modality>``: frozen Qwen3-Omni / Qwen3.8 caches on
  the merged final-fold row index (train + holdout = every subject);
- ``w2v:<name>:<pooling>``: frozen wav2vec2 caches (20 s windows);
- ``acoustic:<name>``: ``acoustic_channel_stats.py`` subject vectors.

Analyses (subjects intersected across sources, counts recorded):

1. 5-class probe, stratified 5-fold CV over subjects; StandardScaler +
   multinomial LogisticRegression (balanced class weights), C chosen on 3
   inner folds by balanced accuracy. Balanced accuracy, macro-F1, pooled
   out-of-fold confusion matrix. Chance = 0.20.
2. Shuffled-label control (5 permutations, C=1).
3. Few-shot curve: n training subjects per dataset (2/5/10/20), 20 draws, C=1,
   test on all other subjects.
4. Erased probe: inside each outer fold, LEACE (dataset concept, equal dataset
   weights) is fit on the standardized training subjects and applied to train
   and test before the probe. Shows whether the erasure generalizes to unseen
   subjects.
5. Window-level probe (neural sources): GroupKFold by subject, at most 10
   windows per subject (seeded), C=1.
6. Overlap of the depression direction with the dataset subspace: on globally
   standardized subject vectors, a depression LogReg (equal dataset mass,
   balanced classes, C by 5-fold CV) gives w; D spans the 4 between-dataset
   mean differences. Reports cos^2(w, D) (random baseline 4/d) and the share of
   the depression score's variance that lies between datasets (eta^2).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.concept_erasure import fit_leace  # noqa: E402
from baselines.merged_subject_mean_heads import SubjectUnits, frozen_matrix, load_frozen_index  # noqa: E402
from baselines.run_merged_head_matrix import features_dir, frozen_caches  # noqa: E402
from src.merged.heads import _load_features  # noqa: E402

SEED = 0
C_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
FEW_SHOT = (2, 5, 10, 20)
FEW_SHOT_DRAWS = 20
SHUFFLES = 5
WINDOWS_PER_SUBJECT = 10
DATASETS = ("androids_interview", "cmdc", "d3tec", "daic", "turkish")


# ---------------------------------------------------------------- loading
def load_merged(features_root: Path, caches_root: Path, daic_audio: Path, language: str, modality: str) -> dict:
    fdir = features_dir(features_root, language, modality, 1337, "final", 0)
    caches = frozen_caches(caches_root, daic_audio, language, modality)
    index, audit = load_frozen_index(caches)
    xs, rows = [], []
    for part in ("outer_train", "outer_holdout"):
        _, r = _load_features(fdir, part)
        xs.append(frozen_matrix(r, index))
        rows += r
    x = np.vstack(xs).astype(np.float64)
    units = SubjectUnits(rows)
    subjects = sorted(units.units)
    return {"subjects": subjects, "x": units.means(x, subjects),
            "dataset": [units.dataset[s] for s in subjects], "label": units.y(subjects).tolist(),
            "windows": (x, [str(r["subject_id"]) for r in rows]),
            "provenance": {"features_dir": str(fdir), "caches": [str(c) for c in caches],
                           "base_models": sorted({c["base_model"] for c in audit["caches"]})}}


def load_w2v(cache_root: Path, pooling: str) -> dict:
    parts = {"androids_interview": ["androids_interview"], "cmdc": ["cmdc"], "d3tec": ["d3tec"],
             "turkish": ["turkish"], "daic": ["daic_test", "daic_dev"]}
    vectors, rows, seen = [], [], set()
    base_models = set()
    for dataset, dirs in parts.items():
        for name in dirs:
            d = cache_root / name / pooling
            base_models.add(json.loads((d / "extraction_metadata.json").read_text()).get("base_model", "?"))
            for npz in sorted(d.glob("*.npz")):
                with np.load(npz) as payload:
                    matrix = np.asarray(payload[payload.files[0]], dtype=np.float64)
                rr = [json.loads(line) for line in npz.with_name(f"{npz.stem}_rows.jsonl").read_text().splitlines()
                      if line.strip()]
                if len(rr) != len(matrix):
                    raise ValueError(f"{npz}: row count mismatch")
                for r, v in zip(rr, matrix):
                    key = (dataset, str(r["sample_id"]))
                    if key in seen:
                        continue
                    seen.add(key)
                    rows.append({"dataset": dataset, "subject_id": f"{dataset}::{r['subject_id']}",
                                 "label": int(r["label"]), "response_id": str(r.get("response_id") or r["sample_id"]),
                                 "sample_id": str(r["sample_id"])})
                    vectors.append(v)
    x = np.stack(vectors)
    units = SubjectUnits(rows)
    subjects = sorted(units.units)
    return {"subjects": subjects, "x": units.means(x, subjects),
            "dataset": [units.dataset[s] for s in subjects], "label": units.y(subjects).tolist(),
            "windows": (x, [r["subject_id"] for r in rows]),
            "provenance": {"cache_root": str(cache_root), "pooling": pooling, "base_models": sorted(base_models)}}


def load_acoustic(path: Path) -> dict:
    with np.load(path) as p:
        return {"subjects": p["subject_id"].tolist(), "x": p["x"].astype(np.float64), "dataset": p["dataset"].tolist(),
                "label": p["label"].astype(int).tolist(), "windows": None,
                "provenance": {"npz": str(path), "feature_names": p["feature_names"].tolist()}}


# ---------------------------------------------------------------- probes
def _probe(c: float) -> LogisticRegression:
    return LogisticRegression(C=c, class_weight="balanced", max_iter=5000)


def _scores(y_true, y_pred) -> dict:
    return {"balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro"))}


def _equal_dataset_weights(g: np.ndarray) -> np.ndarray:
    counts = {d: int(np.sum(g == d)) for d in set(g.tolist())}
    return np.asarray([1.0 / counts[d] for d in g])


def _select_c(x: np.ndarray, g: np.ndarray, erase: bool) -> float:
    inner = StratifiedKFold(3, shuffle=True, random_state=SEED)
    best = (-1.0, None)
    for c in C_GRID:
        preds = np.empty(len(g), dtype=object)
        for tr, te in inner.split(x, g):
            xtr, xte = _prepare(x[tr], x[te], g[tr], erase)
            preds[te] = _probe(c).fit(xtr, g[tr]).predict(xte)
        score = balanced_accuracy_score(g, preds.astype(str))
        if score > best[0] + 1e-12:
            best = (score, c)
    return best[1]


def _prepare(xtr, xte, gtr, erase: bool):
    scaler = StandardScaler().fit(xtr)
    xtr, xte = scaler.transform(xtr), scaler.transform(xte)
    if erase:
        eraser = fit_leace(xtr, gtr, _equal_dataset_weights(gtr))
        xtr, xte = eraser(xtr), eraser(xte)
    return xtr, xte


def probe_cv(x: np.ndarray, g: np.ndarray, erase: bool = False) -> dict:
    outer = StratifiedKFold(5, shuffle=True, random_state=SEED)
    preds = np.empty(len(g), dtype=object)
    chosen = []
    for tr, te in outer.split(x, g):
        c = _select_c(x[tr], g[tr], erase)
        chosen.append(c)
        xtr, xte = _prepare(x[tr], x[te], g[tr], erase)
        preds[te] = _probe(c).fit(xtr, g[tr]).predict(xte)
    preds = preds.astype(str)
    labels = sorted(set(g.tolist()))
    return {**_scores(g, preds), "selected_C": chosen, "labels": labels,
            "confusion_matrix": confusion_matrix(g, preds, labels=labels).tolist(),
            "per_dataset_recall": {d: float(np.mean(preds[g == d] == d)) for d in labels}}


def shuffled_control(x: np.ndarray, g: np.ndarray) -> dict:
    rng = np.random.default_rng(SEED)
    outer = StratifiedKFold(5, shuffle=True, random_state=SEED)
    scores = []
    for _ in range(SHUFFLES):
        gs = rng.permutation(g)
        preds = np.empty(len(g), dtype=object)
        for tr, te in outer.split(x, gs):
            xtr, xte = _prepare(x[tr], x[te], gs[tr], False)
            preds[te] = _probe(1.0).fit(xtr, gs[tr]).predict(xte)
        scores.append(balanced_accuracy_score(gs, preds.astype(str)))
    return {"balanced_accuracy_mean": float(np.mean(scores)), "balanced_accuracy_sd": float(np.std(scores)),
            "permutations": SHUFFLES}


def few_shot(x: np.ndarray, g: np.ndarray) -> dict:
    rng = np.random.default_rng(SEED)
    out = {}
    for n in FEW_SHOT:
        scores = []
        for _ in range(FEW_SHOT_DRAWS):
            tr = np.concatenate([rng.choice(np.nonzero(g == d)[0], size=n, replace=False) for d in sorted(set(g))])
            te = np.setdiff1d(np.arange(len(g)), tr)
            xtr, xte = _prepare(x[tr], x[te], g[tr], False)
            scores.append(balanced_accuracy_score(g[te], _probe(1.0).fit(xtr, g[tr]).predict(xte)))
        out[str(n)] = {"balanced_accuracy_mean": float(np.mean(scores)), "balanced_accuracy_sd": float(np.std(scores))}
    return {"draws": FEW_SHOT_DRAWS, "C": 1.0, "per_dataset_train_subjects": out}


def window_probe(windows, subject_dataset: dict[str, str]) -> dict | None:
    if windows is None:
        return None
    x, subjects = windows
    rng = np.random.default_rng(SEED)
    by_subject = defaultdict(list)
    for i, s in enumerate(subjects):
        if s in subject_dataset:
            by_subject[s].append(i)
    keep = np.sort(np.concatenate([rng.choice(ix, size=min(len(ix), WINDOWS_PER_SUBJECT), replace=False)
                                   for ix in by_subject.values()]))
    xs = x[keep]
    groups = np.asarray([subjects[i] for i in keep])
    g = np.asarray([subject_dataset[s] for s in groups])
    preds = np.empty(len(g), dtype=object)
    for tr, te in GroupKFold(5).split(xs, g, groups):
        xtr, xte = _prepare(xs[tr], xs[te], g[tr], False)
        preds[te] = _probe(1.0).fit(xtr, g[tr]).predict(xte)
    return {**_scores(g, preds.astype(str)), "windows": int(len(g)), "max_windows_per_subject": WINDOWS_PER_SUBJECT,
            "C": 1.0}


def depression_overlap(x: np.ndarray, g: np.ndarray, y: np.ndarray) -> dict:
    z = StandardScaler().fit_transform(x)
    cells = defaultdict(int)
    for d, label in zip(g, y):
        cells[(d, int(label))] += 1
    datasets = sorted(set(g.tolist()))
    w_sample = np.asarray([1.0 / (len(datasets) * 2 * cells[(d, int(label))]) for d, label in zip(g, y)])
    w_sample *= len(w_sample) / w_sample.sum()
    model = LogisticRegressionCV(Cs=list(C_GRID), cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                                 scoring="balanced_accuracy", max_iter=5000).fit(z, y, sample_weight=w_sample)
    w = model.coef_[0]
    overall = np.mean([z[g == d].mean(0) for d in datasets], axis=0)
    diffs = np.stack([z[g == d].mean(0) - overall for d in datasets]).T  # (dims, k)
    u, s, _ = np.linalg.svd(diffs, full_matrices=False)
    basis = u[:, s > 1e-10 * s[0]]
    cos2 = float(np.sum((basis.T @ w) ** 2) / np.sum(w ** 2))
    score = z @ w
    grand = score.mean()
    between = sum(np.sum(g == d) * (score[g == d].mean() - grand) ** 2 for d in datasets)
    total = float(np.sum((score - grand) ** 2))
    return {"cos2_depression_vs_dataset_subspace": cos2, "dataset_subspace_dims": int(basis.shape[1]),
            "random_baseline_cos2": float(basis.shape[1] / z.shape[1]),
            "eta2_depression_score_between_datasets": float(between / total), "depression_C": float(model.C_[0]),
            "note": "in-sample descriptive; the merged head E vs B run is the out-of-sample test"}


# ---------------------------------------------------------------- main
def parse_source(spec: str, args) -> tuple[str, dict]:
    kind, name, *rest = spec.split(":")
    if kind == "merged":
        language, modality = rest
        return name, load_merged(args.merged_features_root, args.frozen_caches_root, args.frozen_daic_audio,
                                 language, modality)
    if kind == "w2v":
        return name, load_w2v(args.w2v_cache_root, rest[0])
    if kind == "acoustic":
        return name, load_acoustic(args.acoustic_npz)
    raise ValueError(f"Unknown source kind {kind!r}.")


def render(result: dict) -> str:
    lines = ["# Dataset (= language) probe on frozen features", "",
             f"Commit `{result['git_head']}`. Subjects per dataset: "
             + ", ".join(f"{d} {n}" for d, n in result["subjects_per_dataset"].items()) + ". Chance = 0.20.", "",
             "| Source | dim | Probe bal. acc | macro-F1 | Shuffled | Erased (LEACE) | Window-level | "
             "Few-shot n=2 / 5 / 20 | cos² dep·dataset (random) | η² dep score |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in result["sources"].items():
        fs = r["few_shot"]["per_dataset_train_subjects"]
        win = r["window_probe"]
        ov = r["depression_overlap"]
        lines.append(
            f"| {name} | {r['dimension']} | {r['probe']['balanced_accuracy']:.3f} | {r['probe']['macro_f1']:.3f} | "
            f"{r['shuffled']['balanced_accuracy_mean']:.3f} | {r['erased_probe']['balanced_accuracy']:.3f} | "
            f"{'—' if win is None else format(win['balanced_accuracy'], '.3f')} | "
            f"{fs['2']['balanced_accuracy_mean']:.3f} / {fs['5']['balanced_accuracy_mean']:.3f} / "
            f"{fs['20']['balanced_accuracy_mean']:.3f} | {ov['cos2_depression_vs_dataset_subspace']:.3f} "
            f"({ov['random_baseline_cos2']:.4f}) | {ov['eta2_depression_score_between_datasets']:.3f} |")
    lines += ["", "Per-dataset recall of the probe (out-of-fold):", ""]
    for name, r in result["sources"].items():
        lines.append(f"- {name}: " + ", ".join(f"{d} {v:.2f}" for d, v in r["probe"]["per_dataset_recall"].items()))
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", action="append", required=True,
                        help="merged:<name>:<language>:<modality> | w2v:<name>:<pooling> | acoustic:<name>")
    parser.add_argument("--merged-features-root", type=Path)
    parser.add_argument("--frozen-caches-root", type=Path)
    parser.add_argument("--frozen-daic-audio", type=Path)
    parser.add_argument("--w2v-cache-root", type=Path)
    parser.add_argument("--acoustic-npz", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--no-intersect", action="store_true", help="Keep each source's own subject set.")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"Refusing to overwrite {args.output_dir}.")
    sources = dict(parse_source(spec, args) for spec in args.source)
    common = set.intersection(*(set(s["subjects"]) for s in sources.values()))
    result = {"git_head": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                         cwd=PROJECT_ROOT).stdout.strip(),
              "git_dirty": bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                                               capture_output=True, text=True, cwd=PROJECT_ROOT).stdout.strip()),
              "seed": SEED, "intersected": not args.no_intersect, "sources": {}}
    for name, src in sources.items():
        keep = [i for i, s in enumerate(src["subjects"]) if args.no_intersect or s in common]
        x = src["x"][keep]
        g = np.asarray([src["dataset"][i] for i in keep])
        y = np.asarray([src["label"][i] for i in keep])
        subject_dataset = {src["subjects"][i]: src["dataset"][i] for i in keep}
        print(f"{name}: {len(keep)} subjects (dropped {len(src['subjects']) - len(keep)}), dim {x.shape[1]}", flush=True)
        result["sources"][name] = {
            "subjects": len(keep), "dropped_by_intersection": len(src["subjects"]) - len(keep),
            "dimension": int(x.shape[1]), "provenance": src["provenance"],
            "probe": probe_cv(x, g), "shuffled": shuffled_control(x, g), "few_shot": few_shot(x, g),
            "erased_probe": probe_cv(x, g, erase=True),
            "window_probe": window_probe(src["windows"], subject_dataset),
            "depression_overlap": depression_overlap(x, g, y)}
        print(json.dumps({k: result["sources"][name][k]["balanced_accuracy"]
                          for k in ("probe", "erased_probe")}), flush=True)
        g_all = g
    result["subjects_per_dataset"] = {d: int(np.sum(g_all == d)) for d in sorted(set(g_all.tolist()))}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "probe_results.json").write_text(json.dumps(result, indent=2))
    (args.output_dir / "probe_results.md").write_text(render(result))
    print(render(result))


if __name__ == "__main__":
    main()
