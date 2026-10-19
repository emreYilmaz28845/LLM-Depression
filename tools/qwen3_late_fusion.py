"""Late fusion of six Qwen3 subject-level scores (oracle, averaging, stacking).

Per subject there are six candidate-label likelihood margins: three training
regimes (unilang = standalone native, multilang = merged native, multilang-en =
merged English) times two modalities (text = Qwen3.8-27B, audio+text =
Qwen3-Omni). All six come from already finished, locally synced runs; nothing
is retrained. Margin > 0 means "Depressed" and reproduces every recorded
prediction (checked on load).

Methods:

* single models (the six inputs) as references,
* oracle: a subject counts as correct if any of the six models is correct
  (upper bound for any combination rule),
* training-free: mean probability (all six, text three, audio+text three) and
  majority vote of six (3-3 ties fall back to the six-model mean),
* stacking: standardized LogReg and fixed shallow XGBoost on the six margins,
  balanced classes, threshold 0.5. CV datasets use leave-one-fold-out over the
  same outer folds the base models used ("per_dataset" and "pooled" with a
  dataset one-hot). DAIC only has official-test scores for the unilang model,
  so it gets leave-one-subject-out within the test set (small-sample extra) and
  a transfer meta-model trained on the four CV datasets (no DAIC data).

Metrics are binary_strict subject-level Macro-F1 / Positive-F1 / UAR; CV
datasets report the unweighted fold mean (paper convention) and the pooled
out-of-fold value. Significance: exact paired prediction-swap test (primary),
label-stratified paired bootstrap CI and exact McNemar on pooled out-of-fold
rows, each fusion method against the best single model of that dataset (picked
on the same rows, which favours the baseline). Holm runs across all fusion
comparisons, per metric. Read-only on evidence.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.daic_statistics import (  # noqa: E402
    exact_mcnemar,
    exact_paired_prediction_swap,
    holm_adjust,
    stratified_paired_bootstrap_many,
)
from src.metrics import classification_metrics  # noqa: E402
from src.utils import save_json, sha256_file  # noqa: E402

WORKTREES = Path("/home/emre/Projects/AudioLLM/worktrees")
DEFAULT_ROOTS = {
    "native": WORKTREES / "LLM-Depression-feat-qwen3-multiseed-native-20261002/output_model/qwen3_multiseed_native_20261002",
    "androids_mergedfolds": WORKTREES
    / "LLM-Depression-feat-qwen3-androids-mergedfolds-20261015/output_model/qwen3_androids_mergedfolds_native_20261015",
    "merged": WORKTREES / "LLM-Depression-feat-qwen3-multiseed-merged-20261002/outputs/symmetric_merged",
}
CV_DATASETS = ("cmdc", "d3tec", "turkish", "androids_interview")
TEST_DATASETS = ("daic",)
MODALITIES = ("text_only", "audio_text")
REGIMES = ("uni", "multi", "multi_en")
MODELS = tuple(f"{regime}_{modality}" for regime in REGIMES for modality in MODALITIES)
BACKBONE = {"text_only": "Qwen3.8-27B", "audio_text": "Qwen3-Omni-30B-A3B"}
METRICS = ("macro_f1", "positive_f1", "macro_recall")
LABELS = {"macro_f1": "Macro-F1", "positive_f1": "Positive-F1", "macro_recall": "UAR"}
XGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 2,
    "learning_rate": 0.05,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "min_child_weight": 1.0,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "n_jobs": 1,
}
LOGREG_PARAMS = {"C": 1.0, "class_weight": "balanced", "max_iter": 1000}


def source_path(roots: dict[str, Path], seed: int, dataset: str, fold: int, regime: str, modality: str) -> Path:
    if regime == "uni":
        if dataset == "androids_interview":
            run = f"q3amf_{modality}_native_s{seed}_f{fold}_r1"
            base = roots["androids_mergedfolds"] / modality / dataset
        else:
            run = f"q3ms_{dataset}_{modality}_native_s{seed}_f{fold}_r1"
            base = roots["native"] / modality / dataset
        return base / run / f"fold_{fold}/best_model/standalone_eval/predictions_subject_level.csv"
    language = "native" if regime == "multi" else "english"
    stage = "final" if dataset in TEST_DATASETS else "cv"
    return (
        roots["merged"] / f"qwen3_pooled_{language}" / modality / f"qmsm_{language}_{modality}_s{seed}"
        / stage / f"fold_{fold}/qwen/{dataset}/predictions_subject_level.csv"
    )


def subject_margin(row: dict[str, str]) -> float:
    """Depressed-minus-non-depressed likelihood margin of one subject row."""
    if row.get("score_margin") not in (None, ""):
        return float(row["score_margin"])
    return float(row["dep_score"]) - float(row["non_score"])


def read_scores(path: Path) -> dict[str, tuple[int, float]]:
    out: dict[str, tuple[int, float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            subject = row["subject_id"].split("::")[-1]
            if subject in out:
                raise ValueError(f"{path}: duplicate subject {subject}.")
            margin = subject_margin(row)
            if not math.isfinite(margin) or margin == 0.0:
                raise ValueError(f"{path}: subject {subject} has no usable margin.")
            if int(margin > 0) != int(row["prediction"]):
                raise ValueError(f"{path}: margin sign disagrees with the recorded prediction for {subject}.")
            out[subject] = (int(row["label"]), margin)
    return out


def load_matched(roots: dict[str, Path], seed: int) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """One row per (dataset, subject) with all six margins; fail closed on any mismatch."""
    rows: list[dict[str, Any]] = []
    files: list[dict[str, str]] = []
    for dataset in CV_DATASETS + TEST_DATASETS:
        folds = [0] if dataset in TEST_DATASETS else range(5)
        for fold in folds:
            per_model = {}
            for regime in REGIMES:
                for modality in MODALITIES:
                    path = source_path(roots, seed, dataset, fold, regime, modality)
                    per_model[f"{regime}_{modality}"] = read_scores(path)
                    files.append({
                        "dataset": dataset, "fold": str(fold), "model": f"{regime}_{modality}",
                        "path": str(path), "sha256": sha256_file(path),
                    })
            subjects = set(per_model[MODELS[0]])
            for model, scores in per_model.items():
                if set(scores) != subjects:
                    raise ValueError(f"{dataset} fold {fold}: {model} has a different subject set.")
            for subject in sorted(subjects):
                labels = {per_model[model][subject][0] for model in MODELS}
                if len(labels) != 1:
                    raise ValueError(f"{dataset} fold {fold}: label disagreement for a subject.")
                rows.append({
                    "dataset": dataset, "fold": fold, "subject_id": subject, "label": labels.pop(),
                    **{model: per_model[model][subject][1] for model in MODELS},
                })
    keys = [(row["dataset"], row["subject_id"]) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("A subject is held out in more than one fold.")
    return rows, files


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def features(rows: list[dict[str, Any]], datasets: tuple[str, ...] = ()) -> np.ndarray:
    x = np.asarray([[row[model] for model in MODELS] for row in rows], dtype=float)
    if datasets:
        onehot = np.asarray([[float(row["dataset"] == name) for name in datasets] for row in rows])
        x = np.hstack([x, onehot])
    return x


def make_logreg(seed: int):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), LogisticRegression(random_state=seed, **LOGREG_PARAMS))


def make_xgb(seed: int, y: np.ndarray):
    from xgboost import XGBClassifier

    positives = int(y.sum())
    weight = (len(y) - positives) / positives if positives else 1.0
    return XGBClassifier(random_state=seed, scale_pos_weight=weight, eval_metric="logloss", **XGB_PARAMS)


def fit_predict(kind: str, seed: int, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    if len(set(y_train.tolist())) < 2:
        raise ValueError("Meta-model training data has a single class.")
    model = make_logreg(seed) if kind == "logreg" else make_xgb(seed, y_train)
    model.fit(x_train, y_train)
    return model.predict_proba(x_test)[:, 1]


def training_free(rows: list[dict[str, Any]]) -> dict[str, list[int]]:
    margins = features(rows)
    probs = sigmoid(margins)
    text = [MODELS.index(m) for m in MODELS if m.endswith("text_only")]
    audio_text = [MODELS.index(m) for m in MODELS if m.endswith("audio_text")]
    votes = (margins > 0).sum(axis=1)
    mean_all = probs.mean(axis=1)
    out = {model: (margins[:, i] > 0).astype(int).tolist() for i, model in enumerate(MODELS)}
    out["mean_all6"] = (mean_all >= 0.5).astype(int).tolist()
    out["mean_text3"] = (probs[:, text].mean(axis=1) >= 0.5).astype(int).tolist()
    out["mean_audio_text3"] = (probs[:, audio_text].mean(axis=1) >= 0.5).astype(int).tolist()
    out["vote_all6"] = np.where(votes == 3, mean_all >= 0.5, votes > 3).astype(int).tolist()
    labels = np.asarray([row["label"] for row in rows])
    any_correct = ((margins > 0).astype(int) == labels[:, None]).any(axis=1)
    out["oracle_any6"] = np.where(any_correct, labels, 1 - labels).astype(int).tolist()
    return out


def agreement(rows: list[dict[str, Any]]) -> dict[str, int]:
    """How often the six models agree; fusion can only change disagreeing subjects."""
    margins = features(rows)
    votes = (margins > 0).astype(int)
    labels = np.asarray([row["label"] for row in rows])
    unanimous = (votes.min(axis=1) == votes.max(axis=1))
    correct = votes == labels[:, None]
    return {
        "subjects": len(rows),
        "all_six_agree": int(unanimous.sum()),
        "all_six_correct": int(correct.all(axis=1).sum()),
        "all_six_wrong": int((~correct).all(axis=1).sum()),
    }


def leave_fold_out(rows: list[dict[str, Any]], kind: str, seed: int, datasets: tuple[str, ...] = ()) -> list[int]:
    x = features(rows, datasets)
    y = np.asarray([row["label"] for row in rows])
    folds = np.asarray([row["fold"] for row in rows])
    probs = np.full(len(rows), np.nan)
    for fold in sorted(set(folds.tolist())):
        test = folds == fold
        probs[test] = fit_predict(kind, seed, x[~test], y[~test], x[test])
    return (probs >= 0.5).astype(int).tolist()


def leave_one_out(rows: list[dict[str, Any]], kind: str, seed: int) -> list[int]:
    x = features(rows)
    y = np.asarray([row["label"] for row in rows])
    out = []
    for i in range(len(rows)):
        mask = np.arange(len(rows)) != i
        out.append(int(fit_predict(kind, seed, x[mask], y[mask], x[i:i + 1])[0] >= 0.5))
    return out


def transfer(train_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]], kind: str, seed: int) -> list[int]:
    y = np.asarray([row["label"] for row in train_rows])
    probs = fit_predict(kind, seed, features(train_rows), y, features(test_rows))
    return (probs >= 0.5).astype(int).tolist()


def strict_metrics(labels: list[int], predictions: list[int]) -> dict[str, float]:
    values = classification_metrics(labels, predictions)
    return {metric: float(values[metric]) for metric in METRICS}


def summarize(rows: list[dict[str, Any]], predictions: list[int]) -> dict[str, Any]:
    labels = [row["label"] for row in rows]
    out: dict[str, Any] = {"subjects": len(rows), "pooled": strict_metrics(labels, predictions)}
    folds = sorted({row["fold"] for row in rows})
    if len(folds) > 1:
        per_fold = []
        for fold in folds:
            idx = [i for i, row in enumerate(rows) if row["fold"] == fold]
            per_fold.append(strict_metrics([labels[i] for i in idx], [predictions[i] for i in idx]))
        out["per_fold"] = per_fold
        out["fold_mean"] = {metric: float(np.mean([f[metric] for f in per_fold])) for metric in METRICS}
    out["headline"] = out.get("fold_mean", out["pooled"])
    return out


def run(rows: list[dict[str, Any]], seed: int) -> dict[str, dict[str, list[int]]]:
    """Predictions per dataset and method, aligned with that dataset's rows."""
    by_dataset = {name: [row for row in rows if row["dataset"] == name] for name in CV_DATASETS + TEST_DATASETS}
    preds: dict[str, dict[str, list[int]]] = {name: training_free(group) for name, group in by_dataset.items()}
    for name in CV_DATASETS:
        for kind in ("logreg", "xgb"):
            preds[name][f"{kind}_per_dataset"] = leave_fold_out(by_dataset[name], kind, seed)
    cv_rows = [row for name in CV_DATASETS for row in by_dataset[name]]
    for kind in ("logreg", "xgb"):
        pooled = leave_fold_out(cv_rows, kind, seed, CV_DATASETS)
        start = 0
        for name in CV_DATASETS:
            size = len(by_dataset[name])
            preds[name][f"{kind}_pooled"] = pooled[start:start + size]
            start += size
        for name in TEST_DATASETS:
            preds[name][f"{kind}_loo"] = leave_one_out(by_dataset[name], kind, seed)
            preds[name][f"{kind}_transfer"] = transfer(cv_rows, by_dataset[name], kind, seed)
    return preds


def significance(
    rows: list[dict[str, Any]], preds: dict[str, dict[str, list[int]]], results: dict[str, dict[str, Any]],
    *, iterations: int, seed: int, alpha: float,
) -> list[dict[str, Any]]:
    comparisons = []
    for dataset, methods in preds.items():
        group = [row for row in rows if row["dataset"] == dataset]
        best = max(MODELS, key=lambda m: (results[dataset][m]["pooled"]["macro_f1"], -MODELS.index(m)))

        def paired(method: str) -> list[dict[str, Any]]:
            return [
                {"subject_id": f"{row['dataset']}::{row['subject_id']}", "label": row["label"], "prediction": p}
                for row, p in zip(group, methods[method])
            ]

        baseline = paired(best)
        for method in methods:
            if method in MODELS or method == "oracle_any6":
                continue
            comparison = paired(method)
            permutation = exact_paired_prediction_swap(baseline, comparison, metrics=METRICS)
            bootstrap = stratified_paired_bootstrap_many(
                baseline, comparison, metrics=METRICS, iterations=iterations, seed=seed
            )
            comparisons.append({
                "dataset": dataset, "baseline": best, "comparison": method, "subjects": len(group),
                "aggregation": "pooled_out_of_fold_subject_level",
                "metrics": {m: {"permutation": permutation[m], "bootstrap": bootstrap[m]} for m in METRICS},
                "mcnemar": exact_mcnemar(baseline, comparison),
            })
    for metric in METRICS:
        adjusted = holm_adjust([c["metrics"][metric]["permutation"]["p_value"] for c in comparisons])
        for c, value in zip(comparisons, adjusted):
            c["metrics"][metric]["permutation"]["p_value_holm"] = value
            c["metrics"][metric]["permutation"]["significant_holm"] = value <= alpha
    return comparisons


def fmt(values: dict[str, float]) -> str:
    return " / ".join(f"{values[m]:.3f}" for m in METRICS)


def render(payload: dict[str, Any]) -> str:
    results = payload["results"]
    lines = [
        "# Qwen3 late fusion: six subject-level scores",
        "",
        f"Seed {payload['seed']}. Text = {BACKBONE['text_only']}, audio+text = {BACKBONE['audio_text']}. "
        "uni = standalone native, multi = merged native, multi_en = merged English.",
        "Cells are Macro-F1 / Positive-F1 / UAR, binary_strict, subject level. CV datasets: unweighted fold mean "
        "(pooled out-of-fold in the JSON). DAIC: official test set (one fold).",
        "",
    ]
    for dataset, methods in results.items():
        best = payload["best_single"][dataset]
        agree = payload["agreement"][dataset]
        lines += [f"## {dataset} (N={agree['subjects']})", "",
                  f"All six models agree on {agree['all_six_agree']} subjects ({agree['all_six_correct']} correct, "
                  f"{agree['all_six_wrong']} wrong); only the other {agree['subjects'] - agree['all_six_agree']} "
                  "can change under any fusion rule.", "",
                  "| Method | Macro-F1 / Positive-F1 / UAR |", "|---|---|"]
        for method, summary in methods.items():
            tag = " (best single)" if method == best else ""
            lines.append(f"| {method}{tag} | {fmt(summary['headline'])} |")
        lines.append("")
    lines += [
        "## Paired significance vs best single model (pooled out-of-fold, Macro-F1)",
        "",
        "Delta = fusion - best single. p = exact paired prediction-swap; Holm across all "
        f"{len(payload['significance'])} fusion comparisons per metric. * = significant after Holm at alpha "
        f"{payload['alpha']}.",
        "",
        "| Dataset | Fusion | Best single | Delta Macro-F1 [95% CI] | p | p (Holm) | McNemar b/c |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in payload["significance"]:
        perm, boot, mc = c["metrics"]["macro_f1"]["permutation"], c["metrics"]["macro_f1"]["bootstrap"], c["mcnemar"]
        mark = " *" if perm["significant_holm"] else ""
        lines.append(
            f"| {c['dataset']} | {c['comparison']} | {c['baseline']} | {perm['observed_delta']:+.3f} "
            f"[{boot['ci_low']:+.3f}, {boot['ci_high']:+.3f}] | {perm['p_value']:.3f} | "
            f"{perm['p_value_holm']:.3f}{mark} | {mc['baseline_only_correct']}/{mc['comparison_only_correct']} |"
        )
    lines += ["", "McNemar b = only best single correct, c = only fusion correct.", ""]
    lines += ["## Caveats", ""] + [f"- {note}" for note in payload["caveats"]] + [""]
    return "\n".join(lines)


CAVEATS = [
    "CMDC and Turkish unilang (standalone) scores are train_val: the same outer fold selected the checkpoint "
    "and is scored, so those input scores are optimistic. Multilang scores are outer holdout.",
    "Androids unilang uses the 2026-10-19 merged-fold standalone rerun so all six models share folds; it is "
    "not the official Androids fold split.",
    "Stacking is leave-one-fold-out on base-model out-of-fold scores. The base models that scored the training "
    "folds saw the test fold during their own training; this is the usual cross-validated stacking setup.",
    "DAIC LogReg/XGBoost leave-one-out uses 46 training subjects per fit: small sample, noisy. DAIC transfer "
    "meta-models never see DAIC labels.",
    "The best single model is picked on the same rows it is compared on, which favours the baseline.",
    "Single seed (7) per base model; no seed variance.",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=7, help="base-model training seed to fuse")
    parser.add_argument("--meta-seed", type=int, default=1337)
    parser.add_argument("--native-root", type=Path, default=DEFAULT_ROOTS["native"])
    parser.add_argument("--androids-root", type=Path, default=DEFAULT_ROOTS["androids_mergedfolds"])
    parser.add_argument("--merged-root", type=Path, default=DEFAULT_ROOTS["merged"])
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/late_fusion/qwen3_seed7")
    parser.add_argument("--bootstrap-iterations", type=int, default=100_000)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    roots = {"native": args.native_root, "androids_mergedfolds": args.androids_root, "merged": args.merged_root}
    rows, files = load_matched(roots, args.seed)
    preds = run(rows, args.meta_seed)
    results: dict[str, dict[str, Any]] = {}
    for dataset, methods in preds.items():
        group = [row for row in rows if row["dataset"] == dataset]
        results[dataset] = {method: summarize(group, p) for method, p in methods.items()}
    comparisons = significance(
        rows, preds, results, iterations=args.bootstrap_iterations, seed=args.meta_seed, alpha=args.alpha
    )
    payload = {
        "schema_version": "qwen3_late_fusion.v1",
        "seed": args.seed,
        "meta_seed": args.meta_seed,
        "models": {f"{r}_{m}": {"regime": r, "modality": m, "backbone": BACKBONE[m]}
                   for r in REGIMES for m in MODALITIES},
        "feature": "candidate-label likelihood margin (Depressed - Non-depressed); probability = sigmoid(margin)",
        "xgb_params": XGB_PARAMS,
        "logreg_params": LOGREG_PARAMS,
        "label_counts": {d: dict(Counter(row["label"] for row in rows if row["dataset"] == d)) for d in preds},
        "best_single": {d: max(MODELS, key=lambda m: (results[d][m]["pooled"]["macro_f1"], -MODELS.index(m)))
                        for d in preds},
        "agreement": {d: agreement([row for row in rows if row["dataset"] == d]) for d in preds},
        "results": results,
        "alpha": args.alpha,
        "bootstrap_iterations": args.bootstrap_iterations,
        "significance": comparisons,
        "caveats": CAVEATS,
        "inputs": files,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(payload, args.output_dir / "report.json")
    with (args.output_dir / "subject_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        method_names = sorted({m for methods in preds.values() for m in methods})
        writer = csv.writer(handle)
        writer.writerow(["dataset", "fold", "subject_id", "label", *[f"margin_{m}" for m in MODELS], *method_names])
        positions: dict[str, int] = defaultdict(int)
        for row in rows:
            i = positions[row["dataset"]]
            positions[row["dataset"]] += 1
            writer.writerow([
                row["dataset"], row["fold"], row["subject_id"], row["label"], *[row[m] for m in MODELS],
                *[preds[row["dataset"]].get(m, [""] * (i + 1))[i] for m in method_names],
            ])
    report = render(payload)
    (args.output_dir / "report.md").write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
