"""Paired t-tests for MFCC XGBoost heads vs each other and vs Qwen3-Omni.

CV datasets: paired t-test over the 5 outer folds on per-fold subject-level
Macro-F1 (binary_strict). Folds are paired only when both sides held out the
same subjects. Two versions are reported:

* standard paired t-test (scipy.stats.ttest_rel, two-sided, df = k - 1);
* Nadeau-Bengio corrected resampled t-test, which inflates the variance by
  (1/k + n_test/n_train) because CV folds share training data.

DAIC has one fixed split, so a fold t-test is impossible; it gets a paired
t-test on per-subject correctness (1 = right, 0 = wrong) instead. That tests
accuracy, not Macro-F1, and is labelled as such. Read-only on evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scipy import stats  # noqa: E402

from src.metrics import classification_metrics  # noqa: E402
from src.utils import read_jsonl, save_json, sha256_file  # noqa: E402
from tools.mfcc_vs_omni_significance import OMNI_CV_ROOT, OMNI_DAIC_ROOT  # noqa: E402

CV_DATASETS = {
    "cmdc": ("CMDC", "cmdc"),
    "d3tec": ("D3TEC", "d3tec"),
    "turkish_pooled": ("Turkish pooled", "turkish"),
    "androids_standalone": ("Androids (standalone folds)", "androids_interview"),
}
COMPARISONS = (
    ("MFCC fixed", "MFCC Optuna-100"),
    ("MFCC fixed", "Omni audio only"),
    ("MFCC Optuna-100", "Omni audio only"),
    ("MFCC fixed", "Omni audio+text"),
    ("MFCC Optuna-100", "Omni audio+text"),
)


def strict_prediction(row: dict[str, Any]) -> int:
    try:
        value = int(float(row["prediction"]))
    except (TypeError, ValueError):
        value = -1
    return value if value in (0, 1) else 1 - int(row["label"])


def read_any(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows = read_jsonl(path)
    else:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    return [{"subject_id": str(r["subject_id"]), "label": int(r["label"]), "prediction": strict_prediction(r)} for r in rows]


def method_files(mfcc_dir: Path, omni_dataset: str, *, daic: bool) -> dict[str, list[Path]]:
    folds = sorted(mfcc_dir.glob("fold_*"), key=lambda path: int(path.name.split("_")[1]))
    root = OMNI_DAIC_ROOT if daic else OMNI_CV_ROOT
    files = {
        "MFCC fixed": [fold / "fixed_heads/xgb_raw/predictions_subject_level.jsonl" for fold in folds],
        "MFCC Optuna-100": [fold / "xgb_optuna100/predictions_subject_level.jsonl" for fold in folds],
    }
    for modality, label in (("audio_only", "Omni audio only"), ("audio_text", "Omni audio+text")):
        paths = []
        for fold in folds:
            index = fold.name.split("_")[1]
            pattern = f"*/fold_{index}/best_model/standalone_eval/predictions_subject_level.csv"
            matches = sorted((root / modality / omni_dataset).glob(pattern))
            if len(matches) != 1:
                raise ValueError(f"{label} {omni_dataset} fold {index}: expected one file, found {len(matches)}.")
            paths.append(matches[0])
        files[label] = paths
    return files


def fold_scores(files: dict[str, list[Path]]) -> tuple[dict[str, list[float]], list[int], list[dict[str, Any]]]:
    scores: dict[str, list[float]] = {name: [] for name in files}
    sizes: list[int] = []
    evidence = []
    for fold_index in range(len(files["MFCC fixed"])):
        reference: set[str] | None = None
        for name, paths in files.items():
            rows = read_any(paths[fold_index])
            subjects = {row["subject_id"] for row in rows}
            if reference is None:
                reference = subjects
                sizes.append(len(subjects))
            elif subjects != reference:
                raise ValueError(f"Fold {fold_index}: {name} held out different subjects.")
            metrics = classification_metrics([r["label"] for r in rows], [r["prediction"] for r in rows])
            scores[name].append(float(metrics["macro_f1"]))
            evidence.append({"method": name, "fold": fold_index, "path": str(paths[fold_index]), "sha256": sha256_file(paths[fold_index])})
    return scores, sizes, evidence


def paired_t(a: list[float], b: list[float], test_train_ratio: float) -> dict[str, Any]:
    diffs = [y - x for x, y in zip(a, b)]
    k = len(diffs)
    mean = statistics.fmean(diffs)
    sd = statistics.stdev(diffs)
    result: dict[str, Any] = {"folds": k, "fold_diffs": diffs, "mean_diff": mean, "sd_diff": sd, "df": k - 1}
    if sd == 0.0:
        same = mean == 0.0
        result.update({"t": 0.0 if same else math.inf, "p": 1.0 if same else 0.0,
                       "t_corrected": 0.0 if same else math.inf, "p_corrected": 1.0 if same else 0.0,
                       "note": "all fold differences equal"})
        return result
    standard = stats.ttest_rel(b, a)
    t_corrected = mean / math.sqrt((1.0 / k + test_train_ratio) * sd**2)
    result.update({
        "t": float(standard.statistic),
        "p": float(standard.pvalue),
        "t_corrected": t_corrected,
        "p_corrected": float(2 * stats.t.sf(abs(t_corrected), k - 1)),
        "test_train_ratio": test_train_ratio,
    })
    return result


def daic_correctness_t(files: dict[str, list[Path]], left: str, right: str) -> dict[str, Any]:
    a = {r["subject_id"]: r for r in read_any(files[left][0])}
    b = {r["subject_id"]: r for r in read_any(files[right][0])}
    if set(a) != set(b):
        raise ValueError(f"DAIC: {left} and {right} scored different subjects.")
    keys = sorted(a)
    x = [int(a[key]["prediction"] == a[key]["label"]) for key in keys]
    y = [int(b[key]["prediction"] == b[key]["label"]) for key in keys]
    diffs = [j - i for i, j in zip(x, y)]
    out: dict[str, Any] = {"subjects": len(keys), "accuracy_left": statistics.fmean(x),
                           "accuracy_right": statistics.fmean(y), "mean_diff": statistics.fmean(diffs), "df": len(keys) - 1}
    if statistics.pstdev(diffs) == 0.0:
        out.update({"t": 0.0, "p": 1.0, "note": "identical correctness on every subject"})
    else:
        test = stats.ttest_rel(y, x)
        out.update({"t": float(test.statistic), "p": float(test.pvalue)})
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT / "outputs/mfcc_xgb")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/significance/mfcc_ttest")
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    evidence: dict[str, Any] = {}
    for mfcc_dataset, (display, omni_dataset) in CV_DATASETS.items():
        files = method_files(args.root / mfcc_dataset, omni_dataset, daic=False)
        scores, sizes, evidence[display] = fold_scores(files)
        total = sum(sizes)
        ratio = statistics.fmean(size / (total - size) for size in sizes)
        for left, right in COMPARISONS:
            rows.append({"dataset": display, "left": left, "right": right, "test": "paired_t_over_folds_macro_f1",
                         "left_fold_mean": statistics.fmean(scores[left]),
                         "right_fold_mean": statistics.fmean(scores[right]),
                         "left_folds": scores[left], "right_folds": scores[right],
                         **paired_t(scores[left], scores[right], ratio)})
    daic_files = method_files(args.root / "daic", "daic", daic=True)
    evidence["DAIC"] = [{"method": name, "path": str(paths[0]), "sha256": sha256_file(paths[0])} for name, paths in daic_files.items()]
    for left, right in COMPARISONS:
        rows.append({"dataset": "DAIC", "left": left, "right": right, "test": "paired_t_subject_correctness_accuracy",
                     **daic_correctness_t(daic_files, left, right)})

    payload = {"schema_version": "mfcc_ttest.v1", "alpha": args.alpha, "delta": "right minus left",
               "comparisons": rows, "evidence": evidence}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(payload, args.output_dir / "report.json")

    def verdict(p: float, diff: float, left: str, right: str) -> str:
        if p >= args.alpha:
            return "No"
        return f"Yes ({right if diff > 0 else left} better)"

    lines = ["# Paired t-tests: MFCC XGBoost and Qwen3-Omni", "",
             "CV datasets: paired t-test over 5 folds on per-fold Macro-F1. Delta = right - left.", "",
             "| Dataset | Left | Right | Left (fold-mean) | Right (fold-mean) | Delta | t | p | Significant? | p (corrected) | Significant (corrected)? |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for row in rows:
        if row["dataset"] == "DAIC":
            continue
        lines.append(
            f"| {row['dataset']} | {row['left']} | {row['right']} | {row['left_fold_mean']:.3f} | {row['right_fold_mean']:.3f} | "
            f"{row['mean_diff']:+.3f} | {row['t']:.2f} | {row['p']:.3f} | {verdict(row['p'], row['mean_diff'], row['left'], row['right'])} | "
            f"{row['p_corrected']:.3f} | {verdict(row['p_corrected'], row['mean_diff'], row['left'], row['right'])} |"
        )
    lines += ["", "DAIC (one fixed split, 47 subjects): paired t-test on per-subject correctness (accuracy, not Macro-F1).", "",
              "| Left | Right | Left accuracy | Right accuracy | Delta | t | p | Significant? |", "|---|---|---|---|---|---|---|---|"]
    for row in rows:
        if row["dataset"] != "DAIC":
            continue
        lines.append(
            f"| {row['left']} | {row['right']} | {row['accuracy_left']:.3f} | {row['accuracy_right']:.3f} | {row['mean_diff']:+.3f} | "
            f"{row['t']:.2f} | {row['p']:.3f} | {verdict(row['p'], row['mean_diff'], row['left'], row['right'])} |"
        )
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
