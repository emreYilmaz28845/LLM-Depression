#!/usr/bin/env python3
"""Compute seed-1337 pooled subject-level F1 and fold-mean ± SD from per-fold
subject predictions for every workbook CV cell.

Read-only tool: it reads per-fold `predictions_subject_level.csv` files and
writes `outputs/workbook_pooled_audit.json`. It never modifies training/eval
artifacts, configs, or the registry, and it never runs train/eval.

Fold-set sources (locked by the workbook-standardization plan):
- Standalone native+EN (D3TEC/Androids/CMDC; TF/LogReg/XGB, Qwen+Gemma):
  the per-fold metrics paths recorded in the Papers three-route evidence file
  (`Papers/presentations/.../three_route_evidence.json`), which was verified
  against the local registry and workbook on 2026-09-07.
- Turkish (native + EN, all routes): the pooled-campaign provenance index
  (`outputs/turkish_pooled_qcond/.../provenance_index.json`), seed-1337 subset.
- Merged CV: per-dataset fold dirs under `outputs/symmetric_merged/.../cv/...`.
- DAIC standalone + merged final DAIC test: official single test, no pooling.

Conventions:
- main result  = pooled: all 5-fold subject predictions in one pool -> single F1
                 (INVALID counts as wrong)
- side result  = fold-mean ± fold-SD over the 5 per-fold F1 values
- seed         = 1337 (Turkish pooled uses its seed-1337 subset)
- merged CV    = per-dataset pooled, then unweighted mean over datasets

Usage:
    python tools/compute_pooled_f1.py [--check]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "workbook_pooled_audit.json"
EVIDENCE = Path(
    "/home/emre/Projects/AudioLLM/Papers/presentations/LLM_Depression_Multilingual_20260906/three_route_evidence.json"
)
TURKISH_REPORT = ROOT / "outputs" / "turkish_pooled_qcond" / "exp-turkish-pooled-qcond-clean-v1-20260903" / "production" / "report_v2"
MODS = ["audio_text", "audio_only", "text_only"]


def read_json(p: Path):
    return json.loads(p.read_text())


def _norm01(v) -> int:
    """Normalize a label/prediction cell (int 0/1 or str '0'/'1'/label text)."""
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float)):
        return int(v)
    s = str(v).strip().lower()
    if s in ("1", "depressed", "positive", "true"):
        return 1
    if s in ("0", "non-depressed", "non_depressed", "negative", "false"):
        return 0
    raise ValueError(f"cannot normalize label/prediction {v!r}")


def _is_invalid(v) -> bool:
    if isinstance(v, (int, float)):
        return int(v) == -1
    return str(v).strip().upper() in ("INVALID", "INVALID_SUBJECT", "ERROR", "-1", "")


def _binary_strict_f1(rows: list[dict]) -> tuple[float, float]:
    """Macro-F1 and positive-F1 from subject rows.

    Binary-strict: INVALID adds a false negative for its true class, never a
    correct answer (matches the repo's confusion-matrix convention).
    """
    tp = fp = fn = tn = 0
    for r in rows:
        gold = _norm01(r["label"])
        pred_raw = r["prediction"]
        if _is_invalid(pred_raw):
            # gold class never answered -> false negative for that class
            if gold == 1:
                fn += 1
            else:
                # negative-class subject with INVALID: repo counts it against
                # the negative class as an extra false negative
                fn += 1
                tn += 0  # not a true negative
            continue
        pred = _norm01(pred_raw)
        if gold == 1 and pred == 1:
            tp += 1
        elif gold == 0 and pred == 1:
            fp += 1
        elif gold == 1 and pred == 0:
            fn += 1
        else:  # gold 0, pred 0
            tn += 1
    pos = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    neg = 2 * tn / (2 * tn + fp + fn) if (2 * tn + fp + fn) else 0.0
    return (pos + neg) / 2, pos


def load_rows(p: Path) -> list[dict]:
    if p.suffix == ".csv":
        with p.open(newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    if p.suffix == ".jsonl":
        rows = []
        with p.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        return rows
    raise ValueError(f"unsupported rows file: {p}")


def subject_csv_for_metrics(metrics_path: Path) -> Path:
    """Subject CSV/JSONL next to a metrics_*.json file (workbook eval dirs)."""
    for name in ("predictions_subject_level.csv", "predictions_subject_level.jsonl"):
        cand = metrics_path.parent / name
        if cand.exists():
            return cand
    raise AssertionError(f"no subject predictions next to {metrics_path}")


def _f1_from_cm(cm: list) -> tuple[float, float]:
    """Macro/positive F1 from a (possibly 3-col) confusion matrix.

    Binary-strict: row 0 = gold Non-depressed [tn, fp, ni?], row 1 = gold
    Depressed [fn, tp, pi?]; INVALID counts as a false negative for its own
    class (matches the repo metric convention).
    """
    tn, fp = cm[0][:2]
    fn, tp = cm[1][:2]
    ni = cm[0][2] if len(cm[0]) > 2 else 0
    pi_ = cm[1][2] if len(cm[1]) > 2 else 0
    pos = 2 * tp / (2 * tp + fp + fn + pi_) if (2 * tp + fp + fn + pi_) else 0.0
    neg = 2 * tn / (2 * tn + fp + fn + ni) if (2 * tn + fp + fn + ni) else 0.0
    return (pos + neg) / 2, pos


def _cm_from_rows(rows: list[dict]) -> list:
    """3-column confusion matrix from subject rows.

    Rows: [gold Non-depressed: tn, fp, ni], [gold Depressed: fn, tp, pi].
    INVALID prediction adds to its gold class's invalid column.
    """
    cm = [[0, 0, 0], [0, 0, 0]]
    for r in rows:
        gold = _norm01(r["label"])
        pred_raw = r.get("prediction", r.get("prediction_text"))
        if _is_invalid(pred_raw):
            cm[1 if gold else 0][2] += 1  # invalid column of gold class
            continue
        pred = _norm01(pred_raw)
        if gold == 1:
            if pred == 1:
                cm[1][1] += 1  # tp
            else:
                cm[1][0] += 1  # fn
        else:
            if pred == 1:
                cm[0][1] += 1  # fp
            else:
                cm[0][0] += 1  # tn
    return cm


def _pool_stats_from_folds(metrics_paths: list[Path]) -> dict:
    """Pooled + fold-mean ± SD from per-fold evidence files.

    Each file may be either a metrics json carrying a confusion matrix, or a
    subject-rows file (predictions_subject_level.csv / .jsonl). Pooled =
    element-wise sum of the per-fold binary-strict confusion matrices, then a
    single F1 from the summed matrix (5-fold subjects in one pool). INVALID
    rows count as false negatives for their own class (repo convention).
    """
    fold_f1s: list[tuple[float, float]] = []
    summed = None
    n_subjects = 0
    for p in metrics_paths:
        if p.suffix in (".jsonl", ".csv"):
            cm = _cm_from_rows(load_rows(p))
        else:
            d = read_json(p)
            if "binary_strict_confusion_matrix" in d:
                cm = d["binary_strict_confusion_matrix"]
            elif "confusion_matrix" in d:
                cm = d["confusion_matrix"]
            else:
                # scalar-only metrics json -> fall back to sidecar predictions
                side = subject_csv_for_metrics(p)
                cm = _cm_from_rows(load_rows(side))
        fold_f1s.append(_f1_from_cm(cm))
        if summed is None:
            ncols = max(len(row) for row in cm)
            summed = [[0] * ncols for _ in cm]
        for i, row in enumerate(cm):
            for j, v in enumerate(row):
                summed[i][j] += v
        n_subjects += sum(sum(row) for row in cm)
    macro_pooled, pos_pooled = _f1_from_cm(summed)
    macro_fm = mean(x[0] for x in fold_f1s)
    pos_fm = mean(x[1] for x in fold_f1s)
    macro_sd = pstdev(x[0] for x in fold_f1s) if len(fold_f1s) > 1 else 0.0
    pos_sd = pstdev(x[1] for x in fold_f1s) if len(fold_f1s) > 1 else 0.0
    return {
        "macro_pooled": round(macro_pooled, 6), "positive_pooled": round(pos_pooled, 6),
        "macro_foldmean": round(macro_fm, 6), "macro_foldsd": round(macro_sd, 6),
        "positive_foldmean": round(pos_fm, 6), "positive_foldsd": round(pos_sd, 6),
        "n_subjects": n_subjects, "n_folds": len(fold_f1s),
        "sources": [str(p) for p in metrics_paths],
    }


def _folder_stats(dirs: list[Path]) -> dict:
    """Pooled + fold-mean from subject CSVs found under each fold dir."""
    metrics = []
    for d in dirs:
        for pat in (
            "best_model/standalone_eval*/metrics_original_teacher_forced.json",
            "eval/best_validation/metrics_original_teacher_forced.json",
            "logreg_raw/metrics.json",
            "xgb_raw/metrics.json",
            "xgb_optuna100_harmonized_v1/metrics.json",
        ):
            hits = sorted(d.glob(pat))
            if hits:
                metrics.append(hits[0])
                break
        else:
            raise AssertionError(f"no metrics under {d}")
    return _pool_stats_from_folds(metrics)


def _evidence_metrics_path(rec: dict) -> list[Path]:
    """Extract per-fold metrics paths from a three_route_evidence record."""
    paths = []
    for fe in rec.get("fold_evidence", []):
        art = fe.get("evaluation_metrics_artifact")
        if art:
            p = ROOT / str(art["path"])
        else:
            p = Path(fe.get("metrics_path", ""))
        if p.exists():
            paths.append(p)
    # Turkish records use provenance keys with metrics artifact paths
    if not paths and "sources" in rec:
        for s in rec.get("sources", []):
            p = Path(s)
            if p.exists() and p.suffix == ".json":
                paths.append(p)
    return paths


def evidence_records() -> list[dict]:
    if not EVIDENCE.exists():
        return []
    return read_json(EVIDENCE).get("records", [])


# --------------------------------------------------------------------------- standalone native+EN
def standalone_rows() -> list[dict]:
    out = []
    for rec in evidence_records():
        ds = rec["dataset"]
        if rec["condition"] == "turkish_pooled":
            continue
        mod, cond, model, route = rec["modality"], rec["condition"], rec["model"], rec["route"]
        if cond == "english" and mod == "audio_only":
            continue  # workbook English rows are A+T and Text-only only
        metrics = _evidence_metrics_path(rec)
        if not metrics:
            out.append({**{k: rec.get(k) for k in ("dataset", "modality", "condition", "model", "route")},
                        "error": "no local metrics paths in evidence record"})
            continue
        stats = _pool_stats_from_folds(metrics)
        if ds == "daic":
            # official single test: no cross-fold pooling; keep the single value
            stats["macro_pooled"] = stats["macro_foldmean"]  # single fold
            stats["positive_pooled"] = stats["positive_foldmean"]
            stats["macro_foldsd"] = 0.0
            stats["positive_foldsd"] = 0.0
            stats["official_test"] = True
        stats.update(dataset=ds, modality=mod, condition=cond, model=model, route=route,
                     cell_group="standalone")
        out.append(stats)
    return out


# --------------------------------------------------------------------------- merged CV
# Qwen merged campaign lives under outputs/symmetric_merged/harmonized_v1/<mod>/<campaign>;
# Gemma merged under outputs/symmetric_merged/gemma4/harmonized_v1/<mod>/<campaign>.
MERGED_ROOTS = {
    "qwen": ROOT / "outputs" / "symmetric_merged" / "harmonized_v1",
    "gemma4": ROOT / "outputs" / "symmetric_merged" / "gemma4" / "harmonized_v1",
}
MERGED_OPTUNA_ROOTS = {
    "qwen": ROOT / "output_model" / "harmonized_v1_merged_optuna100",
    "gemma4": ROOT / "output_model" / "harmonized_v1_gemma4_merged_optuna100",
}
MERGED_DATASETS = ["androids_interview", "cmdc", "d3tec", "daic", "turkish"]


def _pick_merged_optuna_run(root: Path, mod: str) -> Path | None:
    """Pick the merged-optuna CV run dir (skip 'final' DAIC test)."""
    mod_root = root / mod
    if not mod_root.exists():
        return None
    for cand in sorted(mod_root.glob("*")):
        if "_cv" in cand.name and (cand / "fold_0").exists():
            return cand
    return None


def _pooled_by_dataset(files_by_fold: list[Path]) -> tuple[list[float], list[float], list[float]]:
    """Pooled + fold-mean macro-F1 per dataset over the given per-fold files.

    Each file carries a dataset column (merged eval). Returns (per-dataset
    pooled macro, per-dataset fold-mean macro, per-dataset fold-SD) lists.
    """
    ds_pooled = []
    ds_foldmeans = []
    ds_foldsds = []
    for dsk in MERGED_DATASETS:
        summed = None
        per_fold = []
        for f in files_by_fold:
            rows = load_rows(f)
            ds_rows = [r for r in rows if r.get("dataset") == dsk]
            if not ds_rows:
                continue
            cm = _cm_from_rows(ds_rows)
            per_fold.append(cm)
            if summed is None:
                summed = [[0] * 3 for _ in cm]
            for i in range(2):
                for j in range(3):
                    summed[i][j] += cm[i][j]
        if summed is None or len(per_fold) < 5:
            continue
        ds_pooled.append(_f1_from_cm(summed)[0])
        fold_f1s = [_f1_from_cm(cm)[0] for cm in per_fold]
        ds_foldmeans.append(mean(fold_f1s))
        ds_foldsds.append(pstdev(fold_f1s) if len(fold_f1s) > 1 else 0.0)
    return ds_pooled, ds_foldmeans, ds_foldsds


def _pick_campaign(root: Path, mod: str) -> Path | None:
    mod_root = root / mod
    if not mod_root.exists():
        return None
    for cand in sorted(mod_root.glob("*")):
        if "preflight" not in cand.name and "smoke" not in cand.name and (cand / "cv").exists():
            return cand
    return None


def _per_dataset_merged_folds(campaign: Path, owner_model: str) -> dict[str, dict[str, list[Path]]]:
    """Return model/head -> dataset -> list of 5 per-fold subject files.

    TF rows: cv/fold_<n>/<model>/<dataset>/predictions_subject_level.csv
    Heads:   cv/fold_<n>/heads/<head>/predictions_subject_level.csv (dataset col)
    owner_model limits the TF model dirs collected (the campaign root owns one
    backbone; the sibling model dirs are shared evaluation copies).
    """
    out: dict[str, dict[str, list[Path]]] = {}
    for fold_dir in sorted((campaign / "cv").glob("fold_*")):
        mdir = fold_dir / owner_model
        if mdir.is_dir():
            for dsk in MERGED_DATASETS:
                csvp = mdir / dsk / "predictions_subject_level.csv"
                if csvp.exists():
                    out.setdefault(owner_model, {}).setdefault(dsk, []).append(csvp)
        heads = fold_dir / "heads"
        if heads.is_dir():
            for head in ("logreg", "xgb_fixed"):
                csvp = heads / head / "predictions_subject_level.csv"
                if csvp.exists():
                    # single file with a dataset column; record per dataset below
                    out.setdefault(f"head_{head}", {}).setdefault("_all", []).append(csvp)
    return out


def merged_rows() -> list[dict]:
    """Merged CV: per-dataset pooled F1, then unweighted mean over datasets."""
    out = []
    for model, root in MERGED_ROOTS.items():
        for mod in MODS:
            campaign = _pick_campaign(root, mod)
            if not campaign:
                continue
            foldmap = _per_dataset_merged_folds(campaign, owner_model=model)
            # TF rows (this campaign root's own model dir)
            per_ds = foldmap.get(model)
            if per_ds:
                ds_pooled = []
                ds_foldmeans = []
                ds_foldsds = []
                ds_pos_pooled = []
                ds_pos_foldmeans = []
                for dsk in MERGED_DATASETS:
                    files = per_ds.get(dsk, [])
                    if len(files) != 5:
                        continue
                    try:
                        stats = _pool_stats_from_folds(files)
                    except AssertionError:
                        continue
                    ds_pooled.append(stats["macro_pooled"])
                    ds_foldmeans.append(stats["macro_foldmean"])
                    ds_foldsds.append(stats["macro_foldsd"])
                    ds_pos_pooled.append(stats["positive_pooled"])
                    ds_pos_foldmeans.append(stats["positive_foldmean"])
                if len(ds_pooled) == 5:
                    out.append({"dataset": "merged", "modality": mod, "condition": "native",
                                "model": model, "route": "teacher_forced", "cell_group": "merged_cv",
                                "macro_pooled": round(mean(ds_pooled), 6),
                                "macro_foldmean": round(mean(ds_foldmeans), 6),
                                "macro_foldsd": round(mean(ds_foldsds), 6),
                                "positive_pooled": round(mean(ds_pos_pooled), 6),
                                "positive_foldmean": round(mean(ds_pos_foldmeans), 6),
                                "macro_per_dataset_pooled": [round(x, 6) for x in ds_pooled],
                                "sources": [str(f) for ds_files in per_ds.values() for f in ds_files],
                                "n_datasets": 5})
            # Head rows: logreg/xgb_fixed per dataset via the dataset column
            for head in ("logreg", "xgb_fixed"):
                files = foldmap.get(f"head_{head}", {}).get("_all", [])
                if len(files) != 5:
                    continue
                ds_pooled, ds_foldmeans, ds_foldsds = _pooled_by_dataset(files)
                if len(ds_pooled) == 5:
                    out.append({"dataset": "merged", "modality": mod, "condition": "native",
                                "model": model, "route": "xgb_fixed" if head == "xgb_fixed" else "logreg",
                                "cell_group": "merged_cv",
                                "macro_pooled": round(mean(ds_pooled), 6),
                                "macro_foldmean": round(mean(ds_foldmeans), 6),
                                "macro_foldsd": round(mean(ds_foldsds), 6),
                                "macro_per_dataset_pooled": [round(x, 6) for x in ds_pooled],
                                "n_datasets": 5})
            # Optuna-100 merged XGB lives under output_model/..._merged_optuna100
            orun = _pick_merged_optuna_run(MERGED_OPTUNA_ROOTS[model], mod)
            if orun:
                files = sorted((orun).glob("fold_*/xgb_optuna100_harmonized_v1/predictions_subject_level.csv"))
                if len(files) == 5:
                    ds_pooled, ds_foldmeans, ds_foldsds = _pooled_by_dataset(files)
                    if len(ds_pooled) == 5:
                        out.append({"dataset": "merged", "modality": mod, "condition": "native",
                                    "model": model, "route": "xgb_optuna100", "cell_group": "merged_cv",
                                    "macro_pooled": round(mean(ds_pooled), 6),
                                    "macro_foldmean": round(mean(ds_foldmeans), 6),
                                    "macro_foldsd": round(mean(ds_foldsds), 6),
                                    "macro_per_dataset_pooled": [round(x, 6) for x in ds_pooled],
                                    "n_datasets": 5,
                                    "sources": [str(f) for f in files]})
    return out


# --------------------------------------------------------------------------- Turkish pooled
WORKTREE = Path("/home/emre/worktrees/LLM-Depression-exp-turkish-pooled-qcond-clean-v1")


def _resolve_local(p: str) -> Path:
    """Map a provenance artifact path (worktree or GPFS) to a local readable path."""
    p = p.removeprefix("/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression/")
    # worktree-relative output_model/... -> worktree root
    if p.startswith("output_model/") or p.startswith("outputs/"):
        cand = WORKTREE / p
        if cand.exists():
            return cand
    cand = ROOT / p
    if cand.exists():
        return cand
    return cand  # caller asserts


def _backend_route(backend: str) -> str:
    if "teacher_forced" in backend:
        return "teacher_forced"
    if "logreg" in backend:
        return "logreg"
    if "xgb" in backend or "optuna" in backend:
        return "xgb_optuna100"
    return backend


def turkish_rows() -> list[dict]:
    out = []
    idx = TURKISH_REPORT / "provenance_index.json"
    if not idx.exists():
        return out
    pi = read_json(idx)
    # Group seed-1337 records by (modality, transcript_condition, backend-class, backbone).
    by_key: dict[tuple, list[Path]] = {}
    for ref in pi.values():
        if ref.get("seed") != 1337:
            continue
        # metrics artifact carries the binary-strict confusion matrix -> use it
        ma = ref.get("evaluation_metrics_artifact") or {}
        path = _resolve_local(str(ma.get("path", "")))
        if not path.exists():
            pa = ref.get("evaluation_predictions_artifact") or {}
            path = _resolve_local(str(pa.get("path", "")))
        if not path.exists():
            continue
        tcond = ref.get("transcript_condition") or "not_applicable"
        # not_applicable -> native (audio-only has no transcript condition)
        cond = "native" if tcond in ("not_applicable", "native", "turkish") else tcond
        meta = (ref.get("modality"), cond, _backend_route(str(ref.get("backend", ""))),
                str(ref.get("backbone", "")).lower().replace(" ", ""))
        by_key.setdefault(meta, []).append(path)
    for (mod, cond, route, model), paths in by_key.items():
        if len(paths) < 4:
            continue
        stats = _pool_stats_from_folds(sorted(paths))
        stats.update(dataset="turkish", modality=mod, condition=cond, model=model, route=route,
                     cell_group="turkish_pooled")
        out.append(stats)
    return out


def build_rows() -> list[dict]:
    rows = standalone_rows()
    rows += turkish_rows()
    rows += merged_rows()
    return rows


def _check_against_workbook(rows: list[dict]) -> tuple[int, list[str]]:
    """Compare computed pooled values against the current workbook 'Qwen vs Gemma'
    cells for every pooled CV cell (DAIC official test and the Turkish pooled
    campaign rows are excluded: DAIC has no CV and Turkish cells are compared
    against the campaign report elsewhere). Returns (n_checked, mismatches)."""
    import openpyxl

    wb = openpyxl.load_workbook(ROOT / "depression_results_clean.xlsx", data_only=True)
    ws = wb["Qwen vs Gemma"]
    ds_key = {"D3TEC": "d3tec", "Androids Interview": "androids_interview",
              "CMDC": "cmdc", "Turkish": "turkish", "DAIC": "daic"}
    mod_key = {"Audio + Text": "audio_text", "Audio only": "audio_only", "Text only": "text_only"}
    route_key = {"Teacher-forced": "teacher_forced", "LogReg head": "logreg", "XGBoost": "xgb_optuna100"}

    def parse(v):
        try:
            return [float(x) for x in str(v).split(" / ")]
        except Exception:
            return None

    wbmap = {}
    for r in ws.iter_rows(min_row=8, values_only=True):
        exp, ds, mod, method, qwen, gemma = r[0], r[1], r[2], r[3], r[4], r[5]
        if method not in route_key or exp not in ("Standalone", "English") or ds == "DAIC":
            continue
        key = (ds_key[ds], mod_key[mod], "english" if exp == "English" else "native")
        wbmap[(key[0], key[1], key[2], "qwen", route_key[method])] = parse(qwen)
        wbmap[(key[0], key[1], key[2], "gemma4", route_key[method])] = parse(gemma)

    mismatches = []
    checked = 0
    for r in rows:
        if "error" in r or r.get("cell_group") != "standalone":
            continue
        if r["dataset"] in ("turkish", "daic"):
            continue  # turkish = pooled campaign report cells; daic = official test
        key = (r["dataset"], r["modality"], r["condition"], r["model"], r["route"])
        wv = wbmap.get(key)
        if not wv:
            continue
        calc = r["macro_pooled"]
        checked += 1
        if abs(calc - wv[0]) > 0.0003:
            mismatches.append(f"{key}: computed pooled {calc:.4f} vs workbook {wv[0]:.4f}")
    return checked, mismatches


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--check", action="store_true",
                        help="compare computed fold-means against current workbook values")
    args = parser.parse_args()

    rows = build_rows()
    payload = {"convention": "seed1337_pooled", "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    n_ok = sum(1 for r in rows if "error" not in r)
    n_err = len(rows) - n_ok
    print(f"Wrote {len(rows)} rows ({n_ok} ok, {n_err} errors) to {args.out}")
    for r in rows:
        if "error" in r:
            print("  ERR", r.get("dataset"), r.get("modality"), r.get("condition"), r.get("model"),
                  r.get("route"), "->", r["error"][:90])
    if args.check:
        checked, mismatches = _check_against_workbook(rows)
        print(f"--check: {checked} cells compared; {len(mismatches)} mismatches")
        for m in mismatches[:20]:
            print("  MISMATCH", m)
        return 1 if (n_err or mismatches) else 0
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())
