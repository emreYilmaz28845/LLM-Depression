"""Evidence test for the UAR columns of the 'Qwen vs Gemma' sheet.

Every UAR value in scripts/build_clean_workbook.py's *_UAR tables and in the
MERGED_CV_AVG_UAR / MERGED_FINAL_UAR runtime dicts is recomputed here from the
local artifacts the builder's comments point to, then compared with the stored
value. UAR = unweighted average recall = mean of the two class recalls, with an
invalid output counted as a false negative for its true class; it equals the
saved strict ``macro_recall`` key.

Aggregation conventions follow the paired Macro-F1/Positive-F1 tables:
  DAIC official test       -> single fold_0 value
  CMDC / Turkish           -> 5-fold mean of per-fold subject-level metrics
  D3TEC / Androids TF      -> unweighted 5-fold mean of subject-level metrics
  Heads (all datasets)     -> 5-fold mean of per-fold variant_summary.json
  Optuna-100               -> fold-mean of per-fold metrics.json
  Merged CV                -> mean over five per-dataset fold-means
  Merged final             -> DAIC official-test single value
  Turkish pooled           -> mean of the five per-fold macro_recall values
                              recorded in the pooled report provenance index

Turkish standalone/English literal cells stay None in the builder by design (the
current Turkish headline comes from the pooled report); they are covered by
``test_turkish_pooled_uar_lookup`` instead.
"""

from __future__ import annotations

import importlib.util as _ilu
import json
import os
from pathlib import Path
from statistics import mean

import pytest

CODE_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("LLMDEP_EVIDENCE_ROOT", CODE_ROOT))

_spec = _ilu.spec_from_file_location("build_clean_workbook", CODE_ROOT / "scripts/build_clean_workbook.py")
build_clean_workbook = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(build_clean_workbook)

MOD_DIR = {"Audio + Text": "audio_text", "Audio only": "audio_only", "Text only": "text_only"}
DS_DIR = {
    "DAIC": "daic", "CMDC": "cmdc", "Turkish": "turkish_t17_qwen3asr",
    "D3TEC": "d3tec", "Androids Interview": "androids",
}
RUN_DS = {
    "DAIC": "daic", "CMDC": "cmdc", "Turkish": "turkish",
    "D3TEC": "d3tec", "Androids Interview": "androids_interview",
}
HEAD_DS = {
    "DAIC": "daic", "CMDC": "cmdc", "Turkish": "turkish",
    "D3TEC": "d3tec", "Androids Interview": "androids_interview",
}
DATASETS = list(DS_DIR)
EN_DATASETS = ["D3TEC", "Androids Interview", "CMDC", "Turkish"]


def _uar_from_cm(cm):
    ni = cm[0][2] if len(cm[0]) > 2 else 0
    pi = cm[1][2] if len(cm[1]) > 2 else 0
    tn, fp = cm[0][:2]
    fn, tp = cm[1][:2]
    pos = tp / (tp + fn + pi) if (tp + fn + pi) else 0.0
    neg = tn / (tn + fp + ni) if (tn + fp + ni) else 0.0
    return (pos + neg) / 2


def _uar(payload):
    """Strict UAR of a metrics payload: saved macro_recall, else its matrix."""
    if payload.get("macro_recall") is not None:
        return payload["macro_recall"]
    return _uar_from_cm(payload["binary_strict_confusion_matrix"])


def _fold_files(run: Path, pattern: str) -> list[Path]:
    """Per-fold files under a run, deduped by fold index; [] unless five folds."""
    found: dict[int, Path] = {}
    for p in sorted(run.glob(pattern)):
        parts = p.parts
        idx = next((i for i, x in enumerate(parts) if x.startswith("fold_")), None)
        if idx is None:
            continue
        found.setdefault(int(parts[idx].split("_")[1]), p)
    return [found[f] for f in sorted(found)] if len(found) == 5 else []


def _ordered_runs(base: Path, contains: str) -> list[Path]:
    """Non-smoke runs for a family, retry runs (``_r1``, ``_r2``, ...) first:
    retries replace failed attempts and carry the reported values."""
    import re

    def rank(run: Path):
        return (0 if re.search(r"_r\d+$", run.name) else 1, run.name)

    assert base.is_dir(), f"missing artifact root {base}"
    runs = [r for r in base.glob(f"*{contains}*") if "smoke" not in r.name and "preflight" not in r.name]
    return sorted(runs, key=rank)


def _family_folds(base: Path, contains: str, pattern: str, folds: int = 5) -> list[Path]:
    """Per-fold files across the family's runs, deduped by fold index with retry
    runs taking precedence (retries split folds across run directories)."""
    found: dict[int, Path] = {}
    for run in _ordered_runs(base, contains):
        for p in sorted(run.glob(pattern)):
            parts = p.parts
            idx = next((i for i, x in enumerate(parts) if x.startswith("fold_")), None)
            if idx is None:
                continue
            found.setdefault(int(parts[idx].split("_")[1]), p)
    assert len(found) == folds, f"{base} *{contains}* ({pattern}): expected {folds} folds, found {sorted(found)}"
    return [found[f] for f in sorted(found)]


def _family_fold_mean(base: Path, contains: str, pattern: str, folds: int = 5) -> float:
    return mean(_uar(json.loads(p.read_text())) for p in _family_folds(base, contains, pattern, folds))


def _discover_heads(base: Path, contains: str) -> Path:
    """Run directory of a hidden-head family; heads are single-run per cell."""
    for run in _ordered_runs(base, contains):
        if _fold_files(run, "fold_*/variant_summary.json"):
            return run
    raise AssertionError(f"no head run with five folds under {base} matching *{contains}*")


def _fold_mean(run: Path, pattern: str) -> float:
    files = _fold_files(run, pattern)
    assert files, f"{run}: no five-fold layout for {pattern}"
    return mean(_uar(json.loads(p.read_text())) for p in files)


def _head_variant_mean(run: Path, variant: str) -> float:
    files = _fold_files(run, "fold_*/variant_summary.json")
    assert files, f"{run}: no five-fold variant summaries"
    vals = []
    for p in files:
        payload = json.loads(p.read_text())
        vals.append(next(v["macro_recall"] for v in payload if v["variant"] == variant))
    return mean(vals)


def _skip_turkish(dataset):
    if dataset == "Turkish":
        pytest.skip("Turkish literals are pooled-driven; covered by the pooled lookup test")


def _pick_run(base: Path, contains: str) -> Path:
    runs = _ordered_runs(base, contains)
    assert runs, f"no run for {base}/*{contains}*"
    return runs[0]


def _single_fold_uar(base: Path, contains: str, pattern: str) -> float:
    """Official-test cells: the retry-preferred run's single fold_0 metric."""
    run = _pick_run(base, contains)
    files = sorted(run.glob(pattern))
    assert files, f"{run}: no {pattern}"
    return _uar(json.loads(files[0].read_text()))


# ------------------------------------------------------------------ standalone
@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.parametrize("modality", list(MOD_DIR))
def test_standalone_qwen_tf_uar(dataset, modality):
    """The legacy teacher-forced UAR anchors stay reproducible from the artifacts."""
    _skip_turkish(dataset)
    m = MOD_DIR[modality]
    folder = "eval/best_validation" if dataset in ("CMDC", "Turkish") else "best_model/standalone_eval"
    base = ROOT / f"output_model/harmonized_v1/{m}/{DS_DIR[dataset]}"
    rel = f"fold_*/{folder}*/metrics_original_teacher_forced.json"
    if dataset == "DAIC":
        got = _single_fold_uar(base, f"{RUN_DS[dataset]}_{m}", f"fold_0/{folder}/metrics_original_teacher_forced.json")
    else:
        got = _family_fold_mean(base, f"{RUN_DS[dataset]}_{m}", rel)
    assert got == pytest.approx(build_clean_workbook.STANDALONE_QWEN_TF_LEGACY_UAR[(dataset, modality)], abs=1e-5)


@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.parametrize("modality", list(MOD_DIR))
def test_standalone_qwen_logreg_uar(dataset, modality):
    _skip_turkish(dataset)
    m = MOD_DIR[modality]
    base = ROOT / f"outputs/hidden_classifiers/harmonized_v1/{HEAD_DS[dataset]}"
    if dataset == "DAIC":
        run = _pick_run(base, f"{HEAD_DS[dataset]}_{m}")
        payload = json.loads((run / "fold_0/variant_summary.json").read_text())
        got = next(v["macro_recall"] for v in payload if v["variant"] == "logreg_raw")
    else:
        run = _discover_heads(base, f"{HEAD_DS[dataset]}_{m}")
        got = _head_variant_mean(run, "logreg_raw")
    assert got == pytest.approx(build_clean_workbook.STANDALONE_HEADS_UAR[(dataset, modality)][0], abs=1e-6)


@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.parametrize("modality", list(MOD_DIR))
def test_optuna_native_uar(dataset, modality):
    _skip_turkish(dataset)
    m = MOD_DIR[modality]
    ds_dir = HEAD_DS[dataset]
    rel = "fold_*/xgb_optuna100_harmonized_v1/metrics.json"
    folds = 1 if dataset == "DAIC" else 5
    q = _family_fold_mean(ROOT / f"output_model/harmonized_v1_optuna100/{m}/{ds_dir}", f"{ds_dir}_{m}", rel, folds)
    g = _family_fold_mean(ROOT / f"output_model/harmonized_v1_gemma4_optuna100/{m}/{ds_dir}", f"{ds_dir}_{m}", rel, folds)
    assert q == pytest.approx(build_clean_workbook.QWEN_OPTUNA_UAR[(dataset, modality)], abs=1e-5)
    assert g == pytest.approx(build_clean_workbook.GEMMA_OPTUNA_UAR[(dataset, modality)], abs=1e-5)


@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.parametrize("modality", list(MOD_DIR))
def test_gemma_native_tf_uar(dataset, modality):
    _skip_turkish(dataset)
    m = MOD_DIR[modality]
    folder = "eval/best_validation" if dataset in ("CMDC", "Turkish") else "best_model/standalone_eval"
    base = ROOT / f"output_model/harmonized_v1_gemma4/{m}/{DS_DIR[dataset]}"
    rel = f"fold_*/{folder}*/metrics_original_teacher_forced.json"
    if dataset == "DAIC":
        got = _single_fold_uar(base, f"{RUN_DS[dataset]}_{m}", f"fold_0/{folder}/metrics_original_teacher_forced.json")
    else:
        got = _family_fold_mean(base, f"{RUN_DS[dataset]}_{m}", rel)
    assert got == pytest.approx(build_clean_workbook.GEMMA_NATIVE_TF_UAR[(dataset, modality)], abs=1e-5)


@pytest.mark.parametrize("dataset", ["D3TEC", "Androids Interview", "CMDC"])
@pytest.mark.parametrize("modality", list(MOD_DIR))
def test_gemma_native_lr_uar(dataset, modality):
    m = MOD_DIR[modality]
    report = ROOT / f"outputs/experiment_reports/gemma4_harmonized/native_lr/{HEAD_DS[dataset]}_{m}.json"
    assert report.is_file(), report
    folds = json.loads(report.read_text())["folds"]
    assert len(folds) == 5, report.name
    paths = [Path(item["metrics_path"]) for item in folds]
    if not all(p.is_file() for p in paths):
        pytest.skip(f"metrics not local for {report.name}")
    got = mean(_uar(json.loads(p.read_text())) for p in paths)
    assert got == pytest.approx(build_clean_workbook.GEMMA_NATIVE_LR_UAR[(dataset, modality)], abs=1e-5)


# ------------------------------------------------------------------ English
@pytest.mark.parametrize("dataset", EN_DATASETS)
@pytest.mark.parametrize("modality", ["Audio + Text", "Text only"])
def test_en_qwen_tf_uar(dataset, modality):
    _skip_turkish(dataset)
    m = MOD_DIR[modality]
    folder = "eval/best_validation" if dataset == "CMDC" else "best_model/standalone_eval"
    base = ROOT / f"output_model/harmonized_v1_en/{m}/{DS_DIR[dataset]}"
    got = _family_fold_mean(base, f"{RUN_DS[dataset]}_{m}", f"fold_*/{folder}*/metrics_original_teacher_forced.json")
    assert got == pytest.approx(build_clean_workbook.EN_TF_UAR[(dataset, modality)][0], abs=1e-5)


@pytest.mark.parametrize("dataset", EN_DATASETS)
@pytest.mark.parametrize("modality", ["Audio + Text", "Text only"])
def test_en_qwen_logreg_uar(dataset, modality):
    _skip_turkish(dataset)
    m = MOD_DIR[modality]
    base = ROOT / f"outputs/hidden_classifiers/harmonized_v1_en/{HEAD_DS[dataset]}"
    run = _discover_heads(base, f"{HEAD_DS[dataset]}_{m}")
    got = _head_variant_mean(run, "logreg_raw")
    assert got == pytest.approx(build_clean_workbook.EN_LR_UAR[(dataset, modality)][0], abs=1e-6)


@pytest.mark.parametrize("dataset", EN_DATASETS)
@pytest.mark.parametrize("modality", ["Audio + Text", "Text only"])
def test_en_optuna_uar(dataset, modality):
    _skip_turkish(dataset)
    m = MOD_DIR[modality]
    rel = "fold_*/xgb_optuna100_harmonized_v1/metrics.json"
    q = _family_fold_mean(ROOT / f"output_model/harmonized_v1_en_optuna100/{m}/{HEAD_DS[dataset]}",
                          f"{HEAD_DS[dataset]}_{m}", rel)
    g = _family_fold_mean(ROOT / f"output_model/harmonized_v1_en_gemma4_optuna100/{m}/{HEAD_DS[dataset]}",
                          f"{HEAD_DS[dataset]}_{m}", rel)
    assert q == pytest.approx(build_clean_workbook.EN_XGB_UAR[(dataset, modality)][0], abs=1e-5)
    assert g == pytest.approx(build_clean_workbook.EN_XGB_UAR[(dataset, modality)][1], abs=1e-5)


@pytest.mark.parametrize("dataset", EN_DATASETS)
@pytest.mark.parametrize("modality", ["Audio + Text", "Text only"])
def test_en_gemma_tf_uar(dataset, modality):
    _skip_turkish(dataset)
    m = MOD_DIR[modality]
    folder = "eval/best_validation" if dataset == "CMDC" else "best_model/standalone_eval"
    base = ROOT / f"output_model/harmonized_v1_en_gemma4/{m}/{DS_DIR[dataset]}"
    got = _family_fold_mean(base, f"{RUN_DS[dataset]}_{m}", f"fold_*/{folder}*/metrics_original_teacher_forced.json")
    assert got == pytest.approx(build_clean_workbook.EN_TF_UAR[(dataset, modality)][1], abs=1e-5)


@pytest.mark.parametrize("dataset", ["D3TEC", "Androids Interview", "CMDC"])
@pytest.mark.parametrize("modality", ["Audio + Text", "Text only"])
def test_en_gemma_lr_uar(dataset, modality):
    m = MOD_DIR[modality]
    report = ROOT / f"outputs/experiment_reports/gemma4_harmonized/english_lr/{HEAD_DS[dataset]}_{m}.json"
    assert report.is_file(), report
    folds = json.loads(report.read_text())["folds"]
    assert len(folds) == 5, report.name
    paths = [Path(item["metrics_path"]) for item in folds]
    if not all(p.is_file() for p in paths):
        pytest.skip(f"metrics not local for {report.name}")
    got = mean(_uar(json.loads(p.read_text())) for p in paths)
    assert got == pytest.approx(build_clean_workbook.EN_LR_UAR[(dataset, modality)][1], abs=1e-5)


# ------------------------------------------------------------------ merged
@pytest.mark.parametrize("model", ["qwen", "gemma4"])
@pytest.mark.parametrize("modality", list(MOD_DIR))
def test_merged_uar(model, modality):
    m = MOD_DIR[modality]
    build_clean_workbook._load_merged_per_dataset_foldmeans()
    per_ds = [build_clean_workbook.MERGED_PER_DATASET_FOLDMEAN[(m, model, "teacher_forced", ds)][2]
              for ds in build_clean_workbook.DATASET_LABELS]
    assert all(v is not None and 0.0 <= v <= 1.0 for v in per_ds)
    assert build_clean_workbook.MERGED_CV_AVG_UAR[(m, model, "teacher_forced")] == pytest.approx(mean(per_ds), abs=1e-12)
    root = (ROOT / "outputs/symmetric_merged/harmonized_v1" if model == "qwen"
            else ROOT / "outputs/symmetric_merged/gemma4/harmonized_v1") / m
    summaries = sorted(root.glob(f"*/final/fold_0/{model}/summary.json"))
    assert summaries, f"no merged final summary under {root}"
    payload = json.loads(summaries[0].read_text())
    metrics = payload["daic"]["metrics"] if "daic" in payload else payload["metrics"]
    assert build_clean_workbook.MERGED_FINAL_UAR[(m, model, "teacher_forced")] == pytest.approx(_uar(metrics), abs=1e-9)


# ------------------------------------------------------------------ Turkish pooled
def test_turkish_pooled_uar_lookup():
    report_path = build_clean_workbook.TURKISH_POOLED_MIXED_REPORT_PATH
    if not report_path.is_file():
        pytest.skip("pooled report not present")
    build_clean_workbook._build_turkish_pooled_lookup(report_path)
    index = json.loads((report_path.parent / "provenance_index.json").read_text())
    report = json.loads(report_path.read_text())
    checked = 0
    for item in report["tables"]["seed_results"]:
        if item.get("seed") != 1337:
            continue
        folds = []
        for key in str(item["provenance_keys"]).split(","):
            artifact = index[key.strip()]["evaluation_metrics_artifact"]
            folds.append(_uar(json.loads(Path(artifact["path"]).read_text())))
        assert len(folds) == 5
        modality = {"audio_only": "Audio only", "text_only": "Text only", "audio_text": "Audio + Text"}[str(item["modality"])]
        transcript = str(item.get("transcript_condition") or "not_applicable")
        key = (item["model"], modality, transcript, str(item["route"]))
        assert build_clean_workbook.TURKISH_POOLED_MIXED_LOOKUP[key][2] == pytest.approx(mean(folds), abs=1e-9)
        checked += 1
    assert checked == 30
