"""Evidence test for the paired 'Macro-F1 / Positive-F1' workbook cells.

Every positive-F1 value in scripts/build_clean_workbook.py's *_POSF1 tables is
recomputed here from the local artifacts the builder's comments point to, then
compared against the table values. This is the proof that the paired cells
match the same evidence as the macro-F1 they sit next to.

Aggregation conventions (locked by the builder's macro tables):
  DAIC official test       -> single fold_0 value
  CMDC / Turkish           -> 5-fold mean of per-fold subject-level metrics
  D3TEC / Androids TF      -> pooled 5-fold subject-level (concatenated CSVs)
  Heads (all datasets)     -> 5-fold mean of per-fold variant_summary.json
  Optuna-100               -> fold-mean of per-fold evaluations.json
  Merged CV                -> mean over five per-dataset fold-means
  Merged final             -> DAIC official-test single value
"""

from __future__ import annotations

import csv
import glob
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

import importlib.util as _ilu

_BUILDER_PATH = ROOT / "scripts/build_clean_workbook.py"
_spec = _ilu.spec_from_file_location("build_clean_workbook", _BUILDER_PATH)
build_clean_workbook = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(build_clean_workbook)


def _mean(xs):
    xs = [x for x in xs if x is not None]
    assert xs, "no values to average"
    return sum(xs) / len(xs)


def _read_csv(p):
    with open(p, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _f1s_from_preds(rows):
    cm = [[0, 0], [0, 0]]
    for r in rows:
        y = int(float(r["label"]))
        p = int(float(r["prediction"]))
        cm[y][p] += 1
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]

    def f1(prec, rec):
        return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    prec_n = tn / (tn + fn) if (tn + fn) else 0.0
    rec_n = tn / (tn + fp) if (tn + fp) else 0.0
    prec_p = tp / (tp + fp) if (tp + fp) else 0.0
    rec_p = tp / (tp + fn) if (tp + fn) else 0.0
    return (f1(prec_n, rec_n) + f1(prec_p, rec_p)) / 2, f1(prec_p, rec_p), cm


def _pick_run(base: Path, pattern: str) -> Path:
    runs = sorted(base.glob(pattern))
    assert runs, f"no run for {base}/{pattern}"
    r1 = [r for r in runs if r.name.endswith("_r1")]
    return r1[-1] if r1 else runs[-1]


def _fold_metrics(base: Path, run_ds: str, m: str, folder: str) -> list[dict]:
    """Per-fold metrics across retry layouts (standalone_eval* variants, folds
    split across base and _r1 run dirs), deduped per fold."""
    found: dict[int, Path] = {}
    pat = f"{base}/{run_ds}/{m}/harmonized_v1_harmonized_v1_prod_20260809T171705Z_d1e8130b_{run_ds}_{m}*/fold_*/{folder}/metrics_original_teacher_forced.json"
    for p in sorted(glob.glob(str(ROOT / pat))):
        parts = p.split("/")
        fold_idx = next(i for i, x in enumerate(parts) if x.startswith("fold_"))
        fold = int(parts[fold_idx].split("_")[1])
        found.setdefault(fold, Path(p))
    assert len(found) == 5, f"expected 5 folds, found {sorted(found)}"
    return [json.loads(found[f].read_text()) for f in range(5)]


def _gemma_fold_metrics(base: str, m: str, ds_dir: str, run_pat: str, folder: str) -> list[dict]:
    found: dict[int, Path] = {}
    pat = f"{base}/{m}/{ds_dir}/{run_pat}*/fold_*/{folder}*/metrics_original_teacher_forced.json"
    for p in sorted(glob.glob(str(ROOT / pat))):
        parts = p.split("/")
        fold_idx = next(i for i, x in enumerate(parts) if x.startswith("fold_"))
        fold = int(parts[fold_idx].split("_")[1])
        found.setdefault(fold, Path(p))
    return [json.loads(found[f].read_text()) for f in sorted(found)]


def _optuna_pos(base: str, ds_dir: str, run_prefix: str, m: str) -> float:
    pat = f"{base}/{m}/{ds_dir}/{run_prefix}_{ds_dir}_{m}*/fold_*/xgb_optuna100_harmonized_v1/evaluations.json"
    vals = []
    for p in sorted(glob.glob(str(ROOT / pat))):
        d = json.loads(Path(p).read_text())
        for ev in d.get("evaluations", []):
            for mm in ev.get("metrics", []):
                if mm["name"] == "positive_f1":
                    vals.append(mm["value"])
    assert len(vals) in (1, 5), f"expected 1 or 5 folds of evaluations, found {len(vals)}"
    return _mean(vals)


MOD_DIR = {"Audio + Text": "audio_text", "Audio only": "audio_only", "Text only": "text_only"}


@pytest.mark.parametrize(
    "dataset,modality",
    [("DAIC", m) for m in MOD_DIR]
    + [("CMDC", m) for m in MOD_DIR]
    + [("Turkish", m) for m in MOD_DIR],
)
def test_standalone_qwen_posf1_daic_cmdc_turkish(dataset, modality):
    m = MOD_DIR[modality]
    expected = build_clean_workbook.STANDALONE_QWEN_POSF1[(dataset, modality)]
    if dataset == "DAIC":
        run = _pick_run(ROOT / f"output_model/harmonized_v1/{m}/daic",
                        "harmonized_v1_harmonized_v1_prod_20260809T171705Z_d1e8130b_daic_*")
        d = json.loads((run / "fold_0/best_model/standalone_eval/metrics_original_teacher_forced.json").read_text())
        got = d["positive_f1"]
    else:
        ds_dir = "cmdc" if dataset == "CMDC" else "turkish_t17_qwen3asr"
        run_ds = "cmdc" if dataset == "CMDC" else "turkish"
        run = _pick_run(ROOT / f"output_model/harmonized_v1/{m}/{ds_dir}",
                        f"harmonized_v1_harmonized_v1_prod_20260809T171705Z_d1e8130b_{run_ds}_{m}*")
        vals = []
        for fold in range(5):
            d = json.loads((run / f"fold_{fold}/eval/best_validation/metrics_original_teacher_forced.json").read_text())
            vals.append(d["positive_f1"])
        got = _mean(vals)
    assert got == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("dataset,ds_dir,run_ds", [("D3TEC", "d3tec", "d3tec"), ("Androids Interview", "androids", "androids_interview")])
@pytest.mark.parametrize("modality", list(MOD_DIR))
def test_standalone_qwen_posf1_d3tec_androids_pooled(dataset, ds_dir, run_ds, modality):
    m = MOD_DIR[modality]
    expected = build_clean_workbook.STANDALONE_QWEN_POSF1[(dataset, modality)]
    found: dict[int, Path] = {}
    pat = (f"output_model/harmonized_v1/{m}/{ds_dir}/harmonized_v1_harmonized_v1_prod_20260809T171705Z_d1e8130b_{run_ds}_{m}*/"
           f"fold_*/best_model/standalone_eval*/predictions_subject_level.csv")
    for p in sorted(glob.glob(str(ROOT / pat))):
        parts = p.split("/")
        fold_idx = next(i for i, x in enumerate(parts) if x.startswith("fold_"))
        fold = int(parts[fold_idx].split("_")[1])
        found.setdefault(fold, Path(p))
    assert len(found) == 5, f"expected 5 folds, found {sorted(found)}"
    rows = []
    for fold in range(5):
        rows.extend(_read_csv(found[fold]))
    macro, pos, _cm = _f1s_from_preds(rows)
    assert pos == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize(
    "dataset,ds_dir",
    [("DAIC", "daic"), ("CMDC", "cmdc"), ("Turkish", "turkish"),
     ("D3TEC", "d3tec"), ("Androids Interview", "androids_interview")],
)
@pytest.mark.parametrize("modality", list(MOD_DIR))
def test_standalone_heads_posf1(dataset, ds_dir, modality):
    m = MOD_DIR[modality]
    exp_lr, exp_xg = build_clean_workbook.STANDALONE_HEADS_POSF1[(dataset, modality)][:2]
    hc_base = ROOT / "outputs/hidden_classifiers/harmonized_v1" / ds_dir
    run = _pick_run(hc_base, f"harmonized_v1_harmonized_v1_prod_20260809T171705Z_d1e8130b_{ds_dir}_{m}*")
    lr_vals, xg_vals = [], []
    for p in sorted(run.glob("fold_*/variant_summary.json")):
        d = json.loads(p.read_text())
        lr_vals.append(next(v["positive_f1"] for v in d if v["variant"] == "logreg_raw"))
        xg_vals.append(next(v["positive_f1"] for v in d if v["variant"] == "xgb_raw"))
    assert _mean(lr_vals) == pytest.approx(exp_lr, abs=1e-6)
    assert _mean(xg_vals) == pytest.approx(exp_xg, abs=1e-6)


@pytest.mark.parametrize(
    "dataset,ds_dir",
    [("D3TEC", "d3tec"), ("Androids Interview", "androids_interview"),
     ("CMDC", "cmdc"), ("Turkish", "turkish")],
)
@pytest.mark.parametrize("modality", ["Audio + Text", "Text only"])
def test_en_heads_posf1(dataset, ds_dir, modality):
    m = MOD_DIR[modality]
    exp_lr, exp_xg = build_clean_workbook.EN_HEADS_POSF1[(dataset, modality)]
    hc_base = ROOT / "outputs/hidden_classifiers/harmonized_v1_en" / ds_dir
    runs = sorted(hc_base.glob(f"harmonized_v1_en_*_{ds_dir}_{m}*"))
    assert runs
    run = runs[-1]
    lr_vals, xg_vals = [], []
    for p in sorted(run.glob("fold_*/variant_summary.json")):
        d = json.loads(p.read_text())
        lr_vals.append(next(v["positive_f1"] for v in d if v["variant"] == "logreg_raw"))
        xg_vals.append(next(v["positive_f1"] for v in d if v["variant"] == "xgb_raw"))
    assert _mean(lr_vals) == pytest.approx(exp_lr, abs=1e-6)
    assert _mean(xg_vals) == pytest.approx(exp_xg, abs=1e-6)


@pytest.mark.parametrize("dataset,ds_dir", [("D3TEC", "d3tec"), ("Androids Interview", "androids_interview"),
                                            ("CMDC", "cmdc"), ("Turkish", "turkish"), ("DAIC", "daic")])
@pytest.mark.parametrize("modality", list(MOD_DIR))
def test_optuna_native_posf1(dataset, ds_dir, modality):
    m = MOD_DIR[modality]
    exp_q = build_clean_workbook.QWEN_OPTUNA_POSF1[(dataset, modality)]
    exp_g = build_clean_workbook.GEMMA_OPTUNA_POSF1[(dataset, modality)]
    q = _optuna_pos("output_model/harmonized_v1_optuna100", ds_dir, "harmonized_v1_optuna100_optuna100_native_20260815T0330Z_99efc52", m)
    g = _optuna_pos("output_model/harmonized_v1_gemma4_optuna100", ds_dir, "gemma4_harmonized_v1_optuna100_optuna100_native_20260815T0330Z_99efc52", m)
    assert q == pytest.approx(exp_q, abs=1e-5)
    assert g == pytest.approx(exp_g, abs=1e-5)


@pytest.mark.parametrize("dataset,ds_dir", [("D3TEC", "d3tec"), ("Androids Interview", "androids_interview"),
                                            ("CMDC", "cmdc"), ("Turkish", "turkish")])
@pytest.mark.parametrize("modality", ["Audio + Text", "Text only"])
def test_optuna_english_posf1(dataset, ds_dir, modality):
    m = MOD_DIR[modality]
    exp_q, exp_g = build_clean_workbook.EN_XGB_POSF1[(dataset, modality)]
    q = _optuna_pos("output_model/harmonized_v1_en_optuna100", ds_dir, "harmonized_v1_en_optuna100_optuna100_english_20260815T2300Z_a955cdd", m)
    g = _optuna_pos("output_model/harmonized_v1_en_gemma4_optuna100", ds_dir, "gemma4_harmonized_v1_en_optuna100_optuna100_english_20260815T2300Z_a955cdd", m)
    assert q == pytest.approx(exp_q, abs=1e-5)
    assert g == pytest.approx(exp_g, abs=1e-5)


@pytest.mark.parametrize("stage,st", [("cv", "cv"), ("final", "final")])
@pytest.mark.parametrize("modality", list(MOD_DIR))
def test_optuna_merged_posf1(stage, st, modality):
    m = MOD_DIR[modality]
    exp_q, exp_g = build_clean_workbook.MERGED_OPTUNA_POSF1[(stage, modality)]

    def _merged_optuna_pos(base: str, backend: str) -> float:
        pat = f"{base}/{m}/*_merged_optuna100_*_{m}_{st}/fold_*/xgb_optuna100_harmonized_v1/evaluations.json"
        vals = []
        for p in sorted(glob.glob(str(ROOT / pat))):
            d = json.loads(Path(p).read_text())
            for ev in d.get("evaluations", []):
                for mm in ev.get("metrics", []):
                    if mm["name"] == "positive_f1":
                        vals.append(mm["value"])
        assert vals, f"no evaluations found for {pat}"
        return _mean(vals)

    q = _merged_optuna_pos("output_model/harmonized_v1_merged_optuna100", "qwen")
    g = _merged_optuna_pos("output_model/harmonized_v1_gemma4_merged_optuna100", "gemma4")
    assert q == pytest.approx(exp_q, abs=1e-5)
    assert g == pytest.approx(exp_g, abs=1e-5)


@pytest.mark.parametrize(
    "dataset,ds_dir,run_ds,folder",
    [("D3TEC", "d3tec", "d3tec", "best_model/standalone_eval"),
     ("Androids Interview", "androids", "androids_interview", "best_model/standalone_eval"),
     ("CMDC", "cmdc", "cmdc", "eval/best_validation"),
     ("Turkish", "turkish_t17_qwen3asr", "turkish", "eval/best_validation")],
)
@pytest.mark.parametrize("modality", list(MOD_DIR))
def test_gemma_native_tf_posf1(dataset, ds_dir, run_ds, folder, modality):
    m = MOD_DIR[modality]
    expected = build_clean_workbook.GEMMA_NATIVE_TF_POSF1[(dataset, modality)]
    folds = _gemma_fold_metrics(
        "output_model/harmonized_v1_gemma4", m, ds_dir,
        f"gemma4_harmonized_v1_gemma4_v1_prod_20260814T2030Z_1ab337d2_r2_{run_ds}_{m}", folder)
    assert len(folds) == 5, f"expected 5 folds, got {len(folds)}"
    got = _mean([d["positive_f1"] for d in folds])
    assert got == pytest.approx(expected, abs=1e-5)


@pytest.mark.parametrize("modality", list(MOD_DIR))
def test_gemma_daic_tf_posf1(modality):
    expected = build_clean_workbook.GEMMA_NATIVE_TF_POSF1[("DAIC", modality)]
    m = MOD_DIR[modality]
    run = _pick_run(ROOT / f"output_model/harmonized_v1_gemma4/{m}/daic",
                    "gemma4_harmonized_v1_gemma4_v1_prod_20260812T020449Z_cca3f4ae_daic_*")
    d = json.loads((run / "fold_0/best_model/standalone_eval/metrics_original_teacher_forced.json").read_text())
    assert d["positive_f1"] == pytest.approx(expected, abs=1e-5)


@pytest.mark.parametrize(
    "dataset,ds_dir,run_ds",
    [("D3TEC", "d3tec", "d3tec"), ("Androids Interview", "androids_interview", "androids_interview"),
     ("CMDC", "cmdc", "cmdc"), ("Turkish", "turkish", "turkish")],
)
@pytest.mark.parametrize("modality", list(MOD_DIR))
def test_gemma_native_lr_posf1(dataset, ds_dir, run_ds, modality):
    m = MOD_DIR[modality]
    expected = build_clean_workbook.GEMMA_NATIVE_LR_POSF1[(dataset, modality)]
    p = ROOT / f"outputs/experiment_reports/gemma4_harmonized/native_lr/{ds_dir}_{m}.json"
    d = json.loads(p.read_text())
    got = d["aggregation_views"]["fold_mean"]["positive_f1"]
    assert got == pytest.approx(expected, abs=1e-5)


@pytest.mark.parametrize(
    "dataset,ds_dir,run_ds",
    [("D3TEC", "d3tec", "d3tec"), ("Androids Interview", "androids", "androids_interview"),
     ("CMDC", "cmdc", "cmdc"), ("Turkish", "turkish_t17_qwen3asr", "turkish")],
)
@pytest.mark.parametrize("modality", ["Audio + Text", "Text only"])
def test_gemma_en_tf_posf1(dataset, ds_dir, run_ds, modality):
    m = MOD_DIR[modality]
    expected = build_clean_workbook.EN_TF_POSF1[(dataset, modality)][1]
    folder = "eval/best_validation" if dataset in ("CMDC", "Turkish") else "best_model/standalone_eval"
    folds = _gemma_fold_metrics(
        "output_model/harmonized_v1_en_gemma4", m, ds_dir,
        f"gemma4_harmonized_v1_en_gemma4_en_prod_20260815T1300Z_a955cdd_{run_ds}_{m}", folder)
    assert len(folds) == 5, f"expected 5 folds, got {len(folds)}"
    got = _mean([d["positive_f1"] for d in folds])
    assert got == pytest.approx(expected, abs=1e-5)


@pytest.mark.parametrize(
    "dataset,ds_dir",
    [("D3TEC", "d3tec"), ("Androids Interview", "androids_interview"),
     ("CMDC", "cmdc"), ("Turkish", "turkish")],
)
@pytest.mark.parametrize("modality", ["Audio + Text", "Text only"])
def test_gemma_en_lr_posf1(dataset, ds_dir, modality):
    m = MOD_DIR[modality]
    expected = build_clean_workbook.EN_LR_POSF1[(dataset, modality)][1]
    p = ROOT / f"outputs/experiment_reports/gemma4_harmonized/english_lr/{ds_dir}_{m}.json"
    d = json.loads(p.read_text())
    got = d["aggregation_views"]["fold_mean"]["positive_f1"]
    assert got == pytest.approx(expected, abs=1e-5)


@pytest.mark.parametrize("stage", ["cv", "final"])
@pytest.mark.parametrize("modality", list(MOD_DIR))
def test_merged_qwen_posf1(stage, modality):
    m = MOD_DIR[modality]
    exp_tf_q, exp_tf_g = build_clean_workbook.MERGED_TF_POSF1[(stage, modality)]
    exp_lr_q, exp_lr_g = build_clean_workbook.MERGED_LR_POSF1[(stage, modality)]
    qwen_base = ROOT / f"outputs/symmetric_merged/harmonized_v1/{m}/harmonized_v1_prod_20260809T171705Z_d1e8130b/{stage}"
    gemma_model_base = ROOT / f"output_model/symmetric_merged/gemma4/harmonized_v1/{m}/gemma4_merged_v1_prod_20260816T0000Z_d4ff33e/{stage}"
    gemma_outputs_base = ROOT / f"outputs/symmetric_merged/gemma4/harmonized_v1/{m}/gemma4_merged_v1_prod_20260816T0000Z_d4ff33e/{stage}"
    folds = sorted(qwen_base.glob("fold_*"))
    assert folds, f"no folds for qwen merged {stage} {m}"

    def per_dataset_mean(fdir: Path, rel: str) -> float:
        d = json.loads((fdir / rel).read_text())
        return _mean(v["metrics"]["positive_f1"] for v in d.values())

    # Qwen TF
    if stage == "final":
        d = json.loads((folds[0] / "qwen/summary.json").read_text())
        q_tf = d["daic"]["metrics"]["positive_f1"]
        g_tf = json.loads((gemma_model_base / "fold_0/logs/postprocess/final_daic_metrics_original_teacher_forced.json").read_text())["positive_f1"]
    else:
        q_tf = _mean(per_dataset_mean(f, "qwen/summary.json") for f in folds)
        g_tf = _mean(
            json.loads((f / "logs/selection/combined_selection_metrics.json").read_text())["positive_f1"]
            for f in sorted(gemma_model_base.glob("fold_*"))
            if (f / "logs/selection/combined_selection_metrics.json").is_file()
        )
    assert q_tf == pytest.approx(exp_tf_q, abs=1e-5)
    assert g_tf == pytest.approx(exp_tf_g, abs=1e-5)

    # Qwen LR and Gemma LR
    def lr_per_dataset_mean(f: Path) -> float:
        d = json.loads(f.read_text())
        return _mean(v["positive_f1"] for v in d.values())

    q_lr_vals, g_lr_vals = [], []
    for f in folds:
        p = f / "heads/logreg/metrics_by_dataset.json"
        if p.is_file():
            q_lr_vals.append(lr_per_dataset_mean(p))
    gemma_folds = sorted(gemma_outputs_base.glob("fold_*"))
    for f in gemma_folds:
        p = f / "heads/logreg/metrics_by_dataset.json"
        if p.is_file():
            g_lr_vals.append(lr_per_dataset_mean(p))
    assert q_lr_vals, f"no qwen merged LR metrics for {stage} {m}"
    assert g_lr_vals, f"no gemma merged LR metrics for {stage} {m}"
    assert _mean(q_lr_vals) == pytest.approx(exp_lr_q, abs=1e-5)
    assert _mean(g_lr_vals) == pytest.approx(exp_lr_g, abs=1e-5)
