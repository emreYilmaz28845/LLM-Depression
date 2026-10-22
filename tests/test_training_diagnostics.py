from __future__ import annotations

import json
from pathlib import Path

from src.experiment_tracking import diagnostics
from tools import training_diagnostics


def _row(epoch, train, val, f1, pos=0.5, rec=0.5):
    return {"epoch": epoch, "train_loss": train, "inner_val_loss": val, "inner_val_macro_f1": f1,
            "inner_val_positive_f1": pos, "inner_val_recall": rec}


def _flags(result):
    return {f["flag"] for f in result["flags"]}


def test_healthy_run_has_no_flags():
    hist = [_row(1, 0.9, 0.8, 0.60), _row(2, 0.7, 0.7, 0.70), _row(3, 0.6, 0.65, 0.75), _row(4, 0.55, 0.64, 0.74)]
    res = diagnostics.analyze_history(hist)
    assert res["best_epoch"] == 3 and res["best_epoch_source"].startswith("derived")
    assert _flags(res) == set()


def test_best_is_last_is_reported():
    hist = [_row(1, 0.9, 0.8, 0.6), _row(2, 0.7, 0.7, 0.65), _row(3, 0.6, 0.6, 0.7)]
    assert _flags(diagnostics.analyze_history(hist)) == {"best_is_last"}


def test_overfit_needs_two_rising_val_loss_epochs_and_falling_train_loss():
    hist = [_row(1, 0.9, 0.8, 0.7), _row(2, 0.5, 0.6, 0.72), _row(3, 0.3, 0.7, 0.71), _row(4, 0.2, 0.9, 0.70)]
    res = diagnostics.analyze_history(hist)
    assert "overfit" in _flags(res)
    detail = next(f for f in res["flags"] if f["flag"] == "overfit")["detail"]
    assert detail["val_loss_min_epoch"] == 2 and detail["rising_epochs"] == 2
    one_rise = [_row(1, 0.9, 0.8, 0.7), _row(2, 0.5, 0.6, 0.72), _row(3, 0.3, 0.7, 0.71)]
    assert "overfit" not in _flags(diagnostics.analyze_history(one_rise))


def test_late_degradation_threshold():
    hist = [_row(1, 0.9, 0.8, 0.6), _row(2, 0.7, 0.7, 0.80), _row(3, 0.6, 0.7, 0.72)]
    assert "late_degradation" not in _flags(diagnostics.analyze_history(hist))
    hist[2] = _row(3, 0.6, 0.7, 0.70)
    assert "late_degradation" in _flags(diagnostics.analyze_history(hist))


def test_collapse_only_counts_after_best_epoch():
    early_all_positive = [_row(1, 0.9, 0.8, 0.40, pos=0.6, rec=1.0), _row(2, 0.7, 0.7, 0.70), _row(3, 0.6, 0.65, 0.72)]
    assert "collapse" not in _flags(diagnostics.analyze_history(early_all_positive))
    late = [_row(1, 0.9, 0.8, 0.70), _row(2, 0.7, 0.7, 0.72), _row(3, 0.6, 0.65, 0.40, pos=0.0, rec=0.0)]
    res = diagnostics.analyze_history(late)
    detail = next(f for f in res["flags"] if f["flag"] == "collapse")["detail"]
    assert detail["epochs"][0]["side"] == "all_negative"
    pos = [_row(1, 0.9, 0.8, 0.70), _row(2, 0.7, 0.7, 0.72), _row(3, 0.6, 0.65, 0.40, pos=0.6, rec=1.0)]
    assert next(f for f in diagnostics.analyze_history(pos)["flags"] if f["flag"] == "collapse")["detail"]["epochs"][0]["side"] == "all_positive"


def test_nonfinite_loss_and_missing_metrics():
    hist = [_row(1, 0.9, 0.8, 0.6), _row(2, float("nan"), 0.7, 0.65), _row(3, 0.6, 0.6, 0.7)]
    assert "nonfinite" in _flags(diagnostics.analyze_history(hist))
    sparse = [{"epoch": e, "train_loss": 1.0 - 0.1 * e} for e in (1, 2, 3)]
    res = diagnostics.analyze_history(sparse)
    assert res["best_epoch"] is None and "macro_f1" in res["not_recorded"] and res["flags"] == []


def test_short_history_skips_trend_flags():
    res = diagnostics.analyze_history([_row(1, 0.9, 0.8, 0.5), _row(2, 0.7, 0.9, 0.3)])
    assert res["status"] == "short_history" and "overfit" not in _flags(res)


def test_selected_epoch_from_checkpoint_selection_file_wins(tmp_path):
    fold = tmp_path / "run" / "fold_0"
    (fold / "logs").mkdir(parents=True)
    (fold / "logs" / "training_history.json").write_text(json.dumps([_row(1, 0.9, 0.8, 0.6), _row(2, 0.7, 0.7, 0.8), _row(3, 0.6, 0.6, 0.7)]))
    (fold / "logs" / "selected_checkpoint_selection_metrics.json").write_text(json.dumps({"selected_epoch": 3}))
    res = diagnostics.load_fold(fold)
    assert res["best_epoch"] == 3 and res["best_epoch_source"] == "selected_checkpoint_selection_metrics"


def test_missing_and_broken_history_are_na_not_errors(tmp_path):
    (tmp_path / "fold_0").mkdir()
    assert diagnostics.load_fold(tmp_path / "fold_0")["status"] == "n/a"
    (tmp_path / "fold_1" / "logs").mkdir(parents=True)
    (tmp_path / "fold_1" / "logs" / "training_history.json").write_text("{not json")
    assert diagnostics.load_fold(tmp_path / "fold_1")["status"] == "unreadable"


def test_cross_fold_instability():
    def fold(best_epoch, f1):
        return {"status": "ok", "best_epoch": best_epoch, "macro_f1_best": f1, "flags": []}
    stable = diagnostics.analyze_run({"fold_0": fold(3, 0.70), "fold_1": fold(4, 0.72)})
    assert stable["flags"] == []
    unstable = diagnostics.analyze_run({"fold_0": fold(1, 0.50), "fold_1": fold(6, 0.80)})
    assert [f["detail"]["reason"] for f in unstable["flags"]] == ["best_epoch_range", "best_macro_f1_std"]


def test_scan_and_run_cli_write_outputs(tmp_path, capsys):
    root = tmp_path / "output_model"
    for dataset in ("cmdc", "androids"):
        fold = root / "camp" / "audio_only" / dataset / "run_a" / "fold_0"
        (fold / "logs").mkdir(parents=True)
        (fold / "logs" / "training_history.json").write_text(json.dumps([_row(1, 0.9, 0.8, 0.6), _row(2, 0.7, 0.7, 0.7), _row(3, 0.6, 0.6, 0.72)]))
    legacy = root / "audio_text" / "daic" / "run_b" / "fold_0"
    (legacy / "logs").mkdir(parents=True)
    (legacy / "logs" / "training_history.json").write_text("[]")
    out = tmp_path / "out"
    assert training_diagnostics.main(["scan", str(root), "--output", str(out)]) == 0
    rows = (out / "summary.csv").read_text().splitlines()
    assert len(rows) == 4 and any(",n/a," in r or r.endswith("n/a,") or "n/a" in r for r in rows[1:])
    assert "Training diagnostics summary" in (out / "summary.md").read_text()
    detail = json.loads((out / "details.json").read_text())
    assert detail["thresholds"] == diagnostics.DIAGNOSTIC_THRESHOLDS and len(detail["runs"]) == 3
    run_dir = root / "camp" / "audio_only" / "cmdc" / "run_a"
    assert training_diagnostics.main(["run", str(run_dir), "--output", str(tmp_path / "runout")]) == 0
    assert "Training observations" in capsys.readouterr().out
