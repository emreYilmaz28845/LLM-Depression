"""Leave-one-participant-out linear calibration of D3TEC zero-shot PHQ-9 totals, plus item means by group.

D3TEC has no official split, so unlike DAIC v4 (fit on train+dev, applied once to test) the line
gold ~ a + b * predicted_total is fitted on the other 61 participants and applied to the held-out one.
Calibrated label = calibrated score >= 10. Raw metrics are reported next to it; nothing is tuned on the
held-out participant.

Usage: python experiments/manyshot_claude/calibrate_d3tec_phq9.py RUN_DIR [RUN_DIR ...]
Writes RUN_DIR/calibration_lopo.json for each run and prints a summary (mean over runs when several).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import f1_score, roc_auc_score

ITEM_NAMES = ("interest", "down", "sleep", "tired", "appetite", "self", "concentration", "psychomotor", "self-harm")


def analyse(run_dir: Path) -> dict:
    recs = [json.loads(line) for line in (run_dir / "responses.jsonl").read_text().splitlines()]
    invalid = [r["subject_id"] for r in recs if r["phq9_total"] is None]
    recs = [r for r in recs if r["phq9_total"] is not None]
    x = np.array([r["phq9_total"] for r in recs], float)
    y = np.array([r["phq9_score"] for r in recs], float)
    lab = np.array([r["label"] for r in recs])
    cal = np.empty_like(x)
    for i in range(len(x)):
        keep = np.arange(len(x)) != i
        b, a = np.polyfit(x[keep], y[keep], 1)
        cal[i] = a + b * x[i]
    b_all, a_all = np.polyfit(x, y, 1)
    items = np.array([r["phq9_items"] for r in recs], float)
    out = {
        "run": run_dir.name, "n": len(recs), "invalid": invalid,
        "raw": {"spearman": float(spearmanr(x, y)[0]), "mae": float(np.abs(x - y).mean()),
                "macro_f1": float(f1_score(lab, (x >= 10).astype(int), average="macro")),
                "auroc": float(roc_auc_score(lab, x)), "mean_pred": float(x.mean()), "mean_gold": float(y.mean())},
        "lopo_calibrated": {"mae": float(np.abs(cal - y).mean()),
                            "macro_f1": float(f1_score(lab, (cal >= 10).astype(int), average="macro"))},
        "full_fit_line_for_reference": {"intercept": float(a_all), "slope": float(b_all)},
        "baseline_mae_predict_mean": float(np.mean([abs(y[i] - np.delete(y, i).mean()) for i in range(len(y))])),
        "item_means": {name: {"depressed": float(items[lab == 1, j].mean()),
                              "non_depressed": float(items[lab == 0, j].mean())}
                       for j, name in enumerate(ITEM_NAMES)},
        "item9_nonzero": int((items[:, 8] > 0).sum()),
    }
    (run_dir / "calibration_lopo.json").write_text(json.dumps(out, indent=1))
    return out


def main() -> None:
    results = [analyse(Path(d)) for d in sys.argv[1:]]
    for r in results:
        print(json.dumps({k: r[k] for k in ("run", "n", "invalid", "raw", "lopo_calibrated",
                                            "baseline_mae_predict_mean", "item9_nonzero")}, indent=1))
    if len(results) > 1:
        def mean(path):
            vals = [r[path[0]][path[1]] for r in results]
            return round(float(np.mean(vals)), 4), round(float(np.std(vals)), 4)
        print("mean (sd) over runs:", {f"{a}.{b}": mean((a, b)) for a, b in
              [("raw", "spearman"), ("raw", "mae"), ("raw", "macro_f1"), ("raw", "auroc"),
               ("lopo_calibrated", "mae"), ("lopo_calibrated", "macro_f1")]})


if __name__ == "__main__":
    main()
