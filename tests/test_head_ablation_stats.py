import numpy as np
import pytest

from scipy import stats

from baselines.head_ablation_stats import _metrics, _paired_t, analyse


def _records(n_subjects=20, folds=2, flip_b=4, seeds=(1, 2)):
    """Condition lora_new is perfect; frozen_new flips ``flip_b`` subjects per fold (more in text)."""
    records = []
    for route in ("audio_only_native", "text_only_native"):
        for fold in range(folds):
            for i in range(n_subjects):
                subject, label = f"s{fold}_{i}", i % 2
                for cond, seed_list in (("frozen_new", (0,)), ("lora_new", seeds)):
                    for seed in seed_list:
                        flips = flip_b + 2 * (route == "text_only_native") + fold
                        pred = label if cond == "lora_new" or i >= flips else 1 - label
                        records.append({"scope": "standalone", "dataset": "d", "route": route,
                                        "condition": cond, "seed": seed, "fold": fold,
                                        "subject": subject, "label": label, "pred": pred})
    return records


def test_metrics_match_hand_counts():
    m = _metrics(np.array([3.0]), np.array([1.0]), np.array([1.0]), np.array([5.0]))
    assert m["positive_f1"][0] == pytest.approx(6 / 8)
    assert m["macro_f1"][0] == pytest.approx((6 / 8 + 10 / 12) / 2)
    assert m["uar"][0] == pytest.approx((3 / 4 + 5 / 6) / 2)


def test_paired_t_matches_scipy_and_handles_small_n():
    d = [0.05, -0.01, 0.03, 0.02]
    out = _paired_t(d)
    ref = stats.ttest_1samp(d, 0.0)
    assert out["t"] == pytest.approx(ref.statistic) and out["p"] == pytest.approx(ref.pvalue)
    assert out["df"] == 3 and out["ci95"][0] < sum(d) / 4 < out["ci95"][1]
    assert _paired_t([0.1])["p"] is None
    assert _paired_t([0.0, 0.0])["p"] == 1.0


def test_analyse_reports_paired_difference_and_significance():
    result = analyse(_records())
    test = next(t for t in result["tests"]
                if t["group"] == "all" and t["comparison"] == "lora_effect_new_head")
    frozen = {r["route"]: r["macro_f1"] for r in result["table"] if r["condition"] == "frozen_new"}
    route_diffs = [1.0 - v for v in frozen.values()]
    assert test["mean_diff_macro_f1"] == pytest.approx(sum(route_diffs) / 2)
    assert test["n_routes"] == 2 and test["routes_a_better"] == 2 and test["n_route_folds"] == 4
    assert test["p_route"] == pytest.approx(stats.ttest_1samp(route_diffs, 0.0).pvalue)
    assert test["p_fold"] < 0.05 and test["ci95_route"][0] < test["mean_diff_macro_f1"] < test["ci95_route"][1]
    row = next(r for r in result["table"] if r["condition"] == "lora_new")
    assert row["n_seeds"] == 2 and row["n_folds"] == 2 and row["seed_sd_macro_f1"] == 0.0


def test_missing_subject_in_one_condition_fails():
    records = [r for r in _records() if not (r["condition"] == "frozen_new" and r["subject"] == "s0_0"
                                             and r["route"] == "text_only_native")]
    with pytest.raises(ValueError, match="missing"):
        analyse(records)
