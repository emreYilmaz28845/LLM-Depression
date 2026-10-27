import numpy as np
import pytest

from baselines.head_ablation_stats import _holm, _metrics, analyse


def _records(n_subjects=20, folds=2, flip_b=4, seeds=(1, 2)):
    """Condition lora_new is perfect; frozen_new flips ``flip_b`` subjects per fold."""
    records = []
    for route in ("audio_only_native", "text_only_native"):
        for fold in range(folds):
            for i in range(n_subjects):
                subject, label = f"s{fold}_{i}", i % 2
                for cond, seed_list in (("frozen_new", (0,)), ("lora_new", seeds)):
                    for seed in seed_list:
                        pred = label if cond == "lora_new" or i >= flip_b else 1 - label
                        records.append({"scope": "standalone", "dataset": "d", "route": route,
                                        "condition": cond, "seed": seed, "fold": fold,
                                        "subject": subject, "label": label, "pred": pred})
    return records


def test_metrics_match_hand_counts():
    m = _metrics(np.array([3.0]), np.array([1.0]), np.array([1.0]), np.array([5.0]))
    assert m["positive_f1"][0] == pytest.approx(6 / 8)
    assert m["macro_f1"][0] == pytest.approx((6 / 8 + 10 / 12) / 2)
    assert m["uar"][0] == pytest.approx((3 / 4 + 5 / 6) / 2)


def test_holm_is_monotone_and_capped():
    assert _holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert _holm([0.6, 0.9]) == [1.0, 1.0]


def test_analyse_reports_paired_difference_and_significance():
    result = analyse(_records(), replicates=2000, seed=0)
    test = next(t for t in result["tests"]
                if t["group"] == "all" and t["comparison"] == "lora_effect_new_head")
    assert test["mean_diff_macro_f1"] == pytest.approx(1.0 - 0.8)
    assert test["n_routes"] == 2 and test["routes_a_better"] == 2
    assert test["ci95"][0] > 0 and test["p_bootstrap"] < 0.05
    row = next(r for r in result["table"] if r["condition"] == "lora_new")
    assert row["n_seeds"] == 2 and row["n_folds"] == 2 and row["seed_sd_macro_f1"] == 0.0


def test_missing_subject_in_one_condition_fails():
    records = [r for r in _records() if not (r["condition"] == "frozen_new" and r["subject"] == "s0_0"
                                             and r["route"] == "text_only_native")]
    with pytest.raises(ValueError, match="missing"):
        analyse(records, replicates=10, seed=0)
