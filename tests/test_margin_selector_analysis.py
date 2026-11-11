import numpy as np
import pandas as pd
import pytest

from scripts import margin_selector_analysis as m


def test_margin_uses_dep_minus_non_and_checks_prediction():
    d = pd.DataFrame({"dep_score": [1., 0.], "non_score": [0., 2.], "prediction": [1, 0]})
    assert m.margin(d).tolist() == [1., -2.]
    d.loc[0, "prediction"] = 0
    with pytest.raises(ValueError):
        m.margin(d)


def test_margin_rejects_non_finite():
    d = pd.DataFrame({"score_margin": [np.nan, 1.], "prediction": [0, 1]})
    with pytest.raises(ValueError):
        m.margin(d)


def test_rules():
    zt, za = np.array([.5, -2., .1]), np.array([-1., 1., -.05])
    assert m.rule_predict("max_conf", zt, za).tolist() == [0, 0, 1]
    assert m.rule_predict("margin_sum", zt, za).tolist() == [0, 0, 1]
    # tau=inf: neither reliable -> text; tau=0: both reliable -> margin sum
    assert m.rule_predict("advisor_gate", zt, za, float("inf")).tolist() == [1, 0, 1]
    assert m.rule_predict("advisor_gate", zt, za, 0.).tolist() == [0, 0, 1]
    # tau=0.75: only A+T reliable for row 0 -> A+T; both for row 1 -> sum; neither row 2 -> text
    assert m.rule_predict("advisor_gate", zt, za, .75).tolist() == [0, 0, 1]


def test_max_conf_tie_goes_to_text():
    assert m.rule_predict("max_conf", np.array([1.]), np.array([-1.])).tolist() == [1]


def test_select_tau_tie_prefers_larger():
    y = np.array([1, 0]); zt = np.array([1., -1.]); za = np.array([1., -1.])
    tau, _ = m.select_tau(y, zt, za)
    assert tau == float("inf")


def test_uncertainty_zero_when_identical():
    rows = []
    for s in m.SEEDS:
        for f in (0, 1):
            for i in range(6):
                rows.append({"seed": s, "fold": f, "subject_id": f"s{f}{i}", "label": i % 2,
                             "a": i % 2, "b": i % 2})
    df = pd.DataFrame(rows)
    r = m.uncertainty(df, "a", "b", reps=20, seed=1)
    assert r["delta_macro_f1"] == 0 and r["ci95_low"] == 0 and r["ci95_high"] == 0


def test_uncertainty_rejects_seed_subject_mismatch():
    df = pd.DataFrame([{"seed": s, "fold": 0, "subject_id": f"x{s}", "label": 1, "a": 1, "b": 1}
                       for s in m.SEEDS])
    with pytest.raises(ValueError):
        m.uncertainty(df, "a", "b", reps=2, seed=1)


def test_margin_sum_equals_max_conf():
    rng = np.random.default_rng(0)
    zt, za = rng.normal(size=500), rng.normal(size=500)
    assert (m.rule_predict("margin_sum", zt, za) == m.rule_predict("max_conf", zt, za)).all()
