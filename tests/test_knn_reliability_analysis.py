import numpy as np
import pandas as pd
import pytest

from scripts.knn_reliability_analysis import (
    gate, holm, knn_share, overlap, paired, paired_uncertainty, scores, select_threshold,
)


def frame(pred):
    return pd.DataFrame({"subject_id": ["synthetic_a", "synthetic_b", "synthetic_c", "synthetic_d"],
                         "fold": [0] * 4, "label": [0, 0, 1, 1], "prediction": pred})


def test_overlap_partition_and_denominators():
    r = overlap(frame([0, 0, 0, 0]), frame([0, 1, 1, 0]))
    assert [r[k] for k in ["both_correct", "only_text_correct", "only_audio_text_correct", "both_wrong"]] == [1, 1, 1, 1]
    assert r["shared_error_pct_of_union"] == pytest.approx(100 / 3)
    assert r["text_errors_corrected_pct"] == 50


def test_pairing_rejects_silent_drops_duplicates_and_label_mismatch():
    a = frame([0, 0, 1, 1])
    for b in [a.iloc[:-1], pd.concat([a, a.iloc[:1]]), a.assign(label=1)]:
        with pytest.raises(ValueError):
            paired(a, b)


def test_knn_uses_training_normalization_only():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(15, 6)); y = np.arange(15) % 2; q = rng.normal(size=(2, 6))
    a = knn_share(x, y, q)
    # An unrelated query cannot change the first query's neighbours.
    assert a[0] == knn_share(x, y, np.vstack([q[0], np.full(6, 1e6)]))[0]
    with pytest.raises(ValueError):
        knn_share(x[:5], y[:5], q)


def test_gate_all_four_reliability_states():
    pt = np.full(4, .8); pa = np.full(4, .1)
    result = gate(pt, pa, np.array([.2, .8, .2, .8]), np.array([.2, .2, .8, .8]), .6)
    np.testing.assert_allclose(result, [.8, .8, .1, .45])


def test_selection_tie_is_conservative_and_no_eval_arguments():
    threshold, _ = select_threshold(np.array([0, 1]), np.array([.1, .9]), np.array([.1, .9]),
                                    np.ones(2), np.ones(2))
    assert threshold == 1.1


def test_identical_predictions_have_zero_paired_difference():
    d = frame([0, 0, 1, 1]).rename(columns={"prediction": "a"}); d["b"] = d.a
    r = paired_uncertainty(d, "a", "b", 30, 7)
    assert r == {"delta_macro_f1": 0., "ci95_low": 0., "ci95_high": 0., "permutation_p": 1.}
    assert holm([.01, .04, .03]) == pytest.approx([.03, .06, .06])


def test_invalid_prediction_counts_as_wrong():
    assert scores([0, 1], [-1, 1])["accuracy"] == .5
    assert scores([0, 1], [-1, 1])["macro_f1"] == .5
