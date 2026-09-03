"""Three-way pooled scoring incl. the locked text pair rule (plan Steps 6-7)."""

from __future__ import annotations

import pytest

from tools.score_turkish_pooled import score_piles


def _sample(subject: str, sample: str, condition: str, label: int, pred, margin: float) -> dict:
    return {
        "subject_id": subject,
        "sample_id": sample,
        "dataset_variant": condition,
        "label": label,
        "teacher_forced_prediction": pred,
        "score_margin": margin,
    }


def test_audio_piles_use_mean_margin_sign_with_strict_invalid() -> None:
    rows = [
        # s1 gold 1: all piles correct
        _sample("s1", "s1-p1", "pos_only_t17", 1, 1, 0.5),
        _sample("s1", "s1-p2", "pos_only_t17", 1, 1, 0.3),
        _sample("s1", "s1-n1", "negative_only_t17", 1, 0, -0.1),
        _sample("s1", "s1-n2", "negative_only_t17", 1, 1, 0.4),
        # s2 gold 0: pooled margins sum to exactly 0.0 -> INVALID -> wrong
        _sample("s2", "s2-p1", "pos_only_t17", 0, 0, -0.5),
        _sample("s2", "s2-p2", "pos_only_t17", 0, 0, -0.25),
        _sample("s2", "s2-n1", "negative_only_t17", 0, 1, 0.5),
        _sample("s2", "s2-n2", "negative_only_t17", 0, 1, 0.25),
    ]
    piles = score_piles(rows)
    # pos pile: s1 margin +0.8 -> 1 correct; s2 margin -0.75 -> 0 correct
    assert piles["pos_only_t17"]["macro_f1"] == pytest.approx(1.0)
    # neg pile: s1 margin +0.3 -> 1 correct; s2 margin +0.75 -> 1 wrong
    assert piles["negative_only_t17"]["macro_f1"] == pytest.approx(1 / 3)
    # pooled: s1 +1.1 -> 1 correct; s2 0.0 -> INVALID -> wrong, rate 1/2
    assert piles["pooled"]["num_invalid_subjects"] == 1
    assert piles["pooled"]["invalid_subject_rate"] == pytest.approx(0.5)
    assert piles["pooled"]["macro_f1"] == pytest.approx(1 / 3)


def test_text_pair_rule_agreeing_disagreeing_and_invalid() -> None:
    rows = [
        # agreeing pair, gold 1 -> correct
        _sample("t1", "t1-pos", "pos_only_t17", 1, 1, 0.5),
        _sample("t1", "t1-neg", "negative_only_t17", 1, 1, 0.3),
        # disagreeing pair, gold 0, mean margin +0.1 -> predicts 1 -> wrong
        _sample("t2", "t2-pos", "pos_only_t17", 0, 1, 0.6),
        _sample("t2", "t2-neg", "negative_only_t17", 0, 0, -0.4),
        # INVALID-containing pair, gold 1 -> wrong whatever the margin
        _sample("t3", "t3-pos", "pos_only_t17", 1, 1, 0.5),
        _sample("t3", "t3-neg", "negative_only_t17", 1, "INVALID", -0.9),
        # agreeing pair, gold 0 -> correct
        _sample("t4", "t4-pos", "pos_only_t17", 0, 0, -0.5),
        _sample("t4", "t4-neg", "negative_only_t17", 0, 0, -0.2),
    ]
    piles = score_piles(rows, text_pair_mode=True)
    # hand-computed: strict pooled preds [1,1,0,0] vs gold [1,0,1,0]
    assert piles["pooled"]["macro_f1"] == pytest.approx(0.5)
    assert piles["pooled"]["accuracy"] == pytest.approx(0.5)
    assert piles["pooled"]["num_invalid_subjects"] == 1
    # per-pile text uses decoded labels: pos strict [1,1,1,0] -> f1_pos 0.8
    assert piles["pos_only_t17"]["positive_f1"] == pytest.approx(0.8)
    # neg strict [1,0,0,0] (INVALID->wrong) -> macro (2/3+0.8)/2
    assert piles["negative_only_t17"]["macro_f1"] == pytest.approx((2 / 3 + 0.8) / 2)


def test_unknown_condition_raises() -> None:
    rows = [_sample("s1", "s1-x", "pooled_t17", 1, 1, 0.5)]
    with pytest.raises(ValueError):
        score_piles(rows)
