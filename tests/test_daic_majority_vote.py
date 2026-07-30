from __future__ import annotations

import pytest

from scripts.reaggregate_daic_majority_vote import aggregate_majority_rows


def _row(subject: str, chunk: str, label: int, score: float, head: str) -> dict:
    row = {"subject_id": subject, "chunk_id": chunk, "label": label}
    row["teacher_forced_margin" if head == "qwen" else "probability"] = score
    return row


@pytest.mark.parametrize(
    ("head", "scores", "expected", "decision"),
    [
        ("qwen", [1.0, 0.2, -0.1], 1, "strict_positive_majority"),
        ("logreg_raw", [0.9, 0.8, 0.1], 1, "strict_positive_majority"),
        ("xgb_raw", [0.9, 0.1, 0.2], 0, "strict_negative_majority"),
        ("qwen", [2.0, -0.1], 1, "mean_continuous_score_tiebreak"),
        ("logreg_raw", [0.9, 0.4], 1, "mean_continuous_score_tiebreak"),
        ("xgb_raw", [0.6, 0.1], 0, "mean_continuous_score_tiebreak"),
    ],
)
def test_majority_and_tie_rules(
    head: str, scores: list[float], expected: int, decision: str
) -> None:
    rows = [_row("1", f"chunk-{index}", 1, score, head) for index, score in enumerate(scores)]
    result = aggregate_majority_rows(rows, head)[0]
    assert result["prediction"] == expected
    assert result["decision"] == decision


def test_majority_rejects_duplicate_chunks() -> None:
    rows = [
        _row("1", "chunk-1", 0, 0.2, "logreg_raw"),
        _row("1", "chunk-1", 0, 0.8, "logreg_raw"),
    ]
    with pytest.raises(ValueError, match="duplicate chunks"):
        aggregate_majority_rows(rows, "logreg_raw")
