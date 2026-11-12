import pytest

from baselines.lora_fusion import read_predictions


def _write(tmp_path, rows):
    p = tmp_path / "predictions_subject_level.csv"
    p.write_text("subject_id,label,prediction,dep_score,non_score\n" + "\n".join(",".join(map(str, r)) for r in rows))
    return p


def test_read_predictions_margin(tmp_path):
    out = read_predictions(_write(tmp_path, [("a", 1, 1, -1.0, -2.5), ("b", 0, 0, -3.0, -1.0)]))
    assert out["a"] == (1, 1.5, 1) and out["b"] == (0, -2.0, 0)


def test_read_predictions_rejects_sign_mismatch(tmp_path):
    with pytest.raises(ValueError):
        read_predictions(_write(tmp_path, [("a", 1, 0, -1.0, -2.5)]))


def test_read_predictions_rejects_duplicates(tmp_path):
    with pytest.raises(ValueError):
        read_predictions(_write(tmp_path, [("a", 1, 1, -1.0, -2.5), ("a", 1, 1, -1.0, -2.5)]))
