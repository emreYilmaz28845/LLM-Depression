import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "run_claude_manyshot", Path(__file__).resolve().parents[1] / "experiments/manyshot_claude/run_claude_manyshot.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_parse_answer_valid_and_invalid():
    assert mod.parse_answer('{"label": "Depressed", "explanation": "x"}') == (1, "x")
    assert mod.parse_answer('text {"label": "non-depressed", "explanation": "y"} end')[0] == 0
    assert mod.parse_answer("Depressed")[0] is None
    assert mod.parse_answer('{"label": "maybe"}')[0] is None


def test_invalid_counts_as_wrong():
    recs = [{"label": 1, "pred": None}, {"label": 0, "pred": 0}]
    m = mod.metrics(recs)
    assert m["invalid"] == 1 and m["uar"] == 0.5
