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


def test_zeroshot_prompt_has_no_examples_and_same_answer_format():
    zs = mod.SYSTEM_PROMPT_ZEROSHOT
    assert "example" not in zs.lower()
    assert '{"label": "Depressed" or "Non-depressed", "explanation": "<3-6 sentences>"}' in zs
    assert "PHQ-8 >= 10" in zs
    assert "Final participant" not in mod.target_block({"text": "t"}, zero_shot=True)


def test_advisor_prompt_keeps_sentence_verbatim_and_drops_v2_naming_rule():
    adv = mod.PROMPTS["advisor"]
    assert "Base your answer on the examples in the prompt and explain your answer." in adv
    assert "2-3" not in adv and "most similar" not in adv
    assert adv.startswith(mod.SYSTEM_PROMPT.split("Answer with exactly")[0])
    assert adv.endswith(mod.ANSWER_FORMAT)
    assert "### Example 1\n" in mod.shot_block([{"text": "t", "label": 1}])


def test_advisor_zeroshot_prompt_is_advisor_prompt_without_examples():
    zs = mod.SYSTEM_PROMPT_ZEROSHOT_ADVISOR
    assert "example" not in zs.lower()
    assert "Explain your answer." in zs and "PHQ-8 >= 10" in zs
    assert zs.endswith(mod.ANSWER_FORMAT)
