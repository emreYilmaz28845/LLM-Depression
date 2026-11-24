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


def _write_manifest(path, texts):
    import hashlib
    import json
    rows = []
    for i, t in enumerate(texts):
        rows.append(json.dumps({"subject_id": str(300 + i), "split_original": "train", "label": i % 2,
                                "full_participant_transcript": t,
                                "full_participant_transcript_sha256": hashlib.sha256(t.encode()).hexdigest()}))
    path.write_text("\n".join(rows) + "\n")


def test_load_subjects_rejects_transcripts_capped_at_4000_chars(tmp_path):
    import pytest
    m = tmp_path / "m.jsonl"
    _write_manifest(m, ["short text", "x" * mod.TRUNCATION_CAP_CHARS])
    with pytest.raises(ValueError, match="exactly 4000 chars"):
        mod.load_subjects(m)
    _write_manifest(m, ["short text", "x" * (mod.TRUNCATION_CAP_CHARS + 1)])
    assert len(mod.load_subjects(m)) == 2


def test_check_reference_manifest_rejects_other_files(tmp_path):
    import pytest
    m = tmp_path / "m.jsonl"
    _write_manifest(m, ["short text"])
    with pytest.raises(ValueError, match="not the reference DAIC manifest"):
        mod.check_reference_manifest(m)
    assert mod.DEFAULT_MANIFEST.name == "daic_manifest.jsonl"
    assert len(mod.REFERENCE_MANIFEST_SHA256) == 64


def test_cited_examples_reads_singular_plural_and_lists():
    import importlib.util as u
    from pathlib import Path
    sp = u.spec_from_file_location("bes", Path(__file__).resolve().parents[1] / "experiments/manyshot_claude/build_explanation_sheet.py")
    b = u.module_from_spec(sp)
    sp.loader.exec_module(b)
    f = b.cited_examples
    assert f("Most like Example 12 (Depressed) and Example 7.", 142) == [7, 12]
    assert f("Like the examples (e.g., Examples 16, 25, 33) were usually labelled Non-depressed.", 142) == [16, 25, 33]
    assert f("Like depressed examples (e.g., 47, 44, 95), they show anhedonia.", 142) == [44, 47, 95]
    assert f("The depressed examples point to a PHQ-8 score of 10 or higher.", 142) == []
    assert f("Examples 7, 14 and 43, were labelled; example 999 is out of range.", 142) == [7, 14, 43]


def test_cited_examples_ignores_phq_thresholds():
    import importlib.util as u
    from pathlib import Path
    sp = u.spec_from_file_location("bes2", Path(__file__).resolve().parents[1] / "experiments/manyshot_claude/build_explanation_sheet.py")
    b = u.module_from_spec(sp)
    sp.loader.exec_module(b)
    assert b.cited_examples("Like Example 17, the transcript fits the examples with a PHQ-8 score below 10.", 142) == [17]
