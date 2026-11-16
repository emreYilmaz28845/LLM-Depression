from argparse import Namespace
from pathlib import Path

import pytest

from src.features.extract_manyshot_hidden import (
    NOTE,
    Renderer,
    fold_plan,
    inner_groups,
    messages,
    plan_fold,
)

QWEN38_DIR = Path("/media/emre/Backup/AudioLLM/models/Qwen3.8-27B")


class FakeTokenizer:
    """Qwen-style chat template; one token per character."""

    def apply_chat_template(self, msgs, add_generation_prompt, enable_thinking, tokenize):
        assert add_generation_prompt and enable_thinking is False and tokenize is False
        text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in msgs)
        return text + "<|im_start|>assistant\n<think>\n\n</think>\n\n"

    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": [ord(c) for c in text]}


def _pool(n=20, rows_per_subject=2):
    pool = {}
    for i in range(n):
        s = f"s{i:02d}"
        pool[s] = [{"row": {"sample_id": f"{s}_{k}", "subject_id": s, "label": int(i % 3 == 0)},
                    "system": "SYS", "user": f"transcript {s} part {k}"} for k in range(rows_per_subject)]
    return pool


def _args(mode="manyshot"):
    return Namespace(mode=mode, inner_groups=5, inner_seed=1337, shot_seed=0)


def test_inner_groups_stratified_deterministic_and_complete():
    pool = _pool(23)
    fit = sorted(pool)
    labels = {s: pool[s][0]["row"]["label"] for s in pool}
    groups = inner_groups(fit, labels, 5, 1337, 0)
    assert groups == inner_groups(fit, labels, 5, 1337, 0)
    assert groups != inner_groups(fit, labels, 5, 1337, 1)
    assert sorted(s for g in groups for s in g) == fit
    positives = [sum(labels[s] for s in g) for g in groups]
    assert max(positives) - min(positives) <= 1


def test_fold_plan_protocols_and_overlap():
    pool = _pool(20)
    ids = sorted(pool)
    split = {"train_subject_ids": ids[:12], "selection_subject_ids": ids[12:16], "final_eval_subject_ids": ids[16:]}
    p = fold_plan({"protocol": "train_val_test"}, split, pool)
    assert p["outer_train"] == ids[:16] and p["final_eval"] == ids[16:] and p["others"] == ids[12:]
    p = fold_plan({"protocol": "daic_official"}, split, pool)
    assert p["outer_train"] == ids[:12] and p["dev"] == ids[12:16] and p["final_eval"] == ids[16:]
    p = fold_plan({"protocol": "train_val"}, {**split, "final_eval_subject_ids": []}, pool)
    assert p["outer_train"] == ids[:12] and p["final_eval"] == ids[12:16]
    with pytest.raises(ValueError, match="overlap"):
        fold_plan({"protocol": "train_val_test"}, {**split, "selection_subject_ids": ids[10:16]}, pool)
    with pytest.raises(ValueError, match="not in the prompt pool"):
        fold_plan({"protocol": "train_val_test"}, {**split, "final_eval_subject_ids": ["nope"]}, pool)


def test_plan_fold_never_shows_a_subject_its_own_label():
    pool = _pool(20)
    ids = sorted(pool)
    split = {"train_subject_ids": ids[:12], "selection_subject_ids": ids[12:16], "final_eval_subject_ids": ids[16:]}
    route = {"protocol": "train_val_test"}
    plan, sets, _ = plan_fold(route, split, pool, 0, _args(), Renderer(FakeTokenizer()))
    assert len(sets) == 5
    own_count = {}
    for st in sets:
        donors = {r["row"]["subject_id"] for r in st["shots"]}
        assert donors <= set(plan["fit"])
        embedded = {r["subject"] for r in st["rows"]}
        assert not embedded & donors
        assert set(plan["others"]) <= embedded
        for r in st["rows"]:
            if r["own"]:
                own_count[r["subject"]] = own_count.get(r["subject"], 0) + 1
        assert NOTE in "".join(chr(c) for c in st["prefix_ids"])
    # each fit subject is embedded as a fit row in exactly one shot set (once per row)
    assert set(own_count) == set(plan["fit"]) and set(own_count.values()) == {2}


def test_zeroshot_plan_has_one_set_without_note_and_matches_reference_prompt():
    pool = _pool(10, rows_per_subject=1)
    ids = sorted(pool)
    tok = FakeTokenizer()
    for s in pool.values():
        s[0]["reference_prompt_text"] = tok.apply_chat_template(
            [{"role": "system", "content": "SYS"}, {"role": "user", "content": s[0]["user"]}],
            add_generation_prompt=True, enable_thinking=False, tokenize=False).replace("<think>\n\n</think>\n\n", "")
    split = {"train_subject_ids": ids[:6], "selection_subject_ids": ids[6:]}
    _, sets, _ = plan_fold({"protocol": "train_val"}, split, pool, 0, _args("zeroshot"), Renderer(tok))
    assert len(sets) == 1 and sets[0]["shots"] == []
    assert NOTE not in "".join(chr(c) for c in sets[0]["prefix_ids"])
    pool[ids[0]][0]["reference_prompt_text"] = "different"
    with pytest.raises(AssertionError, match="differs from the reference"):
        plan_fold({"protocol": "train_val"}, split, pool, 0, _args("zeroshot"), Renderer(tok))


@pytest.mark.skipif(not (QWEN38_DIR / "tokenizer_config.json").exists(), reason="local Qwen3.8 tokenizer absent")
def test_real_qwen38_template_prefix_is_shared_across_rows():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(QWEN38_DIR), local_files_only=True)
    r = Renderer(tok)
    shots = [{"row": {"label": 1}, "user": "Transcript A. Answer with exactly one label: Depressed or Non-depressed."},
             {"row": {"label": 0}, "user": "Transcript B. Answer with exactly one label: Depressed or Non-depressed."}]
    p1, s1, t1 = r.split(r.text(messages("System prompt.", shots, "Final one.")))
    p2, s2, t2 = r.split(r.text(messages("System prompt.", shots, "Another final transcript, longer.")))
    assert p1 == p2 and s1 != s2
    assert tok(t1)["input_ids"] == p1 + s1
    assert "Non-depressed" in tok.decode(p1) and "Final one." in tok.decode(s1)
    assert tok.decode(s1).endswith("<|im_start|>assistant\n")
    assert r.suffix(t1, t1[:r.cut(t1)], p1, verify=True) == s1
