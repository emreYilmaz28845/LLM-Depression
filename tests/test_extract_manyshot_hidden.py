from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from src.features.extract_manyshot_hidden import (
    HIDDEN_SIZE,
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
            add_generation_prompt=True, enable_thinking=False, tokenize=False)
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
    assert tok.decode(s1).endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert r.suffix(t1, t1[:r.cut(t1)], p1, verify=True) == s1


@pytest.mark.skipif(not (QWEN38_DIR / "tokenizer_config.json").exists(), reason="local Qwen3.8 tokenizer absent")
def test_zeroshot_text_equals_frozen_extractor_prompt():
    from transformers import AutoTokenizer

    from src.features.extract_manyshot_hidden import _reference_prompt
    from src.model.qwen38_lora import Qwen38ProcessorAdapter, render_qwen38_prompt

    tok = AutoTokenizer.from_pretrained(str(QWEN38_DIR), local_files_only=True)
    r = Renderer(tok)
    text = r.text(messages("System prompt.", [], "A transcript."))
    assert text == render_qwen38_prompt(Qwen38ProcessorAdapter(tok), "System prompt.", "A transcript.")
    assert text == _reference_prompt(r, "System prompt.", {"user": "A transcript."})


def _loo_args(**kw):
    base = dict(mode="manyshot", shot_scheme="loo", inner_groups=5, inner_seed=1337, shot_seed=0,
                check_uncached=0, prefill_chunk=8192, model_dir=Path("model"), only_sets=None)
    base.update(kw)
    return Namespace(**base)


def _daic_split(pool):
    ids = sorted(pool)
    return {"train_subject_ids": ids[:10], "selection_subject_ids": ids[10:14], "final_eval_subject_ids": ids[14:]}


def test_loo_plan_one_set_per_fit_subject_plus_one_shared_set():
    pool = _pool(18)
    split = _daic_split(pool)
    plan, sets, _ = plan_fold({"protocol": "daic_official"}, split, pool, 0, _loo_args(), Renderer(FakeTokenizer()))
    assert len(sets) == len(plan["fit"]) + 1
    full_order = [r["row"]["sample_id"] for r in sets[-1]["shots"]]
    assert {r["row"]["subject_id"] for r in sets[-1]["shots"]} == set(plan["fit"])
    assert {r["subject"] for r in sets[-1]["rows"]} == set(plan["others"])
    assert not any(r["own"] for r in sets[-1]["rows"])
    for st, subject in zip(sets[:-1], plan["fit"]):
        donors = {r["row"]["subject_id"] for r in st["shots"]}
        assert donors == set(plan["fit"]) - {subject}
        assert {r["subject"] for r in st["rows"]} == {subject} and all(r["own"] for r in st["rows"])
        # same fixed order as the shared set, with only this subject's turns removed
        assert [r["row"]["sample_id"] for r in st["shots"]] == [
            s for s in full_order if not s.startswith(subject + "_")]
    # dev/test subjects never appear as shots anywhere
    assert not {r["row"]["subject_id"] for st in sets for r in st["shots"]} & set(plan["others"])


class FakeEncoder:
    """Vector = hash of the prompt suffix; scores favour the true label slightly."""

    def __init__(self, labels):
        self.labels = labels

    def prefix(self, ids):
        return list(ids)

    def _vec(self, cache, suffix):
        rng = np.random.default_rng(abs(hash((len(cache), tuple(suffix)))) % (2**32))
        return rng.normal(size=HIDDEN_SIZE).astype(np.float32)

    def row(self, cache, suffix):
        return self._vec(cache, suffix)

    def row_scores(self, cache, suffix, candidates):
        return self._vec(cache, suffix), [-0.1, -0.2], 0.0


def test_loo_sharded_run_resumes_and_assembles(tmp_path, monkeypatch):
    import json

    import src.features.extract_manyshot_hidden as m

    pool = _pool(18, rows_per_subject=1)
    split = _daic_split(pool)
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(split))
    args = _loo_args(answer_scores=True)
    plan, sets, _ = plan_fold({"protocol": "daic_official"}, split, pool, 0, args, Renderer(FakeTokenizer()))
    out_dir = tmp_path / "daic" / "fold_0"
    enc = FakeEncoder(None)
    set_dir = m.sets_dir(out_dir)
    set_dir.mkdir(parents=True)
    for shard in ((0, 2), (1, 2)):
        todo = m.set_todo(out_dir, sets, shard)
        assert todo and all(i % 2 == shard[0] for i in todo)
        for i in todo:
            m.run_set(enc, sets[i], i, set_dir, args, [[1], [2]], None, "daic", 0)
        assert m.set_todo(out_dir, sets, shard) == []  # resume: nothing left for this shard
    prov = {k: "x" for k in ("dataset", "condition", "input_modality", "protocol_id", "manifest_sha256", "manifest",
                             "manifest_file_sha256", "reference_checkpoint_dir", "saved_run_config_sha256",
                             "saved_split_sha256", "split_metadata_sha256")}
    m.assemble({"name": "daic", "protocol": "daic_official"}, prov, 0, split_path, plan, sets, out_dir, args, None)
    meta = json.loads((out_dir / "extraction_metadata.json").read_text())
    assert meta["manyshot"]["shot_scheme"] == "loo" and meta["manyshot"]["n_shot_sets"] == 11
    train = np.load(out_dir / "outer_train.npz")["vectors"]
    assert train.shape == (10, HIDDEN_SIZE)
    assert np.load(out_dir / "final_eval.npz")["vectors"].shape == (4, HIDDEN_SIZE)
    assert np.load(out_dir / "dev" / "final_eval.npz")["vectors"].shape == (4, HIDDEN_SIZE)
    ans = [json.loads(x) for x in (out_dir / "final_eval_answer_scores.jsonl").read_text().splitlines()]
    assert len(ans) == 4 and ans[0]["logprob_Depressed"] == pytest.approx(-0.1)
    assert m.set_todo(out_dir, sets, None) == []


def test_assemble_rejects_a_stale_saved_set(tmp_path):
    import json

    import src.features.extract_manyshot_hidden as m

    pool = _pool(18, rows_per_subject=1)
    split = _daic_split(pool)
    args = _loo_args(answer_scores=False)
    plan, sets, _ = plan_fold({"protocol": "daic_official"}, split, pool, 0, args, Renderer(FakeTokenizer()))
    out_dir = tmp_path / "fold_0"
    set_dir = m.sets_dir(out_dir)
    set_dir.mkdir(parents=True)
    for i in range(len(sets)):
        m.run_set(FakeEncoder(None), sets[i], i, set_dir, args, None, None, "daic", 0)
    meta = json.loads((set_dir / "set_0000.json").read_text())
    meta["prefix_sha256"] = "stale"
    (set_dir / "set_0000.json").write_text(json.dumps(meta))
    with pytest.raises(AssertionError, match="does not match the current plan"):
        m.assemble({"name": "d", "protocol": "daic_official"}, {}, 0, tmp_path, plan, sets, out_dir, args, None)
