import json
from pathlib import Path

import pytest

from experiments.manyshot_qwen_explain import prompt_v3 as pv
from experiments.manyshot_qwen_explain import run_qwen_manyshot_explain as mod


class FakeTokenizer:
    """Qwen-style chat template; one token per character; '|' plays <|im_end|>."""

    eos_token_id = ord("|")

    def apply_chat_template(self, msgs, add_generation_prompt, enable_thinking, tokenize):
        assert add_generation_prompt and enable_thinking is False and tokenize is False
        text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in msgs)
        return text + "<|im_start|>assistant\n<think>\n\n</think>\n\n"

    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": [ord(c) for c in text]}

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(i) for i in ids)

    def convert_tokens_to_ids(self, token):
        return ord("|")


def _subjects(n, splits=None):
    out = {}
    for i in range(n):
        sid = str(300 + i)
        out[sid] = {"subject_id": sid, "split": (splits or (lambda k: "all"))(i), "label": int(i % 3 == 0),
                    "text": f"words of {sid}", "score": float(i)}
    return out


# --------------------------------------------------------------------------- prompt


def test_prompt_has_advisor_sentence_label_rules_and_explanation_first_format():
    for ds, rule in (("daic", "PHQ-8 >= 10"), ("turkish", "BDI >= 17")):
        sp = pv.system_prompt(ds)
        assert "Base your answer on the examples in the prompt and explain your answer." in sp
        assert rule in sp
        assert sp.index('"explanation"') < sp.index('"label"')
        assert "most similar" not in sp  # advisor dropped the 2-3 similar examples requirement
    block = pv.shot_block([{"text": "a", "label": 1}, {"text": "b", "label": 0}])
    assert "### Example 1\nTranscript:\na\nLabel: Depressed" in block
    assert "### Example 2\nTranscript:\nb\nLabel: Non-depressed" in block


def test_parse_answer_and_cited_examples():
    assert pv.parse_answer('{"explanation": "Like Example 3.", "label": "Depressed"}') == (1, "Like Example 3.")
    assert pv.parse_answer('x {"explanation": "e", "label": "non-depressed"} y')[0] == 0
    assert pv.parse_answer('{"explanation": "e", "label": "unsure"}')[0] is None
    assert pv.parse_answer("Depressed")[0] is None
    assert pv.cited_examples("Example 12 (Depressed) and Examples 3 and 7, also example #5, Example 12") == [12, 3, 7, 5]
    assert pv.cited_examples("no numbers here") == []


# --------------------------------------------------------------------------- tasks


def test_daic_tasks_share_shots_and_exclude_test():
    subjects = _subjects(30, lambda i: ("train", "val", "test")[i % 3])
    tasks = mod.build_tasks("daic", subjects, 0)
    test_ids = {s for s, r in subjects.items() if r["split"] == "test"}
    assert {t["target"]["subject_id"] for t in tasks} == test_ids
    first = [s["subject_id"] for s in tasks[0]["shots"]]
    assert len(first) == 20 and not set(first) & test_ids
    assert all([s["subject_id"] for s in t["shots"]] == first for t in tasks)
    assert first == [s["subject_id"] for s in mod.build_tasks("daic", subjects, 0)[0]["shots"]]


def test_turkish_lopo_drops_only_the_target_from_one_global_order():
    subjects = _subjects(12)
    tasks = mod.build_tasks("turkish", subjects, 0)
    assert len(tasks) == 12
    order = None
    for t in tasks:
        ids = [s["subject_id"] for s in t["shots"]]
        assert len(ids) == 11 and t["target"]["subject_id"] not in ids
        full = ids[:]
        if order is None:
            order = full
        # removing the target from the global order gives the same relative order for everyone
        common = [s for s in order if s in ids]
        assert common == [s for s in ids if s in order]


def test_load_turkish_joins_both_question_sets_and_checks_label_rule():
    rows = []
    for sid, score in (("t1", 20.0), ("t2", 5.0)):
        for cond in ("pos_only_t17", "negative_only_t17"):
            for c in (2, 1):
                rows.append({"dataset": "turkish", "subject_id": sid, "dataset_variant": cond,
                             "sample_id": f"{sid}_{cond}_{c}", "chunk_id": str(c), "question_id": "",
                             "audio_path": f"/a/{sid}/{cond}/{c}.wav", "transcript": f"{cond} chunk {c}",
                             "label": int(score >= 17), "score": score})
    subjects = mod.load_turkish(rows)
    text = subjects["t1"]["text"]
    assert text.startswith("[Positive question set]\npos_only_t17 chunk 1\npos_only_t17 chunk 2")
    assert "[Negative question set]\nnegative_only_t17 chunk 1" in text
    assert subjects["t1"]["label"] == 1 and subjects["t2"]["label"] == 0
    rows[0] = {**rows[0], "label": 0}
    with pytest.raises(ValueError):
        mod.load_turkish(rows)
    bad = [{**r, "label": 0} for r in rows if r["subject_id"] == "t1"]
    with pytest.raises(ValueError, match="score >= 17"):
        mod.load_turkish(bad)


def test_plan_shares_daic_prefix_and_gives_each_lopo_target_its_own():
    r = mod.Renderer(FakeTokenizer())
    daic = mod.plan("daic", mod.build_tasks("daic", _subjects(9, lambda i: ("train", "val", "test")[i % 3]), 0), r)
    assert len({p["prefix_sha256"] for p in daic}) == 1
    tr = mod.plan("turkish", mod.build_tasks("turkish", _subjects(5), 0), r)
    assert len({p["prefix_sha256"] for p in tr}) == 5
    for p in daic + tr:
        text = "".join(chr(i) for i in p["prefix_ids"] + p["suffix_ids"])
        assert text.endswith("Return the JSON object now.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
        assert p["target"]["text"] not in "".join(chr(i) for i in p["prefix_ids"])


def test_label_candidates_split_at_the_label_value():
    tok = FakeTokenizer()
    before, cands = mod.label_candidates(tok, '{"explanation": "x", "label": "Depressed"}')
    assert before == '{"explanation": "x", "label": "'
    assert ["".join(map(chr, c)) for c in cands] == ["Depressed", "Non-depressed"]
    assert mod.label_candidates(tok, "no label") is None


# --------------------------------------------------------------------------- generator (tiny real model)


@pytest.fixture(scope="module")
def tiny_generator():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    cfg = transformers.LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256)
    model = transformers.LlamaForCausalLM(cfg).eval()
    gen = object.__new__(mod.QwenGenerator)
    gen.torch, gen.model, gen.decoder, gen.head = torch, model, model.model, model.lm_head
    gen.device, gen.chunk = torch.device("cpu"), 5
    return gen


def test_generation_from_copied_prefix_cache_equals_fresh_prefill(tiny_generator):
    g = tiny_generator
    prefix, suffix = list(range(1, 23)), [30, 31, 32, 33, 34, 35, 36]
    cache = g.prefill(prefix)
    a, _ = g.generate(g.copy(cache), suffix, 12, eos=set())
    b, _ = g.generate(g.copy(cache), suffix, 12, eos=set())
    c, _ = g.generate(None, prefix + suffix, 12, eos=set())
    assert a == b == c and len(a) == 12
    g.chunk = 1000  # chunking does not change the answer
    d, _ = g.generate(None, prefix + suffix, 12, eos=set())
    g.chunk = 5
    assert d == a


def test_candidate_scores_match_a_full_forward(tiny_generator):
    torch = pytest.importorskip("torch")
    g = tiny_generator
    prefix, context, cands = list(range(1, 15)), [20, 21, 22], [[40, 41], [42, 43, 44]]
    scores = g.candidate_scores(g.copy(g.prefill(prefix)), context, cands)
    for cand, s in zip(cands, scores):
        ids = prefix + context + cand
        with torch.inference_mode():
            logits = g.model(input_ids=torch.tensor([ids])).logits[0].float()
        lp = torch.log_softmax(logits, -1)
        n = len(prefix + context)
        ref = float(torch.stack([lp[n - 1 + j, t] for j, t in enumerate(cand)]).mean())
        assert s == pytest.approx(ref, abs=1e-5)


# --------------------------------------------------------------------------- end to end (fake model)


class FakeGenerator:
    """Answers from a script keyed by the target transcript; checks the prefix is reused."""

    answers = {}

    def __init__(self, model_dir, chunk):
        self.prefills = 0

    def prefill(self, ids):
        self.prefills += 1
        return {"ids": list(ids)}

    def copy(self, cache):
        return {"ids": list(cache["ids"])}

    def generate(self, cache, suffix, max_new, eos):
        text = "".join(map(chr, suffix))
        for key, answer in self.answers.items():
            if f"words of {key}\n" in text:
                return [ord(c) for c in answer] + [ord("|")], True
        raise AssertionError("unknown target")

    def candidate_scores(self, cache, context, candidates):
        return [-0.1, -2.0]


def _write_daic_manifest(path: Path, n=12):
    with path.open("w") as f:
        for i in range(n):
            sid = str(300 + i)
            f.write(json.dumps({"subject_id": sid, "split_original": ("train", "val", "test")[i % 3],
                                "label": int(i % 2 == 0), "full_participant_transcript": f"words of {sid}"}) + "\n")


def test_main_end_to_end_resume_invalid_and_citations(tmp_path, monkeypatch):
    import transformers

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: FakeTokenizer())
    manifest = tmp_path / "m.jsonl"
    _write_daic_manifest(manifest)
    # test subjects: 302 (label 1), 305 (0), 308 (1), 311 (0)
    FakeGenerator.answers = {
        "302": '{"explanation": "Similar to Example 1.", "label": "Depressed"}',
        "305": '{"explanation": "Unclear.", "label": "maybe"}',
        "308": '{"explanation": "Like Examples 2 and 3.", "label": "Non-depressed"}',
        "311": '{"explanation": "Calm.", "label": "Non-depressed"}',
    }
    out = tmp_path / "out"
    base = ["--dataset", "daic", "--manifest", str(manifest), "--model-dir", str(tmp_path),
            "--out-dir", str(out), "--allow-other-manifest"]
    with pytest.raises(SystemExit, match="not the daic reference"):
        mod.main(base[:-1])
    mod.main(base + ["--limit", "2"], generator_factory=FakeGenerator)
    assert len((out / "responses.jsonl").read_text().splitlines()) == 2
    mod.main(base, generator_factory=FakeGenerator)  # resumes: only the two missing targets
    recs = [json.loads(line) for line in (out / "responses.jsonl").read_text().splitlines()]
    assert [r["subject_id"] for r in recs] == ["302", "305", "308", "311"]
    by = {r["subject_id"]: r for r in recs}
    assert by["305"]["pred"] is None and by["305"]["pred_text"] == "INVALID"
    info = json.loads((out / "run_info.json").read_text())
    shots = info["example_subjects"]
    assert by["308"]["cited_examples"] == [{"number": 2, "subject_id": shots[1], "label": by["308"]["cited_examples"][0]["label"]},
                                           {"number": 3, "subject_id": shots[2], "label": by["308"]["cited_examples"][1]["label"]}]
    assert by["302"]["label_margin"] == pytest.approx(1.9)
    m = json.loads((out / "metrics.json").read_text())
    assert m["n_subjects"] == 4 and m["invalid"] == 1 and m["complete"]
    # 302 TP, 305 INVALID -> wrong (FP), 308 FN, 311 TN
    assert m["confusion"] == {"tp": 1, "fn": 1, "fp": 1, "tn": 1}
    assert info["manifest_is_reference"] is False


def test_reference_hashes_are_pinned():
    assert mod.REFERENCE_MANIFEST_SHA256["daic"].startswith("8d96abc2a411")
    assert mod.REFERENCE_MANIFEST_SHA256["turkish"].startswith("a0daf6658f11")


def test_metrics_match_sklearn():
    sk = pytest.importorskip("sklearn.metrics")
    import random as _r

    rng = _r.Random(3)
    recs = [{"label": rng.randint(0, 1), "pred": rng.choice([0, 1, None]), "label_margin": rng.random()}
            for _ in range(60)]
    m = mod.metrics(recs)
    y = [r["label"] for r in recs]
    p = [r["pred"] if r["pred"] is not None else 1 - r["label"] for r in recs]
    assert m["macro_f1"] == round(sk.f1_score(y, p, average="macro"), 4)
    assert m["positive_f1"] == round(sk.f1_score(y, p), 4)
    assert m["uar"] == round(sk.recall_score(y, p, average="macro"), 4)
    assert m["auroc_label_margin_secondary"] == round(sk.roc_auc_score(y, [r["label_margin"] for r in recs]), 4)


def test_shards_write_separate_files_and_metrics_need_all(tmp_path, monkeypatch):
    import transformers

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: FakeTokenizer())
    manifest = tmp_path / "m.jsonl"
    _write_daic_manifest(manifest)
    FakeGenerator.answers = {s: '{"explanation": "e", "label": "Depressed"}' for s in ("302", "305", "308", "311")}
    out = tmp_path / "out"
    base = ["--dataset", "daic", "--manifest", str(manifest), "--model-dir", str(tmp_path),
            "--out-dir", str(out), "--allow-other-manifest"]
    mod.main(base + ["--shard", "0/2"], generator_factory=FakeGenerator)
    assert not (out / "metrics.json").exists()
    mod.main(base + ["--shard", "1/2"], generator_factory=FakeGenerator)
    a = [json.loads(x)["subject_id"] for x in (out / "responses.shard0of2.jsonl").read_text().splitlines()]
    b = [json.loads(x)["subject_id"] for x in (out / "responses.shard1of2.jsonl").read_text().splitlines()]
    assert a == ["302", "308"] and b == ["305", "311"]
    assert json.loads((out / "metrics.json").read_text())["n_subjects"] == 4


def test_smoke_gate(tmp_path):
    from experiments.manyshot_qwen_explain import check_smoke

    good = [{"subject_id": s, "pred": 1, "finished": True} for s in ("a", "b", "c")]
    (tmp_path / "responses.jsonl").write_text("\n".join(json.dumps(r) for r in good))
    (tmp_path / "checks.jsonl").write_text(json.dumps({"subject_id": "a", "determinism_identical": True,
                                                       "uncached_identical": False}))
    assert check_smoke.main([str(tmp_path)]) == 0
    (tmp_path / "responses.jsonl").write_text("\n".join(json.dumps({**r, "pred": None}) if r["subject_id"] == "b"
                                                        else json.dumps(r) for r in good))
    assert check_smoke.main([str(tmp_path)]) == 1
    (tmp_path / "responses.jsonl").write_text("\n".join(json.dumps(r) for r in good))
    (tmp_path / "checks.jsonl").write_text(json.dumps({"subject_id": "a", "determinism_identical": False}))
    assert check_smoke.main([str(tmp_path)]) == 1
    assert check_smoke.main([str(tmp_path), "--min-targets", "4"]) == 1
