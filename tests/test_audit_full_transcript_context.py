from scripts.audit_full_transcript_context import audit_dataset, natural_key


class FakeTokenizer:
    unk_token_id = -1

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        tokens = text.replace("<|AUDIO|>", " <|AUDIO|> ").split()
        return {"input_ids": [99 if token == "<|AUDIO|>" else 1 for token in tokens]}


def test_natural_key_orders_numeric_parts():
    assert sorted(["Q10", "Q2", "Q1"], key=natural_key) == ["Q1", "Q2", "Q10"]


def test_audit_expands_single_audio_placeholder():
    detail, summary = audit_dataset(
        "test",
        {"subject": "short transcript"},
        {"subject"},
        FakeTokenizer(),
        audio_token_id=99,
        audio_tokens=750,
        context_limit=8192,
        safety_margin=128,
    )
    assert detail[0]["audio_embedding_tokens_30sec"] == 750
    assert detail[0]["effective_multimodal_tokens"] == (
        detail[0]["text_tokens_with_audio_placeholder_and_label"] - 1 + 750
    )
    assert detail[0]["fits"] is True
    assert summary["audited_subjects"] == 1
