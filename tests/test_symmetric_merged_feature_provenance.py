from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.features.qwen_hidden_collator import PromptOnlyExtractionCollator


class _TextProcessor:
    def __call__(self, **kwargs):
        del kwargs
        return {
            "input_ids": torch.tensor([[1]], dtype=torch.long),
            "attention_mask": torch.tensor([[1]], dtype=torch.long),
        }


def test_feature_rows_preserve_response_hierarchy_fields() -> None:
    example = {
        "prompt_text": "prompt",
        "dataset": "cmdc",
        "sample_id": "cmdc::s::window0",
        "subject_id": "cmdc::s",
        "label": 0,
        "partition": "outer_train",
        "fold": 0,
        "response_id": "",
        "turn_key": "",
        "question_id": "Q1",
        "window_id": "window0",
        "prompt_id": "Q1",
    }
    _, metadata = PromptOnlyExtractionCollator(_TextProcessor())([example])
    assert metadata[0]["question_id"] == "Q1"
    assert metadata[0]["window_id"] == "window0"
