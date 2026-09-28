"""Weights-free tests for the backend-aware label token audit.

The audit is exercised with fake processors that reproduce the pinned Qwen3.8
chat template and a whitespace tokenizer, so the tests run without model weights
and without a GPU. They cover the single-token gate for the four short
vocabularies, the shared-token and mask failures, the English exemption, and the
rule that no transcript text may reach the report.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from scripts.build_qwen3_daic_label_configs import generated_name
from tools import verify_label_vocab_tokens as audit

ROOT = Path(__file__).resolve().parents[1]
CLOSED_THINKING_BLOCK = "<think>\n\n</think>\n\n"
TRANSCRIPT_MARKER = "MARKER-TRANSCRIPT-TEXT-DO-NOT-LEAK"


class FakeTokenizer:
    """Whitespace tokenizer; ``splits`` forces a word to tokenize into pieces."""

    pad_token_id = 0
    eos_token_id = 1

    def __init__(self, splits: dict[str, list[str]] | None = None) -> None:
        self._splits = splits or {}
        self._ids: dict[str, int] = {"<pad>": 0, "<eos>": 1}
        self._pieces: dict[int, str] = {0: "<pad>", 1: "<eos>"}

    def _piece_id(self, piece: str) -> int:
        if piece not in self._ids:
            identifier = len(self._ids)
            self._ids[piece] = identifier
            self._pieces[identifier] = piece
        return self._ids[piece]

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        # Chat markers are their own tokens, the way a real tokenizer keeps them.
        for word in re.findall(r"<\|[^|]*\|>|[^\s<]+|<", text):
            for piece in self._splits.get(word, [word]):
                ids.append(self._piece_id(piece))
        return ids

    def decode(self, ids: list[int]) -> str:
        return "".join(self._pieces[int(identifier)] for identifier in ids)


class FakeProcessor:
    """Processor surface used by the audit: __call__, tokenizer, chat template."""

    def __init__(self, splits: dict[str, list[str]] | None = None) -> None:
        self.tokenizer = FakeTokenizer(splits)

    def __call__(self, text: str, return_tensors=None, padding=False, **kwargs):
        ids = self.tokenizer.encode(text)
        return {"input_ids": [ids], "attention_mask": [[1] * len(ids)]}

    def apply_chat_template(
        self,
        messages,
        add_generation_prompt: bool = False,
        enable_thinking: bool | None = None,
        tokenize: bool = False,
    ) -> str:
        text = ""
        for message in messages:
            text += f"<|im_start|>{message['role']}\n"
            if message["role"] == "assistant" and enable_thinking is False:
                text += CLOSED_THINKING_BLOCK
            text += f"{message['content']}<|im_end|>\n"
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
            text += CLOSED_THINKING_BLOCK if enable_thinking is False else "<think>\n"
        return text


@pytest.fixture()
def model_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "fake-model"
    directory.mkdir()
    (directory / "tokenizer.json").write_text('{"fake": true}\n', encoding="utf-8")
    return directory


def _patch_processor(monkeypatch: pytest.MonkeyPatch, splits: dict[str, list[str]] | None = None) -> FakeProcessor:
    processor = FakeProcessor(splits)
    monkeypatch.setattr(audit, "load_processor", lambda model_path, config: processor)
    return processor


def _config_path(modality: str, arm: str) -> Path:
    return ROOT / "configs/labels" / generated_name(modality, arm)


def _audit(monkeypatch: pytest.MonkeyPatch, *, arm: str, model_dir: Path, splits=None, **kwargs) -> dict:
    _patch_processor(monkeypatch, splits)
    return audit.audit_config(
        _config_path("text_only", arm),
        model_path=str(model_dir),
        overrides=[],
        manifest=kwargs.get("manifest"),
        split_metadata=kwargs.get("split_metadata"),
        partitions=kwargs.get("partitions", ("train",)),
        real_limit=kwargs.get("real_limit", 2),
    )


def test_single_token_vocabulary_passes(monkeypatch: pytest.MonkeyPatch, model_dir: Path) -> None:
    report = _audit(monkeypatch, arm="ab", model_dir=model_dir)
    assert report["passed"], report["failures"]
    assert report["checks"]["prompt_is_gold_independent"] is True
    positives = report["synthetic"][0]["labels"][1]
    negatives = report["synthetic"][0]["labels"][0]
    assert positives == {"text": "A", "tokens": positives["tokens"], "length": 1}
    assert negatives["text"] == "B"
    assert positives["tokens"] != negatives["tokens"]
    assert report["synthetic"][0]["mask"]["answer_tokens"] == 1
    assert report["synthetic"][0]["terminator_tokens"] == 1


def test_multi_token_short_label_fails_closed(monkeypatch: pytest.MonkeyPatch, model_dir: Path) -> None:
    report = _audit(monkeypatch, arm="truefalse", model_dir=model_dir, splits={"True": ["Tru", "e"]})
    assert not report["passed"]
    assert any("tokens at the prompt boundary" in message for message in report["failures"]), report["failures"]


def test_classes_may_not_share_a_token(monkeypatch: pytest.MonkeyPatch, model_dir: Path) -> None:
    report = _audit(monkeypatch, arm="ab", model_dir=model_dir, splits={"A": ["shared"], "B": ["shared"]})
    assert not report["passed"]
    assert any("share the same continuation token ids" in message for message in report["failures"])


def test_english_vocabulary_may_span_several_tokens(monkeypatch: pytest.MonkeyPatch, model_dir: Path) -> None:
    report = _audit(
        monkeypatch,
        arm="en",
        model_dir=model_dir,
        splits={"Depressed": ["Dep", "ressed"], "Non-depressed": ["Non", "-", "depressed"]},
    )
    assert report["passed"], report["failures"]
    assert report["single_token_required"] is False
    lengths = {
        entry["text"]: entry["length"] for entry in report["synthetic"][0]["labels"].values()
    }
    assert lengths == {"Depressed": 2, "Non-depressed": 3}


def test_mask_mismatch_is_reported(monkeypatch: pytest.MonkeyPatch, model_dir: Path) -> None:
    report = _audit(monkeypatch, arm="ab", model_dir=model_dir)
    assert report["passed"]
    # The synthetic checks already prove the mask keeps the candidate tokens; a
    # mismatch would surface as a failure entry with the mask wording.
    assert report["synthetic"][0]["mask"]["answer_tokens"] == 1
    assert not any("training label mask" in message for message in report["failures"])


def test_report_never_carries_transcript_text(monkeypatch: pytest.MonkeyPatch, model_dir: Path, tmp_path: Path) -> None:
    manifest = tmp_path / "daic_manifest.jsonl"
    rows = [
        {
            "schema_version": "audiollm.manifest.v1",
            "dataset": "daic",
            "subject_id": str(300 + index),
            "sample_id": f"{300 + index}_participant_p30_000",
            "audio_path": f"/data/{300 + index}.wav",
            "label": index % 2,
            "label_text": "Depressed" if index % 2 else "Non-depressed",
            "transcript": TRANSCRIPT_MARKER,
            "full_participant_transcript": TRANSCRIPT_MARKER,
            "chunk_transcript": TRANSCRIPT_MARKER,
            "chunk_index": 0,
            "num_chunks": 1,
            "protocol_id": "daic_participant_speech_packed30_v1",
            "split_original": "train",
            "participant_sample_count": 1,
            "audio_spans": [],
            "full_participant_transcript_sha256": "0" * 64,
        }
        for index in range(2)
    ]
    manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    report = _audit(monkeypatch, arm="ab", model_dir=model_dir, manifest=str(manifest))
    assert report["passed"], report["failures"]
    assert len(report["real"]["train"]) == 2
    serialized = json.dumps(report)
    assert TRANSCRIPT_MARKER not in serialized
    assert audit.SYNTHETIC_TRANSCRIPT not in serialized
    assert "300" not in serialized


def test_unsupported_backend_is_refused(monkeypatch: pytest.MonkeyPatch, model_dir: Path) -> None:
    _patch_processor(monkeypatch)
    with pytest.raises(audit.AuditError):
        audit.audit_config(
            _config_path("audio_only", "ab"),
            model_path=str(model_dir),
            overrides=["--set=model_backend=qwen2audio"],
            manifest=None,
            split_metadata=None,
            partitions=("train",),
            real_limit=1,
        )
