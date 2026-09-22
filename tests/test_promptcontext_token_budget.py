from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.audit_promptcontext_token_budget import main as audit_main

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# The locally built harmonized manifests live in the main checkout; worktrees do
# not receive gitignored paths.
LOCAL_MANIFEST_ROOT = Path(
    "/home/emre/Projects/AudioLLM/LLM-Depression/outputs/manifests_harmonized"
)
LOCAL_TOKENIZER = Path("/media/emre/Backup/AudioLLM/models/Qwen3.8-27B")


def test_audit_records_a_proxy_reason_when_the_tokenizer_is_missing(tmp_path: Path) -> None:
    output = tmp_path / "audit.json"
    assert (
        audit_main(
            [
                "--output",
                str(output),
                "--tokenizer",
                str(tmp_path / "no-such-tokenizer"),
                "--max-subjects",
                "1",
            ]
        )
        == 0
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["measurement_kind"] == "not_measured"
    assert payload["tokenizer_error"]
    assert payload["results"]
    assert all(row["measured"] is False for row in payload["results"])
    assert all("tokenizer" in row["reason"] for row in payload["results"])
    assert payload["any_cell_exceeds_context_limit"] is False


@pytest.mark.skipif(
    not LOCAL_MANIFEST_ROOT.is_dir(), reason="local harmonized manifests are not present"
)
def test_longest_local_inputs_fit_the_qwen38_context_window(tmp_path: Path) -> None:
    output = tmp_path / "audit.json"
    assert (
        audit_main(
            [
                "--output",
                str(output),
                "--manifest-root",
                str(LOCAL_MANIFEST_ROOT),
                "--max-subjects",
                "3",
            ]
        )
        == 0
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    if payload["measurement_kind"] != "real_qwen38_tokenizer":
        pytest.skip("the Qwen3.8 tokenizer is not locally available")
    measured = [row for row in payload["results"] if row.get("measured")]
    assert measured, "no cell could be measured"
    assert not payload["any_cell_exceeds_context_limit"]
    assert payload["model_context_limit_tokens"] >= 32768
    for row in measured:
        assert row["max_prompt_tokens"] > 0
        assert row["context_limit_tokens"] == payload["model_context_limit_tokens"]
        assert row["safety_margin_tokens"] > 0
        assert row["max_prompt_tokens"] + row["safety_margin_tokens"] <= row["context_limit_tokens"]
    pooled = [row for row in payload["results"] if row["cell_id"] == "turkish_pooled"]
    assert pooled and pooled[0].get("condition_rows"), "the pooled cell must audit both conditions"
    assert set(pooled[0]["condition_rows"]) == {"positive_sources", "negative_sources"}


def test_audit_defaults_point_at_local_paths() -> None:
    from scripts.audit_promptcontext_token_budget import DEFAULT_MODEL_DIR

    assert DEFAULT_MODEL_DIR == LOCAL_TOKENIZER
