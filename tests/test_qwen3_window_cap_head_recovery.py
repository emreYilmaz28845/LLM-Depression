"""Tests for the tracked exact affected-head recovery module.

Covers real confound selection (exact known old deployments + train+val routes
only), the deployment/key match required for a COMPLETED replacement, terminal
state gating, and whole-wave admission behavior. No SSH, no GPU.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import qwen3_window_cap_head_recovery as recovery  # noqa: E402

OLD = "old-deployment-full-id"
UNKNOWN = "unknown-deployment-full-id"
KEY_ANDROIDS = "androids_interview_audio_only_native_cap25|7|0"
KEY_DAIC = "daic_audio_only_native_cap25|7|0"


def make_entry(key: str, *, attempt: str = "ATT", deployment: str = OLD, extract: str = "1", classifier: str = "2") -> dict:
    return {
        "registry_key": key,
        "attempt_id": attempt,
        "deployment_id": deployment,
        "extract_job_id": extract,
        "classifier_job_id": classifier,
    }


def configure(tmp_path: Path) -> None:
    recovery.configure(
        campaign_dir=tmp_path,
        corrected_deployment="corrected-deployment-full-id",
        old_deployments=(OLD,),
        affected_datasets=("androids_interview", "d3tec"),
    )


def test_confound_record_requires_exact_old_deployment_and_route(tmp_path):
    configure(tmp_path)
    entry = make_entry(KEY_ANDROIDS)
    record = recovery.build_confound_record(entry, KEY_ANDROIDS, "COMPLETED")
    assert record is not None and record["disposition"] == "terminal_confounded"
    assert record["deployment_id"] == OLD
    assert record["mismatch_evidence"]["old_deployment"] == OLD
    # train-only routes are never auto-classified
    assert recovery.build_confound_record(make_entry(KEY_DAIC), KEY_DAIC, "COMPLETED") is None
    # unknown sources fail closed
    unknown = make_entry(KEY_ANDROIDS, deployment=UNKNOWN)
    assert recovery.build_confound_record(unknown, KEY_ANDROIDS, "COMPLETED") is None
    # registry update appends only eligible attempts, with dispositions
    appended = recovery.update_confound_registry(
        {KEY_ANDROIDS: entry, KEY_DAIC: make_entry(KEY_DAIC, attempt="DAIC-ATT")},
        {"2": "PENDING"},
    )
    assert appended == 1
    lines = (tmp_path / "head_confounded_attempts.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    written = json.loads(lines[0])
    assert written["disposition"] == "pending_terminal" and written["attempt_id"] == "ATT"


def test_confound_match_requires_deployment_and_key(tmp_path):
    configure(tmp_path)
    entry = make_entry(KEY_ANDROIDS)
    record = recovery.build_confound_record(entry, KEY_ANDROIDS, "COMPLETED")
    assert recovery.confound_match(record, entry, KEY_ANDROIDS) is True
    other_deployment = make_entry(KEY_ANDROIDS, deployment="other-full-id")
    assert recovery.confound_match(record, other_deployment, KEY_ANDROIDS) is False
    assert recovery.confound_match(record, entry, "other|key") is False
    assert recovery.confound_match(None, entry, KEY_ANDROIDS) is False


def test_select_authorized_terminal_and_confounded_states(tmp_path):
    configure(tmp_path)
    pending = {
        "a": make_entry("a", attempt="ATT-a", extract="11", classifier="12"),
        "b": make_entry("b", attempt="ATT-b", extract="21", classifier="22"),
        "c": make_entry("c", attempt="ATT-c", extract="31", classifier="32"),
        "d": make_entry("d", attempt="ATT-d", extract="41", classifier="42"),
        "e": make_entry("e", attempt="ATT-e", extract="51", classifier="52"),
    }
    states = {
        "11": "COMPLETED", "12": "PENDING",   # a: live classifier -> held
        "21": "COMPLETED", "22": "FAILED",    # b: terminal -> authorized
        "31": "COMPLETED", "32": "COMPLETED", # c: completed with confound -> authorized
        "41": "COMPLETED", "42": "COMPLETED", # d: completed without confound -> held
        "51": "COMPLETED", "52": "COMPLETED", # e: completed, confound mismatch -> held
    }
    confounded = {
        "ATT-c": {"deployment_id": OLD, "registry_key": "c"},
        "ATT-e": {"deployment_id": "wrong-deployment", "registry_key": "e"},
    }
    authorized, held = recovery.select_authorized(pending, states, confounded)
    assert authorized == ["b", "c"]
    assert set(held) == {"a", "d", "e"}


def test_wave_headroom_ok_whole_wave_admission(tmp_path):
    configure(tmp_path)
    recovery.configure(own_cap=80, wave_keys=5)
    assert recovery.wave_headroom_ok(69, 10) is True   # 69 + 2*5 = 79
    assert recovery.wave_headroom_ok(71, 10) is False  # 71 + 10 = 81
    assert recovery.wave_headroom_ok(77, 1) is True    # 77 + 2 = 79
    assert recovery.wave_headroom_ok(77, 10) is False
