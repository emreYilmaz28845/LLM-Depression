"""English-lane wiring for the shared head planning and dispatch contract.

The shared tools resolve a lane's campaign, language, evidence directory and
head tracking identity from the linked experiment group. This test checks the
real English group definition: schema validity, the exact English scope fields,
and, inside the managed worktree where the lane pin exists, the resolved
identity plus the isolation rule that a foreign (Native) campaign override is
refused. Weight-free and network-free.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiment_tracking.schemas import validate_experiment_group  # noqa: E402
from tools import qwen3_heads_dispatch as dispatch  # noqa: E402

GROUP = ROOT / "experiments/definitions/qwen3-multiseed-english-20261002.yaml"
PIN = ROOT / ".agent-pin.json"

EXPECTED_SCOPE = {
    "campaign": "qwen3_multiseed_english_20261002",
    "language": "english",
    "evidence_dir": "outputs/qwen3_multiseed_english_20261002",
    "head_tracking_kind": "qwen3_multiseed_english_20261002_head",
    "head_run_schema": "audiollm.qwen3_multiseed_english_20261002_head_run.v1",
}


def test_group_scope_is_valid_and_english_specific() -> None:
    group = yaml.safe_load(GROUP.read_text(encoding="utf-8"))
    valid, errors = validate_experiment_group(group)
    assert valid, errors
    assert group["scope"] == EXPECTED_SCOPE


@pytest.mark.skipif(not PIN.is_file(), reason="lane pin only exists inside the managed worktree")
def test_lane_identity_resolves_english_values() -> None:
    identity = dispatch.resolve_lane_identity()
    assert identity.campaign == EXPECTED_SCOPE["campaign"]
    assert identity.language == EXPECTED_SCOPE["language"]
    assert identity.evidence_dir == (ROOT / EXPECTED_SCOPE["evidence_dir"]).resolve()
    assert identity.tracking_kind == EXPECTED_SCOPE["head_tracking_kind"]
    assert identity.run_schema == EXPECTED_SCOPE["head_run_schema"]
    assert identity.runtime_root.name == "feat-qwen3-multiseed-english-20261002"
    assert identity.evidence_dir.name == "qwen3_multiseed_english_20261002"


@pytest.mark.skipif(not PIN.is_file(), reason="lane pin only exists inside the managed worktree")
def test_foreign_campaign_override_is_refused() -> None:
    with pytest.raises(dispatch.DispatchError, match="does not match the linked group campaign"):
        dispatch.resolve_lane_identity(campaign_override="qwen3_multiseed_native_20261002")
