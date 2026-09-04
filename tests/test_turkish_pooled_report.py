from __future__ import annotations

from tools.turkish_pooled_qcond_report import _slurm_job_ids


def test_slurm_job_ids_normalize_append_only_correction_without_losing_audit() -> None:
    events = [
        {
            "attempt_id": "attempt-a",
            "fold": 1,
            "job_key": "head",
            "job_type": "hidden_extraction",
            "event_id": "event-a",
            "slurm_job_id": "__RECOVERY_WAIT__ active=366 need=1\n45395989",
        },
        {"slurm_job_id": "45395469"},
        {"slurm_job_id": None},
    ]
    ids, corrections = _slurm_job_ids(events)
    assert ids == ["45395469", "45395989"]
    assert len(corrections) == 1
    assert corrections[0]["resolved_job_ids"] == ["45395989"]
    assert "raw_value" not in corrections[0]
    assert len(corrections[0]["raw_value_sha256"]) == 64
