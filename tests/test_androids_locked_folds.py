import hashlib
import json

import pytest

from src.data.androids import load_androids_interview_locked_folds


SUBJECTS = {f"s{index:02d}" for index in range(10)}


def _write(tmp_path, payload):
    path = tmp_path / "folds.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _payload():
    ordered = sorted(SUBJECTS)
    return {str(fold): {"final_eval_subject_ids": ordered[2 * fold : 2 * fold + 2]} for fold in range(5)}


def test_locked_folds_load_and_partition_every_subject(tmp_path):
    path, sha = _write(tmp_path, _payload())
    folds = load_androids_interview_locked_folds(path, sha, SUBJECTS)
    assert sorted(folds) == [0, 1, 2, 3, 4]
    heldout = [subject for fold in folds.values() for subject in fold["final_eval_subject_ids"]]
    assert sorted(heldout) == sorted(SUBJECTS)
    for fold in folds.values():
        assert set(fold["outer_train_subject_ids"]) == SUBJECTS - set(fold["final_eval_subject_ids"])


def test_locked_folds_refuse_a_wrong_sha256(tmp_path):
    path, _ = _write(tmp_path, _payload())
    with pytest.raises(ValueError, match="sha256"):
        load_androids_interview_locked_folds(path, "0" * 64, SUBJECTS)


def test_locked_folds_refuse_overlap_and_missing_coverage(tmp_path):
    payload = _payload()
    payload["4"]["final_eval_subject_ids"] = payload["0"]["final_eval_subject_ids"]
    path, sha = _write(tmp_path, payload)
    with pytest.raises(ValueError, match="overlap"):
        load_androids_interview_locked_folds(path, sha, SUBJECTS)

    payload = _payload()
    payload["4"]["final_eval_subject_ids"] = []
    path, sha = _write(tmp_path, payload)
    with pytest.raises(ValueError, match="coverage"):
        load_androids_interview_locked_folds(path, sha, SUBJECTS)


def test_locked_folds_refuse_unknown_subjects(tmp_path):
    payload = _payload()
    payload["0"]["final_eval_subject_ids"] = ["unknown"]
    path, sha = _write(tmp_path, payload)
    with pytest.raises(ValueError, match="Unknown"):
        load_androids_interview_locked_folds(path, sha, SUBJECTS)
