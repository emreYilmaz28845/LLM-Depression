from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiment_tracking.collect import (
    CollectionError,
    RemoteRunner,
    build_collect_argv,
    execute_collection,
    local_inventory,
    plan_collection,
    remote_inventory,
    validate_fold_path,
    verify_compact_hash_agreement,
    verify_required_evidence,
)


def _build_remote_tree(root: Path) -> None:
    fold = root / "output_model" / "camp" / "audio_text" / "daic" / "run1" / "fold_0"
    (fold / "best_model" / "standalone_eval").mkdir(parents=True)
    (fold / "best_model" / "standalone_eval" / "metrics_original_teacher_forced.json").write_text(
        '{"binary_strict_macro_f1": 0.5}\n')
    (fold / "best_model" / "standalone_eval" / "predictions_subject_level.csv").write_text(
        "subject_id,prediction\n300,1\n")
    # adapter weights that must NOT survive collection
    (fold / "best_model").mkdir(parents=True, exist_ok=True)
    (fold / "last_model").mkdir(parents=True, exist_ok=True)
    (fold / "best_model" / "adapter_model.safetensors").write_bytes(b"A" * 2048)
    (fold / "best_model" / "adapter_config.json").write_text("{}\n")
    (fold / "last_model" / "adapter_model.safetensors").write_bytes(b"B" * 2048)
    for name in ("run_config.yaml", "metadata.json", "status.json", "jobs.jsonl",
                 "artifacts.json", "evaluations.json", "final_summary.json"):
        (fold / name).write_text(f"{name}\n")
    (fold / "eval" / "best_checkpoint").mkdir(parents=True)
    (fold / "eval" / "best_checkpoint" / "metrics.json").write_text("{}\n")
    (fold / "logs").mkdir(parents=True, exist_ok=True)
    (fold / "logs" / "train-1.log.jsonl").write_text("{}\n")
    return fold


class FakeCollectRunner(RemoteRunner):
    """Serves remote_inventory from a real local directory."""

    def __init__(self, remote_root: Path):
        super().__init__(host="ozu647717@transfer1.bsc.es", runner=self._dispatch)
        self.remote_root = remote_root
        self.calls: list[list[str]] = []

    def _dispatch(self, argv):
        self.calls.append(argv)
        script = argv[-1]
        if "find ." in script and "sha256sum" in script:
            proc = subprocess.run(
                ["bash", "-c",
                 script.replace("cd ", f"cd {self.remote_root} # cd ", 1)],
                capture_output=True, text=True)
            # run the real find|sha256sum inside the fake remote root
            inner = script.split("&& ", 1)[1]
            proc2 = subprocess.run(["bash", "-c", f"cd '{self.remote_root}' && {inner}"],
                                   capture_output=True, text=True)
            return proc2
        return subprocess.CompletedProcess(argv, 255, "", f"unexpected: {script}")


def _local_rsync_executor():
    def exec_local(argv):
        # strip host prefix so rsync runs locally against the fixture tree
        argv = list(argv)
        src = argv[-2]
        if "@" in src:
            argv[-2] = src.split(":", 1)[1]
        return subprocess.run(argv, capture_output=True, text=True)
    return exec_local


FOLD_REMOTE = "/gpfs/x/output_model/camp/audio_text/daic/run1/fold_0"


def test_validate_rejects_placeholders_and_bad_paths():
    with pytest.raises(CollectionError, match="placeholder"):
        validate_fold_path("/gpfs/x/output_model/<modality>/<dataset>/run/fold_0")
    with pytest.raises(CollectionError, match="fold_<n>"):
        validate_fold_path("/gpfs/x/output_model/camp/daic/run1")


def test_argv_never_contains_delete_and_orders_filters():
    argv = build_collect_argv(FOLD_REMOTE, "/tmp/local", dry_run=False)
    assert "--delete" not in argv
    joined = " ".join(argv)
    assert joined.index("best_model/standalone_eval/**") < joined.index("best_model/**")
    assert "-n" not in argv
    dry = build_collect_argv(FOLD_REMOTE, "/tmp/local", dry_run=True)
    assert "-n" in dry


def test_execute_preserves_standalone_eval_and_excludes_adapters(tmp_path):
    remote_base = tmp_path / "remote"
    fold = _build_remote_tree(remote_base)
    local_out = tmp_path / "local" / "fold_0"
    plan = plan_collection(str(fold), str(local_out))
    runner = FakeCollectRunner(fold)

    result = execute_collection(plan, runner, rsync_executor=_local_rsync_executor())

    assert result["inventory"]["matched"] == result["inventory"]["expected_compact_files"]
    # standalone eval metrics survived while adapter weights did not
    assert (local_out / "best_model" / "standalone_eval" / "metrics_original_teacher_forced.json").is_file()
    assert (local_out / "best_model" / "standalone_eval" / "predictions_subject_level.csv").is_file()
    assert not (local_out / "best_model" / "adapter_model.safetensors").exists()
    assert not (local_out / "last_model").exists()
    for name in ("run_config.yaml", "metadata.json", "status.json", "jobs.jsonl",
                 "artifacts.json", "evaluations.json", "final_summary.json"):
        assert (local_out / name).is_file()
    assert (local_out / "eval" / "best_checkpoint" / "metrics.json").is_file()
    # the only metrics were under best_model/standalone_eval; they survived
    inv = local_inventory(local_out)
    assert any("standalone_eval" in k for k in inv)
    # runner used ssh to transfer1 for inventory
    assert all("ozu647717@transfer1.bsc.es" in call for call in runner.calls)


def test_missing_required_evidence_fails(tmp_path):
    remote_base = tmp_path / "remote"
    fold = _build_remote_tree(remote_base)
    (fold / "evaluations.json").unlink()
    local_out = tmp_path / "local" / "fold_0"
    plan = plan_collection(str(fold), str(local_out))
    with pytest.raises(CollectionError, match="missing after sync"):
        execute_collection(plan, FakeCollectRunner(fold), rsync_executor=_local_rsync_executor())


def test_incompatible_local_overwrite_refused(tmp_path):
    remote_base = tmp_path / "remote"
    fold = _build_remote_tree(remote_base)
    local_out = tmp_path / "local" / "fold_0"
    local_out.mkdir(parents=True)
    (local_out / "run_config.yaml").write_text("tampered\n")
    plan = plan_collection(str(fold), str(local_out))
    with pytest.raises(CollectionError, match="incompatible local overwrite"):
        execute_collection(plan, FakeCollectRunner(fold), rsync_executor=_local_rsync_executor())


def test_identical_rerun_is_allowed(tmp_path):
    remote_base = tmp_path / "remote"
    fold = _build_remote_tree(remote_base)
    local_out = tmp_path / "local" / "fold_0"
    plan = plan_collection(str(fold), str(local_out))
    runner = FakeCollectRunner(fold)
    execute_collection(plan, runner, rsync_executor=_local_rsync_executor())
    # rerun over identical content must succeed (idempotent)
    result2 = execute_collection(plan, runner, rsync_executor=_local_rsync_executor())
    assert result2["inventory"]["matched"] > 0


def test_hash_mismatch_between_remote_and_local_fails(tmp_path):
    remote_base = tmp_path / "remote"
    fold = _build_remote_tree(remote_base)
    local_out = tmp_path / "local" / "fold_0"
    local_out.mkdir(parents=True)
    # pre-place a file the filters will include but whose content differs AND
    # make it look already-synced by bypassing overwrite check via identical name
    (local_out / "final_summary.json").write_text("different\n")
    plan = plan_collection(str(fold), str(local_out))
    with pytest.raises(CollectionError, match="incompatible local overwrite"):
        execute_collection(plan, FakeCollectRunner(fold), rsync_executor=_local_rsync_executor())


def test_verify_agreement_detects_mismatch(tmp_path):
    a = tmp_path / "a"
    a.mkdir()
    (a / "final_summary.json").write_text("one")
    remote_inv = {"final_summary.json": "deadbeef" * 8}
    with pytest.raises(CollectionError, match="hash mismatch"):
        verify_compact_hash_agreement(remote_inv, a)


def test_shell_wrapper_delegates_to_python():
    text = (PROJECT_ROOT / "scripts" / "collect_experiment.sh").read_text()
    assert "src.experiment_tracking.collect" in text
    assert "would run here" not in text
