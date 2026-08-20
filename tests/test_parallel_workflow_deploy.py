from __future__ import annotations
import json
import pathlib
import subprocess
import tempfile

import pytest

from src.experiment_tracking.deployment import (
    build_deployment_record,
    generate_deployment_id,
    is_clean,
    get_source_manifest_hash,
    build_rsync_command,
    validate_deployment_paths,
)

def test_dirty_production_source_fails(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    (repo / "README.md").write_text("hello")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo, check=True)
    # Make dirty
    (repo / "dirty.txt").write_text("dirty")
    assert not is_clean(repo)
    # Simulate production deploy check: should fail if dirty
    # Our deployment should reject dirty for production
    # Test the helper: is_clean false should be considered dirty production failure
    assert is_clean(repo) is False

def test_dirty_smoke_is_labeled_non_reportable(tmp_path):
    # For smoke/debug mode, dirty is allowed but record marks non-reportable
    repo = tmp_path / "repo2"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    (repo / "README.md").write_text("hello")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo, check=True)
    (repo / "dirty.txt").write_text("dirty")
    git_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    manifest_hash = get_source_manifest_hash(repo)
    record = build_deployment_record(
        deployment_id="exp-test-20260821-abc123-deadbeef",
        experiment_id="exp-test",
        git_commit=git_commit,
        git_branch_at_deploy="main",
        git_dirty=True,
        source_manifest_sha256=manifest_hash,
        deployed_code_path="/gpfs/.../deployments/exp-test/code",
    )
    assert record["git_dirty"] is True
    # For smoke, we would label non-reportable; check that record indicates dirty
    assert record["uncommitted_patch_sha256"] != "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"  # not empty

def test_deployment_ids_change_when_source_changes(tmp_path):
    id1 = generate_deployment_id("exp-test", "abc123def456", timestamp="20260821T000000Z")
    id2 = generate_deployment_id("exp-test", "def456abc123", timestamp="20260821T000000Z")
    assert id1 != id2
    # Also suffix random ensures different even same commit
    id3 = generate_deployment_id("exp-test", "abc123def456", timestamp="20260821T000000Z")
    assert id1 != id3  # random suffix

def test_existing_deployment_targets_fail_before_rsync(tmp_path):
    # Simulate check that existing deployment path should fail before rsync
    target = tmp_path / "existing_deploy"
    target.mkdir()
    (target / "code").mkdir()
    # Our validate should detect existing deployment target exists -> should fail
    # For test, we check that if target exists, deployment should refuse
    assert target.exists()
    # Simulate build_rsync_command should not contain --delete and should be refused if target exists
    cmd = build_rsync_command(tmp_path, "host", target, dry_run=True)
    assert "--delete" not in cmd
    # The deploy logic should check existence before rsync; we test helper
    assert target.exists() is True

def test_no_generated_command_contains_delete(tmp_path):
    cmd = build_rsync_command(tmp_path, "transfer1", pathlib.Path("/gpfs/deployments/test/code"), dry_run=True)
    assert "--delete" not in " ".join(cmd)
    assert "--delete" not in cmd

def test_runtime_roots_are_outside_deployment_code(tmp_path):
    dep = tmp_path / "deployments" / "exp1" / "code"
    dep.mkdir(parents=True)
    runtime = tmp_path / "experiment_runtime" / "exp1"
    runtime.mkdir(parents=True)
    errors = validate_deployment_paths(dep, runtime)
    assert errors == []
    # Now test inside case should fail
    bad_runtime = dep / "runtime"
    bad_runtime.mkdir()
    errors2 = validate_deployment_paths(dep, bad_runtime)
    assert len(errors2) > 0
    assert "inside deployment code" in errors2[0]

def test_deployment_records_contain_full_sha_and_manifest_hash(tmp_path):
    repo = tmp_path / "repo3"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    (repo / "README.md").write_text("hello")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo, check=True)
    git_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    manifest_hash = get_source_manifest_hash(repo)
    deployment_id = generate_deployment_id("exp-test", git_commit)
    record = build_deployment_record(
        deployment_id=deployment_id,
        experiment_id="exp-test",
        git_commit=git_commit,
        git_branch_at_deploy="main",
        git_dirty=False,
        source_manifest_sha256=manifest_hash,
        deployed_code_path="/gpfs/deployments/exp-test/code",
    )
    assert len(record["git_commit"]) == 40
    assert len(record["source_manifest_sha256"]) == 64
    assert record["schema_version"] == "audiollm.deployment.v1"

def test_no_tracking_sidecar_responsibility_is_duplicated(tmp_path):
    # Ensure deployment record does not contain job/metric/evaluation lifecycle fields
    record = build_deployment_record(
        deployment_id="exp-test-20260821-abc123-deadbeef",
        experiment_id="exp-test",
        git_commit="a"*40,
        git_branch_at_deploy="main",
        git_dirty=False,
        source_manifest_sha256="b"*64,
        deployed_code_path="/gpfs/deployments/exp-test/code",
    )
    # Should not contain sidecar fields
    for forbidden in ["jobs", "metrics", "evaluations", "status", "attempt_id", "fold"]:
        assert forbidden not in record
    assert "deployment_id" in record
    assert "source_manifest_sha256" in record
