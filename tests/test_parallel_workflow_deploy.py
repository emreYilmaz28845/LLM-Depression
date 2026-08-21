from __future__ import annotations

import json
import pathlib
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiment_tracking import deployment as dep
from src.experiment_tracking.deployment import (
    DeploymentError,
    RemoteRunner,
    build_deployment_record,
    build_rsync_command,
    build_source_manifest,
    create_runtime_dirs,
    estimate_transfer_bytes,
    execute_deployment,
    generate_deployment_id,
    get_source_manifest_hash,
    is_clean,
    plan_deployment,
    read_remote_file,
    remote_path_exists,
    require_remote_absent,
    sha256_of_json,
    validate_deployment_paths,
    verify_deployment,
    verify_remote_tree,
    write_remote_file_once,
)


def _init_repo(tmp_path: Path, name: str = "repo") -> Path:
    repo = tmp_path / name
    repo.mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    (repo / "README.md").write_text("hello")
    (repo / "scripts").mkdir()
    (repo / "scripts" / "tool.py").write_text("print('x')\n")
    (repo / ".gitignore").write_text(".provenance/\n")
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo, check=True, capture_output=True)
    return repo


def _write_provenance(repo: Path) -> dict:
    manifest = build_source_manifest(repo)
    prov = repo / ".provenance"
    prov.mkdir(exist_ok=True)
    (prov / "source_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


class FakeRunner(RemoteRunner):
    """Fake SSH runner that records argv and simulates a remote filesystem."""

    def __init__(self, host="ozu647717@transfer1.bsc.es"):
        super().__init__(host=host, runner=self._dispatch)
        self.calls: list[list[str]] = []
        self.remote_files: dict[str, str] = {}
        self.remote_dirs: set[str] = set()
        self._stdin: str | None = None

    def _dispatch(self, argv):
        self.calls.append(argv)
        script = argv[-1]
        # existence check
        if script.startswith("test -e "):
            target = script[len("test -e "):].strip().strip("'")
            return subprocess.CompletedProcess(argv, 0 if target in self.remote_files or target in self.remote_dirs else 1, "", "")
        if script.startswith("cd ") and "find ." in script:
            code = script.split("&&")[0].strip()[3:].strip().strip("'")
            lines = []
            for path in sorted(self.remote_files):
                if path.startswith(code.rstrip("/") + "/"):
                    rel = path[len(code.rstrip("/") + "/"):]
                    lines.append(rel)
            out = "\n".join(lines) + ("\n" if lines else "")
            return subprocess.CompletedProcess(argv, 0, out, "")
        if script.startswith("cd ") and "sha256sum" in script:
            code_part, files_part = script.split("&& sha256sum -- ")
            code = code_part.strip()[3:].strip().strip("'")
            import shlex as _shlex
            wanted = _shlex.split(files_part)
            lines = []
            for rel in wanted:
                full = code.rstrip("/") + "/" + rel
                digest = self.remote_files.get(full)
                if digest is None:
                    return subprocess.CompletedProcess(argv, 1, "", f"sha256sum: {rel}: No such file\n")
                lines.append(f"{digest}  {rel}")
            return subprocess.CompletedProcess(argv, 0, "\n".join(lines) + "\n", "")
        if script.startswith("cat > ") and "&& mv " in script:
            parts = script.split()
            tmp = parts[2].strip("'")
            target = parts[-1].strip("'")
            self.remote_files[target] = self._stdin or ""
            self._stdin = None
            return subprocess.CompletedProcess(argv, 0, "", "")
        if script.startswith("mkdir -p "):
            import shlex as _shlex
            for d in _shlex.split(script[len("mkdir -p "):]):
                self.remote_dirs.add(d)
            return subprocess.CompletedProcess(argv, 0, "", "")
        if script.startswith("cat ") and "&&" not in script:
            target = script[4:].strip().strip("'")
            content = self.remote_files.get(target)
            if content is None:
                return subprocess.CompletedProcess(argv, 1, "", f"cat: {target}: No such file\n")
            return subprocess.CompletedProcess(argv, 0, content, "")
        return subprocess.CompletedProcess(argv, 255, "", f"unexpected fake ssh script: {script}")

    def run(self, script, input=None, timeout=300):
        self._stdin = input
        return self._runner(self._argv_for(script))

    def _argv_for(self, script):
        return ["ssh", *self.ssh_opts, self.host, script]


def _populate_deployed_tree(runner: FakeRunner, repo: Path, code_path: str, manifest: dict, extras=None):
    for f in manifest["files"]:
        runner.remote_files[f"{code_path}/{f['path']}"] = f["sha256"]
    prov = repo / ".provenance" / "source_manifest.json"
    runner.remote_files[f"{code_path}/.provenance/source_manifest.json"] = prov.read_text(encoding="utf-8")
    for extra in extras or []:
        runner.remote_files[f"{code_path}/{extra}"] = "0" * 64


def _rsync_ok_factory(calls):
    def fake_rsync(argv):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, ">f+++++++++ README.md\n", "")
    return fake_rsync


# --- helper-level behavior retained from Phase 3 ---

def test_dirty_production_source_fails(tmp_path):
    repo = _init_repo(tmp_path)
    (repo / "dirty.txt").write_text("dirty")
    assert not is_clean(repo)
    with pytest.raises(DeploymentError, match="dirty production source"):
        plan_deployment(worktree=repo, experiment_id="exp-x", branch="agent/exp-x",
                        allow_dirty=False, deployment_id="d1")


def test_dirty_smoke_is_labeled_non_reportable(tmp_path):
    repo = _init_repo(tmp_path)
    (repo / "dirty.txt").write_text("dirty")
    _write_provenance(repo)
    plan = plan_deployment(worktree=repo, experiment_id="exp-x", branch="agent/exp-x",
                           allow_dirty=True, deployment_id="d1")
    assert plan["git_dirty"] is True
    assert plan["reportable_allowed"] is False


def test_deployment_ids_change_when_source_changes():
    id1 = generate_deployment_id("exp-test", "abc123def456", timestamp="20260821T000000Z")
    id2 = generate_deployment_id("exp-test", "def456abc123", timestamp="20260821T000000Z")
    assert id1 != id2


def test_existing_deployment_targets_fail_before_rsync(tmp_path):
    repo = _init_repo(tmp_path)
    _write_provenance(repo)
    plan = plan_deployment(worktree=repo, experiment_id="exp-x", branch="agent/exp-x",
                           allow_dirty=False, deployment_id="d1")
    runner = FakeRunner()
    runner.remote_dirs.add(plan["deployment_dir"])
    rsync_calls: list[list[str]] = []
    with pytest.raises(DeploymentError, match="already exists"):
        execute_deployment(plan, runner, rsync_executor=_rsync_ok_factory(rsync_calls))
    assert rsync_calls == [], "no rsync may run after an existing-target refusal"


def test_no_generated_command_contains_delete(tmp_path):
    cmd_dry = build_rsync_command(tmp_path, "transfer1", Path("/gpfs/deployments/test/code"), dry_run=True)
    cmd_exec = build_rsync_command(tmp_path, "transfer1", Path("/gpfs/deployments/test/code"), dry_run=False)
    for cmd in (cmd_dry, cmd_exec):
        assert "--delete" not in cmd
        assert all(not t.startswith("--delete") for t in cmd)


def test_runtime_roots_are_outside_deployment_code(tmp_path):
    dep_dir = tmp_path / "deployments" / "exp1" / "code"
    dep_dir.mkdir(parents=True)
    runtime = tmp_path / "experiment_runtime" / "exp1"
    runtime.mkdir(parents=True)
    assert validate_deployment_paths(dep_dir, runtime) == []
    bad_runtime = dep_dir / "runtime"
    bad_runtime.mkdir()
    errors = validate_deployment_paths(dep_dir, bad_runtime)
    assert errors and "inside deployment code" in errors[0]


def test_deployment_records_contain_full_sha_and_manifest_hash(tmp_path):
    repo = _init_repo(tmp_path)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    manifest_hash = get_source_manifest_hash(repo)
    record = build_deployment_record(
        deployment_id="exp-test-20260821-abc123-deadbeef",
        experiment_id="exp-test",
        git_commit=commit,
        git_branch_at_deploy="main",
        git_dirty=False,
        source_manifest_sha256=manifest_hash,
        deployed_code_path="/gpfs/deployments/exp-test/code",
    )
    assert len(record["git_commit"]) == 40
    assert len(record["source_manifest_sha256"]) == 64
    assert record["schema_version"] == "audiollm.deployment.v1"


def test_no_tracking_sidecar_responsibility_is_duplicated():
    record = build_deployment_record(
        deployment_id="exp-test-20260821-abc123-deadbeef",
        experiment_id="exp-test",
        git_commit="a" * 40,
        git_branch_at_deploy="main",
        git_dirty=False,
        source_manifest_sha256="b" * 64,
        deployed_code_path="/gpfs/deployments/exp-test/code",
    )
    for forbidden in ["jobs", "metrics", "evaluations", "status", "attempt_id", "fold"]:
        assert forbidden not in record


# --- R1 behavioral tests: real mocked SSH/rsync/checksum operations ---

def test_plan_requires_captured_provenance(tmp_path):
    repo = _init_repo(tmp_path)
    with pytest.raises(DeploymentError, match="capture_provenance"):
        plan_deployment(worktree=repo, experiment_id="exp-x", branch="agent/exp-x",
                        allow_dirty=False, deployment_id="d1")


def test_execute_runs_real_rsync_then_verifies_then_writes_record_once(tmp_path):
    repo = _init_repo(tmp_path)
    manifest = _write_provenance(repo)
    plan = plan_deployment(worktree=repo, experiment_id="exp-x", branch="agent/exp-x",
                           allow_dirty=False, deployment_id="dep-1")
    runner = FakeRunner()
    # Simulate what rsync will place on the remote (the fake executor does not).
    _populate_deployed_tree(runner, repo, plan["deployed_code_path"], manifest)
    rsync_calls: list[list[str]] = []

    result = execute_deployment(plan, runner, rsync_executor=_rsync_ok_factory(rsync_calls))

    # exactly one dry-run rsync and one execute rsync, neither with -n/--delete at exec time
    assert len(rsync_calls) == 2
    dry_argv, exec_argv = rsync_calls
    assert "-n" in dry_argv
    assert "-n" not in exec_argv
    for argv in (dry_argv, exec_argv):
        assert "--delete" not in argv
        assert argv[0] == "rsync"
        ff_tokens = [x for x in argv if x.startswith("--files-from=")]
        assert len(ff_tokens) == 1
        files_from = Path(ff_tokens[0][len("--files-from="):])
        assert files_from.is_file()
        listed = files_from.read_text().splitlines()
        assert "README.md" in listed and ".provenance/source_manifest.json" in listed
        assert argv[-1].endswith(f"{plan['deployed_code_path']}/")
        assert argv[-1].startswith("ozu647717@transfer1.bsc.es:")

    # absence checks ran over SSH before anything else
    test_e_calls = [c for c in runner.calls if c[-1].startswith("test -e ")]
    assert len(test_e_calls) >= 1
    assert any(plan["deployment_dir"] in c[-1] for c in test_e_calls)

    # remote hash verification used sha256sum against the deployed code dir
    sha_calls = [c for c in runner.calls if "sha256sum" in c[-1]]
    assert sha_calls, "remote checksum verification must run"
    assert all(f"cd '{plan['deployed_code_path']}'" in c[-1] or f"cd {plan['deployed_code_path']}" in c[-1] for c in sha_calls)

    # deployment record written exactly once via temp+mv (new-target-only creation)
    cat_writes = [c for c in runner.calls if c[-1].startswith("cat > ") and "&& mv " in c[-1]]
    assert len(cat_writes) == 1
    assert plan["deployment_record_path"] in cat_writes[0][-1]

    # runtime dirs created outside deployment code
    mkdirs = [c for c in runner.calls if c[-1].startswith("mkdir -p ")]
    assert mkdirs and plan["runtime_root"] in mkdirs[0][-1]
    assert plan["runtime_root"] not in plan["deployed_code_path"]

    assert result["verification"]["ok"] is True
    assert result["verification"]["verified_files"] == manifest["file_count"]


def test_failed_transfer_leaves_no_deployment_record(tmp_path):
    repo = _init_repo(tmp_path)
    _write_provenance(repo)
    plan = plan_deployment(worktree=repo, experiment_id="exp-x", branch="agent/exp-x",
                           allow_dirty=False, deployment_id="dep-fail")

    def failing_rsync(argv):
        if "-n" in argv:
            return subprocess.CompletedProcess(argv, 0, ">f+++++++++ README.md\n", "")
        return subprocess.CompletedProcess(argv, 23, "", "rsync error: connection died")

    runner = FakeRunner()
    with pytest.raises(DeploymentError, match="rsync transfer failed"):
        execute_deployment(plan, runner, rsync_executor=failing_rsync)
    writes = [c for c in runner.calls if c[-1].startswith("cat > ") and "mv" in c[-1]]
    assert writes == [], "a failed transfer must never leave a deployment record"


def test_hash_mismatch_fails_before_record_write(tmp_path):
    repo = _init_repo(tmp_path)
    manifest = _write_provenance(repo)
    plan = plan_deployment(worktree=repo, experiment_id="exp-x", branch="agent/exp-x",
                           allow_dirty=False, deployment_id="dep-bad")
    runner = FakeRunner()
    _populate_deployed_tree(runner, repo, plan["deployed_code_path"], manifest)
    # corrupt one file's remote hash
    victim = manifest["files"][0]["path"]
    runner.remote_files[f"{plan['deployed_code_path']}/{victim}"] = "f" * 64
    with pytest.raises(DeploymentError, match="does not match source manifest"):
        execute_deployment(plan, runner, rsync_executor=_rsync_ok_factory([]))
    writes = [c for c in runner.calls if c[-1].startswith("cat > ") and "mv" in c[-1]]
    assert writes == []


def test_unexpected_file_in_clean_deployment_fails(tmp_path):
    repo = _init_repo(tmp_path)
    manifest = _write_provenance(repo)
    plan = plan_deployment(worktree=repo, experiment_id="exp-x", branch="agent/exp-x",
                           allow_dirty=False, deployment_id="dep-extra")
    runner = FakeRunner()
    _populate_deployed_tree(runner, repo, plan["deployed_code_path"], manifest, extras=["stray.txt"])
    with pytest.raises(DeploymentError, match="unexpected files"):
        execute_deployment(plan, runner, rsync_executor=_rsync_ok_factory([]))


def test_missing_file_in_deployment_fails(tmp_path):
    repo = _init_repo(tmp_path)
    manifest = _write_provenance(repo)
    plan = plan_deployment(worktree=repo, experiment_id="exp-x", branch="agent/exp-x",
                           allow_dirty=False, deployment_id="dep-miss")
    runner = FakeRunner()
    _populate_deployed_tree(runner, repo, plan["deployed_code_path"], manifest)
    victim = "README.md"
    del runner.remote_files[f"{plan['deployed_code_path']}/{victim}"]
    with pytest.raises(DeploymentError, match="missing"):
        execute_deployment(plan, runner, rsync_executor=_rsync_ok_factory([]))


def test_paths_with_spaces_are_quoted_safely(tmp_path):
    runner = FakeRunner()
    weird = "/gpfs/some dir with spaces/code"
    exists = remote_path_exists(runner, weird)
    assert exists is False
    script = runner.calls[0][-1]
    assert "'/gpfs/some dir with spaces/code'" in script
    write_remote_file_once(runner, Path(weird) / "deployment.json", "{}", "record")
    write_script = [c for c in runner.calls if c[-1].startswith("cat > ")][-1][-1]
    assert "'/gpfs/some dir with spaces/code/deployment.json.tmp'" in write_script


def test_verify_deployment_detects_identity_mismatch_and_drift(tmp_path):
    repo = _init_repo(tmp_path)
    manifest = _write_provenance(repo)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    manifest_hash = sha256_of_json(manifest)
    record = build_deployment_record(
        deployment_id="dep-v", experiment_id="exp-v", git_commit=commit,
        git_branch_at_deploy="agent/x", git_dirty=False,
        source_manifest_sha256=manifest_hash,
        deployed_code_path="/gpfs/AudioLLM/deployments/dep-v/code",
    )
    runner = FakeRunner()
    code = record["deployed_code_path"]
    _populate_deployed_tree(runner, repo, code, manifest)
    runner.remote_files["/gpfs/AudioLLM/deployments/dep-v/deployment.json"] = json.dumps(record)

    ok = verify_deployment(runner, "dep-v", remote_base=pathlib.Path("/gpfs/AudioLLM"))
    assert ok["ok"] is True

    # wrong expected commit fails
    with pytest.raises(DeploymentError, match="git_commit"):
        verify_deployment(runner, "dep-v", remote_base=pathlib.Path("/gpfs/AudioLLM"),
                          expected_git_commit="b" * 40)

    # tampered record manifest hash fails
    tampered = dict(record)
    tampered["source_manifest_sha256"] = "e" * 64
    runner.remote_files[f"/gpfs/AudioLLM/deployments/dep-v/deployment.json"] = json.dumps(tampered)
    with pytest.raises(DeploymentError, match="manifest hash drift"):
        verify_deployment(runner, "dep-v", remote_base=pathlib.Path("/gpfs/AudioLLM"))

    # post-deploy drift (unexpected file) fails
    runner.remote_files[f"/gpfs/AudioLLM/deployments/dep-v/deployment.json"] = json.dumps(record)
    runner.remote_files[f"{code}/drifted.txt"] = "a" * 64
    with pytest.raises(DeploymentError, match="unexpected files"):
        verify_deployment(runner, "dep-v", remote_base=pathlib.Path("/gpfs/AudioLLM"))


def test_verify_deployment_fails_when_record_missing():
    runner = FakeRunner()
    with pytest.raises(DeploymentError, match="record missing"):
        verify_deployment(runner, "nope", remote_base=pathlib.Path("/gpfs/AudioLLM"))


def test_runtime_dir_creation_issues_single_mkdir(tmp_path):
    runner = FakeRunner()
    subdirs = create_runtime_dirs(runner, "/gpfs/rt/exp1")
    assert len(subdirs) == 6
    mkdir_calls = [c for c in runner.calls if c[-1].startswith("mkdir -p ")]
    assert len(mkdir_calls) == 1
    assert all(d in mkdir_calls[0][-1] for d in subdirs)


def test_estimate_transfer_bytes_matches_manifest():
    manifest = {"files": [{"size_bytes": 10}, {"size_bytes": 32}]}
    assert estimate_transfer_bytes(manifest) == 42
