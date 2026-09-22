"""Local pre-filter for the configurable training-lane shape.

Runs the real submission wrapper with a stub `sbatch` on PATH, so the argument
construction for the 4-GPU and the 2 x 4-GPU lanes is verified without touching
the cluster. The MN5 job remains the authoritative evidence.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUBMIT = ROOT / "scripts/submit_train_and_eval.sh"
CONFIG = "configs/main/daic_text_only_harmonized_selmacrof1_likelihood_v1_qwen38_27b.yaml"


def _run_submit(tmp_path: Path, *, train_nodes: int, overrides: str = "") -> str:
    stub_dir = tmp_path / "bin"
    stub_dir.mkdir(parents=True, exist_ok=True)
    sbatch = stub_dir / "sbatch"
    sbatch.write_text(
        "#!/usr/bin/env bash\n"
        'echo "SBATCH_CALL: $*"\n'
        'echo "1001;cluster"\n',
        encoding="utf-8",
    )
    sbatch.chmod(0o755)
    log_root = tmp_path / "logs"
    env = {
        **os.environ,
        "PATH": f"{stub_dir}:{os.environ['PATH']}",
        "PROJECT_ROOT": str(ROOT),
        "CONFIG": str(ROOT / CONFIG),
        "FOLD": "0",
        "RUN_NAME": "shape_probe",
        "TRAIN_NODES": str(train_nodes),
        "LOG_ROOT": str(log_root),
        "SKIP_MANIFEST_BUILD": "1",
        "EXTRA_TRAIN_ARGS": overrides,
    }
    result = subprocess.run(
        ["bash", str(SUBMIT)], cwd=ROOT, env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_two_node_lane_requests_two_nodes_and_exports_the_shape(tmp_path: Path) -> None:
    stdout = _run_submit(tmp_path, train_nodes=2, overrides="--set=training.gradient_accumulation_steps=16")
    assert "train_shape: 2 node(s) x 4 GPU(s) = 8 rank(s)" in stdout
    assert "effective_global_batch_size: 128" in stdout
    train_call = [line for line in stdout.splitlines() if "SBATCH_CALL" in line and "run_train_slurm.sh" in line]
    assert train_call, stdout
    assert "--nodes=2" in train_call[0]
    assert "--ntasks=8" in train_call[0]
    assert "--ntasks-per-node=4" in train_call[0]
    assert "--gres=gpu:4" in train_call[0]
    assert "NNODES=2" in train_call[0]
    assert "NPROC_PER_NODE=4" in train_call[0]
    # The evaluation job keeps the single-GPU shape.
    eval_call = [line for line in stdout.splitlines() if "SBATCH_CALL" in line and "run_eval_slurm.sh" in line]
    assert eval_call, stdout
    assert "--nodes=1" in eval_call[0] and "--gres=gpu:1" in eval_call[0]


def test_two_node_lane_refuses_the_default_accumulation(tmp_path: Path) -> None:
    """Eight ranks with accumulation 32 would double the effective batch."""
    stub_dir = tmp_path / "bin"
    stub_dir.mkdir(parents=True, exist_ok=True)
    sbatch = stub_dir / "sbatch"
    sbatch.write_text('#!/usr/bin/env bash\necho "SBATCH_CALL: $*"\necho "1001;cluster"\n', encoding="utf-8")
    sbatch.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{stub_dir}:{os.environ['PATH']}",
        "PROJECT_ROOT": str(ROOT),
        "CONFIG": str(ROOT / CONFIG),
        "FOLD": "0",
        "RUN_NAME": "shape_probe",
        "TRAIN_NODES": "2",
        "LOG_ROOT": str(tmp_path / "logs"),
        "SKIP_MANIFEST_BUILD": "1",
    }
    result = subprocess.run(
        ["bash", str(SUBMIT)], cwd=ROOT, env=env, capture_output=True, text=True
    )
    assert result.returncode == 1
    assert "effective global batch of 128" in result.stderr
    assert "gradient_accumulation_steps=16" in result.stderr


def test_default_lane_keeps_one_node_and_four_gpus(tmp_path: Path) -> None:
    stdout = _run_submit(tmp_path, train_nodes=1)
    assert "train_shape: 1 node(s) x 4 GPU(s) = 4 rank(s)" in stdout
    train_call = [line for line in stdout.splitlines() if "SBATCH_CALL" in line and "run_train_slurm.sh" in line]
    assert "--nodes=1" in train_call[0] and "--ntasks=4" in train_call[0]
    assert "NNODES=1" in train_call[0]
