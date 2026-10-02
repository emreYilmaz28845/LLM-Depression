"""The training/evaluation wrappers must not write bytecode into deployments.

The managed deployment verifier allows final ``__pycache__/*.pyc`` files but a
concurrent import can briefly expose interpreter atomic-write temp files
(``*.pyc.<id>``) as unexpected files and abort a submission before sbatch.
The three wrappers therefore disable bytecode writes explicitly; this test
keeps that isolation from being dropped silently.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

WRAPPERS = (
    "scripts/submit_train_and_eval.sh",
    "scripts/run_train_slurm.sh",
    "scripts/run_eval_slurm.sh",
)


def test_workers_disable_bytecode_writes() -> None:
    for relative in WRAPPERS:
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert "export PYTHONDONTWRITEBYTECODE=1" in text, relative
