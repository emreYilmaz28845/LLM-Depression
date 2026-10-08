"""Tests for the window-cap treatment config generator."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_window_cap_configs as gen  # noqa: E402


def test_treatment_changes_only_window_cap_block() -> None:
    for route_id, rel in gen.BASE_CONFIGS.items():
        base_path = ROOT / rel
        base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
        for pct, fraction in gen.ARMS.items():
            treatment = gen.build_treatment_config(base_path, fraction)
            block = treatment["training"]["window_cap"]
            assert block["enabled"] is True
            assert block["fraction"] == fraction
            assert block["sampling_seed"] == 1337
            assert block["algorithm_version"] == gen.ALGORITHM_VERSION
            assert block["base_config"] == rel
            assert block["base_config_sha256"] == gen.sha256_file(base_path)
            stripped = copy.deepcopy(treatment)
            stripped["training"].pop("window_cap")
            assert stripped == base, f"{route_id} cap{pct} changed more than window_cap"
            name = gen.treatment_name(base_path, pct)
            assert name.endswith(f"_cap{pct}.yaml")


def test_expected_files_cover_thirty_configs() -> None:
    expected = gen.expected_files()
    assert len(expected) == 30
    for name, content in expected.items():
        payload = yaml.safe_load(content)
        assert payload["training"]["window_cap"]["enabled"] is True


def test_check_mode_detects_tampered_config(tmp_path, monkeypatch, capsys) -> None:
    expected = gen.expected_files()
    out = tmp_path / "window_cap"
    out.mkdir()
    for name, content in expected.items():
        (out / name).write_text(content, encoding="utf-8")
    monkeypatch.setattr(gen, "OUTPUT_DIR", out)
    monkeypatch.setattr(sys, "argv", ["build_window_cap_configs.py", "--check"])
    assert gen.main() == 0
    target = out / sorted(expected)[0]
    target.write_text(target.read_text(encoding="utf-8") + "# tampered\n", encoding="utf-8")
    assert gen.main() == 1
    assert "content drift" in capsys.readouterr().err


def test_repo_check_passes() -> None:
    import subprocess

    proc = subprocess.run(
        [sys.executable, "scripts/build_window_cap_configs.py", "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "30 files" in proc.stdout
