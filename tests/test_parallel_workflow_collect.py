import subprocess, sys, pathlib

def test_collect_filter_includes_standalone_eval():
    result = subprocess.run([sys.executable, "tools/exp.py", "collect", "demo", "--dry-run"], capture_output=True, text=True)
    assert result.returncode == 0
    out = result.stdout.lower()
    assert "standalone_eval" in out
    # Should mention excluding adapters but including standalone
    assert "best_model/standalone_eval" in result.stdout or "standalone_eval" in out

def test_collect_dry_run_first():
    result = subprocess.run([sys.executable, "tools/exp.py", "collect", "demo", "--dry-run"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "dry_run" in result.stdout.lower() or "dry-run" in result.stdout.lower()
