from pathlib import Path
import subprocess
import sys

def test_status_command_exists():
    result = subprocess.run([sys.executable, "tools/exp.py", "status", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "status" in result.stdout.lower()

def test_collect_preserves_compact_evidence():
    result = subprocess.run([sys.executable, "tools/exp.py", "collect", "demo", "--dry-run"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "standalone_eval" in result.stdout
    assert "best_model/***" not in result.stdout or "exclude" in result.stdout.lower()

def test_compare_requires_full_qualifiers():
    # Missing required args should fail
    result = subprocess.run([sys.executable, "tools/exp.py", "compare", "--group", "g1", "--attempts", "a1", "--dataset", "daic"], capture_output=True, text=True)
    assert result.returncode != 0

def test_compare_with_full_qualifiers_succeeds():
    result = subprocess.run([sys.executable, "tools/exp.py", "compare", "--group", "g1", "--attempts", "a1,a2", "--dataset", "daic", "--metric", "positive_f1", "--namespace", "headline/binary_strict", "--backend", "original_teacher_forced", "--view", "harmonized_all_windows_full_coverage", "--aggregation", "subject_level"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "compare" in result.stdout.lower()

def test_validate_stub():
    result = subprocess.run([sys.executable, "tools/exp.py", "validate", "demo"], capture_output=True, text=True)
    assert result.returncode == 0
