import subprocess, sys

def test_compare_group_scoped():
    result = subprocess.run([sys.executable, "tools/exp.py", "compare", "--group", "test_group", "--attempts", "a1,a2", "--dataset", "daic", "--metric", "macro_f1", "--namespace", "headline/binary_strict", "--backend", "original_teacher_forced", "--view", "harmonized_all_windows_full_coverage", "--aggregation", "subject_level"], capture_output=True, text=True)
    assert result.returncode == 0
