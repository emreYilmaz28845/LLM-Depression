import pathlib, subprocess, sys

def test_submit_wrapper_has_chdir():
    content = pathlib.Path("scripts/submit_train_and_eval.sh").read_text()
    assert '--chdir="$PROJECT_ROOT"' in content or "--chdir" in content

def test_submit_wrapper_has_evaluation_view_check():
    content = pathlib.Path("scripts/submit_train_and_eval.sh").read_text()
    assert "evaluation_view" in content

def test_submit_wrapper_has_collision_check():
    content = pathlib.Path("scripts/submit_train_and_eval.sh").read_text()
    assert "collision" in content.lower() or 'already exists' in content

def test_submit_wrapper_preserves_gpu_shapes():
    train = pathlib.Path("scripts/run_train_slurm.sh").read_text()
    evalp = pathlib.Path("scripts/run_eval_slurm.sh").read_text()
    assert "--gres=gpu:4" in train
    assert "--gres=gpu:1" in evalp
    assert "NPROC_PER_NODE" in train
