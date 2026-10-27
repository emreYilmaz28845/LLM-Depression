import json
from pathlib import Path

import numpy as np
import pytest

from baselines.subject_mean_logreg_head import (
    DEV_PROTOCOL,
    PACKED30_PROTOCOL,
    TEST_PROTOCOL,
    VARIANT,
    balanced_weights,
    fit_logistic,
    run,
)


def _write_partition(cache: Path, name: str, subjects: list[tuple[str, int, int]], rng, shift: float) -> None:
    rows, vectors = [], []
    for subject, label, chunks in subjects:
        for chunk in range(chunks):
            rows.append({"sample_id": f"{subject}_{chunk:03d}", "subject_id": subject, "label": label})
            vectors.append(rng.normal(size=6) + shift * label)
    np.savez(cache / f"{name}.npz", vectors=np.asarray(vectors, dtype=np.float32))
    (cache / f"{name}_rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def _make_cache(cache: Path, protocol: str, eval_prefix: str, *, train_seed: int = 0) -> None:
    cache.mkdir(parents=True)
    train = [(f"t{i}", int(i % 3 == 0), 2 + i % 4) for i in range(30)]
    held = [(f"{eval_prefix}{i}", int(i % 3 == 0), 3) for i in range(12)]
    _write_partition(cache, "outer_train", train, np.random.default_rng(train_seed), 1.5)
    _write_partition(cache, "final_eval", held, np.random.default_rng(hash(eval_prefix) % 1000), 1.5)
    meta = {
        "dataset": "daic",
        "protocol_id": PACKED30_PROTOCOL,
        "evaluation_provenance": {"evaluation_protocol": protocol},
        "model_backend": "qwen3omni",
        "input_modality": "audio_only",
        "condition": "audio_only_base_omni",
        "fold": 0,
    }
    (cache / "extraction_metadata.json").write_text(json.dumps(meta))


def test_head_fits_selects_on_dev_and_writes_outputs(tmp_path):
    _make_cache(tmp_path / "dev", DEV_PROTOCOL, "d")
    _make_cache(tmp_path / "test", TEST_PROTOCOL, "s")
    summary = run(tmp_path / "dev", tmp_path / "test", tmp_path / "out")
    out = tmp_path / "out" / VARIANT
    selection = json.loads((out / "selection.json").read_text())
    assert selection["selected_l2_penalty"] == summary["selected_l2_penalty"]
    assert all(c["converged"] for c in selection["candidates"])
    for split, n in (("dev", 12), ("test", 12)):
        metrics = json.loads((out / split / "metrics.json").read_text())
        assert metrics["num_subjects"] == n
        assert metrics["macro_f1"] > 0.7  # separable synthetic data
    with pytest.raises(ValueError, match="overwrite"):
        run(tmp_path / "dev", tmp_path / "test", tmp_path / "out")


def test_refuses_different_training_partitions(tmp_path):
    _make_cache(tmp_path / "dev", DEV_PROTOCOL, "d", train_seed=1)
    _make_cache(tmp_path / "test", TEST_PROTOCOL, "s", train_seed=2)
    with pytest.raises(ValueError, match="different outer_train"):
        run(tmp_path / "dev", tmp_path / "test", tmp_path / "out")


def test_refuses_swapped_protocols(tmp_path):
    _make_cache(tmp_path / "dev", TEST_PROTOCOL, "d")
    _make_cache(tmp_path / "test", TEST_PROTOCOL, "s")
    with pytest.raises(ValueError, match="protocol"):
        run(tmp_path / "dev", tmp_path / "test", tmp_path / "out")


def test_mean_logit_equals_logit_of_mean_vector():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(20, 5))
    y = (x[:, 0] > 0).astype(int)
    w, b, info = fit_logistic(x, y, balanced_weights(y), 1.0)
    assert info["converged"]
    chunks = rng.normal(size=(4, 5))
    assert np.isclose((chunks @ w + b).mean(), chunks.mean(0) @ w + b)


def test_balanced_weights_equalize_classes():
    y = np.array([1] * 3 + [0] * 7)
    w = balanced_weights(y)
    assert np.isclose(w[y == 1].sum(), w[y == 0].sum())
