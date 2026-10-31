import json

import numpy as np
import pytest

from baselines.frozen_standalone_heads import run_route
from baselines.layer_select_heads import run

SIGNAL_LAYER = 2


def _write_cache(path, layer, subjects, labels, rng, signal):
    path.mkdir(parents=True)
    rows, vectors = [], []
    for subject, label in zip(subjects, labels):
        for unit in range(2):
            for window in range(2):
                rows.append({"sample_id": f"{subject}_u{unit}_w{window}", "subject_id": subject, "label": label,
                             "response_id": f"{subject}::u{unit}"})
                v = rng.normal(size=6)
                if signal:
                    v[0] += 2.5 * (1 if label else -1)
                vectors.append(v)
    x = np.asarray(vectors, dtype=np.float32)
    half = len(rows) // 2
    for name, sl in (("outer_train", slice(0, half)), ("final_eval", slice(half, None))):
        np.savez(path / f"{name}.npz", vectors=x[sl])
        (path / f"{name}_rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows[sl]))
    (path / "extraction_metadata.json").write_text(json.dumps({
        "weights_source": "base_model_no_adapter", "hidden_layer_index": layer, "manifest_sha256": "m",
        "base_model": "w2v", "condition": "c", "input_modality": "audio_only", "pooling": "p"}))


@pytest.fixture()
def layered(tmp_path):
    subjects = [f"s{i:02d}" for i in range(40)]
    labels = [i % 2 for i in range(40)]
    root = tmp_path / "layers"
    for layer in range(4):
        rng = np.random.default_rng(100 + layer)
        _write_cache(root / f"layer_{layer:02d}", layer, subjects, labels, rng, signal=layer == SIGNAL_LAYER)
    splits = []
    for fold in range(2):
        held = subjects[fold * 10:(fold + 1) * 10]
        split = {"train_subject_ids": [s for s in subjects if s not in held], "selection_subject_ids": held}
        path = tmp_path / f"split_{fold}.json"
        path.write_text(json.dumps(split))
        splits.append(path)
    return root, splits, tmp_path


def test_selects_signal_layer_and_matches_single_layer_head(layered):
    root, splits, tmp = layered
    summary = run(layers_root=root, dev_layers_root=None, split_files=splits, protocol="train_val",
                  output_dir=tmp / "out", route={})
    assert summary["selected_layers"] == [SIGNAL_LAYER, SIGNAL_LAYER]
    single = run_route(cache_dirs=[root / f"layer_{SIGNAL_LAYER:02d}"], dev_caches=None, split_files=splits,
                       protocol="train_val", output_dir=tmp / "single", route={})
    assert summary["fold_mean"] == single["fold_mean"]
    diag = json.loads((tmp / "out" / "per_layer_test.json").read_text())
    assert set(diag["layers"]) == {"0", "1", "2", "3"}
    selection = json.loads((tmp / "out" / "fold_0" / "selection.json").read_text())
    assert selection["selected_layer"] == SIGNAL_LAYER and len(selection["per_layer_best"]) == 4


def test_refuses_mismatched_rows(layered):
    root, splits, tmp = layered
    rows_path = root / "layer_01" / "outer_train_rows.jsonl"
    lines = rows_path.read_text().splitlines()
    first = json.loads(lines[0])
    first["sample_id"] = "other"
    rows_path.write_text("\n".join([json.dumps(first)] + lines[1:]) + "\n")
    with pytest.raises(ValueError, match="rows differ"):
        run(layers_root=root, dev_layers_root=None, split_files=splits, protocol="train_val",
            output_dir=tmp / "out", route={})


def test_requires_contiguous_layers(layered):
    root, splits, tmp = layered
    (root / "layer_01").rename(root / "skipped")
    with pytest.raises(ValueError, match="layer_00..layer_NN"):
        run(layers_root=root, dev_layers_root=None, split_files=splits, protocol="train_val",
            output_dir=tmp / "out", route={})
