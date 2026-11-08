#!/usr/bin/env python3
"""Draw the descriptive UMAP maps of the text replication (needs umap-learn and matplotlib).

Reads ``umap_speakers.npz`` and ``umap_maps.json`` written by ``text_replication_geometry.py``.
Each map z-normalizes its points with the mean / population std of its scaler speakers,
then runs UMAP on the D-dimensional input with the handoff defaults: 2 components,
n_neighbors 15, min_dist 0.1, euclidean, init random, random_state 42,
transform_seed 42, n_epochs 500, n_jobs 1. Left panel: language or role; right panel:
true depression label, on identical coordinates. Maps are label-free and descriptive.
"""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

UMAP_PARAMS = {"n_components": 2, "n_neighbors": 15, "min_dist": 0.1, "metric": "euclidean", "init": "random",
               "random_state": 42, "transform_seed": 42, "n_epochs": 500, "n_jobs": 1}
_DATA: dict = {}


def _init(npz: str) -> None:
    payload = np.load(npz)
    _DATA.update({k: payload[k] for k in payload.files})


def _draw(task: tuple[dict, str, str, str]) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import umap

    spec, maps_dir, coords_dir, text_language = task
    x = _DATA["x"].astype(np.float64)
    keys = np.asarray(spec["scaler_keys"])
    mean, std = x[keys].mean(0), x[keys].std(0)
    std[std < 1e-6] = 1.0
    pts = np.asarray(spec["points"])
    z = (x[pts] - mean) / std
    coords = umap.UMAP(**UMAP_PARAMS).fit_transform(z)
    lang, label, subj = _DATA["language"][pts], _DATA["label"][pts], _DATA["subjects"][pts]
    groups = lang if spec["kind"] == "main" else np.asarray(spec["roles"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for g in sorted(set(groups.tolist())):
        m = groups == g
        axes[0].scatter(coords[m, 0], coords[m, 1], s=12, label=f"{g} ({m.sum()})", alpha=0.8)
    axes[0].legend(fontsize=8, markerscale=1.5)
    axes[0].set_title("Language" if spec["kind"] == "main" else "Role")
    for lab, name in ((0, "Non-depressed"), (1, "Depressed")):
        m = label == lab
        axes[1].scatter(coords[m, 0], coords[m, 1], s=12, label=f"{name} ({m.sum()})", alpha=0.8)
    axes[1].legend(fontsize=8, markerscale=1.5)
    axes[1].set_title("True depression label")
    fig.suptitle(f"{spec['name']} - frozen Qwen3.8 text ({text_language}), UMAP of z-normalized speaker vectors")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(Path(maps_dir) / f"{spec['name']}.png", dpi=110)
    plt.close(fig)
    with (Path(coords_dir) / f"{spec['name']}.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["subject", "language", "label", "group", "umap1", "umap2"])
        for row in zip(subj, lang, label, groups, coords[:, 0], coords[:, 1]):
            w.writerow([*row[:4], f"{row[4]:.5f}", f"{row[5]:.5f}"])
    return spec["name"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--geometry-dir", type=Path, required=True)
    parser.add_argument("--maps-dir", type=Path, required=True)
    parser.add_argument("--coords-dir", type=Path, required=True)
    parser.add_argument("--text-language", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    maps = json.loads((args.geometry_dir / "umap_maps.json").read_text())
    args.maps_dir.mkdir(parents=True, exist_ok=True)
    args.coords_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(m, str(args.maps_dir), str(args.coords_dir), args.text_language) for m in maps]
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init,
                             initargs=(str(args.geometry_dir / "umap_speakers.npz"),)) as pool:
        done = list(pool.map(_draw, tasks))
    (args.coords_dir / "umap_params.json").write_text(json.dumps({"params": UMAP_PARAMS, "maps": done}, indent=1))
    print(len(done), "maps")


if __name__ == "__main__":
    main()
