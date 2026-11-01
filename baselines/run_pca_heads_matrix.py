#!/usr/bin/env python3
"""PCA before the subject-mean L2 head: Qwen3-Omni frozen and wav2vec2 (mean+var, mean).

Every route reuses an existing ``frozen_standalone_heads.py`` reference run:
cache, DAIC dev cache, split files, protocol and route are read from its
``result_config.json`` and the cache and split hashes are checked against the
files on disk (fails closed). Variants per route:

* ``nopca``: the unchanged head; its subject-level predictions must equal the
  reference run's (reproduction check, fails closed);
* ``k<K>``: fixed K PCA components (K=50 is the headline);
* ``ksel``: K chosen with L2 on the inner validation from the K grid.

Writes ``<output-root>/<variant>/<feature>/<dataset>/`` plus ``RUN_MANIFEST.json``
(git commit, dirty flag, routes, reference paths).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.frozen_standalone_heads import run_route
from src.features.hidden_classifier_policy import canonical_sha256, cache_identity
from src.utils import read_json, save_json

DATASETS = ("daic", "d3tec", "androids_interview", "cmdc", "turkish")
K_GRID = (10, 20, 50, 100, 200)


def _git() -> dict:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=PROJECT_ROOT,
                           capture_output=True, text=True).stdout.strip()
    return {"commit": head, "dirty": bool(dirty)}


def _cache_dir(identity: dict, lane_root: Path) -> Path:
    path = Path(identity["outer_train.npz"]["path"])
    return (path if path.is_absolute() else lane_root / path).parent


def _check_identity(cache_dir: Path, expected: dict) -> None:
    actual = cache_identity(cache_dir)
    for name, item in expected.items():
        if actual[name]["sha256"] != item["sha256"]:
            raise ValueError(f"{cache_dir}/{name}: sha256 differs from the reference run.")


def load_reference(ref_dir: Path, lane_root: Path) -> dict:
    config = read_json(ref_dir / "result_config.json")
    caches = [_cache_dir(i, lane_root) for i in config["cache_identities"]]
    for c, i in zip(caches, config["cache_identities"]):
        _check_identity(c, i)
    devs = None
    if config.get("dev_cache_identities"):
        devs = [_cache_dir(i, lane_root) for i in config["dev_cache_identities"]]
        for c, i in zip(devs, config["dev_cache_identities"]):
            _check_identity(c, i)
    splits = []
    for item in config["split_files"]:
        path = Path(item["path"])
        if canonical_sha256(read_json(path)) != item["sha256"]:
            raise ValueError(f"{path}: split hash differs from the reference run.")
        splits.append(path)
    return {"ref_dir": ref_dir, "cache_dirs": caches, "dev_caches": devs, "split_files": splits,
            "protocol": config["protocol"], "route": config["route"], "l2_grid": tuple(config["l2_grid"])}


def _predictions(route_dir: Path) -> list[list[tuple]]:
    folds = []
    for fold_dir in sorted(route_dir.glob("fold_*")):
        rows = [json.loads(line) for line in (fold_dir / "predictions_subject_level.jsonl").read_text().splitlines()]
        folds.append([(r["subject_id"], r["prediction"], round(r["mean_logit"], 9)) for r in rows])
    return folds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--omni-ref-root", required=True, type=Path, help="<dataset>/ reference Omni head runs.")
    parser.add_argument("--w2v-ref-root", required=True, type=Path, help="<pooling>/<dataset>/ reference runs.")
    parser.add_argument("--w2v-lane-root", required=True, type=Path, help="Resolves relative wav2vec2 cache paths.")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--variants", nargs="+", default=["nopca", *(f"k{k}" for k in K_GRID), "ksel"])
    args = parser.parse_args()

    git = _git()
    if git["dirty"]:
        raise SystemExit("Refusing to run from a dirty worktree (provenance).")
    features = {"omni": lambda d: args.omni_ref_root / d,
                "w2v_meanvar": lambda d: args.w2v_ref_root / "meanvar" / d,
                "w2v_mean": lambda d: args.w2v_ref_root / "mean" / d}
    refs = {(f, d): load_reference(path(d), args.w2v_lane_root) for f, path in features.items() for d in DATASETS}
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = {"git": git, "k_grid": list(K_GRID), "variants": args.variants, "started": time.strftime("%FT%TZ", time.gmtime()),
                "routes": {f"{f}/{d}": {"reference": str(r["ref_dir"]), "protocol": r["protocol"],
                                        "cache_dirs": [str(c) for c in r["cache_dirs"]],
                                        "dev_caches": [str(c) for c in r["dev_caches"]] if r["dev_caches"] else None,
                                        "split_files": [str(s) for s in r["split_files"]]}
                           for (f, d), r in refs.items()}}
    save_json(manifest, args.output_root / f"RUN_MANIFEST_{hashlib.sha1(' '.join(args.variants).encode()).hexdigest()[:8]}.json")
    for variant in args.variants:
        if variant == "nopca":
            grid = None
        elif variant == "ksel":
            grid = K_GRID
        else:
            grid = (int(variant[1:]),)
        for (feature, dataset), ref in refs.items():
            out = args.output_root / variant / feature / dataset
            t0 = time.time()
            summary = run_route(cache_dirs=ref["cache_dirs"], dev_caches=ref["dev_caches"],
                                split_files=ref["split_files"], protocol=ref["protocol"], output_dir=out,
                                route={**ref["route"], "pca_variant": variant}, grid=ref["l2_grid"], pca_grid=grid)
            if variant == "nopca" and _predictions(out) != _predictions(ref["ref_dir"]):
                raise SystemExit(f"{feature}/{dataset}: no-PCA rerun does not reproduce {ref['ref_dir']}.")
            print(f"{variant:6s} {feature:12s} {dataset:20s} macroF1 {summary['fold_mean']['macro_f1']:.3f} "
                  f"posF1 {summary['fold_mean']['positive_f1']:.3f}  {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
