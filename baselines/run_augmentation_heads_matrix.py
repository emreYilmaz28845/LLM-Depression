#!/usr/bin/env python3
"""Embedding-level training augmentation for the frozen subject-mean L2 head.

Every route reuses an existing ``frozen_standalone_heads.py`` reference run
(frozen Qwen3-Omni audio, frozen Qwen3.8 text, frozen Omni audio+text; native
language): cache, DAIC dev cache, split files, protocol and route are read from
its ``result_config.json``, and cache and split hashes are checked against the
files on disk (fails closed). Variants:

* ``none``: the unchanged head; its subject-level predictions must equal the
  reference run's (reproduction check, fails closed);
* ``<method>_r<ratio>/seed<k>``: ``smote``, ``mixup``, ``gauss`` or ``wboot``
  with ``ratio`` synthetic rows per real row and class, augmentation seed k.

Writes ``<output-root>/<variant>[/seed<k>]/<dataset>/<route>/`` plus a
``RUN_MANIFEST_<hash>.json`` (git commit, dirty flag, routes, references).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.frozen_standalone_heads import AUG_METHODS, run_route
from src.features.hidden_classifier_policy import cache_identity, canonical_sha256
from src.utils import read_json, save_json

DATASETS = ("daic", "d3tec", "androids_interview", "cmdc", "turkish")
ROUTES = ("audio_only_native", "text_only_native", "audio_text_native")
RATIOS = (1, 4)
SEEDS = (0, 1, 2)


def _git() -> dict:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=PROJECT_ROOT,
                           capture_output=True, text=True).stdout.strip()
    return {"commit": head, "dirty": bool(dirty)}


def _identities(config: dict, key: str) -> list[dict]:
    """v2 stores a list (``<key>_identities``), v1 a single ``<key>_identity``."""
    many = config.get(f"{key}_identities")
    if many:
        return many
    one = config.get(f"{key}_identity")
    return [one] if one else []


def _check(cache_dir: Path, expected: dict) -> None:
    actual = cache_identity(cache_dir)
    for name, item in expected.items():
        if actual[name]["sha256"] != item["sha256"]:
            raise ValueError(f"{cache_dir}/{name}: sha256 differs from the reference run.")


def load_reference(ref_dir: Path) -> dict:
    config = read_json(ref_dir / "result_config.json")
    caches = [Path(i["outer_train.npz"]["path"]).parent for i in _identities(config, "cache")]
    for c, i in zip(caches, _identities(config, "cache")):
        _check(c, i)
    devs = [Path(i["outer_train.npz"]["path"]).parent for i in _identities(config, "dev_cache")] or None
    for c, i in zip(devs or [], _identities(config, "dev_cache")):
        _check(c, i)
    splits = []
    for item in config["split_files"]:
        path = Path(item["path"])
        if canonical_sha256(read_json(path)) != item["sha256"]:
            raise ValueError(f"{path}: split hash differs from the reference run.")
        splits.append(path)
    return {"ref_dir": str(ref_dir), "cache_dirs": [str(c) for c in caches],
            "dev_caches": [str(c) for c in devs] if devs else None, "split_files": [str(s) for s in splits],
            "protocol": config["protocol"], "route": config["route"], "l2_grid": list(config["l2_grid"]),
            "base_model": config.get("base_model"), "reference_schema": config.get("schema_version")}


def _predictions(route_dir: Path) -> list[list[tuple]]:
    folds = []
    for fold_dir in sorted(route_dir.glob("fold_*")):
        rows = [json.loads(line) for line in (fold_dir / "predictions_subject_level.jsonl").read_text().splitlines()]
        folds.append([(r["subject_id"], r["prediction"], round(r["mean_logit"], 9)) for r in rows])
    return folds


def _job(args: tuple) -> str:
    variant, seed, key, ref, out = args
    augment = None
    if variant != "none":
        method, ratio = variant.rsplit("_r", 1)
        augment = {"method": method, "ratio": float(ratio), "seed": seed}
    t0 = time.time()
    summary = run_route(cache_dirs=[Path(c) for c in ref["cache_dirs"]],
                        dev_caches=[Path(c) for c in ref["dev_caches"]] if ref["dev_caches"] else None,
                        split_files=[Path(s) for s in ref["split_files"]], protocol=ref["protocol"],
                        output_dir=Path(out), route={**ref["route"], "augmentation_variant": variant,
                                                     "augmentation_seed": seed},
                        grid=tuple(ref["l2_grid"]), augment=augment)
    if variant == "none" and _predictions(Path(out)) != _predictions(Path(ref["ref_dir"])):
        raise SystemExit(f"{key}: no-augmentation rerun does not reproduce {ref['ref_dir']}.")
    fm = summary["fold_mean"]
    return (f"{variant:9s} s{seed if seed is not None else '-'} {key:40s} macroF1 {fm['macro_f1']:.3f} "
            f"posF1 {fm['positive_f1']:.3f} UAR {fm['uar']:.3f} {time.time() - t0:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ref-root", required=True, type=Path, help="<dataset>/<route>/ reference head runs.")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--variants", nargs="+",
                        default=["none", *(f"{m}_r{r}" for m in AUG_METHODS for r in RATIOS)])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--routes", nargs="+", default=list(ROUTES))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    git = _git()
    if git["dirty"]:
        raise SystemExit("Refusing to run from a dirty worktree (provenance).")
    refs = {f"{d}/{r}": load_reference(args.ref_root / d / r) for d in DATASETS for r in args.routes}
    args.output_root.mkdir(parents=True, exist_ok=True)
    tag = hashlib.sha1(json.dumps([args.variants, args.seeds, args.routes]).encode()).hexdigest()[:8]
    save_json({"git": git, "variants": args.variants, "seeds": args.seeds, "ratios": list(RATIOS),
               "started": time.strftime("%FT%TZ", time.gmtime()), "routes": refs},
              args.output_root / f"RUN_MANIFEST_{tag}.json")
    jobs = []
    for variant in args.variants:
        for seed in ([None] if variant == "none" else args.seeds):
            base = args.output_root / variant / ("" if seed is None else f"seed{seed}")
            for key, ref in refs.items():
                jobs.append((variant, seed, key, ref, str(base / key)))
    with ProcessPoolExecutor(args.workers) as pool:
        for line in pool.map(_job, jobs):
            print(line, flush=True)


if __name__ == "__main__":
    main()
