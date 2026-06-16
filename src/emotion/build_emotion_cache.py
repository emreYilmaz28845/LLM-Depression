"""Thin orchestrator + validator for the offline emotion cache.

Merges SECap extraction shards (if any), and validates coverage of the final
translated cache against a manifest before training. Analogous to
``src/data/build_manifest.py``. Pure JSONL/text work — no SECap, no models — so it
can run in either env.

Examples:
    # merge shard files written by extract_secap --shard i/N
    python -m src.emotion.build_emotion_cache merge \
        --pattern 'outputs/emotion/daic_secap_zh.shard*of*.jsonl' \
        --out outputs/emotion/daic_secap_zh.jsonl

    # validate the final translated cache covers the manifest
    python -m src.emotion.build_emotion_cache validate \
        --cache outputs/emotion/daic_secap_en.jsonl \
        --manifest outputs/manifests/daic_manifest.jsonl
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _cmd_merge(args: argparse.Namespace) -> int:
    paths = sorted(glob.glob(args.pattern))
    if not paths:
        raise SystemExit(f"No shard files matched {args.pattern!r}.")
    merged: dict[str, dict] = {}
    for path in paths:
        for row in _read_jsonl(Path(path)):
            merged[str(row["sample_id"])] = row  # last writer wins
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for sample_id in sorted(merged):
            handle.write(json.dumps(merged[sample_id], ensure_ascii=False) + "\n")
    print(f"[build_emotion_cache] merged {len(paths)} shards -> {out_path} "
          f"({len(merged)} unique sample_ids)", flush=True)
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    cache_rows = _read_jsonl(Path(args.cache))
    cache = {str(row["sample_id"]): row for row in cache_rows}
    manifest_ids = [str(row["sample_id"]) for row in _read_jsonl(Path(args.manifest))]

    missing = [sid for sid in manifest_ids if sid not in cache]
    null_caption = [sid for sid in manifest_ids if sid in cache and not cache[sid].get("emotion_en")]
    covered = len(manifest_ids) - len(missing) - len(null_caption)

    print(f"[build_emotion_cache] coverage vs {args.manifest}:")
    print(f"  total manifest sample_ids : {len(manifest_ids)}")
    print(f"  covered (emotion_en set)  : {covered}")
    print(f"  missing cache rows        : {len(missing)}")
    print(f"  null/failed emotion_en    : {len(null_caption)}")
    if missing[:10]:
        print(f"  e.g. missing: {missing[:10]}")

    if args.strict and (missing or null_caption):
        raise SystemExit("Coverage incomplete under --strict.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    merge = sub.add_parser("merge", help="Merge extraction shard JSONLs.")
    merge.add_argument("--pattern", required=True)
    merge.add_argument("--out", required=True)
    merge.set_defaults(func=_cmd_merge)

    validate = sub.add_parser("validate", help="Validate cache coverage vs a manifest.")
    validate.add_argument("--cache", required=True)
    validate.add_argument("--manifest", required=True)
    validate.add_argument("--strict", action="store_true")
    validate.set_defaults(func=_cmd_validate)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
