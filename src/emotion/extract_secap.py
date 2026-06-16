"""Offline SECap emotion-caption extraction (runs in the ``secap`` env, GPU, once).

Iterates the unique chunk wavs of a manifest and produces a per-``sample_id``
Chinese caption via SECap, writing a resumable JSONL cache. Translation is a
SEPARATE pass (``translate.py``) so a translation failure never forces re-running
the expensive SECap stage.

SECap I/O contract (from ``SECap/model2.py``, see ``secap_implementation.md`` §1.3):
- ``model = MotionAudio()``; ``model.inference([wav])`` with ``wav`` a float32
  numpy waveform @ 16 kHz returns ``(candidates, prompt)`` where ``candidates`` is
  a list of Chinese sentences (``post_processing`` keeps 5) and ``prompt`` is the
  fixed Chinese instruction. ``inference`` runs 8 stochastic generations -> NON
  deterministic, hence the offline freeze.

Example (cluster):
    python -m src.emotion.extract_secap \
        --manifest outputs/manifests/daic_manifest.jsonl \
        --secap-root /gpfs/projects/etur92/SECap \
        --ckpt /gpfs/projects/etur92/SECap/model.ckpt \
        --out outputs/emotion/daic_secap_zh.jsonl \
        --shard 0/8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


SECAP_PROMPT = "请用一句中文简述音频里说话者的情感表现："
TARGET_SR = 16000


def _parse_shard(value: str | None) -> tuple[int, int]:
    if not value:
        return 0, 1
    index_str, total_str = value.split("/", 1)
    index, total = int(index_str), int(total_str)
    if total < 1 or not (0 <= index < total):
        raise ValueError(f"Invalid --shard {value!r}; expected i/N with 0<=i<N.")
    return index, total


def _read_manifest(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _unique_chunks(rows: list[dict]) -> list[dict]:
    seen: set[str] = set()
    unique: list[dict] = []
    for row in rows:
        sample_id = str(row["sample_id"])
        if sample_id in seen:
            continue
        seen.add(sample_id)
        unique.append(row)
    return sorted(unique, key=lambda item: str(item["sample_id"]))


def _already_done(out_path: Path) -> set[str]:
    done: set[str] = set()
    if out_path.exists():
        with out_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(str(json.loads(line)["sample_id"]))
                except (KeyError, json.JSONDecodeError):
                    continue
    return done


def _load_wav(path: str) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if int(sr) != TARGET_SR:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
    return np.asarray(audio, dtype=np.float32)


def _select_medoid(model, candidates: list[str]) -> str:
    """Pick the candidate most central to the set (medoid) via SECap's SimiCal.

    Falls back to the first candidate if similarity scoring is unavailable.
    """
    if not candidates:
        return ""
    if len(candidates) == 1:
        return candidates[0]
    try:
        from model2 import SimiCal  # type: ignore

        scorer = SimiCal()
        best_idx, best_score = 0, -1.0
        for i, cand_i in enumerate(candidates):
            total = 0.0
            for j, cand_j in enumerate(candidates):
                if i == j:
                    continue
                total += float(scorer(cand_i, cand_j))
            if total > best_score:
                best_idx, best_score = i, total
        return candidates[best_idx]
    except Exception:  # noqa: BLE001 - any failure -> deterministic first candidate
        return candidates[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--secap-root", required=True, type=Path, help="SECap repo root (added to sys.path).")
    parser.add_argument("--ckpt", required=True, type=Path, help="SECap model.ckpt (~15 GB).")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--shard", default=None, help="i/N for SLURM array sharding.")
    parser.add_argument("--fast", action="store_true", help="Single greedy generation (smoke test only).")
    parser.add_argument("--limit", type=int, default=0, help="Stop after N chunks (debug).")
    args = parser.parse_args(argv)

    shard_index, shard_total = _parse_shard(args.shard)
    sys.path.insert(0, str(args.secap_root))

    rows = _unique_chunks(_read_manifest(args.manifest))
    rows = [row for i, row in enumerate(rows) if i % shard_total == shard_index]
    out_path = args.out
    if shard_total > 1:
        out_path = out_path.with_name(f"{out_path.stem}.shard{shard_index}of{shard_total}{out_path.suffix}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = _already_done(out_path)
    pending = [row for row in rows if str(row["sample_id"]) not in done]
    if args.limit:
        pending = pending[: args.limit]
    print(f"[extract_secap] shard {shard_index}/{shard_total}: {len(pending)} pending "
          f"({len(done)} already cached) -> {out_path}", flush=True)

    if not pending:
        return 0

    from model2 import MotionAudio  # type: ignore

    model = MotionAudio()
    if hasattr(model, "load_ckpt"):
        model.load_ckpt(str(args.ckpt))

    ckpt_version = f"model.ckpt@{args.ckpt.stat().st_size}" if args.ckpt.exists() else "model.ckpt"

    with out_path.open("a", encoding="utf-8") as handle:
        for n, row in enumerate(pending, start=1):
            sample_id = str(row["sample_id"])
            wav = _load_wav(row["audio_path"])
            try:
                if args.fast and hasattr(model, "inference"):
                    candidates, prompt = model.inference([wav], do_sample=False, num_beams=1)
                else:
                    candidates, prompt = model.inference([wav])
            except TypeError:
                candidates, prompt = model.inference([wav])
            if isinstance(candidates, str):
                candidates = [candidates]
            candidates = [str(c).strip() for c in candidates if str(c).strip()]
            canonical = _select_medoid(model, candidates)
            record = {
                "dataset": row.get("dataset"),
                "subject_id": row.get("subject_id"),
                "sample_id": sample_id,
                "audio_path": row["audio_path"],
                "secap_prompt": prompt if isinstance(prompt, str) else SECAP_PROMPT,
                "emotion_zh": canonical,
                "emotion_zh_candidates": candidates,
                "emotion_en": None,
                "translation_ok": False,
                "secap_model_version": ckpt_version,
                "secap_fast": bool(args.fast),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            if n % 25 == 0 or n == len(pending):
                print(f"[extract_secap] {n}/{len(pending)} done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
