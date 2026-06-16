"""Offline SECap emotion-caption extraction (runs in the ``secap`` env, GPU, once).

Iterates the unique chunk wavs of a manifest and produces a per-``sample_id``
Chinese caption via SECap, writing a resumable JSONL cache. Translation is a
SEPARATE pass (``translate.py``) so a translation failure never forces re-running
the expensive SECap stage.

SECap I/O contract (verified against ``SECap/model2.py`` + ``scripts/inference.py``):
- ``model = MotionAudio()`` (no ckpt arg; sub-model weights load relative to
  ``model2.py``). Checkpoint loads externally:
  ``model.load_state_dict(torch.load(ckpt, map_location='cpu'))`` then
  ``model = model.to('cuda')``.
- ``model.inference([wav])`` takes ONLY a list of float32 numpy waveforms @ 16 kHz
  and returns ``(candidates, prompt)``: ``candidates`` is a list of Chinese
  sentences (``inference`` runs 8 stochastic generations, ``post_processing``
  drops the 3 least-similar and keeps 5), ``prompt`` is the fixed Chinese
  instruction. Generation is hardcoded ``do_sample=True`` -> NON deterministic,
  hence the offline freeze. (``--fast`` cannot reduce generations without editing
  model2; it only pairs with ``--limit`` for small smoke runs.)

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
import types
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


def _locate_secap_dir(secap_root: Path) -> Path:
    """Return the directory that actually contains ``model2.py``.

    The cluster checkout may nest the SECap source under ``secap_root`` (e.g.
    ``secap_root/SECap/model2.py``), so we search a couple of levels deep instead
    of assuming ``model2.py`` sits at the top.
    """
    direct = secap_root / "model2.py"
    if direct.exists():
        return secap_root
    matches = sorted(secap_root.glob("*/model2.py")) + sorted(secap_root.glob("*/*/model2.py"))
    if matches:
        return matches[0].parent
    raise FileNotFoundError(
        f"Could not find model2.py under {secap_root}. Point --secap-root at the "
        f"SECap source dir (the one containing model2.py)."
    )


def _install_lightning_compat_inference_shim() -> None:
    """Provide the tiny Lightning surface SECap needs for offline inference.

    The cluster ``lightning`` package can import optional app/web dependencies
    (FastAPI/Pydantic) before exposing ``lightning.pytorch``. That is unnecessary
    for this script: we only instantiate ``MotionAudio``, load a checkpoint, move
    it to a device, and call ``inference``. A ``torch.nn.Module``-backed
    ``LightningModule`` shim avoids the fragile dependency chain while preserving
    the parts of the API used by SECap model classes.
    """
    import torch

    class InferenceLightningModule(torch.nn.Module):
        @property
        def device(self) -> torch.device:
            try:
                return next(self.parameters()).device
            except StopIteration:
                return torch.device("cpu")

        def save_hyperparameters(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
            return None

        def log(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
            return None

    shim = types.ModuleType("lightning_compat")
    shim.pl = types.SimpleNamespace(LightningModule=InferenceLightningModule)
    sys.modules["lightning_compat"] = shim


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
    # Resolve all I/O paths to absolute BEFORE chdir into the SECap dir.
    manifest_path = args.manifest.resolve()
    ckpt_path = args.ckpt.resolve()
    out_path = args.out.resolve()
    if shard_total > 1:
        out_path = out_path.with_name(f"{out_path.stem}.shard{shard_index}of{shard_total}{out_path.suffix}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    secap_dir = _locate_secap_dir(args.secap_root)
    sys.path.insert(0, str(secap_dir))
    # model2/SimiCal resolve some resources relative to cwd; run from the SECap dir.
    import os
    os.chdir(secap_dir)
    print(f"[extract_secap] SECap source dir: {secap_dir}", flush=True)

    rows = _unique_chunks(_read_manifest(manifest_path))
    rows = [row for i, row in enumerate(rows) if i % shard_total == shard_index]

    done = _already_done(out_path)
    pending = [row for row in rows if str(row["sample_id"]) not in done]
    if args.limit:
        pending = pending[: args.limit]
    print(f"[extract_secap] shard {shard_index}/{shard_total}: {len(pending)} pending "
          f"({len(done)} already cached) -> {out_path}", flush=True)

    if not pending:
        return 0

    import torch
    _install_lightning_compat_inference_shim()
    from model2 import MotionAudio  # type: ignore

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MotionAudio()
    state_dict = torch.load(str(ckpt_path), map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.to(torch.device(device))
    model.eval()

    ckpt_version = f"model.ckpt@{ckpt_path.stat().st_size}" if ckpt_path.exists() else "model.ckpt"

    with out_path.open("a", encoding="utf-8") as handle:
        for n, row in enumerate(pending, start=1):
            sample_id = str(row["sample_id"])
            wav = _load_wav(row["audio_path"])
            # inference() takes only the audio list; it always runs 8 sampled
            # generations and returns the 5 post-processed candidates + the prompt.
            candidates, prompt = model.inference([wav])
            if isinstance(candidates, str):
                candidates = [candidates]
            candidates = [str(c).strip() for c in candidates if str(c).strip()]
            # SECap already keeps the 5 most mutually-similar of 8 samples; take the
            # first as the deterministic canonical caption, store all for audit/QC.
            canonical = candidates[0] if candidates else ""
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
