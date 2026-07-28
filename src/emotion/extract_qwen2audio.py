"""Offline Qwen2-Audio emotion-caption extraction (runs in the ``qwen`` env, GPU, once).

Uses base **Qwen2-Audio-7B-Instruct** itself as an audio-only emotion captioner and
writes the same frozen JSONL cache shape that ``src/data/emotion.py`` already
consumes via ``data.use_emotion``. SECap and Qwen2-Audio are therefore
interchangeable cache producers: training/eval prompt injection is unchanged and
``data.emotion_caption_field: emotion_en`` stays valid.

Unlike SECap (Chinese, hardcoded ``do_sample=True``), Qwen2-Audio captions are
generated with **deterministic** decoding (``do_sample=False``, ``num_beams=1``) and
directly populate ``emotion_en`` in English, so no separate translation pass is
needed. Extraction is still done ONCE offline and frozen; training/eval must never
generate captions dynamically (see ``secap_implementation.md``).

The captioning prompt is intentionally **audio-only** (no transcript) and instructs
the model to describe prosody/affect without inferring depression or diagnosis, so
the emotion channel reflects voice rather than lexical depression cues, and the
base (non-finetuned) checkpoint avoids label leakage.

Example (cluster):
    python -m src.emotion.extract_qwen2audio \
        --manifest outputs/manifests/daic_manifest.jsonl \
        --model /gpfs/.../models/Qwen2-Audio-7B-Instruct \
        --out outputs/emotion/daic_qwen2audio_en.jsonl \
        --shard 0/8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf


EMOTION_PROMPT = (
    "Describe the speaker's emotional state from the speech audio in one short "
    "sentence. Base this only on vocal qualities: pitch, loudness, speed, rhythm, "
    "and tone of voice. Do not reference, quote, or infer what the speaker said, "
    "and do not infer depression or diagnosis."
)
EMOTION_SOURCE = "qwen2audio"
DEFAULT_MAX_NEW_TOKENS = 48


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


def _load_wav(
    path: str,
    target_sr: int,
    start_time: float | None = None,
    end_time: float | None = None,
) -> np.ndarray:
    from src.data.runtime import load_audio_array

    return load_audio_array(path, target_sr, None, False, start_time, end_time)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path, help="Qwen2-Audio-7B-Instruct dir.")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--shard", default=None, help="i/N for SLURM array sharding.")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--limit", type=int, default=0, help="Stop after N chunks (debug).")
    args = parser.parse_args(argv)

    shard_index, shard_total = _parse_shard(args.shard)
    manifest_path = args.manifest.resolve()
    model_path = args.model.resolve()
    out_path = args.out.resolve()
    if shard_total > 1:
        out_path = out_path.with_name(f"{out_path.stem}.shard{shard_index}of{shard_total}{out_path.suffix}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = _unique_chunks(_read_manifest(manifest_path))
    rows = [row for i, row in enumerate(rows) if i % shard_total == shard_index]

    done = _already_done(out_path)
    pending = [row for row in rows if str(row["sample_id"]) not in done]
    if args.limit:
        pending = pending[: args.limit]
    print(f"[extract_qwen2audio] shard {shard_index}/{shard_total}: {len(pending)} pending "
          f"({len(done)} already cached) -> {out_path}", flush=True)

    if not pending:
        return 0

    import torch
    from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(str(model_path))
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        str(model_path), torch_dtype=torch_dtype
    )
    model = model.to(torch.device(device))
    model.eval()

    target_sr = int(processor.feature_extractor.sampling_rate)
    generation_config = {
        "max_new_tokens": int(args.max_new_tokens),
        "do_sample": False,
        "num_beams": 1,
    }
    model_version = f"{model_path.name}"

    # Audio-only conversation; the wav itself is passed via the processor's
    # ``audios`` argument, so the content placeholder carries no path.
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": ""},
                {"type": "text", "text": EMOTION_PROMPT},
            ],
        }
    ]
    chat_text = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )

    with out_path.open("a", encoding="utf-8") as handle:
        for n, row in enumerate(pending, start=1):
            sample_id = str(row["sample_id"])
            caption = ""
            caption_ok = False
            try:
                wav = _load_wav(
                    row["audio_path"],
                    target_sr,
                    row.get("start_time"),
                    row.get("end_time"),
                )
                # transformers==4.55.0's Qwen2AudioProcessor.__call__ takes the
                # audio kwarg as `audio` (singular); `audios` is silently dropped
                # (logged as an invalid kwarg), which fed captions with NO audio
                # conditioning at all.
                inputs = processor(
                    text=chat_text,
                    audio=[wav],
                    sampling_rate=target_sr,
                    return_tensors="pt",
                    padding=True,
                )
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                input_len = inputs["input_ids"].shape[1]
                with torch.no_grad():
                    generated = model.generate(**inputs, **generation_config)
                generated = generated[:, input_len:]
                caption = processor.batch_decode(
                    generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0].strip()
                caption_ok = bool(caption)
            except Exception as exc:  # noqa: BLE001 - log and continue; resumable
                print(f"[extract_qwen2audio] FAILED sample_id={sample_id}: {exc}", flush=True)

            record = {
                "dataset": row.get("dataset"),
                "subject_id": row.get("subject_id"),
                "sample_id": sample_id,
                "audio_path": row["audio_path"],
                "emotion_en": caption or None,
                "emotion_source": EMOTION_SOURCE,
                "emotion_prompt": EMOTION_PROMPT,
                "qwen2audio_model": model_version,
                "generation_config": generation_config,
                "caption_ok": caption_ok,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            if n % 25 == 0 or n == len(pending):
                print(f"[extract_qwen2audio] {n}/{len(pending)} done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
