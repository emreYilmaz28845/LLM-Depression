"""Offline zh->en translation pass over the cached SECap Chinese captions.

Reads the ``extract_secap.py`` JSONL, adds ``emotion_en`` (+ ``translation_ok``,
``translation_model``), writes the final cache consumed by training. Kept separate
from extraction so a translation crash never forces re-running SECap, and so the
translator can be swapped without re-extracting.

Primary model: ``facebook/nllb-200-distilled-600M`` (more stable than opus-mt on
short emotive captions; opus-mt repetition-collapses — see plan §5.3). Deterministic
greedy/beam decode. A repetition/empty-output guard marks degenerate outputs as
failed so the training-time missing-caption policy applies.

Example:
    python -m src.emotion.translate \
        --in outputs/emotion/daic_secap_zh.jsonl \
        --out outputs/emotion/daic_secap_en.jsonl \
        --model facebook/nllb-200-distilled-600M
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _is_degenerate(text: str) -> bool:
    """Detect repetition-collapse / empty output (e.g. 'Happy, happy, happy...')."""
    if not text or not text.strip():
        return True
    words = re.findall(r"[A-Za-z']+", text.lower())
    if len(words) >= 4 and len(set(words)) <= max(1, len(words) // 3):
        return True
    return False


class _Translator:
    def __init__(self, model_name: str, src_lang: str, tgt_lang: str, num_beams: int):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.model_name = model_name
        self.is_nllb = "nllb" in model_name.lower()
        self.tgt_lang = tgt_lang
        self.num_beams = num_beams
        tok_kwargs = {"src_lang": src_lang} if self.is_nllb else {}
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **tok_kwargs)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.eval()

    def translate(self, texts: list[str]) -> list[str]:
        import torch

        if not texts:
            return []
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        gen_kwargs = {"num_beams": self.num_beams, "do_sample": False, "max_new_tokens": 64}
        if self.is_nllb:
            forced = self.tokenizer.convert_tokens_to_ids(self.tgt_lang)
            gen_kwargs["forced_bos_token_id"] = forced
        with torch.no_grad():
            output = self.model.generate(**inputs, **gen_kwargs)
        return self.tokenizer.batch_decode(output, skip_special_tokens=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="in_path", required=True, type=Path)
    parser.add_argument("--out", dest="out_path", required=True, type=Path)
    parser.add_argument("--model", default="facebook/nllb-200-distilled-600M")
    parser.add_argument("--src-lang", default="zho_Hans", help="NLLB source lang code.")
    parser.add_argument("--tgt-lang", default="eng_Latn", help="NLLB target lang code.")
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args(argv)

    rows = _read_jsonl(args.in_path)
    translator = _Translator(args.model, args.src_lang, args.tgt_lang, args.num_beams)

    texts = [str(row.get("emotion_zh") or "") for row in rows]
    translations: list[str] = []
    for start in range(0, len(texts), args.batch_size):
        batch = texts[start : start + args.batch_size]
        translations.extend(translator.translate(batch))
        print(f"[translate] {min(start + args.batch_size, len(texts))}/{len(texts)}", flush=True)

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    n_failed = 0
    with args.out_path.open("w", encoding="utf-8") as handle:
        for row, en in zip(rows, translations):
            en = (en or "").strip()
            ok = bool(row.get("emotion_zh")) and not _is_degenerate(en)
            if not ok:
                n_failed += 1
            record = dict(row)
            record["emotion_en"] = en if ok else None
            record["translation_ok"] = ok
            record["translation_model"] = args.model
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[translate] wrote {len(rows)} rows -> {args.out_path} "
          f"({n_failed} failed/degenerate)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
