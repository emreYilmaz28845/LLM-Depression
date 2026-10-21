#!/usr/bin/env python3
"""Does a Qwen3-Omni Thinker checkpoint actually use the audio? A small probe.

Diagnostic only: it runs on a handful of inputs and its numbers never enter a
results table. It loads the base Thinker once (device-map sharded), attaches the
audio+text and the audio-only LoRA adapters of the same dataset/fold, and runs
three probes on the same selected audio windows:

* ``attention`` — layer-by-layer attention mass from the answer positions onto
  each prompt region (audio tokens, transcript, system prompt, other text, ...).
  The decision position is the last prompt token (it predicts the first label
  token); the label positions are the teacher-forced gold-label tokens. The
  attention rows are recomputed exactly from each attention module's own
  q/k projections, so the model keeps its sdpa kernels; every layer also checks
  the recomputed rows against the module output (``max_rel_err``).
* ``ablation`` — candidate-label likelihood scores with the real audio, with
  silence of the same length, and with the other selected subject's audio.
* ``prosody`` — the base model (and the audio-only adapter) is asked to describe
  speech rate, pitch variation, pauses and loudness before a judgment. A
  silence input is the hallucination control, and simple acoustic measurements
  of the same audio are written beside the descriptions.

Outputs never contain subject identifiers or audio paths in the CSV/JSON
tables; examples are referenced by a short hash of their sample id. The
selection (with subject ids) is written separately to ``selection_private.json``.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

SCHEMA_VERSION = "audiollm.audio_attention_probe.v1"
REGIONS = (
    "sink",
    "system",
    "audio_marker",
    "audio",
    "user_text",
    "transcript",
    "template",
    "label",
)
REGION_INDEX = {name: index for index, name in enumerate(REGIONS)}
TRANSCRIPT_LEAD = "The transcript of the subject's speech is:\n"
AUDIO_PAD = "<|audio_pad|>"
AUDIO_START = "<|audio_start|>"
AUDIO_END = "<|audio_end|>"
ATTENTION_CLASS_NAME = "Qwen3OmniMoeThinkerTextAttention"

PROSODY_SYSTEM = "You are a careful listener who describes how speech sounds."
PROSODY_USER = (
    "Listen to the speech audio. First describe the speaker's prosody, based only on how "
    "the person sounds and not on what they say:\n"
    "1. Speech rate (slow, normal, or fast)\n"
    "2. Pitch variation (monotone, normal, or expressive)\n"
    "3. Pauses (few or many; short or long)\n"
    "4. Loudness and energy (low, normal, or high)\n"
    "If you cannot hear speech, say so. Then, on the last line, write exactly "
    "'Judgment: Depressed' or 'Judgment: Non-depressed'."
)


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested without a model)
# ---------------------------------------------------------------------------


def example_ref(sample_id: str) -> str:
    return hashlib.sha256(str(sample_id).encode("utf-8")).hexdigest()[:16]


def expand_audio_placeholders(text: str, audio_counts: list[int]) -> str:
    """Repeat each ``<|audio_pad|>`` the way the processor expands it."""
    parts = text.split(AUDIO_PAD)
    if len(parts) - 1 != len(audio_counts):
        raise ValueError(
            f"text has {len(parts) - 1} audio placeholders but {len(audio_counts)} audio counts"
        )
    pieces = [parts[0]]
    for count, tail in zip(audio_counts, parts[1:]):
        pieces.append(AUDIO_PAD * int(count))
        pieces.append(tail)
    return "".join(pieces)


def audio_run_lengths(input_ids: list[int], audio_token_id: int) -> list[int]:
    """Lengths of the consecutive audio-token runs, in order."""
    runs: list[int] = []
    current = 0
    for token in input_ids:
        if token == audio_token_id:
            current += 1
        elif current:
            runs.append(current)
            current = 0
    if current:
        runs.append(current)
    return runs


def assign_regions(
    *,
    input_ids: list[int],
    offsets: list[tuple[int, int]] | None,
    text: str,
    prompt_len: int,
    transcript: str,
    audio_token_id: int,
    audio_marker_ids: set[int],
) -> tuple[np.ndarray, str]:
    """Region code per token. Returns the codes and the mapping method used.

    With character offsets, text tokens are split into system / user text /
    transcript / chat template by their character position. Without offsets
    only the id-based regions (audio, markers, label, sink) are exact and every
    other prompt token is ``user_text``.
    """
    regions = np.full(len(input_ids), REGION_INDEX["user_text"], dtype=np.int64)
    method = "ids_only"
    if offsets is not None:
        method = "char_offsets"
        user_start = text.find("<|im_start|>user")
        assistant_start = text.rfind("<|im_start|>assistant")
        transcript_start = transcript_end = -1
        if transcript:
            lead = text.find(TRANSCRIPT_LEAD)
            if lead >= 0 and text.startswith(transcript, lead + len(TRANSCRIPT_LEAD)):
                transcript_start = lead + len(TRANSCRIPT_LEAD)
                transcript_end = transcript_start + len(transcript)
            else:
                method = "char_offsets_transcript_not_found"
        for index, (start, _end) in enumerate(offsets):
            if 0 <= user_start and start < user_start:
                regions[index] = REGION_INDEX["system"]
            elif 0 <= assistant_start <= start:
                regions[index] = REGION_INDEX["template"]
            elif transcript_start <= start < transcript_end:
                regions[index] = REGION_INDEX["transcript"]
    for index, token in enumerate(input_ids):
        if token == audio_token_id:
            regions[index] = REGION_INDEX["audio"]
        elif token in audio_marker_ids:
            regions[index] = REGION_INDEX["audio_marker"]
    regions[prompt_len:] = REGION_INDEX["label"]
    regions[0] = REGION_INDEX["sink"]
    return regions, method


def region_mass(weights: np.ndarray, regions: np.ndarray, n_regions: int = len(REGIONS)) -> np.ndarray:
    """Sum attention weights per region.

    ``weights`` is ``[heads, queries, keys]``; returns ``[heads, queries, regions]``.
    """
    heads, queries, keys = weights.shape
    if regions.shape[0] != keys:
        raise ValueError("regions must have one code per key position")
    one_hot = np.zeros((keys, n_regions), dtype=np.float64)
    one_hot[np.arange(keys), regions] = 1.0
    return weights.astype(np.float64) @ one_hot


def uniform_share(regions: np.ndarray, query_position: int) -> np.ndarray:
    """Share of each region among the keys a causal query can see."""
    visible = regions[: query_position + 1]
    counts = np.bincount(visible, minlength=len(REGIONS)).astype(np.float64)
    return counts / float(len(visible))


def causal_rows_softmax(scores, query_positions):
    """Mask keys after each query position and softmax in float32 (torch)."""
    import torch

    keys = scores.shape[-1]
    key_index = torch.arange(keys, device=scores.device)
    query_index = torch.as_tensor(query_positions, device=scores.device)
    future = key_index.unsqueeze(0) > query_index.unsqueeze(1)
    scores = scores.float().masked_fill(future, float("-inf"))
    return torch.softmax(scores, dim=-1)


def parse_judgment(text: str) -> str:
    for line in reversed([line.strip() for line in text.splitlines() if line.strip()]):
        lowered = line.lower()
        if "judgment" in lowered or "judgement" in lowered:
            if "non-depressed" in lowered or "non depressed" in lowered:
                return "Non-depressed"
            if "depressed" in lowered:
                return "Depressed"
    return "INVALID"


def acoustic_summary(audio: np.ndarray, sampling_rate: int) -> dict[str, Any]:
    """Rough acoustic measures to set beside the model's prosody description."""
    import librosa

    audio = np.asarray(audio, dtype=np.float32)
    duration = len(audio) / float(sampling_rate)
    result: dict[str, Any] = {"duration_s": round(duration, 3)}
    if duration <= 0 or float(np.max(np.abs(audio))) == 0.0:
        result["silent"] = True
        return result
    result["silent"] = False
    intervals = librosa.effects.split(audio, top_db=30, frame_length=1024, hop_length=256)
    voiced = float(sum(end - start for start, end in intervals)) / sampling_rate
    gaps = [
        (intervals[index + 1][0] - intervals[index][1]) / sampling_rate
        for index in range(len(intervals) - 1)
    ]
    pauses = [gap for gap in gaps if gap >= 0.25]
    result.update(
        {
            "speech_time_s": round(voiced, 3),
            "speech_fraction": round(voiced / duration, 3),
            "pauses_ge_250ms": len(pauses),
            "pause_mean_s": round(float(np.mean(pauses)), 3) if pauses else 0.0,
            "pause_total_s": round(float(np.sum(pauses)), 3),
        }
    )
    f0, _voiced_flag, _prob = librosa.pyin(
        audio, fmin=65.0, fmax=400.0, sr=sampling_rate, frame_length=1024
    )
    f0 = f0[np.isfinite(f0)]
    if f0.size:
        semitones = 12.0 * np.log2(f0 / np.median(f0))
        result.update(
            {
                "f0_median_hz": round(float(np.median(f0)), 1),
                "f0_sd_semitones": round(float(np.std(semitones)), 2),
                "f0_range_p5_p95_semitones": round(
                    float(np.percentile(semitones, 95) - np.percentile(semitones, 5)), 2
                ),
            }
        )
    onsets = librosa.onset.onset_detect(y=audio, sr=sampling_rate, units="time")
    result["onsets_per_speech_s"] = round(len(onsets) / voiced, 2) if voiced > 0 else None
    rms = librosa.feature.rms(y=audio)[0]
    rms = rms[rms > 0]
    if rms.size:
        result["rms_dbfs_median"] = round(float(np.median(20.0 * np.log10(rms))), 1)
    return result


# ---------------------------------------------------------------------------
# Attention recorder
# ---------------------------------------------------------------------------


class AttentionRecorder:
    """Forward hooks that recompute selected attention rows of every text layer."""

    def __init__(self, model, query_positions: list[int], regions: np.ndarray, n_regions: int = len(REGIONS)):
        self.query_positions = list(query_positions)
        self.regions = regions
        self.n_regions = int(n_regions)
        self.layers: list[dict[str, Any]] = []
        self.handles = []
        modules = [
            module for module in model.modules() if type(module).__name__ == ATTENTION_CLASS_NAME
        ]
        if not modules:
            raise RuntimeError(f"no {ATTENTION_CLASS_NAME} modules found")
        for module in modules:
            self.handles.append(module.register_forward_hook(self._hook, with_kwargs=True))

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def _hook(self, module, args, kwargs, output):
        import torch

        modeling = sys.modules[type(module).__module__]
        hidden = kwargs.get("hidden_states", args[0] if args else None)
        cos, sin = kwargs["position_embeddings"]
        if hidden.shape[0] != 1:
            raise RuntimeError("the attention probe expects batch size 1")
        with torch.no_grad():
            shape = (*hidden.shape[:-1], -1, module.head_dim)
            query = module.q_norm(module.q_proj(hidden).view(shape)).transpose(1, 2)
            key = module.k_norm(module.k_proj(hidden).view(shape)).transpose(1, 2)
            value = module.v_proj(hidden).view(shape).transpose(1, 2)
            query, key = modeling.apply_rotary_pos_emb(query, key, cos, sin)
            rows = torch.as_tensor(self.query_positions, device=query.device)
            query = query[:, :, rows]
            key = modeling.repeat_kv(key, module.num_key_value_groups)
            value = modeling.repeat_kv(value, module.num_key_value_groups)
            scores = torch.matmul(query.float(), key.float().transpose(2, 3)) * module.scaling
            weights = causal_rows_softmax(scores, self.query_positions)
            # Self-check: the recomputed rows must reproduce the module output.
            recomputed = torch.matmul(weights.to(value.dtype), value)
            recomputed = recomputed.transpose(1, 2).reshape(1, len(self.query_positions), -1)
            recomputed = module.o_proj(recomputed).float()
            actual = output[0][0, rows].float()
            rel_err = (
                (recomputed[0] - actual).norm(dim=-1) / actual.norm(dim=-1).clamp_min(1e-6)
            ).max()
            weights_np = weights[0].cpu().numpy()
        self.layers.append(
            {
                "layer_idx": int(getattr(module, "layer_idx", len(self.layers))),
                "mass": region_mass(weights_np, self.regions, self.n_regions),
                "max_rel_err": float(rel_err.item()),
            }
        )


# ---------------------------------------------------------------------------
# Model-side probes
# ---------------------------------------------------------------------------


def _load_saved_config(fold_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    import yaml

    saved = yaml.safe_load((fold_dir / "run_config.yaml").read_text(encoding="utf-8"))
    if not isinstance(saved, dict) or not isinstance(saved.get("config"), dict):
        raise ValueError(f"invalid run_config.yaml under {fold_dir}")
    return saved, saved["config"]


def _select_subjects(predictions_csv: Path) -> dict[int, str]:
    """First correctly predicted subject of each class, by sorted subject id."""
    with predictions_csv.open(encoding="utf-8", newline="") as handle:
        rows = sorted(csv.DictReader(handle), key=lambda row: str(row["subject_id"]))
    chosen: dict[int, str] = {}
    for row in rows:
        label = int(row["label"])
        if label not in chosen and str(row.get("prediction")) == str(label):
            chosen[label] = str(row["subject_id"])
    if set(chosen) != {0, 1}:
        raise RuntimeError(f"could not find a correct prediction for both classes in {predictions_csv}")
    return chosen


def _processor_inputs(processor, text: str, audio_arrays: list, sampling_rate: int, device):
    import torch

    kwargs: dict[str, Any] = {"text": text, "return_tensors": "pt", "padding": False}
    if audio_arrays:
        kwargs["audio"] = audio_arrays
        kwargs["sampling_rate"] = int(sampling_rate)
    inputs = processor(**kwargs)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    # The backend's audio dtype boundary is a forward pre-hook on the base model,
    # which PEFT's LoraModel.forward bypasses; cast the features here instead.
    features = inputs.get("input_features")
    if features is not None and features.is_floating_point():
        inputs["input_features"] = features.to(torch.bfloat16)
    return inputs


def _tokenizer_offsets(processor, text: str, input_ids: list[int], audio_token_id: int):
    tokenizer = getattr(processor, "tokenizer", processor)
    expanded = expand_audio_placeholders(text, audio_run_lengths(input_ids, audio_token_id))
    try:
        encoded = tokenizer(expanded, add_special_tokens=False, return_offsets_mapping=True)
    except Exception:  # noqa: BLE001 - a slow tokenizer has no offsets
        return None, expanded
    if list(encoded["input_ids"]) != list(input_ids):
        return None, expanded
    return [tuple(pair) for pair in encoded["offset_mapping"]], expanded


def _adapter_context(model, adapter: str | None):
    if adapter is None:
        return model.disable_adapter()
    model.set_adapter(adapter)
    return contextlib.nullcontext()


def run_attention(model, processor, example, label_text, audio_arrays, sampling_rate, device, ids):
    import torch

    prompt = example["prompt_text"]
    prompt_inputs = _processor_inputs(processor, prompt, audio_arrays, sampling_rate, device)
    prompt_len = int(prompt_inputs["input_ids"].shape[1])
    full_inputs = _processor_inputs(processor, prompt + label_text, audio_arrays, sampling_rate, device)
    input_ids = full_inputs["input_ids"][0].tolist()
    if input_ids[:prompt_len] != prompt_inputs["input_ids"][0].tolist():
        raise RuntimeError("prompt tokens are not a prefix of prompt+label tokens")
    offsets, expanded = _tokenizer_offsets(processor, prompt + label_text, input_ids, ids["audio"])
    regions, method = assign_regions(
        input_ids=input_ids,
        offsets=offsets,
        text=expanded,
        prompt_len=prompt_len,
        transcript=str(example.get("transcript", "")) if example.get("transcript") else "",
        audio_token_id=ids["audio"],
        audio_marker_ids={ids["audio_start"], ids["audio_end"]},
    )
    query_positions = list(range(prompt_len - 1, len(input_ids)))
    recorder = AttentionRecorder(model, query_positions, regions)
    try:
        with torch.inference_mode():
            model(**full_inputs, use_cache=False)
    finally:
        recorder.remove()
    layers = sorted(recorder.layers, key=lambda item: item["layer_idx"])
    mass = np.stack([layer["mass"] for layer in layers])  # [layers, heads, queries, regions]
    return {
        "mass": mass,
        "max_rel_err": [layer["max_rel_err"] for layer in layers],
        "regions": regions,
        "query_positions": query_positions,
        "prompt_len": prompt_len,
        "seq_len": len(input_ids),
        "region_method": method,
        "region_token_counts": {
            name: int((regions == index).sum()) for name, index in REGION_INDEX.items()
        },
    }


def score_labels(model, processor, example, config, audio_arrays, sampling_rate, device):
    import torch

    from src.utils import internal_label_text_from_int

    prompt = example["prompt_text"]
    prompt_inputs = _processor_inputs(processor, prompt, audio_arrays, sampling_rate, device)
    prompt_len = int(prompt_inputs["input_ids"].shape[1])
    scores = {}
    for label in (1, 0):
        label_text = internal_label_text_from_int(config, label)
        full = _processor_inputs(processor, prompt + label_text, audio_arrays, sampling_rate, device)
        with torch.inference_mode():
            logits = model(**full, use_cache=False).logits[0]
        target = full["input_ids"][0, prompt_len:]
        selected = logits[prompt_len - 1 : full["input_ids"].shape[1] - 1]
        log_probs = torch.log_softmax(selected.float(), dim=-1)
        scores[label] = float(log_probs.gather(-1, target.unsqueeze(-1)).mean().item())
    return {"dep_score": scores[1], "non_score": scores[0], "margin": scores[1] - scores[0]}


def generate_prosody(model, processor, audio_arrays, sampling_rate, device, max_new_tokens):
    import torch

    from src.data.runtime import QWEN3OMNI_AUDIO_PLACEHOLDER

    text = (
        f"<|im_start|>system\n{PROSODY_SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\nAudio 1: {QWEN3OMNI_AUDIO_PLACEHOLDER}\n{PROSODY_USER}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    inputs = _processor_inputs(processor, text, audio_arrays, sampling_rate, device)
    input_len = int(inputs["input_ids"].shape[1])
    with torch.inference_mode():
        # Stop at the end of the assistant turn; without it the Thinker keeps
        # writing new user/assistant turns until max_new_tokens.
        tokenizer = getattr(processor, "tokenizer", processor)
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=int(tokenizer.convert_tokens_to_ids("<|im_end|>")),
        )
    sequences = generated.sequences if hasattr(generated, "sequences") else generated
    return processor.decode(
        sequences[0, input_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False
    ).strip()


def _layer_rows(condition: dict[str, Any], result: dict[str, Any]) -> list[dict[str, Any]]:
    mass = result["mass"]
    rows = []
    kinds = {"decision": [0], "label_mean": list(range(1, mass.shape[2]))}
    for kind, query_slots in kinds.items():
        if not query_slots:
            continue
        uniform = np.mean(
            [uniform_share(result["regions"], result["query_positions"][slot]) for slot in query_slots],
            axis=0,
        )
        for layer in range(mass.shape[0]):
            per_head = mass[layer][:, query_slots, :].mean(axis=1)  # [heads, regions]
            for region, index in REGION_INDEX.items():
                rows.append(
                    {
                        **condition,
                        "row_kind": kind,
                        "layer": layer,
                        "region": region,
                        "mass_mean_heads": round(float(per_head[:, index].mean()), 6),
                        "mass_max_heads": round(float(per_head[:, index].max()), 6),
                        "uniform_share": round(float(uniform[index]), 6),
                    }
                )
    return rows


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _summary_table(layer_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mean over layers of the decision-row mass for audio and transcript."""
    grouped: dict[tuple, dict[str, list[float]]] = {}
    for row in layer_rows:
        if row["row_kind"] != "decision" or row["region"] not in ("audio", "transcript", "sink"):
            continue
        key = (row["model"], row["adapter"], row["example"])
        bucket = grouped.setdefault(key, {})
        bucket.setdefault(row["region"], []).append(row["mass_mean_heads"])
        bucket.setdefault(f"{row['region']}_uniform", []).append(row["uniform_share"])
    table = []
    for (model, adapter, example), values in sorted(grouped.items()):
        entry = {"model": model, "adapter": adapter, "example": example}
        for region in ("audio", "transcript", "sink"):
            if region in values:
                entry[f"{region}_mass_layer_mean"] = round(float(np.mean(values[region])), 4)
                entry[f"{region}_mass_layer_max"] = round(float(np.max(values[region])), 4)
                entry[f"{region}_uniform_share"] = round(float(values[f"{region}_uniform"][0]), 4)
        table.append(entry)
    return table


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--audio-text-fold-dir", required=True, type=Path)
    parser.add_argument("--audio-only-fold-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=320)
    parser.add_argument("--skip-prosody", action="store_true")
    args = parser.parse_args(argv)

    import torch
    from peft import PeftModel

    from src.data.runtime import build_examples, filter_rows_by_subjects, load_manifest_rows
    from src.evaluate import _load_example_audio
    from src.model.qwen3omni_lora import load_additional_audio_modules
    from src.model.runtime import load_model_for_inference, load_processor, resolve_processor_sampling_rate
    from src.utils import internal_label_text_from_int, sha256_jsonl_rows

    started = time.time()
    out = args.output_dir
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"output dir {out} is not empty; refusing to overwrite")
    out.mkdir(parents=True, exist_ok=True)

    fold_dirs = {"audio_text": args.audio_text_fold_dir, "audio_only": args.audio_only_fold_dir}
    saved, configs = {}, {}
    for name, fold_dir in fold_dirs.items():
        saved[name], configs[name] = _load_saved_config(fold_dir)
    model_paths = {str(saved[name].get("resolved_model_name_or_path")) for name in fold_dirs}
    if len(model_paths) != 1:
        raise RuntimeError(f"the two checkpoints use different base models: {model_paths}")
    model_path = model_paths.pop()

    manifest_path = Path(saved["audio_text"]["manifest_path"])
    manifest_rows = load_manifest_rows(manifest_path)
    manifest_hash = sha256_jsonl_rows(manifest_rows)
    for name in fold_dirs:
        if saved[name].get("manifest_hash") != manifest_hash:
            raise RuntimeError(f"manifest hash mismatch for {name}: {manifest_hash}")

    chosen = _select_subjects(
        fold_dirs["audio_text"] / "best_model" / "standalone_eval" / "predictions_subject_level.csv"
    )
    selected: dict[str, dict[str, Any]] = {}
    for label, subject_id in sorted(chosen.items(), reverse=True):
        alias = "dep" if label == 1 else "nondep"
        rows = filter_rows_by_subjects(manifest_rows, [subject_id])
        per_model = {
            name: sorted(
                build_examples(rows, configs[name], partition_name="final_eval"),
                key=lambda item: str(item["sample_id"]),
            )
            for name in fold_dirs
        }
        middle = per_model["audio_text"][len(per_model["audio_text"]) // 2]
        pair = {}
        for name, examples in per_model.items():
            match = [item for item in examples if item["sample_id"] == middle["sample_id"]]
            if len(match) != 1:
                raise RuntimeError(f"sample {example_ref(middle['sample_id'])} missing for {name}")
            pair[name] = match[0]
        selected[alias] = {"label": label, "subject_id": subject_id, "examples": pair}
    (out / "selection_private.json").write_text(
        json.dumps(
            {
                alias: {
                    "label": item["label"],
                    "subject_id": item["subject_id"],
                    "sample_id": item["examples"]["audio_text"]["sample_id"],
                }
                for alias, item in selected.items()
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    processor = load_processor(fold_dirs["audio_text"] / "best_model", configs["audio_text"])
    sampling_rate = int(resolve_processor_sampling_rate(processor))
    base = load_model_for_inference(model_path, None, configs["audio_text"])
    for name, fold_dir in fold_dirs.items():
        extra = load_additional_audio_modules(base, fold_dir / "best_model")
        if extra.get("enabled") or extra.get("train_projector"):
            raise RuntimeError(f"{name} checkpoint carries extra audio modules; probe does not support it")
    model = PeftModel.from_pretrained(base, str(fold_dirs["audio_text"] / "best_model"), adapter_name="audio_text")
    model.load_adapter(str(fold_dirs["audio_only"] / "best_model"), adapter_name="audio_only")
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = getattr(processor, "tokenizer", processor)
    ids = {
        "audio": int(tokenizer.convert_tokens_to_ids(AUDIO_PAD)),
        "audio_start": int(tokenizer.convert_tokens_to_ids(AUDIO_START)),
        "audio_end": int(tokenizer.convert_tokens_to_ids(AUDIO_END)),
    }

    audio = {
        alias: _load_example_audio(item["examples"]["audio_text"], sampling_rate, False)
        for alias, item in selected.items()
    }
    other = {"dep": "nondep", "nondep": "dep"}

    layer_rows: list[dict[str, Any]] = []
    attention_meta: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []
    for model_name in ("audio_text", "audio_only"):
        for adapter in (model_name, None):
            adapter_tag = "finetuned" if adapter else "base"
            for alias, item in selected.items():
                example = item["examples"][model_name]
                label_text = internal_label_text_from_int(configs[model_name], item["label"])
                condition = {"model": model_name, "adapter": adapter_tag, "example": alias}
                with _adapter_context(model, adapter):
                    result = run_attention(
                        model, processor, example, label_text, audio[alias], sampling_rate, device, ids
                    )
                    np.savez_compressed(
                        out / f"attention_{model_name}_{adapter_tag}_{alias}.npz",
                        mass=result["mass"].astype(np.float32),
                        regions=result["regions"],
                        query_positions=np.asarray(result["query_positions"]),
                        region_names=np.asarray(REGIONS),
                    )
                    layer_rows.extend(_layer_rows(condition, result))
                    attention_meta.append(
                        {
                            **condition,
                            "example_ref": example_ref(example["sample_id"]),
                            "label": item["label"],
                            "prompt_len": result["prompt_len"],
                            "seq_len": result["seq_len"],
                            "region_method": result["region_method"],
                            "region_token_counts": result["region_token_counts"],
                            "self_check_max_rel_err": round(max(result["max_rel_err"]), 5),
                            "layers": len(result["max_rel_err"]),
                        }
                    )
                    for audio_condition, arrays in (
                        ("real", audio[alias]),
                        ("silence", [np.zeros_like(array) for array in audio[alias]]),
                        ("other_subject", audio[other[alias]]),
                    ):
                        scores = score_labels(
                            model, processor, example, configs[model_name], arrays, sampling_rate, device
                        )
                        ablation_rows.append(
                            {
                                **condition,
                                "label": item["label"],
                                "audio_condition": audio_condition,
                                **{key: round(value, 5) for key, value in scores.items()},
                                "prediction": int(scores["margin"] > 0),
                            }
                        )
                print(json.dumps({"done": condition, "elapsed_s": round(time.time() - started)}), flush=True)

    _write_csv(layer_rows, out / "attention_layers.csv")
    _write_csv(ablation_rows, out / "ablation_scores.csv")
    summary_table = _summary_table(layer_rows)
    _write_csv(summary_table, out / "attention_summary.csv")

    prosody_rows: list[dict[str, Any]] = []
    acoustic_rows: list[dict[str, Any]] = []
    if not args.skip_prosody:
        inputs = {alias: audio[alias] for alias in selected}
        inputs["silence"] = [np.zeros_like(array) for array in audio["dep"]]
        for alias, arrays in inputs.items():
            acoustic_rows.append({"input": alias, **acoustic_summary(arrays[0], sampling_rate)})
            for adapter in (None, "audio_only"):
                with _adapter_context(model, adapter):
                    text = generate_prosody(
                        model, processor, arrays, sampling_rate, device, args.max_new_tokens
                    )
                prosody_rows.append(
                    {
                        "input": alias,
                        "label": selected[alias]["label"] if alias in selected else None,
                        "adapter": adapter or "base",
                        "judgment": parse_judgment(text),
                        "response": text,
                    }
                )
        with (out / "prosody_generations.jsonl").open("w", encoding="utf-8") as handle:
            for row in prosody_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        _write_csv(acoustic_rows, out / "acoustic_measures.csv")

    report = {
        "schema_version": SCHEMA_VERSION,
        "model_path": model_path,
        "fold_dirs": {name: str(path) for name, path in fold_dirs.items()},
        "run_names": {name: saved[name].get("run_name") for name in fold_dirs},
        "manifest_path": str(manifest_path),
        "manifest_hash": manifest_hash,
        "split_metadata_hash": {name: saved[name].get("split_metadata_hash") for name in fold_dirs},
        "selection_rule": "first correctly predicted subject per class (sorted subject id) "
        "in the audio_text checkpoint's standalone_eval predictions; middle packed30 chunk",
        "sampling_rate": sampling_rate,
        "token_ids": ids,
        "attention": attention_meta,
        "attention_summary": summary_table,
        "ablation": ablation_rows,
        "acoustic": acoustic_rows,
        "prosody": [{key: value for key, value in row.items()} for row in prosody_rows],
        "torch": torch.__version__,
        "elapsed_s": round(time.time() - started, 1),
    }
    (out / "probe_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": "ok", "output_dir": str(out), "elapsed_s": report["elapsed_s"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
