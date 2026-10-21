#!/usr/bin/env python3
"""Does a Qwen3-Omni checkpoint listen to the speech or to the recording?

Diagnostic only; its numbers never enter a results table. For one fold it
re-runs the exact standalone evaluation (``src.evaluate.evaluate_examples``,
same test subjects, every window, likelihood scoring, subject aggregation) of
an audio-only and an audio+text checkpoint, five times each, with the audio of
every window replaced by:

* ``real`` — the window itself (must reproduce the saved standalone eval);
* ``speech_only`` — only the speech parts of the window, tiled to its length;
* ``room_tone`` — only the non-speech parts of the same recording (the window,
  or the subject's other windows when the window has too little), tiled;
* ``silence`` — zeros of the same length;
* ``other_subject`` — a window of a subject of the other class, fitted to length.

Speech and non-speech are split with an energy threshold (``librosa.effects
.split``, ``top_db`` below the window peak). Levels are never normalized, so
the recording level stays in every condition.

It also records, for one window per subject, the decision-row attention on the
speech and the non-speech audio tokens (audio tokens are mapped to time
linearly over the window).

Tables reference windows by a short hash; per-condition evaluation folders are
the usual standalone-eval artifacts and contain subject ids, like any
``standalone_eval`` directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

SCHEMA_VERSION = "audiollm.audio_recording_shortcut_probe.v1"
CONDITIONS = ("real", "speech_only", "room_tone", "silence", "other_subject")
EXTRA_REGION = "audio_nonspeech"


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested without a model)
# ---------------------------------------------------------------------------


def speech_intervals(audio: np.ndarray, top_db: float) -> np.ndarray:
    """Sample intervals louder than ``top_db`` below the peak, ``[n, 2]``."""
    import librosa

    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0 or float(np.max(np.abs(audio))) == 0.0:
        return np.zeros((0, 2), dtype=np.int64)
    return np.asarray(
        librosa.effects.split(audio, top_db=top_db, frame_length=1024, hop_length=256),
        dtype=np.int64,
    ).reshape(-1, 2)


def complement_intervals(intervals: np.ndarray, length: int) -> np.ndarray:
    gaps = []
    cursor = 0
    for start, end in intervals:
        if start > cursor:
            gaps.append((cursor, int(start)))
        cursor = max(cursor, int(end))
    if cursor < length:
        gaps.append((cursor, int(length)))
    return np.asarray(gaps, dtype=np.int64).reshape(-1, 2)


def gather(audio: np.ndarray, intervals: np.ndarray) -> np.ndarray:
    if len(intervals) == 0:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate([audio[start:end] for start, end in intervals]).astype(np.float32)


def fit_length(piece: np.ndarray, length: int) -> np.ndarray:
    """Tile or crop ``piece`` to ``length`` samples; zeros if it is empty."""
    if length <= 0:
        return np.zeros(0, dtype=np.float32)
    if piece.size == 0:
        return np.zeros(length, dtype=np.float32)
    repeats = int(np.ceil(length / piece.size))
    return np.tile(piece, repeats)[:length].astype(np.float32)


def token_speech_mask(n_tokens: int, n_samples: int, intervals: np.ndarray) -> np.ndarray:
    """True for audio tokens whose time span is at least half speech."""
    speech = np.zeros(n_samples, dtype=bool)
    for start, end in intervals:
        speech[start:end] = True
    edges = np.linspace(0, n_samples, n_tokens + 1).round().astype(int)
    return np.asarray(
        [speech[edges[i] : edges[i + 1]].mean() >= 0.5 if edges[i + 1] > edges[i] else False for i in range(n_tokens)]
    )


def pair_other_class(subject_labels: dict[str, int]) -> dict[str, str]:
    """Pair each subject with one of the other class (sorted ids, cyclic)."""
    by_class = {label: sorted(s for s, value in subject_labels.items() if value == label) for label in (0, 1)}
    if not by_class[0] or not by_class[1]:
        raise ValueError("both classes are needed to pair subjects")
    pairs = {}
    for label in (0, 1):
        others = by_class[1 - label]
        for index, subject in enumerate(by_class[label]):
            pairs[subject] = others[index % len(others)]
    return pairs


def binary_metrics(labels: list[int], predictions: list[int]) -> dict[str, float]:
    """Macro-F1, positive-F1 and UAR; any prediction other than 0/1 counts as wrong."""

    def f1(cls: int) -> float:
        tp = sum(y == cls and p == cls for y, p in zip(labels, predictions))
        fp = sum(y != cls and p == cls for y, p in zip(labels, predictions))
        fn = sum(y == cls and p != cls for y, p in zip(labels, predictions))
        return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0

    recalls = []
    for cls in (0, 1):
        total = labels.count(cls)
        hits = sum(y == cls and p == cls for y, p in zip(labels, predictions))
        recalls.append(hits / total if total else 0.0)
    return {
        "macro_f1": (f1(0) + f1(1)) / 2,
        "positive_f1": f1(1),
        "uar": sum(recalls) / 2,
        "predicted_depressed": sum(p == 1 for p in predictions),
        "n": len(labels),
    }


def dbfs(audio: np.ndarray) -> float | None:
    audio = np.asarray(audio, dtype=np.float64)
    if audio.size == 0:
        return None
    rms = float(np.sqrt(np.mean(audio**2)))
    return round(20.0 * np.log10(rms), 1) if rms > 0 else None


# ---------------------------------------------------------------------------
# Condition audio
# ---------------------------------------------------------------------------


class ConditionAudio:
    """Replacement for ``src.evaluate._load_example_audio`` under one condition."""

    def __init__(self, original_loader, examples, sampling_rate: int, top_db: float, min_room_tone_s: float):
        from scripts.probe_audio_attention import example_ref

        self.original = original_loader
        self.sampling_rate = int(sampling_rate)
        self.top_db = float(top_db)
        self.min_room_tone = int(min_room_tone_s * sampling_rate)
        self.example_ref = example_ref
        self.by_sample = {str(item["sample_id"]): item for item in examples}
        self.by_subject: dict[str, list[str]] = {}
        for item in sorted(examples, key=lambda row: str(row["sample_id"])):
            self.by_subject.setdefault(str(item["subject_id"]), []).append(str(item["sample_id"]))
        self.pairs = pair_other_class({str(item["subject_id"]): int(item["label"]) for item in examples})
        self.cache: dict[str, list[np.ndarray]] = {}
        self.pool: dict[str, np.ndarray] = {}
        self.condition = "real"
        self.stats: dict[tuple[str, str], dict[str, Any]] = {}

    def real(self, sample_id: str) -> list[np.ndarray]:
        if sample_id not in self.cache:
            arrays = self.original(self.by_sample[sample_id], self.sampling_rate, False)
            self.cache[sample_id] = [np.asarray(array, dtype=np.float32) for array in arrays]
        return self.cache[sample_id]

    def subject_room_tone(self, subject_id: str) -> np.ndarray:
        if subject_id not in self.pool:
            pieces = []
            for sample_id in self.by_subject[subject_id]:
                for array in self.real(sample_id):
                    pieces.append(gather(array, complement_intervals(speech_intervals(array, self.top_db), len(array))))
            self.pool[subject_id] = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
        return self.pool[subject_id]

    def __call__(self, example, sampling_rate, silence_audio):
        if int(sampling_rate) != self.sampling_rate:
            raise RuntimeError(f"unexpected sampling rate {sampling_rate}")
        sample_id = str(example["sample_id"])
        subject_id = str(example["subject_id"])
        arrays = self.real(sample_id)
        out = []
        for index, array in enumerate(arrays):
            length = len(array)
            stat: dict[str, Any] = {"samples": length}
            if self.condition == "real":
                result = array
            elif self.condition == "silence":
                result = np.zeros(length, dtype=np.float32)
            elif self.condition in ("speech_only", "room_tone"):
                intervals = speech_intervals(array, self.top_db)
                speech = gather(array, intervals)
                stat["speech_s"] = round(speech.size / self.sampling_rate, 3)
                if self.condition == "speech_only":
                    result = fit_length(speech if speech.size else array, length)
                else:
                    tone = gather(array, complement_intervals(intervals, length))
                    stat["window_room_tone_s"] = round(tone.size / self.sampling_rate, 3)
                    stat["used_subject_pool"] = tone.size < self.min_room_tone
                    if tone.size < self.min_room_tone:
                        tone = self.subject_room_tone(subject_id)
                    stat["room_tone_source_s"] = round(tone.size / self.sampling_rate, 3)
                    stat["room_tone_dbfs"] = dbfs(tone)
                    result = fit_length(tone, length)
            elif self.condition == "other_subject":
                partner = self.pairs[subject_id]
                own_index = self.by_subject[subject_id].index(sample_id)
                partner_windows = self.by_subject[partner]
                partner_array = self.real(partner_windows[own_index % len(partner_windows)])[index]
                result = fit_length(partner_array, length)
            else:
                raise ValueError(f"unknown condition {self.condition}")
            stat["dbfs"] = dbfs(result)
            self.stats[(self.condition, sample_id)] = stat
            out.append(result)
        return out


# ---------------------------------------------------------------------------
# Speech vs non-speech attention
# ---------------------------------------------------------------------------


def speech_attention(model, processor, example, arrays, sampling_rate, device, ids, top_db, keep_weights=False):
    import torch

    from scripts.probe_audio_attention import (
        REGION_INDEX,
        REGIONS,
        AttentionRecorder,
        _processor_inputs,
        _tokenizer_offsets,
        assign_regions,
    )

    prompt = example["prompt_text"]
    inputs = _processor_inputs(processor, prompt, arrays, sampling_rate, device)
    input_ids = inputs["input_ids"][0].tolist()
    prompt_len = len(input_ids)
    offsets, expanded = _tokenizer_offsets(processor, prompt, input_ids, ids["audio"])
    regions, method = assign_regions(
        input_ids=input_ids,
        offsets=offsets,
        text=expanded,
        prompt_len=prompt_len,
        transcript=str(example.get("transcript") or ""),
        audio_token_id=ids["audio"],
        audio_marker_ids={ids["audio_start"], ids["audio_end"]},
    )
    audio_positions = np.flatnonzero(np.asarray(input_ids) == ids["audio"])
    if len(arrays) != 1:
        raise RuntimeError("speech attention expects one audio array per prompt")
    mask = token_speech_mask(len(audio_positions), len(arrays[0]), speech_intervals(arrays[0], top_db))
    names = list(REGIONS) + [EXTRA_REGION]
    regions = regions.copy()
    regions[audio_positions[~mask]] = len(REGIONS)
    query = prompt_len - 1
    recorder = AttentionRecorder(model, [query], regions, n_regions=len(names), keep_weights=keep_weights)
    try:
        with torch.inference_mode():
            model(**inputs, use_cache=False)
    finally:
        recorder.remove()
    layers = sorted(recorder.layers, key=lambda item: item["layer_idx"])
    mass = np.stack([layer["mass"][:, 0, :].mean(axis=0) for layer in layers])  # [layers, regions]
    visible = regions[: query + 1]
    uniform = np.bincount(visible, minlength=len(names)) / float(len(visible))
    return {
        "names": names,
        "mass": mass,
        "uniform": uniform,
        "audio_tokens": int(len(audio_positions)),
        "speech_tokens": int(mask.sum()),
        "region_method": method,
        "max_rel_err": max(layer["max_rel_err"] for layer in layers),
        "speech_index": REGION_INDEX["audio"],
        "nonspeech_index": len(REGIONS),
        "regions": regions,
        "token_weights": np.stack([layer["weights"][0] for layer in layers]) if keep_weights else None,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _read_subject_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _prediction(value: str) -> int:
    value = str(value).strip()
    if value in ("0", "1"):
        return int(value)
    return {"Depressed": 1, "Non-depressed": 0}.get(value, -1)


def _margin(row: dict[str, str]) -> float:
    """Subject margin; some aggregations write only dep_score and non_score."""
    if row.get("score_margin") not in (None, ""):
        return float(row["score_margin"])
    return float(row["dep_score"]) - float(row["non_score"])


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--audio-text-fold-dir", required=True, type=Path)
    parser.add_argument("--audio-only-fold-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--top-db", type=float, default=30.0)
    parser.add_argument("--min-room-tone-s", type=float, default=1.0)
    parser.add_argument("--attention-windows-per-subject", type=int, default=1)
    parser.add_argument(
        "--subjects-per-class",
        type=int,
        default=0,
        help="keep only the first N evaluated subjects of each class (sorted id); 0 keeps all",
    )
    parser.add_argument(
        "--transcript-scope",
        default="",
        help="override data.audio_text_transcript_scope of the audio+text config "
        "(e.g. segment_aligned for Androids); empty keeps the saved value",
    )
    parser.add_argument(
        "--zero-shot",
        action="store_true",
        help="disable the LoRA adapters (base Thinker) and keep each checkpoint's prompt",
    )
    parser.add_argument(
        "--save-token-attention",
        action="store_true",
        help="save token-level attention maps (always on with --attention-only)",
    )
    parser.add_argument(
        "--attention-only",
        action="store_true",
        help="skip the condition evaluations; only record attention and save token-level maps",
    )
    args = parser.parse_args(argv)
    conditions = [item.strip() for item in args.conditions.split(",") if item.strip()]
    unknown = sorted(set(conditions) - set(CONDITIONS))
    if args.attention_only:
        conditions = []
    elif unknown or conditions[0] != "real":
        raise SystemExit(f"conditions must start with 'real' and come from {CONDITIONS}; got {conditions}")

    import torch

    import src.evaluate as evaluate_module
    from scripts.probe_audio_attention import (
        AUDIO_END,
        AUDIO_PAD,
        AUDIO_START,
        _load_saved_config,
        example_ref,
    )
    from src.data.runtime import build_examples, filter_rows_by_subjects, load_manifest_rows
    from src.model.qwen3omni_lora import load_additional_audio_modules
    from src.model.runtime import (
        load_model_for_inference,
        load_processor,
        prepare_backend_examples,
        resolve_processor_sampling_rate,
    )
    from src.utils import sha256_jsonl_rows

    started = time.time()
    out = args.output_dir
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"output dir {out} is not empty; refusing to overwrite")
    out.mkdir(parents=True, exist_ok=True)

    fold_dirs = {"audio_only": args.audio_only_fold_dir, "audio_text": args.audio_text_fold_dir}
    saved, configs, reference = {}, {}, {}
    for name, fold_dir in fold_dirs.items():
        saved[name], configs[name] = _load_saved_config(fold_dir)
        reference[name] = _read_subject_rows(
            fold_dir / "best_model" / "standalone_eval" / "predictions_subject_level.csv"
        )
    saved_scope = configs["audio_text"].get("data", {}).get("audio_text_transcript_scope")
    if args.transcript_scope:
        configs["audio_text"].setdefault("data", {})["audio_text_transcript_scope"] = args.transcript_scope
    model_paths = {str(saved[name].get("resolved_model_name_or_path")) for name in fold_dirs}
    if len(model_paths) != 1:
        raise RuntimeError(f"the two checkpoints use different base models: {model_paths}")
    model_path = model_paths.pop()
    manifest_path = Path(saved["audio_only"]["manifest_path"])
    manifest_rows = load_manifest_rows(manifest_path)
    manifest_hash = sha256_jsonl_rows(manifest_rows)
    for name in fold_dirs:
        if saved[name].get("manifest_hash") != manifest_hash:
            raise RuntimeError(f"manifest hash mismatch for {name}: {manifest_hash}")
    subjects = {name: sorted(row["subject_id"] for row in reference[name]) for name in fold_dirs}
    if subjects["audio_only"] != subjects["audio_text"]:
        raise RuntimeError("the two checkpoints were evaluated on different subjects")
    eval_subjects = subjects["audio_only"]
    if args.subjects_per_class > 0:
        labels = {row["subject_id"]: int(row["label"]) for row in reference["audio_only"]}
        eval_subjects = [
            subject
            for cls in (1, 0)
            for subject in sorted(s for s in eval_subjects if labels[s] == cls)[: args.subjects_per_class]
        ]
        reference = {
            name: [row for row in rows if row["subject_id"] in set(eval_subjects)]
            for name, rows in reference.items()
        }

    processor = load_processor(fold_dirs["audio_only"] / "best_model", configs["audio_only"])
    sampling_rate = int(resolve_processor_sampling_rate(processor))
    examples = {}
    for name in fold_dirs:
        built = build_examples(
            filter_rows_by_subjects(manifest_rows, eval_subjects),
            configs[name],
            partition_name="final_eval",
            truncation_log_path=None,
        )
        examples[name] = prepare_backend_examples(built, configs[name], processor)
    if sorted(map(str, (e["sample_id"] for e in examples["audio_only"]))) != sorted(
        map(str, (e["sample_id"] for e in examples["audio_text"]))
    ):
        raise RuntimeError("audio-only and audio+text examples cover different windows")

    model = load_model_for_inference(model_path, fold_dirs["audio_only"] / "best_model", configs["audio_only"])
    extra = load_additional_audio_modules(model.get_base_model(), fold_dirs["audio_text"] / "best_model")
    if extra.get("enabled") or extra.get("train_projector"):
        raise RuntimeError("audio_text checkpoint carries extra audio modules; probe does not support it")
    model.load_adapter(str(fold_dirs["audio_text"] / "best_model"), adapter_name="audio_text")
    adapters = {"audio_only": "default", "audio_text": "audio_text"}
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = getattr(processor, "tokenizer", processor)
    ids = {
        "audio": int(tokenizer.convert_tokens_to_ids(AUDIO_PAD)),
        "audio_start": int(tokenizer.convert_tokens_to_ids(AUDIO_START)),
        "audio_end": int(tokenizer.convert_tokens_to_ids(AUDIO_END)),
    }

    original_loader = evaluate_module._load_example_audio
    provider = ConditionAudio(
        original_loader, examples["audio_only"], sampling_rate, args.top_db, args.min_room_tone_s
    )
    metric_rows: list[dict[str, Any]] = []
    subject_rows: list[dict[str, Any]] = []
    attention_rows: list[dict[str, Any]] = []
    attention_meta: list[dict[str, Any]] = []
    evaluate_module._load_example_audio = provider
    try:
        for name in ("audio_only", "audio_text"):
            model.set_adapter(adapters[name])
            if args.zero_shot:
                # Base Thinker: the adapter stays loaded but contributes nothing.
                model.base_model.disable_adapter_layers()
            margins_real: dict[str, float] = {}
            preds_real: dict[str, int] = {}
            for condition in conditions:
                provider.condition = condition
                condition_dir = out / "eval" / name / condition
                evaluate_module.evaluate_examples(
                    model,
                    processor,
                    examples[name],
                    configs[name],
                    condition_dir,
                    checkpoint_name=f"{name}_{condition}",
                )
                rows = _read_subject_rows(condition_dir / "predictions_subject_level.csv")
                labels = [int(row["label"]) for row in rows]
                predictions = [_prediction(row["prediction"]) for row in rows]
                margins = {row["subject_id"]: _margin(row) for row in rows}
                current = {row["subject_id"]: _prediction(row["prediction"]) for row in rows}
                if condition == "real":
                    margins_real, preds_real = margins, current
                    saved_preds = {row["subject_id"]: _prediction(row["prediction"]) for row in reference[name]}
                    reproduced = sum(saved_preds[s] == current[s] for s in current)
                else:
                    reproduced = None
                metrics = binary_metrics(labels, predictions)
                metric_rows.append(
                    {
                        "model": name,
                        "condition": condition,
                        **{key: round(value, 4) if isinstance(value, float) else value for key, value in metrics.items()},
                        "flipped_vs_real": sum(current[s] != preds_real[s] for s in current),
                        "mean_margin": round(float(np.mean(list(margins.values()))), 4),
                        "mean_abs_margin_change_vs_real": round(
                            float(np.mean([abs(margins[s] - margins_real[s]) for s in margins])), 4
                        ),
                        "real_matches_saved_eval": reproduced,
                    }
                )
                for row in rows:
                    subject_rows.append(
                        {
                            "model": name,
                            "condition": condition,
                            "subject_ref": example_ref(row["subject_id"]),
                            "label": int(row["label"]),
                            "prediction": _prediction(row["prediction"]),
                            "score_margin": round(_margin(row), 5),
                        }
                    )
                print(json.dumps({"done": [name, condition], **metric_rows[-1], "elapsed_s": round(time.time() - started)}), flush=True)

            # Speech vs non-speech attention on the real audio.
            provider.condition = "real"
            for subject in eval_subjects:
                windows = [e for e in examples[name] if str(e["subject_id"]) == subject]
                windows.sort(key=lambda e: str(e["sample_id"]))
                scored = []
                for example in windows:
                    array = provider.real(str(example["sample_id"]))[0]
                    speech = gather(array, speech_intervals(array, args.top_db)).size / max(1, len(array))
                    scored.append((abs(speech - 0.6), str(example["sample_id"]), example, speech))
                scored.sort(key=lambda item: (item[0], item[1]))
                for _, _, example, speech_fraction in scored[: args.attention_windows_per_subject]:
                    arrays = provider.real(str(example["sample_id"]))
                    result = speech_attention(
                        model, processor, example, arrays, sampling_rate, device, ids, args.top_db,
                        keep_weights=args.attention_only or args.save_token_attention,
                    )
                    ref = example_ref(example["sample_id"])
                    if result["token_weights"] is not None:
                        (out / "token_attention").mkdir(exist_ok=True)
                        np.savez_compressed(
                            out / "token_attention" / f"{name}_{int(example['label'])}_{ref}.npz",
                            weights=result["token_weights"],
                            regions=result["regions"],
                            region_names=np.asarray(result["names"]),
                            label=int(example["label"]),
                        )
                    attention_meta.append(
                        {
                            "model": name,
                            "window_ref": ref,
                            "label": int(example["label"]),
                            "speech_fraction": round(float(speech_fraction), 3),
                            "audio_tokens": result["audio_tokens"],
                            "speech_tokens": result["speech_tokens"],
                            "region_method": result["region_method"],
                            "self_check_max_rel_err": round(float(result["max_rel_err"]), 5),
                        }
                    )
                    for layer in range(result["mass"].shape[0]):
                        for index, region in enumerate(result["names"]):
                            attention_rows.append(
                                {
                                    "model": name,
                                    "window_ref": ref,
                                    "label": int(example["label"]),
                                    "layer": layer,
                                    "region": "audio_speech" if region == "audio" else region,
                                    "mass": round(float(result["mass"][layer, index]), 6),
                                    "uniform_share": round(float(result["uniform"][index]), 6),
                                }
                            )
            print(json.dumps({"done": [name, "attention"], "elapsed_s": round(time.time() - started)}), flush=True)
    finally:
        evaluate_module._load_example_audio = original_loader

    _write_csv(metric_rows, out / "condition_metrics.csv")
    _write_csv(subject_rows, out / "condition_subjects.csv")
    _write_csv(attention_rows, out / "speech_attention_layers.csv")
    _write_csv(attention_meta, out / "speech_attention_windows.csv")
    stat_rows = [
        {"condition": condition, "window_ref": example_ref(sample_id), **stat}
        for (condition, sample_id), stat in sorted(provider.stats.items())
    ]
    _write_csv(stat_rows, out / "condition_audio_stats.csv")

    summary = []
    for name in ("audio_only", "audio_text"):
        rows = [row for row in attention_rows if row["model"] == name]
        if not rows:
            continue
        entry = {"model": name}
        for region in ("audio_speech", EXTRA_REGION, "transcript"):
            values = [row["mass"] for row in rows if row["region"] == region]
            shares = [row["uniform_share"] for row in rows if row["region"] == region]
            if values:
                entry[f"{region}_mass"] = round(float(np.mean(values)), 4)
                entry[f"{region}_uniform_share"] = round(float(np.mean(shares)), 4)
        summary.append(entry)
    report = {
        "schema_version": SCHEMA_VERSION,
        "model_path": model_path,
        "fold_dirs": {name: str(path) for name, path in fold_dirs.items()},
        "run_names": {name: saved[name].get("run_name") for name in fold_dirs},
        "manifest_path": str(manifest_path),
        "manifest_hash": manifest_hash,
        "split_metadata_hash": {name: saved[name].get("split_metadata_hash") for name in fold_dirs},
        "evaluation": "src.evaluate.evaluate_examples with the saved run_config; audio loader replaced per condition",
        "subjects": len(eval_subjects),
        "windows": len(examples["audio_only"]),
        "top_db": args.top_db,
        "min_room_tone_s": args.min_room_tone_s,
        "other_subject_pairing": "each subject paired with a subject of the other class (sorted ids, cyclic); "
        "window i of the subject gets window i mod n of the partner",
        "conditions": conditions,
        "metrics": metric_rows,
        "speech_attention_summary": summary,
        "speech_attention_windows": attention_meta,
        "torch": torch.__version__,
        "zero_shot": bool(args.zero_shot),
        "transcript_scope": {"saved": saved_scope, "used": configs["audio_text"].get("data", {}).get("audio_text_transcript_scope")},
        "subjects_per_class": args.subjects_per_class,
        "elapsed_s": round(time.time() - started, 1),
    }
    (out / "probe_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": "ok", "output_dir": str(out), "elapsed_s": report["elapsed_s"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
