from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import soundfile as sf

from src.data.split_utils import assign_stratified_group_folds, deterministic_inner_split
from src.utils import label_text_from_int


D3TEC_RESPONSE_COUNT = 27
D3TEC_DEFAULT_SEGMENT_SECONDS = 30.0


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_subject_id(value: Any) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    if not text.isdigit():
        raise ValueError(f"Invalid D3TEC participant ID: {value!r}")
    return f"{int(text):03d}"


def response_id(subject_id: str, prompt_id: int) -> str:
    return f"{subject_id}_p{int(prompt_id)}"


def sample_id(subject_id: str, prompt_id: int, segment_index: int) -> str:
    return f"{response_id(subject_id, prompt_id)}_s{int(segment_index)}"


def equal_duration_windows(duration: float, segment_seconds: float = D3TEC_DEFAULT_SEGMENT_SECONDS) -> list[tuple[float, float]]:
    if duration <= 0:
        raise ValueError(f"Audio duration must be positive, got {duration}.")
    if segment_seconds <= 0:
        raise ValueError("segment_seconds must be positive.")
    count = max(1, int(math.ceil(duration / segment_seconds)))
    width = duration / count
    return [
        (index * width, duration if index == count - 1 else (index + 1) * width)
        for index in range(count)
    ]


def _response_audio_path(root: Path, subject_id: str, prompt_id: int, device_dir: str) -> Path:
    suffix = "" if prompt_id == 0 else f"_{prompt_id}"
    if device_dir == "iPhoneSE2020":
        return root / device_dir / f"{subject_id}_cel{suffix}.wav"
    return root / device_dir / f"{subject_id}{suffix}.wav"


def discover_d3tec_response_windows(
    dataset_root: str | Path,
    *,
    segment_seconds: float = D3TEC_DEFAULT_SEGMENT_SECONDS,
) -> list[dict[str, Any]]:
    root = Path(dataset_root)
    metadata_rows = _load_metadata(root / "Dataset.csv")
    rows: list[dict[str, Any]] = []
    for subject_id in sorted(metadata_rows):
        for prompt_id in range(D3TEC_RESPONSE_COUNT):
            audio_path = _response_audio_path(root, subject_id, prompt_id, "SM-27")
            if not audio_path.is_file():
                raise FileNotFoundError(f"Missing canonical D3TEC SM-27 response: {audio_path}")
            info = sf.info(audio_path)
            duration = float(info.frames / info.samplerate)
            windows = equal_duration_windows(duration, segment_seconds)
            for segment_index, (start_time, end_time) in enumerate(windows):
                rows.append(
                    {
                        "dataset": "d3tec",
                        "subject_id": subject_id,
                        "response_id": response_id(subject_id, prompt_id),
                        "sample_id": sample_id(subject_id, prompt_id, segment_index),
                        "prompt_id": prompt_id,
                        "segment_index": segment_index,
                        "num_segments": len(windows),
                        "audio_path": str(audio_path),
                        "start_time": start_time,
                        "end_time": end_time,
                        "segment_duration": end_time - start_time,
                    }
                )
    return rows


def _load_metadata(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing D3TEC authoritative metadata: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        subject_id = normalize_subject_id(row["Participant_ID"])
        if subject_id in result:
            raise ValueError(f"Duplicate D3TEC participant metadata: {subject_id}")
        score = int(str(row["PHQ-9 Score"]).strip())
        result[subject_id] = {
            "subject_id": subject_id,
            "phq9_score": score,
            "label": int(score >= 10),
            "label_text": label_text_from_int(int(score >= 10)),
            "age": str(row.get("Age", "")).strip(),
            "gender": str(row.get("Gender", "")).strip(),
            "institution": str(row.get("Institution", "")).strip(),
            "residence": str(row.get("Lugar de Residencia", "")).strip(),
            "origin": str(row.get("Lugar de Procedencia", "")).strip(),
            "social_class": str(row.get("Social Class", "")).strip(),
            "medicine": str(row.get("Medicine", "")).strip(),
            "physical_condition": str(row.get("Physical Condition", "")).strip(),
            "mental_health_condition": str(row.get("Mental Health Condition", "")).strip(),
            "depression_diagnosis": str(row.get("Depression Diagnosis (level)", "")).strip(),
        }
    return result


def _audit_binary_csv(root: Path, subject_meta: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    path = root / "Dataset_binary_phq9_ge10.csv"
    if not path.is_file():
        return [{"status": "missing_optional_derived_binary_csv", "path": str(path)}]
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    audit: list[dict[str, Any]] = []
    for row in rows:
        subject_id = normalize_subject_id(row.get("Participant_ID", row.get("subject_id", "")))
        candidates = [
            row.get("label"),
            row.get("Label"),
            row.get("PHQ9_Binary"),
            row.get("PHQ-9 Binary"),
            row.get("depressed"),
            row.get("label_binary"),
        ]
        raw = next((str(value).strip() for value in candidates if value not in (None, "")), "")
        if not raw:
            # Some copies retain only the score; the independently derived label
            # remains authoritative and is still recorded in the audit.
            derived = int(int(str(row.get("PHQ-9 Score", subject_meta[subject_id]["phq9_score"]))) >= 10)
        else:
            derived = int(float(raw))
        authoritative = int(subject_meta[subject_id]["label"])
        audit.append(
            {
                "subject_id": subject_id,
                "authoritative_phq9_label": authoritative,
                "derived_csv_label": derived,
                "matches": derived == authoritative,
            }
        )
        if derived != authoritative:
            raise ValueError(
                f"D3TEC derived binary label disagrees with PHQ-9 >= 10 for {subject_id}: "
                f"{derived} != {authoritative}"
            )
    return audit


def _load_transcripts(path: Path, *, id_kind: str) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing D3TEC {id_kind} transcript JSONL: {path}")
    result: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if id_kind == "response":
                subject = normalize_subject_id(row.get("subject_id") or Path(str(row["audio_path"])).stem.split("_")[0])
                prompt = int(row.get("prompt_id", 0 if "_" not in Path(str(row["audio_path"])).stem else Path(str(row["audio_path"])).stem.rsplit("_", 1)[1]))
                stable_id = response_id(subject, prompt)
            else:
                stable_id = str(row.get("sample_id", "")).strip()
            if not stable_id:
                raise ValueError(f"Missing stable ID in {path}:{line_number}")
            if stable_id in result:
                raise ValueError(f"Duplicate D3TEC {id_kind} transcript ID: {stable_id}")
            transcript = str(row.get("transcript", "")).strip()
            if not transcript:
                raise ValueError(f"Empty D3TEC {id_kind} transcript: {stable_id}")
            language = str(row.get("language", "")).strip().lower()
            detected = str(row.get("asr_detected_language", "")).strip().lower()
            if language not in {"es", "spanish"} or (detected and detected not in {"es", "spanish"}):
                raise ValueError(f"Non-Spanish D3TEC {id_kind} transcript: {stable_id}")
            result[stable_id] = {**row, "transcript": transcript}
    return result


def _fold_distribution(
    folds: dict[int, dict[str, list[str]]],
    subject_meta: dict[str, dict[str, Any]],
    *,
    seed: int,
    inner_val_ratio: float,
) -> list[dict[str, Any]]:
    report: list[dict[str, Any]] = []
    labels = {subject_id: int(row["label"]) for subject_id, row in subject_meta.items()}
    for fold, payload in sorted(folds.items()):
        inner = deterministic_inner_split(
            labels,
            payload["outer_train_subject_ids"],
            seed=seed + int(fold),
            val_ratio=inner_val_ratio,
        )
        partitions = {
            "train_inner": inner["train_inner_subject_ids"],
            "val_inner": inner["val_inner_subject_ids"],
            "outer_test": payload["final_eval_subject_ids"],
        }
        item: dict[str, Any] = {"fold": int(fold), "partitions": {}}
        for name, subject_ids in partitions.items():
            item["partitions"][name] = {
                "subject_ids": sorted(subject_ids),
                "label_counts": dict(Counter(subject_meta[s]["label_text"] for s in subject_ids)),
                "gender_counts": dict(Counter(subject_meta[s]["gender"] for s in subject_ids)),
                "institution_counts": dict(Counter(subject_meta[s]["institution"] for s in subject_ids)),
            }
        report.append(item)
    return report


def build_d3tec_manifest(config: dict[str, Any], quarantine: dict[str, Any]) -> dict[str, Any]:
    del quarantine  # D3TEC is complete by contract; missing canonical inputs are fatal.
    root = Path(config["dataset_root"])
    segment_seconds = float(config.get("data", {}).get("segment_seconds", 30.0))
    partition = str(config.get("data", {}).get("segment_partition", "equal_duration"))
    if partition != "equal_duration":
        raise ValueError("D3TEC supports only data.segment_partition=equal_duration.")

    subject_meta = _load_metadata(root / "Dataset.csv")
    label_counts = Counter(int(row["label"]) for row in subject_meta.values())
    if len(subject_meta) != 62 or label_counts != Counter({0: 33, 1: 29}):
        raise ValueError(
            f"Unexpected D3TEC cohort: subjects={len(subject_meta)} labels={dict(label_counts)}; "
            "expected 62 with 33 non-depressed and 29 depressed."
        )
    binary_audit = _audit_binary_csv(root, subject_meta)
    full_path = Path(config.get("full_transcript_path") or root / "transcripts_qwen3_asr_spanish.jsonl")
    segment_path = Path(config.get("segment_transcript_path") or root / "transcripts_qwen3_asr_spanish_segments.jsonl")
    full_transcripts = _load_transcripts(full_path, id_kind="response")
    segment_transcripts = _load_transcripts(segment_path, id_kind="segment")
    windows = discover_d3tec_response_windows(root, segment_seconds=segment_seconds)

    manifest_rows: list[dict[str, Any]] = []
    join_audit_rows: list[dict[str, Any]] = []
    used_full: set[str] = set()
    used_segments: set[str] = set()
    for window in windows:
        subject_id = window["subject_id"]
        rid = window["response_id"]
        sid = window["sample_id"]
        if rid not in full_transcripts or sid not in segment_transcripts:
            raise ValueError(
                f"D3TEC transcript coverage failure for {sid}: "
                f"full={rid in full_transcripts} segment={sid in segment_transcripts}"
            )
        used_full.add(rid)
        used_segments.add(sid)
        prompt_id = int(window["prompt_id"])
        paired = _response_audio_path(root, subject_id, prompt_id, "iPhoneSE2020")
        meta = subject_meta[subject_id]
        row = {
            **window,
            "audio_paths": [window["audio_path"]],
            "transcript": segment_transcripts[sid]["transcript"],
            "segment_transcript": segment_transcripts[sid]["transcript"],
            "full_response_transcript": full_transcripts[rid]["transcript"],
            "transcript_path": str(segment_path),
            "full_transcript_path": str(full_path),
            "label": int(meta["label"]),
            "label_text": meta["label_text"],
            "score": int(meta["phq9_score"]),
            "phq9_score": int(meta["phq9_score"]),
            "language": "es",
            "device": "SM-27",
            "paired_iphone_audio_path": str(paired),
            "paired_iphone_audio_found": paired.is_file(),
            "split_original": "",
            "fold": "",
            "question_id": str(prompt_id),
            "chunk_id": int(window["segment_index"]),
            "modality_mode": "single_audio_segment_aligned_text",
            **{key: value for key, value in meta.items() if key not in {"subject_id", "label", "label_text", "phq9_score"}},
        }
        manifest_rows.append(row)
        join_audit_rows.append(
            {
                "sample_id": sid,
                "response_id": rid,
                "audio_found": Path(window["audio_path"]).is_file(),
                "segment_transcript_found": True,
                "full_transcript_found": True,
                "paired_iphone_found": paired.is_file(),
                "aligned_start_time": window["start_time"],
                "aligned_end_time": window["end_time"],
            }
        )
    extra_full = sorted(set(full_transcripts) - used_full)
    extra_segments = sorted(set(segment_transcripts) - used_segments)
    if extra_full or extra_segments:
        raise ValueError(
            f"Unexpected D3TEC transcript rows: extra_full={extra_full[:10]} "
            f"extra_segments={extra_segments[:10]}"
        )

    labels = {subject_id: int(meta["label"]) for subject_id, meta in subject_meta.items()}
    n_folds = int(config.get("split", {}).get("outer_folds", 5))
    seed = int(config.get("split", {}).get("seed", config["seed"]))
    folds = assign_stratified_group_folds(labels, n_splits=n_folds, seed=seed)
    fold_report = _fold_distribution(
        folds,
        subject_meta,
        seed=seed,
        inner_val_ratio=float(config.get("split", {}).get("inner_val_ratio", 0.2)),
    )
    response_counts = Counter(row["response_id"] for row in manifest_rows)
    chunk_audit = {
        "segment_seconds": segment_seconds,
        "segment_partition": partition,
        "subjects": len(subject_meta),
        "responses": len(response_counts),
        "segments": len(manifest_rows),
        "segments_per_response_distribution": dict(sorted(Counter(response_counts.values()).items())),
        "segments_per_subject": {
            subject_id: sum(1 for row in manifest_rows if row["subject_id"] == subject_id)
            for subject_id in sorted(subject_meta)
        },
        "max_segment_duration": max(float(row["segment_duration"]) for row in manifest_rows),
    }
    # A stable digest makes the fold/audit identity easy to compare across configs.
    fold_hash = hashlib.sha256(
        json.dumps(folds, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "manifest_rows": manifest_rows,
        "subject_rows": [subject_meta[key] for key in sorted(subject_meta)],
        "folds": folds,
        "fold_report": fold_report,
        "join_audit_rows": join_audit_rows,
        "chunk_window_audit": chunk_audit,
        "label_source_audit": binary_audit,
        "fold_hash": fold_hash,
        "transcript_paths": {
            "full_response": str(full_path),
            "segment_aligned": str(segment_path),
        },
        "source_hashes": {
            "dataset_csv_sha256": _sha256_file(root / "Dataset.csv"),
            "derived_binary_csv_sha256": (
                _sha256_file(root / "Dataset_binary_phq9_ge10.csv")
                if (root / "Dataset_binary_phq9_ge10.csv").is_file()
                else None
            ),
            "full_response_transcripts_sha256": _sha256_file(full_path),
            "segment_transcripts_sha256": _sha256_file(segment_path),
            "full_response_transcript_report_sha256": (
                _sha256_file(full_path.with_suffix(".report.json"))
                if full_path.with_suffix(".report.json").is_file()
                else None
            ),
            "segment_transcript_report_sha256": (
                _sha256_file(segment_path.with_suffix(".report.json"))
                if segment_path.with_suffix(".report.json").is_file()
                else None
            ),
        },
    }


def group_examples_by_response(examples: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for example in examples:
        grouped[str(example["response_id"])].append(example)
    return {
        key: sorted(value, key=lambda item: int(item["segment_index"]))
        for key, value in sorted(grouped.items())
    }


def _stable_seed(seed: int, value: str, epoch: int = 0) -> int:
    digest = hashlib.sha256(f"{seed}:{value}:{epoch}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def build_d3tec_training_schedule(
    examples: list[dict[str, Any]],
    *,
    policy: str,
    seed: int,
    virtual_epochs: int,
    responses_per_subject: int = D3TEC_RESPONSE_COUNT,
) -> tuple[list[list[dict[str, Any]]], dict[str, Any]]:
    if virtual_epochs <= 0:
        raise ValueError("virtual_epochs must be positive.")
    grouped = group_examples_by_response(examples)
    subject_ids = sorted({str(item["subject_id"]) for item in examples})
    expected = len(subject_ids) * responses_per_subject
    if len(grouped) != expected:
        raise ValueError(
            f"D3TEC training pool must contain {responses_per_subject} responses per subject: "
            f"found {len(grouped)} for {len(subject_ids)} subjects."
        )
    budget = expected
    epochs: list[list[dict[str, Any]]] = []
    if policy == "rotate_one_per_response":
        for epoch in range(virtual_epochs):
            selected: list[dict[str, Any]] = []
            for rid, segments in grouped.items():
                offset = _stable_seed(seed, rid) % len(segments)
                selected.append(dict(segments[(offset + epoch) % len(segments)]))
            rng = random.Random(_stable_seed(seed, "rotary_epoch", epoch))
            rng.shuffle(selected)
            epochs.append(selected)
    elif policy in {"all_segments_flat", "all_segments_response_normalized"}:
        inventory = [dict(item) for item in sorted(examples, key=lambda item: item["sample_id"])]
        flat: list[dict[str, Any]] = []
        cycle = 0
        while len(flat) < budget * virtual_epochs:
            shuffled = [dict(item) for item in inventory]
            random.Random(_stable_seed(seed, "flat_cycle", cycle)).shuffle(shuffled)
            flat.extend(shuffled)
            cycle += 1
        flat = flat[: budget * virtual_epochs]
        occurrence_counts = Counter(str(item["sample_id"]) for item in flat)
        if policy == "all_segments_response_normalized":
            response_occurrences = Counter(str(item["response_id"]) for item in flat)
            # Raw contributions sum to 1/27 per response and 1 per subject.
            # Multiplying by the global normalization factor preserves those
            # relative totals while making the actual loss weights mean one.
            global_scale = len(flat) / len(subject_ids)
            for item in flat:
                raw_weight = 1.0 / (
                    responses_per_subject * response_occurrences[str(item["response_id"])]
                )
                item["raw_loss_weight"] = raw_weight
                item["loss_weight"] = raw_weight * global_scale
        epochs = [
            flat[start : start + budget]
            for start in range(0, budget * virtual_epochs, budget)
        ]
    else:
        raise ValueError(f"Unsupported D3TEC train_chunk_policy={policy!r}.")

    flat_schedule = [item for epoch in epochs for item in epoch]
    audit = {
        "policy": policy,
        "seed": seed,
        "virtual_epochs": virtual_epochs,
        "examples_per_virtual_epoch": budget,
        "total_examples": len(flat_schedule),
        "subject_count": len(subject_ids),
        "response_count": len(grouped),
        "schedule_sample_ids": [[str(item["sample_id"]) for item in epoch] for epoch in epochs],
        "sample_occurrence_counts": dict(Counter(str(item["sample_id"]) for item in flat_schedule)),
        "mean_loss_weight": (
            sum(float(item.get("loss_weight", 1.0)) for item in flat_schedule) / len(flat_schedule)
        ),
    }
    if policy == "all_segments_response_normalized":
        raw_response_totals: dict[str, float] = defaultdict(float)
        raw_subject_totals: dict[str, float] = defaultdict(float)
        for item in flat_schedule:
            raw = float(item["raw_loss_weight"])
            raw_response_totals[str(item["response_id"])] += raw
            raw_subject_totals[str(item["subject_id"])] += raw
        audit["raw_response_weight_totals"] = dict(raw_response_totals)
        audit["raw_subject_weight_totals"] = dict(raw_subject_totals)
    return epochs, audit
