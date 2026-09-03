from __future__ import annotations

import hashlib
import json
import wave
from pathlib import Path

from scripts.build_turkish_pooled_manifest import main


def _write_sources(root: Path, condition: str, language: str, audio: Path) -> tuple[Path, Path]:
    count = 1051 if condition == "pos_only_t17" else 1170
    manifest = root / f"{condition}_{language}.jsonl"
    rows: list[dict[str, object]] = []
    for index in range(count):
        subject_number = index % 120
        subject = f"s{subject_number:03d}"
        label = int(subject_number < 83)
        transcript = f"{language}-{condition}-{subject_number}-{index}"
        row: dict[str, object] = {
            "dataset": "turkish", "dataset_variant": condition,
            "sample_id": f"{condition}-{subject}-{index}", "subject_id": subject,
            "label": label, "label_text": "Depressed" if label else "Non-depressed",
            "score": 18.0 if label else 10.0, "threshold": 17.0,
            "transcript": transcript, "audio_path": str(audio),
            "audio_paths": [str(audio)], "language": "tr" if language == "native" else "en",
        }
        if language == "english":
            row.update({
                "transcript_variant": "english",
                "translation_sha256": hashlib.sha256(transcript.encode()).hexdigest(),
            })
        rows.append(row)
    manifest.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    split = root / f"{condition}_{language}_folds.json"
    folds = {
        str(fold): {
            "final_eval_subject_ids": [f"s{i:03d}" for i in range(120) if i % 5 == fold],
            "outer_train_subject_ids": [f"s{i:03d}" for i in range(120) if i % 5 != fold],
        }
        for fold in range(5)
    }
    split.write_text(json.dumps(folds, sort_keys=True), encoding="utf-8")
    return manifest, split


def test_manifest_builder_validates_and_writes_the_pooled_contract(tmp_path: Path) -> None:
    audio = tmp_path / "one.wav"
    with wave.open(str(audio), "wb") as handle:
        handle.setnchannels(1); handle.setsampwidth(2); handle.setframerate(16000); handle.writeframes(b"\0\0" * 160)
    sources = {(condition, language): _write_sources(tmp_path, condition, language, audio)
               for condition in ("pos_only_t17", "negative_only_t17")
               for language in ("native", "english")}
    output_root = tmp_path / "pooled"
    args = [
        "--positive-native-manifest", str(sources[("pos_only_t17", "native")][0]),
        "--positive-native-split", str(sources[("pos_only_t17", "native")][1]),
        "--negative-native-manifest", str(sources[("negative_only_t17", "native")][0]),
        "--negative-native-split", str(sources[("negative_only_t17", "native")][1]),
        "--positive-english-manifest", str(sources[("pos_only_t17", "english")][0]),
        "--positive-english-split", str(sources[("pos_only_t17", "english")][1]),
        "--negative-english-manifest", str(sources[("negative_only_t17", "english")][0]),
        "--negative-english-split", str(sources[("negative_only_t17", "english")][1]),
        "--native-output-dir", str(output_root / "manifests/native"),
        "--english-output-dir", str(output_root / "manifests/english"),
        "--native-split-output-dir", str(output_root / "splits/native"),
        "--english-split-output-dir", str(output_root / "splits/english"),
        "--audit-output", str(output_root / "audit.json"),
    ]
    assert main(args) == 0
    audit = json.loads((output_root / "audit.json").read_text(encoding="utf-8"))
    assert audit["status"] == "passed"
    assert audit["pooled_row_counts"] == {"english": 2221, "native": 2221}
    assert audit["pooled_subject_counts"] == {"english": 120, "native": 120}
    assert (output_root / "manifests/native/turkish_manifest.jsonl").is_file()
    assert (output_root / "splits/english/turkish_manifest_metadata.json").is_file()
