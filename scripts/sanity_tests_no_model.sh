#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

run_python() {
  if [[ -n "${CONDA_ENV:-}" ]]; then
    conda run -n "$CONDA_ENV" "$PYTHON_BIN" "$@"
  else
    "$PYTHON_BIN" "$@"
  fi
}

echo "[1/5] Building manifests and split metadata"
"$PROJECT_ROOT/scripts/validate_manifests.sh"

echo "[2/5] Verifying expected audit/split files and EDAIC invariants"
run_python - <<'PY'
import ast
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from src.evaluate import _resolve_final_eval_subject_ids
from src.hpo import validate_hpo_split_mode
from src.train import _resolve_outer_partitions, _resolve_training_subject_splits
from src.utils import load_yaml_with_overrides, resolve_metadata_paths

root = Path("/home/emre/Projects/AudioLLM/LLM-Depression")
required = [
    root / "outputs/manifests/daic_manifest.jsonl",
    root / "outputs/manifests/edaic_manifest.jsonl",
    root / "outputs/manifests/cmdc_manifest.jsonl",
    root / "outputs/manifests/eatd_manifest.jsonl",
    root / "outputs/splits/daic_join_audit.csv",
    root / "outputs/splits/daic_folds.json",
    root / "outputs/splits/daic_fold_report.json",
    root / "outputs/splits/daic_subject_partitions.json",
    root / "outputs/splits/daic_manifest_metadata.json",
    root / "outputs/splits/edaic_join_audit.csv",
    root / "outputs/splits/edaic_folds.json",
    root / "outputs/splits/edaic_fold_report.json",
    root / "outputs/splits/edaic_subject_partitions.json",
    root / "outputs/splits/edaic_manifest_metadata.json",
    root / "outputs/splits/cmdc_fold_report.json",
    root / "outputs/splits/eatd_folds.json",
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit(f"Missing expected outputs: {missing}")

def assert_fold_coverage(label, folds, expected_subject_ids, forbidden_subject_ids):
    expected_subject_set = set(expected_subject_ids)
    forbidden_subject_set = set(forbidden_subject_ids)
    heldout_coverage = set()
    for fold_key, payload in sorted(folds.items(), key=lambda item: int(item[0])):
        heldout_ids = set(payload["final_eval_subject_ids"])
        outer_train_ids = set(payload["outer_train_subject_ids"])
        if heldout_ids & forbidden_subject_set:
            raise SystemExit(f"{label} fold {fold_key} unexpectedly includes forbidden subjects.")
        if heldout_ids & heldout_coverage:
            raise SystemExit(f"{label} held-out fold overlap detected for fold {fold_key}.")
        if heldout_ids & outer_train_ids:
            raise SystemExit(f"{label} fold {fold_key} overlaps outer train and held-out subjects.")
        if heldout_ids | outer_train_ids != expected_subject_set:
            raise SystemExit(f"{label} fold {fold_key} does not partition the expected development pool.")
        heldout_coverage.update(heldout_ids)
    if heldout_coverage != expected_subject_set:
        raise SystemExit(f"{label} held-out fold coverage mismatch.")

daic_config_path = root / "configs/archive/daic/daic_audio_text.yaml"
daic_config = load_yaml_with_overrides(daic_config_path, [])
daic_base = Path(str(daic_config["dataset_root"]))
daic_summary_specs = [
    ("train", daic_base / "train_preprocessing_summary.csv"),
    ("val", daic_base / "dev_preprocessing_summary.csv"),
    ("test", daic_base / "test_preprocessing_summary.csv"),
]
expected_daic_sample_partition_counts = {}
expected_daic_subject_partition_counts = {}
expected_daic_transcript_paths = set()
for partition, summary_path in daic_summary_specs:
    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(line for line in handle if line.strip()))
    expected_daic_subject_partition_counts[partition] = len(rows)
    expected_daic_sample_partition_counts[partition] = 0
    expected_daic_transcript_paths.add(str(summary_path))
    for row in rows:
        segment_files = ast.literal_eval(row["segment_files"])
        if len(segment_files) != int(row["num_segments"]):
            raise SystemExit(
                f"DAIC num_segments mismatch in {summary_path.name} for participant "
                f"{row.get('Participant_ID') or row.get('participant_id')}"
            )
        expected_daic_sample_partition_counts[partition] += len(segment_files)

daic_metadata = resolve_metadata_paths(
    json.loads((root / "outputs/splits/daic_manifest_metadata.json").read_text(encoding="utf-8"))
)
if daic_metadata.get("split_source") != "preprocessed_full_transcript_all_splits":
    raise SystemExit(f"Unexpected DAIC split source: {daic_metadata.get('split_source')}")

daic_manifest_rows = [
    json.loads(line)
    for line in (root / "outputs/manifests/daic_manifest.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
if len(daic_manifest_rows) != sum(expected_daic_sample_partition_counts.values()):
    raise SystemExit(f"Unexpected DAIC sample count: {len(daic_manifest_rows)}")

daic_sample_partition_counts = Counter(row["split_original"] for row in daic_manifest_rows)
if dict(daic_sample_partition_counts) != expected_daic_sample_partition_counts:
    raise SystemExit(
        f"Unexpected DAIC sample partition counts: {dict(daic_sample_partition_counts)} "
        f"expected={expected_daic_sample_partition_counts}"
    )

daic_subject_partition_rows = json.loads((root / "outputs/splits/daic_subject_partitions.json").read_text(encoding="utf-8"))
daic_subject_partition_counts = Counter(row["partition"] for row in daic_subject_partition_rows)
if dict(daic_subject_partition_counts) != expected_daic_subject_partition_counts:
    raise SystemExit(
        f"Unexpected DAIC subject partition counts: {dict(daic_subject_partition_counts)} "
        f"expected={expected_daic_subject_partition_counts}"
    )

daic_partitions_by_subject = defaultdict(set)
for row in daic_subject_partition_rows:
    daic_partitions_by_subject[row["subject_id"]].add(row["partition"])
daic_overlaps = {subject_id: sorted(parts) for subject_id, parts in daic_partitions_by_subject.items() if len(parts) > 1}
if daic_overlaps:
    raise SystemExit(f"DAIC subject partition overlap detected: {list(daic_overlaps.items())[:5]}")
daic_dev_subject_ids = sorted(
    [row["subject_id"] for row in daic_subject_partition_rows if row["partition"] in {"train", "val"}]
)
daic_test_subject_ids = sorted([row["subject_id"] for row in daic_subject_partition_rows if row["partition"] == "test"])
daic_folds = json.loads((root / "outputs/splits/daic_folds.json").read_text(encoding="utf-8"))
daic_fold_report = json.loads((root / "outputs/splits/daic_fold_report.json").read_text(encoding="utf-8"))
if len(daic_fold_report) != 5:
    raise SystemExit(f"Unexpected DAIC fold report length: {len(daic_fold_report)}")
assert_fold_coverage("DAIC", daic_folds, daic_dev_subject_ids, daic_test_subject_ids)

with (root / "outputs/splits/daic_join_audit.csv").open("r", encoding="utf-8", newline="") as handle:
    daic_join_audit_rows = list(csv.DictReader(handle))
if len(daic_join_audit_rows) != len(daic_manifest_rows):
    raise SystemExit(f"Unexpected DAIC join audit row count: {len(daic_join_audit_rows)}")
daic_missing_join_rows = [
    row["sample_id"]
    for row in daic_join_audit_rows
    if row["audio_found"] != "True" or row["transcript_found"] != "True"
]
if daic_missing_join_rows:
    raise SystemExit(f"DAIC join audit has missing pairs: {daic_missing_join_rows[:10]}")

daic_transcripts_by_subject = defaultdict(set)
daic_transcript_paths = set()
for row in daic_manifest_rows:
    daic_transcripts_by_subject[row["subject_id"]].add(row["transcript"])
    daic_transcript_paths.add(row["transcript_path"])
    if not row["transcript"].strip():
        raise SystemExit(f"DAIC manifest has empty transcript for sample_id={row['sample_id']}")
    if "whisper" in row["transcript_path"].lower():
        raise SystemExit(f"DAIC manifest unexpectedly used Whisper transcript cache: {row['transcript_path']}")
bad_daic_transcript_subjects = [
    subject_id for subject_id, transcripts in daic_transcripts_by_subject.items() if len(transcripts) != 1
]
if bad_daic_transcript_subjects:
    raise SystemExit(f"DAIC subjects have inconsistent repeated full transcripts: {bad_daic_transcript_subjects[:10]}")
if daic_transcript_paths != expected_daic_transcript_paths:
    raise SystemExit(
        f"Unexpected DAIC transcript paths: {sorted(daic_transcript_paths)} "
        f"expected={sorted(expected_daic_transcript_paths)}"
    )

daic_training_plan = _resolve_training_subject_splits(
    daic_config,
    daic_metadata,
    {row["subject_id"]: int(row["label"]) for row in daic_manifest_rows},
    0,
)
if daic_training_plan["uses_inner_split"]:
    raise SystemExit("DAIC main config unexpectedly used deterministic inner split.")
if len(daic_training_plan["train_subject_ids"]) != expected_daic_subject_partition_counts["train"]:
    raise SystemExit(
        f"Unexpected DAIC train subject count from train.py logic: {len(daic_training_plan['train_subject_ids'])}"
    )
if len(daic_training_plan["selection_subject_ids"]) != expected_daic_subject_partition_counts["val"]:
    raise SystemExit(
        f"Unexpected DAIC selection subject count from train.py logic: {len(daic_training_plan['selection_subject_ids'])}"
    )
if len(daic_training_plan["final_eval_subject_ids"]) != expected_daic_subject_partition_counts["test"]:
    raise SystemExit(
        f"Unexpected DAIC final-eval subject count from train.py logic: {len(daic_training_plan['final_eval_subject_ids'])}"
    )
daic_final_eval_subject_ids = _resolve_final_eval_subject_ids(daic_config, daic_metadata, 0)
if len(daic_final_eval_subject_ids) != expected_daic_subject_partition_counts["test"]:
    raise SystemExit(
        f"Unexpected DAIC final eval partition resolution from evaluate.py: {len(daic_final_eval_subject_ids)}"
    )

daic_cv_config = load_yaml_with_overrides(daic_config_path, ["split.mode=cv"])
daic_cv_training_plan = _resolve_training_subject_splits(
    daic_cv_config,
    daic_metadata,
    {row["subject_id"]: int(row["label"]) for row in daic_manifest_rows},
    0,
)
expected_daic_cv_holdout_ids = sorted(daic_folds["0"]["final_eval_subject_ids"])
if not daic_cv_training_plan["uses_inner_split"]:
    raise SystemExit("DAIC CV mode unexpectedly skipped the deterministic inner split.")
if daic_cv_training_plan["final_eval_subject_ids"] != expected_daic_cv_holdout_ids:
    raise SystemExit("DAIC CV mode final-eval subjects do not match fold_0 held-out ids.")
if set(daic_cv_training_plan["train_subject_ids"]) & set(expected_daic_cv_holdout_ids):
    raise SystemExit("DAIC CV mode leaked held-out fold subjects into train_inner.")
if set(daic_cv_training_plan["selection_subject_ids"]) & set(expected_daic_cv_holdout_ids):
    raise SystemExit("DAIC CV mode leaked held-out fold subjects into val_inner.")
if _resolve_final_eval_subject_ids(daic_cv_config, daic_metadata, 0) != expected_daic_cv_holdout_ids:
    raise SystemExit("DAIC CV mode evaluate.py resolution does not match fold_0 held-out ids.")

daic_full_train_config = load_yaml_with_overrides(daic_config_path, ["split.mode=full_train"])
daic_full_train_plan = _resolve_training_subject_splits(
    daic_full_train_config,
    daic_metadata,
    {row["subject_id"]: int(row["label"]) for row in daic_manifest_rows},
    0,
)
if daic_full_train_plan["selection_subject_ids"]:
    raise SystemExit("DAIC full_train mode unexpectedly created selection subjects.")
if len(daic_full_train_plan["train_subject_ids"]) != len(daic_dev_subject_ids):
    raise SystemExit("DAIC full_train mode did not train on the full pooled train+val subject pool.")
if daic_full_train_plan["final_eval_subject_ids"] != daic_test_subject_ids:
    raise SystemExit("DAIC full_train mode did not keep official test as the final eval split.")
if _resolve_final_eval_subject_ids(daic_full_train_config, daic_metadata, 0) != daic_test_subject_ids:
    raise SystemExit("DAIC full_train evaluate.py resolution did not keep official test.")

edaic_manifest_rows = [
    json.loads(line)
    for line in (root / "outputs/manifests/edaic_manifest.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
if len(edaic_manifest_rows) != 3080:
    raise SystemExit(f"Unexpected EDAIC sample count: {len(edaic_manifest_rows)}")

edaic_subject_ids = {row["subject_id"] for row in edaic_manifest_rows}
if len(edaic_subject_ids) != 275:
    raise SystemExit(f"Unexpected EDAIC subject count: {len(edaic_subject_ids)}")

sample_partition_counts = Counter(row["split_original"] for row in edaic_manifest_rows)
expected_sample_partition_counts = {"train": 1815, "val": 620, "test": 645}
if dict(sample_partition_counts) != expected_sample_partition_counts:
    raise SystemExit(
        f"Unexpected EDAIC sample partition counts: {dict(sample_partition_counts)} "
        f"expected={expected_sample_partition_counts}"
    )

subject_partition_rows = json.loads((root / "outputs/splits/edaic_subject_partitions.json").read_text(encoding="utf-8"))
subject_partition_counts = Counter(row["partition"] for row in subject_partition_rows)
expected_subject_partition_counts = {"train": 163, "val": 56, "test": 56}
if dict(subject_partition_counts) != expected_subject_partition_counts:
    raise SystemExit(
        f"Unexpected EDAIC subject partition counts: {dict(subject_partition_counts)} "
        f"expected={expected_subject_partition_counts}"
    )

partitions_by_subject = defaultdict(set)
for row in subject_partition_rows:
    partitions_by_subject[row["subject_id"]].add(row["partition"])
overlaps = {subject_id: sorted(parts) for subject_id, parts in partitions_by_subject.items() if len(parts) > 1}
if overlaps:
    raise SystemExit(f"EDAIC subject partition overlap detected: {list(overlaps.items())[:5]}")
edaic_dev_subject_ids = sorted([row["subject_id"] for row in subject_partition_rows if row["partition"] in {"train", "val"}])
edaic_test_subject_ids = sorted([row["subject_id"] for row in subject_partition_rows if row["partition"] == "test"])
edaic_folds = json.loads((root / "outputs/splits/edaic_folds.json").read_text(encoding="utf-8"))
edaic_fold_report = json.loads((root / "outputs/splits/edaic_fold_report.json").read_text(encoding="utf-8"))
if len(edaic_fold_report) != 5:
    raise SystemExit(f"Unexpected EDAIC fold report length: {len(edaic_fold_report)}")
assert_fold_coverage("EDAIC", edaic_folds, edaic_dev_subject_ids, edaic_test_subject_ids)

with (root / "outputs/splits/edaic_join_audit.csv").open("r", encoding="utf-8", newline="") as handle:
    join_audit_rows = list(csv.DictReader(handle))
if len(join_audit_rows) != 3080:
    raise SystemExit(f"Unexpected EDAIC join audit row count: {len(join_audit_rows)}")
missing_join_rows = [
    row["sample_id"]
    for row in join_audit_rows
    if row["audio_found"] != "True" or row["transcript_found"] != "True"
]
if missing_join_rows:
    raise SystemExit(f"EDAIC join audit has missing pairs: {missing_join_rows[:10]}")

if not any("_random_segment_" in row["sample_id"] for row in edaic_manifest_rows):
    raise SystemExit("EDAIC manifest is missing random_segment samples.")
if not any("_segment_" in row["sample_id"] and "_random_segment_" not in row["sample_id"] for row in edaic_manifest_rows):
    raise SystemExit("EDAIC manifest is missing segment samples.")

transcripts_by_subject = defaultdict(set)
for row in edaic_manifest_rows:
    transcripts_by_subject[row["subject_id"]].add(row["transcript"])
bad_transcript_subjects = [
    subject_id for subject_id, transcripts in transcripts_by_subject.items() if len(transcripts) != 1
]
if bad_transcript_subjects:
    raise SystemExit(f"EDAIC subjects have inconsistent repeated full transcripts: {bad_transcript_subjects[:10]}")

edaic_config_path = root / "configs/archive/edaic/edaic_audio_text.yaml"
edaic_config = load_yaml_with_overrides(edaic_config_path, [])
edaic_metadata = resolve_metadata_paths(
    json.loads((root / "outputs/splits/edaic_manifest_metadata.json").read_text(encoding="utf-8"))
)
outer_partitions = _resolve_outer_partitions(edaic_config, edaic_metadata, 0)
if len(outer_partitions["outer_train_subject_ids"]) != 219:
    raise SystemExit(
        f"Unexpected EDAIC outer-train subject count from train.py logic: "
        f"{len(outer_partitions['outer_train_subject_ids'])}"
    )
if len(outer_partitions["final_eval_subject_ids"]) != 56:
    raise SystemExit(
        f"Unexpected EDAIC final-eval subject count from train.py logic: "
        f"{len(outer_partitions['final_eval_subject_ids'])}"
    )
edaic_training_plan = _resolve_training_subject_splits(
    edaic_config,
    edaic_metadata,
    {row["subject_id"]: int(row["label"]) for row in edaic_manifest_rows},
    0,
)
if not edaic_training_plan["uses_inner_split"]:
    raise SystemExit("EDAIC fixed mode unexpectedly skipped the deterministic inner split.")
if len(edaic_training_plan["selection_subject_ids"]) == 0:
    raise SystemExit("EDAIC fixed mode unexpectedly produced an empty inner validation split.")

edaic_cv_config = load_yaml_with_overrides(edaic_config_path, ["split.mode=cv"])
edaic_cv_training_plan = _resolve_training_subject_splits(
    edaic_cv_config,
    edaic_metadata,
    {row["subject_id"]: int(row["label"]) for row in edaic_manifest_rows},
    0,
)
expected_edaic_cv_holdout_ids = sorted(edaic_folds["0"]["final_eval_subject_ids"])
if not edaic_cv_training_plan["uses_inner_split"]:
    raise SystemExit("EDAIC CV mode unexpectedly skipped the deterministic inner split.")
if edaic_cv_training_plan["final_eval_subject_ids"] != expected_edaic_cv_holdout_ids:
    raise SystemExit("EDAIC CV mode final-eval subjects do not match fold_0 held-out ids.")
if set(edaic_cv_training_plan["train_subject_ids"]) & set(expected_edaic_cv_holdout_ids):
    raise SystemExit("EDAIC CV mode leaked held-out fold subjects into train_inner.")
if set(edaic_cv_training_plan["selection_subject_ids"]) & set(expected_edaic_cv_holdout_ids):
    raise SystemExit("EDAIC CV mode leaked held-out fold subjects into val_inner.")
if _resolve_final_eval_subject_ids(edaic_cv_config, edaic_metadata, 0) != expected_edaic_cv_holdout_ids:
    raise SystemExit("EDAIC CV mode evaluate.py resolution does not match fold_0 held-out ids.")

edaic_full_train_config = load_yaml_with_overrides(edaic_config_path, ["split.mode=full_train"])
edaic_full_train_plan = _resolve_training_subject_splits(
    edaic_full_train_config,
    edaic_metadata,
    {row["subject_id"]: int(row["label"]) for row in edaic_manifest_rows},
    0,
)
if edaic_full_train_plan["selection_subject_ids"]:
    raise SystemExit("EDAIC full_train mode unexpectedly created selection subjects.")
if len(edaic_full_train_plan["train_subject_ids"]) != len(edaic_dev_subject_ids):
    raise SystemExit("EDAIC full_train mode did not train on the full pooled train+val subject pool.")
if edaic_full_train_plan["final_eval_subject_ids"] != edaic_test_subject_ids:
    raise SystemExit("EDAIC full_train mode did not keep official test as the final eval split.")
if _resolve_final_eval_subject_ids(edaic_full_train_config, edaic_metadata, 0) != edaic_test_subject_ids:
    raise SystemExit("EDAIC full_train evaluate.py resolution did not keep official test.")

try:
    validate_hpo_split_mode(load_yaml_with_overrides(daic_config_path, ["split.mode=full_train"]))
except ValueError:
    pass
else:
    raise SystemExit("hpo.py did not reject split.mode=full_train.")

print("All expected manifest/split outputs are present.")
print("DAIC manifest invariants passed.")
print("EDAIC manifest invariants passed.")
PY

echo "[3/5] Verifying configurable evaluation aggregation and modality rendering"
run_python - <<'PY'
import json
import tempfile
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import torch

from src.aggregate import aggregate_predictions
from src.data.split_utils import deterministic_inner_split
from src.data.runtime import build_examples
import src.evaluate as evaluate_mod
from src.model.qwen2audio_lora import build_lora_config, resolve_lora_layer_selection
from src.utils import (
    PREDICTION_MODE_GENERATION,
    PREDICTION_MODE_LIKELIHOOD,
    PREDICTION_MODE_ORIGINAL_TEACHER_FORCED,
    load_yaml_with_overrides,
    resolve_aggregation_level,
    resolve_input_modality,
)

root = Path("/home/emre/Projects/AudioLLM/LLM-Depression")
config_paths = sorted(path for path in (root / "configs").glob("*.yaml") if path.name != "quarantines.yaml")
missing_key = []
for path in config_paths:
    config = load_yaml_with_overrides(path, [])
    evaluation_cfg = config.get("evaluation", {})
    if "sample_prediction_mode" in evaluation_cfg and "aggregation_level" not in evaluation_cfg:
        missing_key.append(path.name)
    if "aggregation_level" in evaluation_cfg:
        assert resolve_aggregation_level(config) == "subject", path.name
if missing_key:
    raise SystemExit(f"Configs missing evaluation.aggregation_level: {missing_key}")

segment_config = load_yaml_with_overrides(root / "configs/archive/edaic/edaic_audio_text.yaml", ["evaluation.aggregation_level=segment"])
if resolve_aggregation_level(segment_config) != "segment":
    raise SystemExit("Config override for evaluation.aggregation_level=segment did not resolve correctly.")
invalid_config = load_yaml_with_overrides(root / "configs/archive/edaic/edaic_audio_text.yaml", ["evaluation.aggregation_level=bad_level"])
try:
    resolve_aggregation_level(invalid_config)
except ValueError:
    pass
else:
    raise SystemExit("Invalid evaluation.aggregation_level value did not raise ValueError.")

audio_only_config = load_yaml_with_overrides(root / "configs/archive/daic/daic_audio_only.yaml", [])
if resolve_input_modality(audio_only_config) != "audio_only":
    raise SystemExit("configs/archive/daic/daic_audio_only.yaml did not resolve to audio_only modality.")
single_row = {
    "dataset": "daic",
    "subject_id": "smoke_subject",
    "sample_id": "smoke_subject_0",
    "audio_path": "/tmp/fake.wav",
    "transcript": "This transcript should be dropped in audio-only mode.",
    "label": 1,
    "label_text": "Depressed",
    "question_id": "",
}
audio_only_example = build_examples([single_row], audio_only_config, partition_name="smoke")[0]
assert audio_only_example["input_modality"] == "audio_only"
assert audio_only_example["transcript"] == ""
assert "Audio 1: <|audio_bos|><|AUDIO|><|audio_eos|>" in audio_only_example["prompt_text"]
assert "The transcript of the subject's speech is:" not in audio_only_example["prompt_text"]
assert audio_only_example["training_text"] == (
    f"{audio_only_example['prompt_text']}{audio_only_example['internal_label_text']}<|im_end|>\n"
)

text_only_config = load_yaml_with_overrides(root / "configs/archive/daic/daic_text_only.yaml", [])
if resolve_input_modality(text_only_config) != "text_only":
    raise SystemExit("configs/archive/daic/daic_text_only.yaml did not resolve to text_only modality.")
text_only_example = build_examples([single_row], text_only_config, partition_name="smoke")[0]
assert text_only_example["input_modality"] == "text_only"
assert text_only_example["transcript"] == single_row["transcript"]
assert text_only_example["audio_paths"] == []
assert text_only_example["audio_clip_seconds"] == []
assert "Audio 1: <|audio_bos|><|AUDIO|><|audio_eos|>" not in text_only_example["prompt_text"]
assert "The transcript of the subject's speech is:" in text_only_example["prompt_text"]
assert text_only_example["training_text"] == (
    f"{text_only_example['prompt_text']}{text_only_example['internal_label_text']}<|im_end|>\n"
)

daic_manifest_rows = [
    json.loads(line)
    for line in (root / "outputs/manifests/daic_manifest.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
daic_train_rows = [row for row in daic_manifest_rows if row["split_original"] == "train"]
daic_text_only_examples = build_examples(daic_train_rows, text_only_config, partition_name="train")
daic_train_subject_count = len({row["subject_id"] for row in daic_train_rows})
assert len(daic_text_only_examples) == daic_train_subject_count
assert all(example["sample_id"] == example["subject_id"] for example in daic_text_only_examples)
assert all(example["audio_paths"] == [] for example in daic_text_only_examples)

ab_config = load_yaml_with_overrides(root / "configs/archive/daic/daic_audio_text_hybrid_ab.yaml", [])
ab_example = build_examples([single_row], ab_config, partition_name="smoke")[0]
assert "Use this label legend:" in ab_example["prompt_text"]
assert "A = Depressed" in ab_example["prompt_text"]
assert "B = Non-depressed" in ab_example["prompt_text"]
assert "Answer with exactly one label: A or B." in ab_example["prompt_text"]

eatd_audio_only_config = load_yaml_with_overrides(root / "configs/archive/eatd/eatd_audio_only.yaml", [])
if resolve_input_modality(eatd_audio_only_config) != "audio_only":
    raise SystemExit("configs/archive/eatd/eatd_audio_only.yaml did not resolve to audio_only modality.")
eatd_manifest_rows = [
    json.loads(line)
    for line in (root / "outputs/manifests/eatd_manifest.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
rows_by_subject = defaultdict(list)
for row in eatd_manifest_rows:
    rows_by_subject[row["subject_id"]].append(row)
eatd_subject_rows = next(
    rows
    for _, rows in sorted(rows_by_subject.items())
    if {row["question_id"] for row in rows} == {"negative", "neutral", "positive"}
)
eatd_audio_only_example = build_examples(eatd_subject_rows, eatd_audio_only_config, partition_name="smoke")[0]
assert eatd_audio_only_example["input_modality"] == "audio_only"
assert eatd_audio_only_example["transcript"] == ""
assert "Audio 1: <|audio_bos|><|AUDIO|><|audio_eos|>" in eatd_audio_only_example["prompt_text"]
assert "Audio 3: <|audio_bos|><|AUDIO|><|audio_eos|>" in eatd_audio_only_example["prompt_text"]
assert "The transcript of the subject's speech is:" not in eatd_audio_only_example["prompt_text"]
assert "three responses: negative, neutral, and positive." in eatd_audio_only_example["prompt_text"]
assert eatd_audio_only_example["training_text"] == (
    f"{eatd_audio_only_example['prompt_text']}{eatd_audio_only_example['internal_label_text']}<|im_end|>\n"
)

edaic_text_only_config = load_yaml_with_overrides(root / "configs/archive/edaic/edaic_text_only.yaml", [])
if resolve_input_modality(edaic_text_only_config) != "text_only":
    raise SystemExit("configs/archive/edaic/edaic_text_only.yaml did not resolve to text_only modality.")
edaic_manifest_rows = [
    json.loads(line)
    for line in (root / "outputs/manifests/edaic_manifest.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
edaic_subject_partition_rows = json.loads((root / "outputs/splits/edaic_subject_partitions.json").read_text(encoding="utf-8"))
edaic_subject_labels = {row["subject_id"]: int(row["label"]) for row in edaic_subject_partition_rows}
edaic_dev_subject_ids = sorted(
    [row["subject_id"] for row in edaic_subject_partition_rows if row["partition"] in {"train", "val"}]
)
edaic_test_subject_ids = sorted(
    [row["subject_id"] for row in edaic_subject_partition_rows if row["partition"] == "test"]
)
edaic_inner_split = deterministic_inner_split(
    edaic_subject_labels,
    edaic_dev_subject_ids,
    seed=int(edaic_text_only_config["split"]["seed"]),
    val_ratio=float(edaic_text_only_config["split"]["inner_val_ratio"]),
)
assert len(edaic_inner_split["train_inner_subject_ids"]) == 175
edaic_train_inner_rows = [
    row for row in edaic_manifest_rows if row["subject_id"] in set(edaic_inner_split["train_inner_subject_ids"])
]
edaic_val_inner_rows = [
    row for row in edaic_manifest_rows if row["subject_id"] in set(edaic_inner_split["val_inner_subject_ids"])
]
edaic_final_eval_rows = [
    row for row in edaic_manifest_rows if row["subject_id"] in set(edaic_test_subject_ids)
]
assert len(build_examples(edaic_train_inner_rows, edaic_text_only_config, partition_name="train_inner")) == 175
assert len(build_examples(edaic_val_inner_rows, edaic_text_only_config, partition_name="val_inner")) == len(
    edaic_inner_split["val_inner_subject_ids"]
)
assert len(build_examples(edaic_final_eval_rows, edaic_text_only_config, partition_name="final_eval")) == len(
    edaic_test_subject_ids
)

invalid_modality_config = load_yaml_with_overrides(
    root / "configs/archive/daic/daic_audio_text.yaml",
    ["data.use_audio=false", "data.use_text=false"],
)
try:
    resolve_input_modality(invalid_modality_config)
except ValueError:
    pass
else:
    raise SystemExit("Invalid data.use_audio/data.use_text combination did not raise ValueError.")

last2_config = load_yaml_with_overrides(root / "configs/archive/edaic/edaic_audio_text_reg3.yaml", [])
if int(last2_config["lora"]["last_n_layers"]) != 2:
    raise SystemExit("Expected configs/archive/edaic/edaic_audio_text_reg3.yaml to set lora.last_n_layers=2.")

dummy_model = SimpleNamespace(config=SimpleNamespace(text_config=SimpleNamespace(num_hidden_layers=32)))
base_lora_config = load_yaml_with_overrides(root / "configs/archive/edaic/edaic_audio_text.yaml", [])
base_peft_cfg, base_layer_selection = build_lora_config(base_lora_config, dummy_model)
assert getattr(base_peft_cfg, "layers_to_transform", None) is None
assert base_layer_selection["requested_last_n_layers"] is None
assert base_layer_selection["decoder_hidden_layer_count"] == 32
assert base_layer_selection["layers_to_transform"] is None

last2_override = load_yaml_with_overrides(root / "configs/archive/edaic/edaic_audio_text.yaml", ["lora.last_n_layers=2"])
last2_peft_cfg, last2_layer_selection = build_lora_config(last2_override, dummy_model)
assert last2_layer_selection["requested_last_n_layers"] == 2
assert last2_layer_selection["decoder_hidden_layer_count"] == 32
assert last2_layer_selection["layers_to_transform"] == [30, 31]
assert list(last2_peft_cfg.layers_to_transform) == [30, 31]

zero_lora_cfg = load_yaml_with_overrides(root / "configs/archive/edaic/edaic_audio_text.yaml", ["lora.last_n_layers=0"])
zero_layer_selection = resolve_lora_layer_selection(zero_lora_cfg, dummy_model)
if zero_layer_selection["requested_last_n_layers"] is not None:
    raise SystemExit("lora.last_n_layers=0 must resolve to None (unset) like False.")

for override in ["lora.last_n_layers=-1", "lora.last_n_layers=40", "lora.last_n_layers=two"]:
    bad_lora_cfg = load_yaml_with_overrides(root / "configs/archive/edaic/edaic_audio_text.yaml", [override])
    try:
        resolve_lora_layer_selection(bad_lora_cfg, dummy_model)
    except ValueError:
        pass
    else:
        raise SystemExit(f"Invalid {override} did not raise ValueError.")

likelihood_rows = [
    {
        "subject_id": "S1",
        "sample_id": "S1_1",
        "label": 1,
        "label_text": "Depressed",
        "likelihood_prediction": 1,
        "likelihood_prediction_text": "Depressed",
        "dep_score": 0.9,
        "non_score": 0.1,
    },
    {
        "subject_id": "S1",
        "sample_id": "S1_2",
        "label": 1,
        "label_text": "Depressed",
        "likelihood_prediction": 0,
        "likelihood_prediction_text": "Non-depressed",
        "dep_score": 0.8,
        "non_score": 0.2,
    },
    {
        "subject_id": "S2",
        "sample_id": "S2_1",
        "label": 0,
        "label_text": "Non-depressed",
        "likelihood_prediction": 0,
        "likelihood_prediction_text": "Non-depressed",
        "dep_score": 0.1,
        "non_score": 0.9,
    },
    {
        "subject_id": "S2",
        "sample_id": "S2_2",
        "label": 0,
        "label_text": "Non-depressed",
        "likelihood_prediction": 1,
        "likelihood_prediction_text": "Depressed",
        "dep_score": 0.2,
        "non_score": 0.8,
    },
]
headline_rows, headline_metrics, subject_rows, subject_metrics = aggregate_predictions(
    likelihood_rows,
    mode=PREDICTION_MODE_LIKELIHOOD,
    aggregation_level="subject",
)
assert headline_rows == subject_rows
assert headline_metrics["aggregation_level"] == "subject"
assert headline_metrics["num_subjects"] == 2
assert headline_metrics["predicted_depressed_subjects"] == 1
assert headline_metrics["predicted_non_depressed_subjects"] == 1
assert subject_metrics["num_valid_subject_predictions"] == 2

headline_rows, headline_metrics, subject_rows, subject_metrics = aggregate_predictions(
    likelihood_rows,
    mode=PREDICTION_MODE_LIKELIHOOD,
    aggregation_level="segment",
)
assert headline_metrics["aggregation_level"] == "segment"
assert headline_metrics["num_segments"] == 4
assert headline_metrics["num_units"] == 4
assert headline_metrics["predicted_depressed_segments"] == 2
assert headline_metrics["predicted_non_depressed_segments"] == 2
assert subject_metrics["num_subjects"] == 2

generation_rows = [
    {
        "subject_id": "G1",
        "sample_id": "G1_1",
        "label": 1,
        "label_text": "Depressed",
        "parsed_prediction": 1,
        "generation_text": "Depressed",
    },
    {
        "subject_id": "G1",
        "sample_id": "G1_2",
        "label": 1,
        "label_text": "Depressed",
        "parsed_prediction": "",
        "generation_text": "maybe",
    },
    {
        "subject_id": "G2",
        "sample_id": "G2_1",
        "label": 0,
        "label_text": "Non-depressed",
        "parsed_prediction": 0,
        "generation_text": "Non-depressed",
    },
]
headline_rows, headline_metrics, _, _ = aggregate_predictions(
    generation_rows,
    mode=PREDICTION_MODE_GENERATION,
    aggregation_level="segment",
)
assert headline_metrics["invalid_generations"] == 1
assert headline_metrics["invalid_segments"] == 1
assert abs(headline_metrics["valid_only_accuracy"] - 1.0) < 1e-9
assert abs(headline_metrics["accuracy"] - (2 / 3)) < 1e-9

teacher_rows = [
    {
        "subject_id": "T1",
        "sample_id": "T1_1",
        "label": 1,
        "label_text": "Depressed",
        "teacher_forced_prediction": "",
        "teacher_forced_decoded_text": "INVALID",
        "dep_score": 0.4,
        "non_score": 0.6,
    },
    {
        "subject_id": "T2",
        "sample_id": "T2_1",
        "label": 0,
        "label_text": "Non-depressed",
        "teacher_forced_prediction": 0,
        "teacher_forced_decoded_text": "B",
        "dep_score": 0.1,
        "non_score": 0.9,
    },
]
headline_rows, headline_metrics, _, _ = aggregate_predictions(
    teacher_rows,
    mode=PREDICTION_MODE_ORIGINAL_TEACHER_FORCED,
    aggregation_level="segment",
)
assert headline_metrics["invalid_teacher_forced_predictions"] == 1
assert headline_metrics["invalid_segments"] == 1
assert abs(headline_metrics["accuracy"] - 0.5) < 1e-9
assert abs(headline_metrics["valid_only_accuracy"] - 1.0) < 1e-9


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.config = type("Config", (), {"use_cache": True})()


def fake_generation_predict(model, processor, example, config, device, silence_audio, checkpoint_name):
    return {
        **evaluate_mod._base_sample_row(example, checkpoint_name, PREDICTION_MODE_GENERATION),
        "generation_text": "Depressed" if example["label"] == 1 else "INVALID",
        "parsed_prediction": 1 if example["label"] == 1 else "",
        "generation_prediction_text": "Depressed" if example["label"] == 1 else "INVALID",
    }


examples = [
    {
        "subject_id": "E1",
        "sample_id": "E1_1",
        "label": 1,
        "label_text": "Depressed",
        "internal_label_text": "A",
        "prompt_text": "prompt",
        "audio_paths": [],
        "audio_clip_seconds": [],
    },
    {
        "subject_id": "E1",
        "sample_id": "E1_2",
        "label": 1,
        "label_text": "Depressed",
        "internal_label_text": "A",
        "prompt_text": "prompt",
        "audio_paths": [],
        "audio_clip_seconds": [],
    },
    {
        "subject_id": "E2",
        "sample_id": "E2_1",
        "label": 0,
        "label_text": "Non-depressed",
        "internal_label_text": "B",
        "prompt_text": "prompt",
        "audio_paths": [],
        "audio_clip_seconds": [],
    },
]
config = load_yaml_with_overrides(
    root / "configs/archive/edaic/edaic_audio_text.yaml",
    [
        "evaluation.sample_prediction_mode=generation",
        "evaluation.aggregation_level=segment",
    ],
)
original_backend = evaluate_mod._prediction_backend
evaluate_mod._prediction_backend = lambda mode: fake_generation_predict
try:
    with tempfile.TemporaryDirectory() as tmpdir:
        payload = evaluate_mod.evaluate_examples(
            DummyModel(),
            processor=None,
            examples=examples,
            config=config,
            output_dir=tmpdir,
            checkpoint_name="dummy_checkpoint",
            sample_prediction_mode=PREDICTION_MODE_GENERATION,
        )
        assert payload["active_backend"] == PREDICTION_MODE_GENERATION
        assert payload["active_aggregation_level"] == "segment"
        assert payload["headline_metrics"]["aggregation_level"] == "segment"
        assert payload["subject_metrics"]["aggregation_level"] == "subject"
        assert len(payload["headline_rows"]) == 3
        assert len(payload["subject_rows"]) == 2
        assert (Path(tmpdir) / "predictions_headline_level.csv").exists()
        assert (Path(tmpdir) / "predictions_subject_level.csv").exists()
        assert (Path(tmpdir) / "metrics_generation.json").exists()
        assert (Path(tmpdir) / "metrics_subject_level_generation.json").exists()
finally:
    evaluate_mod._prediction_backend = original_backend

print("Evaluation aggregation checks passed.")
print("Audio-only prompt rendering checks passed.")
PY

echo "[4/5] Validating DepAdapter helper round-trip"
run_python - <<'PY'
import tempfile
import types

import torch
from transformers.modeling_outputs import BaseModelOutput

from src.model.qwen2audio_lora import (
    attach_dep_adapter,
    configure_trainable_audio_modules,
    load_additional_audio_modules,
    resolve_audio_adapter_config,
    save_additional_audio_modules,
    summarize_audio_module_state,
)


class DummyEncoder(torch.nn.Module):
    def __init__(self, d_model: int = 8):
        super().__init__()
        self.config = types.SimpleNamespace(d_model=d_model)

    def forward(
        self,
        input_features,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        hidden = input_features + 1.0
        if return_dict is False:
            return (hidden,)
        return BaseModelOutput(last_hidden_state=hidden, hidden_states=None, attentions=None)


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.audio_tower = DummyEncoder()
        self.multi_modal_projector = torch.nn.Linear(8, 8)
        self.config = types.SimpleNamespace(use_cache=True)


config = {
    "audio_adapter": {
        "enabled": True,
        "adapter_dim": 4,
        "dropout": 0.0,
        "train_projector": True,
    }
}
audio_cfg = resolve_audio_adapter_config(config)
model = DummyModel()
attach_dep_adapter(model, audio_cfg)
configure_trainable_audio_modules(model, audio_cfg)
state = summarize_audio_module_state(model)
assert state["adapter_attached"], state
assert state["adapter_trainable_params"] > 0, state
assert state["projector_trainable_params"] > 0, state

forward_output = model.audio_tower(input_features=torch.zeros(2, 3, 8))
assert forward_output.last_hidden_state.shape == (2, 3, 8)

for parameter in model.audio_tower.audio_adapter.parameters():
    torch.nn.init.constant_(parameter, 0.25)
for parameter in model.multi_modal_projector.parameters():
    torch.nn.init.constant_(parameter, 0.5)

with tempfile.TemporaryDirectory() as tmpdir:
    metadata = save_additional_audio_modules(model, tmpdir, config=config)
    restored = DummyModel()
    loaded = load_additional_audio_modules(restored, tmpdir)
    assert metadata["enabled"] is True, metadata
    assert metadata["train_projector"] is True, metadata
    assert loaded["adapter_state_loaded"] is True, loaded
    assert loaded["projector_state_loaded"] is True, loaded
    for key, value in model.audio_tower.audio_adapter.state_dict().items():
        assert torch.equal(value, restored.audio_tower.audio_adapter.state_dict()[key]), key
    for key, value in model.multi_modal_projector.state_dict().items():
        assert torch.equal(value, restored.multi_modal_projector.state_dict()[key]), key

print("DepAdapter helper round-trip passed.")
PY

echo "[emotion] Verifying emotion-augmented prompt injection + K-chunk alignment"
run_python "$PROJECT_ROOT/scripts/test_emotion_injection.py"

echo "[emotion] Verifying Qwen2-Audio emotion cache (emotion_en, no emotion_zh) is a drop-in"
run_python "$PROJECT_ROOT/scripts/test_emotion_qwen2audio_cache.py"

echo "[5/5] Printing split source summary"
run_python - <<'PY'
import json
from pathlib import Path
root = Path("/home/emre/Projects/AudioLLM/LLM-Depression/outputs/splits")
for name in ["daic", "edaic", "cmdc", "eatd"]:
    path = root / f"{name}_manifest_metadata.json"
    data = json.loads(path.read_text())
    print(name, json.dumps({k: data[k] for k in data if k in {"dataset", "split_source", "split_source_notes", "manifest_hash"}}, ensure_ascii=False))
PY
