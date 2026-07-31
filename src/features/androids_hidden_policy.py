"""Protocol-specific policy for Androids hidden-state classifiers.

The repository already contains hidden-state heads for other datasets.  The
Androids corpus needs a different, explicit aggregation contract: window
probabilities are averaged within a turn and turn probabilities are averaged
within a subject.  Keeping that contract here makes the extraction metadata,
fixed heads, Optuna objective, audit, and workbook updater use the same
definition.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from src.metrics import binary_auroc, classification_metrics


ANDROID_DATASET = "androids_interview"
ANDROID_MODALITIES = ("audio_only", "audio_text", "text_only")
ANDROID_HEADS = ("logreg_raw", "xgb_raw", "xgb_optuna_150t_d6")
ANDROID_HIDDEN_CACHE_SCHEMA = "androids_hidden_cache.v1"
ANDROID_HIDDEN_FIXED_SCHEMA = "androids_hidden_fixed_classifier.v1"
ANDROID_HIDDEN_OPTUNA_SCHEMA = "androids_hidden_xgb_optuna.v1"
ANDROID_AGGREGATION_POLICY = "window_mean_to_turn_mean_to_subject_mean"
ANDROID_TEXT_AGGREGATION_POLICY = "one_vector_per_subject"
ANDROID_THRESHOLD = 0.5
ANDROID_MANIFEST_HASH = "01a351f7277e4763a8bb9e4983bba190b265becafafca6d7ee04bdcfc948cbed"
ANDROID_SPLIT_HASH = "f75dd2ba7bb324af26de8c5ae3497d2108e6b50815c0ef6cbcade7de70992518"
ANDROID_SUBJECT_COUNT = 116
ANDROID_CONTROL_COUNT = 52
ANDROID_PATIENT_COUNT = 64

CACHE_ARTIFACT_NAMES = (
    "outer_train.npz",
    "outer_train_rows.jsonl",
    "final_eval.npz",
    "final_eval_rows.jsonl",
    "extraction_metadata.json",
)


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "size_bytes": int(path.stat().st_size),
        "sha256": file_sha256(path),
    }


def cache_identity(cache_dir: Path) -> dict[str, Any]:
    return {name: file_identity(cache_dir / name) for name in CACHE_ARTIFACT_NAMES}


def compact_cache_identity(cache_dir: Path) -> dict[str, Any]:
    """Return identities for cache artifacts retained in a compact sync.

    The hidden vectors are deliberately not copied back from MN5 by default.
    The compact rows and metadata are sufficient to audit provenance,
    aggregation, labels, and the classifier outputs after the remote audit has
    already validated the vectors in place.
    """
    return {
        name: file_identity(cache_dir / name)
        for name in CACHE_ARTIFACT_NAMES
        if not name.endswith(".npz") and (cache_dir / name).is_file()
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"Expected an object in {path}:{line_number}.")
                rows.append(value)
    return rows


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True)
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in row.items()
                }
            )


def write_sha256_manifest(root: Path, output: Path, *, exclude_suffixes: tuple[str, ...] = ()) -> None:
    """Write a deterministic relative-path checksum manifest."""
    entries: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path == output:
            continue
        if any(path.name.endswith(suffix) for suffix in exclude_suffixes):
            continue
        entries.append(f"{file_sha256(path)}\t{path.relative_to(root).as_posix()}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(entries) + ("\n" if entries else ""), encoding="utf-8")


def modality_policy(modality: str) -> str:
    if modality not in ANDROID_MODALITIES:
        raise ValueError(f"Unsupported Androids modality: {modality!r}")
    return ANDROID_TEXT_AGGREGATION_POLICY if modality == "text_only" else ANDROID_AGGREGATION_POLICY


def validate_androids_cache_metadata(
    metadata: dict[str, Any],
    *,
    modality: str | None = None,
    fold: int | None = None,
    source_commit: str | None = None,
    require_production: bool = False,
) -> None:
    if metadata.get("schema_version") != ANDROID_HIDDEN_CACHE_SCHEMA:
        raise ValueError(
            "Refusing an incompatible Androids hidden cache schema: "
            f"{metadata.get('schema_version')!r}."
        )
    if str(metadata.get("dataset")) != ANDROID_DATASET:
        raise ValueError("Androids hidden cache has the wrong dataset.")
    resolved_modality = str(metadata.get("modality") or metadata.get("input_modality") or "")
    if resolved_modality not in ANDROID_MODALITIES:
        raise ValueError(f"Androids cache has invalid modality {resolved_modality!r}.")
    if modality is not None and resolved_modality != modality:
        raise ValueError(f"Cache modality {resolved_modality!r} != requested {modality!r}.")
    if fold is not None and int(metadata.get("fold", -1)) != int(fold):
        raise ValueError(f"Cache fold {metadata.get('fold')!r} != requested {fold!r}.")
    if source_commit is not None and str(metadata.get("source_commit")) != source_commit:
        raise ValueError("Cache source commit does not match the requested provenance.")
    if metadata.get("manifest_sha256") != ANDROID_MANIFEST_HASH:
        raise ValueError("Cache manifest hash does not match the frozen Androids manifest.")
    if metadata.get("split_metadata_sha256") != ANDROID_SPLIT_HASH:
        raise ValueError("Cache split hash does not match the frozen Androids official folds.")
    if resolved_modality == "audio_text" and metadata.get("audio_text_transcript_scope") != "segment_aligned":
        raise ValueError("Androids hidden Audio + Text must use the segment-aligned source checkpoint.")
    if metadata.get("audio_text_transcript_scope") == "full_turn":
        raise ValueError("Full-turn Androids hidden caches are not part of this experiment.")
    if metadata.get("aggregation_policy") != modality_policy(resolved_modality):
        raise ValueError("Cache aggregation policy is incompatible with the Androids protocol.")
    if require_production and metadata.get("max_examples") is not None:
        raise ValueError("A smoke/truncated cache cannot be used for production acceptance.")


def load_androids_cache(
    cache_dir: Path,
    *,
    modality: str | None = None,
    fold: int | None = None,
    source_commit: str | None = None,
    require_production: bool = False,
    require_vectors: bool = True,
) -> tuple[np.ndarray, list[dict[str, Any]], np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    for name in CACHE_ARTIFACT_NAMES:
        if not require_vectors and name.endswith(".npz"):
            continue
        path = cache_dir / name
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing Androids hidden cache artifact: {path}")
    metadata = read_json(cache_dir / "extraction_metadata.json")
    validate_androids_cache_metadata(
        metadata,
        modality=modality,
        fold=fold,
        source_commit=source_commit,
        require_production=require_production,
    )

    def load_partition(name: str) -> tuple[np.ndarray, list[dict[str, Any]]]:
        rows = read_jsonl(cache_dir / f"{name}_rows.jsonl")
        vector_path = cache_dir / f"{name}.npz"
        if vector_path.is_file():
            with np.load(vector_path) as payload:
                vectors = np.asarray(payload["vectors"], dtype=np.float32)
        elif require_vectors:
            raise FileNotFoundError(f"Missing Androids hidden vectors: {vector_path}")
        else:
            dimension = int(metadata.get("vector_dimension", 0))
            if dimension < 1:
                raise ValueError("Compact Androids cache metadata lacks vector_dimension.")
            vectors = np.zeros((len(rows), dimension), dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(rows) or vectors.shape[1] < 1:
            raise ValueError(f"Invalid Androids {name} cache shape {vectors.shape} for {len(rows)} rows.")
        if not bool(np.isfinite(vectors).all()):
            raise ValueError(f"Androids {name} cache contains non-finite values.")
        sample_ids = [str(row.get("sample_id", "")) for row in rows]
        if not all(sample_ids) or len(sample_ids) != len(set(sample_ids)):
            raise ValueError(f"Androids {name} cache has missing or duplicate sample IDs.")
        return vectors, rows

    train_x, train_rows = load_partition("outer_train")
    eval_x, eval_rows = load_partition("final_eval")
    if train_x.shape[1] != eval_x.shape[1]:
        raise ValueError("Androids training and held-out hidden dimensions differ.")
    train_subjects = {str(row["subject_id"]) for row in train_rows}
    eval_subjects = {str(row["subject_id"]) for row in eval_rows}
    if train_subjects & eval_subjects:
        raise ValueError("Androids cache contains outer train/held-out subject leakage.")
    return train_x, train_rows, eval_x, eval_rows, metadata


def _android_row_key(row: dict[str, Any]) -> tuple[str, int]:
    response_id = str(row.get("response_id", "")).strip()
    if not response_id:
        raise ValueError("Androids audio hidden rows require response_id metadata.")
    if "window_index" not in row:
        raise ValueError(f"Androids audio hidden row {row.get('sample_id')} lacks window_index.")
    return response_id, int(row["window_index"])


def validate_androids_row_inventory(rows: list[dict[str, Any]], modality: str) -> dict[str, Any]:
    """Validate complete turn/window metadata and return inventory counts."""
    if modality == "text_only":
        subjects = {str(row["subject_id"]) for row in rows}
        if len(rows) != len(subjects) or any(str(row["sample_id"]) != str(row["subject_id"]) for row in rows):
            raise ValueError("Androids text-only cache must contain one vector per subject.")
        for row in rows:
            required = (
                "source_turn_count",
                "source_window_count",
                "source_turn_ids",
                "source_window_ids",
                "source_window_inventory_sha256",
            )
            if any(key not in row for key in required):
                raise ValueError("Androids text-only metadata lacks source inventory fields.")
            if int(row["source_turn_count"]) < 1 or int(row["source_window_count"]) < 1:
                raise ValueError("Androids text-only metadata lacks source turn/window counts.")
            if len(row["source_turn_ids"]) != int(row["source_turn_count"]):
                raise ValueError("Androids text-only source turn inventory is inconsistent.")
            if len(row["source_window_ids"]) != int(row["source_window_count"]):
                raise ValueError("Androids text-only source window inventory is inconsistent.")
            if not str(row["source_window_inventory_sha256"]):
                raise ValueError("Androids text-only source window inventory hash is empty.")
        return {
            "row_count": len(rows),
            "subject_count": len(subjects),
            "turn_count": sum(int(row["source_turn_count"]) for row in rows),
            "window_count": sum(int(row["source_window_count"]) for row in rows),
        }

    subjects: dict[str, set[str]] = defaultdict(set)
    turns: dict[str, set[int]] = defaultdict(set)
    windows: dict[str, list[int]] = defaultdict(list)
    labels: dict[str, int] = {}
    declared: dict[str, int] = {}
    turn_subject: dict[str, str] = {}
    sample_ids: set[str] = set()
    for row in rows:
        sample_id = str(row.get("sample_id", ""))
        if not sample_id or sample_id in sample_ids:
            raise ValueError(f"Duplicate/missing Androids sample_id: {sample_id!r}.")
        sample_ids.add(sample_id)
        subject_id = str(row["subject_id"])
        label = int(row["label"])
        if subject_id in labels and labels[subject_id] != label:
            raise ValueError(f"Androids subject {subject_id} has inconsistent labels.")
        labels[subject_id] = label
        required = (
            "recording_id",
            "turn_id",
            "turn_key",
            "response_id",
            "window_id",
            "window_index",
            "num_windows",
            "num_segments",
            "segment_index",
            "start_time",
            "end_time",
            "segment_duration",
            "turn_duration",
        )
        if any(key not in row for key in required):
            raise ValueError(f"Androids audio hidden row {sample_id} lacks complete turn/window metadata.")
        response_id, window_index = _android_row_key(row)
        prior_subject = turn_subject.setdefault(response_id, subject_id)
        if prior_subject != subject_id:
            raise ValueError(f"Androids turn {response_id} spans multiple subjects.")
        if str(row.get("window_id", "")) != f"{response_id}_w{window_index:02d}":
            raise ValueError(f"Androids window_id is inconsistent for {sample_id}.")
        num_windows = int(row.get("num_windows", row.get("num_segments", 0)))
        num_segments = int(row.get("num_segments", 0))
        if num_windows < 1 or num_segments != num_windows:
            raise ValueError(f"Androids window count metadata is invalid for {sample_id}.")
        if not math.isfinite(float(row["start_time"])) or not math.isfinite(float(row["end_time"])):
            raise ValueError(f"Androids interval metadata is invalid for {sample_id}.")
        if float(row["end_time"]) <= float(row["start_time"]):
            raise ValueError(f"Androids interval is not positive for {sample_id}.")
        if float(row["segment_duration"]) <= 0 or float(row["turn_duration"]) <= 0:
            raise ValueError(f"Androids duration metadata is not positive for {sample_id}.")
        if int(row["segment_index"]) != window_index:
            raise ValueError(f"Androids segment/window indices differ for {sample_id}.")
        subjects[subject_id].add(response_id)
        turns[response_id].add(window_index)
        windows[response_id].append(window_index)
        prior = declared.setdefault(response_id, num_windows)
        if prior != num_windows:
            raise ValueError(f"Androids turn {response_id} has inconsistent window counts.")
    for response_id, indices in turns.items():
        expected = set(range(declared[response_id]))
        if indices != expected:
            raise ValueError(
                f"Androids turn {response_id} is incomplete: observed={sorted(indices)} expected={sorted(expected)}."
            )
    return {
        "row_count": len(rows),
        "subject_count": len(subjects),
        "turn_count": len(turns),
        "window_count": len(rows),
        "turns_per_subject": dict(sorted(Counter(len(value) for value in subjects.values()).items())),
        "windows_per_turn": dict(sorted(Counter(len(value) for value in turns.values()).items())),
    }


def androids_training_weights(
    rows: list[dict[str, Any]],
    modality: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return the frozen Androids classifier fit weights and an audit."""
    if not rows:
        raise ValueError("Androids classifier fitting requires non-empty training rows.")
    inventory = validate_androids_row_inventory(rows, modality)
    if modality == "text_only":
        weights = np.ones(len(rows), dtype=np.float64)
        subject_totals = {str(row["subject_id"]): 1.0 for row in rows}
        audit = {
            "schema_version": "androids_hidden_weight_audit.v1",
            "policy": ANDROID_TEXT_AGGREGATION_POLICY,
            "formula": "unit_weight_one_vector_per_subject",
            "row_count": len(rows),
            "subject_count": len(subject_totals),
            "mean_weight": 1.0,
            "equal_subject_totals": True,
            "equal_turn_weight_within_subject": True,
            "subject_weight_totals": subject_totals,
            "inventory": inventory,
        }
        return weights, audit

    turns_by_subject: dict[str, set[str]] = defaultdict(set)
    windows_by_turn: Counter[str] = Counter()
    turn_subject: dict[str, str] = {}
    for row in rows:
        subject_id = str(row["subject_id"])
        response_id = str(row["response_id"])
        turns_by_subject[subject_id].add(response_id)
        windows_by_turn[response_id] += 1
        prior = turn_subject.setdefault(response_id, subject_id)
        if prior != subject_id:
            raise ValueError(f"Androids turn {response_id} spans multiple subjects.")
    raw = np.asarray(
        [
            1.0
            / (
                len(turns_by_subject[str(row["subject_id"])])
                * windows_by_turn[str(row["response_id"])]
            )
            for row in rows
        ],
        dtype=np.float64,
    )
    weights = raw / raw.mean()
    subject_totals: defaultdict[str, float] = defaultdict(float)
    turn_totals: defaultdict[str, float] = defaultdict(float)
    for row, weight in zip(rows, weights.tolist()):
        subject_totals[str(row["subject_id"])] += float(weight)
        turn_totals[str(row["response_id"])] += float(weight)
    turn_totals_by_subject: defaultdict[str, list[float]] = defaultdict(list)
    for response_id, total in turn_totals.items():
        turn_totals_by_subject[turn_subject[response_id]].append(total)
    equal_subject_totals = len({round(value, 10) for value in subject_totals.values()}) == 1
    equal_turn_weight = all(
        len({round(value, 10) for value in values}) == 1
        for values in turn_totals_by_subject.values()
    )
    if not equal_subject_totals or not equal_turn_weight:
        raise AssertionError("Androids classifier weights do not satisfy the balance contract.")
    audit = {
        "schema_version": "androids_hidden_weight_audit.v1",
        "policy": ANDROID_AGGREGATION_POLICY,
        "formula": "1 / (turns_for_subject * windows_for_turn), rescaled_to_mean_one",
        "row_count": len(rows),
        "subject_count": len(turns_by_subject),
        "turn_count": len(windows_by_turn),
        "mean_weight": float(weights.mean()),
        "raw_subject_weight_totals": {
            subject_id: 1.0 for subject_id in sorted(turns_by_subject)
        },
        "subject_weight_totals": dict(sorted(subject_totals.items())),
        "turn_weight_totals": dict(sorted(turn_totals.items())),
        "turn_weight_totals_by_subject": {
            subject_id: sorted(values)
            for subject_id, values in sorted(turn_totals_by_subject.items())
        },
        "equal_subject_totals": equal_subject_totals,
        "equal_turn_weight_within_subject": equal_turn_weight,
        "turns_per_subject": {
            subject_id: len(turns)
            for subject_id, turns in sorted(turns_by_subject.items())
        },
        "windows_per_turn": dict(sorted(windows_by_turn.items())),
        "inventory": inventory,
    }
    return weights, audit


def _strict_probability_prediction(probability: float) -> int:
    if probability > ANDROID_THRESHOLD:
        return 1
    if probability < ANDROID_THRESHOLD:
        return 0
    return -1


def _metrics_for_subject_rows(subject_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not subject_rows:
        raise ValueError("Androids aggregation produced no subject rows.")
    y_true = [int(row["label"]) for row in subject_rows]
    raw_predictions = [int(row["prediction"]) for row in subject_rows]
    ties = sum(prediction == -1 for prediction in raw_predictions)
    # Androids headline metrics treat an exact 0.5 tie as an incorrect answer.
    strict_predictions = [
        prediction if prediction in (0, 1) else 1 - label
        for prediction, label in zip(raw_predictions, y_true)
    ]
    metrics = classification_metrics(y_true, strict_predictions)
    tn, fp = metrics["confusion_matrix"][0]
    fn, _ = metrics["confusion_matrix"][1]
    negative_precision = tn / (tn + fn) if tn + fn else 0.0
    negative_recall = tn / (tn + fp) if tn + fp else 0.0
    metrics["negative_f1"] = (
        2 * negative_precision * negative_recall / (negative_precision + negative_recall)
        if negative_precision + negative_recall
        else 0.0
    )
    probabilities = [float(row["probability"]) for row in subject_rows]
    metrics.update(
        {
            "auroc": binary_auroc(y_true, probabilities),
            "threshold": ANDROID_THRESHOLD,
            "tie_count": ties,
            "invalid_subjects": ties,
            "num_subjects": len(subject_rows),
            "support_negative": sum(label == 0 for label in y_true),
            "support_positive": sum(label == 1 for label in y_true),
            "strict_tie_policy": "exact_0.5_is_invalid_and_counted_wrong",
            "aggregation_policy": subject_rows[0].get("aggregation_policy"),
        }
    )
    valid_rows = [row for row in subject_rows if int(row["prediction"]) in (0, 1)]
    if valid_rows and len(valid_rows) != len(subject_rows):
        valid_metrics = classification_metrics(
            [int(row["label"]) for row in valid_rows],
            [int(row["prediction"]) for row in valid_rows],
        )
        metrics["valid_only_metrics"] = valid_metrics
    else:
        metrics["valid_only_metrics"] = None
    return metrics


def aggregate_androids_hidden_predictions(
    sample_rows: list[dict[str, Any]],
    modality: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Aggregate classifier probabilities from windows to turns to subjects."""
    if not sample_rows:
        raise ValueError("Androids classifier aggregation requires sample rows.")
    if modality == "text_only":
        by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in sample_rows:
            by_subject[str(row["subject_id"])].append(row)
        if any(len(rows) != 1 for rows in by_subject.values()):
            raise ValueError("Androids text-only classifier requires one vector per subject.")
        subject_rows = []
        for subject_id, rows in sorted(by_subject.items()):
            row = rows[0]
            probability = float(row["probability"])
            prediction = _strict_probability_prediction(probability)
            subject_rows.append(
                {
                    "subject_id": subject_id,
                    "label": int(row["label"]),
                    "probability": probability,
                    "prediction": prediction,
                    "predicted_class": prediction,
                    "num_turns": int(row.get("source_turn_count", 0)),
                    "num_windows": int(row.get("source_window_count", 0)),
                    "aggregation_policy": ANDROID_TEXT_AGGREGATION_POLICY,
                    "classifier_variant": row.get("classifier_variant"),
                }
            )
        return [], subject_rows, _metrics_for_subject_rows(subject_rows)

    by_turn: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sample_rows:
        response_id = str(row.get("response_id", "")).strip()
        if not response_id:
            raise ValueError("Androids audio classifier sample rows require response_id.")
        by_turn[response_id].append(row)
    turn_rows: list[dict[str, Any]] = []
    for response_id, rows in sorted(by_turn.items()):
        labels = {int(row["label"]) for row in rows}
        subjects = {str(row["subject_id"]) for row in rows}
        if len(labels) != 1 or len(subjects) != 1:
            raise ValueError(f"Androids turn {response_id} has inconsistent subject or label metadata.")
        expected = int(rows[0].get("num_windows", rows[0].get("num_segments", len(rows))))
        indices = {int(row["window_index"]) for row in rows}
        if expected != len(rows) or indices != set(range(expected)):
            raise ValueError(f"Androids turn {response_id} is incomplete during aggregation.")
        probability = float(np.mean([float(row["probability"]) for row in rows]))
        prediction = _strict_probability_prediction(probability)
        turn_rows.append(
            {
                "subject_id": next(iter(subjects)),
                "response_id": response_id,
                "label": next(iter(labels)),
                "probability": probability,
                "prediction": prediction,
                "predicted_class": prediction,
                "num_windows": len(rows),
                "window_ids": [str(row["sample_id"]) for row in sorted(rows, key=lambda item: int(item["window_index"]))],
                "aggregation_policy": ANDROID_AGGREGATION_POLICY,
                "classifier_variant": rows[0].get("classifier_variant"),
            }
        )
    by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in turn_rows:
        by_subject[str(row["subject_id"])].append(row)
    subject_rows = []
    for subject_id, rows in sorted(by_subject.items()):
        labels = {int(row["label"]) for row in rows}
        if len(labels) != 1:
            raise ValueError(f"Androids subject {subject_id} has inconsistent turn labels.")
        probability = float(np.mean([float(row["probability"]) for row in rows]))
        prediction = _strict_probability_prediction(probability)
        subject_rows.append(
            {
                "subject_id": subject_id,
                "label": next(iter(labels)),
                "probability": probability,
                "prediction": prediction,
                "predicted_class": prediction,
                "num_turns": len(rows),
                "num_windows": sum(int(row["num_windows"]) for row in rows),
                "turn_ids": [str(row["response_id"]) for row in rows],
                "aggregation_policy": ANDROID_AGGREGATION_POLICY,
                "classifier_variant": rows[0].get("classifier_variant"),
            }
        )
    return turn_rows, subject_rows, _metrics_for_subject_rows(subject_rows)


def metrics_close(expected: dict[str, Any], observed: dict[str, Any], *, atol: float = 1e-10) -> bool:
    """Compare saved/recomputed metrics while allowing JSON float round trips."""
    keys = {
        "accuracy",
        "positive_f1",
        "precision",
        "recall",
        "macro_f1",
        "negative_f1",
        "auroc",
        "support_negative",
        "support_positive",
        "num_subjects",
        "tie_count",
        "invalid_subjects",
        "confusion_matrix",
    }
    for key in keys:
        if key not in expected or key not in observed:
            return False
        left, right = expected[key], observed[key]
        if isinstance(left, (float, int)) and isinstance(right, (float, int)):
            if not math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=atol):
                return False
        elif left != right:
            return False
    return True
