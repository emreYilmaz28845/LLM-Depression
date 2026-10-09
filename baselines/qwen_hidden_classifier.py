from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.aggregate import (
    aggregate_binary_classifier_predictions,
    aggregate_binary_classifier_response_rows,
)
from src.features.hidden_classifier_policy import (
    PACKED30_AGGREGATION_POLICY,
    TURKISH_POOLED_TEXT_PAIR_POLICY,
    cache_identity,
    canonical_sha256,
    classifier_aggregation_policy,
    is_packed30_rows,
    response_normalized_sample_weights,
)
from src.sampling import (
    SAMPLING_MODE_NONE,
    SAMPLING_MODE_SUBJECT_OVERSAMPLE,
    build_no_sampling_audit,
    build_subject_oversampling,
)
from src.utils import read_json, read_jsonl, save_json, write_jsonl


PRIMARY_VARIANTS = ("logreg_raw", "xgb_raw", "xgb_pca32", "logreg_pca32", "xgb_pca64")
CONTROL_VARIANTS = ("majority_class", "xgb_raw_shuffled_labels")
LEGACY_SAMPLING_MODE = "legacy"
FIXED_RESULT_SCHEMA_VERSION = "qwen_hidden_fixed_classifier.v2"

QWEN_PREDICTION_BACKEND = "qwen_hidden_classifier"
QWEN_LOGREG_PREDICTION_BACKEND = "qwen_hidden_logreg_raw"
QWEN_XGB_PREDICTION_BACKEND = "qwen_hidden_xgb_raw"
GEMMA4_LOGREG_PREDICTION_BACKEND = "gemma4_hidden_logreg_raw"
GEMMA4_XGB_PREDICTION_BACKEND = "gemma4_hidden_xgb_raw"
GEMMA4_VARIANTS = ("logreg_raw", "xgb_raw")
OFFICIALDEV_PROTOCOL = "daic_official_train_inner_split_dev_evaluation"


def resolve_prediction_backend(
    metadata: dict[str, Any], variant: str, *, protocol_backend_mode: str | None = None
) -> str:
    """Map the extraction backend and classifier variant to the exact
    prediction-backend identity written into every prediction, metric, and
    evaluation record."""
    if str(metadata.get("model_backend", "")).strip().lower() != "gemma4":
        if protocol_backend_mode == "native_en_text_heads_v2":
            if variant == "logreg_raw":
                return QWEN_LOGREG_PREDICTION_BACKEND
            if variant == "xgb_raw":
                return QWEN_XGB_PREDICTION_BACKEND
        if (
            str(metadata.get("evaluation_provenance", {}).get("evaluation_protocol", ""))
            == OFFICIALDEV_PROTOCOL
        ):
            if variant not in GEMMA4_VARIANTS:
                raise ValueError(
                    f"Official-development DAIC campaign accepts only "
                    f"{sorted(GEMMA4_VARIANTS)}, got {variant!r}."
                )
            if variant == "logreg_raw":
                return QWEN_LOGREG_PREDICTION_BACKEND
            return QWEN_XGB_PREDICTION_BACKEND
        return QWEN_PREDICTION_BACKEND
    if variant not in GEMMA4_VARIANTS:
        raise ValueError(
            f"Gemma fixed-head campaign accepts only {sorted(GEMMA4_VARIANTS)}, "
            f"got {variant!r}."
        )
    if variant == "logreg_raw":
        return GEMMA4_LOGREG_PREDICTION_BACKEND
    return GEMMA4_XGB_PREDICTION_BACKEND


def _load_partition(cache_dir: Path, name: str) -> tuple[np.ndarray, list[dict[str, Any]]]:
    with np.load(cache_dir / f"{name}.npz") as payload:
        vectors = np.asarray(payload["vectors"], dtype=np.float32)
    rows = read_jsonl(cache_dir / f"{name}_rows.jsonl")
    if vectors.ndim != 2 or vectors.shape[0] != len(rows):
        raise ValueError(f"Invalid {name} cache shape {vectors.shape} for {len(rows)} rows.")
    if not bool(np.isfinite(vectors).all()):
        raise ValueError(f"{name} cache contains non-finite values.")
    sample_ids = [str(row["sample_id"]) for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"Duplicate sample IDs in {name} cache.")
    return vectors, rows


def _load_authoritative_split(cache_dir: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    """Load and hash-verify the parent fit's authoritative split for mask binding.

    The extraction metadata records the exact ``saved_split`` path (the fit's
    ``logs/split_used.json``) and its sha256. A train mask may only be applied
    when that split is present and unmodified: the masked subjects must equal
    the split's ``train_subject_ids`` and the head's extra pool subjects must
    be exactly the split's ``val_inner_subject_ids``.
    """
    saved = str(metadata.get("saved_split") or "").strip()
    recorded_sha = str(metadata.get("saved_split_sha256") or "").strip()
    if not saved or not recorded_sha:
        raise ValueError(
            "train mask requires the authoritative parent split; extraction "
            "metadata has no saved_split/saved_split_sha256"
        )
    path = Path(saved)
    if not path.is_file():
        raise ValueError(f"authoritative parent split is missing: {saved}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != recorded_sha:
        raise ValueError(
            "authoritative parent split hash mismatch: "
            f"recorded {recorded_sha!r}, actual {actual!r}"
        )
    split = read_json(path)
    if not isinstance(split, dict):
        raise ValueError(f"authoritative parent split is not an object: {saved}")
    return split


def _validate_train_mask(
    mask: dict[str, Any], expected_selection_sha256: str | None = None
) -> dict[str, Any]:
    """Fail-closed validation of a window-cap mask artifact before fitting.

    The recorded ``selection_sha256`` is never trusted on its own: the canonical
    hash is recomputed from ``algorithm_version``, ``sampling_seed``,
    ``fraction`` and ``subjects`` with the training definition
    (``src.data.window_cap.compute_selection_sha256``), so a mutated membership
    payload cannot keep a stale accepted hash. Schema and algorithm identity
    must match, sample ids must be unique, and the expected hash (recorded in
    run_config) must equal both the recorded and recomputed values.
    """
    if not isinstance(mask, dict):
        raise ValueError("train mask must be a JSON object")
    from src.data.window_cap import (
        ALGORITHM_VERSION,
        MASK_SCHEMA_VERSION,
        compute_selection_sha256,
    )

    schema_version = str(mask.get("schema_version") or "")
    if schema_version != MASK_SCHEMA_VERSION:
        raise ValueError(
            f"train mask schema_version {schema_version!r} != {MASK_SCHEMA_VERSION!r}"
        )
    algorithm_version = str(mask.get("algorithm_version") or "")
    if algorithm_version != ALGORITHM_VERSION:
        raise ValueError(
            f"train mask algorithm_version {algorithm_version!r} != {ALGORITHM_VERSION!r}"
        )
    fraction = mask.get("fraction")
    if (
        isinstance(fraction, bool)
        or not isinstance(fraction, (int, float))
        or not 0.0 < float(fraction) <= 1.0
    ):
        raise ValueError(f"train mask fraction is invalid: {fraction!r}")
    sampling_seed = mask.get("sampling_seed")
    if isinstance(sampling_seed, bool) or not isinstance(sampling_seed, int):
        raise ValueError(f"train mask sampling_seed is invalid: {sampling_seed!r}")
    subjects = mask.get("subjects")
    if not isinstance(subjects, dict) or not subjects:
        raise ValueError("train mask has no subject selections")
    seen: set[str] = set()
    for subject_id, ids in subjects.items():
        if not isinstance(ids, list) or not ids:
            raise ValueError(f"train mask subject {subject_id!r} has no sample ids")
        for item in ids:
            if not isinstance(item, str) or not item:
                raise ValueError(
                    f"train mask subject {subject_id!r} has a non-string sample id: {item!r}"
                )
            if item in seen:
                raise ValueError(f"train mask lists sample id {item!r} more than once")
            seen.add(item)
    recomputed = compute_selection_sha256(
        algorithm_version, sampling_seed, float(fraction), subjects
    )
    recorded = str(mask.get("selection_sha256") or "")
    if recomputed != recorded:
        raise ValueError(
            "train mask selection_sha256 does not match its membership: "
            f"recorded {recorded!r}, recomputed {recomputed!r}"
        )
    if expected_selection_sha256 is not None:
        expected = str(expected_selection_sha256)
        if recomputed != expected:
            raise ValueError(
                f"train mask selection_sha256 {recomputed!r} != expected {expected!r} "
                "from run_config"
            )
    return mask


def _apply_train_mask(
    train_x: np.ndarray,
    train_rows: list[dict[str, Any]],
    mask: dict[str, Any],
    *,
    authoritative_split: dict[str, Any],
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    """Filter outer_train rows to the capped membership; evaluation stays full.

    The mask artifact is the training-time ``window_cap_mask.json`` built over
    the fit's authoritative training pool (the parent split's
    ``train_subject_ids``). The head's ``outer_train`` partition is
    train + inner-val, so before any filtering this verifies, fail-closed:

    * the outer_train subjects are exactly the authoritative
      train + ``val_inner_subject_ids`` set;
    * the masked subjects equal the authoritative training split exactly
      (a truncated or extended mask is refused, not just hash-checked);
    * the recorded baseline membership hash matches the cache rows restricted
      to the masked subjects (the training pool);
    * every selected sample id exists in the cache, belongs to its declared
      subject, and ids are unique.

    Rows outside the masked subjects are the authoritative inner-val pool and
    are dropped by the filter; the final_eval partition is never touched.
    """
    baseline_sha = str(mask.get("baseline_input_sha256") or "")
    if not baseline_sha:
        raise ValueError(
            "train mask has no baseline_input_sha256; cannot verify the exact "
            "training-pool subject/sample membership"
        )
    subjects = mask.get("subjects")
    if not isinstance(subjects, dict) or not subjects:
        raise ValueError("train mask has no subject selections")
    mask_subjects = {str(subject) for subject in subjects}
    train_split_subjects = {
        str(subject) for subject in (authoritative_split.get("train_subject_ids") or [])
    }
    val_split_subjects = {
        str(subject) for subject in (authoritative_split.get("val_inner_subject_ids") or [])
    }
    if not train_split_subjects:
        raise ValueError("authoritative parent split has no train_subject_ids")
    row_subjects = {str(row["subject_id"]) for row in train_rows}
    expected_pool = train_split_subjects | val_split_subjects
    if row_subjects != expected_pool:
        raise ValueError(
            "outer_train subjects are not exactly the authoritative train+inner-val "
            f"set; only_in_cache={sorted(row_subjects - expected_pool)[:5]} "
            f"missing_from_cache={sorted(expected_pool - row_subjects)[:5]}"
        )
    if mask_subjects != train_split_subjects:
        raise ValueError(
            "train mask subjects do not equal the authoritative training split; "
            f"missing_training_subjects={sorted(train_split_subjects - mask_subjects)[:5]} "
            f"extra_mask_subjects={sorted(mask_subjects - train_split_subjects)[:5]}"
        )
    training_rows = [
        row for row in train_rows if str(row["subject_id"]) in mask_subjects
    ]
    canonical_membership = sorted(
        [str(row["subject_id"]), str(row["sample_id"])] for row in training_rows
    )
    recomputed_baseline = hashlib.sha256(
        json.dumps(canonical_membership, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if recomputed_baseline != baseline_sha:
        raise ValueError(
            "train mask baseline_input_sha256 does not match the training-pool "
            f"subject/sample membership: recorded {baseline_sha!r}, "
            f"recomputed {recomputed_baseline!r}. Diagnose any intentional cache "
            "membership difference before changing policy."
        )
    row_ids = [str(row["sample_id"]) for row in train_rows]
    row_owner = {sample_id: str(row["subject_id"]) for sample_id, row in zip(row_ids, train_rows)}
    selected: set[str] = set()
    for subject_id, ids in subjects.items():
        if not isinstance(ids, list) or not ids:
            raise ValueError("every masked subject needs at least one sample id")
        for item in ids:
            sample_id = str(item)
            if sample_id in selected:
                raise ValueError(f"train mask lists sample id {sample_id!r} more than once")
            owner = row_owner.get(sample_id)
            if owner is None:
                raise ValueError(f"train mask sample id {sample_id!r} is missing from outer_train")
            if owner != str(subject_id):
                raise ValueError(
                    f"train mask assigns sample {sample_id!r} to subject {subject_id!r} "
                    f"but the outer_train cache has it under {owner!r}"
                )
            selected.add(sample_id)
    keep = [index for index, sample_id in enumerate(row_ids) if sample_id in selected]
    filtered_rows = [train_rows[index] for index in keep]
    metadata = {
        "selection_sha256": mask.get("selection_sha256"),
        "baseline_input_sha256": mask.get("baseline_input_sha256"),
        "fraction": mask.get("fraction"),
        "sampling_seed": mask.get("sampling_seed"),
        "algorithm_version": mask.get("algorithm_version"),
        "available_rows": len(training_rows),
        "selected_rows": len(filtered_rows),
        "selected_subject_count": len(mask_subjects),
    }
    return train_x[keep], filtered_rows, metadata


def _variant_pipeline(variant: str, seed: int, *, unweighted: bool = False):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pca_components = 32 if "pca32" in variant else 64 if "pca64" in variant else None
    if variant.startswith("xgb"):
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise RuntimeError("XGBoost variants require xgboost==2.1.4.") from exc
        estimator = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=300,
            learning_rate=0.03,
            max_depth=2,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.25,
            reg_alpha=1.0,
            reg_lambda=10.0,
            scale_pos_weight=1.0,
            tree_method="hist",
            random_state=seed,
            n_jobs=1,
        )
        steps = []
        if pca_components:
            steps.append(("pca", PCA(n_components=pca_components, svd_solver="full")))
        steps.append(("classifier", estimator))
        return Pipeline(steps), pca_components
    if variant.startswith("logreg"):
        steps = []
        if pca_components:
            steps.append(("pca", PCA(n_components=pca_components, svd_solver="full")))
        steps.extend(
            [
                ("scale", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=1.0,
                        class_weight=None if unweighted else "balanced",
                        max_iter=5000,
                        random_state=seed,
                        solver="liblinear",
                    ),
                ),
            ]
        )
        return Pipeline(steps), pca_components
    raise ValueError(f"Unsupported fitted variant: {variant}")


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value) if isinstance(value, (list, dict)) else value
                    for key, value in row.items()
                }
            )


def _metrics_with_negative_f1(metrics: dict[str, Any]) -> dict[str, Any]:
    tn, fp = metrics["confusion_matrix"][0]
    fn, _ = metrics["confusion_matrix"][1]
    precision = tn / (tn + fn) if tn + fn else 0.0
    recall = tn / (tn + fp) if tn + fp else 0.0
    output = dict(metrics)
    output["negative_f1"] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return output


def shuffled_subject_labels(rows: list[dict[str, Any]], seed: int) -> np.ndarray:
    """Permute labels between subjects while preserving response groups."""
    labels_by_subject: dict[str, int] = {}
    for row in rows:
        subject_id = str(row["subject_id"])
        label = int(row["label"])
        if subject_id in labels_by_subject and labels_by_subject[subject_id] != label:
            raise ValueError(f"Subject {subject_id} has inconsistent training labels.")
        labels_by_subject[subject_id] = label
    subject_ids = sorted(labels_by_subject)
    shuffled = np.random.default_rng(seed).permutation(
        [labels_by_subject[subject_id] for subject_id in subject_ids]
    )
    shuffled_by_subject = dict(zip(subject_ids, shuffled.tolist()))
    return np.asarray(
        [shuffled_by_subject[str(row["subject_id"])] for row in rows],
        dtype=np.int64,
    )


def majority_subject_control(rows: list[dict[str, Any]]) -> tuple[int, float]:
    labels_by_subject: dict[str, int] = {}
    for row in rows:
        subject_id = str(row["subject_id"])
        label = int(row["label"])
        if subject_id in labels_by_subject and labels_by_subject[subject_id] != label:
            raise ValueError(f"Subject {subject_id} has inconsistent training labels.")
        labels_by_subject[subject_id] = label
    labels = np.asarray(list(labels_by_subject.values()), dtype=np.int64)
    if not len(labels):
        raise ValueError("Majority control requires at least one training subject.")
    return int(np.bincount(labels, minlength=2).argmax()), float(labels.mean())


def _enforce_gemma_daic_contract(
    metadata: dict[str, Any],
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    train_subjects: set[str],
) -> None:
    """Enforce the fixed Gemma DAIC scientific contract on the extracted cache.

    Refuses non-finite vectors, incomplete chunk coverage, unequal
    subject-normalized fit weights, and wrong official train/test subject
    counts. Production must fit on exactly 107 official training subjects and
    evaluate exactly 47 official test subjects; a count mismatch is a hard
    stop, never a row-drop. Isolated smokes carry ``subject_selection_sha256``
    in their cache identity and are exempt from the exact production counts
    only.
    """
    smoke_cache = bool(metadata.get("cache_config", {}).get("subject_selection_sha256"))
    if not smoke_cache:
        if len(train_subjects) != 107:
            raise ValueError(
                f"Gemma DAIC fit requires exactly 107 official training subjects, "
                f"got {len(train_subjects)}. Do not fit on validation subjects."
            )
        test_subjects = {str(row["subject_id"]) for row in test_rows}
        if len(test_subjects) != 47:
            raise ValueError(
                f"Gemma DAIC evaluation requires exactly 47 official test subjects, "
                f"got {len(test_subjects)}."
            )
    modality = str(metadata.get("input_modality", ""))
    packed30 = modality in {"audio_only", "audio_text"}
    if packed30:
        _enforce_complete_chunk_coverage(train_rows, "outer_train")
        _enforce_complete_chunk_coverage(test_rows, "final_eval")


def _enforce_officialdev_contract(
    metadata: dict[str, Any],
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    train_subjects: set[str],
) -> None:
    """Enforce the DAIC official-development fixed-head contract.

    The fit set must be exactly the 86 inner-training subjects and the
    evaluation set exactly the 35 official development subjects, for either
    backbone. Count mismatches are hard stops, never row-drops. Isolated
    smokes are exempt from the exact production counts only.
    """
    smoke_cache = bool(metadata.get("cache_config", {}).get("subject_selection_sha256"))
    if not smoke_cache:
        if len(train_subjects) != 86:
            raise ValueError(
                "Official-development DAIC fit requires exactly 86 inner-training "
                f"subjects, got {len(train_subjects)}. Do not fit on selection or "
                "development subjects."
            )
        test_subjects = {str(row["subject_id"]) for row in test_rows}
        if len(test_subjects) != 35:
            raise ValueError(
                "Official-development DAIC evaluation requires exactly 35 "
                f"official development subjects, got {len(test_subjects)}."
            )
    modality = str(metadata.get("input_modality", ""))
    if modality in {"audio_only", "audio_text"}:
        _enforce_complete_chunk_coverage(train_rows, "outer_train")
        _enforce_complete_chunk_coverage(test_rows, "final_eval")


def _enforce_gemma_dependency_versions() -> None:
    """Require the locked classifier library versions on the Gemma path.

    Reads the module ``__version__`` attributes: the hidden dependency
    directory is a ``pip --target`` install whose dist-info metadata is not
    always visible to ``importlib.metadata``.
    """
    try:
        import sklearn
        import xgboost
    except Exception as error:
        raise RuntimeError(
            "Gemma fixed heads require scikit-learn 1.7.0 and xgboost 2.1.4."
        ) from error
    sklearn_version = str(getattr(sklearn, "__version__", ""))
    xgboost_version = str(getattr(xgboost, "__version__", ""))
    if sklearn_version != "1.7.0" or xgboost_version != "2.1.4":
        raise RuntimeError(
            f"Gemma fixed heads require scikit-learn 1.7.0 and xgboost 2.1.4; "
            f"got scikit-learn {sklearn_version} and xgboost {xgboost_version}."
        )


def _enforce_complete_chunk_coverage(
    rows: list[dict[str, Any]], partition: str
) -> None:
    """Refuse a packed30 partition with missing or duplicated chunk indices."""
    from collections import Counter, defaultdict

    by_subject: dict[str, list[int]] = defaultdict(list)
    declared_by_subject: dict[str, int] = {}
    for row in rows:
        subject_id = str(row["subject_id"])
        chunk_index = row.get("chunk_index")
        num_chunks = row.get("num_chunks")
        if chunk_index is None or num_chunks is None:
            raise ValueError(
                f"Gemma packed30 {partition} rows require chunk_index and "
                f"num_chunks; sample_id={row.get('sample_id')}."
            )
        by_subject[subject_id].append(int(chunk_index))
        declared = int(num_chunks)
        if (
            subject_id in declared_by_subject
            and declared_by_subject[subject_id] != declared
        ):
            raise ValueError(
                f"Gemma packed30 {partition} subject {subject_id} has "
                f"inconsistent num_chunks declarations."
            )
        declared_by_subject[subject_id] = declared
    for subject_id, indices in sorted(by_subject.items()):
        counts = Counter(indices)
        duplicates = [index for index, count in counts.items() if count > 1]
        if duplicates:
            raise ValueError(
                f"Gemma packed30 {partition} subject {subject_id} has duplicate "
                f"chunk indices: {duplicates[:10]}."
            )
        expected = set(range(declared_by_subject[subject_id]))
        missing = sorted(expected - set(indices))
        if missing:
            raise ValueError(
                f"Gemma packed30 {partition} subject {subject_id} is missing "
                f"chunks: {missing[:10]}."
            )


def run_variant(
    cache_dir: Path,
    output_root: Path,
    variant: str,
    seed: int,
    sampling_mode: str = LEGACY_SAMPLING_MODE,
    oversampling_ratio: float | None = None,
    oversampling_seed: int = 1337,
    protocol_backend_mode: str | None = None,
    train_mask: dict[str, Any] | None = None,
) -> dict[str, Any]:
    train_x, train_rows = _load_partition(cache_dir, "outer_train")
    test_x, test_rows = _load_partition(cache_dir, "final_eval")
    train_subjects = {str(row["subject_id"]) for row in train_rows}
    test_subjects = {str(row["subject_id"]) for row in test_rows}
    overlap = sorted(train_subjects & test_subjects)
    if overlap:
        raise ValueError(f"Training/held-out subject leakage: {overlap[:10]}")
    train_y = np.asarray([int(row["label"]) for row in train_rows], dtype=np.int64)
    test_y = np.asarray([int(row["label"]) for row in test_rows], dtype=np.int64)
    if set(train_y.tolist()) != {0, 1}:
        raise ValueError("Training cache must contain both classes.")
    metadata = read_json(cache_dir / "extraction_metadata.json")
    prediction_backend = resolve_prediction_backend(
        metadata, variant, protocol_backend_mode=protocol_backend_mode
    )
    evaluation_protocol = str(
        metadata.get("evaluation_provenance", {}).get("evaluation_protocol", "")
    )
    if evaluation_protocol == OFFICIALDEV_PROTOCOL:
        if str(metadata.get("model_backend", "")).strip().lower() == "gemma4":
            _enforce_gemma_dependency_versions()
        _enforce_officialdev_contract(metadata, train_rows, test_rows, train_subjects)
    elif str(metadata.get("model_backend", "")).strip().lower() == "gemma4":
        _enforce_gemma_dependency_versions()
        if evaluation_protocol == "daic_official_train_fit_locked_test_evaluation":
            # The exact official-train/official-test contract is DAIC-specific.
            _enforce_gemma_daic_contract(metadata, train_rows, test_rows, train_subjects)
        # Harmonized (non-DAIC) Gemma caches enforce the generic invariants
        # already proven above: both classes, disjoint fit/holdout subjects,
        # finite vectors, equal subject totals via the weight audit below.
    train_mask_metadata: dict[str, Any] | None = None
    if train_mask is not None:
        _validate_train_mask(train_mask)
        authoritative_split = _load_authoritative_split(cache_dir, metadata)
        train_x, train_rows, train_mask_metadata = _apply_train_mask(
            train_x, train_rows, train_mask, authoritative_split=authoritative_split
        )
        train_y = np.asarray([int(row["label"]) for row in train_rows], dtype=np.int64)
        train_subjects = {str(row["subject_id"]) for row in train_rows}
        if set(train_y.tolist()) != {0, 1}:
            raise ValueError("Capped training rows must contain both classes.")
        overlap = sorted(train_subjects & test_subjects)
        if overlap:
            raise ValueError(f"Training/held-out subject leakage after mask: {overlap[:10]}")
    result_identity = {
        "schema_version": FIXED_RESULT_SCHEMA_VERSION,
        "variant": variant,
        "seed": int(seed),
        "sampling_mode": sampling_mode,
        "oversampling_ratio": oversampling_ratio,
        "oversampling_seed": int(oversampling_seed),
        "model_backend": metadata.get("model_backend"),
        "prediction_backend": prediction_backend,
        "cache_identity": cache_identity(cache_dir),
        "aggregation_policy": classifier_aggregation_policy(metadata),
    }
    if train_mask_metadata is not None:
        result_identity["train_mask"] = train_mask_metadata
    result_identity["config_sha256"] = canonical_sha256(result_identity)
    variant_dir = output_root / variant
    identity_path = variant_dir / "result_config.json"
    required = {
        "predictions_sample_level.jsonl",
        "predictions_sample_level.csv",
        "predictions_subject_level.jsonl",
        "predictions_subject_level.csv",
        "metrics.json",
        "classifier_metadata.json",
        "sampling_audit.json",
    }
    if variant != "majority_class":
        required.add("pipeline.joblib")
    if (
        str(metadata.get("dataset", "")).lower() == "d3tec"
        and metadata.get("input_modality") != "text_only"
    ):
        required.update(
            {
                "predictions_response_level.jsonl",
                "predictions_response_level.csv",
                "metrics_response_level.json",
            }
        )
    if variant_dir.exists() and any(variant_dir.iterdir()):
        if not identity_path.is_file():
            raise ValueError(
                f"Non-empty fixed-head output has no result_config.json: {variant_dir}. "
                "Refusing to overwrite partial or legacy output."
            )
        if read_json(identity_path) != result_identity:
            raise ValueError(f"Existing fixed-head output is incompatible: {variant_dir}.")
        if not all((variant_dir / name).is_file() for name in required):
            raise ValueError(f"Existing fixed-head output is partial: {variant_dir}.")
        return {"variant": variant, **read_json(variant_dir / "metrics.json")}
    variant_dir.mkdir(parents=True, exist_ok=True)
    save_json(result_identity, identity_path)
    requested_components = None
    effective_components = None
    fit_y = train_y.copy()
    if sampling_mode == SAMPLING_MODE_SUBJECT_OVERSAMPLE:
        sampling = build_subject_oversampling(
            train_rows,
            ratio=oversampling_ratio,
            seed=oversampling_seed,
            expected_minority_label=0,
            evaluation_rows=test_rows,
        )
    elif sampling_mode in {SAMPLING_MODE_NONE, LEGACY_SAMPLING_MODE}:
        sampling = build_no_sampling_audit(
            train_rows,
            seed=oversampling_seed,
            evaluation_rows=test_rows,
        )
    else:
        raise ValueError(f"Unsupported sampling_mode {sampling_mode!r}.")
    fit_indices = np.asarray(sampling.indices, dtype=np.int64)
    if sampling_mode != LEGACY_SAMPLING_MODE:
        sampling_identity = {
            "schema_version": "fixed_hidden_sampling_identity.v1",
            "sampling_mode": sampling_mode,
            "oversampling_ratio": oversampling_ratio,
            "oversampling_seed": int(oversampling_seed),
            "source_row_assignments_sha256": sampling.audit[
                "source_row_assignments_sha256"
            ],
            "source_subject_assignments_sha256": sampling.audit[
                "source_subject_assignments_sha256"
            ],
        }
        identity_path = output_root / "sampling_config.json"
        if identity_path.exists() and read_json(identity_path) != sampling_identity:
            raise ValueError("Existing fixed-head sampling configuration is incompatible.")
        save_json(sampling_identity, identity_path)
    if variant == "majority_class":
        prediction, probability = majority_subject_control(train_rows)
        probabilities = np.full(test_y.shape, probability, dtype=np.float64)
        predictions = np.full(test_y.shape, prediction, dtype=np.int64)
        fit_weight_audit = response_normalized_sample_weights(
            [train_rows[index] for index in fit_indices.tolist()],
            metadata,
        )[1]
    else:
        fitted_variant = "xgb_raw" if variant == "xgb_raw_shuffled_labels" else variant
        fitted, requested_components = _variant_pipeline(
            fitted_variant,
            seed,
            unweighted=sampling_mode != LEGACY_SAMPLING_MODE,
        )
        if requested_components:
            matrix_limit = min(train_x.shape)
            if requested_components > matrix_limit:
                raise ValueError(
                    f"PCA-{requested_components} exceeds training matrix limit {matrix_limit}."
                )
            effective_components = requested_components
        if variant == "xgb_raw_shuffled_labels":
            fit_y = shuffled_subject_labels(train_rows, seed)
        fit_rows = [train_rows[index] for index in fit_indices.tolist()]
        fit_weights, fit_weight_audit = response_normalized_sample_weights(
            fit_rows,
            metadata,
        )
        fitted.fit(
            train_x[fit_indices],
            fit_y[fit_indices],
            classifier__sample_weight=fit_weights,
        )
        probabilities = np.asarray(fitted.predict_proba(test_x)[:, 1], dtype=np.float64)
        predictions = (probabilities >= 0.5).astype(np.int64)
        import joblib

        joblib.dump(fitted, variant_dir / "pipeline.joblib")
    condition = str(metadata.get("condition") or metadata["input_modality"])
    packed30 = is_packed30_rows(metadata)
    sample_rows = []
    for row, probability, prediction in zip(test_rows, probabilities.tolist(), predictions.tolist()):
        classifier_row = {
            "dataset": metadata["dataset"],
            "modality": metadata["input_modality"],
            "condition": condition,
            "dataset_variant": metadata.get("dataset_variant", ""),
            "fold": int(metadata["fold"]),
            "sample_id": str(row["sample_id"]),
            "subject_id": str(row["subject_id"]),
            **{
                key: row[key]
                for key in ("response_id", "prompt_id", "segment_index", "num_segments")
                if key in row
            },
            "label": int(row["label"]),
            "probability": float(probability),
            "predicted_class": int(prediction),
            "checkpoint": metadata["checkpoint_dir"],
            "classifier_variant": variant,
            "prediction_backend": prediction_backend,
            "sampling_mode": sampling_mode,
            "oversampling_ratio": oversampling_ratio,
            "oversampling_seed": int(oversampling_seed),
            "aggregation_policy": classifier_aggregation_policy(metadata),
        }
        if metadata.get("dataset_variant") == "pooled_t17":
            classifier_row["question_condition"] = str(row.get("question_condition", ""))
            if metadata.get("input_modality") == "text_only":
                classifier_row["aggregation_policy"] = TURKISH_POOLED_TEXT_PAIR_POLICY
        if packed30:
            classifier_row["protocol_id"] = str(metadata.get("protocol_id", ""))
            classifier_row["classifier_aggregation"] = PACKED30_AGGREGATION_POLICY
        sample_rows.append(classifier_row)
    subject_rows, metrics = aggregate_binary_classifier_predictions(
        sample_rows, prediction_backend=prediction_backend
    )
    for row in subject_rows:
        row.update(
            {
                "dataset": metadata["dataset"],
                "modality": metadata["input_modality"],
                "condition": condition,
                "fold": int(metadata["fold"]),
                "classifier_variant": variant,
                "prediction_backend": prediction_backend,
                "sampling_mode": sampling_mode,
                "oversampling_ratio": oversampling_ratio,
                "oversampling_seed": int(oversampling_seed),
            }
        )
        if packed30:
            row["protocol_id"] = str(metadata.get("protocol_id", ""))
    metrics = _metrics_with_negative_f1(metrics)
    response_rows = []
    if str(metadata["dataset"]).lower() == "d3tec" and metadata["input_modality"] != "text_only":
        response_rows, response_metrics = aggregate_binary_classifier_response_rows(sample_rows)
        for row in response_rows:
            row.update(
                {
                    "dataset": metadata["dataset"],
                    "modality": metadata["input_modality"],
                    "condition": condition,
                    "fold": int(metadata["fold"]),
                    "classifier_variant": variant,
                }
            )
        write_jsonl(response_rows, variant_dir / "predictions_response_level.jsonl")
        _write_csv(response_rows, variant_dir / "predictions_response_level.csv")
        save_json(response_metrics, variant_dir / "metrics_response_level.json")
    artifact_metadata = {
        "dataset": metadata["dataset"],
        "modality": metadata["input_modality"],
        "condition": condition,
        "fold": int(metadata["fold"]),
        "classifier_variant": variant,
        "prediction_backend": prediction_backend,
        "seed": seed,
        "sampling_mode": sampling_mode,
        "oversampling_ratio": oversampling_ratio,
        "oversampling_seed": int(oversampling_seed),
        "sampling_audit": sampling.audit,
        "fit_weight_audit": fit_weight_audit,
        "weight_policy": fit_weight_audit["policy"],
        "aggregation_policy": classifier_aggregation_policy(metadata),
        "cache_identity": result_identity["cache_identity"],
        "cache_identity_sha256": canonical_sha256(result_identity["cache_identity"]),
        "checkpoint_hashes": {
            "adapter_config_sha256": metadata.get("adapter_config_sha256"),
            "adapter_sha256": metadata.get("adapter_sha256"),
        },
        "split_hashes": {
            "saved_split_sha256": metadata.get("saved_split_sha256"),
            "split_metadata_sha256": metadata.get("split_metadata_sha256"),
            "manifest_sha256": metadata.get("manifest_sha256"),
        },
        "parent_attempt_id": metadata.get("parent_attempt_id"),
        "threshold": 0.5,
        "input_dimension": int(train_x.shape[1]),
        "requested_pca_components": requested_components,
        "effective_pca_components": effective_components,
        "training_row_ids": [str(row["sample_id"]) for row in train_rows],
        "training_subject_ids": sorted(train_subjects),
        "heldout_subject_ids": sorted(test_subjects),
        "shuffled_training_labels": variant == "xgb_raw_shuffled_labels",
        "label_shuffle_unit": "subject" if variant == "xgb_raw_shuffled_labels" else None,
        "extraction_metadata": str(cache_dir / "extraction_metadata.json"),
        "result_config_sha256": result_identity["config_sha256"],
        "response_prediction_count": len(response_rows) if response_rows else None,
    }
    if train_mask_metadata is not None:
        artifact_metadata["train_mask"] = train_mask_metadata
    write_jsonl(sample_rows, variant_dir / "predictions_sample_level.jsonl")
    write_jsonl(subject_rows, variant_dir / "predictions_subject_level.jsonl")
    _write_csv(sample_rows, variant_dir / "predictions_sample_level.csv")
    _write_csv(subject_rows, variant_dir / "predictions_subject_level.csv")
    save_json(metrics, variant_dir / "metrics.json")
    save_json(artifact_metadata, variant_dir / "classifier_metadata.json")
    save_json(sampling.audit, variant_dir / "sampling_audit.json")
    return {"variant": variant, **metrics}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate classifiers on cached Qwen hidden vectors.")
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--variants", nargs="+", default=list(PRIMARY_VARIANTS + CONTROL_VARIANTS))
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--sampling-mode",
        choices=(
            LEGACY_SAMPLING_MODE,
            SAMPLING_MODE_NONE,
            SAMPLING_MODE_SUBJECT_OVERSAMPLE,
        ),
        default=LEGACY_SAMPLING_MODE,
    )
    parser.add_argument("--oversampling-ratio", type=float)
    parser.add_argument("--oversampling-seed", type=int, default=1337)
    parser.add_argument(
        "--protocol-backend-mode",
        choices=("native_en_text_heads_v2",),
        default=None,
        help="Use method-specific Qwen backend qualifiers for the v2 text-head study.",
    )
    parser.add_argument(
        "--train-mask",
        type=Path,
        default=None,
        help="window_cap_mask.json produced by the capped training run.",
    )
    parser.add_argument(
        "--train-mask-sha256",
        default=None,
        help="Expected selection_sha256 of --train-mask (recorded in run_config).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_mask = None
    if bool(args.train_mask) != bool(args.train_mask_sha256):
        raise SystemExit("--train-mask and --train-mask-sha256 must be provided together.")
    if args.train_mask:
        train_mask = read_json(args.train_mask)
        try:
            _validate_train_mask(train_mask, expected_selection_sha256=args.train_mask_sha256)
        except ValueError as exc:
            raise SystemExit(f"train mask rejected: {exc}") from exc
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = [
        run_variant(
            args.cache_dir,
            args.output_dir,
            variant,
            args.seed,
            sampling_mode=args.sampling_mode,
            oversampling_ratio=args.oversampling_ratio,
            oversampling_seed=args.oversampling_seed,
            protocol_backend_mode=args.protocol_backend_mode,
            train_mask=train_mask,
        )
        for variant in args.variants
    ]
    save_json(summaries, args.output_dir / "variant_summary.json")
    _write_csv(summaries, args.output_dir / "variant_summary.csv")
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
