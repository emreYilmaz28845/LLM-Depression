"""N2 subject-level mean-pooling and gated-attention MIL over frozen WavLM bags.

The module extends, but never mutates, the immutable N0/N1 protocol produced by
``src.baselines.acoustic_crossfold``.  It uses four cached 2,304-dimensional
WavLM chunk vectors per development subject, performs nested subject-level model
selection, and keeps the official test partition locked.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# Deterministic CUDA GEMMs require this before importing torch.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import soundfile as sf
import torch
from torch import nn

from src.baselines.acoustic_crossfold import (
    BOOTSTRAP_REPEATS,
    DEFAULT_EGEMAPS_CACHE,
    DEFAULT_MANIFEST,
    DEFAULT_OUTPUT,
    DEFAULT_PARTITIONS,
    DEFAULT_WAVLM_CACHE,
    METRIC_NAMES,
    PRIMARY_COMPARISON_METRICS,
    CachePaths,
    atomic_json,
    atomic_jsonl,
    dependency_versions,
    derangement,
    derived_seed,
    evaluate_binary,
    git_provenance,
    natural_key,
    read_json,
    read_jsonl,
    sha256_file,
    validate_caches_from_frozen,
    validate_oof_coverage,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL_ROOT = DEFAULT_OUTPUT
DEFAULT_N2_ROOT = DEFAULT_PROTOCOL_ROOT / "n2"

ARCHITECTURES = ("mean_pooling", "gated_attention")
INSTANCE_DIMENSION = 2304
PROJECTION_DIMENSION = 128
ATTENTION_HIDDEN_DIMENSION = 64
DROPOUT = 0.2
LEARNING_RATES = (1e-4, 3e-4)
WEIGHT_DECAYS = (0.01, 0.1)
MAX_EPOCHS = 200
PATIENCE = 20
MIN_DELTA = 1e-4
MIL_SEEDS = (20260721, 20260722, 20260723, 20260724, 20260725)
SHUFFLE_SEED = 20260726
BOOTSTRAP_SEED = 20260727
THRESHOLD = 0.5
ACTIVE_FRAME_RMS_THRESHOLD = 0.01
SILENCE_ABSOLUTE_THRESHOLD = 1e-4
CLIPPING_ABSOLUTE_THRESHOLD = 0.999
FRAME_MILLISECONDS = 25


@dataclass(frozen=True)
class N2Paths:
    base: CachePaths
    n2_root: Path


@dataclass
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        transformed = (np.asarray(values, dtype=np.float32) - self.mean) / self.scale
        if not np.isfinite(transformed).all():
            raise ValueError("Standardized WavLM bags contain non-finite values")
        return transformed.astype(np.float32, copy=False)


class InstanceProjection(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(INSTANCE_DIMENSION, PROJECTION_DIMENSION)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, bags: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.activation(self.projection(bags)))


class MeanPoolingMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.instances = InstanceProjection()
        self.head = nn.Sequential(
            nn.LayerNorm(PROJECTION_DIMENSION),
            nn.Dropout(DROPOUT),
            nn.Linear(PROJECTION_DIMENSION, 1),
        )

    def forward(self, bags: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.instances(bags)
        attention = torch.full(
            encoded.shape[:2],
            1.0 / encoded.shape[1],
            dtype=encoded.dtype,
            device=encoded.device,
        )
        logits = self.head(encoded.mean(dim=1)).squeeze(-1)
        return logits, attention


class GatedAttentionMIL(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.instances = InstanceProjection()
        self.attention_v = nn.Linear(PROJECTION_DIMENSION, ATTENTION_HIDDEN_DIMENSION)
        self.attention_u = nn.Linear(PROJECTION_DIMENSION, ATTENTION_HIDDEN_DIMENSION)
        self.attention_w = nn.Linear(ATTENTION_HIDDEN_DIMENSION, 1, bias=False)
        self.head = nn.Sequential(
            nn.LayerNorm(PROJECTION_DIMENSION),
            nn.Dropout(DROPOUT),
            nn.Linear(PROJECTION_DIMENSION, 1),
        )

    def forward(self, bags: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.instances(bags)
        gated = torch.tanh(self.attention_v(encoded)) * torch.sigmoid(self.attention_u(encoded))
        attention = torch.softmax(self.attention_w(gated).squeeze(-1), dim=1)
        pooled = torch.sum(attention.unsqueeze(-1) * encoded, dim=1)
        logits = self.head(pooled).squeeze(-1)
        return logits, attention


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def architecture_model(name: str) -> nn.Module:
    if name == "mean_pooling":
        model: nn.Module = MeanPoolingMLP()
    elif name == "gated_attention":
        model = GatedAttentionMIL()
    else:
        raise ValueError(f"Unknown N2 architecture: {name}")
    parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if parameters >= 1_000_000:
        raise ValueError(f"{name} exceeds the locked one-million-parameter ceiling: {parameters}")
    return model


def trainable_parameter_counts() -> dict[str, int]:
    return {
        name: sum(
            parameter.numel()
            for parameter in architecture_model(name).parameters()
            if parameter.requires_grad
        )
        for name in ARCHITECTURES
    }


def set_deterministic_state(seed: int) -> None:
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def fit_standardizer(bags: np.ndarray) -> Standardizer:
    values = np.asarray(bags, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (4, INSTANCE_DIMENSION):
        raise ValueError(f"Expected [subjects,4,2304] training bags, found {values.shape}")
    flat = values.reshape(-1, INSTANCE_DIMENSION)
    mean = flat.mean(axis=0).astype(np.float32)
    scale = flat.std(axis=0, ddof=0).astype(np.float32)
    scale[scale < 1e-8] = 1.0
    if not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise ValueError("Training-derived standardizer is non-finite")
    return Standardizer(mean=mean, scale=scale)


def load_bags(
    selection: Mapping[str, Any],
    wavlm_vectors: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    bags: dict[str, np.ndarray] = {}
    for subject_id, subject in selection["subjects"].items():
        samples = sorted(subject["samples"], key=lambda row: int(row["selected_position"]))
        if len(samples) != 4 or [int(row["selected_position"]) for row in samples] != [0, 1, 2, 3]:
            raise ValueError(f"Subject {subject_id} does not have the frozen ordered K=4 bag")
        try:
            bag = np.stack([wavlm_vectors[str(row["sample_id"])] for row in samples])
        except KeyError as exc:
            raise ValueError(f"Missing frozen WavLM vector for {exc.args[0]}") from exc
        if bag.shape != (4, INSTANCE_DIMENSION) or not np.isfinite(bag).all():
            raise ValueError(f"Invalid WavLM bag for subject {subject_id}: {bag.shape}")
        bags[str(subject_id)] = bag.astype(np.float32)
    return bags


def subject_metadata(selection: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(subject_id): {
            "label": int(row["label"]),
            "original_partition": str(row["original_partition"]),
            "samples": list(row["samples"]),
        }
        for subject_id, row in selection["subjects"].items()
    }


def _labels(metadata: Mapping[str, dict[str, Any]], ids: Sequence[str]) -> np.ndarray:
    return np.asarray([int(metadata[str(subject_id)]["label"]) for subject_id in ids], dtype=np.int64)


def _bag_matrix(
    bags: Mapping[str, np.ndarray],
    target_ids: Sequence[str],
    source_ids: Sequence[str] | None = None,
) -> np.ndarray:
    if source_ids is None:
        source_ids = target_ids
    if len(target_ids) != len(source_ids):
        raise ValueError("Target/source subject lists differ in length")
    values = np.stack([bags[str(subject_id)] for subject_id in source_ids]).astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Bag matrix contains non-finite values")
    return values


def _state_copy(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def train_early_stopping(
    architecture: str,
    train_bags: np.ndarray,
    train_labels: np.ndarray,
    validation_bags: np.ndarray,
    validation_labels: np.ndarray,
    *,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    set_deterministic_state(seed)
    standardizer = fit_standardizer(train_bags)
    train_x = torch.from_numpy(standardizer.transform(train_bags)).to(device)
    train_y = torch.from_numpy(train_labels.astype(np.float32)).to(device)
    validation_x = torch.from_numpy(standardizer.transform(validation_bags)).to(device)
    validation_y = torch.from_numpy(validation_labels.astype(np.float32)).to(device)
    model = architecture_model(architecture).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    criterion = nn.BCEWithLogitsLoss()
    best_loss = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    epochs_ran = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(train_x)
        loss = criterion(logits, train_y)
        if not torch.isfinite(loss):
            raise ValueError("Non-finite N2 training loss")
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.inference_mode():
            validation_logits, _ = model(validation_x)
            validation_loss = float(criterion(validation_logits, validation_y).item())
        epochs_ran = epoch
        if validation_loss < best_loss - MIN_DELTA:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = _state_copy(model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                break
    if best_state is None or best_epoch < 1:
        raise RuntimeError("Early stopping never recorded a finite validation state")
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        logits, attention = model(validation_x)
        probability = torch.sigmoid(logits).cpu().numpy().astype(np.float64)
    return {
        "probabilities": probability,
        "attention": attention.cpu().numpy().astype(np.float64),
        "best_epoch": best_epoch,
        "epochs_ran": epochs_ran,
        "best_validation_log_loss": best_loss,
    }


def train_fixed_epochs(
    architecture: str,
    train_bags: np.ndarray,
    train_labels: np.ndarray,
    evaluation_bags: np.ndarray,
    *,
    learning_rate: float,
    weight_decay: float,
    epochs: int,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    if epochs < 1 or epochs > MAX_EPOCHS:
        raise ValueError(f"Final refit epochs must be in [1,{MAX_EPOCHS}], found {epochs}")
    set_deterministic_state(seed)
    standardizer = fit_standardizer(train_bags)
    train_x = torch.from_numpy(standardizer.transform(train_bags)).to(device)
    train_y = torch.from_numpy(train_labels.astype(np.float32)).to(device)
    evaluation_x = torch.from_numpy(standardizer.transform(evaluation_bags)).to(device)
    model = architecture_model(architecture).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    criterion = nn.BCEWithLogitsLoss()
    final_loss = math.nan
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(train_x)
        loss = criterion(logits, train_y)
        if not torch.isfinite(loss):
            raise ValueError("Non-finite N2 final-refit loss")
        loss.backward()
        optimizer.step()
        final_loss = float(loss.item())
    model.eval()
    with torch.inference_mode():
        logits, attention = model(evaluation_x)
        probability = torch.sigmoid(logits).cpu().numpy().astype(np.float64)
    return {
        "probabilities": probability,
        "attention": attention.cpu().numpy().astype(np.float64),
        "final_training_loss": final_loss,
        "standardizer_mean_sha256": hashlib.sha256(standardizer.mean.tobytes()).hexdigest(),
        "standardizer_scale_sha256": hashlib.sha256(standardizer.scale.tobytes()).hexdigest(),
    }


def hyperparameter_grid() -> list[dict[str, float]]:
    return [
        {"learning_rate": learning_rate, "weight_decay": weight_decay}
        for learning_rate in LEARNING_RATES
        for weight_decay in WEIGHT_DECAYS
    ]


def select_hyperparameters(
    architecture: str,
    bags: Mapping[str, np.ndarray],
    metadata: Mapping[str, dict[str, Any]],
    outer: Mapping[str, Any],
    *,
    mil_seed: int,
    device: torch.device,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    outer_fold = int(outer["outer_fold"])
    outer_train_ids = set(str(value) for value in outer["train_subject_ids"])
    for candidate in hyperparameter_grid():
        rows: list[tuple[str, int, float]] = []
        inner_records: list[dict[str, Any]] = []
        for inner in outer["inner_folds"]:
            inner_fold = int(inner["inner_fold"])
            train_ids = [str(value) for value in inner["train_subject_ids"]]
            validation_ids = [str(value) for value in inner["validation_subject_ids"]]
            result = train_early_stopping(
                architecture,
                _bag_matrix(bags, train_ids),
                _labels(metadata, train_ids),
                _bag_matrix(bags, validation_ids),
                _labels(metadata, validation_ids),
                learning_rate=candidate["learning_rate"],
                weight_decay=candidate["weight_decay"],
                # Hold initialization/dropout randomness fixed across the four
                # hyperparameter candidates within this seed/fold comparison.
                seed=derived_seed(mil_seed, outer_fold, inner_fold, 1),
                device=device,
            )
            rows.extend(
                (subject_id, int(label), float(probability))
                for subject_id, label, probability in zip(
                    validation_ids,
                    _labels(metadata, validation_ids),
                    result["probabilities"],
                )
            )
            inner_records.append(
                {
                    "inner_fold": inner_fold,
                    "best_epoch": result["best_epoch"],
                    "epochs_ran": result["epochs_ran"],
                    "best_validation_log_loss": result["best_validation_log_loss"],
                }
            )
        validate_oof_coverage(
            rows,
            expected_subject_ids=outer_train_ids,
            id_index=0,
            context=f"N2 {architecture} inner OOF outer={outer_fold} seed={mil_seed}",
        )
        rows.sort(key=lambda row: natural_key(row[0]))
        metrics = evaluate_binary([row[1] for row in rows], [row[2] for row in rows])
        epochs = [int(row["best_epoch"]) for row in inner_records]
        final_epochs = int(math.floor(float(np.median(epochs)) + 0.5))
        candidates.append(
            {
                **candidate,
                "pooled_inner_oof_metrics": metrics,
                "final_refit_epochs": final_epochs,
                "inner_folds": inner_records,
            }
        )
    selected = min(
        candidates,
        key=lambda row: (
            float(row["pooled_inner_oof_metrics"]["log_loss"]),
            float(row["learning_rate"]),
            -float(row["weight_decay"]),
        ),
    )
    return {"selected": selected, "candidates": candidates}


def fit_outer_real_and_shuffle(
    architecture: str,
    bags: Mapping[str, np.ndarray],
    metadata: Mapping[str, dict[str, Any]],
    outer: Mapping[str, Any],
    *,
    mil_seed: int,
    seed_index: int,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    outer_fold = int(outer["outer_fold"])
    selection = select_hyperparameters(
        architecture,
        bags,
        metadata,
        outer,
        mil_seed=mil_seed,
        device=device,
    )
    selected = selection["selected"]
    train_ids = [str(value) for value in outer["train_subject_ids"]]
    holdout_ids = [str(value) for value in outer["holdout_subject_ids"]]
    train_y = _labels(metadata, train_ids)
    holdout_y = _labels(metadata, holdout_ids)
    final_seed = derived_seed(mil_seed, outer_fold, 1000)
    real = train_fixed_epochs(
        architecture,
        _bag_matrix(bags, train_ids),
        train_y,
        _bag_matrix(bags, holdout_ids),
        learning_rate=float(selected["learning_rate"]),
        weight_decay=float(selected["weight_decay"]),
        epochs=int(selected["final_refit_epochs"]),
        seed=final_seed,
        device=device,
    )
    train_map = derangement(
        train_ids,
        derived_seed(SHUFFLE_SEED, seed_index, outer_fold, 0),
    )
    holdout_map = derangement(
        holdout_ids,
        derived_seed(SHUFFLE_SEED, seed_index, outer_fold, 1),
    )
    shuffled = train_fixed_epochs(
        architecture,
        _bag_matrix(bags, train_ids, [train_map[target] for target in train_ids]),
        train_y,
        _bag_matrix(bags, holdout_ids, [holdout_map[target] for target in holdout_ids]),
        learning_rate=float(selected["learning_rate"]),
        weight_decay=float(selected["weight_decay"]),
        epochs=int(selected["final_refit_epochs"]),
        # The paired control shares initialization/dropout randomness with the
        # real refit; only the subject-bundle alignment changes.
        seed=final_seed,
        device=device,
    )
    real_rows = [
        {
            "architecture": architecture,
            "condition": "real",
            "seed": mil_seed,
            "seed_index": seed_index,
            "outer_fold": outer_fold,
            "subject_id": subject_id,
            "source_subject_id": subject_id,
            "label": int(label),
            "probability": float(probability),
            "prediction": int(probability >= THRESHOLD),
            "attention": [float(value) for value in attention],
        }
        for subject_id, label, probability, attention in zip(
            holdout_ids,
            holdout_y,
            real["probabilities"],
            real["attention"],
        )
    ]
    shuffled_rows = [
        {
            "architecture": architecture,
            "condition": "shuffled_bundle",
            "seed": mil_seed,
            "seed_index": seed_index,
            "outer_fold": outer_fold,
            "subject_id": subject_id,
            "source_subject_id": holdout_map[subject_id],
            "label": int(label),
            "probability": float(probability),
            "prediction": int(probability >= THRESHOLD),
            "attention": [float(value) for value in attention],
        }
        for subject_id, label, probability, attention in zip(
            holdout_ids,
            holdout_y,
            shuffled["probabilities"],
            shuffled["attention"],
        )
    ]
    record = {
        "architecture": architecture,
        "seed": mil_seed,
        "outer_fold": outer_fold,
        "selection": selection,
        "real_refit": {
            "final_training_loss": real["final_training_loss"],
            "standardizer_mean_sha256": real["standardizer_mean_sha256"],
            "standardizer_scale_sha256": real["standardizer_scale_sha256"],
        },
        "shuffled_refit": {
            "control_policy": "reuse real-selected hyperparameters and epoch count",
            "final_training_loss": shuffled["final_training_loss"],
            "standardizer_mean_sha256": shuffled["standardizer_mean_sha256"],
            "standardizer_scale_sha256": shuffled["standardizer_scale_sha256"],
        },
    }
    return real_rows, shuffled_rows, record


def _ordered_scores(
    rows: Sequence[dict[str, Any]],
    development_ids: Sequence[str],
    metadata: Mapping[str, dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    by_subject = {str(row["subject_id"]): row for row in rows}
    if len(by_subject) != len(rows) or set(by_subject) != set(development_ids):
        raise ValueError("N2 OOF rows do not cover development subjects exactly once")
    y = _labels(metadata, development_ids)
    scores = np.asarray([float(by_subject[subject_id]["probability"]) for subject_id in development_ids])
    return y, scores


def paired_seed_subject_bootstrap(
    y: np.ndarray,
    real_scores: np.ndarray,
    shuffled_scores: np.ndarray,
    *,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    if real_scores.shape != shuffled_scores.shape or real_scores.ndim != 2:
        raise ValueError("Real/shuffled N2 score matrices must both be [seeds, subjects]")
    real_seed_metrics = [evaluate_binary(y, scores) for scores in real_scores]
    shuffled_seed_metrics = [evaluate_binary(y, scores) for scores in shuffled_scores]
    point = {
        metric: float(
            np.mean(
                [
                    float(real[metric]) - float(shuffled[metric])
                    for real, shuffled in zip(real_seed_metrics, shuffled_seed_metrics)
                    if real.get(metric) is not None and shuffled.get(metric) is not None
                ]
            )
        )
        for metric in PRIMARY_COMPARISON_METRICS
    }
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {metric: [] for metric in PRIMARY_COMPARISON_METRICS}
    correct_probability: list[float] = []
    for _ in range(repeats):
        indices = rng.integers(0, len(y), len(y))
        seed_index = int(rng.integers(0, real_scores.shape[0]))
        real = evaluate_binary(y[indices], real_scores[seed_index, indices])
        shuffled = evaluate_binary(y[indices], shuffled_scores[seed_index, indices])
        for metric in PRIMARY_COMPARISON_METRICS:
            if real.get(metric) is not None and shuffled.get(metric) is not None:
                samples[metric].append(float(real[metric]) - float(shuffled[metric]))
        signed = 2 * y[indices] - 1
        correct_probability.append(
            float(
                np.mean(
                    signed
                    * (
                        real_scores[seed_index, indices]
                        - shuffled_scores[seed_index, indices]
                    )
                )
            )
        )
    return {
        "unit": "subject with MIL seed sampled per replicate",
        "method": "paired percentile subject-and-seed bootstrap",
        "repeats": repeats,
        "seed": seed,
        "point_mean_seedwise_real_minus_shuffled": point,
        "intervals": {
            metric: {
                "lower_2.5pct": float(np.quantile(values, 0.025)),
                "upper_97.5pct": float(np.quantile(values, 0.975)),
                "valid_replicates": len(values),
            }
            for metric, values in samples.items()
            if values
        },
        "mean_correct_class_probability_difference": {
            "point": float(
                np.mean((2 * y - 1) * (real_scores.mean(axis=0) - shuffled_scores.mean(axis=0)))
            ),
            "lower_2.5pct": float(np.quantile(correct_probability, 0.025)),
            "upper_97.5pct": float(np.quantile(correct_probability, 0.975)),
        },
    }


def summarize_architecture(
    architecture: str,
    real_rows: Sequence[dict[str, Any]],
    shuffled_rows: Sequence[dict[str, Any]],
    development_ids: Sequence[str],
    metadata: Mapping[str, dict[str, Any]],
) -> dict[str, Any]:
    real_matrix = np.empty((len(MIL_SEEDS), len(development_ids)), dtype=np.float64)
    shuffled_matrix = np.empty_like(real_matrix)
    seed_rows: list[dict[str, Any]] = []
    for seed_index, mil_seed in enumerate(MIL_SEEDS):
        real_seed_rows = [row for row in real_rows if int(row["seed"]) == mil_seed]
        shuffled_seed_rows = [row for row in shuffled_rows if int(row["seed"]) == mil_seed]
        y, real_scores = _ordered_scores(real_seed_rows, development_ids, metadata)
        shuffled_y, shuffled_scores = _ordered_scores(shuffled_seed_rows, development_ids, metadata)
        if not np.array_equal(y, shuffled_y):
            raise ValueError("N2 shuffled labels changed")
        real_matrix[seed_index] = real_scores
        shuffled_matrix[seed_index] = shuffled_scores
        real_metrics = evaluate_binary(y, real_scores)
        shuffled_metrics = evaluate_binary(y, shuffled_scores)
        seed_fold_metrics = []
        assignment = {
            str(row["subject_id"]): int(row["outer_fold"]) for row in real_seed_rows
        }
        for fold in range(5):
            indices = np.asarray(
                [
                    index
                    for index, subject_id in enumerate(development_ids)
                    if assignment[subject_id] == fold
                ]
            )
            seed_fold_metrics.append(
                {
                    "outer_fold": fold,
                    "real_metrics": evaluate_binary(y[indices], real_scores[indices]),
                    "shuffled_metrics": evaluate_binary(
                        y[indices], shuffled_scores[indices]
                    ),
                }
            )
        seed_rows.append(
            {
                "seed": mil_seed,
                "real_metrics": real_metrics,
                "shuffled_metrics": shuffled_metrics,
                "fold_metrics": seed_fold_metrics,
                "real_minus_shuffled": {
                    metric: (
                        None
                        if real_metrics.get(metric) is None or shuffled_metrics.get(metric) is None
                        else float(real_metrics[metric]) - float(shuffled_metrics[metric])
                    )
                    for metric in PRIMARY_COMPARISON_METRICS
                },
            }
        )
    ensemble_real = real_matrix.mean(axis=0)
    ensemble_shuffled = shuffled_matrix.mean(axis=0)
    paired = paired_seed_subject_bootstrap(
        y,
        real_matrix,
        shuffled_matrix,
        repeats=BOOTSTRAP_REPEATS,
        seed=derived_seed(BOOTSTRAP_SEED, 0 if architecture == "mean_pooling" else 1),
    )
    fold_directions: dict[str, Any] = {}
    assignment = {
        str(row["subject_id"]): int(row["outer_fold"])
        for row in real_rows
        if int(row["seed"]) == MIL_SEEDS[0]
    }
    for metric in ("auroc", "balanced_accuracy"):
        records = []
        for fold in range(5):
            indices = np.asarray(
                [index for index, subject_id in enumerate(development_ids) if assignment[subject_id] == fold]
            )
            real_value = evaluate_binary(y[indices], ensemble_real[indices])[metric]
            shuffled_value = evaluate_binary(y[indices], ensemble_shuffled[indices])[metric]
            records.append(
                {
                    "outer_fold": fold,
                    "real": real_value,
                    "shuffled": shuffled_value,
                    "real_minus_shuffled": float(real_value) - float(shuffled_value),
                    "positive": bool(float(real_value) > float(shuffled_value)),
                }
            )
        fold_directions[metric] = {
            "positive_fold_count": sum(row["positive"] for row in records),
            "records": records,
        }
    auc_ci = paired["intervals"]["auroc"]
    balanced_ci = paired["intervals"]["balanced_accuracy"]
    auc_seed_positive = sum(
        float(row["real_minus_shuffled"]["auroc"]) > 0 for row in seed_rows
    )
    balanced_seed_positive = sum(
        float(row["real_minus_shuffled"]["balanced_accuracy"]) > 0 for row in seed_rows
    )
    return {
        "schema_version": 1,
        "architecture": architecture,
        "unit": "subject",
        "official_test_status": "locked; no test prediction was computed",
        "seed_metrics": seed_rows,
        "across_seed_summary": {
            condition: {
                metric: {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=0)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }
                for metric in METRIC_NAMES
                if (
                    values := [
                        float(row[f"{condition}_metrics"][metric])
                        for row in seed_rows
                        if row[f"{condition}_metrics"].get(metric) is not None
                    ]
                )
            }
            for condition in ("real", "shuffled")
        },
        "ensemble_mean_probability": {
            "real_metrics": evaluate_binary(y, ensemble_real),
            "shuffled_metrics": evaluate_binary(y, ensemble_shuffled),
        },
        "paired_subject_seed_bootstrap_95pct": paired,
        "fold_directions": fold_directions,
        "development_gate_components": {
            "not_a_final_gate_decision": True,
            "pooled_ensemble_auroc_above_0.5": bool(
                evaluate_binary(y, ensemble_real)["auroc"] > 0.5
            ),
            "auroc_delta_ci_excludes_zero_positive": bool(auc_ci["lower_2.5pct"] > 0),
            "balanced_accuracy_delta_ci_excludes_zero_positive": bool(
                balanced_ci["lower_2.5pct"] > 0
            ),
            "auroc_positive_in_at_least_four_outer_folds": bool(
                fold_directions["auroc"]["positive_fold_count"] >= 4
            ),
            "balanced_accuracy_positive_in_at_least_four_outer_folds": bool(
                fold_directions["balanced_accuracy"]["positive_fold_count"] >= 4
            ),
            "auroc_positive_in_at_least_four_seeds": bool(auc_seed_positive >= 4),
            "balanced_accuracy_positive_in_at_least_four_seeds": bool(
                balanced_seed_positive >= 4
            ),
            "remaining_if_development_passes": ["N3 winner freeze", "N4 one-time locked test"],
        },
    }


def freeze_n2(paths: N2Paths) -> dict[str, Any]:
    marker_path = paths.n2_root / "n2_protocol_freeze.json"
    if marker_path.exists():
        verified = verify_n2_protocol(paths)
        return {
            "status": "already_frozen_and_verified",
            "n2_spec_sha256": verified["marker"]["n2_spec_sha256"],
        }
    if paths.n2_root.exists() and any(paths.n2_root.iterdir()):
        raise FileExistsError(f"Refusing to freeze N2 into non-empty directory: {paths.n2_root}")
    paths.n2_root.mkdir(parents=True, exist_ok=True)
    validated, _, wavlm = validate_caches_from_frozen(paths.base)
    parameter_counts = trainable_parameter_counts()
    spec = {
        "schema_version": 1,
        "experiment_id": "daic_acoustic_mil_n2_20260715",
        "status": "frozen",
        "frozen_at_utc": utc_now(),
        "parent_n0_n1": {
            "experiment_spec_path": str(paths.base.output / "experiment_spec.json"),
            "experiment_spec_sha256": validated["freeze"]["experiment_spec_sha256"],
            "fold_assignments_sha256": validated["freeze"]["fold_assignments_sha256"],
            "selected_k4_samples_sha256": validated["freeze"]["selected_k4_samples_sha256"],
        },
        "official_test_policy": {
            "status": "locked",
            "predictions_permitted_in_n2": False,
            "unlock_condition": "development gate passes and N3 freezes one winner",
        },
        "bags": {
            "instances_per_subject": 4,
            "instance_dimension": INSTANCE_DIMENSION,
            "source": "validated frozen WavLM Base+ layers 6/7/8 chunk vectors",
            "source_validation": wavlm.validation,
            "standardization": (
                "per-feature population mean/std over outer/inner training bags only; "
                "scales below 1e-8 become 1"
            ),
        },
        "models": {
            "mean_pooling": {
                "instance_projection": [INSTANCE_DIMENSION, PROJECTION_DIMENSION],
                "pooling": "arithmetic mean over four encoded chunks",
                "head": "LayerNorm(128), Dropout(0.2), Linear(128,1)",
                "trainable_parameters": parameter_counts["mean_pooling"],
            },
            "gated_attention": {
                "instance_projection": [INSTANCE_DIMENSION, PROJECTION_DIMENSION],
                "attention_hidden_dimension": ATTENTION_HIDDEN_DIMENSION,
                "pooling": "tanh(Vh) * sigmoid(Uh), linear score, softmax over four chunks",
                "head": "LayerNorm(128), Dropout(0.2), Linear(128,1)",
                "trainable_parameters": parameter_counts["gated_attention"],
            },
            "encoder_frozen": True,
            "diagnosis_losses_per_subject": 1,
            "chunk_level_diagnosis_loss": False,
        },
        "training": {
            "optimizer": "AdamW",
            "learning_rate_grid": list(LEARNING_RATES),
            "weight_decay_grid": list(WEIGHT_DECAYS),
            "dropout": DROPOUT,
            "loss": "unweighted BCEWithLogitsLoss",
            "batching": "one full subject-bag batch per training partition per epoch",
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "minimum_log_loss_improvement": MIN_DELTA,
            "selection": "minimum pooled inner-OOF log loss",
            "selection_ties": "smaller learning rate, then larger weight decay",
            "final_refit_epochs": "rounded median best epoch across selected candidate inner folds",
            "mil_seeds": list(MIL_SEEDS),
            "candidate_randomness": (
                "all hyperparameter candidates share initialization/dropout seed within each "
                "MIL seed, outer fold, and inner fold"
            ),
            "threshold": THRESHOLD,
        },
        "shuffled_control": {
            "unit": "complete K=4 subject bag",
            "mapping": "independent derangement inside outer train and outer holdout",
            "one_derangement_per_mil_seed": True,
            "seed": SHUFFLE_SEED,
            "hyperparameters": (
                "reuse the corresponding real model's inner-selected learning rate, "
                "weight decay, and final epoch count without inspecting shuffled outcomes"
            ),
            "paired_randomness": (
                "real and shuffled final refits share initialization/dropout seed within "
                "each MIL seed and outer fold"
            ),
        },
        "analysis": {
            "metrics": list(METRIC_NAMES),
            "bootstrap_repeats": BOOTSTRAP_REPEATS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap": "paired subject bootstrap with one of five MIL seeds sampled per replicate",
            "seed_stability": "positive real-minus-shuffled direction in at least four of five seeds",
            "attention_diagnostics": [
                "normalized entropy",
                "selected position",
                "numeric chunk ordinal",
                "duration",
                "RMS",
                "silence fraction",
                "active-frame fraction",
                "clipping fraction",
            ],
        },
        "audio_nuisance_definitions": {
            "silence": f"absolute sample amplitude < {SILENCE_ABSOLUTE_THRESHOLD}",
            "active_frame": (
                f"non-overlapping {FRAME_MILLISECONDS}ms mono frame RMS >= "
                f"{ACTIVE_FRAME_RMS_THRESHOLD}"
            ),
            "clipping": f"absolute sample amplitude >= {CLIPPING_ABSOLUTE_THRESHOLD}",
            "scope": "development subjects only",
        },
        "smoke_requirement": {
            "outer_fold": 0,
            "seed": MIL_SEEDS[0],
            "architectures": list(ARCHITECTURES),
            "must_complete_before_full_run": True,
        },
        "provenance": {
            "analysis_code_path": str(Path(__file__).resolve()),
            "analysis_code_sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    spec_path = paths.n2_root / "n2_experiment_spec.json"
    atomic_json(spec_path, spec)
    marker = {
        "schema_version": 1,
        "status": "frozen",
        "n2_spec_sha256": sha256_file(spec_path),
        "analysis_code_sha256": sha256_file(Path(__file__).resolve()),
        "parent_experiment_spec_sha256": validated["freeze"]["experiment_spec_sha256"],
        "official_test_predictions_created": False,
    }
    atomic_json(marker_path, marker)
    return {
        "status": "frozen",
        "n2_spec_sha256": marker["n2_spec_sha256"],
        "trainable_parameters": parameter_counts,
        "official_test_predictions_created": False,
    }


def verify_n2_protocol(paths: N2Paths) -> dict[str, Any]:
    marker_path = paths.n2_root / "n2_protocol_freeze.json"
    spec_path = paths.n2_root / "n2_experiment_spec.json"
    if not marker_path.is_file() or not spec_path.is_file():
        raise FileNotFoundError("N2 protocol has not been frozen")
    marker = read_json(marker_path)
    if marker.get("status") != "frozen" or marker.get("official_test_predictions_created") is not False:
        raise ValueError("Invalid N2 freeze marker or locked-test state")
    if sha256_file(spec_path) != marker.get("n2_spec_sha256"):
        raise ValueError("N2 experiment specification changed after freeze")
    if sha256_file(Path(__file__).resolve()) != marker.get("analysis_code_sha256"):
        raise ValueError("N2 analysis code changed after freeze; use a new experiment ID")
    parent = read_json(paths.base.output / "protocol_freeze.json")
    if parent.get("experiment_spec_sha256") != marker.get("parent_experiment_spec_sha256"):
        raise ValueError("Parent N0/N1 protocol differs from the N2 freeze")
    return {"status": "verified", "marker": marker, "spec": read_json(spec_path)}


def load_n2_inputs(paths: N2Paths) -> tuple[dict[str, Any], dict[str, Any], dict[str, np.ndarray]]:
    verify_n2_protocol(paths)
    _, _, wavlm = validate_caches_from_frozen(paths.base)
    selection = read_json(paths.base.output / "selected_k4_samples.json")
    folds = read_json(paths.base.output / "fold_assignments.json")
    bags = load_bags(selection, wavlm.vectors)
    return selection, folds, bags


def run_smoke(paths: N2Paths, device_name: str) -> dict[str, Any]:
    verify_n2_protocol(paths)
    smoke_dir = paths.n2_root / "smoke"
    complete_path = smoke_dir / "complete.json"
    if complete_path.is_file():
        completion = read_json(complete_path)
        result_path = smoke_dir / "smoke_result.json"
        if sha256_file(result_path) != completion["smoke_result_sha256"]:
            raise ValueError("N2 smoke artifact changed")
        return read_json(result_path)
    if smoke_dir.exists():
        raise FileExistsError(f"Incomplete N2 smoke directory exists: {smoke_dir}")
    selection, folds, bags = load_n2_inputs(paths)
    metadata = subject_metadata(selection)
    device = resolve_device(device_name)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    architecture_results: list[dict[str, Any]] = []
    outer = folds["folds"][0]
    for architecture in ARCHITECTURES:
        real, shuffled, record = fit_outer_real_and_shuffle(
            architecture,
            bags,
            metadata,
            outer,
            mil_seed=MIL_SEEDS[0],
            seed_index=0,
            device=device,
        )
        architecture_results.append(
            {
                "architecture": architecture,
                "record": record,
                "real_holdout_metrics": evaluate_binary(
                    [row["label"] for row in real], [row["probability"] for row in real]
                ),
                "shuffled_holdout_metrics": evaluate_binary(
                    [row["label"] for row in shuffled],
                    [row["probability"] for row in shuffled],
                ),
            }
        )
    elapsed = time.perf_counter() - started
    result = {
        "schema_version": 1,
        "status": "passed",
        "completed_at_utc": utc_now(),
        "device": str(device),
        "outer_fold": 0,
        "seed": MIL_SEEDS[0],
        "elapsed_seconds": elapsed,
        "estimated_full_seconds_linear_extrapolation": elapsed * 25,
        "estimated_full_hours_linear_extrapolation": elapsed * 25 / 3600,
        "cuda_peak_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        ),
        "cuda_peak_reserved_bytes": (
            int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None
        ),
        "architectures": architecture_results,
        "official_test_predictions_created": False,
    }
    smoke_dir.mkdir(parents=True)
    atomic_json(smoke_dir / "smoke_result.json", result)
    atomic_json(
        complete_path,
        {
            "status": "complete_immutable",
            "smoke_result_sha256": sha256_file(smoke_dir / "smoke_result.json"),
            "official_test_predictions_created": False,
        },
    )
    return {
        "status": result["status"],
        "device": result["device"],
        "elapsed_seconds": result["elapsed_seconds"],
        "estimated_full_hours_linear_extrapolation": result[
            "estimated_full_hours_linear_extrapolation"
        ],
        "cuda_peak_allocated_bytes": result["cuda_peak_allocated_bytes"],
        "cuda_peak_reserved_bytes": result["cuda_peak_reserved_bytes"],
        "official_test_predictions_created": False,
    }


def audio_nuisance_rows(
    selection: Mapping[str, Any],
    development_ids: set[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for subject_id in sorted(development_ids, key=natural_key):
        subject = selection["subjects"][subject_id]
        for sample in sorted(subject["samples"], key=lambda row: int(row["selected_position"])):
            audio_path = Path(str(sample["audio_path"]))
            values, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
            channels = int(values.shape[1])
            mono = values.mean(axis=1, dtype=np.float32)
            if mono.size == 0 or not np.isfinite(mono).all():
                raise ValueError(f"Invalid audio while computing N2 nuisance data: {audio_path}")
            absolute = np.abs(mono)
            rms = float(np.sqrt(np.mean(np.square(mono, dtype=np.float64))))
            frame_samples = max(1, int(round(int(sample_rate) * FRAME_MILLISECONDS / 1000)))
            frame_count = mono.size // frame_samples
            if frame_count:
                framed = mono[: frame_count * frame_samples].reshape(frame_count, frame_samples)
                frame_rms = np.sqrt(np.mean(np.square(framed, dtype=np.float64), axis=1))
                active_fraction = float(np.mean(frame_rms >= ACTIVE_FRAME_RMS_THRESHOLD))
            else:
                active_fraction = float(rms >= ACTIVE_FRAME_RMS_THRESHOLD)
            rows.append(
                {
                    "subject_id": subject_id,
                    "sample_id": str(sample["sample_id"]),
                    "label": int(subject["label"]),
                    "selected_position": int(sample["selected_position"]),
                    "numeric_chunk_number": int(sample["numeric_chunk_number"]),
                    "sample_kind": str(sample["sample_kind"]),
                    "audio_sha256": str(sample["audio_sha256"]),
                    "duration_seconds": float(mono.size / int(sample_rate)),
                    "sample_rate": int(sample_rate),
                    "channels": channels,
                    "rms": rms,
                    "silence_fraction": float(np.mean(absolute < SILENCE_ABSOLUTE_THRESHOLD)),
                    "zero_fraction": float(np.mean(mono == 0)),
                    "active_frame_fraction": active_fraction,
                    "clipping_fraction": float(np.mean(absolute >= CLIPPING_ABSOLUTE_THRESHOLD)),
                }
            )
    return rows


def _safe_spearman(x: Sequence[float], y: Sequence[float]) -> dict[str, Any]:
    from scipy.stats import spearmanr

    x_values = np.asarray(x, dtype=np.float64)
    y_values = np.asarray(y, dtype=np.float64)
    if len(x_values) < 3 or np.all(x_values == x_values[0]) or np.all(y_values == y_values[0]):
        return {"rho": None, "p_value": None, "defined": False}
    result = spearmanr(x_values, y_values)
    return {
        "rho": float(result.statistic),
        "p_value": float(result.pvalue),
        "defined": True,
    }


def build_nuisance_diagnostics(
    paths: N2Paths,
    selection: Mapping[str, Any],
    folds: Mapping[str, Any],
    gated_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    development_ids = set(str(value) for value in folds["development_subject_ids"])
    nuisance = audio_nuisance_rows(selection, development_ids)
    nuisance_by_sample = {str(row["sample_id"]): row for row in nuisance}
    nuisance_path = paths.n2_root / "audio_nuisance.jsonl"
    atomic_jsonl(nuisance_path, nuisance)
    attention_rows: list[dict[str, Any]] = []
    entropies: list[float] = []
    highest_positions: Counter[int] = Counter()
    for prediction in gated_rows:
        subject_id = str(prediction["subject_id"])
        samples = sorted(
            selection["subjects"][subject_id]["samples"],
            key=lambda row: int(row["selected_position"]),
        )
        attention = np.asarray(prediction["attention"], dtype=np.float64)
        if attention.shape != (4,) or not np.isfinite(attention).all() or not np.isclose(attention.sum(), 1):
            raise ValueError(f"Invalid gated attention weights for {subject_id}")
        entropy = float(-np.sum(attention * np.log(np.clip(attention, 1e-12, 1))) / np.log(4))
        entropies.append(entropy)
        highest_positions[int(np.argmax(attention))] += 1
        for sample, weight in zip(samples, attention):
            row = nuisance_by_sample[str(sample["sample_id"])]
            attention_rows.append(
                {
                    **row,
                    "seed": int(prediction["seed"]),
                    "outer_fold": int(prediction["outer_fold"]),
                    "attention_weight": float(weight),
                    "normalized_attention_entropy": entropy,
                }
            )
    attention_path = paths.n2_root / "gated_attention_diagnostics.jsonl"
    atomic_jsonl(attention_path, attention_rows)
    variables = (
        "selected_position",
        "numeric_chunk_number",
        "duration_seconds",
        "rms",
        "silence_fraction",
        "zero_fraction",
        "active_frame_fraction",
        "clipping_fraction",
    )
    correlations = {
        variable: _safe_spearman(
            [float(row[variable]) for row in attention_rows],
            [float(row["attention_weight"]) for row in attention_rows],
        )
        for variable in variables
    }
    subject_aggregates: dict[str, dict[str, Any]] = {}
    for subject_id in sorted(development_ids, key=natural_key):
        rows = [row for row in nuisance if row["subject_id"] == subject_id]
        subject_aggregates[subject_id] = {
            "label": int(rows[0]["label"]),
            **{
                variable: float(np.mean([float(row[variable]) for row in rows]))
                for variable in variables
                if variable not in {"selected_position", "numeric_chunk_number"}
            },
        }
    label_associations = {
        variable: _safe_spearman(
            [row[variable] for row in subject_aggregates.values()],
            [row["label"] for row in subject_aggregates.values()],
        )
        for variable in (
            "duration_seconds",
            "rms",
            "silence_fraction",
            "zero_fraction",
            "active_frame_fraction",
            "clipping_fraction",
        )
    }
    hash_groups: dict[str, list[str]] = defaultdict(list)
    for row in nuisance:
        hash_groups[str(row["audio_sha256"])].append(str(row["sample_id"]))
    duplicates = [sample_ids for sample_ids in hash_groups.values() if len(sample_ids) > 1]
    sample_kind_by_label = Counter((int(row["label"]), str(row["sample_kind"])) for row in nuisance)
    summary = {
        "schema_version": 1,
        "scope": "142 development subjects only",
        "sample_count": len(nuisance),
        "exact_duplicate_groups": duplicates,
        "exact_duplicate_group_count": len(duplicates),
        "sample_kind_by_label": {
            f"label_{label}_{kind}": count
            for (label, kind), count in sorted(sample_kind_by_label.items())
        },
        "attention": {
            "row_count": len(attention_rows),
            "normalized_entropy": {
                "mean": float(np.mean(entropies)),
                "std": float(np.std(entropies, ddof=0)),
                "min": float(np.min(entropies)),
                "max": float(np.max(entropies)),
            },
            "highest_attention_selected_position_counts": {
                str(position): highest_positions[position] for position in range(4)
            },
            "pooled_spearman_correlations": correlations,
        },
        "subject_mean_nuisance_label_spearman": label_associations,
        "limitations": [
            "Sample suffix order is ordinal only, not verified chronology.",
            "All chunks within a subject share preprocessing kind, which remains perfectly label-associated.",
            "Local files cannot establish participant-only speech or interviewer exclusion.",
            "Attention is diagnostic and is not evidence of clinical relevance.",
        ],
        "artifacts": {
            "audio_nuisance_jsonl": nuisance_path.name,
            "audio_nuisance_sha256": sha256_file(nuisance_path),
            "attention_jsonl": attention_path.name,
            "attention_sha256": sha256_file(attention_path),
        },
    }
    atomic_json(paths.n2_root / "nuisance_diagnostics.json", summary)
    return summary


def run_full(paths: N2Paths, device_name: str) -> dict[str, Any]:
    verify_n2_protocol(paths)
    smoke_complete = paths.n2_root / "smoke/complete.json"
    if not smoke_complete.is_file():
        raise RuntimeError("The frozen one-fold N2 smoke run must complete before the full run")
    results_root = paths.n2_root / "results"
    if results_root.exists():
        complete = paths.n2_root / "n2_complete.json"
        if not complete.is_file():
            raise FileExistsError(f"Incomplete N2 results directory exists: {results_root}")
        verify_n2_outputs(paths)
        return {"status": "already_complete_and_verified"}
    selection, folds, bags = load_n2_inputs(paths)
    metadata = subject_metadata(selection)
    development_ids = [str(value) for value in folds["development_subject_ids"]]
    device = resolve_device(device_name)
    results_root.mkdir(parents=True)
    started = time.perf_counter()
    metrics_by_architecture: dict[str, Any] = {}
    gated_real_rows: list[dict[str, Any]] = []
    for architecture in ARCHITECTURES:
        architecture_started = time.perf_counter()
        architecture_dir = results_root / architecture
        architecture_dir.mkdir()
        real_rows: list[dict[str, Any]] = []
        shuffled_rows: list[dict[str, Any]] = []
        selection_records: list[dict[str, Any]] = []
        for seed_index, mil_seed in enumerate(MIL_SEEDS):
            for outer in folds["folds"]:
                print(
                    f"N2 {architecture}: seed {seed_index + 1}/{len(MIL_SEEDS)} "
                    f"outer {int(outer['outer_fold']) + 1}/5",
                    flush=True,
                )
                real, shuffled, record = fit_outer_real_and_shuffle(
                    architecture,
                    bags,
                    metadata,
                    outer,
                    mil_seed=mil_seed,
                    seed_index=seed_index,
                    device=device,
                )
                real_rows.extend(real)
                shuffled_rows.extend(shuffled)
                selection_records.append(record)
        for mil_seed in MIL_SEEDS:
            validate_oof_coverage(
                [row for row in real_rows if int(row["seed"]) == mil_seed],
                expected_subject_ids=set(development_ids),
                context=f"N2 {architecture} real seed={mil_seed}",
            )
            validate_oof_coverage(
                [row for row in shuffled_rows if int(row["seed"]) == mil_seed],
                expected_subject_ids=set(development_ids),
                context=f"N2 {architecture} shuffled seed={mil_seed}",
            )
        real_rows.sort(key=lambda row: (int(row["seed"]), natural_key(row["subject_id"])))
        shuffled_rows.sort(key=lambda row: (int(row["seed"]), natural_key(row["subject_id"])))
        metrics = summarize_architecture(
            architecture,
            real_rows,
            shuffled_rows,
            development_ids,
            metadata,
        )
        metrics["elapsed_seconds"] = time.perf_counter() - architecture_started
        atomic_jsonl(architecture_dir / "oof_predictions.jsonl", real_rows)
        atomic_jsonl(architecture_dir / "shuffled_oof_predictions.jsonl", shuffled_rows)
        atomic_json(architecture_dir / "selection_and_refit.json", selection_records)
        atomic_json(architecture_dir / "metrics.json", metrics)
        artifacts = sorted(
            (path for path in architecture_dir.iterdir() if path.is_file()),
            key=lambda path: path.name,
        )
        atomic_json(
            architecture_dir / "complete.json",
            {
                "status": "complete_immutable",
                "artifact_sha256": {path.name: sha256_file(path) for path in artifacts},
                "official_test_predictions_created": False,
            },
        )
        metrics_by_architecture[architecture] = metrics
        if architecture == "gated_attention":
            gated_real_rows = real_rows
    nuisance = build_nuisance_diagnostics(paths, selection, folds, gated_real_rows)
    total_elapsed = time.perf_counter() - started
    provenance = {
        "schema_version": 1,
        "completed_at_utc": utc_now(),
        "command": shlex.join([sys.executable, *sys.argv]),
        "elapsed_seconds": total_elapsed,
        "device": str(device),
        "n2_spec_sha256": sha256_file(paths.n2_root / "n2_experiment_spec.json"),
        "analysis_code_sha256": sha256_file(Path(__file__).resolve()),
        "parent_protocol_sha256": sha256_file(paths.base.output / "experiment_spec.json"),
        "repository": git_provenance(),
        "dependencies": {
            **dependency_versions(),
            "torch": torch.__version__,
            "soundfile": importlib.metadata.version("soundfile"),
        },
        "official_test_predictions_created": False,
        "leakage_guards": {
            "one_oof_prediction_per_subject_per_seed": True,
            "outer_and_inner_subject_isolation_reused_from_frozen_N0": True,
            "standardizer_fit_on_training_bags_only": True,
            "complete_bundle_derangement_within_outer_partition": True,
            "official_test_used": False,
        },
    }
    atomic_json(paths.n2_root / "provenance.json", provenance)
    render_report(paths, metrics_by_architecture, nuisance, total_elapsed)
    root_artifacts = [
        paths.n2_root / "nuisance_diagnostics.json",
        paths.n2_root / "audio_nuisance.jsonl",
        paths.n2_root / "gated_attention_diagnostics.jsonl",
        paths.n2_root / "provenance.json",
        paths.n2_root / "N2_RESULTS.md",
    ]
    completion = {
        "schema_version": 1,
        "status": "complete_immutable",
        "architectures": list(ARCHITECTURES),
        "root_artifact_sha256": {path.name: sha256_file(path) for path in root_artifacts},
        "official_test_predictions_created": False,
    }
    atomic_json(paths.n2_root / "n2_complete.json", completion)
    return {
        "status": "N2_complete",
        "elapsed_seconds": total_elapsed,
        "official_test_status": "locked",
        "development_gate": {
            architecture: metrics["development_gate_components"]
            for architecture, metrics in metrics_by_architecture.items()
        },
    }


def render_report(
    paths: N2Paths,
    metrics_by_architecture: Mapping[str, Any],
    nuisance: Mapping[str, Any],
    elapsed_seconds: float,
) -> None:
    lines = [
        "# DAIC N2 subject-level MIL results",
        "",
        f"Generated: {utc_now()}",
        "",
        "The official 47-subject test set remains locked. Results below are pooled development "
        "OOF predictions, averaged across five fixed seeds.",
        "",
        "| Model | AUROC | AUPRC | Balanced acc. | Log loss | AUROC delta 95% CI | Balanced-acc. delta 95% CI | Positive AUROC seeds/folds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for architecture in ARCHITECTURES:
        metrics = metrics_by_architecture[architecture]
        ensemble = metrics["ensemble_mean_probability"]["real_metrics"]
        paired = metrics["paired_subject_seed_bootstrap_95pct"]
        auc_point = paired["point_mean_seedwise_real_minus_shuffled"]["auroc"]
        bal_point = paired["point_mean_seedwise_real_minus_shuffled"]["balanced_accuracy"]
        auc_ci = paired["intervals"]["auroc"]
        bal_ci = paired["intervals"]["balanced_accuracy"]
        seed_positive = sum(
            float(row["real_minus_shuffled"]["auroc"]) > 0 for row in metrics["seed_metrics"]
        )
        fold_positive = metrics["fold_directions"]["auroc"]["positive_fold_count"]
        lines.append(
            f"| {architecture} | {ensemble['auroc']:.4f} | {ensemble['auprc']:.4f} | "
            f"{ensemble['balanced_accuracy']:.4f} | {ensemble['log_loss']:.4f} | "
            f"{auc_point:.4f} [{auc_ci['lower_2.5pct']:.4f}, {auc_ci['upper_97.5pct']:.4f}] | "
            f"{bal_point:.4f} [{bal_ci['lower_2.5pct']:.4f}, {bal_ci['upper_97.5pct']:.4f}] | "
            f"{seed_positive}/5 seeds, {fold_positive}/5 folds |"
        )
    gated_gate = metrics_by_architecture["gated_attention"]["development_gate_components"]
    dev_pass = all(
        gated_gate[key]
        for key in (
            "pooled_ensemble_auroc_above_0.5",
            "auroc_delta_ci_excludes_zero_positive",
            "balanced_accuracy_delta_ci_excludes_zero_positive",
            "auroc_positive_in_at_least_four_outer_folds",
            "auroc_positive_in_at_least_four_seeds",
        )
    )
    lines.extend(
        [
            "",
            f"Development gate status for gated MIL: **{'passes development components' if dev_pass else 'does not pass development components'}**.",
            "",
            f"Runtime: {elapsed_seconds / 3600:.2f} hours. Mean normalized attention entropy: "
            f"{nuisance['attention']['normalized_entropy']['mean']:.4f}.",
            "",
            "This does not constitute the complete gate because the official test remains locked. "
            "Attention is diagnostic only, and preprocessing kind remains perfectly label-associated.",
            "",
        ]
    )
    destination = paths.n2_root / "N2_RESULTS.md"
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=destination.parent, prefix=f".{destination.name}.", delete=False
    ) as handle:
        handle.write("\n".join(lines))
        temporary = Path(handle.name)
    os.replace(temporary, destination)


def verify_n2_outputs(paths: N2Paths) -> dict[str, Any]:
    verify_n2_protocol(paths)
    smoke_complete = read_json(paths.n2_root / "smoke/complete.json")
    if smoke_complete.get("official_test_predictions_created") is not False:
        raise ValueError("Invalid N2 smoke locked-test state")
    smoke_result = paths.n2_root / "smoke/smoke_result.json"
    if sha256_file(smoke_result) != smoke_complete["smoke_result_sha256"]:
        raise ValueError("N2 smoke artifact changed")
    complete_path = paths.n2_root / "n2_complete.json"
    if not complete_path.is_file():
        return {
            "status": "protocol_and_smoke_verified",
            "full_run_complete": False,
            "official_test_predictions_created": False,
        }
    complete = read_json(complete_path)
    if complete.get("official_test_predictions_created") is not False:
        raise ValueError("Invalid N2 completion locked-test state")
    for architecture in ARCHITECTURES:
        architecture_dir = paths.n2_root / "results" / architecture
        architecture_complete = read_json(architecture_dir / "complete.json")
        for filename, expected_hash in architecture_complete["artifact_sha256"].items():
            path = architecture_dir / filename
            if not path.is_file() or sha256_file(path) != expected_hash:
                raise ValueError(f"N2 immutable artifact changed: {path}")
        real = read_jsonl(architecture_dir / "oof_predictions.jsonl")
        shuffled = read_jsonl(architecture_dir / "shuffled_oof_predictions.jsonl")
        development = set(read_json(paths.base.output / "fold_assignments.json")["development_subject_ids"])
        for mil_seed in MIL_SEEDS:
            validate_oof_coverage(
                [row for row in real if int(row["seed"]) == mil_seed],
                expected_subject_ids=development,
                context=f"verify N2 {architecture} real seed={mil_seed}",
            )
            validate_oof_coverage(
                [row for row in shuffled if int(row["seed"]) == mil_seed],
                expected_subject_ids=development,
                context=f"verify N2 {architecture} shuffled seed={mil_seed}",
            )
    for filename, expected_hash in complete["root_artifact_sha256"].items():
        path = paths.n2_root / filename
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ValueError(f"N2 root artifact changed: {path}")
    return {
        "status": "verified",
        "full_run_complete": True,
        "architectures": list(ARCHITECTURES),
        "official_test_predictions_created": False,
    }


def add_path_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--partitions", default=str(DEFAULT_PARTITIONS))
    parser.add_argument("--egemaps-cache", default=str(DEFAULT_EGEMAPS_CACHE))
    parser.add_argument("--wavlm-cache", default=str(DEFAULT_WAVLM_CACHE))
    parser.add_argument("--protocol-root", default=str(DEFAULT_PROTOCOL_ROOT))
    parser.add_argument("--n2-root", default=None)


def paths_from_args(args: argparse.Namespace) -> N2Paths:
    protocol_root = Path(args.protocol_root).expanduser().resolve()
    n2_root = (
        Path(args.n2_root).expanduser().resolve()
        if args.n2_root
        else protocol_root / "n2"
    )
    return N2Paths(
        base=CachePaths(
            manifest=Path(args.manifest).expanduser().resolve(),
            partitions=Path(args.partitions).expanduser().resolve(),
            egemaps=Path(args.egemaps_cache).expanduser().resolve(),
            wavlm=Path(args.wavlm_cache).expanduser().resolve(),
            output=protocol_root,
        ),
        n2_root=n2_root,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze")
    add_path_arguments(freeze)
    smoke = subparsers.add_parser("smoke")
    add_path_arguments(smoke)
    smoke.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    run = subparsers.add_parser("run")
    add_path_arguments(run)
    run.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    all_parser = subparsers.add_parser("all")
    add_path_arguments(all_parser)
    all_parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    verify = subparsers.add_parser("verify")
    add_path_arguments(verify)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = paths_from_args(args)
    if args.command == "freeze":
        result = freeze_n2(paths)
    elif args.command == "smoke":
        result = run_smoke(paths, args.device)
    elif args.command == "run":
        result = run_full(paths, args.device)
    elif args.command == "all":
        freeze_n2(paths)
        run_smoke(paths, args.device)
        result = run_full(paths, args.device)
    elif args.command == "verify":
        result = verify_n2_outputs(paths)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
