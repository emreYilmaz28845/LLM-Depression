"""Warm-started linear classification head for LP-FT (linear probe, then fine-tune).

The head is the frozen subject-mean L2 logistic head of
``baselines/frozen_standalone_heads.py`` with its train-only standardisation
folded into the weights, so it applies to raw final-layer hidden vectors:

    logit(x) = ((x - mean) / scale) @ w + b = x @ (w / scale) + (b - (mean / scale) @ w)

Because the head is linear and the frozen pipeline averages standardised chunk
vectors per subject, the subject logit equals the mean of the chunk logits.

The head lives outside FSDP: every rank holds a full fp32 replica, its gradients
are all-reduced by hand and it has its own optimizer. Only output tensors of the
FSDP forward touch it, so ``use_orig_params`` views are never read outside the
forward.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np

HEAD_INIT_SCHEMA_VERSION = "lpft_head_init.v1"
HEAD_STATE_FILENAME = "lpft_head.pt"
OBJECTIVE_HEAD_BCE = "head_bce"


def fold_standardization(mean: np.ndarray, scale: np.ndarray, weight: np.ndarray, bias: float) -> tuple[np.ndarray, float]:
    """Fold ``(x - mean) / scale`` into a linear head on raw vectors."""
    mean = np.asarray(mean, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    if not (mean.shape == scale.shape == weight.shape) or weight.ndim != 1:
        raise ValueError("mean, scale and weight must be 1-D vectors of the same length.")
    if bool((scale <= 0).any()):
        raise ValueError("scale must be positive.")
    folded = weight / scale
    return folded, float(bias) - float((mean / scale) @ weight)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_head_init(path: str | Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        data = {key: payload[key] for key in payload.files}
    required = {"schema_version", "weight", "bias", "scale", "l2_penalty"}
    missing = required - set(data)
    if missing:
        raise ValueError(f"{path}: head init is missing {sorted(missing)}.")
    if str(data["schema_version"]) != HEAD_INIT_SCHEMA_VERSION:
        raise ValueError(f"{path}: unexpected schema {data['schema_version']!r}.")
    return {
        "weight": np.asarray(data["weight"], dtype=np.float64),
        "bias": float(np.asarray(data["bias"]).reshape(())),
        "scale": np.asarray(data["scale"], dtype=np.float64),
        "l2_penalty": float(np.asarray(data["l2_penalty"]).reshape(())),
        "sha256": file_sha256(path),
    }


def subject_mean_logits(sample_logits: list[float], rows: list[dict[str, Any]]) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Mean chunk logit per subject (equal weight per chunk), subjects in sorted order."""
    if len(sample_logits) != len(rows):
        raise ValueError("One logit per row is required.")
    sums: dict[str, list[float]] = {}
    labels: dict[str, int] = {}
    for logit, row in zip(sample_logits, rows):
        subject = str(row["subject_id"])
        label = int(row["label"])
        if labels.setdefault(subject, label) != label:
            raise ValueError(f"Subject {subject} has conflicting labels.")
        sums.setdefault(subject, []).append(float(logit))
    subjects = sorted(sums)
    return (
        subjects,
        np.asarray([labels[s] for s in subjects], dtype=np.int64),
        np.asarray([float(np.mean(sums[s])) for s in subjects], dtype=np.float64),
    )


def subject_predictions(subjects: list[str], labels: np.ndarray, logits: np.ndarray,
                        counts: dict[str, int] | None = None) -> list[dict[str, Any]]:
    return [
        {
            "subject_id": s,
            "label": int(y),
            "prediction": int(z > 0),
            "probability": float(1.0 / (1.0 + math.exp(-z))),
            "mean_logit": float(z),
            "num_samples": int((counts or {}).get(s, 0)),
        }
        for s, y, z in zip(subjects, labels.tolist(), logits.tolist())
    ]


def headline_from_head_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Map head metrics onto the keys the training loop's selection code reads."""
    tn, fp = metrics["confusion_matrix"][0]
    fn, tp = metrics["confusion_matrix"][1]
    total = tn + fp + fn + tp
    return {
        **metrics,
        "accuracy": float((tp + tn) / total) if total else 0.0,
        "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
        "auroc": float(metrics.get("auroc") or 0.0),
    }


def _torch():
    import torch  # noqa: PLC0415

    return torch


def build_head_module(init: dict[str, Any], device):
    torch = _torch()

    class LPFTHead(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(init["weight"], dtype=torch.float32))
            self.bias = torch.nn.Parameter(torch.tensor(float(init["bias"]), dtype=torch.float32))
            self.register_buffer("scale", torch.tensor(init["scale"], dtype=torch.float32))
            self.l2_penalty = float(init["l2_penalty"])

        def forward(self, hidden):
            return hidden.float() @ self.weight + self.bias

        def l2_term(self):
            # Same penalty as the frozen fit: 0.5 * lambda * ||w_standardised||^2,
            # with w_standardised = w_folded * scale.
            return 0.5 * self.l2_penalty * torch.sum((self.weight * self.scale) ** 2)

    return LPFTHead().to(device)


def last_prompt_hidden(hidden, prompt_lens):
    """Rows of ``hidden`` [B, T, D] at position ``prompt_len - 1`` of each example."""
    torch = _torch()
    index = prompt_lens.to(hidden.device).long() - 1
    if bool((index < 0).any()) or bool((index >= hidden.shape[1]).any()):
        raise ValueError("prompt_len outside the sequence.")
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), index]


def all_reduce_mean_grads(module) -> None:
    torch = _torch()
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return
    world = torch.distributed.get_world_size()
    for parameter in module.parameters():
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        torch.distributed.all_reduce(parameter.grad)
        parameter.grad.div_(world)


def save_head(module, path: str | Path, meta: dict[str, Any]) -> str:
    torch = _torch()
    state = {
        "weight": module.weight.detach().cpu().double().numpy(),
        "bias": float(module.bias.detach().cpu().item()),
        "scale": module.scale.detach().cpu().double().numpy(),
        "l2_penalty": float(module.l2_penalty),
        "meta": meta,
    }
    torch.save(state, str(path))
    return file_sha256(path)


def load_head_state(path: str | Path) -> dict[str, Any]:
    torch = _torch()
    state = torch.load(str(path), map_location="cpu", weights_only=False)
    state["weight"] = np.asarray(state["weight"], dtype=np.float64)
    state["sha256"] = file_sha256(path)
    return state
