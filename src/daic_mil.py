from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch


def candidate_mean_token_logprob(model, inputs: dict[str, torch.Tensor], prompt_len: int) -> torch.Tensor:
    """Differentiable counterpart of evaluation.score_candidate_label."""
    target_ids = inputs["input_ids"][0, prompt_len:]
    outputs = model(**inputs)
    logits = outputs.logits[0, prompt_len - 1 : inputs["input_ids"].shape[1] - 1]
    token_log_probs = torch.log_softmax(logits, dim=-1).gather(
        -1, target_ids.unsqueeze(-1)
    ).squeeze(-1)
    if token_log_probs.numel() == 0:
        raise ValueError("Candidate label produced no target tokens.")
    return token_log_probs.mean()


def streaming_subject_mil_backward(
    examples: Sequence[Any], *, label: int,
    margin_fn: Callable[[Any], torch.Tensor],
    backward_fn: Callable[[torch.Tensor], None] | None = None,
) -> dict[str, float]:
    """Exact memory-safe two-pass BCE over a subject's mean candidate margin.

    ``margin_fn`` must return the same mean-token log-probability margin used at
    evaluation. Its first invocation per example runs under ``no_grad``; the
    second is recomputed with gradients. No optimizer step occurs here, making
    it impossible to update midway through a subject.
    """
    if not examples:
        raise ValueError("MIL subjects must contain at least one chunk.")
    with torch.no_grad():
        first_pass = [margin_fn(example).detach() for example in examples]
        mean_margin = torch.stack(first_pass).mean()
        target = torch.as_tensor(float(label), device=mean_margin.device, dtype=mean_margin.dtype)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(mean_margin, target)
        dloss_dmean = torch.sigmoid(mean_margin) - target
    backward = backward_fn or (lambda value: value.backward())
    coefficient = dloss_dmean.detach() / len(examples)
    for example in examples:
        backward(coefficient * margin_fn(example))
    return {
        "mean_margin": float(mean_margin.item()),
        "loss": float(loss.item()),
        "dloss_dmean": float(dloss_dmean.item()),
        "num_chunks": len(examples),
    }
