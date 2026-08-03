from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch


def candidate_mean_token_logprob(model, inputs: dict[str, torch.Tensor], prompt_len: int) -> torch.Tensor:
    """Differentiable counterpart of evaluation.score_candidate_label."""
    input_ids = inputs["input_ids"]
    if input_ids.ndim != 2 or int(input_ids.shape[0]) != 1:
        raise ValueError("MIL candidate scoring requires a single unpadded example.")
    prompt_len = int(prompt_len)
    if prompt_len < 1 or prompt_len >= int(input_ids.shape[1]):
        raise ValueError("prompt_len must leave at least one candidate token.")
    target_ids = input_ids[0, prompt_len:]
    outputs = model(**inputs)
    logits = outputs.logits
    if logits.ndim != 3 or int(logits.shape[0]) != 1:
        raise ValueError("MIL candidate scoring expected logits with batch size one.")
    if int(logits.shape[1]) < int(input_ids.shape[1]):
        raise ValueError("Model logits are shorter than the candidate input sequence.")
    logits = logits[0, prompt_len - 1 : int(input_ids.shape[1]) - 1]
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
        first_pass = []
        for example in examples:
            margin = margin_fn(example)
            if not isinstance(margin, torch.Tensor) or margin.numel() != 1:
                raise ValueError("margin_fn must return one scalar tensor per chunk.")
            first_pass.append(margin.detach().reshape(()))
        mean_margin = torch.stack(first_pass).mean()
        target = torch.as_tensor(float(label), device=mean_margin.device, dtype=mean_margin.dtype)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(mean_margin, target)
        dloss_dmean = torch.sigmoid(mean_margin) - target
    backward = backward_fn or (lambda value: value.backward())
    coefficient = dloss_dmean.detach() / len(examples)
    for example in examples:
        margin = margin_fn(example)
        if not isinstance(margin, torch.Tensor) or margin.numel() != 1:
            raise ValueError("margin_fn must return one scalar tensor per chunk.")
        backward(coefficient * margin.reshape(()))
    return {
        "mean_margin": float(mean_margin.item()),
        "loss": float(loss.item()),
        "dloss_dmean": float(dloss_dmean.item()),
        "num_chunks": len(examples),
    }
