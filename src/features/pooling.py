from __future__ import annotations

import torch


def aligned_attention_mask(
    hidden: torch.Tensor,
    input_attention_mask: torch.Tensor,
    output_attention_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, str]:
    """Return a mask aligned to decoder hidden states.

    Qwen2-Audio can expand audio placeholders inside the model. For the primary
    extractor we process exactly one unpadded example at a time, so an expanded
    sequence is entirely valid even when the input text mask no longer aligns.
    """
    if hidden.ndim != 3:
        raise ValueError(f"Expected [batch, sequence, hidden] tensor, got {tuple(hidden.shape)}.")
    for mask, source in (
        (output_attention_mask, "model_output"),
        (input_attention_mask, "processor_input"),
    ):
        if mask is not None and tuple(mask.shape) == tuple(hidden.shape[:2]):
            return mask.to(device=hidden.device), source
    if hidden.shape[0] == 1 and input_attention_mask.shape[0] == 1:
        if not bool(torch.all(input_attention_mask == 1)):
            raise ValueError("Cannot synthesize an expanded audio mask from a padded input.")
        return torch.ones(hidden.shape[:2], dtype=torch.long, device=hidden.device), "batch1_all_valid"
    raise ValueError(
        "Decoder hidden states and attention mask do not align; use batch size 1 "
        "for audio-expanded extraction or provide the model's updated mask."
    )


def last_valid_token(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    if hidden.ndim != 3 or attention_mask.ndim != 2:
        raise ValueError("Expected hidden [B,S,H] and attention mask [B,S].")
    if tuple(hidden.shape[:2]) != tuple(attention_mask.shape):
        raise ValueError(
            f"Hidden/mask shape mismatch: {tuple(hidden.shape[:2])} versus {tuple(attention_mask.shape)}."
        )
    valid_counts = attention_mask.long().sum(dim=1)
    if bool(torch.any(valid_counts <= 0)):
        raise ValueError("Every extraction input must contain at least one valid token.")
    batch_indices = torch.arange(hidden.shape[0], device=hidden.device)
    return hidden[batch_indices, valid_counts - 1].float()
