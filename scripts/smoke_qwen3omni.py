"""Local de-risking smoke tests for the Qwen3-Omni backend (see QWEN3_OMNI_IMPLEMENTATION.md §6).

Designed to run on a disk- and VRAM-constrained box (e.g. a single RTX 4090 with ~45 GB free):
the full 30B-A3B weights are NEVER downloaded here.

  Tier A  - real processor only (small download, ~tens of MB). Drives one DAIC-style example
            through the repo prompt builder + collator and asserts:
              * the audio placeholder token round-trips through the tokenizer,
              * audio feature tensors (input_features / feature_attention_mask) exist + are sane,
              * label masking sets prompt positions to -100 and keeps only the label tokens.
            This catches the highest-risk unknowns (placeholder token, processor keys) cheaply.

  Tier B  - tiny RANDOM-config end-to-end (NO weight download). Loads only the model's
            config.json, shrinks it to a couple of layers / few experts, instantiates a random
            model, then exercises: disable_talker -> attention-only LoRA -> forward/backward ->
            save_pretrained -> reload -> one likelihood score. Validates every code path the new
            backend will rely on without ever touching 70 GB of real weights.

Ship gate: Tier A and Tier B both green => push repo + configs to the server. Tier C (4-bit
single-sample inference) is intentionally omitted here to respect the disk budget.

Usage:
    python scripts/smoke_qwen3omni.py --tier all
    python scripts/smoke_qwen3omni.py --tier A
    python scripts/smoke_qwen3omni.py --tier B --device cpu

Exit 0 on success; non-zero (with a clear message) on the first failed assertion.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.runtime import (
    AUDIO_PLACEHOLDER,
    build_prompt_text,
    build_training_text,
    render_user_prompt_text,
)
from src.model.collator import Qwen2AudioSFTCollator
from src.utils import internal_label_text_from_int, resolve_input_modality

DEFAULT_MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
SAMPLING_RATE = 16_000


# --------------------------------------------------------------------------------------
# Shared fixtures
# --------------------------------------------------------------------------------------
def _smoke_config(*, use_audio: bool = True, use_text: bool = True) -> dict:
    """A minimal in-memory config that satisfies the prompt builder + label helpers.

    Mirrors configs/daic_audio_text.yaml's prompt/labels blocks; no paths or datasets needed.
    """
    return {
        "data": {"use_audio": use_audio, "use_text": use_text},
        "labels": {"label_vocab_version": "legacy_english_labels"},
        "prompt": {
            "system": "You are a psychologist analyzing speech and transcript information for depression screening.",
            "user_template": (
                "{audio_context_block}\n"
                "{transcript_block}Based on the {decision_basis}, determine whether the subject "
                "is {label_descriptor}.\n{label_instruction}"
            ),
            "prompt_language": "english",
        },
    }


def _build_example(config: dict, *, with_audio: bool, label: int = 1) -> dict:
    """One collator-ready example dict (matches Qwen2AudioSFTCollator's expectations)."""
    transcript = "Lately I have felt very tired and I do not enjoy things the way I used to."
    user_text = render_user_prompt_text(config, transcript, is_subject_bundle=False)
    num_audios = 1 if with_audio else 0
    prompt_text = build_prompt_text(
        config["prompt"]["system"],
        user_text,
        num_audios=num_audios,
        use_audio=with_audio,
    )
    label_text = internal_label_text_from_int(config, label)
    # ~1 s of synthetic 16 kHz audio so the feature extractor produces a real mel tensor.
    audio_arrays = [np.zeros(SAMPLING_RATE, dtype=np.float32)] if with_audio else []
    return {
        "audio_arrays": audio_arrays,
        "training_text": build_training_text(prompt_text, label_text),
        "prompt_text": prompt_text,
        "sample_id": "smoke-0",
        "subject_id": "smoke-subject",
        "label": label,
    }


# --------------------------------------------------------------------------------------
# Tier A
# --------------------------------------------------------------------------------------
def tier_a(model_id: str) -> None:
    print(f"[Tier A] loading processor only from {model_id} (no weights) ...")
    try:
        from transformers import Qwen3OmniMoeProcessor

        processor = Qwen3OmniMoeProcessor.from_pretrained(model_id)
    except ImportError as exc:
        # Fall back so the prompt/collator plumbing can still be smoke-tested before the
        # transformers-from-source env exists; flag clearly that the *real* processor checks
        # (placeholder token, feature keys) did NOT run.
        raise SystemExit(
            "[Tier A] Qwen3OmniMoeProcessor not importable. Install transformers from source "
            "(pip install git+https://github.com/huggingface/transformers) and retry. "
            f"Underlying error: {exc}"
        )

    config = _smoke_config(use_audio=True, use_text=True)
    assert resolve_input_modality(config) == "audio_text"
    example = _build_example(config, with_audio=True, label=1)

    # 1) Audio placeholder must be REAL single special tokens, not literal text.
    # Qwen3-Omni differs from Qwen2-Audio here: it uses <|audio_start|>/<|audio_pad|>/<|audio_end|>,
    # NOT <|audio_bos|>/<|AUDIO|>/<|audio_eos|>. Derive the native placeholder from the processor
    # and assert each piece is a single special token, so a wrong placeholder fails loudly here
    # instead of silently misaligning audio features at forward time on the server.
    tok = processor.tokenizer
    native_tokens = [processor.audio_bos_token, processor.audio_token, processor.audio_eos_token]
    for token in native_tokens:
        ids = tok(token, add_special_tokens=False)["input_ids"]
        assert len(ids) == 1, (
            f"[Tier A] expected {token!r} to be a single special token, got ids={ids}. "
            "Processor/tokenizer mismatch."
        )
    native_placeholder = "".join(native_tokens)
    if AUDIO_PLACEHOLDER != native_placeholder:
        print(
            f"[Tier A] NOTE: repo AUDIO_PLACEHOLDER={AUDIO_PLACEHOLDER!r} != Qwen3-Omni "
            f"native={native_placeholder!r}. The qwen3omni backend must emit the native one."
        )
    # The single audio_token must expand to multiple frame positions once audio is attached.
    pad_id = tok(processor.audio_token, add_special_tokens=False)["input_ids"][0]
    probe = processor(
        text=native_placeholder,
        audio=[example["audio_arrays"][0]],
        sampling_rate=SAMPLING_RATE,
        return_tensors=None,
    )
    probe_ids = np.asarray(probe["input_ids"]).reshape(-1).tolist()
    n_pad = sum(1 for i in probe_ids if i == pad_id)
    assert n_pad > 1, (
        f"[Tier A] audio_token {processor.audio_token!r} did not expand to frame positions "
        f"(found {n_pad}); audio feature alignment would break."
    )
    print(f"[Tier A] placeholder OK -> native={native_placeholder!r} expands to {n_pad} frame positions")

    # 2) Collator produces audio feature tensors + label masking.
    collator = Qwen2AudioSFTCollator(processor=processor, debug=True)
    batch = collator([example])
    for key in ("input_ids", "attention_mask", "labels"):
        assert key in batch, f"[Tier A] missing batch key {key!r}"
    assert "input_features" in batch and "feature_attention_mask" in batch, (
        "[Tier A] processor did not yield input_features/feature_attention_mask. "
        f"Got keys: {sorted(batch.keys())}. Adapt the collator to Qwen3OmniMoeProcessor's "
        "audio output keys."
    )
    labels = batch["labels"][0]
    assert (labels == -100).any(), "[Tier A] no masked prompt positions in labels"
    assert (labels != -100).any(), "[Tier A] all label positions masked - nothing to learn"
    n_supervised = int((labels != -100).sum())
    print(
        f"[Tier A] collator OK | input_ids={tuple(batch['input_ids'].shape)} "
        f"input_features={tuple(batch['input_features'].shape)} supervised_tokens={n_supervised}"
    )
    print("[Tier A] PASS")


# --------------------------------------------------------------------------------------
# Tier B
# --------------------------------------------------------------------------------------
def _shrink_thinker_config(model_id: str):
    """Download only config.json and return a heavily shrunk THINKER config.

    The top-level Qwen3OmniMoeForConditionalGeneration has no standard CausalLM forward (it routes
    thinker+talker), so training/likelihood targets the Thinker. We instantiate
    Qwen3OmniMoeThinkerForConditionalGeneration directly from thinker_config - the talker is never
    built. No weights are fetched (config.json only).
    """
    from transformers import AutoConfig

    full = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    thinker = getattr(full, "thinker_config", full)

    def _shrink(node) -> None:
        for attr, value in {
            "num_hidden_layers": 2,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "decoder_sparse_step": 1,
            "hidden_size": 256,
            "intermediate_size": 256,
            "moe_intermediate_size": 256,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 64,
            "encoder_layers": 2,
        }.items():
            if hasattr(node, attr):
                setattr(node, attr, value)
        for sub_attr in ("text_config", "audio_config", "vision_config"):
            sub = getattr(node, sub_attr, None)
            if sub is not None and sub is not node:
                _shrink(sub)

    _shrink(thinker)
    return thinker


def tier_b(model_id: str, device: str) -> None:
    import torch
    from peft import LoraConfig, get_peft_model

    print(f"[Tier B] loading config.json only from {model_id} (no weights) and shrinking ...")
    try:
        from transformers import Qwen3OmniMoeThinkerForConditionalGeneration
    except ImportError as exc:
        raise SystemExit(
            "[Tier B] Qwen3OmniMoeThinkerForConditionalGeneration not importable. Install "
            f"transformers from source and retry. Underlying error: {exc}"
        )

    thinker_cfg = _shrink_thinker_config(model_id)
    print("[Tier B] instantiating RANDOM tiny THINKER from shrunk config (no checkpoint) ...")
    model = Qwen3OmniMoeThinkerForConditionalGeneration(thinker_cfg).to(device)

    # Attention-only LoRA (MoE expert FFNs deliberately excluded; see plan §3 caveat).
    lora = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    # One forward/backward on a trivially small text batch (text-only keeps Tier B audio-agnostic;
    # audio feature wiring is covered by Tier A with the real processor).
    text_cfg = getattr(thinker_cfg, "text_config", thinker_cfg)
    vocab = int(getattr(text_cfg, "vocab_size", getattr(thinker_cfg, "vocab_size", 32000)))
    input_ids = torch.randint(0, max(vocab, 2), (1, 16), device=device)
    labels = input_ids.clone()
    labels[:, :8] = -100  # mimic prompt masking
    model.train()
    out = model(input_ids=input_ids, labels=labels)
    assert out.loss is not None and torch.isfinite(out.loss), "[Tier B] non-finite training loss"
    out.loss.backward()
    grad_norm = sum(
        float(p.grad.norm()) for p in model.parameters() if p.requires_grad and p.grad is not None
    )
    assert grad_norm > 0.0, "[Tier B] zero gradient on LoRA params"
    print(f"[Tier B] forward/backward OK | loss={float(out.loss):.4f} lora_grad_norm={grad_norm:.4f}")

    # Save adapter -> reload -> one likelihood score (the headline eval path, in miniature).
    with tempfile.TemporaryDirectory() as tmp:
        model.save_pretrained(tmp)
        from peft import PeftModel
        from transformers import Qwen3OmniMoeThinkerForConditionalGeneration

        reload_base = Qwen3OmniMoeThinkerForConditionalGeneration(thinker_cfg).to(device)
        reloaded = PeftModel.from_pretrained(reload_base, tmp).eval()
        with torch.no_grad():
            scored = reloaded(input_ids=input_ids, labels=labels)
            logits = scored.logits[0]
            log_probs = torch.log_softmax(logits[7:15], dim=-1)
            token_lp = log_probs.gather(-1, input_ids[0, 8:16].unsqueeze(-1)).squeeze(-1)
        margin = float(token_lp.mean())
        assert np.isfinite(margin), "[Tier B] non-finite likelihood score after reload"
    print(f"[Tier B] save/reload/score OK | mean_token_logprob={margin:.4f}")
    print("[Tier B] PASS")


# --------------------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", choices=["A", "B", "all"], default="all")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", default="cpu", help="cpu or cuda (Tier B tiny model fits a 4090)")
    args = parser.parse_args()

    if args.tier in ("A", "all"):
        tier_a(args.model_id)
    if args.tier in ("B", "all"):
        tier_b(args.model_id, args.device)

    print("\nAll requested smoke tiers PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
