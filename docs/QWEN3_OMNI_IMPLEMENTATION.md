# Qwen3-Omni Integration Plan

Goal: add **Qwen/Qwen3-Omni-30B-A3B-Instruct** (Thinker-only) as an audio+text / audio-only
backend in this repo, alongside the existing `Qwen2-Audio-7B-Instruct` path, without breaking
the text-only path. Headline metric stays likelihood-based (`Depressed` vs `Non-depressed`
token log-prob); generation is secondary, exactly as today.

This is the real-and-supported successor of the model the repo already uses, so the existing
**encoder + projector + decoder** pattern transfers conceptually. The work is a new backend
module plus a handful of name/shape adaptations — not a rewrite.

**Local env:** a `qwen3omni` conda env already exists for the smoke tests
(`/home/emre/miniconda3/envs/qwen3omni`, transformers-from-source + torch + peft). Run anything
local with `CONDA_ENV=qwen3omni`. This is for plumbing/correctness only — the 30B never runs here.

### Ablation note — text-only as a same-backbone control

The everyday text-only path stays on the cheap dedicated model (`text_lora` + dense
`Qwen2-7B-Instruct`); running the 35B omni model just to read a transcript is pure overkill.

**But for the ablation study, run Qwen3-Omni-Thinker in all three modes — audio+text, audio-only,
and text-only — on the one backbone.** Today's text-only baseline uses a *different* model
(Qwen2-7B) than the audio model (Qwen2-Audio), which confounds "audio vs text" with "model A vs
model B." Same-backbone text-only is the clean **no-audio control**: the audio-mode lift is then
attributable purely to the audio, not to a model swap — directly relevant to the documented
"audio is the overfit liability" finding.

Wiring caveat: this control does **not** go through `text_lora` (that path uses `AutoTokenizer`
/`AutoModelForCausalLM` and won't route into the omni Thinker). Run it through the new
`qwen3omni` backend with `data.use_audio=false` — i.e. the omni model fed text-only, no audio
placeholder and no `input_features`.

---

## 1. Variant decision (locked)

Use **Instruct, Thinker-only**:

- Load `Qwen3OmniMoeForConditionalGeneration`, then call **`model.disable_talker()`** (saves
  ~10 GB, drops speech synthesis we never use). Likelihood scoring only needs the Thinker's LM head.
- **Not Thinking**: it emits chain-of-thought before the answer, so the label token is no longer
  the immediate next token after `assistant\n` — that fights [likelihood scoring](src/evaluate.py#L210-L229)
  and our eval-determinism rule. Revisit only if we pivot to generative-reasoning eval.
- **Not Captioner**: audio-only captioning specialist, off-label as a classifier. Keep it on the
  shelf as an optional audio-description generator for an emotion-injection ablation (mirrors the
  SECap emotion path), not as the head.

### Model facts (verified from the model card)

| Item | Value |
|---|---|
| Class / processor | `Qwen3OmniMoeForConditionalGeneration` / `Qwen3OmniMoeProcessor` |
| Params | MoE ~35B total, ~3B active (30B-A3B) |
| Arch | Thinker–Talker, AuT audio encoder + projector → MoE text decoder |
| Precision | BF16 (no official FP8; ~23 community quants exist) |
| transformers | install from source (`pip install git+https://github.com/huggingface/transformers`) |
| Attention | `flash_attention_2` required for the quoted memory numbers |
| Inference VRAM (BF16, FA2) | Instruct ~78–89 GB for 15–30 s media; lower for our short audio chunks but weights alone ≈ ~70 GB |

---

## 2. Hardware reality

- **Server (training): 4×H100 80 GB now, can go to 8+.** ~70 GB of BF16 weights does *not*
  leave enough room on a single 80 GB card for training activations + grad checkpointing, so we
  **must shard** (FSDP or `device_map`). LoRA-only is mandatory; full fine-tune is out. 4×H100 is
  workable; 8×H100 gives comfortable headroom for larger batch / longer audio.
- **Local box: single RTX 4090 24 GB → cannot load the full model**, not even for inference
  (weights alone ≈ 3× the card). So local work is **plumbing/correctness smoke tests only**
  (Section 6), never a real run. The real LoRA training and the real metrics happen on the server.

---

## 3. What's coupled to Qwen2-Audio today (the change surface)

| Concern | Current (Qwen2-Audio) | Qwen3-Omni change |
|---|---|---|
| Model class | `Qwen2AudioForConditionalGeneration` ([qwen2audio_lora.py:239](src/model/qwen2audio_lora.py#L239)) | `Qwen3OmniMoeForConditionalGeneration` + `disable_talker()` |
| Module access | `base_model.audio_tower`, `base_model.multi_modal_projector` ([:76](src/model/qwen2audio_lora.py#L76), [:139](src/model/qwen2audio_lora.py#L139)) | nested under **`.thinker`** (e.g. `model.thinker.audio_tower`, `model.thinker.…projector`) |
| `_unwrap_base_model` | strips PEFT wrapper | also descend into `.thinker` |
| Processor | `AutoProcessor` → mel `input_features` + `feature_attention_mask` ([collator.py:90-94](src/model/collator.py#L90-L94)) | `Qwen3OmniMoeProcessor`; verify key names + audio arg API |
| Audio placeholder | `<\|audio_bos\|><\|AUDIO\|><\|audio_eos\|>` ([data/runtime.py:39](src/data/runtime.py#L39)) | **verify** Qwen3-Omni's placeholder token; update if different |
| Chat template | hand-built ChatML `<\|im_start\|>/<\|im_end\|>` ([data/runtime.py:164-171](src/data/runtime.py#L164-L171)) | same ChatML family; confirm system/assistant turn markers unchanged |
| LoRA layer count | `config.text_config.num_hidden_layers` ([lora_common.py:8-25](src/model/lora_common.py#L8-L25)) | thinker text config is **nested deeper** (e.g. `config.thinker_config.text_config.num_hidden_layers`) — extend the resolver |
| LoRA exclude regex | `.*audio_tower.*\|.*multi_modal_projector.*` ([lora_common.py:89](src/model/lora_common.py#L89)) | update to match Qwen3-Omni encoder/projector module names |
| LoRA targets | `q/k/v/o_proj` + `gate_proj/up_proj/down_proj` ([config](configs/daic_audio_text.yaml#L43-L50)) | **MoE caveat below** — start attention-only |
| Adapter dim | `audio_tower.config.d_model` ([:87](src/model/qwen2audio_lora.py#L87)) | read Qwen3-Omni encoder hidden size |

### MoE / LoRA targeting caveat (important)

In an MoE decoder, `gate_proj`/`up_proj`/`down_proj` live **inside every expert**. Targeting them
attaches LoRA to *all* experts across all layers — a huge adapter blow-up and a routing-instability
risk on tiny clinical data. **Start with attention-only LoRA** (`q_proj,k_proj,v_proj,o_proj`) for
the Qwen3-Omni configs. Add expert FFN targets later only as a deliberate ablation. (Also note: the
MoE router is usually named `gate`, distinct from the FFN `gate_proj` — don't accidentally LoRA the
router.)

---

## 4. Implementation steps

### 4.1 New backend module `src/model/qwen3omni_lora.py`
Copy `qwen2audio_lora.py` and adapt:
- Target the **Thinker** as the trainable CausalLM (smoke §5.4), not the omni wrapper. Preferred:
  `Qwen3OmniMoeThinkerForConditionalGeneration` (talker never built). If the Thinker class can't load
  weights straight from the Instruct repo, fall back to loading
  `Qwen3OmniMoeForConditionalGeneration(... attn_implementation="flash_attention_2")` and taking
  `.thinker`. Either way, `get_peft_model` wraps the Thinker.
- `_unwrap_base_model` → descend into `.thinker` when the wrapper is used.
- Emit the **native audio placeholder** `<|audio_start|><|audio_pad|><|audio_end|>` (smoke §5.1).
- `attach_dep_adapter` / `configure_trainable_audio_modules` / `enforce_audio_encoder_freeze`:
  repoint `audio_tower` and projector lookups to the thinker submodules and the correct names;
  keep the freeze-guard semantics (still valuable — encoder LoRA leak is the documented overfit trap).
- Keep `save_additional_audio_modules` / projector save-load, repointed.

### 4.2 Backend selector `src/model/runtime.py`
Add an explicit `model_backend` config switch (`qwen2audio` | `qwen3omni` | `text`, default
`qwen2audio`) so `_backend()` can return the new module ([runtime.py:13-16](src/model/runtime.py#L13-L16)).
The explicit switch must **win over the modality default**: text-only normally routes to
`text_lora`, but the same-backbone control (§1 ablation note) sets `model_backend: qwen3omni`
with `data.use_audio=false` so text-only is served by the omni Thinker, not the dense text model.
When `model_backend` is unset, keep today's behavior (modality → `text_lora` for text-only).

### 4.3 Collator `src/model/collator.py`
- Confirm `Qwen3OmniMoeProcessor(text=…, audio=…, sampling_rate=…)` returns `input_features` +
  `feature_attention_mask` (or adapt key names / use the documented `process_mm_info` helper).
- Verify the per-sample concat/pad of audio features still holds; adjust dtype/shape if the new
  feature extractor differs. Label-masking logic ([collator.py:46-47](src/model/collator.py#L46-L47)) is unchanged.

### 4.4 LoRA resolver `src/model/lora_common.py`
- Extend `_resolve_decoder_hidden_layer_count` to look under `thinker_config.text_config`.
- Update the default `exclude_modules` regex to the real Qwen3-Omni encoder/projector names.

### 4.5 Prompt builder `src/data/runtime.py`
- If the audio placeholder token differs, update `AUDIO_PLACEHOLDER` ([:39](src/data/runtime.py#L39)).
- Confirm ChatML turn markers; adjust `build_prompt_text` / `build_training_text` if needed.

### 4.6 Configs
For the ablation grid, add one config per mode so all three share the omni backbone:
- `configs/daic_audio_text_qwen3omni.yaml` — `use_audio: true`, `use_text: true`
- `configs/daic_audio_only_qwen3omni.yaml` — `use_audio: true`, `use_text: false`
- `configs/daic_text_only_qwen3omni.yaml` — `use_audio: false`, `use_text: true` (no-audio **control**)

(+ edaic siblings.) All set:
- `model_name_or_path: Qwen/Qwen3-Omni-30B-A3B-Instruct`, `model_backend: qwen3omni`.
- LoRA `target_modules: [q_proj,k_proj,v_proj,o_proj]` (attention-only, see §3 caveat).
- Keep `bf16: true`, `gradient_checkpointing: true`, `per_device_batch=1`, grad-accum as today.

Keep the existing dense-`Qwen2-7B` text-only configs untouched as the everyday path; the
`*_text_only_qwen3omni.yaml` config exists only as the same-backbone control.

### 4.7 Evaluate
No change expected — likelihood scoring is LM-head only ([evaluate.py:210-229](src/evaluate.py#L210-L229)).
Verify the label tokens (`Depressed`/`Non-depressed`) tokenize sanely under the new tokenizer.

---

## 5. Unknowns — RESOLVED locally via the smoke gate

The smoke tests (Section 6) were run against the real processor + config (no weights), resolving
most of Section 5 before touching the server:

1. **Audio placeholder is DIFFERENT from Qwen2-Audio.** Qwen3-Omni uses
   **`<|audio_start|><|audio_pad|><|audio_end|>`** (single special tokens 151669 / 151675 / 151670),
   NOT `<|audio_bos|><|AUDIO|><|audio_eos|>`. The old string tokenizes as *literal text* and would
   silently misalign audio features. The processor exposes these as `processor.audio_bos_token` /
   `processor.audio_token` / `processor.audio_eos_token`; the single `<|audio_pad|>` expands to N
   frame positions when audio is attached (13 for ~1 s). **The `qwen3omni` backend must emit the
   native placeholder** (don't reuse `src/data/runtime.AUDIO_PLACEHOLDER`).
2. **ChatML carries over.** `<|im_start|>` (151644) / `<|im_end|>` (151645) are single special
   tokens, same as Qwen2 — `build_prompt_text`'s turn structure is reusable.
3. **Processor audio keys carry over.** `Qwen3OmniMoeProcessor(text=…, audio=…, sampling_rate=…)`
   yields `input_features` + `feature_attention_mask`, so the existing collator shape logic holds.
4. **Train/score the Thinker, not the wrapper.** `Qwen3OmniMoeForConditionalGeneration` has **no
   standard `forward(input_ids=…)`** (it routes thinker+talker and errored with
   `_forward_unimplemented`). Use the standalone **`Qwen3OmniMoeThinkerForConditionalGeneration`**
   (and its `thinker_config`) as the trainable CausalLM. This is cleaner than `disable_talker()` —
   the talker is never built. LoRA + forward/backward + `save_pretrained` + reload + likelihood
   scoring all verified on a tiny random Thinker.
5. **Config nesting**: decoder layer count lives at `config.thinker_config.text_config.num_hidden_layers`
   — extend `_resolve_decoder_hidden_layer_count` accordingly (§4.4).

Still confirm on the server with real weights: exact `audio_tower`/projector submodule names under
the Thinker (for the exclude regex + freeze guard), and whether the Thinker class can load its
weights directly from the full Instruct checkpoint or must be pulled from the loaded omni model's
`.thinker`. A 3-line `print(model)` on first load settles both.

### Environment pins discovered
- `transformers` from source (5.13.0.dev0 verified) **requires a matching peft from source**
  (peft 0.19.1 release crashes on adapter reload with
  `WeightConverter.__init__() got an unexpected keyword argument 'distributed_operation'`;
  **peft 0.19.2.dev0 from git fixes it**). Pin both from source together.
- The omni `Qwen3OmniMoeProcessor` pulls an image-processor sub-component → **`Pillow` + `torchvision`
  are required** even for audio-only use.

---

## 6. Local 4090 smoke tests (before shipping to the server)

We can't run the 30B model locally, but we can de-risk almost all the *code* on the 4090/CPU.
Add `scripts/smoke_qwen3omni.py` (and a `scripts/smoke_qwen3omni.sh` wrapper) with three tiers:

**Tier A — processor + prompt + collator (real processor, CPU, minutes).**
Download only the *processor/tokenizer* (small), build one DAIC example through
`render_user_prompt_text` → `build_prompt_text` → the collator, and assert:
- the audio placeholder token round-trips through the tokenizer,
- `input_features` / `feature_attention_mask` keys exist with sane shapes,
- label masking sets prompt positions to `-100` and leaves only the label tokens.
This catches the highest-risk items (§5.2, §5.3) cheaply, with no GPU weights.

**Tier B — tiny random-config end-to-end (CPU or 4090, minutes).**
Instantiate `Qwen3OmniMoeForConditionalGeneration` from a **shrunk config** (e.g. 2 thinker layers,
tiny hidden, few experts) with random weights — no checkpoint download. Run:
`disable_talker()` → `get_peft_model` (attention-only LoRA) → one forward/backward on a collated
batch → `save_pretrained` → reload via the inference path → one likelihood score.
This exercises every code path we wrote (unwrap, freeze guard, adapter save/load, layer resolver)
without ever needing 70 GB of real weights. Wire it into a `pytest` so it runs in CI-style.

**Tier C — optional 4-bit single-sample inference (4090, tight).**
A community 4-bit quant of the 30B (~18–20 GB) *may* fit on the 4090 for a no-grad forward with
`disable_talker()`. If it loads, run one real likelihood score end-to-end to sanity-check that the
*actual* tokenizer/encoder produce a finite `Depressed`/`Non-depressed` margin. Treat as best-effort:
if it OOMs, skip — Tiers A+B are the gate, not C. Never train in 4-bit here; this is a smoke check only.

Run via the existing local env:
```bash
CONDA_ENV=qwen3omni ./scripts/smoke_qwen3omni.sh              # Tier A + B, CPU
CONDA_ENV=qwen3omni DEVICE=cuda ./scripts/smoke_qwen3omni.sh  # Tier B tiny model on the 4090
```

**Gate to ship:** Tier A and Tier B green ⇒ push repo + configs to the server. Tier C is a bonus signal.

---

## 7. Server run plan

1. **Env**: build a sibling of the MN5 env with `transformers` from source + `flash-attn`; pin and
   capture via `scripts/capture_environment.sh`. Keep the existing Qwen2-Audio env intact so old
   results stay reproducible.
2. **Sharding**: LoRA + FSDP (full-shard) or `device_map="auto"` across the GPUs under `torchrun`.
   Start at 4×H100; if activations/audio length push memory, move to 8×H100 rather than shrinking
   the model. Keep `gradient_checkpointing: true`, `per_device_batch=1`, grad-accum for effective batch.
3. **First job**: a single DAIC fold, attention-only LoRA, short max audio, to validate the §5
   unknowns and measure real VRAM before launching the full grid.
4. **Then**: replicate the existing DAIC/EDAIC (and audio-only) grid with the new backend. These are
   **new experiments** — they do not invalidate or overwrite the Qwen2-Audio checkpoints/results.

---

## 8. Acceptance criteria

- [ ] Tier A + B smoke tests pass on the local box.
- [ ] One DAIC fold trains on the server without OOM; freeze-guard logs show **0 trainable LoRA params
      under the audio encoder**.
- [ ] Likelihood eval produces a finite, deterministic `Depressed − Non-depressed` margin and a
      subject-level AUROC on official `test`.
- [ ] Results land in a **new** results table column; Qwen2-Audio numbers untouched.

## 9. Risks & watch-items

- **Overfitting**: 35B capacity on tiny clinical corpora amplifies the documented audio-overfit
  liability. Mitigate with attention-only LoRA, frozen audio encoder (keep the guard), small rank.
- **MoE LoRA blow-up / routing instability** if expert FFNs are targeted — avoid initially (§3).
- **transformers-from-source churn** — pin a commit; a moving API can silently change module names.
- **Audio length × MoE memory** — long subject-audio bundles may force 8 GPUs; budget for it.
- **Tokenizer differences** — confirm the label words tokenize to stable, short token sequences.
