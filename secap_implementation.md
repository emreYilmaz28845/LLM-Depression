# SECap Emotion-Augmented Prompting — Implementation Plan

**Status:** planning only (no code written yet).
**Goal:** add a 4th prompt mode `audio+text+emotion` that injects natural-language
emotional descriptions (from SECap, translated zh→en) into the depression-classification
prompt, following the emotion-augmented instruction recipe of *DepressInstruct*
(`Papers/DepresInstruct.pdf`) using SECap (`Papers/secap.pdf`).

---

## TL;DR / Recommendations

1. **Do emotion extraction OFFLINE and cache it.** SECap is a HuBERT + Q-Former +
   Chinese-LLaMA-7B stack that runs **8 stochastic generations per clip** — it cannot
   live in the dataloader (VRAM, speed) and it is **non-deterministic**, which would
   violate the repo's hard eval-determinism rule. Generate once, freeze, look up by key.
2. **One caption per chunk, cached per chunk**, keyed by `sample_id`. This is the atomic
   unit and matches the existing per-chunk manifest. Subject-level prompts then assemble
   the K per-chunk descriptions.
3. **Translation:** SECap emits Chinese. Add a zh→en stage. Recommend **`Helsinki-NLP/opus-mt-zh-en`**
   (Marian, ~300 MB, offline, batch-friendly) as primary; **`facebook/nllb-200-distilled-600M`**
   as the higher-fluency fallback. Cache **both** the Chinese caption and the English translation.
4. **K-chunk interaction is the crux.** Training re-samples K of N chunks per epoch in
   `AudioTextDataset.__getitem__`, but `prompt_text` is currently baked once at build time.
   If per-chunk descriptions are baked statically they will **misalign** with the randomly
   swapped audio. Fix: **re-render the prompt in `__getitem__`** so descriptions track the
   chunks actually fed. Eval stays on the baked deterministic view (already aligned).
5. **Implement as a `data.use_emotion` flag** layered on the existing modality resolution,
   not a new `INPUT_MODALITY`. `audio+text+emotion` = `use_audio + use_text + use_emotion`.
6. **Cross-lingual caveat:** SECap is trained on Mandarin (Chinese HuBERT + Chinese LLaMA).
   DAIC/EDAIC are English. Q-Former is designed to keep prosody and discard content, so
   transfer is plausible but **unvalidated** — gate the rollout behind a manual QC sample
   and a with/without-emotion ablation. CMDC/EATD (Chinese) are in-domain validation points.

---

## 1. Architecture Overview

### 1.1 Pipeline

```
                       OFFLINE (one-time, secap conda env, on cluster GPU)
 manifest chunk wav (16 kHz mono, ~30 s)
        │
        ▼
 ┌──────────────┐   Chinese caption    ┌──────────────────┐   English caption
 │   SECap      │ ───────────────────► │  zh→en translate │ ──────────────────►  cache
 │ inference()  │  情感描述: 语速较慢… │  (opus-mt-zh-en) │  "Speaking rate is        (JSONL, per sample_id)
 └──────────────┘                      └──────────────────┘   slow, voice calm…"
        ▲
   model.ckpt (15 GB)

                       TRAIN / EVAL (existing qwen_mn5 env)
 build_examples() ── loads emotion cache ── injects English description into prompt
        │
        ▼
 audio+text+emotion prompt ─► Qwen2-Audio-7B (LoRA) ─► Depressed / Non-depressed
```

### 1.2 Component responsibilities

| Component | New? | Env | Runs |
|---|---|---|---|
| SECap extractor (`MotionAudio.inference`) | wrap existing | `secap` (Chinese LLaMA stack) | offline, GPU, once |
| zh→en translator | new | `secap` (or a tiny standalone env) | offline, once |
| Emotion cache (JSONL) | new | — | artifact on disk |
| Cache loader + prompt injection | new | `qwen_mn5` (train/eval) | every run, lookup only |
| `data.use_emotion` flag + templates | new | `qwen_mn5` | every run |

**Key design principle:** the heavy/Chinese/non-deterministic SECap world is fully
decoupled from the training world. Training/eval never import SECap; they read a frozen
text cache. This mirrors how the repo already separates manifest building from training.

### 1.3 SECap I/O facts (verified from local code)

- Entry point: `MotionAudio().inference(wavform)` where `wavform = [np.float32 waveform @ 16 kHz]`
  (`SECap/scripts/inference.py`, `SECap/model2.py:183`).
- Encoder is **`TencentGameMate/chinese-hubert-large`**; decoder is **`minlik/chinese-llama-7b-merged`**
  (`model2.py:32-34`). Output language is **Chinese**.
- Fixed inference prompt: `"请用一句中文简述音频里说话者的情感表现："` ("briefly describe in one
  Chinese sentence the speaker's emotion") — `model2.py:210`. This is SECap's analogue of the
  DepressInstruct prompt *"Please describe the speaker's emotional state in one sentence."*
- `inference()` runs **8 generations** with `do_sample=True, top_k=10, top_p=0.95, num_beams=5`
  (`model2.py:228-246`), then `post_processing()` drops the 3 least-similar and **returns 5
  candidate sentences** + the prompt (`model2.py:252-254`). → **non-deterministic**; we must
  freeze one realization in the cache.
- DAIC/EDAIC chunks are already **16 kHz mono ~30 s** → no resampling needed (SECap resamples
  to 16 kHz anyway).

---

## 2. Required Code Changes (file-by-file)

### 2.1 New: offline emotion extraction (lives next to SECap / runs in `secap` env)
- `src/emotion/extract_secap.py` (or under `scripts/`): batch driver.
  - Load `MotionAudio`, `model.ckpt`; iterate **unique chunk wavs** from a manifest.
  - For each chunk: `cands, _ = model.inference([wav])`; pick a **canonical** caption
    (recommend the medoid of the 5 candidates via the existing `SimiCal`; store all 5).
  - Write JSONL row per `sample_id` with the Chinese caption + candidates + metadata.
  - Resumable (skip `sample_id`s already cached); shardable for SLURM array jobs.
  - Optional `--fast` mode (single greedy `do_sample=False` generation) for smoke tests only.
- `src/emotion/translate.py`: zh→en batch translation (see §5). Reads the SECap JSONL,
  adds `emotion_en`, writes the final cache. Can be folded into `extract_secap.py` as a
  second pass, but keep it a **separate stage/function** so a translation failure never
  forces re-running SECap.
- `src/emotion/build_emotion_cache.py`: thin orchestrator (extract → translate → validate
  coverage against the manifest). Analogous to `src/data/build_manifest.py`.

> These three files import SECap and therefore only run in the `secap` env. The training
> code (`src/...`) must **not** import them.

### 2.2 `src/utils.py`
- Add an emotion flag resolver, e.g. `use_emotion(config) -> bool` reading `data.use_emotion`
  (default `False`). Guard: emotion requires audio (it is derived from audio); raise if
  `use_emotion and not use_audio`.
- Optionally update `_decision_basis` to append "and emotional description(s)" when emotion is on.
- Add helpers for the new prompt placeholders (§3): `{emotion_block}` (single) and per-chunk
  rendering for subject mode.

### 2.3 `src/data/runtime.py` (the core change)
- **Cache loading:** add a loader `load_emotion_cache(path) -> dict[sample_id -> en_caption]`
  called once in `build_examples`. Pass the map down to the example builders.
- **Single-chunk path** (`_base_example_from_row`, `runtime.py:183`): look up the chunk's
  English description by `row["sample_id"]`; inject into the user prompt via a new
  `{emotion_block}` placeholder. Trivially aligned (one chunk ↔ one description).
- **Subject K-chunk path** (`_build_subject_level_audio_examples`, `runtime.py:318`):
  - Bake the deterministic prompt using the **deterministic evenly-spaced K chunks' captions**,
    in chunk order, interleaved with `Audio i:` (used verbatim by eval — aligned).
  - Additionally store on the example: `chunk_caption_by_path: {audio_path -> en_caption}`
    (or `{sample_id -> en}` + a path→sample_id map) covering the **full** `subject_chunk_paths`
    pool, so the training dataset can re-render after random sampling.
- **`_audio_prompt_block`** (`runtime.py:51`) / **`build_prompt_text`** (`runtime.py:133`):
  extend to accept an optional ordered list of per-audio emotion strings and **interleave**
  `Audio i:` with `Emotional description i:` (see §3.2).
- **`AudioTextDataset.__getitem__` / `_resolve_audio_plan`** (`runtime.py:704-744`):
  when `chunk_sampling == "random"` **and** emotion is on, after sampling the K paths,
  look up their captions and **re-render `prompt_text` + `training_text`** so the
  descriptions match the sampled audio order. When deterministic, keep the baked text.
  - This is the only way to keep per-chunk alignment under per-epoch chunk resampling.
  - `training_text` re-render is cheap (string format); the collator already reads
    `example["prompt_text"]` / `example["training_text"]` fresh each call
    (`src/model/collator.py:28-29`), so no collator change is needed.

### 2.4 Configs
- New presets, e.g. `configs/daic_subject_audio_text_emotion_*.yaml`,
  `configs/daic_audio_text_emotion.yaml`, EDAIC equivalents. Each adds:
  - `data.use_emotion: true`
  - `data.emotion_cache_path: ${PROJECT_ROOT}/outputs/emotion/${dataset}_secap_en.jsonl`
  - `data.emotion_on_missing: neutral_fallback` (see §5.4)
  - new `prompt.system` ("You are a clinical expert in depression assessment from speech,
    transcript, and emotional cues.") and a `prompt.user_template` /
    `prompt.subject_user_template` carrying the emotion placeholders.
- `src/utils.py` `OPTIONAL_OVERRIDE_PATHS` may need the new `data.*` keys so `--set` works.

### 2.5 Validation / sanity
- Extend `scripts/sanity_tests_no_model.sh` with an emotion-injection check (§9): coverage,
  per-chunk alignment, and fallback behavior — all without loading any model.

---

## 3. New Prompt Format

### 3.1 Single chunk (`sample_mode: chunk`, `audio+text+emotion`)

```
<|im_start|>system
You are a clinical expert in depression assessment from speech, transcript, and emotional cues.
<|im_end|>
<|im_start|>user
Audio 1: <|audio_bos|><|AUDIO|><|audio_eos|>
The subject's speech audio is provided.
Please determine whether the speaker is depressed or non-depressed based on the speech audio, transcript, and emotional description.
Emotional description: The speaking rate is slow, the voice is calm, and the emotion carries a sense of helplessness.
Transcript: ...
<|im_end|>
<|im_start|>assistant
```

### 3.2 Subject K-chunk (`sample_mode: subject_audio`) — per-chunk descriptions

```
<|im_start|>system
You are a clinical expert in depression assessment from speech, transcript, and emotional cues.
<|im_end|>
<|im_start|>user
Audio 1: <|audio_bos|><|AUDIO|><|audio_eos|>
Emotional description 1: ...
Audio 2: <|audio_bos|><|AUDIO|><|audio_eos|>
Emotional description 2: ...
Audio 3: <|audio_bos|><|AUDIO|><|audio_eos|>
Emotional description 3: ...
Audio 4: <|audio_bos|><|AUDIO|><|audio_eos|>
Emotional description 4: ...
The subject's speech audio segments are provided.
Please determine whether the speaker is depressed or non-depressed based on the speech audio segments, transcript, and emotional descriptions.
Transcript: ...
<|im_end|>
<|im_start|>assistant
```

This matches the user's target and DepressInstruct's "one emotion sentence per audio".
The interleaving (`Audio i:` immediately followed by `Emotional description i:`) makes the
chunk↔description binding explicit to the model.

### 3.3 Template mechanics
- Today `build_prompt_text` prepends a plain `Audio i:` block then the user text
  (`runtime.py:139-145`). For emotion mode, the audio block becomes an **interleaved** block
  built from `(audio_placeholder_i, emotion_i)` pairs.
- New placeholders: `{emotion_block}` for single-chunk; subject mode builds the interleaved
  block programmatically (not a simple `str.format`, because count is variable).
- Keep the non-emotion templates untouched; emotion templates are new presets.

---

## 4. Dataset Modifications

### 4.1 Emotion cache schema (per `sample_id`)

`outputs/emotion/<dataset>_secap_en.jsonl`, one row per chunk:

```json
{
  "dataset": "daic",
  "subject_id": "300",
  "sample_id": "300_random_segment_1",
  "audio_path": ".../300_random_segment_1.wav",
  "secap_prompt": "请用一句中文简述音频里说话者的情感表现：",
  "emotion_zh": "语速较慢，声音平缓，情绪中带着无奈",
  "emotion_zh_candidates": ["...", "...", "...", "...", "..."],
  "emotion_en": "The speaking rate is slow, the voice is calm, and the emotion carries a sense of helplessness.",
  "secap_model_version": "model.ckpt@<sha/size>",
  "translation_model": "Helsinki-NLP/opus-mt-zh-en",
  "translation_ok": true
}
```

- **Key = `sample_id`** (portable across local/cluster; `audio_path` differs between
  `/media/emre/Backup/...` and `/gpfs/...`). Store `audio_path` for debugging only.
- Version fields let the loader detect a stale cache and refuse/regenerate.

### 4.2 Injection points
- `build_examples` (`runtime.py:438`) loads the cache once and threads the
  `sample_id -> emotion_en` map into both builders.
- Single-chunk / EATD response mode: inject by `sample_id`.
- Subject mode: bake deterministic K descriptions + carry the full per-chunk map for
  training re-render (§2.3).

### 4.3 K-chunk alignment — the critical analysis (user's Q1/Q2/Q3)

**Recommendation: option 1 (one description per chunk), with dynamic prompt re-rendering.**
Cache is per chunk regardless. For the *prompt*:

- **Per-chunk (recommended, primary).** Insert the K descriptions of the chunks actually
  used. Correct because emotion is a property of each clip. **Requires** re-rendering in
  `__getitem__` under `chunk_sampling="random"` (training), because the baked `prompt_text`
  is fixed to the deterministic chunk view while training swaps in different chunks each epoch.
  Without re-render, "Emotional description 1" would describe a chunk whose audio is **not**
  the `Audio 1` being fed → silent label-irrelevant noise / mislabeling. Eval uses the baked
  deterministic view, so it is already aligned with no extra work.
- **Subject-level merged summary (option 2, fallback / ablation).** One emotion paragraph
  summarizing the subject, invariant to which chunks are sampled → `prompt_text` stays static,
  no re-render, no alignment risk. Cost: a summarization step (concatenate the K/all captions,
  or LLM-summarize) and loss of per-chunk granularity. **Must be chunk-count-agnostic**
  (never say "across 15 segments") to avoid re-introducing the DAIC chunk-count→label leak.
- **Both (option 3).** Useful only as an ablation; start with per-chunk.

Decision: ship **per-chunk + dynamic re-render** as the default (matches the user's target
format and DepressInstruct). Keep **subject-summary** as a config-selectable ablation
(`data.emotion_granularity: per_chunk | subject_summary`) and as the safe fallback if dynamic
re-render proves troublesome.

> Aside: the existing subject-mode prompt is generic ("K segments sampled from the interview")
> and does **not** name chunks, which is why audio swapping is currently harmless. Adding
> per-chunk text is exactly what breaks that invariance — hence the re-render requirement.

---

## 5. Translation Pipeline

### 5.1 Why translation is needed
SECap's decoder is Chinese LLaMA with a Chinese prompt → Chinese captions
(`model2.py:210`). DepressInstruct presents English emotion sentences, so it (implicitly)
translated. The repo's prompts are English. → a zh→en stage is required.

### 5.2 Options compared

| Option | Size | Quality (short emotive zh) | Offline | Batch on cluster | Notes |
|---|---|---|---|---|---|
| **Helsinki-NLP/opus-mt-zh-en** (Marian) | ~77 M / ~300 MB | good for short sentences | yes | excellent (fast, CPU-OK) | lightest; occasional idiom slips |
| **facebook/nllb-200-distilled-600M** | 600 M | higher fluency, robust on idioms | yes | good (GPU preferred) | heavier but still small; explicit src/tgt lang codes |
| Qwen2.5-Instruct as translator | 7B+ | highest, controllable | yes | reuses existing infra | overkill; heavy; non-deterministic unless greedy |

### 5.3 Recommendation
- **Primary: `Helsinki-NLP/opus-mt-zh-en`** — lightweight, offline, deterministic (greedy),
  trivially batched; ~5 k short sentences in well under a minute. Negligible vs SECap cost.
- **Fallback for fluency: `facebook/nllb-200-distilled-600M`** (`zho_Hans`→`eng_Latn`).
  Worth using if QC shows opus-mt produces stilted English on emotional idioms.
- Reuse `Qwen2.5` only if you want one model to *both* clean and translate; not needed initially.

### 5.4 Operational answers to the prompt's questions
- **Run translation immediately after SECap?** Run it as a **separate pass over the cached
  Chinese captions**, not interleaved per-clip. This isolates failures (a translation crash
  never forces re-running the expensive SECap pass) and lets you swap translators without
  re-extracting.
- **Cache both zh and en?** **Yes.** `emotion_zh` (+ candidates) for audit/reproducibility/QC
  and to allow re-translation; `emotion_en` for the prompt.
- **Translation failures?** Deterministic greedy decode; on failure: retry once, then set
  `emotion_en=null, translation_ok=false` and keep `emotion_zh`. At prompt-build time, if
  `emotion_en` is missing, fall back per `data.emotion_on_missing`:
  - `neutral_fallback` (default): a fixed neutral sentence
    (e.g. *"No reliable emotional description is available for this segment."*) so training
    never crashes and the model can learn to ignore it.
  - `drop_emotion_line`: omit that chunk's `Emotional description i:` line.
  - `error`: hard-fail (use during cache QC).
  Log counts of fallbacks; a high rate is a red flag for the whole approach.

---

## 6. Caching Strategy: offline vs online

### 6.1 Comparison

| | **Offline (recommended)** | Online (in dataloader) |
|---|---|---|
| When | once, before training | every `__getitem__` |
| Determinism | frozen → satisfies eval rule | SECap `do_sample=True` → **violates** eval rule |
| VRAM | SECap GPU job separate from training | SECap-7B + Qwen2-Audio-7B co-resident → infeasible |
| Speed | amortized one-time | 8 generations/clip × K × every step → intractable |
| Env | runs in `secap` env, isolated | forces SECap deps into training env (conflict) |
| QC | captions inspectable before training (DepressInstruct does manual checking) | none |
| Cost of repeat runs | zero (reuse cache) | pays full cost every epoch |

**Online is rejected on every axis** (determinism, memory, speed, env). The repo's
eval-determinism rule (`experiment_handoff.md:123`, and the project memory note) alone is
decisive: SECap sampling in the eval path would make val/test metrics random.

### 6.2 Cost estimate for offline extraction (one-time)
- Chunk counts (measured): **DAIC 2170**, **EDAIC 3080** (≈5250). +CMDC/EATD if extended.
- Per clip SECap ≈ 8 sampled generations (≤50 tokens, beam 5) + similarity post-proc.
  Rough order **~3–8 s/clip** on one modern GPU → **~5–12 GPU-hours** total for DAIC+EDAIC,
  fully **parallelizable** across GPUs / SLURM array shards (e.g. 8 shards → <2 h wall).
  A `--fast` single-greedy mode cuts this ~8× for smoke tests.
- Translation: **negligible** (seconds–minutes for ~5 k short sentences).
- Storage: a few MB of JSONL. Re-runs of training cost **zero** extra emotion compute.

### 6.3 Cache hygiene
- Version stamp (`secap_model_version`, `translation_model`, `secap_prompt`); loader warns/refuses
  on mismatch.
- One cache file per dataset under `outputs/emotion/`.
- Coverage check: every manifest `sample_id` must have a cache row before training (else fallback).

---

## 7. Training / Inference Flow

```
STEP 0 (offline, secap env, cluster GPU, once)
  python src/emotion/extract_secap.py  --manifest outputs/manifests/daic_manifest.jsonl \
                                       --ckpt /gpfs/projects/etur92/SECap/model.ckpt \
                                       --out  outputs/emotion/daic_secap_zh.jsonl   [--shard i/N]
  python src/emotion/translate.py      --in  outputs/emotion/daic_secap_zh.jsonl \
                                       --out outputs/emotion/daic_secap_en.jsonl \
                                       --model Helsinki-NLP/opus-mt-zh-en
  python src/emotion/build_emotion_cache.py --validate-against outputs/manifests/daic_manifest.jsonl

STEP 1 (train, qwen_mn5 env) — unchanged commands + emotion config
  torchrun ... src/train.py --config configs/daic_subject_audio_text_emotion_k4.yaml --fold 0
     └─ build_examples loads outputs/emotion/daic_secap_en.jsonl
     └─ train: chunk_sampling="random" + re-render prompt per sampled chunks
     └─ val/test: deterministic baked prompt (aligned), frozen captions

STEP 2 (eval, qwen_mn5 env) — unchanged, reads same cache
  python src/evaluate.py --config configs/..._emotion_k4.yaml --checkpoint_dir .../best_model
```

- **Env separation is hard:** Step 0 uses `/home/emre/miniconda3/envs/secap` (local smoke)
  / the cluster SECap env; Steps 1–2 use `qwen_mn5`. No cross-imports.
- **Determinism preserved:** train re-render uses the seeded dataset RNG; eval uses baked
  deterministic text + frozen cache. No SECap call in the train/eval path.
- Eval path (`src/evaluate.py:658` → `build_examples` → `example["prompt_text"]`) needs no
  change beyond loading the cache, because it already consumes baked prompts deterministically.

---

## 8. Risks and Failure Modes

1. **Cross-lingual domain shift (highest risk).** SECap = Chinese HuBERT + Chinese LLaMA,
   trained on Mandarin emotional speech; DAIC/EDAIC are English. Captions may be generic,
   biased, or wrong. *Mitigations:* manual QC on a random sample (DepressInstruct manually
   checked all captions); validate first on **CMDC/EATD (Chinese, in-domain)**; require a
   with/without-emotion **ablation** before trusting gains; treat the English result as
   experimental.
2. **Violating the eval-determinism rule.** SECap is stochastic. *Mitigation:* offline only,
   frozen cache; no SECap import in train/eval; cache version-stamped. (See project memory:
   "val/test metrics must be deterministic".)
3. **Per-chunk misalignment under random K-sampling.** Baked static descriptions + swapped
   audio. *Mitigation:* dynamic re-render in `__getitem__` (§4.3); add an alignment unit test (§9).
4. **New overfitting shortcut / leakage.** Project memory ("audio is the overfit liability")
   warns the audio path overfits speaker/site. Emotion text is audio-derived and could add
   another overfittable shortcut, *or* help — unknown a priori. *Mitigations:* heavy reg
   presets as elsewhere; ablation; ensure summaries (if used) are **chunk-count-agnostic** so
   the DAIC chunk-count→label leak isn't re-introduced via text.
5. **Local SECap env broken (CUDA/deps).** Per the brief, local may be unusable.
   *Mitigation:* plan for the **cluster** as the real target; use local only for a `--fast`
   single-clip smoke test; don't sink time into local CUDA fixes. The 15 GB `model.ckpt` and
   `weights/` already exist locally and on `/gpfs/projects/etur92/SECap`.
6. **Translation quality / idioms.** Short emotive Chinese can translate stiffly.
   *Mitigation:* opus-mt → NLLB fallback; QC; cache zh to allow re-translation without re-extract.
7. **Cache/manifest drift.** Re-preprocessing changes `sample_id`s or chunk sets.
   *Mitigation:* coverage validation step; version stamps; key by `sample_id`.
8. **Prompt length blow-up.** K interleaved descriptions + transcript + K×~750 audio tokens
   can crowd context. *Mitigation:* one **short** sentence per chunk (SECap already targets one
   sentence); keep `transcript_max_chars`; monitor the existing audio-budget audit.
9. **`model.inference()` prints / returns 5 sentences.** Not a clean API. *Mitigation:* wrap
   it (don't call `test_step`); select a canonical caption (medoid); store candidates.

---

## 9. Validation Plan

**A. SECap smoke test (local, `--fast`, may be skipped if env broken).**
Run one DAIC chunk through `model.inference` → confirm a Chinese caption is produced; then
opus-mt → confirm an English sentence. Proves the I/O contract end-to-end on one file.

**B. Translation spot-check.** Translate ~30 cached captions; eyeball fluency/faithfulness;
decide opus-mt vs NLLB.

**C. Cache coverage (no model).** Assert every manifest `sample_id` (DAIC, EDAIC) has a cache
row; report fallback count. Wire into `scripts/sanity_tests_no_model.sh`.

**D. Prompt-injection / alignment unit tests (no model).**
- Single-chunk: rendered prompt contains the chunk's English description.
- Subject deterministic: K `Emotional description i:` lines match the K deterministic chunks,
  in order.
- Subject random (seeded): after `chunk_sampling="random"`, the re-rendered descriptions match
  the **sampled** chunk order (the alignment guarantee of §4.3).
- Fallback: a missing `emotion_en` yields the configured neutral fallback, not a crash.

**E. Determinism check.** Two eval runs on the same checkpoint/config produce identical
prompts and metrics (cache frozen, eval deterministic).

**F. Modeling ablation (the real validation).** On DAIC (and EDAIC) subject-audio reg preset:
`audio+text` vs `audio+text+emotion`, same seeds/folds, compare headline macro-F1 / strict
metrics. Repeat on CMDC/EATD (Chinese, in-domain for SECap) as a sanity ceiling. Only adopt
emotion if it helps without inflating the train/val gap (overfit guard).

---

## Open Decisions (need user input before coding)

1. **Emotion granularity default:** per-chunk (recommended) vs subject-summary vs both-as-ablation.
2. **Translator:** opus-mt-zh-en (recommended) vs NLLB-600M — or decide after the §9B spot-check.
3. **Canonical caption selection** from SECap's 5 candidates: medoid (recommended) vs first vs
   keep-all-and-let-prompt-use-one.
4. **Datasets in scope:** DAIC + EDAIC only, or also CMDC/EATD (recommended for in-domain
   validation, adds ~extraction cost).
5. **Missing-caption policy:** `neutral_fallback` (recommended) vs `drop_emotion_line`.
6. **System prompt wording** for the emotion mode (proposed in §2.4 / §3).

---

## Appendix — measured facts

- DAIC manifest: 2170 chunk rows, 189 subjects, 10–15 chunks/subject (count encodes label:
  non-dep=10, dep=15 → keep K fixed; never expose count in text).
- EDAIC manifest: 3080 rows, 275 subjects, 10–15 chunks/subject.
- Chunk wavs: 16 kHz mono ~30 s PCM → directly SECap-compatible.
- SECap assets: `model.ckpt` (~15 GB) + `weights/` present locally
  (`/media/emre/Backup/AudioLLM/SECap`) and on cluster (`/gpfs/projects/etur92/SECap`).
- Repo determinism rule: `experiment_handoff.md:123`; eval reads baked deterministic
  `audio_paths`; training randomness confined to `AudioTextDataset.__getitem__`.
- Relevant code anchors: prompt build `src/data/runtime.py:51,133`; single-chunk example
  `:183`; subject example `:318`; random sampling `:704`; collator reads per-example
  `prompt_text`/`training_text` `src/model/collator.py:28`.
```
