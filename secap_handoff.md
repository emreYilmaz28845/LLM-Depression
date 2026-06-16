# SECap Emotion-Augmented Prompting — Handoff

**Status (2026-06-16):** code complete and unit-tested (no model). The offline
SECap cache has **not** been generated yet, and the with/without-emotion ablation
(the real validation) has **not** been run.

This doc lets a fresh agent continue. The original design lives in
[`secap_implementation.md`](secap_implementation.md); this file records what was
actually built, the decisions taken, how to run it, and what's left.

---

## 1. What this feature is

A 4th prompt mode `audio+text+emotion` that injects natural-language emotional
descriptions (from **SECap**, a HuBERT + Q-Former + Chinese-LLaMA-7B captioner,
translated zh→en) into the depression-classification prompt, following the
emotion-augmented recipe of *DepressInstruct*. Target prompt format = one short
emotion sentence per audio chunk, interleaved with the audio placeholders.

**Hard constraint:** SECap is heavy + **non-deterministic** (8 sampled generations
per clip), so it must run **offline once** and be frozen in a text cache. Training
and eval never import SECap — they read the cache by `sample_id`. This protects the
repo's eval-determinism rule (val/test metrics must never depend on random
sampling — see the `[[eval-determinism-rule]]` memory and `experiment_handoff.md`).

---

## 2. Decisions taken (all = plan recommendations)

| # | Decision | Choice |
|---|---|---|
| 1 | Emotion granularity | **per-chunk** caption + dynamic re-render under random K-sampling |
| 2 | Translator | **`facebook/nllb-200-distilled-600M`** (opus-mt repetition-collapses on short captions) |
| 3 | Canonical caption from SECap's 5 candidates | **medoid** via `SimiCal` (store all candidates) |
| 4 | Datasets in scope | **DAIC + EDAIC** (EATD/CMDC not wired) |
| 5 | Missing-caption policy | **`neutral_fallback`** (fixed neutral sentence) |
| 6 | System prompt | "You are a clinical expert in depression assessment from speech, transcript, and emotional cues." |

Not wired (deliberately): EATD's 3-response subject bundle. CMDC/EATD (Chinese,
in-domain for SECap) would be good extra validation but are out of current scope.

---

## 3. Files

### Training/eval side (env `qwen_mn5`, reads cache only — NO SECap imports)
- **`src/data/emotion.py`** (new) — frozen-cache loader (`load_emotion_cache`),
  coverage report, missing-caption policy (`neutral_fallback` / `drop_emotion_line`
  / `error`), `use_emotion(config)` resolver, and the two prompt mechanics:
  `single_chunk_emotion_block` and `interleaved_audio_emotion_block`. Neutral
  fallback string = `NEUTRAL_FALLBACK_CAPTION`.
- **`src/data/runtime.py`** (edited) — emotion threaded through:
  - `build_examples`: loads cache once + logs coverage (`report_cache_coverage`).
  - `render_user_prompt_text`: new `{emotion_block}` placeholder.
  - `build_prompt_text`: optional `emotion_captions` → interleaved
    `Audio i:` / `Emotional description i:` block.
  - `_base_example_from_row` (single-chunk): inject caption by `sample_id`.
  - `_build_subject_level_audio_examples` (subject K-chunk): bake deterministic
    interleaved prompt; **also store `chunk_caption_by_path` + re-render metadata**
    (`emotion_user_text`, `emotion_system_prompt`, `emotion_internal_label_text`).
  - `AudioTextDataset.__getitem__` + `_rerender_emotion_prompt`: **the critical
    bit** — under `chunk_sampling="random"` (training only) re-render the prompt so
    each `Emotional description i` tracks the chunk actually sampled. Eval is
    deterministic and keeps the baked prompt (already aligned).

### Offline drivers (env `secap`, import SECap/translators — run ONCE on cluster)
- **`src/emotion/extract_secap.py`** — SECap zh-caption extraction. Resumable
  (skips cached `sample_id`s), shardable (`--shard i/N`), medoid selection,
  `--fast` (single greedy gen) for smoke tests. Writes `emotion_zh` +
  `emotion_zh_candidates`, leaves `emotion_en=null`.
- **`src/emotion/translate.py`** — zh→en pass (NLLB primary), deterministic decode,
  repetition-collapse/empty guard → marks degenerate as `translation_ok=false`.
- **`src/emotion/build_emotion_cache.py`** — `merge` (combine shard files) +
  `validate` (coverage vs manifest). Pure JSONL, no model.
- **`src/emotion/__init__.py`** — warns: training code must not import this package.

### Configs (new presets)
- `configs/daic_audio_text_emotion.yaml` (single-chunk)
- `configs/daic_subject_audio_text_emotion_k4.yaml` (subject K=4, primary target)
- `configs/edaic_audio_text_emotion.yaml`
- `configs/edaic_subject_audio_text_emotion_k4.yaml`

Each adds under `data:`: `use_emotion: true`,
`emotion_cache_path: ${PROJECT_ROOT}/outputs/emotion/<dataset>_secap_en.jsonl`,
`emotion_on_missing: neutral_fallback`. Subject presets inherit the tuned recipe
(DAIC reg1 / EDAIC reg2, frozen encoder + macro-F1 selection + TF eval).

### Tests
- **`scripts/test_emotion_injection.py`** — no-model checks: single-chunk injection,
  deterministic interleave alignment, **random re-render tracks sampled order**,
  drop-policy keeps audio count, missing→fallback. Wired into
  `scripts/sanity_tests_no_model.sh` (the `[emotion]` step).

### SLURM (offline pipeline)
- **`scripts/run_emotion_extract_slurm.sh`** — extract, 1 GPU, array job (1 shard/task).
- **`scripts/run_emotion_translate_slurm.sh`** — translate, 1 GPU.
- **`scripts/run_emotion_build_cache_slurm.sh`** — merge/validate, CPU (`MODE=merge|validate`).
- **`scripts/submit_emotion_pipeline.sh`** — orchestrator, chains
  extract(array) → merge(if SHARDS>1) → translate → validate with `afterok` deps.

All four default the SECap env to
`/gpfs/projects/etur92/ozu647717/venvs/secap_rebuilt/bin/activate` (override
`SECAP_ENV_ACTIVATE`, or `SECAP_CONDA_ENV` to use a conda env). Account `etur92`,
queue `acc_ehpc`, matching `submit_train_and_eval.sh`.

---

## 4. Cache schema (`outputs/emotion/<dataset>_secap_en.jsonl`, one row per chunk)

Key = `sample_id` (portable; `audio_path` differs local vs cluster). Fields:
`dataset, subject_id, sample_id, audio_path, secap_prompt, emotion_zh,
emotion_zh_candidates, emotion_en, translation_ok, secap_model_version,
translation_model, secap_fast`. `emotion_en=null` ⇒ training applies the
missing-caption policy.

---

## 5. How to run

### Step 0 — offline cache (cluster, secap env, once)
```bash
# Full DAIC across 8 shards (extract → merge → translate → validate, chained):
DATASET=daic SHARDS=8  scripts/submit_emotion_pipeline.sh
DATASET=edaic SHARDS=8 scripts/submit_emotion_pipeline.sh

# Smoke test first (single clip, greedy, validate the SECap API contract):
DATASET=daic SHARDS=1 FAST=1 LIMIT=4 scripts/submit_emotion_pipeline.sh
```
Key env overrides: `SECAP_ROOT` (default `/gpfs/projects/etur92/SECap`), `CKPT`,
`TRANSLATE_MODEL`, `MANIFEST`, `OUT_ZH`, `OUT_EN`. Produces
`outputs/emotion/<dataset>_secap_en.jsonl`.

### Step 1/2 — train + eval (qwen_mn5 env, normal flow)
```bash
CONFIG=$PROJECT_ROOT/configs/daic_subject_audio_text_emotion_k4.yaml \
  scripts/submit_train_and_eval.sh
```
`build_examples` auto-loads the cache; train re-renders per sampled chunk; eval
uses the baked deterministic prompt. No other change needed.

### No-model sanity (local, any env with deps)
```bash
python scripts/test_emotion_injection.py        # all checks pass
```

---

## 6. Verified locally

- `scripts/test_emotion_injection.py` — all 5 alignment/injection checks pass.
- End-to-end prompt render through `build_examples` for both DAIC emotion configs
  reproduces the plan's §3.1 (single) and §3.2 (subject interleaved) formats with
  correct `<|AUDIO|>` counts.
- Original plan smoke test (in `secap_implementation.md` §"Smoke-test results"):
  SECap ran on a real English DAIC clip (CPU, local CUDA dead), produced a coherent
  but generic prosodic caption; NLLB translated cleanly, opus-mt collapsed.

---

## 7. What's left (priority order)

1. **Generate the cache** on the cluster (Step 0) for DAIC + EDAIC. `extract_secap.py`
   was corrected against the real `SECap/model2.py` + `scripts/inference.py`:
   `MotionAudio()` (no ckpt arg) → `load_state_dict(torch.load(ckpt))` → `.to(cuda)`;
   `inference([wav])` takes audio only, runs 8 sampled gens, returns 5 candidates +
   prompt (canonical = first candidate; all stored). `--secap-root` is auto-resolved
   to the dir containing `model2.py` (handles nested cluster layout), and the job
   `chdir`s there. Still smoke-test with `FAST=1 LIMIT=4` before the full run.
   NOTE: the first cluster attempt hit `ModuleNotFoundError: model2` because
   `SECAP_ROOT` (`/gpfs/projects/etur92/SECap`) didn't have `model2.py` at top level
   — the auto-locate fix addresses this, but verify the smoke run finds it.
2. **Manual QC** a random sample of `emotion_en` captions (DepressInstruct did this).
   High `translation_ok=false` rate or generic/wrong captions = red flag.
3. **Ablation (the real validation):** `audio+text` vs `audio+text+emotion`, same
   seeds/folds, on DAIC + EDAIC subject presets. Compare headline macro-F1 / strict
   metrics **and the train/val gap** (emotion is audio-derived → could add an
   overfit shortcut; see `[[edaic-overfitting-investigation]]`). Only adopt if it
   helps without inflating the gap.
4. Optional: extend to CMDC/EATD (Chinese, in-domain for SECap) as a sanity ceiling.

---

## 8. Risks / gotchas

- **Cross-lingual shift (highest risk):** SECap is Mandarin-trained; DAIC/EDAIC are
  English. Captions may be generic/biased. Q-Former is meant to keep prosody and
  drop content, so transfer is plausible but **unvalidated** — hence QC + ablation.
- **Leakage:** DAIC chunk count encodes the label (non-dep=10, dep=15). K is held
  fixed; never let any emotion text mention chunk counts. Per-chunk captions are
  count-agnostic by construction.
- **Env separation is hard:** Step 0 = `secap` env (or
  `…/venvs/secap_rebuilt`); Steps 1–2 = `qwen_mn5`. No cross-imports — enforced by
  keeping SECap only under `src/emotion/`.
- **Translator env (from plan smoke test):** `transformers==4.29.0` +
  `huggingface_hub==1.8.0` can raise `use_auth_token` errors; pin compatible
  versions or pre-download translator weights. `sentencepiece` needed for NLLB.
- **SLURM array cap:** orchestrator uses `--array=0-$((SHARDS-1))`. If the cluster
  caps concurrent array tasks, add `%N` throttle (not yet parametrized).

---

## 9. Related memory notes
`[[secap-emotion-mode]]`, `[[eval-determinism-rule]]`,
`[[edaic-overfitting-investigation]]`, `[[subject-audio-kchunk-mode]]`,
`[[audio-encoder-lora-leak-freeze]]`.
