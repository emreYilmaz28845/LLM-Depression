# Turkish Depression Dataset — Training Pipeline Plan

Implementation plan for adding a Turkish depression dataset to the `LLM-Depression`
pipeline. Based on (a) a full read of the existing repo and (b) direct inspection of
the Turkish dataset on this PC.

- **Local dataset path (this PC, for building/inspection):** `/media/emre/Backup/AudioLLM/Datasets/Turkish`
- **Server dataset path (for training runs):** `/gpfs/projects/etur92/ozu647717/AudioLLM/Datasets/Turkish`
- All configs parameterize this as `${TURKISH_DATASET_ROOT:-/gpfs/.../Turkish}` so the same config runs on both.

> **Status: PLAN ONLY.** No code has been modified and nothing has been trained.
> Every dataset number below was measured directly from the local copy.

---

## 0. TL;DR / Final Recommendation

- **Leakage unit = `patient_id`** (120 subjects). Split at subject level only.
- **No audio chunking needed.** Every file is already a ≤20 s, 16 kHz segment (max 20.0 s, 0 files > 30 s). Qwen2-Audio's 30 s window holds each file whole.
- **Split protocol = stratified group 5-fold CV at subject level, repeated over seeds {1337, 7, 2024}** (mirrors the EATD path). N=120 is too small for a fixed train/val/test split to be stable; CV uses every subject as test exactly once. Inner split (20 %) selects checkpoints. No separate locked test by default.
- **Primary model = audio+text Qwen2-Audio, chunk-level** (`sample_mode: response`-style, one example per file, aggregated to subject at eval). This needs **zero changes to `runtime.py`**.
- **Strongest enhancement = subject-level K-chunk audio+text** (`sample_mode: subject_audio`, K=4) — reuses DAIC's proven overfitting-mitigation path, needs a small `runtime.py` change to handle Turkish's per-segment transcripts.
- **Sanity baselines first:** text-only Qwen2-7B and the cheap wav2vec2/`features` classifier (features are already in the CSV).
- **Headline metric = subject-level macro-F1 + depressed-class F1** via mean-likelihood aggregation. Add AUROC (continuous score already available from the likelihood backend).

---

## 1. Repo Investigation (what exists today)

### 1.1 Repo structure

```
LLM-Depression/
├── configs/                 # ~100 YAML presets, one per dataset×modality×regime
├── scripts/                 # slurm launchers, n-fold runners, sanity tests
├── src/
│   ├── data/
│   │   ├── build_manifest.py # CLI entrypoint: dataset → manifest + split metadata
│   │   ├── daic.py           # DAIC-WOZ manifest builder
│   │   ├── edaic.py          # E-DAIC manifest builder
│   │   ├── cmdc.py           # CMDC manifest builder
│   │   ├── eatd.py           # EATD-Corpus manifest builder
│   │   ├── split_utils.py    # fold/inner-split logic (stratified group folds, CV)
│   │   ├── validation.py     # label/audio/transcript asserts, partition-uniqueness
│   │   ├── runtime.py        # example building, prompt construction, Dataset, audio load
│   │   └── emotion.py         # optional SECap/Qwen2-Audio emotion-caption injection
│   ├── model/
│   │   ├── runtime.py         # model/processor load, LoRA, eval/train mode switches
│   │   ├── qwen2audio_lora.py # Qwen2-Audio + LoRA + DepAdapter
│   │   ├── text_lora.py       # text-only Qwen2 + LoRA
│   │   ├── collator.py        # Qwen2AudioSFTCollator (mel + tokenize + label mask)
│   │   └── lora_common.py
│   ├── train.py               # training entrypoint (accelerate/torchrun)
│   ├── evaluate.py            # standalone eval entrypoint (likelihood/generation/TF)
│   ├── aggregate.py           # subject/segment-level aggregation + metrics assembly
│   ├── metrics.py             # binary + multiclass-diagnostic metrics (NO AUROC)
│   ├── hpo.py                 # Optuna HPO
│   └── utils.py               # config resolution, label vocab, modality, seeding
├── SIGNAL_FLOW.md             # end-to-end stage-by-stage description (accurate)
└── README.md                  # commands per dataset/modality
```

### 1.2 Responsibility map (exact locations)

| Concern | Where | Notes |
|---|---|---|
| **Manifest CLI** | [build_manifest.py](src/data/build_manifest.py) `build_for_config()` (L65), dispatch L76–85 | dataset name → `build_*_manifest()`; saves jsonl/csv + split metadata json |
| **Dataset preprocessing / join** | `src/data/{daic,edaic,cmdc,eatd}.py` | join labels + transcripts + audio paths into manifest rows |
| **Transcript loading** | [daic.py `_load_whisper_transcripts`](src/data/daic.py#L95) | reads `whisper_transcripts.jsonl` `{audio_path, transcript, language}`, keys by filename → sample_id regex |
| **Label creation** | per dataset; e.g. [eatd.py L39](src/data/eatd.py#L39) `int(index_sds >= 53.0)` | threshold → binary; `label_text_from_int` ([utils.py L385](src/utils.py#L385)) |
| **Split (fixed)** | [train.py `_resolve_fixed_outer_partitions`](src/train.py#L164) reads `*_subject_partitions.json` | DAIC/EDAIC official train/val/test |
| **Split (CV folds)** | [split_utils.py `assign_stratified_group_folds`](src/data/split_utils.py#L310), `build_partition_scoped_stratified_folds` (L284) | subject-level stratified group K-fold |
| **Inner val split** | [split_utils.py `deterministic_inner_split`](src/data/split_utils.py#L337) | carves val from outer-train, class-balanced, deterministic |
| **Split-integrity asserts** | [validation.py `assert_subject_partition_uniqueness`](src/data/validation.py#L126); [split_utils.py `validate_non_overlapping_folds`](src/data/split_utils.py#L159) | fail loudly on subject overlap |
| **Audio chunking** | **None in repo.** Chunks arrive pre-cut. `load_audio_array` ([runtime.py L636](src/data/runtime.py#L636)) only trims to `max_seconds` + resamples to 16 kHz | see SIGNAL_FLOW L42–44 |
| **K-chunk subject sampling** | [runtime.py `_build_subject_level_audio_examples`](src/data/runtime.py#L348) + `AudioTextDataset._resolve_audio_plan` (L780) | deterministic eval view, random per-epoch train view |
| **Prompt construction** | [runtime.py `render_user_prompt_text`](src/data/runtime.py#L109), `build_prompt_text` (L144); templates in config `prompt.*` | English prompt, original-language transcript |
| **Collation** | [collator.py `Qwen2AudioSFTCollator`](src/model/collator.py) | mel (128×3000, 30 s window) + tokenize + label mask `-100` on prompt |
| **Training entrypoint** | [train.py `main()`](src/train.py#L615) | `python src/train.py --config ... --fold k --run_name ...` |
| **Eval entrypoint** | [evaluate.py](src/evaluate.py); likelihood scoring `score_candidate_label` (L208), `_predict_sample_likelihood` (L258) | emits `dep_score`/`non_score` per sample |
| **Subject aggregation** | [aggregate.py `aggregate_likelihood_predictions`](src/aggregate.py#L152) (mean logit), `_aggregate_majority_vote_predictions` (L214) (vote) | |
| **Metrics** | [metrics.py `classification_metrics`](src/metrics.py#L26) | acc, pos-F1, macro-F1, weighted-F1, precision/recall, confusion. **No AUROC.** |

### 1.3 How existing datasets are prepared & trained

**DAIC-WOZ** ([daic.py](src/data/daic.py)) — auto-detects layout. Preprocessed layout reads
`train/dev/test_preprocessing_summary.csv` (columns `Participant_ID`, `PHQ8_Binary`,
`segment_files`, `full_transcript`) + `*_audio_segments/{subject}/{subject}_segment_N.wav`.
One manifest row per pre-cut chunk; **every chunk of a subject repeats the subject's full
transcript**. Produces fixed `subject_partitions.json` (official train/val/test) **and**
5-fold stratified CV folds over the train+val dev pool. Subject id = numeric filename prefix.

**E-DAIC** ([edaic.py](src/data/edaic.py)) — same preprocessed shape (`participant_id`,
`is_depressed`, `num_segments`, `segment_files`, `full_transcript`); strict per-row
validation of segment counts and filename↔subject match. Fixed partitions + 5-fold CV. 275
subjects, 3080 segments (asserted in the no-model sanity test).

**EATD** ([eatd.py](src/data/eatd.py)) — pooled corpus, **no official test split**. Per subject:
`negative/neutral/positive.{wav,txt}` + `label.txt`/`new_label.txt` (SDS). Label =
`int(index_sds >= 53.0)`. Builds **3-fold stratified group CV** directly via
`assign_stratified_group_folds(...)` (no fixed partitions). 162 subjects (30 dep / 132 non-dep,
hard-asserted). This is the closest analog to "a new dataset with no official split" — **but
Turkish is structurally closer to DAIC** (many segments/subject) than to EATD (3 fixed responses).

**CMDC** ([cmdc.py](src/data/cmdc.py)) — official folds parsed from an `.xlsx` workbook;
subject ids `MDD##`/`HC##`.

**Training/eval flow (all datasets):** `build_manifest.py` writes
`outputs/manifests/{ds}_manifest.jsonl` + `outputs/splits/{ds}_*` once (cached;
`train.py` auto-builds if missing via `_load_metadata_or_build`, [train.py L140](src/train.py#L140)).
`train.py` resolves subject splits → builds examples → SFT with LoRA → selects best
checkpoint on inner-val metric. `evaluate.py` scores held-out subjects (likelihood is the
headline). `aggregate.py` rolls chunk predictions up to subject level.

---

## 2. Turkish Dataset Inspection (measured)

Root contents:

```
all-files/                                       1186 .wav
metadata_turkish_t25_binary_merged.csv           1051 rows  ← merged (depression + comorbid)
metadata_w2v2_scores_mfcc_with_3_feat_depression_t25_binary.csv   654 rows  (depression cohort)
metadata_w2v2_scores_mfcc_with_3_feat_comorbid_t25_binary.csv     397 rows  (comorbid cohort)
metadata_w2v2_scores_mfcc_with_3_feat_{depression,comorbid}.csv   (pre-binary variants)
thershold.txt                                    threshold rule
whisper_transcripts.jsonl                        1186 lines  (one per audio file)
```

**`thershold.txt`:**
```
Use threshold 25 for Turkish binary classification.
Rule:
- depressed: depresyon_skoru >= 25
- non_depressed: depresyon_skoru < 25
```

**CSV schema** (all metadata CSVs share it):
`file_name, label, depresyon_skoru, anksiyete_skoru, temp, features, patient_id,
w2v2_predicted_score, age, medeni_hal, egitim, meslek, label_t25, target_t25`

- `file_name` — e.g. `Datasets/Turkish/all-files/ak2-1-1-depr.wav` (relative; **use basename**).
- `patient_id` — **the subject id** (e.g. `ak2`); equals the filename prefix in 100 % of rows.
- `depresyon_skoru` — depression score (range **3–49**, mean 21.3, median 21).
- `target_t25` ∈ {`depressed`,`non_depressed`}, `label_t25` ∈ {0,1} — **the binary depression label** (`depresyon_skoru >= 25`); 413 positive files exactly match `skoru>=25`. **Use this, not `label`.**
- `label` ∈ {0,1} = **cohort/comorbidity flag**, NOT depression (0 = depression-only 654 files, 1 = comorbid 397 files). Do **not** use as the target.
- `anksiyete_skoru` (anxiety), `age`, `medeni_hal` (marital), `egitim` (education), `meslek` (occupation) — demographics.
- `features` — inline 608-dim w2v2/MFCC vector; `w2v2_predicted_score` — a regression model's predicted score. (Enable the cheap classical baseline; not needed for the LLM path.)

**Transcripts (`whisper_transcripts.jsonl`):** `{"audio_path": "../../.../all-files/<bn>.wav",
"transcript": "...", "language": "tr"}`. **One transcript per file (per-segment)** — each file
has its own distinct transcript (NOT a repeated full-interview transcript like DAIC). Join by
**basename**.

**Subject / file structure (measured on the merged CSV):**

| Quantity | Value |
|---|---|
| Unique subjects (`patient_id`) | **120** |
| Labeled files | **1051** (413 depressed / 638 non-depressed at file level) |
| Subject-level balance | **46 depressed / 74 non-depressed** (38.3 % positive) |
| Files per subject | min 5, **median 9**, mean 8.76, max 19 |
| Sessions per subject | always 1 (`-1-` field) |
| Label consistency within subject | **100 %** (0 subjects with mixed labels) |
| Audio total (labeled) | ≈ 5.54 h |
| Filename pattern | `{patient_id}-1-{segment}-{cohort}.wav`, cohort ∈ {`depr`,`depr+ank`,`ank+depr`,`dep+ank`} |

**Durations (1051 labeled files):** min 0.31 s, p25 = median = p75 = p95 = **20.0 s**, max **20.0 s**,
**0 files > 30 s**. Sample rate **16 kHz** (already the pipeline target). → Files are pre-segmented to a 20 s cap.

**Granularity:** transcripts and labels map to **segments (individual files)**, and there are
**multiple segments per subject**. This is the DAIC/EDAIC shape (multi-segment, subject-level
label), so the existing chunk-level and subject-K-chunk machinery applies almost directly.

### Inconsistencies / things to handle

1. **135 unlabeled audio files** (1186 on disk + 1186 transcripts, but only 1051 in the merged CSV). They have audio + transcripts but no label row → **drop** (record in an audit file).
2. **1 labeled transcript mis-tagged language** (`cy2-1-9-ank+depr.wav` → `pt`); the other non-`tr` tags (`en`, `''`) are among the unlabeled extras. The transcript text is still a Whisper transcription of Turkish audio; keep it, just flag.
3. **Three encodings of "comorbid"** in filenames (`depr+ank`, `ank+depr`, `dep+ank`). The `label`/cohort column is authoritative; don't parse the tag for the target.
4. **`features`/`temp` CSV columns are large** (608-dim inline vector). The manifest builder must read with `csv.field_size_limit` raised and must **not** copy `features` into the manifest.

---

## 3. Leakage Analysis

**Correct leakage-safe unit: `patient_id`.** Each subject contributes 5–19 segments that share
one label; segments of the same subject are highly correlated (same speaker, same recording
session). Splitting at the file/chunk level would put a subject's segments in both train and
test → severe optimistic bias.

**Is current repo splitting chunk- or subject-level?** **Subject-level**, everywhere that matters:
`assign_stratified_group_folds` ([split_utils.py L310](src/data/split_utils.py#L310)) and the
fixed-partition path both key on `subject_id`; `filter_rows_by_subjects`
([runtime.py L55](src/data/runtime.py#L55)) restricts a partition's manifest rows by subject id;
aggregation groups chunk predictions by `subject_id`. Chunk-level appears only *inside* a
partition (multiple examples per subject) and in optional `aggregation_level: segment`
diagnostics — never across the split boundary.

**Where Turkish leakage could still creep in:**
1. **Wrong subject id derivation.** Must use the CSV `patient_id` (authoritative). A naive
   "prefix before first `-`" works today (0 mismatches) but is fragile to ids containing `-`.
2. **Chunk-level example expansion** (primary approach) creates many examples per subject — safe
   **only because** folds are subject-keyed. Any future code that splits the *manifest rows*
   directly (instead of subject ids) would leak.
3. **Subject-count → label correlation:** checked and **absent** (depressed mean 8.98 files vs
   non-depressed 8.62) — so Turkish does **not** have DAIC's "chunk-count encodes label" problem.
   (Fixed-K subject_audio still recommended for the multimodal path for a bounded audio budget.)
4. **The 135 unlabeled files** must never enter any split.

**Validation checks that fail loudly (proposed):** a `verify_turkish_split_integrity()` (new,
mirrors the no-model sanity proof in [scripts/sanity_tests_no_model.sh L54](scripts/sanity_tests_no_model.sh#L54)):

```python
def verify_turkish_split_integrity(manifest_rows, folds, inner_val_ratio, seed):
    subj_of = {r["sample_id"]: r["subject_id"] for r in manifest_rows}
    all_subjects = set(subj_of.values())

    # (a) every subject is held out in exactly one fold; folds partition all subjects
    holdout_seen = Counter()
    for k, payload in folds.items():
        tr = set(payload["outer_train_subject_ids"]); ho = set(payload["final_eval_subject_ids"])
        assert tr.isdisjoint(ho), f"fold {k}: train∩holdout leak {sorted(tr & ho)[:5]}"
        assert tr | ho == all_subjects, f"fold {k} does not partition all subjects"
        for s in ho: holdout_seen[s] += 1
    assert all(c == 1 for c in holdout_seen.values()), "subject held out in !=1 fold"
    assert set(holdout_seen) == all_subjects, "fold coverage != all subjects"

    # (b) inner split (per fold) keeps train_inner ∩ val_inner empty at subject level
    labels = {r["subject_id"]: int(r["label"]) for r in manifest_rows}
    for k, payload in folds.items():
        inner = deterministic_inner_split(labels, payload["outer_train_subject_ids"],
                                          seed=seed + int(k), val_ratio=inner_val_ratio)
        ti, vi = set(inner["train_inner_subject_ids"]), set(inner["val_inner_subject_ids"])
        assert ti.isdisjoint(vi) and vi.isdisjoint(set(payload["final_eval_subject_ids"]))

    # (c) no audio file maps to two subjects
    by_file = defaultdict(set)
    for r in manifest_rows: by_file[Path(r["audio_path"]).name].add(r["subject_id"])
    assert all(len(v) == 1 for v in by_file.values()), "a file maps to >1 subject"
```

Run it inside `build_turkish_manifest` (raise on failure) and again as a standalone gate before
any training. Existing `assert_subject_partition_uniqueness` and `validate_non_overlapping_folds`
are reused for free.

**Do not** balance val/test by oversampling (handoff rule + [memory: eval-determinism-rule]).
Class imbalance is handled **only inside training folds** (see §7).

---

## 4. Chunking Decision

**Decision: no chunking.** Inputs are already ≤20 s, 16 kHz, pre-segmented; Qwen2-Audio's
feature extractor pads/trims each clip to a single 30 s / 3000-mel / 750-token window
([SIGNAL_FLOW L206–215](SIGNAL_FLOW.md)), so a 20 s file fits whole with zero truncation.

| Option | Verdict for Turkish |
|---|---|
| **A. No chunking / full file** | ✅ Files are already ≤20 s. Each file = one model-ready chunk. |
| B. Fixed 30 s chunks (DepressInstruct) | ❌ Pointless — nothing exceeds 30 s; would just pad. |
| C. Short 7–10 s chunks | ❌ Would *re-segment* already-short clips, multiply noisy labels, add an offline step with no benefit. |
| **D. Subject-level K-chunk sampling** | ✅ Best for the multimodal model — bounds audio/subject (median 9 × 20 s ≈ 180 s otherwise), reuses DAIC's `subject_audio` path, mitigates speaker memorization. |

- **Primary: D for audio+text** — `sample_mode: subject_audio`, **K = 4**, `max_audio_seconds_per_chunk: 20`. Train samples 4 of the subject's files per epoch (fresh view); eval uses a deterministic evenly-spaced 4-file view. (Needs the small `runtime.py` change in §8 to handle per-segment transcripts.)
- **Fallback / first integration: A at chunk level** — one example per file, no `runtime.py` change, aggregate to subject at eval. Also the right choice for the **audio-only** baseline (more, shorter independent acoustic samples) and the classical baseline.

No offline chunker script is required. (If a future raw/un-segmented Turkish drop appears, add `scripts/chunk_turkish_audio.py` producing 20–30 s windows + per-window transcripts; not needed now.)

---

## 5. Split Strategy

N = 120 subjects (46 dep / 74 non-dep). Recommendation by size:

| Option | Decision |
|---|---|
| A. Fixed train/val/test | ❌ as the main protocol — at N=120 a single 15 % test (≈18 subj, ≈7 positive) is too noisy; estimates swing with the draw. |
| **B. Stratified group K-fold CV** | ✅ **Primary.** Every subject is test exactly once; stratify folds by subject label; group by `patient_id`. |
| C. Nested CV | ➖ Overkill here; the existing **inner deterministic split** (val carved from each fold's train) already gives leakage-safe checkpoint selection without a full nested loop. |

**Chosen protocol (mirrors EATD, [eatd.py L117](src/data/eatd.py#L117)):**
- **5-fold** stratified group CV (`assign_stratified_group_folds(subject_labels, n_splits=5, seed)`).
  Each holdout ≈ 24 subjects (≈ 9 positive) — enough signal; 3-fold is the fallback if positives
  per fold feel too thin.
- **Inner val = 20 %** of each fold's training subjects (`deterministic_inner_split`,
  `inner_val_ratio: 0.2`) for checkpoint selection / early stopping.
- **Repeat over seeds {1337, 7, 2024}** (seeds already used across configs); report **mean ± std**
  across folds × seeds.
- **No separate locked test by default** (the fold holdout *is* the test). If a locked test is
  wanted later, carve a stratified ~20 % test once and run 5-fold CV on the remaining 96 — but at
  N=120 this wastes scarce positives; prefer pure CV.
- **Class balance handled only inside training folds** (§7). Folds, inner val, and holdouts keep
  the natural class ratio.

Fold layout (k=0 example): train 96 subj → inner-train ≈ 77 / inner-val ≈ 19; holdout ≈ 24.

---

## 6. Manifest Design

One JSONL row per **labeled file** (chunk), schema-compatible with existing datasets
(superset of DAIC/EATD rows so `build_examples`, `validation.py`, `aggregate.py` work unchanged):

```jsonc
{
  "dataset": "turkish",
  "subject_id": "ak2",                       // = patient_id (leakage unit)
  "sample_id": "ak2-1-1-depr",               // basename without .wav (unique; 1051 of them)
  "audio_path": "<root>/all-files/ak2-1-1-depr.wav",
  "audio_paths": ["<root>/all-files/ak2-1-1-depr.wav"],
  "transcript": "...",                        // this file's own Whisper transcript
  "transcript_path": "<root>/whisper_transcripts.jsonl",
  "label": 0,                                 // int: 1 if depresyon_skoru>=threshold else 0
  "label_text": "Non-depressed",             // label_text_from_int(label)
  "score": 18.0,                              // depresyon_skoru
  "split_original": "all",                    // no official split; folds live in folds.json
  "fold": "",
  "chunk_id": "1",                            // segment index from filename
  "question_id": "",
  "start_time": "", "end_time": "",
  "language": "tr",
  "gender": null,
  "modality_mode": "single_audio_single_text",
  // Turkish-specific extras (ignored by core code, useful for analysis/stratification):
  "comorbid": 0,                              // from CSV `label` (0=depression,1=comorbid)
  "anxiety_score": 15.0,
  "threshold": 25,
  "w2v2_predicted_score": 21.52
}
```

- **Label derivation:** `label = int(float(depresyon_skoru) >= config.threshold)` with
  `threshold` default 25, **and assert it equals `label_t25`/`target_t25`** (guardrail; raise on
  mismatch). Making threshold a config knob enables sensitivity analysis without re-exporting data.
- **Generate once and cache.** `build_manifest.py` already writes the manifest + split metadata
  and `train.py` auto-rebuilds only when stale ([train.py L140](src/train.py#L140)). Same model as
  every other dataset.
- **Output paths** (unchanged convention):
  - `outputs/manifests/turkish_manifest.jsonl` (+ `.csv`)
  - `outputs/splits/turkish_folds.json`, `turkish_fold_report.json`, `turkish_subjects.json`,
    `turkish_join_audit.csv`, `turkish_extra_file_audit.json`, `turkish_manifest_metadata.json`
- **`prompt_mode`/modality is NOT a manifest field** in this repo — it comes from the config
  (`data.use_audio/use_text`, `data.sample_mode`). One manifest serves all modalities (audio+text,
  audio-only, text-only), exactly like DAIC.

---

## 7. Training Approaches (compare & recommend)

Common: LoRA (r=16, α=32, dropout 0.05) on Qwen2 decoder proj layers; bf16; likelihood headline;
subject-level aggregation; **class imbalance handled train-only** via weighted loss or a
`WeightedRandomSampler` on the train `DataLoader` (the val/test sets stay natural — never oversampled).

| # | Approach | Difficulty | Compute | Risk | Metric | Stage |
|---|---|---|---|---|---|---|
| **A** | **Text-only Qwen2-7B-Instruct** (Whisper transcript only) | Low (config only) | Low (no audio) | Captures only ASR/lexical signal; Whisper errors propagate | subj macro-F1 / AUROC | **First** — strongest, cheapest sanity baseline |
| **B** | **Audio-only Qwen2-Audio** (raw audio) | Low (config only) | Med | Noisier; chunk labels are subject labels | subj macro-F1 | First/second — isolates acoustic cues |
| **C** | **Audio+text Qwen2-Audio, chunk-level** | Low (config; **no `runtime.py` change**) | Med–High | Mild overfitting to speaker; per-file transcript is short | subj macro-F1 / AUROC | **First multimodal / primary baseline** |
| **C′** | **Audio+text Qwen2-Audio, subject K-chunk** (`subject_audio`, K=4) | Med (small `runtime.py` change, §8) | High | Best overfitting control; needs transcript-concat handling | subj macro-F1 / AUROC | **Primary headline once C works** |
| **D** | Audio+text+**emotion** captions | Med–High (needs offline emotion cache for `tr`) | High | Emotion extractor may not transfer to Turkish | subj macro-F1 | **Later** — only after C/C′ beat baselines [memory: secap-emotion-mode] |
| **E** | **wav2vec2 / `features` classifier** (logreg/MLP on the 608-dim CSV `features` or `w2v2_predicted_score`) | Low (standalone, outside LLM repo) | Very low | Not the LLM pipeline; ceiling/sanity only | subj macro-F1 / AUROC | First — cheap CV sanity + comparison point |

**Recommended ordering:** E + A (cheap sanity) → B and C (multimodal baselines) → **C′ headline** →
D only if C′ clears the baselines. Implement C first because it needs no `runtime.py` change.

---

## 8. Recommended Pipeline (concrete, step-by-step)

### Files to MODIFY

1. **`src/data/build_manifest.py`** — register Turkish in the dispatch
   ([L13–16 imports, L76–85 dispatch](src/data/build_manifest.py#L76)):
   ```python
   from src.data.turkish import build_turkish_manifest
   ...
   elif dataset_name == "turkish":
       result = build_turkish_manifest(config, quarantine)
   ```
   (`assert_transcripts`/`assert_audio_exists`/`print_class_counts` already run afterward; the
   save logic already handles `folds`, `subject_rows`, `extra_file_audit`, `join_audit_rows`.)

2. **`src/data/runtime.py`** — *only needed for the subject-level modes (C′ / text-only subject doc).*
   Turkish has **per-segment transcripts**, so the existing subject-level builders, which assert
   "exactly one transcript per subject" ([L264–268 text](src/data/runtime.py#L264),
   [L407–412 subject_audio](src/data/runtime.py#L407)), would raise. Two minimal options:
   - **Add `"turkish"` to the two gates** ([L516 text_only](src/data/runtime.py#L516),
     [L525 subject_audio](src/data/runtime.py#L525)) **and** relax the transcript handling to
     **concatenate** a subject's segment transcripts (newest behavior gated by e.g.
     `data.multi_transcript: concat`, default strict so DAIC/EDAIC are untouched):
     ```python
     transcript_values = [str(r["transcript"]).strip() for r in rows]
     if config["data"].get("multi_transcript") == "concat":
         transcript = "\n".join(t for t in transcript_values if t)
     elif len(set(transcript_values)) != 1:
         raise ValueError(...)   # existing strict path
     ```
   - **Or** skip this entirely for the first milestone and run **chunk-level** (approach C) which
     needs **no** `runtime.py` change (generic per-row path, [L536–550](src/data/runtime.py#L536)).

3. **`src/metrics.py` + `src/aggregate.py` + `src/evaluate.py`** *(optional, recommended)* — add AUROC:
   ```python
   # metrics.py — rank-based AUROC, no sklearn dependency
   def binary_auroc(y_true, scores):
       pos = [s for y, s in zip(y_true, scores) if y == 1]
       neg = [s for y, s in zip(y_true, scores) if y == 0]
       if not pos or not neg: return 0.0
       order = sorted(range(len(scores)), key=lambda i: scores[i])
       ranks = [0.0]*len(scores); i = 0
       while i < len(scores):                  # average ties
           j = i
           while j+1 < len(scores) and scores[order[j+1]] == scores[order[i]]: j += 1
           r = (i + j) / 2.0 + 1.0
           for k in range(i, j+1): ranks[order[k]] = r
           i = j + 1
       rank_pos = sum(ranks[i] for i, y in enumerate(y_true) if y == 1)
       return (rank_pos - len(pos)*(len(pos)+1)/2.0) / (len(pos)*len(neg))
   ```
   In `aggregate_likelihood_predictions` ([aggregate.py L152](src/aggregate.py#L152)) the per-subject
   continuous score is `mean_dep - mean_non` (already computed) → call `binary_auroc(y_true,
   [r["dep_score"]-r["non_score"] for r in subject_rows])` and add `"auroc"` to the metrics dict.

4. **`README.md`** — add a "Turkish Training / Eval" section.

5. **`configs/quarantines.yaml`** *(optional)* — add a `turkish:` entry if specific files must be dropped beyond the automatic unlabeled-file audit.

### Files to CREATE

| New file | Purpose |
|---|---|
| `src/data/turkish.py` | `build_turkish_manifest(config, quarantine)` — join CSV+transcripts+audio, derive labels, build 5-fold stratified group folds, run `verify_turkish_split_integrity`, return manifest/subjects/folds/audits. |
| `scripts/inspect_turkish.py` | Reproducible dataset report (counts, durations, join coverage, anomalies) — the §2 numbers. |
| `configs/turkish_text_only.yaml` | Approach A (Qwen2-7B-Instruct). |
| `configs/turkish_audio_only.yaml` | Approach B. |
| `configs/turkish_audio_text.yaml` | Approach C (chunk-level) — **primary first milestone**. |
| `configs/turkish_subject_audio_text.yaml` | Approach C′ (`sample_mode: subject_audio`, K=4) — headline. |
| `scripts/run_turkish_5fold.sh` | Loop folds 0–4 + `summarize_runs.py` (mirrors `run_eatd_3fold.sh`). |
| `baselines/turkish_features_clf.py` *(optional)* | Approach E — logreg/MLP on CSV `features`, subject-grouped 5-fold. |

### `build_turkish_manifest` pseudocode

```python
def build_turkish_manifest(config, quarantine):
    root = Path(config["dataset_root"])
    csv_path = root / config.get("metadata_csv", "metadata_turkish_t25_binary_merged.csv")
    threshold = float(config.get("threshold", 25))
    transcripts = load_whisper_transcripts(root / "whisper_transcripts.jsonl")  # key: basename

    csv.field_size_limit(10**9)                      # `features` column is large
    manifest_rows, subject_labels, join_audit, extra = [], {}, [], []
    for r in read_csv(csv_path):
        bn = Path(r["file_name"]).name
        wav = root / "all-files" / bn
        score = float(r["depresyon_skoru"])
        label = int(score >= threshold)
        assert label == int(r["label_t25"]), f"label mismatch {bn}"      # guardrail
        tr = transcripts.get(bn)
        found = wav.exists() and tr and tr["transcript"].strip()
        join_audit.append({"sample_id": bn[:-4], "subject_id": r["patient_id"],
                           "audio_found": wav.exists(), "transcript_found": bool(tr), "label": label})
        if not found:
            if is_quarantined_missing(quarantine, "turkish", bn[:-4]): continue
            raise FileNotFoundError(bn)
        if tr["language"] != "tr": extra.append({"file": bn, "language": tr["language"]})
        subject_labels.setdefault(r["patient_id"], label)
        assert subject_labels[r["patient_id"]] == label, f"mixed label for {r['patient_id']}"
        manifest_rows.append({ ... schema in §6 ... })

    # unlabeled files on disk → audit only (never enter splits)
    extra += [{"file": p.name, "reason": "no_label_row"}
              for p in (root/"all-files").glob("*.wav") if p.name not in csv_basenames]

    folds = assign_stratified_group_folds(subject_labels, n_splits=int(config["split"]["outer_folds"]),
                                          seed=int(config["split"]["seed"]))
    verify_turkish_split_integrity(manifest_rows, folds,
                                   float(config["split"]["inner_val_ratio"]), int(config["split"]["seed"]))
    return {"manifest_rows": manifest_rows,
            "subject_rows": [{"subject_id": s, "label": l, "label_text": label_text_from_int(l)}
                             for s, l in sorted(subject_labels.items())],
            "folds": folds,
            "fold_report": subject_fold_report(folds, subject_labels),
            "join_audit_rows": join_audit,
            "extra_file_audit": extra}
```

### Prompt templates (English prompt, Turkish transcript — repo "Core Rules")

Reuse DAIC's templates verbatim (`render_user_prompt_text` fills the blocks):
```yaml
prompt:
  system: You are a psychologist analyzing speech and transcript information for depression screening.
  user_template: |-
    {audio_context_block}
    {transcript_block}Based on the {decision_basis}, determine whether the subject is {label_descriptor}.
    {label_instruction}
  subject_user_template: |-           # only for subject_audio config
    {audio_context_block}
    {transcript_block}Based on the {decision_basis}, determine whether the subject is {label_descriptor}.
    {label_instruction}
  prompt_language: english
labels:
  label_vocab_version: legacy_english_labels   # internal+external labels = Depressed / Non-depressed
```
(Text-only uses the "analyzing transcript information" system prompt, as in `daic_text_only.yaml`.)

### Example config — `configs/turkish_audio_text.yaml` (primary, chunk-level)

```yaml
dataset: turkish
seed: 1337
dataset_root: ${TURKISH_DATASET_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/Datasets/Turkish}
metadata_csv: metadata_turkish_t25_binary_merged.csv
threshold: 25
quarantine_path: ${PROJECT_ROOT}/configs/quarantines.yaml
model_name_or_path: /gpfs/projects/etur92/ozu647717/models/Qwen2-Audio-7B-Instruct
output_dirs:
  manifest_dir: ${PROJECT_ROOT}/outputs/manifests
  split_dir: ${PROJECT_ROOT}/outputs/splits
  run_root: ${PROJECT_ROOT}/output_model/audio_text/turkish
prompt: { system: "You are a psychologist analyzing speech and transcript information for depression screening.", user_template: "...", prompt_language: english }
labels: { label_vocab_version: legacy_english_labels }
data:
  use_audio: true
  use_text: true
  sample_mode: response          # chunk-level: one example per file
  max_audio_seconds_per_chunk: 20.0
  transcript_max_chars: 2000
  allow_empty_transcript: false
split:
  mode: cv                        # CV folds (no official partitions)
  outer_folds: 5
  inner_val_ratio: 0.2
  seed: 1337
lora: { rank: 16, alpha: 32, dropout: 0.05, bias: none, target_modules: [q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj] }
training:
  num_train_epochs: 8
  learning_rate: 2.0e-4
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  bf16: true
  gradient_checkpointing: true
  selection_metric: inner_val_macro_f1
  early_stopping: { enabled: true, metric: inner_val_macro_f1, mode: max, patience: 3 }
evaluation: { sample_prediction_mode: likelihood, aggregation_level: subject }
```
The `subject_audio` config (C′) is identical except `sample_mode: subject_audio`,
`chunks_per_subject: 4`, `multi_transcript: concat`, and adds `subject_user_template`.

---

## 9. Metrics & Evaluation

- **Primary unit = subject** (`evaluation.aggregation_level: subject`). Aggregate chunk predictions
  with the **likelihood backend** = **mean log-prob per subject** (`aggregate_likelihood_predictions`,
  [aggregate.py L152](src/aggregate.py#L152)): `pred = 1 if mean(dep_score) > mean(non_score)`.
  (Majority vote is available via the generation / teacher-forced backends; mean-likelihood is the
  headline because it is deterministic.)
- **Report (subject level):**
  - Accuracy
  - **Macro-F1** (primary; robust to the 38 %/62 % imbalance)
  - **Depressed-class F1** (`positive_f1`)
  - Precision / recall (positive class) + macro precision/recall
  - Confusion matrix `[[TN, FP], [FN, TP]]`
  - **AUROC** — add per §8 using per-subject `mean_dep − mean_non` as the score.
- **Chunk/segment metrics = diagnostic only** (`--set evaluation.aggregation_level=segment`),
  never the headline (chunk "labels" are inherited subject labels → not independent).
- **Cross-fold reporting:** mean ± std of subject-level macro-F1 / dep-F1 / AUROC across 5 folds ×
  3 seeds (`summarize_runs.py` aggregates run outputs).
- **Optional stratified analysis** (data already supports it): report metrics split by cohort
  (`comorbid` flag) and by `depresyon_skoru` band, to see whether comorbid subjects drive errors.

---

## 10. Deliverables — Commands

**Build manifest + folds (run on whichever machine has the data):**
```bash
# local build/inspection
export TURKISH_DATASET_ROOT=/media/emre/Backup/AudioLLM/Datasets/Turkish
python scripts/inspect_turkish.py --root "$TURKISH_DATASET_ROOT"          # report §2 numbers
python src/data/build_manifest.py --config configs/turkish_audio_text.yaml
# → outputs/manifests/turkish_manifest.jsonl + outputs/splits/turkish_*.json|csv
```
(On the server, set `TURKISH_DATASET_ROOT=/gpfs/.../Turkish`; same command. `train.py` will also
auto-build if the manifest is missing.)

**Split-integrity gate (fails loudly on leakage):**
```bash
python -c "from src.data.turkish import verify_turkish_split_integrity, build_turkish_manifest; \
from src.utils import load_yaml_with_overrides as L; c=L('configs/turkish_audio_text.yaml',[]); \
r=build_turkish_manifest(c,{}); print('folds:',{k:len(v['final_eval_subject_ids']) for k,v in r['folds'].items()})"
```

**Smoke test on ~6 subjects (1 fold, 1 epoch) — before any full run:**
```bash
python src/train.py --config configs/turkish_audio_text.yaml --fold 0 --run_name turkish_smoke \
  --set training.num_train_epochs=1 \
  --set split.smoke_subject_limit=6           # (add a tiny subject-cap honored in build_examples/inspect)
# or, model-free first: ./scripts/sanity_tests_no_model.sh  (after adding turkish asserts)
```

**Full training (5-fold, primary chunk-level audio+text):**
```bash
RUN_NAME=turkish_audio_text ./scripts/run_turkish_5fold.sh      # loops folds 0..4 + summarize
# single fold, multi-GPU:
torchrun --nproc_per_node=4 src/train.py --config configs/turkish_audio_text.yaml \
  --fold 0 --run_name turkish_audio_text
```

**Headline subject-K-chunk run (after the small runtime.py change):**
```bash
RUN_NAME=turkish_subj_k4 NPROC_PER_NODE=4 \
  ./scripts/run_turkish_5fold.sh   # using configs/turkish_subject_audio_text.yaml
```

**Baselines (cheap, run first):**
```bash
# text-only LLM
export TEXT_MODEL_PATH=/media/emre/Backup/AudioLLM/models/Qwen2-7B-Instruct   # local
torchrun --nproc_per_node=4 src/train.py --config configs/turkish_text_only.yaml --fold 0 --run_name turkish_text_only
# classical features baseline (subject-grouped 5-fold, no GPU)
python baselines/turkish_features_clf.py --root "$TURKISH_DATASET_ROOT" --folds 5
```

**Standalone evaluation of a checkpoint:**
```bash
python src/evaluate.py --config configs/turkish_audio_text.yaml --fold 0 \
  --checkpoint_dir output_model/audio_text/turkish/turkish_audio_text/fold_0/best_model
```

**Seed sweep:** rerun each with `--set seed=7` / `--set seed=2024 --set split.seed=2024` and
`--run_name ..._s7` etc., then `summarize_runs.py` over the run roots.

---

## Risks / Unknowns / Questions for confirmation

1. **Which cohort CSV is the training set?** Default = **merged** (`metadata_turkish_t25_binary_merged.csv`,
   120 subjects = depression + comorbid). Alternatives: depression-only (fewer subjects) or
   depression vs comorbid stratified reporting. The target (`target_t25`) is depression regardless of
   comorbidity, so merged maximizes data; **confirm this is the intended population.** (Switchable via
   `metadata_csv` in config.)
2. **Threshold:** fixed at **25** per `thershold.txt` and used as a config knob. Confirm no
   subject-level re-derivation is wanted (currently file-level `depresyon_skoru`; labels are 100 %
   consistent within subject, so it doesn't matter — but worth a nod).
3. **135 unlabeled audio files** → dropped (audited). Confirm they shouldn't be labeled/used.
4. **Subject-level transcript handling for C′/text-only-subject:** Turkish has per-segment
   transcripts. Plan = concatenate per subject (`multi_transcript: concat`). Confirm vs. keeping
   text strictly chunk-level. (Chunk-level needs zero `runtime.py` change and is the first milestone.)
5. **Locked held-out test?** Default = none (pure 5-fold CV). Confirm CV-only is acceptable for the
   paper/report, or reserve a stratified test split.
6. **`features` column origin** (608-dim w2v2/MFCC) and `w2v2_predicted_score`: assumed precomputed
   acoustic features for the classical baseline. Confirm they're trustworthy if Approach E is used as
   a comparison number.
7. **Model paths on the server:** configs assume `/gpfs/.../models/Qwen2-Audio-7B-Instruct` and
   `${TEXT_MODEL_PATH}` for Qwen2-7B-Instruct (per existing configs/README). Confirm both exist on MN5.
8. **AUROC addition** touches shared `metrics.py`/`aggregate.py`. Low risk (additive key), but it
   changes metrics JSON for all datasets — confirm that's fine, or gate it behind a flag.
