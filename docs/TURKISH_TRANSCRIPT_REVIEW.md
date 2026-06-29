# Turkish Transcript Review

Date: 2026-06-26

## Scope

This review combines:

- a deterministic pass from `python scripts/review_turkish_transcripts.py`
- a semantic/conceptual pass over all `very_bad` clips and representative `ok` / `minor_issue` clips with subject context

Deterministic outputs were written to:

- `outputs/reviews/turkish_transcripts/turkish_transcript_review.csv`
- `outputs/reviews/turkish_transcripts/turkish_transcript_review.jsonl`
- `outputs/reviews/turkish_transcripts/turkish_transcript_review_summary.json`

## Deterministic Summary

- Total transcripts reviewed: 1186
- Labeled transcripts in merged metadata: 1051
- Unlabeled transcripts: 135
- Subjects all / labeled: 135 / 120

Quality labels across all 1186 clips:

- `ok`: 917 (77.32%)
- `minor_issue`: 266 (22.43%)
- `needs_audio_check`: 0
- `likely_bad`: 0
- `very_bad`: 3 (0.25%)

Quality labels on the labeled subset only:

- `ok`: 806 (76.69%)
- `minor_issue`: 244 (23.22%)
- `very_bad`: 1 (0.10%)

Issue counts across all clips:

- `truncated`: 247
- `too_short`: 24
- `repetition`: 20
- `wrong_context`: 3
- `language_mismatch`: 2
- `odd_characters`: 2
- `semantic_break`: 2

## Bottom Line

The corpus is viable for downstream depression classification after removing a very small set of clearly corrupted clips. The labeled training subset is materially cleaner than the raw directory: only 1 of 1051 labeled clips is clearly unusable.

The main practical concern is not gross corruption. It is transcript form:

- many clips end with ellipsis but still contain useful semantic content
- a small number are tiny continuation fragments that add almost no information
- some clips contain interviewer prompt bleed-through, which matters more for text-only modeling than for subject-level multimodal aggregation

## Semantic Findings

### 1. The true hard failures are rare and obvious

Three clips are genuine semantic outliers and should be excluded:

- `ak3-1-11-ank.wav` -> English fragment: `The`
- `cy2-1-9-ank+depr.wav` -> Portuguese fragment: `mas sim, já.`
- `sb2-1-9-ank+depr.wav` -> unrelated Korean news snippet: `MBC 뉴스 김성현입니다.`

Only `cy2-1-9-ank+depr.wav` is in the labeled subset. The other two do not affect supervised training if training uses `metadata_turkish_t25_binary_merged.csv`.

### 2. Most `truncated` flags are still semantically usable

This is the largest issue bucket, but it is mostly a formatting/segmentation problem rather than semantic failure.

- 247 clips are flagged `truncated`
- 232 of those are `ellipsis_only`
- only 15 are likely continuation fragments
- median truncated clip length is still 40 words and 20.0 seconds

Observed pattern:

- Many ellipsis cases still contain a coherent answer and only lose the last phrase.
- Example: `aa1-1-11-ank.wav` is clearly meaningful despite ending mid-thought.
- The low-value cases are the tail fragments such as `aa1-1-12-ank.wav` -> `düşüncelerim olmuyor`

Interpretation:

- For subject-level aggregation, most ellipsis-only clips should be kept.
- For clip-level text classification, the 15 continuation fragments should be dropped.

### 3. Short clips are usually low-information, not misleading

- 24 clips are flagged `too_short`
- 15 of those are also continuation fragments
- 7 are pure short answers without stronger corruption markers

Representative examples:

- harmless short answer: `at1-1-10-depr.wav` -> `Yok yani.`
- terminal closing: `ht1-1-11-depr+ank.wav` -> `Görüşürüz.`
- clipped tail: `ss2-1-5-depr.wav` -> `Böyle`

Interpretation:

- These clips usually do not inject wrong content.
- They are mostly just low-yield training samples.
- Dropping them is reasonable for text-only or clip-level setups.

### 4. Repetition is usually conversational, but a few cases deserve caution

- 20 clips are flagged `repetition`
- no clip crossed the script threshold into `likely_bad`

What I observed:

- Many repetitions look like normal hesitation, self-repair, or emphasis in spontaneous speech.
- Example: `ag1-1-2-depr+ank.wav` repeats the same idea but still conveys a coherent narrative.
- A smaller subset looks closer to unstable ASR looping or verbal perseveration.
- Strongest example: `ny3-1-7-depr.wav`, which devolves into repeated `bu insanın ...`

Interpretation:

- Repetition does not appear widespread enough to invalidate the corpus.
- It is worth manually spot-checking the most loop-like cases before using transcripts for clip-level text modeling.

### 5. Interviewer prompt bleed-through is a real conceptual issue

This is the most important non-obvious finding from the semantic pass.

- 50 clips (4.22%) show clear interviewer/prompt contamination
- 42 labeled clips (4.00% of labeled data) show the same issue

Typical markers:

- `Peki`
- `Efendim?`
- `Bugün ne yaptınız buraya gelinceye kadar?`
- `Bir olay yaşadın mı?`
- `Başka neler?`

Representative example:

- `hy1-1-1-depr.wav` includes both the prompt and the subject response in the same transcript

Why it matters:

- For subject-level multimodal models, this is tolerable if it stays a small fraction of the total text.
- For text-only models, interviewer text dilutes subject signal and can make clip-level samples less clean.
- If prompts are consistent across subjects, they probably do not create label leakage, but they do reduce transcript purity.

### 6. ASR noise is common, but meaning is usually preserved

Across `ok` and `minor_issue` samples, I saw frequent orthographic and lexical errors, but most transcripts remained interpretable at the sentence or discourse level.

Common pattern:

- malformed words
- grammatical drift
- phonetic substitutions
- disfluencies rendered literally

Despite that, the emotional and autobiographical content usually remains recoverable, which is what matters for downstream depression classification.

## Recommendation

### Safe to use as-is for subject-level aggregation after minimal cleanup

Recommended minimum exclusions:

- `ak3-1-11-ank.wav`
- `cy2-1-9-ank+depr.wav`
- `sb2-1-9-ank+depr.wav`

Recommended additional exclusions for text-only or clip-level experiments:

- all 15 likely continuation fragments
- optionally all 24 `too_short` clips
- optionally the strongest repetition outliers such as `ny3-1-7-depr.wav`

Recommended improvement if transcript quality matters materially:

- strip interviewer turns / prompt bleed-through before text-only training

## Practical Assessment

If the goal is subject-level depression classification with transcript aggregation, the Turkish transcript set is usable and mostly semantically intact after removing the 3 hard-failure clips.

If the goal is clip-level text classification, I would treat the current transcripts as usable only after a stricter cleanup pass that removes:

- the 3 corrupted clips
- continuation fragments
- very short low-information clips
- interviewer-contaminated segments when possible

## Files Produced

- `TURKISH_TRANSCRIPT_REVIEW.md`
- `outputs/reviews/turkish_transcripts/turkish_transcript_review.csv`
- `outputs/reviews/turkish_transcripts/turkish_transcript_review.jsonl`
- `outputs/reviews/turkish_transcripts/turkish_transcript_review_summary.json`
