# Turkish Results Mismatch Investigation

## Current Problem

The Turkish audio-only reruns did not reproduce the older audio-only scores, even after fixing the failed transcript row for `cy2-1-9-ank+depr.wav`.

The unexpected case is strongest for BDI >= 21:

| Run | ACC | F1 | Precision | Recall |
| --- | ---: | ---: | ---: | ---: |
| Old audio-only | 0.642 | 0.688 | 0.625 | 0.778 |
| Transcript repaired | 0.599 | 0.601 | 0.627 | 0.631 |
| Transcript repaired, `cy2-1-9` fixed | 0.582 | 0.557 | 0.582 | 0.594 |

For BDI >= 17, the fixed run is much closer to the old result:

| Run | ACC | F1 | Precision | Recall |
| --- | ---: | ---: | ---: | ---: |
| Old audio-only | 0.717 | 0.814 | 0.744 | 0.913 |
| Transcript repaired | 0.727 | 0.813 | 0.765 | 0.894 |
| Transcript repaired, `cy2-1-9` fixed | 0.716 | 0.806 | 0.764 | 0.867 |

## Initial Root Cause Found

The transcript repair process marked one labeled chunk as `FAIL`:

`cy2-1-9-ank+depr.wav`

The manifest builder skips rows with `repair_status == "FAIL"`, so the first repaired manifest removed one audio chunk even for audio-only training.

That changed row counts:

| Threshold | Old rows | First repaired rows | Change | Subject count |
| --- | ---: | ---: | ---: | --- |
| T17 | 1051 | 1050 | -1 chunk | unchanged |
| T21 | 1051 | 1050 | -1 chunk | unchanged |

The subject split did not change. The missing chunk was in fold 1 validation and in folds 0, 2, 3, and 4 training.

## Fix Applied

The repaired transcript row was corrected to:

```json
{"audio_path": "../../../../../home/emre/Projects/AudioLLM/Datasets/Turkish/all-files/cy2-1-9-ank+depr.wav", "transcript": "olamazsın yavrum.", "language": "tr", "repair_status": "REPAIRED", "repair_actions": ["manual_corrected_transcript"], "manual_review_recommended": false, "manual_review_reason_codes": [], "original_transcript": "mas sim, já."}
```

After rerunning, the fixed audio-only runs restored the validation sample IDs and row counts to the old runs for every fold.

## What Still Does Not Make Sense

Even with matching configs, matching split hashes, and matching validation sample IDs, the T21 audio-only fixed run remained worse than the old run.

Verified same across old and fixed T21/T17 audio-only:

- `selection_metric`: `inner_val_macro_f1`
- `selection_metric_mode`: `max`
- prediction backend: `likelihood`
- aggregation: `subject`
- input modality: `audio_only`
- split metadata hash: unchanged
- validation sample IDs: unchanged after the `cy2-1-9` fix

The old and fixed run configs are identical after ignoring expected provenance differences:

- `transcript_file`
- `manifest_hash`

## Evidence Of Divergence

T21 old vs fixed selected different best epochs:

| Fold | Old selected epoch | Fixed selected epoch |
| ---: | ---: | ---: |
| 0 | 3 | 7 |
| 1 | 2 | 2 |
| 2 | 6 | 3 |
| 3 | 6 | 4 |
| 4 | 5 | 1 |

T21 old vs fixed subject-level predictions changed for 39 subjects:

- 27 subjects flipped from predicted depressed to non-depressed.
- 12 subjects flipped from predicted non-depressed to depressed.

The pooled confusion matrix changed:

| Run | Confusion matrix |
| --- | --- |
| Old | `[[29, 29], [14, 48]]` |
| Fixed | `[[33, 25], [25, 37]]` |

So the T21 F1 drop is mainly a recall drop: true positives decreased from 48 to 37 and false negatives increased from 14 to 25.

## Current Hypotheses

1. Training nondeterminism is large enough to move T21 audio-only results substantially.
2. The old and new jobs may not have used byte-identical code or environment, because run artifacts do not record git commit or full environment state.
3. Some unexpected path may still depend on manifest content, even though audio-only prompt construction sets transcript text to empty when `data.use_text: false`.

## What We Tried

1. Added secondary `transcript_repaired` rows to `depression_results_table_no_emo.csv`.
2. Identified the missing labeled chunk `cy2-1-9-ank+depr`.
3. Corrected that row in `whisper_transcripts_repaired.jsonl`.
4. Reran repaired configs with `RUN_NAME_PREFIX=train_val_t*_rep_transcript_cy219fixed`.
5. Added `transcript_repaired_cy219fixed` rows to `depression_results_table_no_emo.csv`.
6. Verified old vs fixed configs, split hashes, validation sample IDs, and selection metric.

## What We Are Trying Next

Rerun the current configs against the original transcript file, using:

```bash
EXTRA_TRAIN_ARGS="--set transcript_file=whisper_transcripts.jsonl"
```

This requires `scripts/run_turkish_5fold.sh` to pass `EXTRA_TRAIN_ARGS` to the shared manifest prebuild. The script was updated to do that.

The purpose is to test whether the original-transcript rerun reproduces the old result. If it does not, then the mismatch is likely training/environment nondeterminism. If it does reproduce the old result while the fixed repaired run remains worse, then the repaired manifest content is affecting audio-only behavior in an unexpected way.

