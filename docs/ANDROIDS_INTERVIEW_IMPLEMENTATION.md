# ANDROIDS Interview dual-transcript implementation

The experiment described in
`docs/ANDROIDS_INTERVIEW_DUAL_TRANSCRIPT_IMPLEMENTATION_EXPERIMENT_PLAN.md`
is implemented behind `dataset: androids_interview`.

## Implemented data contract

`src/data/androids.py` discovers only participant turns under
`Interview-Task/audio_clip`, parses diagnostic-safe subject IDs such as
`05_P`, and treats `P` as depressed and `C` as non-depressed. It fails unless
the authoritative local corpus contains:

- 116 subjects: 64 patients and 52 controls;
- 874 parent turns;
- 1,302 equal-duration windows at the declared 30-second boundary.

Audio remains in the original WAVs. Every manifest row stores `start_time`,
`end_time`, `segment_transcript`, and `full_turn_transcript`. Segment cache
rows must match the canonical audio path and interval to within one
microsecond. Missing, duplicate, extra, empty, non-Italian, and
interval-mismatched transcript rows are fatal.

Only columns 8-12 in the `Interview` block of `fold-lists.csv` are parsed.
The five held-out sets contain 24, 23, 23, 23, and 23 subjects. Fold overlap,
unknown recordings, and incomplete pooled coverage are fatal.

## Segment ASR

Generate the aligned cache in the local `qwen3asr` environment:

```bash
python scripts/transcribe_multilingual_qwen3asr.py \
  --preset androids_interview \
  --androids-interview-segments \
  --segment-seconds 30 \
  --batch-size 8 \
  --list-files
```

Use repeatable `--include` filters and `--out` for the predeclared short,
30-60-second, and 530-second smoke turns:

```bash
python scripts/transcribe_multilingual_qwen3asr.py \
  --preset androids_interview \
  --androids-interview-segments \
  --segment-seconds 30 \
  --batch-size 8 \
  --include '57_CF25_3_2_w*.wav' \
  --include '49_CM54_4_1_w*.wav' \
  --include '05_PM53_4_1_w*.wav' \
  --out /tmp/androids_interview_segment_smoke.jsonl
```

Then run the full cache with `--resume`:

```bash
python scripts/transcribe_multilingual_qwen3asr.py \
  --preset androids_interview \
  --androids-interview-segments \
  --segment-seconds 30 \
  --batch-size 8 \
  --resume
```

The default artifacts are:

```text
interview_transcripts_qwen3_asr_italian_segments.jsonl
interview_transcripts_qwen3_asr_italian_segments.report.json
```

## Runtime conditions

The four configs are in `configs/experiments/androids_interview/`:

- `androids_interview_audio_only.yaml`
- `androids_interview_audio_text_segment_aligned.yaml`
- `androids_interview_audio_text_full_turn.yaml`
- `androids_interview_text_only.yaml`

For audio+text, `data.audio_text_transcript_scope` selects either the exact
window transcript or the complete parent-turn transcript. Audio-only ignores
this selector. Text-only deduplicates windows to parent turns, orders turns
numerically, concatenates each full turn once, and emits one example per
subject. The observed maximum subject transcript is 8,185 characters after
turn headings, so the configured 12,000-character cap does not truncate the
current corpus.

All audio conditions train on every window. The loss weight is
`1 / (subject turn count * parent-turn window count)`, rescaled to mean one.
Evaluation uses equal window-to-turn and turn-to-subject aggregation.

## Local validation

```bash
python -m py_compile \
  src/data/androids.py src/data/runtime.py src/train.py \
  scripts/transcribe_multilingual_qwen3asr.py \
  scripts/audit_androids_interview.py \
  scripts/audit_androids_interview_inputs.py

conda run -n llmdep4090 env PYTHONPATH=. pytest -q \
  tests/test_androids_interview_pipeline.py \
  tests/test_d3tec_pipeline.py \
  tests/test_daic_chunking.py

bash -n \
  scripts/sync_androids_interview_inputs_to_mn5.sh \
  scripts/run_androids_interview_worker_slurm.sh \
  scripts/run_androids_interview_summary_slurm.sh \
  scripts/run_androids_interview_audit_slurm.sh \
  scripts/submit_androids_interview_smoke.sh \
  scripts/submit_androids_interview_matrix.sh
```

After the real segment cache exists, run the complete manifest/runtime and
no-model 30-second input audit:

```bash
conda run -n llmdep4090 env PYTHONPATH=. \
  python scripts/audit_androids_interview_inputs.py
```

## MN5 execution

Input synchronization is dry-run by default and transfers only the four
transcript/report artifacts:

```bash
DRY_RUN=1 bash scripts/sync_androids_interview_inputs_to_mn5.sh
```

The smoke wrapper plans four concurrent fold-0 GPU jobs and one dependent CPU
audit:

```bash
DRY_RUN=1 RUN_ID=<unique-id> \
  bash scripts/submit_androids_interview_smoke.sh
```

Production refuses to submit without a passed smoke audit. It chains five
folds within each of four concurrent condition chains, then submits four
summaries and one matrix audit: 20 GPU jobs and 25 jobs total.

```bash
DRY_RUN=1 RUN_ID=<unique-id> \
  SMOKE_AUDIT_PATH=<passed-smoke-audit.json> \
  bash scripts/submit_androids_interview_matrix.sh
```

Set `DRY_RUN=0` only after reviewing the printed paths, dependencies, job
counts, provenance, and collision checks. Submission and monitoring must use
`alogin1.bsc.es`; input/result transfer must use `transfer1.bsc.es`.

The final audit requires five folds and 116 unique pooled subjects per
condition, common manifest/fold hashes, complete sample/response/subject
artifacts, the declared transcript scopes, macro-F1 selection, patience 3,
best-only saving, and leakage proofs. It also emits the predeclared paired
10,000-resample subject bootstrap for segment-aligned minus full-turn macro-F1,
positive-F1, and accuracy.

After the remote audit passes, retrieval is also dry-run-first and excludes
`best_model/` and `last_model/`:

```bash
DRY_RUN=1 RUN_ID=<production-run-id> \
  bash scripts/sync_androids_interview_results_from_mn5.sh
```

## Current execution status

The experiment was executed end to end on 2026-07-30.

### ASR and input acceptance

Qwen3-ASR generated all 1,302 declared segment rows with complete interval
coverage, no empty transcripts, and no failed windows. The synchronized input
artifacts have these SHA-256 hashes:

- full-turn JSONL:
  `c0c7e46352aad91f4ef2c158582d56b1b14c174c29ced60ff6ef2ab8208bcc24`
- full-turn report:
  `80ce118a27a05911c5fb06a50532d28821a39ae7817c7903f712c3bb1263d92d`
- segment-aligned JSONL:
  `66dffce3f3c00b96ea759aff98410662bffba91fd8e4243713cb8e47a61e500f`
- segment-aligned report:
  `6e85f3db200ca7216ce4cce464762be2d11fc434f54b842dae9e0d689d5ac505`

Both the local and MN5 input audits passed. The remote manifest hash was
`01a351f7277e4763a8bb9e4983bba190b265becafafca6d7ee04bdcfc948cbed`;
the official fold-content hash was
`dce1d7bafc21014927c6d4e9604c6f85699deeb4801381210eb17daade931bd9`,
and the emitted split-metadata hash used by the run audit was
`f75dd2ba7bb324af26de8c5ae3497d2108e6b50815c0ef6cbcade7de70992518`.
All audio examples remained at or below 29.988 seconds and all transcript
truncation counts were zero.

### MN5 execution

The passed smoke run was
`androids_interview_smoke_20260730T144546Z`. The production run was
`androids_interview_prod_20260730T145948Z`, executed from source commit
`caf7fbf2bd33d599c90687300080e880cad8b599`.

All 25 production jobs completed with exit code `0:0`:

- 20 H100 fold-training jobs;
- four condition-summary jobs;
- one final matrix-audit job.

The final audit status is `passed`. It confirms five folds, 116 pooled
subjects, label counts 52 control and 64 patient, common manifest/fold hashes,
complete prediction artifacts, and zero transcript truncation for every
condition. The remote audit SHA-256 is
`043273cdf6b9a33e094a4634c00e7e956ce55fd4ea8494afcb241c0a73f2e52c`.
A fatal-pattern scan across the production Slurm logs found no traceback, OOM,
killed process, missing-file, or interval error.

### Pooled held-out results

The values below pool the best-checkpoint outer-fold predictions over all 116
subjects.

| Condition | Accuracy | Macro-F1 | Positive F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| Audio only | 0.8966 | 0.8950 | 0.9077 | 0.8939 | 0.9219 |
| Audio + segment-aligned text | 0.8448 | 0.8437 | 0.8571 | 0.8710 | 0.8438 |
| Audio + full-turn text | 0.8448 | 0.8446 | 0.8500 | 0.9107 | 0.7969 |
| Text only | 0.8707 | 0.8695 | 0.8819 | 0.8889 | 0.8750 |

The predeclared 10,000-resample paired subject bootstrap found no clear
segment-aligned advantage over full-turn text. Segment-aligned minus full-turn
macro-F1 had mean delta `-0.0004` with 95% CI `[-0.0666, 0.0619]`;
accuracy had mean delta `0.0006` with CI `[-0.0603, 0.0603]`; positive F1 had
mean delta `0.0077` with CI `[-0.0537, 0.0694]`. Every interval crosses zero.

### Retrieval and independent local audit

Compact results were retrieved without `best_model/` or `last_model/`.
The retrieval generated a 1,677-file local SHA-256 manifest. The independent
local matrix audit also passed with the same manifest hash, split-metadata
hash, five folds, 116 subjects per condition, and identical bootstrap
confidence bounds and probabilities. Its tiny mean-delta differences are
floating-point rounding at approximately `1e-18`.

The retrieved acceptance artifacts are under:

```text
outputs/androids_interview_matrix/androids_interview_prod_20260730T145948Z/
```

The result-sync destination was corrected after this verification so future
retrievals always preserve the run-ID directory. A regression test covers that
path contract. Final validation: `85 passed, 1 skipped`.
