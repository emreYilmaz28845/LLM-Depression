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

Implementation and local structural validation are complete. The real
1,302-row segment-aligned ASR cache has not been generated by this code change,
and no MN5 jobs have been submitted. Those are explicit data-preparation and
compute stages, not silently fabricated implementation artifacts.
