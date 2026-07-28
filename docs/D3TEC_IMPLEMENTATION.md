# D3TEC implementation runbook

The implementation for `D3TEC_IMPLEMENTATION_EXPERIMENT_PLAN.md` is isolated
behind `dataset: d3tec`. Existing dataset behavior is unchanged.

## Local data preparation

Generate the canonical segment-aligned transcript cache in the `qwen3asr`
environment:

```bash
python scripts/transcribe_multilingual_qwen3asr.py \
  --preset d3tec \
  --d3tec-segments \
  --batch-size 16
```

The command slices temporary equal-duration windows for ASR only. Its public
JSONL rows point to the original SM-27 WAV and record `start_time`, `end_time`,
`response_id`, `sample_id`, `prompt_id`, and `segment_index`. It never writes
duplicate dataset WAVs. Use `--resume` after an interruption. Use
`--list-files` to inspect all canonical windows without loading the model.

Build and validate the shared manifest:

```bash
python src/data/build_manifest.py \
  --config configs/experiments/d3tec/d3tec_audio_only_rotary.yaml
```

The builder deliberately fails on missing, duplicate, empty, non-Spanish, or
extra transcript rows. It also verifies the authoritative PHQ-9 labels against
the derived binary CSV and emits manifest CSV/JSONL, subject metadata, folds,
fold distributions, transcript joins, chunk windows, label-source audit, input
hashes, manifest hash, and fold hash.

## Experiment configs

The seven production configs live in `configs/experiments/d3tec/`:

- `d3tec_audio_only_{rotary,flat,normalized}.yaml`
- `d3tec_audio_text_{rotary,flat,normalized}.yaml`
- `d3tec_text_only.yaml`

Audio configs use eight matched virtual epochs and one example per optimizer
input. Validation and outer evaluation always use every segment. Text-only
constructs one subject example from the original 27 full-response transcripts
in numeric prompt order.

## Local checks

```bash
python -m py_compile \
  src/data/d3tec.py src/data/runtime.py src/aggregate.py \
  src/evaluate.py src/train.py scripts/audit_d3tec_matrix.py

bash -n \
  scripts/run_d3tec_worker_slurm.sh \
  scripts/submit_d3tec_smoke.sh \
  scripts/submit_d3tec_matrix.sh

pytest -q tests/test_d3tec_pipeline.py
```

The D3TEC tests cover interval slicing, contiguous windows, rotary balance and
coverage, flat/normalized schedule identity, normalized response/subject
weights, and response-weighted hierarchical aggregation.

## MN5 smoke and production

After following `DEVICES.md` and `MN5_AGENT_EXECUTION_RUNBOOK.md`, first inspect
the transcript transfer and compare local/remote hashes:

```bash
DRY_RUN=1 bash scripts/sync_d3tec_inputs_to_mn5.sh
DRY_RUN=0 bash scripts/sync_d3tec_inputs_to_mn5.sh
```

The transfer helper uses `transfer1`, never uses `--delete`, and transfers only
the two transcript JSONLs and their QC reports.

Then inspect
the three smoke submissions:

```bash
DRY_RUN=1 RUN_ID=<unique-id> bash scripts/submit_d3tec_smoke.sh
```

Submit only after the paths and run ID are correct:

```bash
DRY_RUN=0 RUN_ID=<unique-id> bash scripts/submit_d3tec_smoke.sh
```

After all three smoke jobs pass, inspect the production topology:

```bash
DRY_RUN=1 RUN_ID=<unique-id> bash scripts/submit_d3tec_matrix.sh
```

A clean dry run reports 35 GPU jobs, seven summary jobs, and one final matrix
audit. Production submission uses:

```bash
DRY_RUN=0 \
SMOKE_AUDIT_PATH=outputs/d3tec_smoke/<smoke-run-id>/audit.json \
RUN_ID=<unique-id> \
bash scripts/submit_d3tec_matrix.sh
```

Each config is a sequential five-fold `afterok` chain. The seven chains run
independently. Every GPU job trains, selects by inner-validation macro-F1
(AUROC/loss/earlier-epoch tie-breaks), and evaluates the selected checkpoint
once on the outer holdout. Existing audited folds are skipped; incomplete
colliding fold directories cause a hard failure rather than being overwritten.

The final CPU audit requires exactly 62 pooled out-of-fold subjects per config,
checks leakage and artifact coverage, requires common manifest/split hashes,
writes the seven-config result table, and runs the predeclared 10,000-resample
paired subject bootstraps.

## Completed MN5 production run

The production matrix completed on 2026-07-29:

- source commit: `201ec0affb9c8ca7239a8a0c37b462538cb9091d`;
- smoke ID: `d3tec_smoke_20260728T134717Z`;
- production ID: `d3tec_prod_20260728T135954Z`;
- smoke jobs: `43923944`-`43923947`;
- production: 35 GPU folds, seven summaries, and matrix audit `43924656`;
- Slurm result: all 43 production jobs `COMPLETED` with `ExitCode=0:0`;
- elapsed production wall-clock window: approximately 8 hours 22 minutes;
- audio fold runtimes: approximately 1 hour 27 minutes to 1 hour 49 minutes;
- text-only fold runtimes: approximately 4 to 6 minutes.

The pooled, strict 62-subject out-of-fold results are:

| Configuration | Accuracy | Positive F1 | Macro F1 | AUROC | Invalid subjects |
|---|---:|---:|---:|---:|---:|
| Audio-only rotary | 0.581 | 0.435 | 0.551 | 0.536 | 0 |
| Audio-only flat | 0.581 | 0.500 | 0.569 | 0.515 | 0 |
| Audio-only normalized | 0.581 | 0.500 | 0.569 | 0.576 | 0 |
| Audio+text rotary | 0.565 | 0.471 | 0.550 | 0.554 | 0 |
| Audio+text flat | 0.548 | 0.440 | 0.531 | 0.549 | 0 |
| Audio+text normalized | 0.613 | 0.429 | 0.568 | 0.574 | 0 |
| Text-only | 0.516 | 0.423 | 0.503 | 0.619 | 0 |

The selected epochs by fold were:

- audio-only rotary: `7, 7, 5, 5, 6`;
- audio-only flat: `1, 3, 8, 6, 2`;
- audio-only normalized: `7, 6, 8, 6, 6`;
- audio+text rotary: `8, 7, 7, 4, 8`;
- audio+text flat: `3, 7, 8, 2, 7`;
- audio+text normalized: `5, 6, 5, 5, 7`;
- text-only: `5, 2, 8, 6, 6`.

All six predeclared 10,000-resample paired policy comparisons had 95% confidence
intervals spanning zero for macro-F1, positive-F1, and accuracy. The matrix
therefore does not establish a statistically clear winner among chunk policies.
Audio-only flat and normalized produced the same hard subject predictions and
classification metrics, but their score margins differed, yielding different
AUROC values.

Strict parsing recorded many invalid segment predictions (881-943 pooled per
audio configuration), while hierarchical response/subject aggregation produced
zero invalid responses and zero invalid subjects. These invalid segment counts
remain part of the reported protocol and must not be silently discarded.

The synchronized local artifacts are under:

```text
outputs/d3tec_matrix/d3tec_prod_20260728T135954Z/
outputs/d3tec_jobs/d3tec_prod_20260728T135954Z.tsv
outputs/manifests_d3tec/
outputs/splits_d3tec/
logs/slurm_d3tec/
output_model/experiments/d3tec/
```

`best_model/` and `last_model/` were deliberately not retrieved. The local
matrix re-audit passed and reproduced the remote result CSV exactly. The audited
manifest hash is
`67a62eb73b4ab7e0cd810b81af5e424f6bf9deea9cfdbc322fb32057a6e6f799`;
the split hash is
`a672e309fb193d7fd76e7283f5f42828c33713fe770ccbbf90f1ed72bf3fc15c`.

Interpret the results with the predeclared limitations: only 62 subjects,
subject-level labels inherited by segments, unavailable prompt semantics,
machine-generated Spanish transcripts for audio+text, and SM-27-only audio.
