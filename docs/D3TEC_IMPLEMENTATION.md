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
