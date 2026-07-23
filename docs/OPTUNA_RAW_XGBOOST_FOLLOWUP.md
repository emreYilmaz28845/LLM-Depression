# Raw XGBoost Optuna Follow-up Experiment

Status: implementation and execution guide.

Read `docs/DEVICES.md` and `docs/ENVIRONMENT_NOTES.md` before transferring
files or submitting jobs. The completed 50-trial results under
`xgb_optuna_raw/` are immutable inputs for comparison and are never resumed by
this experiment.

## Experiment identities

| Purpose | Result directory / experiment ID |
|---|---|
| 150 trials, original depth range, inner seed 1337 | `xgb_optuna_raw_t150_d6_seed1337_inner1337` |
| 150 trials, depth up to 8, inner seed 1337 | `xgb_optuna_raw_t150_d8_seed1337_inner1337` |
| 150 trials, original depth range, inner seed 7 | `xgb_optuna_raw_t150_d6_seed1337_inner7` |
| 150 trials, original depth range, inner seed 2024 | `xgb_optuna_raw_t150_d6_seed1337_inner2024` |

`seed1337` fixes both the Optuna sampler and XGBoost. `inner<N>` controls only
the stratified subject-fold assignment. Every directory contains its own
SQLite study, configuration hash, trials, model, predictions, and metrics.

The `standard_d6` search profile is identical to the original experiment.
The `depth8` profile changes only the upper bound of `max_depth` from 6 to 8.

## Local validation

From the repository root:

```bash
python -m unittest tests.test_qwen_hidden_optuna -v
python -m unittest tests.test_qwen_hidden_pipeline -v
python -m py_compile \
  baselines/qwen_hidden_xgb_optuna.py \
  baselines/summarize_qwen_hidden_optuna_stability.py \
  scripts/build_qwen_hidden_optuna_followup_matrix.py
bash -n \
  scripts/run_qwen_hidden_optuna_slurm.sh \
  scripts/submit_qwen_hidden_optuna_matrix.sh
```

Generate the deterministic manifests under ignored output storage:

```bash
mkdir -p outputs/optuna_followup_manifests
python scripts/build_qwen_hidden_optuna_followup_matrix.py \
  --stage stage1 \
  --output outputs/optuna_followup_manifests/stage1.yaml
python scripts/build_qwen_hidden_optuna_followup_matrix.py \
  --stage pilot \
  --output outputs/optuna_followup_manifests/pilot.yaml
```

The manifests must report exactly 33 and 22 jobs respectively.

## MN5 preflight and smoke

Synchronize only after committing and pushing the tested source. Follow the
selective rsync and scheduler checks in `docs/DEVICES.md`. Do not execute
Python training on `transfer1`; submit through the intended MN5 Slurm
scheduler.

On the cluster repository, activate:

```bash
module purge
module load bsc/1.0
module load miniforge/24.3.0-0
source /gpfs/projects/etur92/ozu647717/venvs/qwen_mn5_rebuilt/bin/activate
export PYTHONPATH="$PWD/.deps/qwen_hidden:$PWD${PYTHONPATH:+:$PYTHONPATH}"
python -c "import optuna, xgboost, sklearn; print(optuna.__version__, xgboost.__version__, sklearn.__version__)"
```

Before production, run two-trial smoke studies in uniquely named directories:

```text
xgb_optuna_raw_smoke_t2_d6_seed1337_inner1337
xgb_optuna_raw_smoke_t2_d8_seed1337_inner7
```

Use a cache containing repeated responses per subject. Run the first smoke
twice and verify that the second invocation resumes/skips at exactly two
completed trials. Confirm that the two smoke directories and the existing
`xgb_optuna_raw` directory remain distinct, and that final evaluation is
loaded only after tuning.

## Staged production execution

For every manifest, inspect it and dry-run before submission:

```bash
MATRIX="$PWD/outputs/optuna_followup_manifests/<stage>.yaml" \
DRY_RUN=1 \
bash scripts/submit_qwen_hidden_optuna_matrix.sh
```

If the count and paths are correct, submit by changing `DRY_RUN=1` to
`DRY_RUN=0`. The wrapper uses one CPU-only Slurm task, 20 CPUs, account
`etur92`, QoS `acc_ehpc`, and a four-hour limit. Record every returned job ID.

### Stage 1

Submit `stage1.yaml`: 33 studies × 150 trials. Wait for all studies to
complete and audit them before generating later result-dependent stages.

```bash
python scripts/audit_qwen_hidden_optuna_manifest.py \
  --matrix outputs/optuna_followup_manifests/stage1.yaml \
  --results-root outputs/hidden_classifiers \
  --output outputs/optuna_followup_manifests/stage1_audit.json
```

### Depth-8 sensitivity

Generate this manifest only after stage 1 is complete:

```bash
python scripts/build_qwen_hidden_optuna_followup_matrix.py \
  --stage depth8 \
  --results-root outputs/hidden_classifiers \
  --output outputs/optuna_followup_manifests/depth8.yaml
```

The selector reads only stage-1 `best_params.json` and metadata. If any fold
of a dataset–condition selects `max_depth=6`, all outer folds of that
condition are included. An empty manifest means no depth study is required.

### Inner-fold seed pilot

Submit `pilot.yaml`: inner seeds 7 and 2024 for DAIC text-only, CMDC
audio+text, and Turkish text-only. The seed-1337 counterparts are reused from
stage 1, producing a three-seed panel without duplicate jobs.

After all 22 pilot jobs finish:

```bash
python baselines/summarize_qwen_hidden_optuna_stability.py \
  --root outputs/hidden_classifiers \
  --output-dir outputs/hidden_classifiers/optuna_stability \
  --gate-threshold 0.03
```

The summarizer pools outer subjects separately within each seed. It never
concatenates predictions from different seeds. It sets `expand_all=true` if
the pooled primary-metric range is at least 0.03 for any representative
condition. DAIC/CMDC use positive F1 and Turkish uses macro-F1.

### Conditional full seed expansion

Generate and submit the expansion only when the summary gate triggers:

```bash
python scripts/build_qwen_hidden_optuna_followup_matrix.py \
  --stage expansion \
  --stability-summary outputs/hidden_classifiers/optuna_stability/stability_summary.json \
  --output outputs/optuna_followup_manifests/expansion.yaml
```

The expansion contains exactly 44 jobs: inner seeds 7 and 2024 for the
remaining 22 outer evaluations. The builder refuses to create it when the
gate is false.

## Acceptance audit and reporting

For every submitted study verify:

- exactly 150 completed trials and one final fit;
- matching experiment ID, profile, seeds, cache hashes, and output path;
- three disjoint inner subject folds with complete validation coverage;
- zero outer train/evaluation subject overlap;
- all final artifacts present;
- no failed jobs, tracebacks, OOM events, or GPU requests.

Run `scripts/audit_qwen_hidden_optuna_manifest.py` against every non-empty
completed manifest. Do not generate a result-dependent next stage until the
preceding audit passes.

Run `baselines/summarize_qwen_hidden.py` after syncing results. Report the
original 50-trial result, 150-trial result, depth-8 sensitivity, and seed
stability as separate variants. Never choose the best seed using outer
metrics. Preserve the warning that Turkish results are table-aligned
outer-validation estimates because the underlying Qwen checkpoints were
selected on those validation folds.
