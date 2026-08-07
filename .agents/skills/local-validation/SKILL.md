---
name: local-validation
description: Select and run proportionate local validation for changes in this repository, including environment checks, targeted and full pytest suites, shell and Python syntax checks, config/document consistency audits, model-free dataset sanity checks, and optional model smokes. Use after editing code, scripts, configs, docs, or repo skills; before opening a PR or synchronizing source to MN5; and when reporting whether a local change is ready.
---

# Validate repository changes locally

Prove the changed scope without claiming checks that did not run. Read `AGENTS.md`, inspect the worktree, and preserve unrelated user changes.

## Establish the environment and scope

Start at the repository root:

```bash
git status --short
git diff --stat
git diff -- <intended-files>
conda activate llmdep4090
python --version
```

The default `base` environment lacks PyTorch. Use `llmdep4090` for repository tests unless a workflow document explicitly requires another environment. Never install or upgrade packages as incidental validation.

List the changed files and classify them as Python, shell, configuration, documentation/skills, data pipeline, model-dependent, or experiment-tracking. Select the smallest checks that exercise the changed behavior, then widen coverage when shared code or scientific semantics changed.

## Run cheap structural checks first

- Python: `python -m py_compile <changed.py>` when import-time execution is unnecessary.
- Shell: `bash -n <changed.sh>`.
- Skills: `python /home/emre/.codex/skills/.system/skill-creator/scripts/quick_validate.py <skill-dir>`.
- YAML/docs: inspect referenced paths with `rg`/`test -e`, compare claims with current canonical YAML and CLI `--help`, and search for superseded paths or protocol names.
- Git: ensure generated artifacts, credentials, ignored outputs, and unrelated changes are not staged.

Do not treat syntax checks as behavioral tests.

## Choose behavioral tests

Always invoke pytest as a module from the repository root:

```bash
python -m pytest tests/test_<relevant>.py -q
```

Useful routing:

- `src/experiment_tracking` or `tools/exp.py`: `tests/test_experiment_tracking_*.py`, `tests/test_experiment_registry.py`, `tests/test_experiment_cli.py`, and relevant report/W&B tests.
- `src/data`, sampling, manifests, or splits: the matching dataset tests plus `tests/test_runtime_multispan_audio.py` when audio example construction changes.
- `src/evaluate.py`, aggregation, or metrics: matching aggregation/chunking tests and experiment qualification/report tests.
- `src/features` or hidden classifiers: `tests/test_qwen_hidden_pipeline.py`, `tests/test_qwen_hidden_optuna.py`, and dataset-specific hidden-classifier tests.
- `src/merged`: `tests/test_symmetric_merged_*.py`.
- `src/translation`: `tests/test_translation_pipeline.py`.
- Slurm/submission scripts: relevant submission/workflow CLI tests plus `bash -n` for every changed shell file.

Run the full suite when shared runtime/model/data behavior changes or before publishing a substantial cross-cutting change:

```bash
python -m pytest tests/
```

Never use bare `pytest`; package resolution depends on `python -m pytest` from the repository root.

## Gate dataset and model checks

Run the model-free sanity suite only when its dataset roots are available and the task concerns manifests/splits or warrants end-to-end data validation:

```bash
./scripts/sanity_tests_no_model.sh
```

It creates or updates ignored manifest/audit outputs. Inspect its current source first because its dataset coverage may include archived compatibility fixtures.

Run model-loading, GPU, or local smoke checks only when they materially validate the change. Verify the matching base-model path, GPU memory, environment, and explicit tiny overrides first. Do not present a local single-GPU smoke as equivalent to MN5 production training.

Never initiate SSH, rsync, Slurm, W&B cloud export, or other external mutation under this skill. Use the relevant operational skill and authorization boundary.

## Validate scientific and documentation consistency

For canonical config or documentation changes, verify directly against every current `configs/main/*.yaml`:

- teacher-forced `original_teacher_forced` headline backend;
- positive-F1 selection and early stopping, mode `max`;
- `headline/binary_strict_*` reporting and no teacher-forced AUROC claim;
- frozen audio encoder defaults unless an explicit experiment opts in;
- `${PROJECT_ROOT}/configs/quarantines.yaml` remains referenced;
- DAIC audio examples keep constant K=4 per bundle and state the evaluation view;
- dataset thresholds, modalities, filenames, and counts match the directory inventory.

Treat `configs/README.md` and current YAMLs as more authoritative than general prose. Correct contradictions rather than documenting both as if compatible.

## Report the validation honestly

Return:

```text
Environment:
Changed scope:
Commands run:
Passed:
Failed:
Skipped and reason:
Generated artifacts:
Unrelated worktree changes preserved:
Readiness: ready | not ready
```

Include exact command outcomes. A skipped dataset/model/cluster check is not a pass. Do not report scientific results discovered during validation without applying `provenance-reporting`.
