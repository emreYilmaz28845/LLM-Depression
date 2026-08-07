---
name: experiment-tracking
description: Operate and inspect this repository's implemented experiment-tracking system, including experiment groups, logical runs, attempts, folds, Git identity, lifecycle sidecars, SQLite registry queries, deterministic reports, W&B export, legacy import, and evaluation idempotency. Use when planning or registering an experiment, examining lifecycle evidence, importing or querying runs, generating reports, exporting evidence, or changing the tracking implementation.
---

# Operate experiment tracking

Treat `docs/AudioLLM_Experiment_Workflow_Implementation_Plan_v2.md` as the architecture and execution contract, and the current source, schemas, CLI help, and tests as the executable truth. When changing the implementation, read Sections 25–28 and every file named by the active task before editing. Do not use dated smoke results as procedural truth.

## Keep responsibilities separate

- Use this skill for identity, sidecars, lifecycle state, registry, reports, and W&B evidence export.
- Use `mn5-cluster-ops` for remote deployment, Slurm, monitoring, and synchronization.
- Use `provenance-reporting` before presenting or writing any metric.
- Use `git-pr` after validating agent-made code, configuration, or methodology changes.

## Preserve identity and Git provenance

Use the hierarchy:

```text
experiment group
  -> logical run
    -> attempt
      -> fold
        -> jobs
          -> evaluations, metrics, and artifacts
```

One coherent branch/PR may produce many attempts, seeds, and folds. Store the full Git SHA, branch, clean/dirty status, source/deployment hash, and Issue/PR identifiers when available. Production experiments require reconstructable committed source; explicitly labeled dirty-source smoke runs are not reportable.

Attempt IDs have the form `<UTC timestamp>-<sanitized logical run>-<git SHA prefix>-<random suffix>`. Never reuse an attempt for a rerun.

## Preserve lifecycle and sidecars

Use the enforced lifecycle:

`PLANNED -> DEPLOYED -> SUBMITTED -> RUNNING -> COMPLETED_ON_MN5 -> SYNCED_LOCALLY -> LOCALLY_VALIDATED -> REPORTABLE`

Use `IMPORTED_LEGACY -> LOCALLY_VALIDATED -> REPORTABLE` only when the evidence qualifies. Move failed or cancelled work to `SUPERSEDED` through a new attempt; never relabel it complete.

Keep these sidecars beside `run_config.yaml`:

- `metadata.json`: attempt, source, configuration, seed, manifest, and split identity.
- `status.json`: current state and transition history; require `state == history[-1].to`.
- `jobs.jsonl`: append-only job events; derive current job state from the latest event per job key.
- `artifacts.json`: hashed artifact records and location flags.
- `evaluations.json`: idempotent evaluation records keyed by evaluation ID.

Keep the writable SQLite registry local at `outputs/experiment_registry/experiments.sqlite`. Never let Slurm workers write it.

## Plan and inspect with current CLIs

Inspect each CLI's `--help` before relying on optional arguments. Safe local entrypoints include:

```bash
bash scripts/submit_experiment.sh plan --group <group.yaml> --config <config.yaml> --seeds "7 1337 2024" --folds "0"
bash scripts/collect_experiment.sh --attempt <attempt-id> --fold <n> --output <dir> --dry-run
python tools/rebuild_experiment_registry.py --scan-root <path> --dry-run
python tools/import_experiment.py --run-dir <run-dir> --dry-run
python tools/exp.py list
python tools/exp.py show <attempt-id> --fold <n>
python tools/exp.py provenance <metric-id>
python tools/exp.py jobs --failed
python tools/exp.py best --dataset <d> --metric <m> --namespace <ns> --backend <b> --view <v> --aggregation <a>
```

Require all six qualifiers for `best`. Treat the plan and collection scripts according to their current implementation; generated dry-run commands are not proof that deployment or collection occurred.

## Generate reports and export evidence

```bash
python tools/generate_run_report.py --attempt-id <id> --fold <n>
python tools/generate_group_report.py --attempts <csv> --metric-name <m> --namespace <ns> --backend <b> --view <v> --aggregation <a>
python tools/export_run_to_wandb.py --attempt-id <id> --db outputs/experiment_registry/experiments.sqlite --mode dry_run
```

- Keep reports deterministic unless `--with-timestamp` is explicitly requested.
- Leave researcher conclusions blank unless supplied by the researcher.
- Refuse incompatible group aggregation and ambiguous legacy evidence.
- Keep W&B post-run and evidence-first. Never import or call W&B in training/evaluation.
- Use dry-run by default. Real cloud export requires explicit authorization and local authentication.
- Exclude transcripts, prompts, subject identifiers, dataset paths, and credentials from exported payloads.

## Preserve evaluation idempotency

Derive each evaluation ID from its attempt, fold, dataset, split, protocol, checkpoint role/path, backend, view, aggregation, namespace, and metrics hash. Skip identical content with the same ID; refuse different content under the same ID. A different view or aggregation creates a different record.

After changes, run the relevant targeted tests and the task acceptance gate. Use the Section 28 handoff fields and distinguish observed outputs from plans.
