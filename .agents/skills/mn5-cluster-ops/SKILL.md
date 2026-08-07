---
name: mn5-cluster-ops
description: Execute the authorized MareNostrum 5 experiment lifecycle for this repository, including branch-aware deployment, environment checks, Slurm submission and monitoring, selective rsync, failure and resubmission handling, compact-evidence retrieval, and local validation. Use for any MN5 connectivity, transfer, submission, cancellation, monitoring, remote inspection, or result-sync task.
---

# Operate experiments on MN5

Read `docs/DEVICES.md` and `docs/MN5_AGENT_EXECUTION_RUNBOOK.md` completely before every cluster or rsync workflow. Read the selected experiment-specific plan and the actual wrappers/configs completely before submission. These current files override cached endpoint, resource, and command details in this skill.

Submission is not completion. Finish only when every authorized job is accounted for, compact evidence and logs are local, audits pass locally, lifecycle state is updated, and any requested reporting is complete.

## Enforce authorization

Perform relevant read-only inspection without additional approval. Obtain explicit user authorization before submitting or cancelling jobs, performing a real local-to-cluster synchronization that can overwrite remote files, deleting anything, changing shared environments/resources, or starting a real cloud export. Approval for one mutation does not authorize later mutations.

Never execute training or evaluation directly on a transfer or login node.

## Resolve endpoints dynamically

- Use `ozu647717@transfer1.bsc.es` for rsync and read-only GPFS inspection.
- Probe `alogin2` first during the documented migration, then the documented `alogin1` fallback; use only a scheduler login that exposes `sbatch`, `squeue`, and `sacct`.
- Use `/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression` as the permanent project/results root unless the current device documentation says otherwise.
- Treat a failed `sinfo` check as a reason to investigate, not permission to submit through `transfer1`.

Do not expose credentials during connectivity checks.

## Deploy reconstructable source

Before transfer, record the branch, full commit SHA, clean/dirty state, selected configs and overrides, expected jobs, resources, output root, manifests, artifacts, and storage estimate. Production runs require clean committed source.

When using an optional branch workspace:

- Derive a collision-resistant workspace ID from the sanitized branch and a unique SHA prefix.
- Keep the execution workspace separate from the permanent `output_model` root.
- Audit wrappers for `PROJECT_ROOT` support and override `output_dirs.run_root` with an absolute permanent path.
- Keep generated manifests/splits in durable evidence or copy them with hashes before retiring the workspace.
- Never copy datasets, base models, checkpoints, or generated outputs into the disposable code workspace.

Dry-run every selective transfer, review destination changes, transfer without `--delete`, and verify remote checksums or the deployment manifest.

## Initialize and submit

Use the current environment commands from `docs/DEVICES.md`. Add the documented project-local `PYTHONPATH` only for hidden-state workflows and verify required imports without installing packages.

For a canonical single-fold run, after authorization and preflight:

```bash
CONFIG="$PWD/configs/main/<cfg>.yaml" RUN_NAME=<unique> FOLD=0 bash scripts/submit_train_and_eval.sh
```

Use experiment-specific wrappers for matrices, hidden classifiers, translation, merged runs, or other specialized workflows. Do not force those workflows through the canonical single-fold wrapper. Set and explicitly synchronize `EXPERIMENT_CONTEXT` when tracking a new run. Never reuse a run name until continuation and collision behavior are proven safe.

Run a uniquely named smoke job before a production matrix when required by the runbook. Record every returned job ID and dependency.

## Monitor to terminal evidence

Use both queue and accounting data:

```bash
squeue -j <ids> -o "%.18i %.10T %.12M %.10l %.6D %.30j %.20R"
sacct -j <ids> --format=JobIDRaw,JobName%32,State,ExitCode,Elapsed,MaxRSS,AllocCPUS,NodeList --units=G
```

- Require top-level `COMPLETED` with `ExitCode=0:0` and the expected artifacts.
- Treat an empty queue as unknown until reconciled with `sacct`.
- Interpret stderr with scheduler state and artifacts; warnings alone do not prove failure.
- Poll at sensible intervals, never block for more than 60 seconds in one wait, and resume from recorded IDs rather than resubmitting.

## Handle failures without corrupting identity

Record the job ID, state, exit code, node, elapsed time, logs, source/config hashes, and partial artifacts. Fix only the demonstrated cause, validate locally and remotely, and resubmit only failed or missing scope after authorization. Use a new attempt for a rerun and preserve the failed-to-rerun chain.

Do not delete studies/output directories, change experiment IDs silently, overwrite incompatible configuration hashes, cancel healthy jobs, or install packages as an incidental fix.

## Retrieve and validate evidence

Dry-run and then selectively retrieve through `transfer1`:

- metrics and final summaries;
- subject-level predictions;
- `run_config.yaml` and all tracking sidecars;
- audit JSONs and logs;
- only specifically required `best_model` adapters.

Exclude bulk `best_model` and `last_model` directories by default, but never exclude compact evidence. Never use `--delete`. Re-run the same audits and summarizers locally, compare local and remote counts/hashes, then use `provenance-reporting` before presenting any result.
