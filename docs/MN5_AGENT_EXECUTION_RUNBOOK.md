# MN5 Agent Execution Runbook

Last verified: 2026-08-01 (Europe/Istanbul).

This is the operational playbook for an agent that must do more than implement
an experiment. It covers transferring tested code to MareNostrum 5 (MN5),
submitting Slurm work, monitoring it to a terminal state, diagnosing failures,
synchronizing artifacts back to the local repository, validating the returned
results, updating reports, and completing the Git handoff.

Read `docs/DEVICES.md` first. It remains authoritative for device roles,
paths, environments, checkpoint policy, and safety boundaries. Read the
experiment-specific plan or runbook as well. For the raw hidden-state XGBoost
Optuna work, also read `docs/OPTUNA_RAW_XGBOOST_FOLLOWUP.md`.

Do not put passwords, private keys, tokens, or MFA material in this file or in
agent prompts.

## 1. The host distinction that agents must understand

The BSC endpoints have different jobs:

| Purpose | Endpoint | What to do there |
|---|---|---|
| File transfer | `ozu647717@transfer1.bsc.es` | `rsync`, remote file inspection, `du`, checksums |
| MN5 scheduler login | `ozu647717@alogin1.bsc.es` (or the currently available equivalent, presently `alogin2.bsc.es`) | `sbatch`, `squeue`, `sacct`, job/log inspection |
| Compute | Slurm-allocated nodes | Python training/evaluation launched by `sbatch` |

Known project path on the shared GPFS filesystem:

```text
/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression
```

Both endpoints see that same GPFS project tree. Transfer through `transfer1`;
submit and monitor through the available scheduler login. During the current
MN5 migration, `alogin1` may be unavailable and `alogin2` is the appropriate
replacement; the Slurm commands and project path are unchanged. Never run
the experiment directly on either login endpoint.

As verified on 2026-07-25 for the original login, and rechecked on
2026-08-01 through `alogin2` during the operating-system migration:

- `transfer1` accepted non-interactive SSH and exposed `rsync`.
- The reachable scheduler login (`alogin2` during the migration) accepted
  non-interactive SSH and exposed `sbatch`, `squeue`, and `sacct`.
- `alogin1` may be unavailable while its migration is in progress; this does
  not change the scheduler commands or the shared GPFS project path.
- The earlier `sinfo` check on `alogin1` returned an access/permission error.
  This is not by itself proof that submission is unavailable; verify
  account/QoS from the selected Slurm script and use a smoke job on whichever
  scheduler login is reachable.

Always re-run the lightweight connectivity checks because endpoints and
policies can change:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=15 \
  ozu647717@transfer1.bsc.es \
  'hostname; command -v rsync; test -d /gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression'

ssh -o BatchMode=yes -o ConnectTimeout=15 \
  ozu647717@alogin2.bsc.es \
  'hostname; command -v sbatch; command -v squeue; command -v sacct'

# If alogin2 is unavailable after the migration, repeat the check against
# alogin1 and use whichever scheduler login is currently reachable.
```

If either command fails, report the exact error. Do not invent another
hostname, alter SSH configuration, or expose credentials.

## 2. Runtime environment

On the reachable scheduler login (currently `alogin2`), enter the project and
initialize the environment before running preflight commands or submission
wrappers:

```bash
cd /gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression
module purge
module load bsc/1.0
module load miniforge/24.3.0-0
source /gpfs/projects/etur92/ozu647717/venvs/qwen_mn5_rebuilt/bin/activate
```

The raw hidden-state classifier dependencies are project-local. The base
environment alone does not expose XGBoost. For that workflow, this export is
required:

```bash
export PYTHONPATH="$PWD/.deps/qwen_hidden:$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

Then verify imports and versions:

```bash
python -c "
import optuna, xgboost, sklearn
print('optuna', optuna.__version__)
print('xgboost', xgboost.__version__)
print('sklearn', sklearn.__version__)
"
```

Versions verified on 2026-07-25:

```text
Optuna 4.4.0
XGBoost 2.1.4
scikit-learn 1.7.0
```

Do not install or upgrade packages on the shared environment as an incidental
fix. First check the selected worker script for its intended environment and
`PYTHONPATH`.

## 3. What “run the experiment” means

Unless the user narrows the scope, a request to implement and run an
experiment means the agent should complete this lifecycle:

1. Inspect the local worktree and read all relevant instructions.
2. Implement and validate locally.
3. Commit and push the tested code when authorized.
4. Dry-run the local-to-GPFS transfer.
5. Transfer only the necessary tested source/configuration files.
6. Verify remote checksums or file contents.
7. Verify the MN5 runtime and scheduler commands.
8. Run a small, uniquely named smoke job.
9. Monitor the smoke job to a terminal Slurm state and inspect its artifacts.
10. Test restart/idempotency when the experiment is designed to resume.
11. Dry-run the production submission and verify the exact job count.
12. Submit production jobs and record every job ID.
13. Monitor all jobs until none remain pending or running.
14. Use `sacct` and logs to prove terminal success; an empty `squeue` is not
    sufficient.
15. Run experiment-specific audits on GPFS.
16. Dry-run the GPFS-to-local artifact transfer.
17. Synchronize results, manifests, and logs back locally.
18. Re-run audits and summarizers locally.
19. Update compact result tables and reports.
20. Commit and push the final tracked deliverables when authorized.

Submission is not completion. The terminal condition is validated local
artifacts and reporting, not merely a list of Slurm job IDs.

## 4. Local preflight

Start from the local repository:

```bash
cd /home/emre/Projects/AudioLLM/LLM-Depression
git status --short
git branch --show-current
git log -1 --oneline
hostname
df -h .
```

Preserve unrelated user changes. Do not use destructive Git commands to
obtain a clean tree.

Read the selected scripts completely:

```bash
sed -n '1,260p' scripts/<submit-wrapper>.sh
sed -n '1,220p' scripts/<slurm-worker>.sh
sed -n '1,260p' configs/<selected-config-or-matrix>.yaml
```

Check shell and Python syntax and run the relevant tests. For example:

```bash
bash -n scripts/<submit-wrapper>.sh scripts/<slurm-worker>.sh
python -m py_compile path/to/changed_module.py
python -m unittest <relevant-test-module> -v
```

Before transfer, write down:

- local Git commit;
- exact config or matrix path;
- expected job count;
- experiment/run ID;
- output directory;
- objective and trial/epoch count;
- CPU/GPU allocation and wall time;
- expected artifacts;
- restart behavior;
- estimated storage.

Do not sync an ambiguous dirty source tree to GPFS. Commit the intended source
first when the user has authorized Git publication.

## 5. Safe local-to-GPFS synchronization

### 5.1 Preferred repository sync

The repository helper captures provenance and transfers through `transfer1`:

```bash
cd /home/emre/Projects/AudioLLM/LLM-Depression
bash scripts/sync_to_cluster.sh
```

It excludes `.git/`, respects `.gitignore`, and writes a `.provenance`
snapshot. Inspect the script before use.

Because the helper performs the real transfer immediately, use a manual
selective dry-run first when the remote tree might contain newer work.

The helper's `.gitignore` filter also means it will not transfer ignored
runtime inputs such as generated manifests under `outputs/`, nor the ignored
`docs/` directory. If a submitted job needs an ignored generated manifest,
transfer that exact file explicitly with selective rsync after reviewing a
dry run. Do not assume the repository helper copied it.

### 5.2 Selective rsync

Define task-specific paths:

```bash
LOCAL_PROJECT=/home/emre/Projects/AudioLLM/LLM-Depression
REMOTE_HOST=ozu647717@transfer1.bsc.es
REMOTE_PROJECT=/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression
cd "$LOCAL_PROJECT"
```

Preview only the implementation files required by the job:

```bash
rsync -avhn --itemize-changes --relative \
  baselines/<worker>.py \
  scripts/<submit-wrapper>.sh \
  scripts/<slurm-worker>.sh \
  configs/<matrix>.yaml \
  "$REMOTE_HOST:$REMOTE_PROJECT/"
```

Review every listed target. Then repeat without `-n`:

```bash
rsync -avh --itemize-changes --relative \
  baselines/<worker>.py \
  scripts/<submit-wrapper>.sh \
  scripts/<slurm-worker>.sh \
  configs/<matrix>.yaml \
  "$REMOTE_HOST:$REMOTE_PROJECT/"
```

Never add `--delete` to a cluster sync unless deletion was explicitly
requested and its complete effect was reviewed.

### 5.3 Verify the transfer

Do not assume a successful rsync exit means the intended source is active.
Compare checksums for the selected files:

```bash
sha256sum \
  baselines/<worker>.py \
  scripts/<submit-wrapper>.sh \
  scripts/<slurm-worker>.sh \
  configs/<matrix>.yaml

ssh "$REMOTE_HOST" \
  "cd '$REMOTE_PROJECT' && sha256sum \
    baselines/<worker>.py \
    scripts/<submit-wrapper>.sh \
    scripts/<slurm-worker>.sh \
    configs/<matrix>.yaml"
```

Also inspect the remote provenance snapshot or exact file headers. Do not
overwrite remote results, caches, checkpoints, or unrelated configs while
transferring code.

## 6. Remote preflight on the scheduler login

Open the scheduler-login session (currently `alogin2`; use `alogin1` if it is
the reachable equivalent):

```bash
ssh ozu647717@alogin2.bsc.es
```

Then:

```bash
set -euo pipefail
PROJECT_ROOT=/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression
cd "$PROJECT_ROOT"

module purge
module load bsc/1.0
module load miniforge/24.3.0-0
source /gpfs/projects/etur92/ozu647717/venvs/qwen_mn5_rebuilt/bin/activate

export PYTHONPATH="$PWD/.deps/qwen_hidden:$PWD${PYTHONPATH:+:$PYTHONPATH}"
python -V
python -c "import optuna, xgboost, sklearn; print(optuna.__version__, xgboost.__version__, sklearn.__version__)"

bash -n scripts/<submit-wrapper>.sh scripts/<slurm-worker>.sh
test -f configs/<matrix>.yaml
```

Inspect Slurm directives in the actual worker:

```bash
sed -n '1,80p' scripts/<slurm-worker>.sh
```

Confirm:

- account and QoS;
- CPU/GPU request;
- task and node counts;
- wall time;
- project working directory;
- environment activation;
- log directory;
- no accidental GPU request for CPU-only work.

For the raw XGBoost Optuna worker, the intended resources are one node, one
task, 20 CPUs, no GPU, account `etur92`, QoS `acc_ehpc`, and a four-hour
limit.

Check inputs and free space:

```bash
du -sh outputs/hidden_features outputs/hidden_classifiers 2>/dev/null
df -h "$PROJECT_ROOT"
```

Use task-specific checks rather than recursively listing large GPFS trees.

## 7. Smoke before production

A smoke run must:

- use a unique output/experiment ID;
- be small enough to finish quickly;
- execute the real Slurm worker and real environment;
- exercise representative data, including repeated responses if aggregation
  matters;
- not overwrite production or completed results;
- produce the normal artifact set.

For resumable Optuna studies, use two trials, wait for completion, invoke the
same smoke again, and verify that it remains at exactly two completed trials
instead of adding two more.

Submit through the repository wrapper or `sbatch`; never invoke the Python
training command directly on a login node.

Immediately record:

```text
job ID
experiment/run ID
config or matrix
output directory
submission timestamp
Git/provenance commit
```

Monitor and audit the smoke using the next two sections. Production submission
must wait until the smoke is proven successful.

## 8. Dry-run and production submission

Every matrix/wrapper that supports a dry run must be exercised first. For the
raw hidden-state Optuna matrix:

```bash
MATRIX="$PWD/outputs/optuna_followup_manifests/<stage>.yaml" \
DRY_RUN=1 \
bash scripts/submit_qwen_hidden_optuna_matrix.sh
```

Verify:

- exact expected job count;
- exact experiment ID and output path;
- correct objective per dataset;
- expected fold list;
- expected target trials;
- no collision with completed output directories;
- no GPU flags for CPU-only jobs.

Only then submit:

```bash
MATRIX="$PWD/outputs/optuna_followup_manifests/<stage>.yaml" \
DRY_RUN=0 \
bash scripts/submit_qwen_hidden_optuna_matrix.sh | tee /tmp/<stage>-submission.txt
```

Capture the numeric job IDs from the output. Do not rely solely on scrollback.
If the wrapper reports completed jobs as skipped, reconcile
`submitted + skipped` with the manifest count.

Do not blindly submit the same matrix again after a partial failure. First
inspect the wrapper's resume/collision checks and each output's configuration
hash.

## 9. Monitoring jobs correctly

### 9.1 Active queue

For known job IDs:

```bash
squeue -j <comma-separated-job-ids> \
  -o "%.18i %.10T %.12M %.10l %.6D %.30j %.20R"
```

For all of the user's active jobs:

```bash
squeue -u "$USER" \
  -o "%.18i %.10T %.12M %.10l %.6D %.30j %.20R"
```

Useful pending reasons include resources, priority, QoS limits, and
dependencies. A pending job is not failed.

### 9.2 Terminal accounting

When a job disappears from `squeue`, query `sacct`:

```bash
sacct -j <comma-separated-job-ids> \
  --format=JobIDRaw,JobName%32,State,ExitCode,Elapsed,MaxRSS,AllocCPUS,NodeList \
  --units=G
```

Require the top-level jobs to reach `COMPLETED` with `ExitCode=0:0`. Check for:

```text
FAILED
CANCELLED
TIMEOUT
OUT_OF_MEMORY
NODE_FAIL
PREEMPTED
BOOT_FAIL
```

An empty `squeue` does not mean success.

### 9.3 Logs

Inspect logs while jobs run:

```bash
tail -n 80 logs/<experiment-log-directory>/*.out
tail -n 80 logs/<experiment-log-directory>/*.err

rg -n -i \
  'traceback|error|exception|out of memory|oom|killed|segmentation|failed' \
  logs/<experiment-log-directory>
```

A non-empty `.err` file is not automatically failure—libraries may emit
warnings there. Reconcile the message with Slurm state, exit code, and
artifacts.

### 9.4 Agent behavior during long jobs

The agent should remain responsible for the run:

- poll at sensible intervals;
- do not use one blocking sleep longer than 60 seconds;
- give the user concise progress updates during ongoing work;
- use available wait/monitor mechanisms rather than abandoning the task;
- inspect newly written logs between queue polls;
- continue until every submitted job reaches a terminal state.

If the conversation system automatically continues the task, resume from the
recorded job IDs instead of resubmitting.

## 10. Failure handling

When a job fails:

1. Record the exact job ID, state, exit code, node, and elapsed time.
2. Inspect its `.out` and `.err`.
3. Verify the remote code/config checksum and environment versions.
4. Inspect the output directory for partial state.
5. Determine whether the workflow is safely resumable.
6. Fix only the demonstrated cause.
7. Validate the fix locally and remotely.
8. Re-run a smoke if the worker/runtime changed.
9. Resubmit only the failed/missing scope.

Do not:

- delete a SQLite study or output directory merely to make a rerun easy;
- change an experiment ID silently;
- overwrite an incompatible configuration hash;
- cancel healthy jobs because another job failed;
- label a run successful because most jobs completed.

If three consecutive attempts hit the same external blocker and no safe
progress remains, report the evidence and request user action.

## 11. Remote acceptance audit

After all jobs complete, run the experiment-specific audit on GPFS before
syncing. For the Optuna manifests:

```bash
python scripts/audit_qwen_hidden_optuna_manifest.py \
  --matrix outputs/optuna_followup_manifests/<stage>.yaml \
  --results-root outputs/hidden_classifiers \
  --output outputs/optuna_followup_manifests/<stage>_audit.json
```

Confirm:

- expected number of studies;
- expected completed trials;
- complete fold and subject coverage;
- no subject leakage;
- configuration/provenance match;
- all required final artifacts;
- no failed trial states;
- zero failed/missing Slurm jobs.

Do not start a result-dependent stage until the preceding stage passes its
audit.

## 12. Safe GPFS-to-local synchronization

Use `transfer1`, not a scheduler login, for artifact transfer.

Define:

```bash
LOCAL_PROJECT=/home/emre/Projects/AudioLLM/LLM-Depression
REMOTE_HOST=ozu647717@transfer1.bsc.es
REMOTE_PROJECT=/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression
EXPERIMENT_ID=<exact-experiment-id>
```

### 12.1 Preview result transfer

For deeply nested classifier outputs, include directories so rsync can reach
the named experiment directory, include that directory recursively, and
exclude unrelated files:

```bash
rsync -avhn --itemize-changes --prune-empty-dirs \
  --include='*/' \
  --include="$EXPERIMENT_ID/***" \
  --exclude='*' \
  "$REMOTE_HOST:$REMOTE_PROJECT/outputs/hidden_classifiers/" \
  "$LOCAL_PROJECT/outputs/hidden_classifiers/"
```

Review the dry run, then remove `-n`:

```bash
rsync -avh --itemize-changes --partial --prune-empty-dirs \
  --include='*/' \
  --include="$EXPERIMENT_ID/***" \
  --exclude='*' \
  "$REMOTE_HOST:$REMOTE_PROJECT/outputs/hidden_classifiers/" \
  "$LOCAL_PROJECT/outputs/hidden_classifiers/"
```

### 12.2 Sync logs

```bash
rsync -avhn --itemize-changes \
  "$REMOTE_HOST:$REMOTE_PROJECT/logs/slurm_qwen_hidden_optuna/$EXPERIMENT_ID/" \
  "$LOCAL_PROJECT/logs/slurm_qwen_hidden_optuna/$EXPERIMENT_ID/"
```

After review, repeat without `-n`.

### 12.3 Sync manifests and audits

Use explicit files or a narrowly scoped directory:

```bash
rsync -avhn --itemize-changes \
  "$REMOTE_HOST:$REMOTE_PROJECT/outputs/optuna_followup_manifests/" \
  "$LOCAL_PROJECT/outputs/optuna_followup_manifests/"
```

Do not sync model/checkpoint directories by default. Follow the selective
checkpoint rules in `docs/DEVICES.md`.

Never use `--delete` during result retrieval. Remote GPFS remains the
authoritative source until local verification completes.

## 13. Local verification and reporting

After transfer:

```bash
cd /home/emre/Projects/AudioLLM/LLM-Depression
du -sh outputs/hidden_classifiers logs/slurm_qwen_hidden_optuna
```

Re-run the same audits locally and compare their counts with the remote audit.
Search synchronized logs:

```bash
rg -n -i \
  'traceback|out of memory|oom|killed|segmentation|failed' \
  logs/slurm_qwen_hidden_optuna/<experiment-id>
```

Run the relevant summarizer, inspect the compact CSV manually, and verify that
the reported variants use the intended selection rule. For seed-stability
experiments, never choose the best seed using outer-fold metrics.

Update:

- compact tracked result tables;
- full report;
- job/study/trial counts;
- storage and runtime observations;
- limitations and evaluation warnings;
- exact experiment IDs and provenance.

Do not commit all ignored outputs just because they were synchronized. Commit
only the tracked deliverables requested by the user, unless artifact
publication was explicitly requested.

## 14. Git handoff

Before committing:

```bash
git status --short
git diff --stat
git diff -- <tracked-text-deliverables>
```

Binary files should be validated with their native reader before staging.
Preserve unrelated user changes.

When authorized:

```bash
git add -- <intended-files>
git commit -m "<specific outcome>"
git push origin main
git status --short
```

If a new file under `docs/` is ignored by this repository's `.gitignore`, add
that exact file deliberately with `git add -f`; do not force-add the whole
documentation directory.

The final handoff should state:

- what ran;
- job IDs or job-ID ranges;
- terminal Slurm result;
- audits performed;
- artifacts synchronized;
- reports updated;
- storage used;
- commit and push result;
- remaining scientific limitations.

## 15. Copy-paste assignment for another agent

Replace the bracketed fields before giving this to an agent:

```text
Read docs/DEVICES.md and docs/MN5_AGENT_EXECUTION_RUNBOOK.md completely, then
read [EXPERIMENT-SPECIFIC PLAN/RUNBOOK] completely.

Your task is operational, not advisory. Implement and locally validate
[EXPERIMENT]. After the implementation is proven and Git publication is
authorized, selectively synchronize the tested source to MN5 through
ozu647717@transfer1.bsc.es. Use
/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression as the remote project.
Verify remote checksums.

Use the currently reachable scheduler login (`alogin2` during the current
migration; `alogin1` otherwise) for Slurm submission and monitoring. Do not
run Python training on a login or transfer node. Initialize the documented module,
environment, and project-local PYTHONPATH. Dry-run the submission and verify
the exact expected job count, resources, paths, experiment IDs, and collision
behavior.

Submit a uniquely named smoke job first. Monitor it through squeue, sacct,
logs, exit code, and artifact validation. Test restart/idempotency when the
workflow is resumable. Only after the smoke passes, submit the authorized
production jobs and record every job ID.

Remain responsible for the jobs until all are terminal. An empty squeue is
not success: require sacct COMPLETED with ExitCode 0:0, inspect logs, and run
the experiment-specific acceptance audits. Diagnose and safely retry only
failed or missing scope; do not delete studies, overwrite incompatible output,
or resubmit healthy work.

After successful remote audits, dry-run and then rsync the exact results,
manifests/audits, and logs back through transfer1. Do not use --delete and do
not retrieve large checkpoints unless requested. Re-run audits and
summarizers locally, update [RESULT CSV/REPORT], verify all expected artifacts
and scientific warnings, then commit and push the intended tracked
deliverables if authorized.

Do not stop after implementation, rsync, submission, or initial queue
monitoring. The terminal condition is: all authorized jobs are accounted for,
results and logs are local, audits pass, reporting is updated, and the final
Git state is reported. Keep the user informed during long-running work.
```

## 16. Fast checklist

```text
[ ] Read DEVICES.md and the experiment runbook
[ ] Inspect local Git state and selected scripts/configs
[ ] Run local tests and dry runs
[ ] Record commit, IDs, expected jobs, resources, output paths
[ ] Verify transfer1 and the currently reachable scheduler-login connectivity
[ ] Dry-run selective local-to-GPFS rsync
[ ] Sync and verify remote checksums
[ ] Initialize modules, environment, and PYTHONPATH
[ ] Submit and fully validate a unique smoke job
[ ] Dry-run production and reconcile job count
[ ] Submit production and record every job ID
[ ] Monitor with squeue, sacct, and logs until terminal
[ ] Run remote acceptance audits
[ ] Dry-run GPFS-to-local result/log sync
[ ] Sync exact artifacts without --delete
[ ] Re-run audits/summarizers locally
[ ] Update reports and compact result tables
[ ] Commit/push intended deliverables when authorized
[ ] Report job outcomes, artifacts, audits, limitations, and Git commit
```
