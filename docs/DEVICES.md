# Devices, Hosts, and Runtime Environments

Last verified: 2026-07-25 (Europe/Istanbul).

This file is an operational handoff for agents working on this repository. Read it before running commands locally, transferring files, or submitting cluster jobs. Paths and resource availability can change, so re-run the lightweight checks below before expensive or destructive actions.

## At a glance

| Context | Host / endpoint | Main purpose | GPU / scheduler | Repository path |
|---|---|---|---|---|
| Local workspace | `audiolab-server1` | Development, analysis, lightweight tests, storing synced results/checkpoints | One NVIDIA RTX 4090, 24 GB | `/home/emre/Projects/AudioLLM/LLM-Depression` |
| BSC transfer endpoint | `ozu647717@transfer1.bsc.es` | Moving data to/from the shared GPFS filesystem and read-only inspection | Do not run training directly here; `sinfo` currently shows storage partitions only | `/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression` |
| MN5 scheduler login | `ozu647717@alogin1.bsc.es` | `sbatch`, `squeue`, `sacct`, and job/log inspection | Submit jobs here; do not run training directly on the login node | Same GPFS repository path |
| MareNostrum 5 Slurm jobs | Allocated compute nodes such as `as01...` / `as02...` | GPU training and evaluation | Existing logs show NVIDIA H100 64/65 GB nodes; jobs use Slurm account `etur92` and QoS `acc_ehpc` | Same GPFS repository path |

The local and BSC repository trees intentionally have the same relative structure. They are separate copies; a change in one does not appear in the other until it is synchronized.

## 1. Local environment

Verified host details:

- Hostname: `audiolab-server1`
- User: `emre`
- OS/kernel family: Linux x86-64
- GPU: NVIDIA GeForce RTX 4090, 24,564 MiB reported
- Project: `/home/emre/Projects/AudioLLM/LLM-Depression`
- Backup/data root: `/media/emre/Backup/AudioLLM`
- Local Conda installation: `/home/emre/miniconda3`
- Known Conda environments: `llmdep4090`, `qwen3asr`, and `secap`

The shell may start in Conda `base`; do not assume it is the correct ML environment. Select an environment according to the task and verify imports before model execution.

Useful local checks:

```bash
hostname
nvidia-smi
conda env list
git status --short
df -h /home/emre/Projects/AudioLLM/LLM-Depression
```

Local usage guidance:

- Prefer local execution for code edits, parsing logs, summarizing results, unit/sanity tests, and small experiments.
- The RTX 4090 can run suitable single-GPU work, but do not assume a four-GPU/H100 training configuration will fit or behave identically.
- Config defaults often contain BSC absolute paths. Set local dataset/model paths explicitly when running locally.
- A local Qwen2 text model was observed at `/media/emre/Backup/AudioLLM/models/Qwen2-7B-Instruct`.
- Do not assume the Qwen2-Audio base model exists locally; verify before evaluation.
- `/media/emre/Backup/AudioLLM/qwen_mn5_rebuilt` is a copied cluster environment, not a portable authoritative environment. Its absolute interpreter references are broken locally. See `docs/ENVIRONMENT_NOTES.md`.

## 2. BSC transfer endpoint and shared storage

Known working non-interactive endpoint:

```text
ozu647717@transfer1.bsc.es
```

Known shared paths:

```text
/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression
/gpfs/projects/etur92/ozu647717/AudioLLM/Datasets
/gpfs/projects/etur92/ozu647717/models/Qwen2-Audio-7B-Instruct
/gpfs/projects/etur92/ozu647717/models/Qwen2-7B-Instruct
/gpfs/projects/etur92/ozu647717/venvs/qwen_mn5_rebuilt
```

`transfer1` is the data-transfer endpoint. It currently has `ssh`, `rsync`, `sbatch`, and `squeue` available, but availability of those commands does **not** prove that it is the correct place to submit a compute job. On 2026-07-21, `sinfo` there exposed only `projects`, `tapes`, and `archive`, not the H100 compute resources seen in training logs.

Therefore:

1. Use `transfer1` freely for authorized read-only inspection and file transfer.
2. Never run Python training directly on `transfer1`.
3. Use the currently designated MN5 scheduler login,
   `ozu647717@alogin1.bsc.es`, for `sbatch`, `squeue`, and `sacct`.
4. `sinfo` on `alogin1` returned an access/permission error on 2026-07-25,
   although `sbatch`, `squeue`, and `sacct` were present and prior jobs had
   completed successfully. Treat a failed `sinfo` as a reason to verify with a
   smoke job, not as permission to submit through `transfer1`.

Safe connectivity checks:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=15 \
  ozu647717@transfer1.bsc.es 'hostname; command -v rsync; command -v sbatch; sinfo'

ssh -o BatchMode=yes -o ConnectTimeout=15 \
  ozu647717@alogin1.bsc.es \
  'hostname; command -v sbatch; command -v squeue; command -v sacct'
```

Do not print private keys, tokens, credential files, or unrelated shell history.

## 3. MareNostrum 5 compute jobs

Training and evaluation must be scheduled through Slurm. Existing repository scripts currently declare:

Submit and monitor from:

```text
ozu647717@alogin1.bsc.es
```

Use `transfer1` for rsync. Both endpoints see the same GPFS project tree.

| Workload | Script | Default resources |
|---|---|---|
| Training | `scripts/run_train_slurm.sh` | 1 node, 4 tasks, 4 GPUs, 20 CPUs/task, 72 hours |
| Evaluation | `scripts/run_eval_slurm.sh` | 1 node, 1 task, 1 GPU, 20 CPUs/task, 24 hours |
| CV orchestration | `scripts/run_chain_submit_slurm.sh` and dataset wrappers | Small CPU submission job that chains train/eval jobs |
| Turkish CV | `scripts/run_turkish_5fold.sh` | Submission/orchestration job; folds are chained |

Common Slurm settings in these scripts:

```text
account: etur92
QoS: acc_ehpc
project working directory: /gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression
```

Compute runtime initialization:

```bash
module purge
module load bsc/1.0
module load miniforge/24.3.0-0
source /gpfs/projects/etur92/ozu647717/venvs/qwen_mn5_rebuilt/bin/activate
```

For the raw hidden-state classifier/Optuna environment, also export:

```bash
export PYTHONPATH="$PWD/.deps/qwen_hidden:$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

That project-local dependency path was required to expose Optuna 4.4.0,
XGBoost 2.1.4, and scikit-learn 1.7.0 on 2026-07-25.

Authoritative package versions recorded for the cluster runtime include:

```text
Python 3.10.14
torch 2.3.0+cu121
transformers 4.55.0
accelerate 1.8.1
peft 0.17.0
```

Treat the Slurm headers as defaults, not timeless facts. Before submission, inspect the selected scripts and verify account, QoS/partition, GPU count, wall time, dependency chain, config, folds, and run name.

## 4. Synchronization directions

### Local to BSC

The repository-provided route is:

```bash
cd /home/emre/Projects/AudioLLM/LLM-Depression
bash scripts/sync_to_cluster.sh
```

This captures local Git provenance, excludes `.git/`, respects `.gitignore`, and writes to:

```text
ozu647717@transfer1.bsc.es:/gpfs/projects/etur92/ozu647717/AudioLLM
```

Before using it, always:

- inspect `git status --short`;
- inspect local untracked/user changes;
- ensure the BSC copy does not contain newer work that would be overwritten;
- use an `rsync --dry-run` equivalent when scope is uncertain;
- never add `--delete` unless the user explicitly requests deletion and the effect has been reviewed.

### BSC to local: routine results sync

The user's usual storage-saving command is:

```bash
rsync -avz \
  --exclude='best_model/' \
  --exclude='last_model/' \
  ozu647717@transfer1.bsc.es:/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression/output_model/ \
  /home/emre/Projects/AudioLLM/LLM-Depression/output_model/
```

Logs can be synchronized separately:

```bash
rsync -avz \
  ozu647717@transfer1.bsc.es:/gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression/logs/ \
  /home/emre/Projects/AudioLLM/LLM-Depression/logs/
```

### Selective checkpoint retrieval

Do not sync every checkpoint by default. Identify the exact run and fold directories from logs and `final_summary_active.csv`, check remote size with `du`, then retrieve only the required `best_model/` directories. The evaluated model is normally `best_model`; `last_model` is a different checkpoint and should not be substituted silently.

On 2026-07-20, the 33 `best_model` directories aligned with `depression_results_table_no_emo.csv` were selectively downloaded. They occupy approximately 5.85 GB in total and are LoRA adapter checkpoints, not copies of the full 7B base models.

## 5. Checkpoints and model loading

The checkpoint layout is generally:

```text
output_model/<modality>/<dataset>/<run_name>/fold_<n>/
├── best_model/
│   ├── adapter_model.safetensors
│   ├── adapter_config.json
│   └── ...tokenizer/processor files...
├── last_model/
├── logs/
└── run_config.yaml
```

Important implications:

- `adapter_model.safetensors` is a LoRA adapter (roughly 160 MB in the aligned runs), not a standalone Qwen model.
- Loading/evaluation also requires the correct base model.
- Audio and audio+text runs normally use Qwen2-Audio-7B-Instruct.
- Text-only runs normally use Qwen2-7B-Instruct.
- `run_config.yaml` is the best record of the resolved config, overrides, manifest hash, split hash, fold, selection protocol, and base-model path.
- A similarly named checkpoint is not sufficient evidence of provenance. Match config, run name, fold, backend, aggregation level, and metrics.

## 6. Standard job workflow for an agent

For a requested training run:

1. Inspect `git status` locally and preserve unrelated user changes.
2. Read the chosen YAML and the wrapper/submission scripts completely.
3. Run appropriate lightweight validation locally.
4. Show or internally verify the exact config(s), run name(s), folds, number of jobs, GPUs/job, wall time, and checkpoint policy.
5. Synchronize code only if authorized and needed.
6. Verify the remote code/config and runtime environment.
7. Verify the host is connected to the intended compute scheduler.
8. Submit through the repository wrapper, not by running `src/train.py` on a login/transfer host.
9. Record returned job IDs.
10. Monitor with `squeue`/`sacct` and inspect repository logs.
11. Sync result summaries and logs back locally.
12. Retrieve `best_model` checkpoints only when requested or clearly required.

Typical single-fold wrapper pattern, to be run on the correct BSC submission host after verification:

```bash
cd /gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression
CONFIG="$PWD/configs/main/<config>.yaml" \
RUN_NAME='<unique-run-name>' \
FOLD=0 \
bash scripts/submit_train_and_eval.sh
```

Do not reuse an existing run name unless continuation/overwrite behavior has been explicitly checked.

## 7. Authorization and safety boundary

Technical SSH access is not blanket permission to mutate the cluster.

An agent may perform relevant read-only inspection when needed. Explicit user authorization is required before actions such as:

- submitting costly training or evaluation jobs;
- cancelling jobs;
- overwriting remote code, configs, manifests, or checkpoints;
- deleting local or remote data;
- changing permissions, allocations, environment packages, or shared resources.

Never:

- run GPU training on a transfer/login node;
- expose credentials or private keys;
- bypass MFA, quotas, scheduler policy, or access controls;
- use destructive Git commands to discard user work;
- assume an old log's filename reflects its resolved configuration—inspect `run_config.yaml` and resolved-config log blocks.

## 8. Fast orientation checklist

A new agent should begin with:

```bash
cd /home/emre/Projects/AudioLLM/LLM-Depression
git status --short
hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
sed -n '1,240p' docs/DEVICES.md
sed -n '1,220p' docs/ENVIRONMENT_NOTES.md
```

If BSC access is needed, first perform a read-only connectivity/scheduler check. If training is requested, do not submit until the exact workload and scheduler target are verified.

For the complete operational lifecycle—including selective rsync, smoke jobs,
monitoring through terminal Slurm accounting, result synchronization, audits,
and Git handoff—read `docs/MN5_AGENT_EXECUTION_RUNBOOK.md`.
