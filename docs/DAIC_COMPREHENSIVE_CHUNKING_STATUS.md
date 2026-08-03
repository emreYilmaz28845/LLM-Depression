# DAIC Comprehensive Chunking: Implementation Status and MN5 Operator Handoff

Last updated: 2026-08-04 (Europe/Istanbul)

## Stop point and current state

The comprehensive DAIC chunking implementation is present in the local workspace. No MN5 host was contacted, no files were transferred, and no Slurm job was submitted. This is the intentional stop point requested by the user.

The next agent owns only remote preflight, smoke submission, and continuous monitoring. It must not redesign the protocol or silently weaken an audit.

Important repository state:

- The implementation is not committed or pushed.
- `docs/SYMMETRIC_MERGED_CURRENT_STATUS.md` was already modified by the user and is unrelated. Preserve it exactly; do not include it in an implementation commit unless the user separately asks.
- Use a new run ID. Never reuse `daic_k_prod_20260730_204c550` or any earlier output directory.
- The official test partition is forbidden before a hash-recorded final winner authorization.

## What was implemented

- Six core protocols: `jr4`, `jt4`, `ja4`, `ir4`, `ian`, and `iaf`.
- Fixed five-fold development CV with split seed `1337` and model seeds `1337`, `2027`, `3407`.
- Deterministic random/rotary joint schedules and minimum-cover joint schedules with per-epoch membership, weight, ordering, and audio-exposure audit rows.
- Independent rotary/all schedules with raw and mean-one effective weights; `iaf` deliberately uses equal row weights.
- Deterministic shuffled subject-block order.
- Joint `fixed4`, minimum-cover, and exactly-15-bundle views. Fixed-15 gives every 10-chunk subject six occurrences per chunk and every 15-chunk subject four.
- Independent all-chunk, deterministic matched-10, and 1,000 cached matched-10 resamples.
- Mean, median, 10% trimmed mean, majority vote with margin tie-break, and maximum-margin aggregation.
- Evaluation materializes one subject row plus metrics/prediction artifacts for every configured secondary aggregation.
- Exact two-pass subject mean-margin MIL. It uses evaluation-identical mean token log-probability candidate margins and performs no optimizer update until all chunks of a subject have backpropagated.
- Matrix expansion for `smoke`, `core`, `focused`, and `final`; config/selection/implementation hashes; collision-safe matrix creation; staged Slurm arrays; distinct four-GPU standard training and one-GPU MIL array submission.
- Comprehensive structural/schedule/Slurm audit helpers, paired bootstrap/McNemar/Holm utilities, and reproducible report/CSV entrypoints.

Primary files to read before operating:

```text
docs/DAIC_COMPREHENSIVE_CHUNKING_IMPLEMENTATION_EXPERIMENT_PLAN.md
docs/DEVICES.md
docs/MN5_AGENT_EXECUTION_RUNBOOK.md
configs/experiments/daic_chunking/comprehensive_matrix.yaml
scripts/build_daic_comprehensive_matrix.py
scripts/submit_daic_comprehensive_matrix.sh
scripts/run_daic_comprehensive_array_slurm.sh
scripts/run_daic_comprehensive_task.py
scripts/audit_daic_comprehensive.py
scripts/collect_daic_slurm_accounting.py
scripts/select_daic_comprehensive_protocol.py
scripts/authorize_daic_comprehensive_test.py
scripts/collect_daic_comprehensive_oof.py
```

## Local validation already completed

The implementation was syntax-checked and the original DAIC chunking tests passed. The complete focused suite must be rerun after the final workspace changes before transfer:

```bash
cd /home/emre/Projects/AudioLLM/LLM-Depression
/home/emre/miniconda3/envs/llmdep4090/bin/python -m py_compile \
  src/aggregate.py src/daic_chunking.py src/daic_mil.py \
  src/daic_comprehensive_audit.py src/daic_statistics.py \
  src/data/runtime.py src/data/build_manifest.py src/evaluate.py \
  src/features/extract_qwen_hidden.py src/train.py \
  scripts/build_daic_comprehensive_matrix.py \
  scripts/run_daic_comprehensive_task.py \
  scripts/audit_daic_comprehensive.py scripts/report_daic_comprehensive.py \
  scripts/select_daic_comprehensive_protocol.py \
  scripts/authorize_daic_comprehensive_test.py \
  scripts/collect_daic_comprehensive_oof.py
bash -n \
  scripts/submit_daic_comprehensive_matrix.sh \
  scripts/run_daic_comprehensive_array_slurm.sh \
  scripts/run_eval_slurm.sh \
  scripts/run_daic_chunking_hidden_slurm.sh \
  scripts/run_daic_chunking_classical_slurm.sh
/home/emre/miniconda3/envs/llmdep4090/bin/python -m pytest -q \
  tests/test_daic_chunking.py tests/test_daic_comprehensive_chunking.py
```

Expected matrix contracts:

- Smoke: seven training cells (six core plus MIL), with evaluation, hidden extraction, and one bundled classical task per cell.
- Core: 90 training + 90 evaluation + 90 hidden + 90 bundled classical = 360 tasks.
- Core training cells: 6 protocols x 5 folds x 3 model seeds.
- Fold membership is always controlled by `split.seed=1337`; model seed never changes it.

## Files to transfer

First inspect `git status --short` and make a focused commit if the user authorizes committing. Do not push unless explicitly authorized. Transfer only these implementation paths plus `.provenance`; never use `--delete`:

```text
src/aggregate.py
src/daic_chunking.py
src/daic_mil.py
src/daic_comprehensive_audit.py
src/daic_statistics.py
src/data/runtime.py
src/data/build_manifest.py
src/evaluate.py
src/features/extract_qwen_hidden.py
src/train.py
configs/experiments/daic_chunking/comprehensive_matrix.yaml
scripts/audit_daic_comprehensive.py
scripts/build_daic_comprehensive_matrix.py
scripts/collect_daic_slurm_accounting.py
scripts/report_daic_comprehensive.py
scripts/select_daic_comprehensive_protocol.py
scripts/authorize_daic_comprehensive_test.py
scripts/collect_daic_comprehensive_oof.py
scripts/run_daic_comprehensive_array_slurm.sh
scripts/run_daic_comprehensive_task.py
scripts/submit_daic_comprehensive_matrix.sh
scripts/run_train_slurm.sh
scripts/run_eval_slurm.sh
scripts/run_daic_chunking_hidden_slurm.sh
scripts/run_daic_chunking_classical_slurm.sh
tests/test_daic_comprehensive_chunking.py
docs/DAIC_COMPREHENSIVE_CHUNKING_STATUS.md
```

Use `transfer1` for transfer and a scheduler login for Slurm:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=15 ozu647717@transfer1.bsc.es \
  'hostname; command -v rsync; test -d /gpfs/projects/etur92/ozu647717/AudioLLM/LLM-Depression'
ssh -o BatchMode=yes -o ConnectTimeout=15 ozu647717@alogin2.bsc.es \
  'hostname; command -v sbatch; command -v squeue; command -v sacct'
```

If `alogin2` is unavailable, try `alogin1`. Never submit from `transfer1`, and never run Python training on a login node.

Perform `rsync -avhn --relative` first, review every destination, then repeat without `-n`. Compare local and remote `sha256sum` for every transferred source/config/script.

## Remote smoke preparation

On the reachable scheduler login:

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
python -c "import torch,transformers,accelerate,peft,xgboost,sklearn; print(torch.__version__,transformers.__version__,accelerate.__version__,peft.__version__,xgboost.__version__,sklearn.__version__)"
bash -n scripts/submit_daic_comprehensive_matrix.sh scripts/run_daic_comprehensive_array_slurm.sh
df -h "$PROJECT_ROOT"
```

Choose a unique smoke ID, for example `daic_comp_smoke_20260804_<short-hash>`. Do not copy this literal ID without replacing `<short-hash>`.

```bash
RUN_ID=daic_comp_smoke_20260804_<short-hash>
export RUN_ID

python scripts/build_daic_comprehensive_matrix.py \
  --run-id "$RUN_ID" --stage smoke

python scripts/audit_daic_comprehensive.py \
  --matrix "outputs/daic_chunking_comprehensive/$RUN_ID/matrix_smoke.json"
```

Build the shared manifest/folds once before arrays start. The task workers intentionally use `SKIP_MANIFEST_BUILD=1` to avoid 7 concurrent writers:

```bash
python src/data/build_manifest.py \
  --config configs/experiments/daic_chunking/joint_random_k4.yaml \
  --set split.mode=cv \
  --set split.cv_protocol=train_val_test \
  --set split.outer_folds=5 \
  --set split.seed=1337 \
  --set "output_dirs.manifest_dir=$PROJECT_ROOT/outputs/daic_chunking_comprehensive/$RUN_ID/shared/manifests" \
  --set "output_dirs.split_dir=$PROJECT_ROOT/outputs/daic_chunking_comprehensive/$RUN_ID/shared/splits"
```

Verify the metadata, manifest counts, folds, and test exclusion before submission:

```bash
test -f "outputs/daic_chunking_comprehensive/$RUN_ID/shared/splits/daic_manifest_metadata.json"
python - <<'PY'
import json, os
from pathlib import Path
r=Path('outputs/daic_chunking_comprehensive')/os.environ['RUN_ID']/ 'shared/splits'
m=json.loads((r/'daic_manifest_metadata.json').read_text())
print(json.dumps({k:m.get(k) for k in ('manifest_hash','fold_hash','manifest_path','folds_path')}, indent=2))
PY
```

Dry-run first and reconcile the seven training cells and all dependencies:

```bash
RUN_ID="$RUN_ID" STAGE=smoke DRY_RUN=1 MAX_CONCURRENT_TRAIN=4 \
  bash scripts/submit_daic_comprehensive_matrix.sh 2>&1 | tee \
  "outputs/daic_chunking_comprehensive/$RUN_ID/submission_smoke_dry_run.txt"
```

Only after the dry run is correct, start smoke:

```bash
RUN_ID="$RUN_ID" STAGE=smoke DRY_RUN=0 MAX_CONCURRENT_TRAIN=4 \
  bash scripts/submit_daic_comprehensive_matrix.sh | tee \
  "outputs/daic_chunking_comprehensive/$RUN_ID/submission_smoke_stdout.txt"
```

The script writes `submission_smoke.json`. Record every train, evaluation, hidden, and classical array job ID immediately.

## Monitoring loop: do not stop at submission

Poll every 5-10 minutes while jobs are active. Do not use a blocking sleep longer than 60 seconds inside an agent tool call. The human-facing progress update should include counts by state, active array indices, the latest completed cell, and any error found.

```bash
RUN_ID=<the exact smoke ID>
SUB="outputs/daic_chunking_comprehensive/$RUN_ID/submission_smoke.json"
python - "$SUB" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]))
print(','.join(j for group in p['arrays'].values() for j in group['job_ids']))
PY
```

Use the printed comma-separated IDs as `JOB_IDS`:

```bash
squeue -j "$JOB_IDS" -o '%.22i %.12T %.10M %.6D %R'
sacct -j "$JOB_IDS" -X --format=JobIDRaw,JobName%45,State,ExitCode,Elapsed,AllocCPUS,AllocTRES%45 -P
```

An empty `squeue` is not success. Success requires every authoritative top-level array and every array element to be terminal `COMPLETED` with `ExitCode=0:0`.

Continuously scan only this run's logs:

```bash
LOG_ROOT="logs/daic_chunking_comprehensive/$RUN_ID/smoke"
find "$LOG_ROOT" -type f \( -name '*.out' -o -name '*.err' -o -name '*.log' \) -print0 | \
  xargs -0 -r grep -Ein 'traceback|out of memory|cuda.*error|killed|nan|inf|invalid value|no space|time limit|cancelled|failed' || true
```

For every match, inspect context and reconcile it with `sacct`. Do not treat harmless words in config dumps as failures. For a real failure, record job/array ID, state, exit code, node, elapsed time, last 100 log lines, partial artifacts, and config hash. Repair only the demonstrated cause, preserve the failed attempt, rebuild the matrix under a new smoke run ID if code/config hashes change, and rerun only the affected scope when safe.

Smoke acceptance requires:

- all six schedule paths completed for one epoch;
- one MIL update for at least one subject from each class;
- `fixed4`, `mincover4`, `fixed15`, `all`, `matched10_even`, and cached `matched10_resampled` artifacts;
- hidden caches and both classical heads;
- best-model, prediction, schedule, split, config, and provenance artifacts;
- no unresolved traceback/OOM/killed/invalid-value log event;
- remote audit passes with zero failures;
- measured peak memory and wall time fit production resource requests.

Run the smoke audit after terminal accounting has been saved:

```bash
python scripts/collect_daic_slurm_accounting.py \
  --matrix "outputs/daic_chunking_comprehensive/$RUN_ID/matrix_smoke.json" \
  --submission "outputs/daic_chunking_comprehensive/$RUN_ID/submission_smoke.json" \
  --output "outputs/daic_chunking_comprehensive/$RUN_ID/slurm_accounting_smoke.jsonl"
python scripts/audit_daic_comprehensive.py \
  --matrix "outputs/daic_chunking_comprehensive/$RUN_ID/matrix_smoke.json" \
  --artifact-root "$PROJECT_ROOT/output_model/daic_chunking_comprehensive/$RUN_ID/smoke" \
  --slurm-accounting "outputs/daic_chunking_comprehensive/$RUN_ID/slurm_accounting_smoke.jsonl" \
  --require-artifacts
```

If this audit reports missing authoritative Slurm accounting, save and map `sacct` rows before calling it final. Never delete a failed audit.

## Production core: only after smoke passes

Use a new production ID. Build and audit the 360-task matrix, prepare the shared manifest exactly as above under the new ID, and dry-run:

```bash
RUN_ID=daic_comp_core_20260804_<short-hash>
export RUN_ID
python scripts/build_daic_comprehensive_matrix.py --run-id "$RUN_ID" --stage core
python scripts/audit_daic_comprehensive.py \
  --matrix "outputs/daic_chunking_comprehensive/$RUN_ID/matrix_core.json"
RUN_ID="$RUN_ID" STAGE=core DRY_RUN=1 MAX_CONCURRENT_TRAIN=4 \
  bash scripts/submit_daic_comprehensive_matrix.sh
```

The dry run must show exactly 90 training indices and downstream arrays of 90 each. Submit only after reconciling all 6 x 5 x 3 cells. The default training concurrency is four four-H100 jobs. Monitor until every task is terminal; then run the core audit and generate the selection artifact. Do not create focused or final matrices by hand.

Focused requires a JSON selection artifact containing `leading_joint` and `leading_independent`. Final requires `winner`, `final_epoch_count`, the locked `aggregation_view`, and an embedded `winner_protocol` if the winner came from focused follow-up. Matrix commands are:

```bash
python scripts/build_daic_comprehensive_matrix.py \
  --run-id "$RUN_ID" --stage focused --selection path/to/core_selection.json
python scripts/build_daic_comprehensive_matrix.py \
  --run-id "$RUN_ID" --stage final --selection path/to/final_selection.json
```

Before final matrix creation, write `FINAL_TEST_AUTHORIZED.json` containing the winner ID, selection-artifact SHA-256, timestamp, and implementation/config hashes. No test artifact may exist before this marker.

## What the next agent must report back

For each stage, report:

- exact run ID and implementation commit/provenance hash;
- all array job IDs and counts by terminal state;
- retries and the original failed job IDs;
- peak memory, elapsed time, and storage used;
- audit path and pass/fail count;
- output, log, matrix, submission-manifest, and selection-artifact paths;
- whether anything was committed, pushed, transferred, or retrieved;
- any protocol limitation. Never describe non-significance as equivalence.

Do not retrieve `best_model/`, `last_model/`, or large hidden caches by default. Use `transfer1`, dry-run rsync first, and retrieve configs, raw predictions, metrics, schedules, audits, reports, Slurm accounting, and relevant logs.
