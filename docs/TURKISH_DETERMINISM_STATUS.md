# Turkish Determinism & Provenance — Status

Working log for the T21 reproducibility effort. Companion to
`TURKISH_RESULTS_MISMATCH.md` (the investigation) and
`TURKISH_DETERMINISM_PLAN.md` (the tiered plan).

Last updated: 2026-06-29 (replicas in — determinism confirmed).

---

## TL;DR

The Turkish audio-only T21 rerun gap is **unseeded GPU training
nondeterminism**, not transcript/manifest content (proven by code, see
Conclusion below). We have:

- **Tier 0 (provenance): DONE** — every run now records its exact code + env.
- **Tier 1 (determinism flags): DONE** — best-effort reproducibility wired in.
- **Replicas in (2026-06-29): CONFIRMED.** Two T21 Tier-1 runs landed within
  **Δ0.015 pos-F1 / Δ0.000 macro-F1** on audio-only, vs the old **Δ0.131** gap
  (0.688→0.557). Determinism worked at the CV-aggregate level; the mismatch is
  explained as unseeded nondeterminism, now controlled. See "Results" below.

---

## Conclusion of the investigation (why it's nondeterminism)

For an audio-only run, transcript content cannot reach training:

- examples are sorted by `sample_id`, not manifest order (`src/data/runtime.py:593`)
- `transcript=""` when `use_text=false` (`src/data/runtime.py:268`)
- the weighted sampler is seeded from labels + order only (`src/train.py:405-420`)
- audio is cropped deterministically, no augmentation configured

So once the `cy2-1-9` row fix restored the row set, the audio-only pipeline is
byte-identical between old and repaired runs. The residual T21 drift is GPU
compute nondeterminism, amplified by epoch selection + early stopping + subject
likelihood aggregation, and worst at T21 because it sits at ~50/50 balance with a
near-chance audio signal (AUROC 0.57, see `TURKISH_DATASET_STATS.md §5`). T17 is
stable only because its 69% positive base rate makes "predict majority" a stable
attractor — its high F1 ≈ the all-positive baseline, not real discrimination.

---

## What has been done

### Tier 0 — Provenance (DONE, verified on cluster)

| File | Role |
| --- | --- |
| `scripts/capture_provenance.sh` | local: snapshots `git_commit` + branch + dirty list + `git diff HEAD` into `.provenance/` |
| `scripts/sync_to_cluster.sh` | local: runs capture, then rsync (force-includes `.provenance/`, excludes `.git/`) |
| `scripts/run_train_slurm.sh` | cluster: copies shipped `.provenance/` + live `pip freeze` into `logs/.../provenance-<jobid>/` per run |
| `.gitignore` | ignores `.provenance/` so it never dirties the tree |

Verified on BSC: `.provenance/` transferred (commit `b3c8493`, 231 KB
`uncommitted.patch`); dry-run stamping produced `pip_freeze.txt` with **99
packages on the cluster vs 88 locally** — exactly the env gap that was previously
unrecorded (old hypothesis #2). Closes the "no run records its code/env" gap.

Reproduce any run: `git checkout <git_commit.txt>` then
`git apply uncommitted.patch`.

### Tier 1 — Determinism flags (DONE, not yet run on GPU)

| File | Change |
| --- | --- |
| `src/utils.py` | `set_seed(seed, deterministic=True)` sets `cudnn.deterministic=True`, `cudnn.benchmark=False`, `use_deterministic_algorithms(True, warn_only=True)` |
| `src/train.py` | passes `deterministic=config.training.deterministic` (**default True**) |
| `scripts/run_train_slurm.sh` | exports `CUBLAS_WORKSPACE_CONFIG=:4096:8` + `PYTHONHASHSEED=0` before `torchrun`, logs them |

- `warn_only=True` is mandatory: Qwen2-Audio's encoder has ops with no
  deterministic CUDA kernel. Expect one-time `UserWarning: ... does not have a
  deterministic implementation` lines in the log — harmless. A *crash* on a
  determinism error would mean an op slipped past `warn_only`.
- ON by default for **all** datasets. Opt out with
  `training.deterministic: false` or `--set training.deterministic=false`.
- **Best-effort, not bitwise** — bf16 precision + the `warn_only` ops still
  wobble. Shrinks drift; won't perfectly freeze T21.

---

## Results (2026-06-29) — VERDICT: CLOSE, determinism worked

Two T21 Tier-1 submissions ran (each = 3 configs × 5 folds), both with the
determinism env confirmed live in the logs (`CUBLAS_WORKSPACE_CONFIG=:4096:8
PYTHONHASHSEED=0` + the expected one harmless Flash-Attention `warn_only` warning):

```bash
# Run 1 — canonical
CONFIGS=".../turkish_audio_only_t21.yaml .../turkish_text_only_t21.yaml .../turkish_audio_text_t21.yaml" \
  RUN_NAME_PREFIX=train_val_t21_rep_transcript sbatch --export=ALL scripts/run_turkish_5fold.sh
# Run 2 — replica (same command, different prefix)
  RUN_NAME_PREFIX=train_val_t21_rep_transcript_rep2 sbatch --export=ALL scripts/run_turkish_5fold.sh
```

**Decision rule:** compare audio-only T21 between the two replicas.

| Audio-only T21 | pos-F1 (mean±std) | macro-F1 | AUROC |
| --- | --- | --- | --- |
| rep1 (`_rep_transcript`)      | 0.655 ± 0.054 | 0.629 | 0.638 |
| rep2 (`_rep_transcript_rep2`) | 0.640 ± 0.086 | 0.629 | 0.667 |
| **Δ rep1↔rep2**               | **0.015**     | **0.000** | 0.029 |
| *cy219fixed (pre-Tier-1)*     | *0.557 ± 0.186* | *0.560* | *0.623* |

Old unexplained drift was **0.688 → 0.557 (Δ0.131)**; the two Tier-1 replicas land
within **Δ0.015 pos-F1 / Δ0.000 macro-F1** — an ~9× reduction in run-to-run
spread, and the cy219fixed std (0.186) is ~4× the replicas'. **→ "Close" branch:
determinism worked, mismatch explained, proceed to write up.** Other modalities
corroborate (text-only Δ0.021 pos-F1; audio+text Δ0.015, rep2 std just 0.035).

**Caveat — stable at the aggregate, NOT bitwise.** Selected-epoch means diverge
(audio-only rep1 3.8 vs rep2 5.8) and individual folds still swing (fold_3
0.643→0.522) — the bf16 + `warn_only` Flash-Attention op, exactly as predicted.
This is *why* Tier 2 (AUROC + macro-F1 headline, mean ± std over seeds) remains
the right reporting story; these two replicas are seeds 1–2.

### Verify each job stamped provenance

```bash
ls logs/slurm_train/turkish/provenance-<JOBID>/
grep -E "Code provenance|Determinism env" logs/slurm_train/turkish/train-<JOBID>-*.log
```

---

## Known footgun — manifest collision

All three T21 configs share `outputs/manifests_t21/` and `outputs/splits_t21/`,
and `build_manifest.py` always writes `turkish_manifest.jsonl` there regardless
of transcript file. So:

- **Safe:** the two replica runs above (identical transcript → identical
  manifest content).
- **NOT safe concurrently:** a repaired-transcript run and an
  original-transcript run (`--set transcript_file=whisper_transcripts.jsonl`) —
  they clobber the same manifest, and queued jobs read whichever built last. Run
  one transcript variant at a time, or give the other its own
  `manifest_dir`/`split_dir`.

(For audio-only, original vs repaired transcript is a no-op anyway — that
comparison only matters for text-only / audio+text.)

---

## Not done yet / next steps

- [x] **Commit** the Tier 0 + Tier 1 work (committed in `2170150`).
- [x] Run the two T21 replicas and apply the decision rule above (CLOSE — see
      Results; determinism confirmed at the aggregate level).
- [ ] **Run T17 deterministically (2-seed pair).** The existing T17 runs are from
      2026-06-26 — *before* the determinism commit (`2170150`, 2026-06-29) — and
      have no replica, so T17 has never been through the deterministic pipeline.
      Re-run with prefixes `train_val_t17_det` / `train_val_t17_det_rep2` so the
      definitive results table has both thresholds on the same footing. T21 is
      already done (reuse the 06-29 rep1/rep2). T17 cells in
      `depression_results_table_no_emo.csv` are TODO until this lands.
- [ ] **Tier 2 reporting** (the real fix for trustworthy numbers): AUROC +
      macro-F1 as headline, add an all-positive-baseline column, report N-seed
      mean ± std. See `TURKISH_DETERMINISM_PLAN.md §Tier 2`.
- [ ] Optional: stamp provenance in `scripts/run_eval_slurm.sh` too (only
      training-side is wired now).
- [ ] Optional: separate-output-dir recipe if the original-transcript
      text-modality comparison is wanted.
