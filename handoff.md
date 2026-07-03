# Reproducibility Handoff

Date: 2026-07-03. Repository:
`/home/emre/Projects/AudioLLM/LLM-Depression`.

Paper:
`/home/emre/Projects/AudioLLM/Papers/DepresInstruct.pdf`.

This project is trying to understand why local results do not match the
DepresInstruct paper. Full model training is run on the BSC cluster, not locally.
Do not start expensive local training.

## Key Files

- Investigation report: `docs/REPRODUCIBILITY_INVESTIGATION.md`
- Experiment results report: `docs/REPRODUCIBILITY_EXPERIMENT_RESULTS.md`
- Original no-emotion result table:
  `depression_results_table_no_emo.csv`
- New diagnostic configs already used:
  - `configs/experiments/daic_audio_text_valeval_tf.yaml`
  - `configs/experiments/daic_audio_only_valeval_tf.yaml`
  - `configs/experiments/daic_text_only_valeval_tf.yaml`
  - `configs/experiments/cmdc_audio_text_20ep_tf.yaml`
  - `configs/experiments/turkish_t17_audio_only_selmacro_tf.yaml`
  - `configs/experiments/turkish_t17_text_only_selmacro_tf.yaml`
  - `configs/experiments/turkish_t17_audio_text_selmacro_tf.yaml`

There are also currently untracked configs in `configs/experiments/` for DAIC
valeval selmacro and Turkish T21 selmacro. Treat them as user/other-agent work;
inspect before editing.

## Verified Experiment Results

### DAIC Valeval

Source: standalone eval logs under `logs/slurm_eval/daic/`.

These runs changed DAIC final evaluation to the paper-like AVEC dev partition.
Support sanity check passed: `12 dep / 23 non`.

| Modality | ACC | Pos F1 | Precision | Recall | Macro F1 | Confusion |
|---|---:|---:|---:|---:|---:|---|
| Audio + Text | 0.686 | 0.353 | 0.600 | 0.250 | 0.573 | `[[21, 2], [9, 3]]` |
| Audio Only | 0.343 | 0.511 | 0.343 | 1.000 | 0.255 | `[[0, 23], [0, 12]]` |
| Text Only | 0.714 | 0.444 | 0.667 | 0.333 | 0.626 | `[[21, 2], [8, 4]]` |

Conclusion: DAIC was previously compared on the wrong final partition, but
partition alignment alone does not recover the paper. Remaining mismatch is
likely chunk-level/aggregation/training-protocol related.

### CMDC 20ep

Source:
`output_model/experiments/cmdc_20ep/audio_text/cmdc_20ep_cmdc_audio_text_20ep_tf/final_summary_active.csv`.

| Summary | ACC | Pos F1 | Precision | Recall | Macro F1 | Confusion |
|---|---:|---:|---:|---:|---:|---|
| Mean over folds | 0.951 | 0.938 | 0.933 | 0.960 | 0.948 | - |
| Pooled | 0.949 | 0.926 | 0.893 | 0.962 | 0.943 | `[[49, 3], [1, 25]]` |

Previous CMDC audio+text row was about `ACC=0.908`, `Pos F1=0.896`.

Important: the run was configured for `num_train_epochs: 20`, but early stopping
stopped at epochs 4-6. Best epochs were 1-3, and inner-val positive F1 saturated
at 1.0 in every fold. So the result confirms that training beyond 3 epochs helps,
but it does not prove full 20-epoch training is needed.

### Turkish T17 Selmacro

Source:
`output_model/experiments/turkish_t17_selmacro/*/*/final_summary_active.csv`.

| Modality | ACC | Pos F1 | Precision | Recall | Macro F1 | Support | Confusion |
|---|---:|---:|---:|---:|---:|---|---|
| Audio Only | 0.733 | 0.828 | 0.748 | 0.928 | 0.618 | 37 non / 83 dep | `[[11, 26], [6, 77]]` |
| Text Only | 0.733 | 0.805 | 0.815 | 0.795 | 0.692 | 37 non / 83 dep | `[[22, 15], [17, 66]]` |
| Audio + Text | 0.658 | 0.752 | 0.756 | 0.747 | 0.602 | 37 non / 83 dep | `[[17, 20], [21, 62]]` |

Conclusion: macro-F1 checkpoint selection helps Turkish T17, especially
text-only. Text-only is the healthiest T17 result here.

### Turkish T21 Qwen3-ASR PosF1

Source:
`output_model/{audio_only,audio_text,text_only}/turkish_t21_qwen3asr/*/final_summary_active.csv`.

| Modality | ACC | Pos F1 | Precision | Recall | Macro F1 | Support | Confusion |
|---|---:|---:|---:|---:|---:|---|---|
| Audio Only | 0.583 | 0.706 | 0.556 | 0.968 | 0.496 | 58 non / 62 dep | `[[10, 48], [2, 60]]` |
| Text Only | 0.592 | 0.707 | 0.562 | 0.952 | 0.518 | 58 non / 62 dep | `[[12, 46], [3, 59]]` |
| Audio + Text | 0.550 | 0.697 | 0.534 | 1.000 | 0.413 | 58 non / 62 dep | `[[4, 54], [0, 62]]` |

Conclusion: positive-F1 selection still causes majority-positive collapse on
T21. Audio+text is the worst case: it detects every depressed subject but nearly
all non-depressed subjects are false positives.

## Current Interpretation

1. DAIC: the final partition mismatch is real, but not sufficient. The next
   meaningful tests require implementing DAIC chunk-level / all-30s-chunk
   evaluation and explicit aggregation modes.
2. CMDC: more than 3 epochs helps, but the paper gap is now more likely protocol
   mismatch. The next meaningful test is paper-like fixed split plus
   utterance-level scoring.
3. Turkish: positive-F1 checkpoint selection is a bad choice for these class
   balances. Macro-F1 selection should be preferred. T21 should be rerun with
   macro-F1 selection.
4. Audio+text fusion is not reliably better than single-modality models in the
   Turkish runs.

## Recommended Next Work

### 1. CMDC Paper-Like Fixed Split + Utterance-Level Scoring

Goal: test whether the paper's CMDC score is due to fixed split and utterance
level metrics rather than 5-fold subject-level CV.

Likely required code changes:

- Add a CMDC split mode for a fixed 60-subject train set and held-out eval set.
- Add/elevate an evaluation mode that scores utterances/responses directly
  instead of aggregating to subjects.
- Ensure metrics report utterance-level support and confusion matrix.
- Keep subject-level CV behavior unchanged for existing configs.

This is probably the highest-signal next implementation.

### 2. DAIC Chunk-Level / Aggregation Evaluation

Goal: test whether the DAIC paper values come from chunk-level scoring or a
different chunk-to-subject aggregation.

Likely required code changes:

- Build/evaluate all participant 30s chunks, not only a limited subject audio
  budget.
- Support multiple DAIC aggregation choices:
  - chunk-level metrics;
  - majority vote to subject;
  - mean probability/likelihood to subject;
  - any-positive or thresholded positive fraction.
- Report both chunk-level and subject-level metrics in one eval output.

### 3. Joint Balanced Instruction Tuning

Goal: test whether paper results depend on training one model on a merged,
balanced instruction dataset across DAIC + CMDC + EATD.

Likely required code changes:

- Build a merged train manifest across multiple datasets.
- Preserve dataset-specific final eval.
- Add class-balancing logic for the merged instruction set.
- Track dataset source in samples so eval remains separable.

This is larger than the CMDC/DAIC evaluation changes.

### 4. Turkish T21 Macro-F1 Selection

Goal: test whether T21 behaves like T17 once checkpoint selection uses macro F1.

There are untracked configs that appear to target this:

- `configs/experiments/turkish_t21_audio_only_selmacro_tf_qwen3asr.yaml`
- `configs/experiments/turkish_t21_text_only_selmacro_tf_qwen3asr.yaml`
- `configs/experiments/turkish_t21_audio_text_selmacro_tf_qwen3asr.yaml`

Before running, inspect them against the T17 selmacro configs and the main T21
configs. This likely needs only config validation and cluster execution, not code
changes.

## Cluster / Local Execution Notes

- Do not run full training locally.
- Use local commands only for reading files, parsing summaries, and lightweight
  sanity checks.
- Cluster outputs are synced back into:
  - `logs/`
  - `output_model/`
- Existing scripts of interest:
  - `scripts/run_daic_fixed.sh`
  - `scripts/run_cmdc_cv.sh`
  - `scripts/run_turkish_5fold.sh`
  - `scripts/sync_to_cluster.sh`

## Caution For Next Agent

- The paper appears internally inconsistent about aggregation levels.
- Do not assume exact reproduction is possible from configs alone.
- Avoid mixing old logs with new runs. Use run names/job IDs and official
  `final_summary_active.csv` files when available.
- The current git worktree has untracked experiment configs. Do not overwrite or
  delete them without checking their contents.
