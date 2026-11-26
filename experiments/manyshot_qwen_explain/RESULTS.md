# Qwen3.8-27B many-shot classification with explanation: results

Runner: `run_qwen_manyshot_explain.py` (advisor prompt v3, one greedy answer per subject: label, explanation, cited
examples). Hardware: MN5 acc, 2x H100 per job, `acc_debug` 2 h jobs chained by `scripts/mn5_debug_chain.sh`
(`--time-budget-min 100`, resume from the output dir). Metrics are pooled subject-level, INVALID counts as wrong.
All numbers below were recomputed locally with sklearn from `responses.jsonl`.

## DAIC-WOZ, official test (47 subjects, 14 depressed), 142 train+dev shots in fixed order

| Method (same 47 test subjects) | macro-F1 | pos-F1 | UAR |
|---|---|---|---|
| **Qwen3.8 142-shot + explanation (this run)** | **0.816** | **0.765** | **0.858** |
| Qwen3.8 142-shot likelihood, label only (OZY-159) | 0.776 | 0.690 | 0.781 |
| Opus v4 zero-shot PHQ-8 items, calibrated (mean of 4 runs) | 0.813 | 0.735 | 0.808 |
| Opus v4 zero-shot PHQ-8 items, raw total | 0.776 | 0.667 | 0.755 |

- Confusion: TP 13, FN 1, FP 7, TN 26. 0 INVALID, 0 answers cut by the token limit (max 246 tokens).
- 41/47 explanations cite labelled examples. Label-margin AUROC (secondary) 0.851.
- Cost: shared 142-shot prefix prefilled once (451 s), then ~27 s per target; peak GPU 59.5 GB.
- n = 47: one subject moves macro-F1 by about 0.02.
- Not the OZY-158 reference protocol (that is LOPO over all 189 subjects).

## Turkish, T17 (BDI >= 17), LOPO over 120 participants, full transcripts

| Method | macro-F1 | pos-F1 | UAR |
|---|---|---|---|
| **Qwen3.8 119-shot LOPO + explanation (this run)** | **0.634** | **0.832** | **0.626** |
| OZY-158 reference, frozen text embedding + ridge | 0.622 | 0.764 | — |

- Confusion: TP 77, FN 6, FP 25, TN 12. The model predicts depressed for 102/120 (true: 83/120), so the gain is
  mostly on the positive class; the healthy class stays weak (12/37 correct).
- 0 INVALID, 0 cut answers (max 229 tokens), 97/120 explanations cite examples. Label-margin AUROC 0.674.
- Cost: every target has its own ~194k-token prompt (no shared prefix): ~300 s per target, peak 51.8 GB, 7 jobs
  (19 targets per 100-minute job). Run only on MN5; transcripts never left the cluster.

## DAIC-WOZ zero-shot PHQ-8 item scoring (Claude v4 prompt, unchanged), official test 47

Calibration as in `experiments/manyshot_claude/calibrate_v4.py`: OLS of PHQ-8 total on the 8 item scores, fit on
the 142 train+dev subjects scored by the same model, applied unchanged to test. Label = score >= 10.

| Model | Score | RMSE | MAE | Spearman | macro-F1 | pos-F1 | UAR |
|---|---|---|---|---|---|---|---|
| Qwen3.8 (1 run) | raw total | 4.47 | 3.34 | 0.787 | 0.755 | 0.640 | 0.740 |
| Qwen3.8 (1 run) | calibrated | 4.41 | 3.44 | 0.794 | 0.755 | 0.667 | 0.766 |
| Opus v4 (mean of 4 runs) | raw total | 4.49 | 3.28 | 0.802 | 0.776 | 0.667 | 0.755 |
| Opus v4 (mean of 4 runs) | calibrated | 4.29 | 3.13 | 0.787 | 0.813 | 0.735 | 0.808 |

- Qwen ranks subjects about as well as Opus (Spearman 0.79 vs 0.80) and has a similar raw error, but the
  calibration does not help it: bias goes from -0.75 to +1.05 and MAE gets slightly worse. The fitted weights are
  less plausible than Opus's (interest item weight -0.58).
- Raw on train+dev: RMSE 3.98, MAE 3.06, Spearman 0.758, bias -1.65.
- 0 INVALID on both splits, no prompt shots, prompts 600-5800 tokens; 11 min (test) and 40 min (train+dev).
- For the label, Qwen many-shot with explanation (0.816) is clearly better than Qwen zero-shot PHQ scoring (0.755).

## Run notes

- `run_info.json` on MN5 has an empty `git_commit` (deployments have no `.git`). Commits are in the deployment
  directory names: Turkish and DAIC used `94959fb`, PHQ-8 used `f4a6db9` (prompt code identical for the
  classification task).
- Turkish first ran in `acc_ehpc`, where jobs stayed PENDING for over a day; they were cancelled and the run
  moved to `acc_debug` 2 h chains. Every `acc_debug` job finished COMPLETED; no retry was needed.
- Jobs: DAIC 49406023; Turkish 49407054, 49410169, 49412626, 49414210, 49415618, 49416874, 49418119;
  PHQ-8 test 49418562, train+dev 49418749.
- Outputs (not in git): `outputs/qwen_manyshot_explain/{full_daic,full_turkish,phq8_test,phq8_traindev}`,
  `outputs/qwen_manyshot_explain/phq8_calibration.json`.
