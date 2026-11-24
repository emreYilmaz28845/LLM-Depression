# DAIC many-shot prompting with Claude Opus (OZY-162)

Claude Opus 5.5 (`claude-opus-5-5`, effort medium, `claude -p`) classifies each official DAIC test subject
directly. There are no embeddings and no head. The system prompt holds the instructions plus all 142 official
train+dev subjects as labelled shots (42 Depressed / 100 Non-depressed, fixed order, shot seed 0). The user turn
is the test subject's participant transcript (`full_participant_transcript` from the reference DAIC manifest, the same text as the frozen
Qwen3.8 embeddings, OZY-158, and OZY-148/159). Claude answers with a JSON label and an explanation.

## Full-transcript results (2026-11-24) — use these

**Correction.** Every Claude run before 2026-11-24 read the local manifest
`outputs/manifests_daic_participant_packed30/daic_manifest.jsonl` (sha256 bfb9597ff28e…), where
`full_participant_transcript` is capped at 4000 characters (128 of 189 subjects are exactly 4000 after
stripping). Those runs saw only the first ~800 words of each interview. The frozen Qwen embeddings (OZY-158)
and OZY-148/159 used full transcripts. The runner now defaults to the reference manifest (copy of MN5
`experiment_runtime/feat-qwen38-standalone-promptcontext-20260922/manifests/daic/daic_manifest.jsonl`, sha256
8d96abc2a411…), refuses any other sha256 or any transcript of exactly 4000 chars, and records the manifest
source and sha256 in `run_info.json`. Old numbers are kept below, labelled "first 4000 chars".

All runs: DAIC-WOZ official test (47 subjects, 14 Depressed), Claude Opus 5.5, effort medium, one answer per
subject, binary_strict (INVALID counts as wrong). Many-shot = all 142 official train+dev subjects as numbered
labelled examples (shot seed 0). Full-text shot block: 1.005M chars, 327.9k input tokens (from `usage`);
the longest request was 335.8k tokens, well under the 1M context.

| Run | Text | n | macro-F1 | pos-F1 | UAR | TP / FN / FP / TN | INVALID | same answer old vs new | Artifacts (manifest sha256) |
|---|---|---|---|---|---|---|---|---|---|
| Many-shot, advisor prompt | first 4000 chars | 47 | 0.766 | 0.667 | 0.761 | 9 / 5 / 4 / 29 | 0 |  | `daic_claude_advisor_manyshot_effmedium` (bfb9597ff28e) |
| Many-shot, advisor prompt | full transcript | 47 | 0.743 | 0.667 | 0.772 | 11 / 3 / 8 / 25 | 0 | 41/47 | `daic_fulltext_claude_advisor_manyshot_effmedium` (8d96abc2a411) |
| Many-shot, prompt v2 | first 4000 chars | 47 | 0.796 | 0.714 | 0.796 | 10 / 4 / 4 / 29 | 0 |  | `ozy162_claude_manyshot_v2_effmedium` (bfb9597ff28e) |
| Many-shot, prompt v2 | full transcript | 47 | 0.763 | 0.688 | 0.787 | 11 / 3 / 7 / 26 | 0 | 43/47 | `daic_fulltext_claude_v2_manyshot_effmedium` (8d96abc2a411) |
| Zero-shot, advisor prompt | first 4000 chars | 47 | 0.755 | 0.640 | 0.740 | 8 / 6 / 3 / 30 | 0 |  | `daic_claude_advisor_zeroshot_effmedium` (bfb9597ff28e) |
| Zero-shot, advisor prompt | full transcript | 47 | 0.763 | 0.688 | 0.787 | 11 / 3 / 7 / 26 | 0 | 40/47 | `daic_fulltext_claude_advisor_zeroshot_effmedium` (8d96abc2a411) |
| Zero-shot, original prompt | first 4000 chars | 47 | 0.755 | 0.640 | 0.740 | 8 / 6 / 3 / 30 | 0 |  | `ozy_claude_zeroshot_effmedium` (bfb9597ff28e) |
| Zero-shot, original prompt | full transcript | 47 | 0.724 | 0.647 | 0.756 | 11 / 3 / 9 / 24 | 0 | 38/47 | `daic_fulltext_claude_zeroshot_effmedium` (8d96abc2a411) |
| Many-shot, prompt v1 (not rerun) | first 4000 chars | 47 | 0.766 | 0.667 | 0.761 | 9 / 5 / 4 / 29 | 0 |  | `ozy162_claude_manyshot_effmedium` (bfb9597ff28e) |

Table made by `experiments/manyshot_claude/compare_runs.py` from each run's `responses.jsonl`. The "first 4000
chars" v1/v2/original zero-shot artifacts are in the OZY-162 lane
(`worktrees/LLM-Depression-feat-ozy-162-daic-claude-manyshot/outputs/`); all others are in this lane's `outputs/`.

What changed with full text:

- Full text makes Claude call more people Depressed: FN drops to 3 in every run (from 4-6), FP rises to 7-9
  (from 3-4). Positive-F1 and UAR move little; macro-F1 drops for many-shot (v2 0.796 -> 0.763).
- With full text, the four runs give almost the same answers: advisor zero-shot and v2 many-shot agree on 47/47
  subjects, advisor many-shot agrees with each of them on 46/47, the original zero-shot with advisor zero-shot on
  45/47. So **the 142 examples do not change the label** on full transcripts; the earlier "many-shot gain"
  (0.755 -> 0.797) does not hold. Many-shot still differs in what the explanation can cite.
- The advisor prompt (Cenk Hoca, 2026-11-23: his sentence as is, "Base your answer on the examples in the
  prompt and explain your answer"; examples stay numbered, no rule to name similar ones) gives explanations that
  cite at least one numbered example in 31/47 subjects (25/47 cite two or more), often as a list such as
  "(e.g., Examples 16, 25, 33)"; v2, which requires naming 2-3 examples, cites them in 47/47 (counted with
  `build_explanation_sheet.cited_examples`, which reads singular, plural and listed citations). Explanations are in
  the advisor's Google Sheet (tabs Setup, Many-shot v2/v3, Zero-shot v1/v3; errors first; PHQ-8 column; anonymous
  subject numbers, no DAIC IDs); local workbook `outputs/daic_fulltext_claude_advisor_explanations.xlsx`.
- 47 subjects with 14 positive: one flipped subject moves macro-F1 by ~0.02-0.03 and positive-F1 by ~0.04,
  and the v1 vs v2 runs (first 4000 chars) already differed by 0.03. None of the differences above is beyond
  run-to-run variation.

Full-text cost (API-equivalent, notional, Claude Pro): advisor many-shot $4.66, v2 many-shot $7.38 (it wrote
its own shot cache), advisor zero-shot $1.64, original zero-shot $1.73; plus a 3-subject smoke
(`daic_fulltext_claude_advisor_manyshot_smoke`, $2.84) and a 311k-token synthetic-filler context probe ($2.49,
no DAIC data). Code: commit a16ce34 (run_info git_commit).

## Zero-shot v4 and v5: current symptoms instead of history (2026-11-24)

Why: in the four full-text runs the same 10 subjects are wrong every time. Their explanations show the model
leaning on diagnosis history: false positives have a past diagnosis, treatment or heavy stress but PHQ-8 5-9;
false negatives deny a diagnosis or describe recovery but have PHQ-8 12-18. Two zero-shot prompts test this
(prompt text in `run_claude_manyshot.py`, `ZEROSHOT_ONLY`; approved by the user before the run):

- v4: the model scores the eight PHQ-8 items for the last two weeks (0-3 each, best estimate where the transcript
  has no direct evidence); code sets Depressed = item sum >= 10 (the PHQ-8 cutoff, not tuned).
- v5: zero-shot v1 plus one paragraph: judge the last two weeks; past diagnosis, treatment or hard events are not
  by themselves evidence, and denying a diagnosis is not evidence against.

| Run (full transcripts, 47 subjects) | macro-F1 | pos-F1 | UAR | TP / FN / FP / TN | INVALID |
|---|---|---|---|---|---|
| Zero-shot v1 (reference) | 0.724 | 0.647 | 0.756 | 11 / 3 / 9 / 24 | 0 |
| Zero-shot v4 (PHQ-8 items) | 0.776 | 0.667 | 0.755 | 8 / 6 / 2 / 31 | 0 |
| Zero-shot v5 (last two weeks) | 0.766 | 0.667 | 0.761 | 9 / 5 / 4 / 29 | 0 |

- Both prompts cut false positives (v1 9 -> v4 2, v5 4) and add false negatives (3 -> 6, 5). Of the 10
  always-wrong subjects, v4 fixes 5 (all false positives) and v5 fixes 3; the new errors are all depressed
  subjects at PHQ-8 10-12, just above the cutoff. Positive-F1 and UAR do not improve; macro-F1 rises because the
  majority class gains. Single runs, n=47: the gain is within run-to-run variation.
- v4's item total tracks the real PHQ-8 score: Spearman rho 0.817 (n=47), mean absolute error 3.2 points, AUROC
  of the total for the binary label 0.879. The model under-scores slightly (mean 6.1 vs 7.0), which is why
  borderline depressed subjects fall below 10. Depressed vs non-depressed item means differ most on sleep
  (2.2 vs 0.9), feeling down (1.7 vs 0.8) and feeling bad about oneself (1.7 vs 0.9); appetite is flat (0.2 vs 0.2).
- Cost (API-equivalent, Claude Pro): v4 $1.60, v5 $1.78, v4 smoke (3 subjects) ~$0.09. Artifacts:
  `outputs/daic_fulltext_claude_zs_v4_effmedium/` (responses include `phq8_items`, `phq8_total`) and
  `outputs/daic_fulltext_claude_zs_v5_effmedium/`. Code: commit 0d13f35.

## Zero-shot v4 as a PHQ-8 score: repeat runs and literature (2026-11-24)

Zero-shot v4 run 4 times (same prompt, same 47 official test subjects, effort medium; no temperature control).
RMSE/MAE/Spearman compare the item total (0-24) with the real PHQ-8 score.

| Run | macro-F1 | pos-F1 | UAR | TP / FN / FP / TN | RMSE | MAE | Spearman |
|---|---|---|---|---|---|---|---|
| r1 | 0.776 | 0.667 | 0.755 | 8 / 6 / 2 / 31 | 4.36 | 3.21 | 0.817 |
| r2 | 0.776 | 0.667 | 0.755 | 8 / 6 / 2 / 31 | 4.43 | 3.23 | 0.812 |
| r3 | 0.776 | 0.667 | 0.755 | 8 / 6 / 2 / 31 | 4.75 | 3.45 | 0.757 |
| r4 | 0.776 | 0.667 | 0.755 | 8 / 6 / 2 / 31 | 4.42 | 3.23 | 0.821 |
| mean (SD) | 0.776 (0) | 0.667 (0) | 0.755 (0) | | 4.49 (0.18) | 3.28 (0.11) | 0.802 (0.030) |

- The binary result is the same in all 4 runs (45/47 subjects get the same label; the 2 that flip cancel out).
  Per-subject totals move by 0.45 points (mean SD). INVALID 0 in every run.
- Constant baselines on the same 47 subjects: train+dev mean (6.67) RMSE 6.41 / MAE 5.43; train+dev median (5.5)
  RMSE 6.57 / MAE 5.33.
- The model compresses the scale: subjects with PHQ-8 >= 10 are under-scored by 5.9 points on average, subjects
  below 10 over-scored by 1.3 (run r1).

Published DAIC-WOZ test results (checked 2026-11-24 against the papers; MAE only where RMSE is only in figures):

| Method | Test n | Training | RMSE | MAE |
|---|---|---|---|---|
| Lee, Han & Woo 2026, PLOS Digit Health: 23 zero-shot LLaMA-3.1-70B features + linear regression | 46 | regression on 140 | figure only | 2.91 |
| Same, only the 8 PHQ-symptom features + regression | 46 | regression on 140 | figure only | 2.97 |
| **Ours: Opus 5.5 zero-shot v4, item total used directly (mean of 4 runs)** | 47 | none | 4.49 | 3.28 |
| Lee et al.: LLM asked directly for a severity score | 46 | none | figure only | 3.51 |
| Earlier text models cited by Lee et al. | 46-47 | yes | | 3.46-3.80 |
| Schmidt et al. 2025 (PTTSD, arXiv 2511.04476), best DAIC | "56" (paper's count) | yes | 4.77 | 3.55 |
| Stepanov et al. 2017 (AVEC 2017 winner, arXiv 1711.06095), test | AVEC test | yes | 4.94 | 4.11 |

Caveats: Lee et al. drop 3 subjects without interviewer text (Test Set 1 = official test minus one) and keep the
interviewer's questions in the prompt; our prompt has the participant's side only. PTTSD reports 56 test subjects,
which does not match the official 47. Opus may have seen DAIC-WOZ in pre-training; this cannot be ruled out.
Artifacts: `outputs/daic_fulltext_claude_zs_v4_effmedium{,_r2,_r3,_r4}/`; cost r2-r4 $0.57 each (API-equivalent).

## Zero-shot v4 + linear calibration: plan, written before the run (2026-11-24)

Same protocol as OZY-159 (`daic_traindev_test`): fit on official train+dev (142), official test (47) locked and
used once.
1. Run zero-shot v4 on the 142 train+dev subjects (`--targets traindev`); the prompt has no examples, so no subject
   sees another subject's label.
2. Fit one ordinary least-squares regression on those 142: PHQ-8 total ~ intercept + 8 item scores. No
   hyperparameter, no selection, no refit.
3. Apply it unchanged to each of the 4 existing test runs (r1-r4); report mean (SD) over the 4 runs and r1 alone.
   Label = predicted score >= 10 (fixed). Report RMSE, MAE, Spearman, macro-F1, pos-F1, UAR next to the raw item
   total. Subjects whose v4 answer is INVALID are dropped from the fit and counted wrong on test.

## First 4000 chars only (superseded, kept for the record)

Everything below was run on the truncated manifest and is kept only for comparison with the table above.

**Was the reference result before the truncation was found: prompt v2** (user decision, 2026-11-20). In v2 the explanation must name the 2-3 most
similar labelled examples by number and label, as Cenk Hoca asked ("explaining by looking at the examples is
important"). Run artifacts: `outputs/ozy162_claude_manyshot_v2_effmedium/`.

| Method (DAIC official test, 47 subjects) | macro-F1 | positive-F1 | UAR |
|---|---|---|---|
| Claude Opus 5.5 many-shot prompting, prompt v2 (first 4000 chars) | 0.797 | 0.714 | 0.797 |
| Claude Opus 5.5 zero-shot prompting (no shots, `--n-shots 0`; first 4000 chars) | 0.755 | 0.640 | 0.740 |
| Qwen3.8 A: frozen zero-shot embedding + head (OZY-148) | 0.735 | | |
| Qwen3.8 B: many-shot embedding + head (OZY-148) | 0.750 | | |
| Qwen3.8 C: Enes many-shot logprob (`many_fit`) | 0.776 | | |

Confusion (subject-level): TP 10, FN 4, FP 4, TN 29; 14 predicted positive vs 14 true positives. There were
0 INVALID answers, and INVALID would count as wrong (binary_strict). All 47 explanations name at least 2
numbered examples, and all 152 cited example labels match the real shot labels (checked against the shot
order). The Qwen rows are copied from `experiments/manyshot/RESULTS.md`; only macro-F1 is reported there.

Superseded, not a reference: an earlier full run with prompt v1, where comparing with the examples was optional
(`outputs/ozy162_claude_manyshot_effmedium/`), gave macro-F1 0.766 / positive-F1 0.667 / UAR 0.761. It agrees
with v2 on 46/47 subjects, so the gap is within run-to-run variation and should not be read as a prompt effect.

### Zero-shot (Cenk Hoca, 2026-11-23; first 4000 chars)

Same 47 test subjects, model, effort, label definition and JSON answer format as v2, but the system prompt has
no labelled examples, so the explanation can cite only the participant's transcript. Run artifacts:
`outputs/ozy_claude_zeroshot_effmedium/` (3-subject smoke: `outputs/ozy_claude_zeroshot_smoke_effmedium/`).

- Confusion: TP 8, FN 6, FP 3, TN 30; 11 predicted positive vs 14 true. 0 INVALID answers.
- Zero-shot and many-shot v2 give the same label on 44/47 subjects. The many-shot gain (0.755 -> 0.797 macro-F1)
  comes from 3 subjects, which is within the run-to-run variation seen between v1 and v2 (46/47 agreement).
- The explanations describe symptoms in the transcript (low mood, sleep, anhedonia, hopelessness, stressors,
  isolation). Zero-shot errors look like this: a participant who reports a depression diagnosis and
  hopelessness but has PHQ-8 < 10 is called Depressed; a positive, active participant whose interview has
  few direct symptom answers is called Non-depressed. Many-shot explanations can also point to similar
  labelled examples, which zero-shot cannot.
- Cost ~$1.14 API-equivalent (notional, Claude Pro), ~885 s total call time.
- Code: commit 1052976.

### Caveats

- Single run. Opus 5.5 has no temperature control, so a rerun can give different answers. The raw answers
  are stored per subject.
- 47 test subjects (14 positive): one flipped positive moves positive-F1 by about 0.04.
- The shots use the official train+dev subjects (OZY-159 protocol). The Qwen rows A/B/C use train shots with
  dev for selection, so the shot pools differ.

### Provenance (first-4000-chars runs)

- Code: `experiments/manyshot_claude/run_claude_manyshot.py`, commit eed1c25 (run_info git_commit = worktree HEAD).
- Manifest: `outputs/manifests_daic_participant_packed30/daic_manifest.jsonl` (sha256 in run_info.json).
- v2: commit 89f9b2c, `--prompt-version v2`, artifacts in `outputs/ozy162_claude_manyshot_v2_effmedium/`
  (API-equivalent ~$2.89, ~450 s). The 3-subject v2 smoke run is in `outputs/ozy162_smoke_v2_effmedium/`
  and is not part of the results.
- v1 artifacts (local, gitignored): `outputs/ozy162_claude_manyshot_effmedium/` (run_info.json, responses.jsonl
  with raw answers and usage, metrics.json, run.log).
- Run 2026-11-20 on the user's Claude Pro subscription. All 47 calls read the shot prefix from the prompt
  cache. The API-equivalent cost was about $2.79 (notional, not billed). Wall time was about 400 s.
- Set aside, not used in the results: `outputs/ozy162_smoke_singleblock_nocache/` (2 subjects, earlier
  single-block layout, high effort) and `outputs/ozy162_partial_effhigh_stopped/` (3 subjects, high effort,
  stopped when the user chose medium).
