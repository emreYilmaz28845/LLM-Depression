# DAIC many-shot prompting with Claude Opus (OZY-162)

Claude Opus 5.5 (`claude-opus-5-5`, effort medium, `claude -p`) classifies each official DAIC test subject
directly. There are no embeddings and no head. The system prompt holds the instructions plus all 142 official
train+dev subjects as labelled shots (42 Depressed / 100 Non-depressed, fixed order, shot seed 0). The user turn
is the test subject's participant transcript (`full_participant_transcript`, the same text as Qwen3.8
text_only_native). Claude answers with a JSON label and an explanation.

**Reference result: prompt v2 only** (user decision, 2026-11-20). In v2 the explanation must name the 2-3 most
similar labelled examples by number and label, as Cenk Hoca asked ("explaining by looking at the examples is
important"). Run artifacts: `outputs/ozy162_claude_manyshot_v2_effmedium/`.

| Method (DAIC official test, 47 subjects) | macro-F1 | positive-F1 | UAR |
|---|---|---|---|
| **Claude Opus 5.5 many-shot prompting, prompt v2** | **0.797** | **0.714** | **0.797** |
| Claude Opus 5.5 zero-shot prompting (no shots, `--n-shots 0`) | 0.755 | 0.640 | 0.740 |
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

## Zero-shot (Cenk Hoca, 2026-11-23)

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

## Caveats

- Single run. Opus 5.5 has no temperature control, so a rerun can give different answers. The raw answers
  are stored per subject.
- 47 test subjects (14 positive): one flipped positive moves positive-F1 by about 0.04.
- The shots use the official train+dev subjects (OZY-159 protocol). The Qwen rows A/B/C use train shots with
  dev for selection, so the shot pools differ.

## Provenance

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
