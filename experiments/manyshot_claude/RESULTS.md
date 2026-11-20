# DAIC many-shot prompting with Claude Opus (OZY-162)

Claude Opus 5.5 (`claude-opus-5-5`, effort medium, `claude -p`) classifies each official DAIC test subject
directly. There are no embeddings and no head. The system prompt holds the instructions plus all 142 official
train+dev subjects as labelled shots (42 Depressed / 100 Non-depressed, fixed order, shot seed 0). The user turn
is the test subject's participant transcript (`full_participant_transcript`, the same text as Qwen3.8
text_only_native). Claude answers with a JSON label and an explanation.

| Method (DAIC official test, 47 subjects) | macro-F1 | positive-F1 | UAR |
|---|---|---|---|
| Claude Opus 5.5 many-shot prompting, label from its answer (this run) | 0.766 | 0.667 | 0.761 |
| Qwen3.8 A: frozen zero-shot embedding + head (OZY-148) | 0.735 | | |
| Qwen3.8 B: many-shot embedding + head (OZY-148) | 0.750 | | |
| Qwen3.8 C: Enes many-shot logprob (`many_fit`) | 0.776 | | |

Confusion (subject-level): TP 9, FN 5, FP 4, TN 29. 13 subjects predicted positive vs 14 true positives.
There were 0 INVALID answers, and INVALID would count as wrong (binary_strict). The Qwen rows are copied
from `experiments/manyshot/RESULTS.md`; only macro-F1 is reported there.

## Caveats

- Single run. Opus 5.5 has no temperature control, so a rerun can give different answers. The raw answers
  are stored per subject.
- 47 test subjects (14 positive): one flipped positive moves positive-F1 by about 0.04.
- The shots use the official train+dev subjects (OZY-159 protocol). The Qwen rows A/B/C use train shots with
  dev for selection, so the shot pools differ.

## Provenance

- Code: `experiments/manyshot_claude/run_claude_manyshot.py`, commit eed1c25 (run_info git_commit = worktree HEAD).
- Manifest: `outputs/manifests_daic_participant_packed30/daic_manifest.jsonl` (sha256 in run_info.json).
- Artifacts (local, gitignored): `outputs/ozy162_claude_manyshot_effmedium/` (run_info.json, responses.jsonl
  with raw answers and usage, metrics.json, run.log).
- Run 2026-11-20 on the user's Claude Pro subscription. All 47 calls read the shot prefix from the prompt
  cache. The API-equivalent cost was about $2.79 (notional, not billed). Wall time was about 400 s.
- Set aside, not used in the results: `outputs/ozy162_smoke_singleblock_nocache/` (2 subjects, earlier
  single-block layout, high effort) and `outputs/ozy162_partial_effhigh_stopped/` (3 subjects, high effort,
  stopped when the user chose medium).
