# DAIC many-shot prompting with Claude Opus (OZY-162)

Claude Opus 5.5 (`claude-opus-5-5`, effort medium, `claude -p`) classifies each official DAIC test subject
directly. There are no embeddings and no head. The system prompt holds the instructions plus all 142 official
train+dev subjects as labelled shots (42 Depressed / 100 Non-depressed, fixed order, shot seed 0). The user turn
is the test subject's participant transcript (`full_participant_transcript`, the same text as Qwen3.8
text_only_native). Claude answers with a JSON label and an explanation.

| Method (DAIC official test, 47 subjects) | macro-F1 | positive-F1 | UAR |
|---|---|---|---|
| Claude Opus 5.5 many-shot, prompt v1 (explanation may compare with the examples) | 0.766 | 0.667 | 0.761 |
| Claude Opus 5.5 many-shot, prompt v2 (explanation must name the most similar labelled examples) | 0.797 | 0.714 | 0.797 |
| Qwen3.8 A: frozen zero-shot embedding + head (OZY-148) | 0.735 | | |
| Qwen3.8 B: many-shot embedding + head (OZY-148) | 0.750 | | |
| Qwen3.8 C: Enes many-shot logprob (`many_fit`) | 0.776 | | |

Confusion (subject-level): v1 TP 9, FN 5, FP 4, TN 29; v2 TP 10, FN 4, FP 4, TN 29.
v1 and v2 agree on 46/47 subjects. The whole difference is one depressed subject that v2 got right, which is
within run-to-run variation, so this is not evidence that v2 classifies better. What v2 changes is the
explanations: 47/47 name at least 2 numbered examples (v1: 13/47 cite any example number). All 152 cited
example labels match the real shot labels (checked against the shot order).
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
