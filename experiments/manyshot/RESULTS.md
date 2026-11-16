# Many-shot-context embeddings + logistic head (OZY-148)

Qwen3.8-27B, text_only_native routes. Native macro-F1, mean over outer folds (DAIC: official test, 1 split).
Same head for A and B: `baselines/frozen_standalone_heads.py` (subject-mean L2 logistic, balanced
weights, L2 chosen on inner CV, threshold 0.5).

| Route | A: frozen zero-shot embedding + head | B: many-shot embedding + head | C: Enes many-shot logprob (`many_fit`) |
|---|---|---|---|
| Androids | 0.888 | 0.869 | 0.861 |
| CMDC | 0.970 | 0.970 | 0.986 |
| D3TEC | 0.652 | 0.463 | 0.625 |
| DAIC | 0.735 | 0.750 | 0.776 |
| Turkish | 0.651 | 0.700 | 0.676 |
| Mean (5) | 0.779 | 0.750 | 0.785 |

Per-fold macro-F1, B: Androids [0.869, 0.833, 0.869, 0.956, 0.818]; CMDC [1, 1, 1, 0.928, 0.921];
D3TEC [0.462, 0.513, 0.458, 0.667, 0.214]; Turkish [0.667, 0.833, 0.497, 0.767, 0.735].
A: D3TEC [0.921, 0.381, 0.575, 0.748, 0.633]; Turkish [0.548, 0.74, 0.521, 0.753, 0.692].
D3TEC and CMDC test folds hold 11-17 subjects, so fold-level differences are noisy.

## How B is built (leak-free)

- Prompt = system prompt + note + all outer-fit transcripts with their gold labels as user/assistant turns
  + the target transcript; vector = final post-norm hidden state at the last prompt token (end of the
  closed thinking block), same position as the frozen cache.
- Fit subjects are split into 5 label-stratified inner groups; a fit subject is embedded with shots from
  the other 4 groups only (never its own label). Selection/final/DAIC-dev subjects are embedded with each
  of the 5 shot sets and averaged.
- Shared prefix prefilled once per shot set (HF hybrid cache, deep-copied per row).

## Checks

- Zero-shot reproduction vs frozen cache (androids fold 0, 116 rows): min cosine 0.998.
- Cached vs uncached forward: cosine >= 0.9997. Determinism: max abs diff 0.0 on every fold.
- Arm A rerun with the same head code reproduces the published frozen numbers exactly.

## Caveat

B fit rows see 4/5 of the shots while evaluation rows are 5-set averages; this train/eval
representation shift may explain part of the D3TEC drop. Not tested.

## Provenance (MN5)

- Extraction: jobs 49117824 (non-Turkish, 2h04), 49117825 (Turkish); smoke 49117300. Code commit 00c87cd,
  deployment feat-ozy-148-manyshot-embedding-20261116-20261116T163303Z-00c87cd6-b492162e.
- Caches: experiment_runtime/feat-ozy-148-manyshot-embedding-20261116/full/manyshot/.
- Heads: experiment_runtime/feat-ozy-148-manyshot-embedding-20261116/heads/{A_frozen,B_manyshot}/, code
  commit 08a94a8 (deployment ...T212404Z-08a94a80-8b08ea56; its deploy step timed out before writing
  deployment.json, deployed head/extractor files verified identical to the commit by sha256).
- C: Enes's compare5.json (read-only, his MN5 directory), arm `many_fit`.
