# Local Repair and Run Findings

Date: 2026-07-14

Repository: `/home/emre/Projects/AudioLLM/LLM-Depression`

Repository commit: `f2e48744ade4b2dfcd344fdd56c95fae20727d2e`

This document is the explicitly authorized implementation-and-execution follow-up to
[`investigation_findings.md`](investigation_findings.md) and
[`investigation_plan.md`](investigation_plan.md). The earlier forensic report remains an
accurate record of a read-only investigation. The changes and runs below happened only after
the user authorized local cleanup, package/model installation, repairs, and execution.

## 1. Executive outcome

The prioritized local run set is complete. The main result is more conservative than
the original single-view checkpoint score suggests:

- The checkpoint demonstrably uses the transcript. Across eight balanced audio views,
  transcript shuffling reduced first-token AUROC by `0.3377` with a subject-paired 95% CI of
  `[0.1286, 0.5395]`, and reduced balanced accuracy by `0.2338`
  `[0.0833, 0.3941]`.
- Audio changes predictions, but a stable benefit from the correctly paired audio was not
  established. Real audio versus silence and real audio versus across-subject shuffled audio
  both had correct-class margin confidence intervals crossing zero. Under the investigation
  plan's preregistered rule, the checkpoint is **weakly used**, not `used` and not `ignored`.
- Same-class audio shuffling was essentially indistinguishable from real audio. This is
  compatible with class-correlated acoustic or preprocessing shortcuts, but does not prove
  that such a shortcut is the cause.
- Two controlled frozen-audio ceilings were null on the fixed DAIC split: eGeMAPS AUROC
  `0.5000` and WavLM Base+ AUROC `0.5195`, both with wide intervals spanning chance and
  non-significant shuffled-audio controls.
- The exact one-example Qwen2-Audio forward/backward audit did not fit on the RTX 4090. It
  failed honestly at the specified full K=4 protocol, wrote an immutable OOM artifact, and did
  not reduce K, text, audio, precision, or model size.
- The measurement repairs and all supporting local code passed `54/54` unit tests, the WavLM
  no-model test, Python compilation, shell syntax checking, and `git diff --check`.

The evidence supports **text dominance with unstable/weak checkpoint audio use**. It does not
support a claim that clinically meaningful acoustic reasoning has been learned, and it also
does not support the broader claim that DAIC audio contains no depression-related signal.

## 2. Run-status and gate matrix

| Item | Local result | Decision |
|---|---|---|
| R0 deterministic evaluation repair | Implemented and tested | Complete |
| E0 legacy one-view perturbation audit | All 8 conditions, 47 subjects each | Complete; diagnostic anchor only |
| E0 numeric-balanced eight-view audit | 8 views x 8 conditions x 47 subjects | Complete; primary E0 result |
| E0 paired uncertainty | 12 reports per protocol, 10,000 subject bootstraps | Complete |
| Exact gradient-flow audit | CUDA OOM during full forward/backward | Attempt complete; wiring evidence remains unresolved |
| E1 eGeMAPSv02 ceiling | Fixed K=4, 189 subjects | Complete; null on this fixed split |
| E1 WavLM Base+ ceiling | Fixed K=4, layers 6/7/8, 189 subjects | Complete; null on this fixed split |
| E1 cross-fold and MIL pooling | Not run | Still needed before a general acoustic-ceiling claim |
| emotion2vec ceiling | Pruned | Lower priority after two null fixed-split ceilings; not evidence against emotion2vec |
| E2 capacity-controlled Qwen2-Audio training | Not run | Gate closed by unstable E0 audio benefit and null fixed-split E1 ceilings |
| E3 controlled fusion | Not run | Requires a reproducibly useful unimodal audio branch first |
| E4 alignment rebuild | Not run | Blocked by missing raw timestamps, diarization, and participant-channel provenance |

## 3. Local environment, storage, and compute

No existing environment was mutated. An isolated environment was created at
`/home/emre/miniconda3/envs/llmdep4090` and all runs used `PYTHONNOUSERSITE=1`.

| Component | Version or size |
|---|---:|
| GPU | NVIDIA RTX 4090, 24 GiB |
| Python | 3.10.20 |
| PyTorch | 2.3.0+cu121 |
| Transformers | 4.55.0 |
| PEFT | 0.17.0 |
| Accelerate | 1.8.1 |
| NumPy / scikit-learn / SciPy | 1.26.4 / 1.7.0 / 1.15.3 |
| Isolated Conda environment | 6,052,168,341 bytes = 5.637 GiB |
| Local WavLM Base+ model | 755,197,889 bytes = 0.703 GiB |
| Provisioned openSMILE bundle | 21,201,616 bytes = 20.22 MiB |
| New canonical baseline and E0 outputs | about 39.6 MiB |
| Total incremental environment/model/tool/canonical-output footprint | 6,870,143,121 bytes = 6.398 GiB |
| Free root-filesystem space after the earlier cleanup and these runs | 82,675,363,840 bytes = 77.00 GiB (`df -h`: 78G) |

The incremental total excludes the pre-existing Qwen2-Audio base model
(`/home/emre/models/Qwen2-Audio-7B-Instruct`, 16,806,520,030 bytes) and copied checkpoint
(177,301,117 bytes).

Peak E0 inference allocation was 18,405,195,264 bytes (17.14 GiB), with
24,163,385,344 bytes (22.50 GiB) reserved. Full deterministic inference therefore fits on the
4090. The exact backward audit does not.

## 4. Repairs and new measurement code

### 4.1 Deterministic R0 repair

The following production evaluation paths were repaired:

- [`src/evaluate.py`](../src/evaluate.py) now enters a deterministic evaluation context,
  uses inference mode, disables KV-cache use for direct scoring, and restores the caller's
  original train/eval state even when evaluation raises.
- [`src/model/qwen2audio_lora.py`](../src/model/qwen2audio_lora.py) and
  [`src/model/text_lora.py`](../src/model/text_lora.py) now explicitly load inference models in
  BF16 on CUDA, matching the training precision rather than relying on an implicit default.
- Runtime summaries now record device, model dtype, parameter dtypes, and bounded CUDA peak
  allocation/reservation fields.

These changes address the verified train-mode/dropout evaluation defect. They do not rewrite
or retroactively validate historical artifacts.

### 4.2 E0 measurement implementation

[`src/e0_perturbations.py`](../src/e0_perturbations.py) implements immutable subject-level
artifacts for:

1. real transcript + real audio;
2. real transcript + silence;
3. real transcript + across-subject shuffled audio;
4. real transcript + same-class shuffled audio;
5. shuffled transcript + real audio;
6. transcript-removed real audio;
7. transcript-removed silence;
8. transcript-removed shuffled audio.

The primary score compares the checkpoint-native first label tokens `Dep` (token ID `7839`)
and `Non` (token ID `8121`) from the prompt-only forward pass. It is not an A/B scorer.
Full-candidate likelihood is retained only as a secondary diagnostic because `Depressed` and
`Non-depressed` have unequal continuation lengths.

The numeric-balanced family materializes eight genuinely distinct K=4 views per subject using
numeric suffix order. The schedule is content-independent and balances each subject's chunk
exposure to within one occurrence. The global schedule SHA-256 is:

`4e2b27a48a1117e75e7c20aa46054571d24b884d7a28b2621884b20530d43d17`

Numeric order is a deterministic ordinal, not timestamp-verified chronology. The legacy
lexical K=4 view is kept as a separate anchor and is never pooled with the numeric-balanced
family.

[`scripts/summarize_e0_views.py`](../scripts/summarize_e0_views.py) requires exactly eight
consistent view roots, averages each subject's continuous margin across views before applying
the decision threshold, then performs paired subject bootstrap comparisons. Input artifacts,
code hashes, and generated outputs are pinned in its provenance file.

### 4.3 Baseline and audit implementations

- [`src/baselines/egemaps_ceiling.py`](../src/baselines/egemaps_ceiling.py) and
  [`scripts/run_egemaps_ceiling.sh`](../scripts/run_egemaps_ceiling.sh) implement the fixed-K
  eGeMAPSv02 subject ceiling.
- [`baselines/wavlm_frozen_subject_baseline.py`](../baselines/wavlm_frozen_subject_baseline.py)
  implements the frozen WavLM subject ceiling and shuffled-bundle control.
- [`scripts/e0_gradient_audit.py`](../scripts/e0_gradient_audit.py) pins the full checkpoint,
  input example, target span, optimizer membership, trainable groups, and no-fallback OOM
  policy for one exact forward/backward without an optimizer step.

## 5. Input and checkpoint provenance

All canonical runs used:

| Input | Pinned value |
|---|---|
| DAIC manifest | `outputs/manifests/daic_manifest.jsonl` |
| Manifest SHA-256 | `e31385760a0536a06f9ff38fe20e3eab9fa5dd6736c38de5bb8cd577438f61e3` |
| Subject partitions | `outputs/splits/daic_subject_partitions.json` |
| Partition SHA-256 | `12b1a48cbfcc77771c6047ffa040616d16a66e99f8ad6d956687ffc5fb4d5fe4` |
| Train / validation / test subjects | 107 / 35 / 47 |
| Test support | 33 negative / 14 positive |
| Adapter checkpoint | `output_model/audits/e0/checkpoints/daic_posf1_tf_daic_audio_text_selmacrof1_tf/fold_0/best_model` |
| Adapter SHA-256 | `06b6f7592dfdfd9a0864acd7be59e80661a474324e43c3f14080e4e6e7ce5ed2` |
| Base model | `/home/emre/models/Qwen2-Audio-7B-Instruct` |

The adapter contains 448 LoRA tensors spanning all 32 decoder layers and contains no
audio-tower or projector adapter tensors, consistent with those modules being frozen. This is
a static checkpoint fact; it does not establish whether gradients or decisions depend on
audio.

## 6. E0 legacy single-view results

Canonical root:
`output_model/audits/e0/direct_first_token_legacy_k4_all8_deterministic_20260714`

Each condition contains 47 unique test subjects and an immutable prediction JSONL. The model
was loaded once for the complete eight-condition run.

| Condition | Accuracy | Balanced acc. | Macro F1 | Positive F1 | AUROC | AUPRC |
|---|---:|---:|---:|---:|---:|---:|
| Real | 0.7660 | 0.8333 | 0.7590 | 0.7179 | 0.9123 | 0.7925 |
| Silence | 0.7021 | 0.7262 | 0.6849 | 0.6111 | 0.8604 | 0.7546 |
| Across-subject audio shuffle | 0.6170 | 0.6656 | 0.6083 | 0.5500 | 0.8355 | 0.7369 |
| Same-class audio shuffle | 0.7660 | 0.8128 | 0.7549 | 0.7027 | 0.9167 | 0.8033 |
| Transcript shuffle | 0.5532 | 0.5996 | 0.5458 | 0.4878 | 0.5920 | 0.3998 |
| Transcript removed, real audio | 0.6383 | 0.7219 | 0.6357 | 0.6047 | 0.8160 | 0.6091 |
| Transcript removed, silence | 0.7021 | 0.5000 | 0.4125 | 0.0000 | 0.5000 | 0.2979 |
| Transcript removed, audio shuffle | 0.3404 | 0.3658 | 0.3356 | 0.2791 | 0.3755 | 0.2830 |

Primary first-token paired differences are reference minus control, with 10,000 subject-level
bootstrap repetitions:

| Contrast | Correct-class margin delta [95% CI] | AUROC delta [95% CI] | Balanced-accuracy delta [95% CI] |
|---|---:|---:|---:|
| Real - silence | 0.0731 [-0.2846, 0.4415] | 0.0520 [-0.0065, 0.1320] | 0.1071 [-0.0155, 0.2476] |
| Real - audio shuffle | 0.3684 [-0.0758, 0.8444] | 0.0768 [-0.0011, 0.1810] | 0.1677 [0.0316, 0.3167] |
| Real - same-class shuffle | -0.0239 [-0.4628, 0.4003] | -0.0043 [-0.0520, 0.0470] | 0.0206 [-0.0769, 0.1313] |
| Real - transcript shuffle | 1.8816 [0.5332, 3.1848] | 0.3203 [0.1167, 0.5197] | 0.2338 [0.0770, 0.3982] |
| Audio-only real - silence | 0.5346 [-0.3059, 1.3750] | 0.3160 [0.1770, 0.4292] | 0.2219 [0.1031, 0.3235] |
| Audio-only real - shuffle | 1.6170 [0.6649, 2.5346] | 0.4405 [0.2048, 0.6606] | 0.3561 [0.1432, 0.5521] |

The legacy run alone would be classified `weakly used`: audio perturbations change outputs,
but the primary continuous-margin intervals do not establish a stable aligned-audio benefit.
The much larger transcript-shuffle loss establishes transcript dependence.

An unplanned continuation root repeated the last four conditions. Its four prediction JSONLs
are byte-identical to the canonical run, providing an additional determinism check:
`output_model/audits/e0/direct_first_token_legacy_k4_remaining4_deterministic_20260714`.

## 7. E0 numeric-balanced eight-view result

Canonical view root:
`output_model/audits/e0/numeric_balanced_k4_8views_deterministic_20260714`

Canonical aggregate root:
`output_model/audits/e0/numeric_balanced_k4_8view_summary_20260714`

An independent read-only audit recomputed all primary and secondary metrics, artifact hashes,
subject sets, labels, assignments, K=4 paths, scorer decisions, and perturbation mappings. It
found zero errors across all 3,008 prediction rows (`8 views x 8 conditions x 47 subjects`).
Every root has completion provenance. Each subject has eight unique numeric-ascending K=4
bundles with balanced exposure.

### 7.1 Metrics after averaging subject margins across views

| Condition | Accuracy | Balanced acc. | Macro F1 | Positive F1 | AUROC | AUPRC |
|---|---:|---:|---:|---:|---:|---:|
| Real | 0.7447 | 0.8182 | 0.7389 | 0.7000 | 0.9134 | 0.7882 |
| Silence | 0.7021 | 0.7262 | 0.6849 | 0.6111 | 0.8604 | 0.7546 |
| Across-subject audio shuffle | 0.6596 | 0.7165 | 0.6519 | 0.6000 | 0.8387 | 0.7441 |
| Same-class audio shuffle | 0.7660 | 0.8128 | 0.7549 | 0.7027 | 0.9026 | 0.7757 |
| Transcript shuffle | 0.5319 | 0.5844 | 0.5266 | 0.4762 | 0.5758 | 0.4094 |
| Transcript removed, real audio | 0.6809 | 0.7522 | 0.6756 | 0.6341 | 0.8009 | 0.5924 |
| Transcript removed, silence | 0.7021 | 0.5000 | 0.4125 | 0.0000 | 0.5000 | 0.2979 |
| Transcript removed, audio shuffle | 0.4255 | 0.4470 | 0.4160 | 0.3415 | 0.4318 | 0.3036 |

Real-condition per-view variation was material but bounded: AUROC ranged from `0.8680` to
`0.9167`, balanced accuracy from `0.7673` to `0.8333`, macro F1 from `0.6954` to `0.7590`, and
positive F1 from `0.6500` to `0.7179`. This is why the legacy view is not treated as the sole
result.

### 7.2 Paired eight-view comparisons

Each subject's score was averaged across exactly eight views before the 10,000-repetition
paired bootstrap.

| Contrast | Correct-class margin delta [95% CI] | AUROC delta [95% CI] | Balanced-accuracy delta [95% CI] |
|---|---:|---:|---:|
| Real - silence | -0.0495 [-0.3689, 0.2778] | 0.0530 [-0.0077, 0.1337] | 0.0920 [-0.0278, 0.2273] |
| Real - audio shuffle | 0.3376 [-0.0263, 0.7404] | 0.0747 [-0.0088, 0.1855] | 0.1017 [-0.0152, 0.2336] |
| Real - same-class shuffle | -0.0140 [-0.3767, 0.3258] | 0.0108 [-0.0409, 0.0808] | 0.0054 [-0.0882, 0.1131] |
| Real - transcript shuffle | 1.9048 [0.5404, 3.2254] | 0.3377 [0.1286, 0.5395] | 0.2338 [0.0833, 0.3941] |
| Audio-only real - silence | 0.2839 [-0.3348, 0.9076] | 0.3009 [0.1573, 0.4216] | 0.2522 [0.1362, 0.3529] |
| Audio-only real - shuffle | 0.9914 [0.3394, 1.6366] | 0.3690 [0.1286, 0.5928] | 0.3052 [0.1288, 0.4784] |

Real versus silence changed 8/47 final predictions; real versus across-subject audio shuffle
changed 10/47; real versus same-class shuffle changed 7/47; and real versus transcript shuffle
changed 16/47. These changes rule out literal invariance, but prediction sensitivity alone is
not evidence that the checkpoint uses clinically valid aligned acoustics.

### 7.3 E0 decision

The eight-view decision is **weakly used**:

- `used` fails because the correct-class margin intervals for real-minus-silence and
  real-minus-shuffle both include zero, as do the corresponding AUROC and balanced-accuracy
  intervals.
- `ignored` is too strong because audio perturbations change subject margins and hard
  predictions, and the transcript-removed control distinguishes real from shuffled audio.
- `shortcut-dependent` is plausible but not proven. Same-class shuffling retains nearly all
  performance, and DAIC preprocessing kind is perfectly associated with the label
  (`random_segment` for label 0, `segment` for label 1), but the current artifacts cannot
  identify which acoustic/source property drives that retention.

The transcript gate passes decisively. The aligned-audio-benefit gate does not.

The transcript-removed conditions are controls on the same multimodal checkpoint, not a clean
separately trained audio-only model. They remove the user transcript block and change the
decision instruction to audio, while the checkpoint's original system prompt still mentions
transcript information.

## 8. Exact gradient-flow audit attempt

Canonical artifact:
`output_model/audits/e0/gradient_audit_legacy_k4_subject300_20260714/gradient_audit_failure.json`

The audit selected test subject `300`, retained the exact legacy K=4 audio bundle, constructed
the full supervised `Non-depressed<|im_end|>\n` target, enabled non-reentrant gradient
checkpointing, and attempted exactly one BF16 forward/backward with no optimizer step.

Result: `failed_cuda_oom` during `single_forward_backward`.

| Memory field | Bytes | GiB |
|---|---:|---:|
| Maximum allocated | 22,022,879,232 | 20.51 |
| Maximum reserved | 22,972,203,008 | 21.39 |
| Allocated at failure | 20,851,637,248 | 19.42 |
| Reserved at failure | 21,336,424,448 | 19.87 |
| Failed additional allocation request | about 2.03 GiB | 2.03 |

The failure artifact explicitly records `fallback_attempted: false`, with no K, audio, text,
precision, or model reduction. Exit code `2` is the intended protocol outcome for an exact OOM.

No gradient conclusion can be drawn from this attempt. In particular, it does not prove that
the audio input has zero gradient. The exact audit needs a GPU with more than 24 GiB usable
VRAM, preferably a 40/48/80 GiB device, or a separately preregistered offload protocol.

## 9. E1 eGeMAPSv02 acoustic ceiling

Canonical artifacts:
`outputs/baselines/daic_egemaps_v02_fixedk4/`

Protocol:

- 189 subjects and 756 selected chunks;
- four numerically ordered, evenly spaced chunks per subject;
- 88 eGeMAPSv02 functionals per chunk;
- subject mean plus population standard deviation = 176 features;
- regularized logistic regression, validation-selected `C=0.001`;
- one final fixed test evaluation, subject bootstrap intervals, and 100 within-partition
  shuffled-bundle repetitions.

| Metric | Test value | Subject-bootstrap 95% CI |
|---|---:|---:|
| Accuracy | 0.4894 | [0.3404, 0.6383] |
| AUROC | 0.5000 | [0.3179, 0.6768] |
| AUPRC | 0.3188 | [0.1921, 0.5715] |
| Balanced accuracy | 0.5335 | [0.3678, 0.6899] |
| Macro F1 | 0.4835 | [0.3374, 0.6170] |
| Positive F1 | 0.4286 | — |

Shuffled-audio empirical p-values were `0.4950` for AUROC, `0.4257` for balanced accuracy,
and `0.5050` for macro F1. This is a null fixed-split result.

## 10. E1 frozen WavLM Base+ acoustic ceiling

Canonical artifact:
`outputs/baselines/e1b_wavlm_base_plus_daic_layers678_full_final/runs/full_numeric_k4_layers678/results.json`

Protocol:

- `microsoft/wavlm-base-plus`, pinned revision
  `4c66d4806a428f2e922ccfa1a962776e232d487b`;
- four numerically ordered, evenly spaced chunks per subject;
- time mean of transformer layers 6, 7, and 8 per chunk;
- subject mean plus population standard deviation = 4,608 features;
- frozen encoder and regularized logistic regression, selected `C=0.0001`;
- 1,000 subject bootstraps and 100 within-partition shuffled-bundle repetitions.

| Metric | Test value | Subject-bootstrap 95% CI |
|---|---:|---:|
| Accuracy | 0.4894 | [0.3404, 0.6176] |
| AUROC | 0.5195 | [0.3228, 0.7104] |
| Average precision | 0.3888 | [0.2046, 0.6303] |
| Balanced accuracy | 0.5335 | [0.3785, 0.6813] |
| Macro F1 | 0.4835 | [0.3392, 0.6168] |
| Positive F1 | 0.4286 | [0.2222, 0.6047] |

Shuffled-audio empirical p-values were `0.3762` for AUROC, `0.2574` for average precision and
balanced accuracy, and `0.3762` for macro F1. Extraction processed 756 chunks / 22,680 seconds
of audio in 52.43 seconds, was exactly repeatable (`max_abs_diff=0`), and peaked at 0.872 GiB
allocated CUDA memory.

This is also a null fixed-split result. It only evaluates frozen layers 6/7/8 with mean/stat
pooling on preprocessed chunks. It does not rule out useful information in other layers,
fine-tuned encoders, participant-only speech, temporal/MIL pooling, or other representations.

## 11. Combined interpretation and next-run decision

Direct observations:

1. Transcript shuffling causes a large, robust performance loss.
2. Main multimodal real-versus-silence/shuffle audio gains are not robust under paired
   eight-view uncertainty.
3. Same-class audio shuffling retains nearly all multimodal performance.
4. Transcript-removed real audio is better than shuffled/silent audio on several metrics, but
   this is an altered prompt on the same multimodal checkpoint and has an unstable
   correct-margin benefit against silence.
5. eGeMAPS and one frozen WavLM representation do not beat their controls on the fixed split.

Inference: the checkpoint is primarily text-driven and is responsive to audio, but the local
evidence does not show a stable benefit from the correctly paired audio bundle. The retained
same-class performance makes shortcut use a serious alternative explanation.

Consequences for the proposed runs:

- Do not launch E2 Qwen2-Audio retraining yet. Its precondition—a stable audio integration
  signal or a convincing acoustic ceiling—was not met, and the exact backward does not fit on
  the 4090.
- Do not launch E3 learned fusion yet. A useful and repeatable audio score must first be shown.
- If more local evidence is desired, the next defensible work is cross-fold fixed-K eGeMAPS and
  WavLM, followed by a small subject-level MIL/attention pooling comparison. Folds and model
  selection must be preregistered before touching outcomes.
- A 40 GiB or larger GPU should have enough headroom for the exact gradient audit. The proposed
  smaller E2 rank-4/8 configurations were not memory-benchmarked on the 4090; the OOM applies
  only to the existing audited rank-16 checkpoint. E2's scientific gate remains closed
  regardless of that unresolved memory benchmark.
- E4 requires source data with reliable participant-only timestamps/diarization. The current
  manifests cannot repair alignment retrospectively.

## 12. Verification

Completed checks:

- `54/54` unit tests passed across deterministic R0 evaluation, E0 perturbations, numeric
  eight-view materialization, paired comparisons, aggregation, eGeMAPS, and gradient-audit
  orchestration.
- WavLM frozen-baseline no-model test passed.
- All changed/new Python files compiled with `py_compile`.
- `scripts/run_egemaps_ceiling.sh` passed `bash -n`.
- `git diff --check` passed.
- Independent view audit: 3,008/3,008 rows valid, all recomputed metrics/hashes matched, all
  eight completion provenance files present, zero detected errors.

The worktree is intentionally uncommitted. Existing unrelated user work was not deleted or
reset.

## 13. Canonical and noncanonical artifacts

Reportable canonical artifacts:

- Legacy E0: `output_model/audits/e0/direct_first_token_legacy_k4_all8_deterministic_20260714`
- Legacy paired E0: `output_model/audits/e0/direct_first_token_comparisons_20260714`
- Numeric eight-view E0:
  `output_model/audits/e0/numeric_balanced_k4_8views_deterministic_20260714`
- Numeric eight-view aggregate:
  `output_model/audits/e0/numeric_balanced_k4_8view_summary_20260714`
- Gradient OOM artifact:
  `output_model/audits/e0/gradient_audit_legacy_k4_subject300_20260714`
- eGeMAPS: `outputs/baselines/daic_egemaps_v02_fixedk4`
- WavLM:
  `outputs/baselines/e1b_wavlm_base_plus_daic_layers678_full_final/runs/full_numeric_k4_layers678`

Do not report these as final results:

- `output_model/audits/e0/direct_first_token_legacy_k4_all8_20260714`: interrupted
  nondeterministic attempt with no completion provenance;
- E0 one-subject preflight and exploratory scorer directories;
- `outputs/baselines/e1b_wavlm_base_plus_daic`: CPU smoke;
- `outputs/baselines/e1b_wavlm_base_plus_daic_layers678` and
  `outputs/baselines/e1b_wavlm_base_plus_daic_layers678_final`: smoke/intermediate roots.

No artifact was deleted while preparing this report.

## 14. Reproduction commands

Run from the repository root.

### eGeMAPS

```bash
PYTHONNOUSERSITE=1 \
PYTHON_BIN=/home/emre/miniconda3/envs/llmdep4090/bin/python \
OUTPUT_DIR=outputs/baselines/daic_egemaps_v02_fixedk4_reproduction \
bash scripts/run_egemaps_ceiling.sh
```

### WavLM Base+

```bash
PYTHONNOUSERSITE=1 \
/home/emre/miniconda3/envs/llmdep4090/bin/python \
  baselines/wavlm_frozen_subject_baseline.py \
  --model-dir /home/emre/models/WavLM-Base-Plus \
  --manifest outputs/manifests/daic_manifest.jsonl \
  --partitions outputs/splits/daic_subject_partitions.json \
  --output-root outputs/baselines/e1b_wavlm_base_plus_daic_layers678_reproduction \
  --run-label full_numeric_k4_layers678_reproduction \
  --device cuda \
  --chunks-per-subject 4 \
  --verify-repeat \
  --shuffle-repeats 100 \
  --bootstrap-repeats 1000
```

### Numeric-balanced E0 views

```bash
for view in 0 1 2 3 4 5 6 7; do
  CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  PYTHONHASHSEED=0 \
  PYTHONNOUSERSITE=1 \
  /home/emre/miniconda3/envs/llmdep4090/bin/python -m src.e0_perturbations \
    --checkpoint-dir \
      output_model/audits/e0/checkpoints/daic_posf1_tf_daic_audio_text_selmacrof1_tf/fold_0/best_model \
    --model-name-or-path /home/emre/models/Qwen2-Audio-7B-Instruct \
    --output-dir \
      output_model/audits/e0/numeric_balanced_k4_8views_deterministic_20260714/view_${view} \
    --partition test \
    --seed 1337 \
    --expected-k 4 \
    --view-family numeric_balanced_k4 \
    --view-index "${view}" \
    --include-candidate-likelihood \
    --progress-every 47
done
```

### Eight-view aggregation

```bash
PYTHONNOUSERSITE=1 \
/home/emre/miniconda3/envs/llmdep4090/bin/python -m scripts.summarize_e0_views \
  --view-root view_0=output_model/audits/e0/numeric_balanced_k4_8views_deterministic_20260714/view_0 \
  --view-root view_1=output_model/audits/e0/numeric_balanced_k4_8views_deterministic_20260714/view_1 \
  --view-root view_2=output_model/audits/e0/numeric_balanced_k4_8views_deterministic_20260714/view_2 \
  --view-root view_3=output_model/audits/e0/numeric_balanced_k4_8views_deterministic_20260714/view_3 \
  --view-root view_4=output_model/audits/e0/numeric_balanced_k4_8views_deterministic_20260714/view_4 \
  --view-root view_5=output_model/audits/e0/numeric_balanced_k4_8views_deterministic_20260714/view_5 \
  --view-root view_6=output_model/audits/e0/numeric_balanced_k4_8views_deterministic_20260714/view_6 \
  --view-root view_7=output_model/audits/e0/numeric_balanced_k4_8views_deterministic_20260714/view_7 \
  --output-dir output_model/audits/e0/numeric_balanced_k4_8view_summary_20260714 \
  --seed 1337
```

### Exact backward audit

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
PYTHONHASHSEED=0 \
PYTHONNOUSERSITE=1 \
/home/emre/miniconda3/envs/llmdep4090/bin/python scripts/e0_gradient_audit.py \
  --model-name-or-path /home/emre/models/Qwen2-Audio-7B-Instruct \
  --output-dir output_model/audits/e0/gradient_audit_legacy_k4_subject300_20260714
```

E0 condition roots, the eight-view aggregator, and the gradient audit enforce immutable
outputs. The two baseline implementations do not uniformly enforce non-overwrite. All
reproduction commands above therefore use, or should be changed to use, fresh output directory
names; do not target the canonical roots.

## 15. Missing evidence and limitations

- The exact gradient-flow result is unavailable because the full backward exceeds 24 GiB.
- Both acoustic ceilings use one fixed split and simple mean/statistics pooling. Cross-fold and
  MIL evidence is missing.
- The audio files lack sufficient local provenance to prove participant-only speech,
  timestamp alignment, interviewer exclusion, or chronological correctness.
- Same-class retention cannot distinguish depression acoustics from speaker, recorder,
  preprocessing-kind, silence, duration, or other class-correlated shortcuts.
- The E0 test set has only 47 subjects; paired intervals are therefore wide even after
  averaging eight views.
- No new training was performed. The report determines whether new training is justified; it
  does not estimate the performance of a repaired capacity-controlled model.

These limitations are why the conclusion is deliberately narrow: transcript dependence is
established, stable aligned-audio benefit is not, and the two tested fixed-split acoustic
representations are null.
