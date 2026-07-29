# D3TEC Hidden-State Classifier Report — 2026-07-29

## Protocol

Five predeclared outer folds were evaluated for the normalized D3TEC audio+text and audio-only checkpoints and the D3TEC text-only checkpoints. Gold labels remained external to Qwen inputs. Audio segments were weighted by the inverse number of segments in their response, rescaled to mean one. Predictions were aggregated segment → response → subject with 27 equal response votes. Text-only used one vector and prediction per subject.

The primary selection metric was pooled outer-fold Macro-F1. Optuna used 150 `standard_d6` trials, threshold 0.5, three subject-stratified inner folds, and sampler/model seed 1337. Inner seeds 7 and 2024 were used only for stability analysis and never selected by outer performance.

## Slurm execution

- Run registry: `outputs/d3tec_hidden_jobs/d3tec_hidden_norm_20260729T094602Z.tsv`
- Registered jobs: 72
- Terminal accounting rows: 72
- Retries: 0
- Aggregate allocated job runtime: 27h 45m 55s

| Stage | Jobs | Aggregate runtime |
|---|---:|---:|
| extraction | 15 | 0h 48m 53s |
| fixed_audit | 1 | 0h 00m 12s |
| fixed_heads | 15 | 0h 06m 59s |
| fixed_smoke | 1 | 0h 00m 18s |
| gpu_smoke | 1 | 0h 01m 18s |
| optuna | 35 | 26h 47m 16s |
| optuna_smoke_first | 1 | 0h 00m 22s |
| optuna_smoke_repeat | 1 | 0h 00m 21s |
| stability_audit | 1 | 0h 00m 04s |
| stage1_audit | 1 | 0h 00m 12s |

Remote retained artifact footprint:
- `audit_bytes`: 1020.53 KiB
- `hidden_classifier_bytes`: 739.86 MiB
- `hidden_feature_cache_bytes`: 162.66 MiB
- `registry_bytes`: 21.46 KiB
- `run_log_bytes`: 2.93 MiB

Registered source commits: `bbdfe0dc51f86586257b591e0d4fe0f659f87fca`, `f59892586346f284f70674b448b89943db6cd8df`, `ff6eb7017135178c026cdb358b1da504b8b91e7b`.

## Headline pooled results

| Modality | Head | Accuracy | PosF1 | Macro-F1 | Negative F1 | AUROC | Confusion matrix |
|---|---|---:|---:|---:|---:|---:|---|
| audio_only | LogReg raw | 0.564516 | 0.509091 | 0.558893 | 0.608696 | 0.584117 | `[[21, 12], [15, 14]]` |
| audio_only | XGBoost Optuna raw | 0.580645 | 0.500000 | 0.569444 | 0.638889 | 0.593521 | `[[23, 10], [16, 13]]` |
| audio_only | XGBoost fixed raw | 0.548387 | 0.500000 | 0.544118 | 0.588235 | 0.585162 | `[[20, 13], [15, 14]]` |
| audio_text | LogReg raw | 0.580645 | 0.480000 | 0.564324 | 0.648649 | 0.563218 | `[[24, 9], [17, 12]]` |
| audio_text | XGBoost Optuna raw | 0.564516 | 0.470588 | 0.550363 | 0.630137 | 0.555904 | `[[23, 10], [17, 12]]` |
| audio_text | XGBoost fixed raw | 0.532258 | 0.408163 | 0.510748 | 0.613333 | 0.557994 | `[[23, 10], [19, 10]]` |
| text_only | LogReg raw | 0.580645 | 0.566667 | 0.580208 | 0.593750 | 0.609195 | `[[19, 14], [12, 17]]` |
| text_only | XGBoost Optuna raw | 0.532258 | 0.472727 | 0.526219 | 0.579710 | 0.574190 | `[[20, 13], [16, 13]]` |
| text_only | XGBoost fixed raw | 0.483871 | 0.448276 | 0.481714 | 0.515152 | 0.561651 | `[[17, 16], [16, 13]]` |

## Headline fold metrics

| Modality | Head | Fold | Subjects | Accuracy | PosF1 | Macro-F1 | AUROC |
|---|---|---:|---:|---:|---:|---:|---:|
| audio_only | LogReg raw | 0 | 13 | 0.846154 | 0.857143 | 0.845238 | 1.000000 |
| audio_only | LogReg raw | 1 | 13 | 0.461538 | 0.222222 | 0.405229 | 0.476190 |
| audio_only | LogReg raw | 2 | 13 | 0.538462 | 0.500000 | 0.535714 | 0.523810 |
| audio_only | LogReg raw | 3 | 12 | 0.500000 | 0.250000 | 0.437500 | 0.333333 |
| audio_only | LogReg raw | 4 | 11 | 0.454545 | 0.500000 | 0.450000 | 0.600000 |
| audio_only | XGBoost Optuna raw | 0 | 13 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| audio_only | XGBoost Optuna raw | 1 | 13 | 0.538462 | 0.400000 | 0.512500 | 0.547619 |
| audio_only | XGBoost Optuna raw | 2 | 13 | 0.384615 | 0.333333 | 0.380952 | 0.428571 |
| audio_only | XGBoost Optuna raw | 3 | 12 | 0.500000 | 0.250000 | 0.437500 | 0.388889 |
| audio_only | XGBoost Optuna raw | 4 | 11 | 0.454545 | 0.400000 | 0.450000 | 0.566667 |
| audio_only | XGBoost fixed raw | 0 | 13 | 0.846154 | 0.857143 | 0.845238 | 1.000000 |
| audio_only | XGBoost fixed raw | 1 | 13 | 0.538462 | 0.400000 | 0.512500 | 0.547619 |
| audio_only | XGBoost fixed raw | 2 | 13 | 0.384615 | 0.333333 | 0.380952 | 0.452381 |
| audio_only | XGBoost fixed raw | 3 | 12 | 0.500000 | 0.250000 | 0.437500 | 0.416667 |
| audio_only | XGBoost fixed raw | 4 | 11 | 0.454545 | 0.500000 | 0.450000 | 0.533333 |
| audio_text | LogReg raw | 0 | 13 | 0.923077 | 0.923077 | 0.923077 | 0.976190 |
| audio_text | LogReg raw | 1 | 13 | 0.461538 | 0.222222 | 0.405229 | 0.428571 |
| audio_text | LogReg raw | 2 | 13 | 0.461538 | 0.363636 | 0.448485 | 0.547619 |
| audio_text | LogReg raw | 3 | 12 | 0.500000 | 0.250000 | 0.437500 | 0.250000 |
| audio_text | LogReg raw | 4 | 11 | 0.545455 | 0.444444 | 0.529915 | 0.600000 |
| audio_text | XGBoost Optuna raw | 0 | 13 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| audio_text | XGBoost Optuna raw | 1 | 13 | 0.461538 | 0.363636 | 0.448485 | 0.571429 |
| audio_text | XGBoost Optuna raw | 2 | 13 | 0.384615 | 0.200000 | 0.350000 | 0.404762 |
| audio_text | XGBoost Optuna raw | 3 | 12 | 0.500000 | 0.250000 | 0.437500 | 0.250000 |
| audio_text | XGBoost Optuna raw | 4 | 11 | 0.454545 | 0.400000 | 0.450000 | 0.566667 |
| audio_text | XGBoost fixed raw | 0 | 13 | 0.923077 | 0.909091 | 0.921212 | 0.928571 |
| audio_text | XGBoost fixed raw | 1 | 13 | 0.384615 | 0.200000 | 0.350000 | 0.571429 |
| audio_text | XGBoost fixed raw | 2 | 13 | 0.384615 | 0.200000 | 0.350000 | 0.404762 |
| audio_text | XGBoost fixed raw | 3 | 12 | 0.500000 | 0.250000 | 0.437500 | 0.250000 |
| audio_text | XGBoost fixed raw | 4 | 11 | 0.454545 | 0.400000 | 0.450000 | 0.533333 |
| text_only | LogReg raw | 0 | 13 | 0.615385 | 0.444444 | 0.575163 | 0.833333 |
| text_only | LogReg raw | 1 | 13 | 0.461538 | 0.363636 | 0.448485 | 0.547619 |
| text_only | LogReg raw | 2 | 13 | 0.615385 | 0.545455 | 0.606061 | 0.452381 |
| text_only | LogReg raw | 3 | 12 | 0.666667 | 0.750000 | 0.625000 | 0.861111 |
| text_only | LogReg raw | 4 | 11 | 0.545455 | 0.615385 | 0.529915 | 0.600000 |
| text_only | XGBoost Optuna raw | 0 | 13 | 0.692308 | 0.500000 | 0.638889 | 0.726190 |
| text_only | XGBoost Optuna raw | 1 | 13 | 0.461538 | 0.363636 | 0.448485 | 0.452381 |
| text_only | XGBoost Optuna raw | 2 | 13 | 0.461538 | 0.222222 | 0.405229 | 0.452381 |
| text_only | XGBoost Optuna raw | 3 | 12 | 0.416667 | 0.533333 | 0.377778 | 0.611111 |
| text_only | XGBoost Optuna raw | 4 | 11 | 0.636364 | 0.666667 | 0.633333 | 0.600000 |
| text_only | XGBoost fixed raw | 0 | 13 | 0.615385 | 0.444444 | 0.575163 | 0.833333 |
| text_only | XGBoost fixed raw | 1 | 13 | 0.461538 | 0.363636 | 0.448485 | 0.488095 |
| text_only | XGBoost fixed raw | 2 | 13 | 0.461538 | 0.363636 | 0.448485 | 0.380952 |
| text_only | XGBoost fixed raw | 3 | 12 | 0.416667 | 0.533333 | 0.377778 | 0.527778 |
| text_only | XGBoost fixed raw | 4 | 11 | 0.454545 | 0.500000 | 0.450000 | 0.600000 |

## Stability

The maximum pooled Macro-F1 range across the two pilot modalities was 0.021698, against the predeclared 0.03 gate. Conditional audio-only expansion: **not triggered**.

- `audio_text_normalized`: range 0.018817; inner seed 7=0.555142, inner seed 1337=0.550363, inner seed 2024=0.536325
- `text_only`: range 0.021698; inner seed 7=0.526219, inner seed 1337=0.526219, inner seed 2024=0.547917

## Controls and baselines

- All-positive baseline PosF1: 58/91 = 0.637363.
- Female-positive sex-rule Macro-F1: 0.595828.
- `audio_only` / `majority_class`: Macro-F1 0.347368.
- `audio_only` / `xgb_raw_shuffled_labels`: Macro-F1 0.530811.
- `audio_text` / `majority_class`: Macro-F1 0.347368.
- `audio_text` / `xgb_raw_shuffled_labels`: Macro-F1 0.536325.
- `text_only` / `majority_class`: Macro-F1 0.347368.
- `text_only` / `xgb_raw_shuffled_labels`: Macro-F1 0.602564.

## Gender-stratified headline errors

| Modality | Head | Group | Subjects | Errors | Error rate |
|---|---|---|---:|---:|---:|
| audio_only | LogReg raw | Female | 36 | 20 | 0.555556 |
| audio_only | LogReg raw | Male | 26 | 7 | 0.269231 |
| audio_only | XGBoost Optuna raw | Female | 36 | 18 | 0.500000 |
| audio_only | XGBoost Optuna raw | Male | 26 | 8 | 0.307692 |
| audio_only | XGBoost fixed raw | Female | 36 | 20 | 0.555556 |
| audio_only | XGBoost fixed raw | Male | 26 | 8 | 0.307692 |
| audio_text | LogReg raw | Female | 36 | 18 | 0.500000 |
| audio_text | LogReg raw | Male | 26 | 8 | 0.307692 |
| audio_text | XGBoost Optuna raw | Female | 36 | 20 | 0.555556 |
| audio_text | XGBoost Optuna raw | Male | 26 | 7 | 0.269231 |
| audio_text | XGBoost fixed raw | Female | 36 | 21 | 0.583333 |
| audio_text | XGBoost fixed raw | Male | 26 | 8 | 0.307692 |
| text_only | LogReg raw | Female | 36 | 14 | 0.388889 |
| text_only | LogReg raw | Male | 26 | 12 | 0.461538 |
| text_only | XGBoost Optuna raw | Female | 36 | 16 | 0.444444 |
| text_only | XGBoost Optuna raw | Male | 26 | 13 | 0.500000 |
| text_only | XGBoost fixed raw | Female | 36 | 17 | 0.472222 |
| text_only | XGBoost fixed raw | Male | 26 | 15 | 0.576923 |

## Audit and provenance

The local acceptance audit passed for 23 condition/head combinations, exactly 62 pooled held-out subjects ({'0': 33, '1': 29}), subject-disjoint inner and outer partitions, 27 response predictions per audio subject, complete response-weight audits, and the expected manifest, split, checkpoint, cache, model, prediction, metric, and provenance artifacts.

Gender-stratified error counts and rates are stored per result in the acceptance audit: `outputs/d3tec_hidden_audits/d3tec_hidden_norm_20260729T094602Z/stage1_acceptance.json`.

## Limitations

- The panel contains 62 participants, so fold and subgroup estimates remain noisy.
- Inner-seed stability does not quantify checkpoint-training seed variability.
- The gender analysis is descriptive and not evidence of a causal relationship.
- Hidden-state heads reuse representations from supervised LoRA checkpoints; they are downstream probes, not independently trained foundation models.

## Slurm accounting appendix

| Job ID | Stage | Modality | Fold | Experiment | State | Exit | Elapsed |
|---:|---|---|---:|---|---|---:|---:|
| 43954678 | gpu_smoke | audio_text | 0 | `extraction_metadata` | COMPLETED | 0:0 | 00:01:18 |
| 43954679 | fixed_smoke | audio_text | 0 | `fixed_controls` | COMPLETED | 0:0 | 00:00:18 |
| 43954680 | optuna_smoke_first | audio_text | 0 | `xgb_optuna_raw_smoke_t2_seed1337` | COMPLETED | 0:0 | 00:00:22 |
| 43954681 | optuna_smoke_repeat | audio_text | 0 | `xgb_optuna_raw_smoke_t2_seed1337` | COMPLETED | 0:0 | 00:00:21 |
| 43955378 | extraction | audio_text | 0 | `hidden_cache` | COMPLETED | 0:0 | 00:04:39 |
| 43955379 | fixed_heads | audio_text | 0 | `fixed_matrix` | COMPLETED | 0:0 | 00:00:35 |
| 43955380 | extraction | audio_text | 1 | `hidden_cache` | COMPLETED | 0:0 | 00:04:04 |
| 43955381 | fixed_heads | audio_text | 1 | `fixed_matrix` | COMPLETED | 0:0 | 00:00:35 |
| 43955382 | extraction | audio_text | 2 | `hidden_cache` | COMPLETED | 0:0 | 00:04:07 |
| 43955383 | fixed_heads | audio_text | 2 | `fixed_matrix` | COMPLETED | 0:0 | 00:00:35 |
| 43955384 | extraction | audio_text | 3 | `hidden_cache` | COMPLETED | 0:0 | 00:04:10 |
| 43955385 | fixed_heads | audio_text | 3 | `fixed_matrix` | COMPLETED | 0:0 | 00:00:39 |
| 43955386 | extraction | audio_text | 4 | `hidden_cache` | COMPLETED | 0:0 | 00:04:07 |
| 43955387 | fixed_heads | audio_text | 4 | `fixed_matrix` | COMPLETED | 0:0 | 00:00:35 |
| 43955388 | extraction | audio_only | 0 | `hidden_cache` | COMPLETED | 0:0 | 00:04:17 |
| 43955389 | fixed_heads | audio_only | 0 | `fixed_matrix` | COMPLETED | 0:0 | 00:00:39 |
| 43955390 | extraction | audio_only | 1 | `hidden_cache` | COMPLETED | 0:0 | 00:04:20 |
| 43955391 | fixed_heads | audio_only | 1 | `fixed_matrix` | COMPLETED | 0:0 | 00:00:39 |
| 43955392 | extraction | audio_only | 2 | `hidden_cache` | COMPLETED | 0:0 | 00:04:20 |
| 43955393 | fixed_heads | audio_only | 2 | `fixed_matrix` | COMPLETED | 0:0 | 00:00:39 |
| 43955394 | extraction | audio_only | 3 | `hidden_cache` | COMPLETED | 0:0 | 00:04:19 |
| 43955395 | fixed_heads | audio_only | 3 | `fixed_matrix` | COMPLETED | 0:0 | 00:00:39 |
| 43955396 | extraction | audio_only | 4 | `hidden_cache` | COMPLETED | 0:0 | 00:03:59 |
| 43955397 | fixed_heads | audio_only | 4 | `fixed_matrix` | COMPLETED | 0:0 | 00:00:36 |
| 43955398 | extraction | text_only | 0 | `hidden_cache` | COMPLETED | 0:0 | 00:00:59 |
| 43955399 | fixed_heads | text_only | 0 | `fixed_matrix` | COMPLETED | 0:0 | 00:00:08 |
| 43955400 | extraction | text_only | 1 | `hidden_cache` | COMPLETED | 0:0 | 00:01:23 |
| 43955401 | fixed_heads | text_only | 1 | `fixed_matrix` | COMPLETED | 0:0 | 00:00:06 |
| 43955402 | extraction | text_only | 2 | `hidden_cache` | COMPLETED | 0:0 | 00:01:23 |
| 43955403 | fixed_heads | text_only | 2 | `fixed_matrix` | COMPLETED | 0:0 | 00:00:13 |
| 43955404 | extraction | text_only | 3 | `hidden_cache` | COMPLETED | 0:0 | 00:01:23 |
| 43955405 | fixed_heads | text_only | 3 | `fixed_matrix` | COMPLETED | 0:0 | 00:00:13 |
| 43955406 | extraction | text_only | 4 | `hidden_cache` | COMPLETED | 0:0 | 00:01:23 |
| 43955407 | fixed_heads | text_only | 4 | `fixed_matrix` | COMPLETED | 0:0 | 00:00:08 |
| 43956257 | fixed_audit | all | all | `fixed_acceptance` | COMPLETED | 0:0 | 00:00:12 |
| 43956307 | optuna | audio_text | 0 | `xgb_optuna_raw_t150_d6_seed1337_inner1337` | COMPLETED | 0:0 | 00:55:28 |
| 43956308 | optuna | audio_text | 1 | `xgb_optuna_raw_t150_d6_seed1337_inner1337` | COMPLETED | 0:0 | 02:11:54 |
| 43956309 | optuna | audio_text | 2 | `xgb_optuna_raw_t150_d6_seed1337_inner1337` | COMPLETED | 0:0 | 01:44:54 |
| 43956310 | optuna | audio_text | 3 | `xgb_optuna_raw_t150_d6_seed1337_inner1337` | COMPLETED | 0:0 | 00:58:24 |
| 43956311 | optuna | audio_text | 4 | `xgb_optuna_raw_t150_d6_seed1337_inner1337` | COMPLETED | 0:0 | 01:47:03 |
| 43956312 | optuna | audio_only | 0 | `xgb_optuna_raw_t150_d6_seed1337_inner1337` | COMPLETED | 0:0 | 02:29:21 |
| 43956313 | optuna | audio_only | 1 | `xgb_optuna_raw_t150_d6_seed1337_inner1337` | COMPLETED | 0:0 | 00:51:56 |
| 43956314 | optuna | audio_only | 2 | `xgb_optuna_raw_t150_d6_seed1337_inner1337` | COMPLETED | 0:0 | 00:34:51 |
| 43956315 | optuna | audio_only | 3 | `xgb_optuna_raw_t150_d6_seed1337_inner1337` | COMPLETED | 0:0 | 01:45:48 |
| 43956316 | optuna | audio_only | 4 | `xgb_optuna_raw_t150_d6_seed1337_inner1337` | COMPLETED | 0:0 | 00:27:20 |
| 43956317 | optuna | text_only | 0 | `xgb_optuna_raw_t150_d6_seed1337_inner1337` | COMPLETED | 0:0 | 00:02:35 |
| 43956318 | optuna | text_only | 1 | `xgb_optuna_raw_t150_d6_seed1337_inner1337` | COMPLETED | 0:0 | 00:03:17 |
| 43956319 | optuna | text_only | 2 | `xgb_optuna_raw_t150_d6_seed1337_inner1337` | COMPLETED | 0:0 | 00:02:39 |
| 43956320 | optuna | text_only | 3 | `xgb_optuna_raw_t150_d6_seed1337_inner1337` | COMPLETED | 0:0 | 00:04:05 |
| 43956321 | optuna | text_only | 4 | `xgb_optuna_raw_t150_d6_seed1337_inner1337` | COMPLETED | 0:0 | 00:02:36 |
| 43964203 | stage1_audit | all | all | `stage1_acceptance` | COMPLETED | 0:0 | 00:00:12 |
| 43964345 | optuna | audio_text | 0 | `xgb_optuna_raw_t150_d6_seed1337_inner7` | COMPLETED | 0:0 | 00:31:58 |
| 43964346 | optuna | audio_text | 1 | `xgb_optuna_raw_t150_d6_seed1337_inner7` | COMPLETED | 0:0 | 01:36:08 |
| 43964347 | optuna | audio_text | 2 | `xgb_optuna_raw_t150_d6_seed1337_inner7` | COMPLETED | 0:0 | 00:49:10 |
| 43964348 | optuna | audio_text | 3 | `xgb_optuna_raw_t150_d6_seed1337_inner7` | COMPLETED | 0:0 | 01:46:41 |
| 43964349 | optuna | audio_text | 4 | `xgb_optuna_raw_t150_d6_seed1337_inner7` | COMPLETED | 0:0 | 01:12:11 |
| 43964350 | optuna | text_only | 0 | `xgb_optuna_raw_t150_d6_seed1337_inner7` | COMPLETED | 0:0 | 00:01:49 |
| 43964351 | optuna | text_only | 1 | `xgb_optuna_raw_t150_d6_seed1337_inner7` | COMPLETED | 0:0 | 00:02:57 |
| 43964352 | optuna | text_only | 2 | `xgb_optuna_raw_t150_d6_seed1337_inner7` | COMPLETED | 0:0 | 00:03:35 |
| 43964353 | optuna | text_only | 3 | `xgb_optuna_raw_t150_d6_seed1337_inner7` | COMPLETED | 0:0 | 00:02:12 |
| 43964354 | optuna | text_only | 4 | `xgb_optuna_raw_t150_d6_seed1337_inner7` | COMPLETED | 0:0 | 00:02:52 |
| 43964355 | optuna | audio_text | 0 | `xgb_optuna_raw_t150_d6_seed1337_inner2024` | COMPLETED | 0:0 | 01:01:29 |
| 43964356 | optuna | audio_text | 1 | `xgb_optuna_raw_t150_d6_seed1337_inner2024` | COMPLETED | 0:0 | 01:36:15 |
| 43964357 | optuna | audio_text | 2 | `xgb_optuna_raw_t150_d6_seed1337_inner2024` | COMPLETED | 0:0 | 01:26:04 |
| 43964358 | optuna | audio_text | 3 | `xgb_optuna_raw_t150_d6_seed1337_inner2024` | COMPLETED | 0:0 | 01:09:45 |
| 43964359 | optuna | audio_text | 4 | `xgb_optuna_raw_t150_d6_seed1337_inner2024` | COMPLETED | 0:0 | 01:08:52 |
| 43964360 | optuna | text_only | 0 | `xgb_optuna_raw_t150_d6_seed1337_inner2024` | COMPLETED | 0:0 | 00:02:27 |
| 43964361 | optuna | text_only | 1 | `xgb_optuna_raw_t150_d6_seed1337_inner2024` | COMPLETED | 0:0 | 00:03:23 |
| 43964362 | optuna | text_only | 2 | `xgb_optuna_raw_t150_d6_seed1337_inner2024` | COMPLETED | 0:0 | 00:02:20 |
| 43964363 | optuna | text_only | 3 | `xgb_optuna_raw_t150_d6_seed1337_inner2024` | COMPLETED | 0:0 | 00:02:14 |
| 43964364 | optuna | text_only | 4 | `xgb_optuna_raw_t150_d6_seed1337_inner2024` | COMPLETED | 0:0 | 00:02:43 |
| 43965263 | stability_audit | all | - | `stability_gate` | COMPLETED | 0:0 | 00:00:04 |
