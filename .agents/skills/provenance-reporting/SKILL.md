---
name: provenance-reporting
description: Validate and report this repository's experiment results with complete provenance, canonical headline semantics, local artifact verification, qualified aggregation and evaluation views, workbook generation, ambiguity rejection, hidden-classifier conventions, and resubmission history. Use before showing any metric, writing a result or PR summary, selecting a best run, updating reports/workbooks, or validating results synchronized from MN5.
---

# Report only traceable results

Treat a bare number as a bug. Read the mandatory reporting section in `AGENTS.md` before reporting results, and use current local evidence rather than copied prose or dated examples.

## Require a complete provenance chain

For every reported metric, provide or link:

- experiment group/logical run or run name, attempt ID, and fold;
- branch and full Git SHA when available;
- authoritative `run_config.yaml` path, including configuration, manifest, and split hashes plus seed;
- evaluated checkpoint role and path, normally `best_model`;
- metric name and namespace;
- evaluation backend and view;
- aggregation convention, such as fold-mean or pooled subject-level;
- local metrics/predictions/audit artifact path;
- evaluation ID and relevant job/resubmission IDs when tracked.

If any required link is absent, locate it, reject the value as ambiguous, mark it legacy-unmigrated, or label it `MN5-only, not locally verifiable` as appropriate. Never infer or invent it.

Use this compact presentation pattern:

```text
metric=value; dataset/modality; group or run; attempt; fold(s); checkpoint;
namespace; backend; view; aggregation; config path + hashes; local evidence path;
evaluation ID; Slurm/resubmission IDs; verification status
```

## Verify headline semantics

- Recompute or verify synced headline values locally before writing them anywhere.
- Use `headline/binary_strict_*`; count invalid output as wrong and ignore `valid_only_*` for headlines.
- Use `original_teacher_forced` for the canonical recipe and positive-F1 checkpoint selection (`inner_val_positive_f1`, mode `max`).
- Do not report AUROC from teacher-forced hard labels.
- Use `best_model`; never silently substitute `last_model`.
- State fold-mean versus pooled subject-level aggregation explicitly.

Resolve rather than inherit known ambiguities:

- DAIC fixed-K versus full-coverage K4 evaluation;
- D3TEC normalized versus rotary recipes;
- merged retrain versus smoke identities for each modality;
- Optuna and Subject-OS not-run states, which remain blank;
- any multiple plausible metric files or evaluation records.

## Use the registry and generated reports

Prefer qualified registry queries and deterministic reports over manual transcription:

```bash
python tools/exp.py provenance <metric-id>
python tools/exp.py best --dataset <d> --metric <m> --namespace <ns> --backend <b> --view <v> --aggregation <a>
python tools/generate_run_report.py --attempt-id <id> --fold <n>
python tools/generate_group_report.py --attempts <csv> --metric-name <m> --namespace <ns> --backend <b> --view <v> --aggregation <a>
```

Refuse underqualified best-run queries and incompatible group aggregation. Leave researcher conclusions blank unless the researcher supplies them.

## Generate the canonical workbook

Generate `depression_results_clean.xlsx` only with `scripts/build_clean_workbook.py`; never hand-edit cells.

```bash
python scripts/build_clean_workbook.py
python scripts/build_clean_workbook.py --detailed
python tools/export_selected_results.py --selection <selection.yaml>
python scripts/build_clean_workbook.py --validate-selected <selected-results.json>
```

Add new values to the script's source tables with provenance, regenerate, and validate selections. Preserve blank not-run fields. Represent missing records as `legacy-unmigrated`, never zero.

## Handle specialized and failed runs

- For hidden classifiers, cite the `configs/features` matrix, classifier metadata, pooled five-fold subject-level convention when used, local fold artifacts, and all job IDs.
- For translated or merged layouts, use the experiment-tracking adapters and preserve the actual source/config identity.
- For reruns, cite failed job IDs, failure evidence, replacement attempt and job IDs, and the successful local artifact.
- Do not treat a run name or log filename as configuration evidence.
- Do not hand-edit `.provenance`; it is regenerated during cluster synchronization.

## Parallel workflow — deployment and group provenance

- Include deployment ID/hash (`deployments/<id>/deployment.json`, `source_manifest_sha256`) and stacked source ancestry (parent branch/SHA) in provenance
- Require explicit comparison group and attempt list for winner claims (`--group <id> --attempts <csv>` with full qualifiers: dataset, metric, namespace, backend, view, aggregation); refuse underqualified or global `best` for winner selection

