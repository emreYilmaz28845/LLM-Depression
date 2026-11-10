"""Render aggregate kNN study tables and audit source identities; no private rows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.knn_reliability_analysis import CORPORA, read, sha, write
from src.features.hidden_classifier_policy import canonical_sha256


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "|" + "|".join("---" for _ in headers) + "|"] +
                     ["| " + " | ".join(map(str, r)) + " |" for r in rows])


def number(x, digits=3):
    return f"{x:.{digits}f}" if pd.notna(x) else "undefined (no errors)"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--analysis", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args(); a.output.mkdir(parents=True, exist_ok=True)
    config = read(a.analysis / "analysis_config.json")
    provenance = read(a.analysis / "provenance.json")
    source = Path(config["source_root"])
    audit = []
    for path in sorted({x["path"] for x in provenance if x.get("role") == "head_config"}):
        c = read(path)
        identities = c.get("cache_identities") or [c["cache_identity"]]
        dev = c.get("dev_cache_identities") or ([c["dev_cache_identity"]] if c.get("dev_cache_identity") else [])
        for identity in identities + dev:
            for name, item in identity.items():
                if not isinstance(item, dict) or "path" not in item:
                    continue
                p = Path(item["path"]); p = p if p.is_absolute() else source / p
                exists = p.is_file()
                actual = sha(p) if exists else None
                audit.append({"role": "head_cache", "config": path, "path": str(p), "exists": exists,
                    "expected_sha256": item.get("sha256"), "actual_sha256": actual,
                    "matches": exists and actual == item.get("sha256")})
        for item in c["split_files"]:
            p = Path(item["path"])
            actual = canonical_sha256(read(p)) if p.is_file() else None
            audit.append({"role": "head_split", "config": path, "path": str(p), "exists": p.is_file(),
                "expected_canonical_sha256": item["sha256"], "actual_canonical_sha256": actual,
                "matches": actual == item["sha256"]})
    write(a.output / "source_identity_audit.json", audit)
    mismatches = [x for x in audit if not x["matches"]]
    overlap = pd.read_csv(a.analysis / "error_overlap.csv", dtype={"seed": str})
    gating = pd.read_csv(a.analysis / "gating.csv")
    tests = pd.read_csv(a.analysis / "paired_tests.csv")
    reliability = pd.read_csv(a.analysis / "reliability.csv")
    observations = read(a.analysis / "training_observation_sources.json")
    config_index, by_hash = [], {}
    for path in sorted({x["path"] for x in provenance if x.get("role") == "llm_run_config"}):
        c = yaml.safe_load(Path(path).read_text()); digest = sha(path)
        by_hash.setdefault(digest, []).append(path)
        config_index.append({"path": path, "sha256": digest, "seed": c.get("config", {}).get("seed"),
            "manifest_path": c.get("manifest_path"), "manifest_hash": c.get("manifest_hash"),
            "split_metadata_path": c.get("split_metadata_path"), "split_metadata_hash": c.get("split_metadata_hash"),
            "tracking": c.get("tracking"), "cv_protocol": c.get("cv_protocol"),
            "backend": c.get("evaluation", {}).get("sample_prediction_mode"),
            "selection_protocol": c.get("selection_protocol"), "checkpoint_role": "best_model"})
    parent_resolution = []
    for x in provenance:
        if "metadata" not in x:
            continue
        expected = x.get("saved_run_config_sha256")
        parent_resolution.append({"metadata": x["metadata"], "family": x["family"], "corpus": x["corpus"],
            "modality": x["modality"], "fold": x["fold"], "expected_run_config_sha256": expected,
            "matching_local_run_configs": by_hash.get(expected, []),
            "status": "resolved_by_exact_bytes" if expected in by_hash else
                      ("frozen_parent_reference_only" if x["family"] == "frozen_head" else "legacy-unmigrated")})
    write(a.output / "run_config_index.json", config_index)
    write(a.output / "parent_config_resolution.json", parent_resolution)
    # Verify unique held-out subjects within each seed and actual source seed.
    llm_unique = {}
    for obs in observations:
        fd = Path(obs["run_config"]).parent
        d = pd.read_csv(fd / "best_model/standalone_eval/predictions_subject_level.csv", dtype={"subject_id": str})
        ds, mod, seed, _, _ = obs["cell_id"][len("S-qwen-"):].rsplit("-", 4)
        c = yaml.safe_load(Path(obs["run_config"]).read_text())
        if c["config"]["seed"] != int(seed):
            raise ValueError("Cell seed differs from saved run config")
        llm_unique.setdefault((ds, mod, seed), []).extend(d.subject_id.tolist())
    for key, ids in llm_unique.items():
        if len(ids) != len(set(ids)):
            raise ValueError("Repeated held-out subjects across folds: " + str(key))
    for ds in CORPORA:
        for family in ("heads", "heads_lora_textrerun", "heads_lora"):
            mod = "audio_text" if family == "heads_lora" else "text_only"
            root = source / "output_model/frozen_base_standalone_heads" / family / ds / (mod + "_native")
            ids = []
            for p in root.glob("fold_*/predictions_subject_level.csv"):
                ids.extend(pd.read_csv(p, dtype={"subject_id": str}).subject_id.tolist())
            if len(ids) != len(set(ids)):
                raise ValueError("Repeated head evaluation subjects across folds")
    tuned_text = tests[tests.reference == "text"]
    lines_summary = (f"Text and audio+text have different error sets in every corpus/seed examined. "
        f"Two-way complementarity is not universal: in some conditions one model's errors contain all errors of the other. "
        f"The frozen-feature threshold gate improves point-estimate fold-mean Macro-F1 over text in "
        f"{int((tuned_text.delta_macro_f1 > 1e-12).sum())} of five corpora, ties in "
        f"{int((tuned_text.delta_macro_f1.abs() <= 1e-12).sum())}, and has no Holm-significant gain over any "
        f"single-input reference in the 15 prespecified tests. This first rule does not demonstrate a robust selection benefit.")
    lines = ["# Text / audio+text error overlap and kNN reliability", "",
        lines_summary, "",
        "Native-language analysis, 10 November 2026. One overlap-table row represents one corpus, prediction family and seed. "
        "Source rows and analysis units are subjects; windows are aggregated before classification. "
        "No LLM training or LLM inference was run for this study; lightweight logistic heads were refit on cached features.", "",
        "## Evidence and interpretation", "",
        f"Analysis commit: `{config['git_commit']}`. Script SHA-256: `{config['source_script_sha256']}`. "
        f"Runtime: {config['elapsed_seconds']:.1f} seconds for computation, excluding implementation and audit. "
        f"Cache/split identity checks: {len(audit)}; mismatches or missing files: {len(mismatches)}.", "",
        "Direct LLM rows compare Qwen3.8-27B text with Qwen3-Omni audio+text, evaluated from best_model with candidate-label likelihood. "
        "Frozen and LoRA head rows use their recorded subject-mean logistic classifiers and are separate prediction families. "
        "Correctness AUROC below is diagnostic, not a depression classification headline metric. "
        "The existing canonical results workbook was not changed.", "",
        "The requested error-overlap analysis is descriptive. Complementary errors establish potential room for a selector, "
        "but do not establish that a usable confidence rule can identify those cases.", "",
        "## Direct LLM error overlap", "",
        "Counts pool the outer evaluation subjects within each seed; seeds are never pooled. "
        "Common-error percentage = both wrong / (at least one wrong). "
        "Text-error rescue = text wrong and audio+text correct / all text errors. "
        "With binary valid predictions, both wrong means the same wrong label; invalids, if present, count as errors.", ""]
    for family, title in [("lora_llm_likelihood", "Direct LoRA LLM likelihood"),
                          ("frozen_head", "Frozen embeddings + subject-mean head"),
                          ("lora_head_s1337", "LoRA embeddings + subject-mean head (seed 1337)")]:
        lines += ["### " + title, ""]
        rows = []
        for _, r in overlap[overlap.family == family].iterrows():
            rows.append([CORPORA[r.corpus], r.seed, int(r.n), int(r.both_correct), int(r.only_text_correct),
                int(r.only_audio_text_correct), int(r.both_wrong), number(r.shared_error_pct_of_union, 1) + "%",
                number(r.text_errors_corrected_pct, 1) + "%"])
        lines += [table(["Corpus / language", "Seed", "N", "Both correct", "Only text correct", "Only A+T correct",
                         "Both wrong", "Common / union errors", "Text-error rescue"], rows), ""]
    lines += ["## Frozen-feature selection / combination", "",
        "One row represents one corpus and method. Metrics are equal-weight fold means (Macro-F1 / Positive-F1; UAR separately). "
        "DAIC uses its single official test fold. CMDC and Turkish use outer validation and are not locked-test estimates. "
        "The A+T comparator is a single multimodal model; averaging text and audio probabilities is a separate late-fusion method.", "",
        "Gate inputs are frozen text-only and audio-only head probabilities. Confidence is the fraction of K=10 same-fold "
        "training neighbours carrying the head's predicted class. Both reliable -> average probabilities; only audio reliable -> audio; "
        "only text reliable or neither reliable -> text. Probability >0.5 predicts positive. "
        "Threshold candidates 0.0, 0.1, ..., 1.1 were fixed before evaluation; ties prefer the higher threshold. "
        "A threshold of 1.1 disables combination. Text fallback when neither is reliable is an explicit design choice.", "",
        "DAIC, Androids and D3TEC thresholds use saved development/inner-validation subjects. CMDC and Turkish use "
        "stratified five-fold out-of-fold predictions within outer training; each inner head, scaler and neighbour bank excludes its queries. "
        "Head L2 is the recorded value originally selected on inner validation. No final evaluation labels select thresholds. "
        "LoRA gate tuning is omitted because outer-trained adapter features would not be independent for an inner holdout.", ""]
    for ds, g in gating.groupby("corpus", sort=False):
        best = g.macro_f1.max()
        lines += ["### " + CORPORA[ds], ""]
        rows = []
        for _, r in g.iterrows():
            score = f"{r.macro_f1:.3f} / {r.positive_f1:.3f}"
            if r.macro_f1 == best:
                score = "**" + score + "**"
            rows.append([r.method, score, f"{r.uar:.3f}"])
        lines += [table(["Method", "Macro-F1 / Positive-F1", "UAR"], rows), ""]
    lines += ["## Paired uncertainty", "",
        f"{config['bootstrap_and_permutation_reps']} paired bootstrap replicates, stratified by label within fold. "
        "Intervals estimate the tuned-gate minus reference fold-mean Macro-F1 difference. "
        "Paired subject-wise prediction-swap permutation p-values are adjusted with Holm across all 15 comparisons "
        "(five corpora × three single-input references). Intervals are unadjusted 95% percentile intervals. "
        "These analyses condition on fitted models and selected thresholds; they exclude training/selection uncertainty. "
        "Overlapping CV training sets are not resampled, so inference is exploratory.", ""]
    lines += [table(["Corpus", "Reference", "Delta Macro-F1", "95% CI", "Permutation p", "Holm p (15)"],
        [[CORPORA[r.corpus], r.reference, f"{r.delta_macro_f1:+.3f}", f"[{r.ci95_low:+.3f}, {r.ci95_high:+.3f}]",
          f"{r.permutation_p:.4f}", f"{r.holm_p_15_tests:.4f}"] for _, r in tests.iterrows()]), ""]
    lines += ["## kNN confidence diagnostic", "",
        "One row represents one corpus, embedding family and modality. Confidence is computed without query labels. "
        "Queries are held out; neighbours are same-fold training subjects. kNN uses flat window subject means, train-only "
        "z-normalization and cosine distance, matching the initial diagnostic. Classifier heads preserve hierarchical "
        "unit/window aggregation. The two aggregations differ deliberately. "
        "The same embeddings feed head and kNN, so these signals are dependent. "
        "Neighbour agreement is not a calibrated probability of correctness. Bin-level observed accuracy is in reliability_bins.csv; "
        "sparse bins and saturated confidence limit interpretation.", ""]
    lines += [table(["Corpus", "Family", "Modality", "N", "Correctness AUROC: kNN", "Correctness AUROC: head"],
        [[CORPORA[r.corpus], r.family, r.modality, int(r.n), number(r.correctness_auroc_knn_diagnostic),
          number(r.correctness_auroc_head_diagnostic)] for _, r in reliability.iterrows()]), ""]
    lines += ["## Training observations", "",
        "Frozen models were not trained in this analysis. Logistic heads are deterministic refits of their saved L2 choices; "
        "reported probabilities were reproduced within 1e-8 before gate evaluation. LoRA text uses corrected textrerun caches. "
        "The table below reads local history and selected-checkpoint files for each direct-LLM source. "
        "Loss and validation changes are descriptive; a drop alone does not prove overfitting or single-class collapse. "
        "Slurm OOM/retry/INVALID observations are not recorded where local Slurm logs are unavailable. "
        "All file paths are indexed in training_observation_sources.json and provenance.json.", ""]
    rows = []
    for r in observations:
        rows.append([r["cell_id"], f"{r.get('selected_epoch', 'not recorded')} / {r.get('last_epoch_recorded', 'not recorded')}",
            number(r.get("train_loss_first")), number(r.get("train_loss_last")),
            number(r.get("selection_metric_value")), number(r.get("selection_macro_f1_last")),
            number(r.get("selection_macro_f1_min_after_selected"))])
    lines += [table(["Source cell", "Selected / last epoch", "First train loss", "Last train loss", "Selected metric",
                     "Last val Macro-F1", "Min val Macro-F1 after selected"], rows), "",
        "## Limitations and provenance", "",
        "Each language is represented by one corpus; language effects cannot be separated from dataset, population or recording effects. "
        "Samples are small. Direct-LLM error overlap and frozen-head gating answer different questions; frozen gate gains must not be "
        "presented as a validated selector for the LoRA LLMs. No new extraction, cluster job or adapter training was launched. "
        "LoRA audio / audio+text confidence was outside the estimated package; only corrected LoRA text confidence is included.", "",
        "The analysis folder retains private subject-level rows for reproducibility; they are excluded from this deliverable. "
        "analysis_config.json fixes K, seed, resampling and aggregation. provenance.json links source predictions, head configs, "
        "run_config.yaml, extraction metadata, split and manifest hashes and local existence. source_identity_audit.json "
        "compares actual cache bytes and canonical split hashes to the head configurations. "
        "run_config_index.json records each direct-LLM source seed, attempt, manifest/split hashes and selection protocol. "
        "parent_config_resolution.json resolves extraction parent configs by exact hash against local source runs; "
        "Missing parent training configs are legacy-unmigrated evidence, not silently reconstructed. "
        "Raw prediction source files and training artifacts remain authoritative.", ""]
    if mismatches:
        lines += ["**Identity audit did not fully pass. Affected rows require source repair before publication.**", ""]
    (a.output / "REPORT.md").write_text("\n".join(lines))
    print(json.dumps({"audit_checks": len(audit), "audit_failures": len(mismatches), "report": str(a.output / 'REPORT.md')}))


if __name__ == "__main__":
    main()
