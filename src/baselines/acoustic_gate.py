"""Apply the preregistered N3 development gate to immutable N1/N2 artifacts."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.baselines.acoustic_crossfold import (
    DEFAULT_OUTPUT,
    CachePaths,
    DEFAULT_EGEMAPS_CACHE,
    DEFAULT_MANIFEST,
    DEFAULT_PARTITIONS,
    DEFAULT_WAVLM_CACHE,
    git_provenance,
    read_json,
    sha256_file,
    verify_outputs,
)
from src.baselines.acoustic_mil import N2Paths, verify_n2_outputs


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def evaluate_development_candidate(
    *,
    name: str,
    pooled_auroc: float,
    auroc_delta_interval: Mapping[str, float],
    balanced_delta_interval: Mapping[str, float],
    auroc_positive_folds: int,
    balanced_positive_folds: int,
    auroc_positive_seeds: int | None,
    balanced_positive_seeds: int | None,
) -> dict[str, Any]:
    criteria = {
        "pooled_auroc_above_0.5": bool(pooled_auroc > 0.5),
        "auroc_delta_ci_excludes_zero_positive": bool(
            float(auroc_delta_interval["lower_2.5pct"]) > 0
        ),
        "balanced_accuracy_delta_ci_excludes_zero_positive": bool(
            float(balanced_delta_interval["lower_2.5pct"]) > 0
        ),
        "auroc_positive_in_at_least_four_outer_folds": bool(auroc_positive_folds >= 4),
        "balanced_accuracy_positive_in_at_least_four_outer_folds": bool(
            balanced_positive_folds >= 4
        ),
    }
    if auroc_positive_seeds is not None and balanced_positive_seeds is not None:
        criteria.update(
            auroc_positive_in_at_least_four_seeds=bool(auroc_positive_seeds >= 4),
            balanced_accuracy_positive_in_at_least_four_seeds=bool(
                balanced_positive_seeds >= 4
            ),
        )
    return {
        "candidate": name,
        "criteria": criteria,
        "development_gate_pass": bool(all(criteria.values())),
        "failed_criteria": [key for key, value in criteria.items() if not value],
    }


def linear_candidate(name: str, metrics: Mapping[str, Any]) -> dict[str, Any]:
    paired = metrics["real_minus_shuffled_paired_bootstrap_95pct"]
    directions = metrics["shuffled_bundle_control"]["fold_directions"]
    return evaluate_development_candidate(
        name=name,
        pooled_auroc=float(metrics["real"]["pooled_oof_metrics"]["auroc"]),
        auroc_delta_interval=paired["intervals"]["auroc"],
        balanced_delta_interval=paired["intervals"]["balanced_accuracy"],
        auroc_positive_folds=int(directions["auroc"]["positive_fold_count"]),
        balanced_positive_folds=int(
            directions["balanced_accuracy"]["positive_fold_count"]
        ),
        auroc_positive_seeds=None,
        balanced_positive_seeds=None,
    )


def mil_candidate(name: str, metrics: Mapping[str, Any]) -> dict[str, Any]:
    paired = metrics["paired_subject_seed_bootstrap_95pct"]
    directions = metrics["fold_directions"]
    auroc_positive_seeds = sum(
        float(row["real_minus_shuffled"]["auroc"]) > 0
        for row in metrics["seed_metrics"]
    )
    balanced_positive_seeds = sum(
        float(row["real_minus_shuffled"]["balanced_accuracy"]) > 0
        for row in metrics["seed_metrics"]
    )
    return evaluate_development_candidate(
        name=name,
        pooled_auroc=float(
            metrics["ensemble_mean_probability"]["real_metrics"]["auroc"]
        ),
        auroc_delta_interval=paired["intervals"]["auroc"],
        balanced_delta_interval=paired["intervals"]["balanced_accuracy"],
        auroc_positive_folds=int(directions["auroc"]["positive_fold_count"]),
        balanced_positive_folds=int(
            directions["balanced_accuracy"]["positive_fold_count"]
        ),
        auroc_positive_seeds=auroc_positive_seeds,
        balanced_positive_seeds=balanced_positive_seeds,
    )


def render_report(decision: Mapping[str, Any]) -> str:
    lines = [
        "# DAIC N3 acoustic decision gate",
        "",
        f"Generated: {decision['completed_at_utc']}",
        "",
        "Decision: **fail the current fixed-K preprocessed DAIC audio protocol at the development gate**.",
        "",
        "No candidate satisfied all necessary development conditions. Therefore the official "
        "47-subject test set remains locked, no winner is frozen, and the conditional small-Qwen "
        "memory probe or training step is not authorized by this protocol.",
        "",
        "| Candidate | Development pass | Failed criteria |",
        "|---|---:|---|",
    ]
    for candidate in decision["candidates"]:
        failed = ", ".join(candidate["failed_criteria"]) or "none"
        lines.append(
            f"| {candidate['candidate']} | {'yes' if candidate['development_gate_pass'] else 'no'} | {failed} |"
        )
    lines.extend(
        [
            "",
            "The gated-attention model improved pooled OOF discrimination, but its paired "
            "real-minus-shuffled AUROC and balanced-accuracy intervals crossed zero. Attention "
            "was nearly uniform, so the sparse-informative-chunk hypothesis was not supported.",
            "",
            "Required interpretation:",
            "",
            "> No reproducible depression signal was established for the tested fixed-K "
            "preprocessed DAIC audio protocol.",
            "",
            "This conclusion does not apply to raw participant-only speech, other speech "
            "representations, other depression datasets, or audio rebuilt with timestamps and "
            "interviewer exclusion.",
            "",
        ]
    )
    return "\n".join(lines)


def apply_gate(protocol_root: Path) -> dict[str, Any]:
    n3_root = protocol_root / "n3"
    completion_path = n3_root / "n3_gate_complete.json"
    if completion_path.is_file():
        completion = read_json(completion_path)
        for filename, expected_hash in completion["artifact_sha256"].items():
            path = n3_root / filename
            if not path.is_file() or sha256_file(path) != expected_hash:
                raise ValueError(f"N3 immutable artifact changed: {path}")
        return read_json(n3_root / "n3_gate_decision.json")
    if n3_root.exists():
        raise FileExistsError(f"Incomplete N3 directory exists: {n3_root}")

    base_paths = CachePaths(
        manifest=DEFAULT_MANIFEST.resolve(),
        partitions=DEFAULT_PARTITIONS.resolve(),
        egemaps=DEFAULT_EGEMAPS_CACHE.resolve(),
        wavlm=DEFAULT_WAVLM_CACHE.resolve(),
        output=protocol_root,
    )
    n1_verification = verify_outputs(base_paths)
    n2_paths = N2Paths(base=base_paths, n2_root=protocol_root / "n2")
    n2_verification = verify_n2_outputs(n2_paths)
    if not n2_verification.get("full_run_complete"):
        raise RuntimeError("N2 must be complete before applying N3")

    metric_paths = {
        "egemaps_linear": protocol_root / "n1/egemaps/metrics.json",
        "wavlm_linear": protocol_root / "n1/wavlm/metrics.json",
        "mean_pooling": protocol_root / "n2/results/mean_pooling/metrics.json",
        "gated_attention": protocol_root / "n2/results/gated_attention/metrics.json",
    }
    metrics = {name: read_json(path) for name, path in metric_paths.items()}
    candidates = [
        linear_candidate("egemaps_linear", metrics["egemaps_linear"]),
        linear_candidate("wavlm_linear", metrics["wavlm_linear"]),
        mil_candidate("mean_pooling", metrics["mean_pooling"]),
        mil_candidate("gated_attention", metrics["gated_attention"]),
    ]
    eligible = [row["candidate"] for row in candidates if row["development_gate_pass"]]
    if eligible:
        raise RuntimeError(
            "At least one development candidate passed; this fail-only N3 implementation "
            f"must not choose a winner implicitly: {eligible}"
        )
    nuisance = read_json(protocol_root / "n2/nuisance_diagnostics.json")
    decision = {
        "schema_version": 1,
        "status": "complete",
        "completed_at_utc": utc_now(),
        "decision": "fail_current_audio_protocol_at_development_gate",
        "reason": (
            "No candidate satisfied the necessary paired AUROC and balanced-accuracy "
            "confidence-interval conditions; a complete pass is impossible regardless of "
            "the locked-test outcome."
        ),
        "candidates": candidates,
        "official_test": {
            "status": "locked",
            "predictions_created": False,
            "action": "do_not_evaluate_without_a_development-eligible_frozen_winner",
        },
        "conditional_follow_up": {
            "small_qwen_memory_probe": "do_not_run",
            "new_qwen2_audio_training": "stop_for_current_preprocessed_chunks",
            "model_focus": "text-driven depression screening with explicit labeling",
            "data_priority": (
                "obtain raw participant-only audio, timestamps, diarization, and interviewer "
                "exclusion metadata before revisiting audio"
            ),
        },
        "required_interpretation": (
            "No reproducible depression signal was established for the tested fixed-K "
            "preprocessed DAIC audio protocol."
        ),
        "scope_limitations": [
            "Do not generalize to raw participant-only speech.",
            "Do not generalize to all speech representations or depression datasets.",
            "Preprocessing kind remains perfectly label-associated.",
            "Participant-only speech and interviewer exclusion cannot be established locally.",
        ],
        "nuisance_summary": {
            "mean_normalized_attention_entropy": nuisance["attention"][
                "normalized_entropy"
            ]["mean"],
            "exact_duplicate_group_count": nuisance["exact_duplicate_group_count"],
            "sample_kind_by_label": nuisance["sample_kind_by_label"],
        },
        "input_verification": {
            "n1": n1_verification,
            "n2": n2_verification,
            "artifact_sha256": {
                name: sha256_file(path) for name, path in metric_paths.items()
            },
            "n2_nuisance_sha256": sha256_file(
                protocol_root / "n2/nuisance_diagnostics.json"
            ),
        },
        "provenance": {
            "analysis_code_path": str(Path(__file__).resolve()),
            "analysis_code_sha256": sha256_file(Path(__file__).resolve()),
            "repository": git_provenance(),
        },
    }
    n3_root.mkdir(parents=True)
    decision_path = n3_root / "n3_gate_decision.json"
    report_path = n3_root / "N3_GATE_DECISION.md"
    from src.baselines.acoustic_crossfold import atomic_json

    atomic_json(decision_path, decision)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=n3_root, prefix=f".{report_path.name}.", delete=False
    ) as handle:
        handle.write(render_report(decision))
        temporary = Path(handle.name)
    os.replace(temporary, report_path)
    completion = {
        "schema_version": 1,
        "status": "complete_immutable",
        "artifact_sha256": {
            decision_path.name: sha256_file(decision_path),
            report_path.name: sha256_file(report_path),
        },
        "official_test_predictions_created": False,
    }
    atomic_json(completion_path, completion)
    return decision


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-root", default=str(DEFAULT_OUTPUT))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    decision = apply_gate(Path(args.protocol_root).expanduser().resolve())
    print(
        json.dumps(
            {
                "status": decision["status"],
                "decision": decision["decision"],
                "official_test": decision["official_test"],
                "conditional_follow_up": decision["conditional_follow_up"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
