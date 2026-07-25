from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from src.utils import read_json


TIE_ORDER = ("audio_only", "text_only", "audio_text")


def _jobs(modalities: list[str], folds: list[int], ratio: float) -> list[dict]:
    ratio_token = f"ros{int(round(ratio * 100)):03d}"
    jobs = []
    for modality in modalities:
        config = (
            "configs/experiments/turkish_oversampling/"
            f"turkish_t17_{modality}_selmacro_tf_qwen3asr.yaml"
        )
        for profile in ("weighted", "oversampled"):
            run_name = (
                f"t17_selmacro_qwen3asr_weighted_{modality}"
                if profile == "weighted"
                else f"t17_selmacro_qwen3asr_{ratio_token}_os1337_{modality}"
            )
            for fold in folds:
                jobs.append(
                    {
                        "modality": modality,
                        "config": config,
                        "fold": fold,
                        "profile": profile,
                        "run_name": run_name,
                        "sampling_mode": (
                            "weighted_sampler"
                            if profile == "weighted"
                            else "minority_subject_oversample"
                        ),
                        "oversampling_ratio": None if profile == "weighted" else ratio,
                        "oversampling_seed": None if profile == "weighted" else 1337,
                        "chain_key": f"{modality}:{profile}",
                    }
                )
    return jobs


def build(stage: str, optuna_summary: Path, pilot_summary: Path | None) -> dict:
    optuna = read_json(optuna_summary)
    if not optuna.get("proceed_to_qwen"):
        raise ValueError("Stage-3 gate did not authorize Qwen.")
    ratio = float(optuna["selected_ratio"])
    if stage == "pilot":
        decisions = {
            row["condition"]: float(row["mean_macro_f1_gain"])
            for row in optuna["decisions"]
            if row["condition"] in optuna["qualifying_modalities"]
        }
        selected = min(
            decisions,
            key=lambda modality: (-decisions[modality], TIE_ORDER.index(modality)),
        )
        jobs = _jobs([selected], [0, 1], ratio)
        return {
            "schema_version": "turkish_oversampling_qwen_matrix.v1",
            "stage": "pilot",
            "selected_ratio": ratio,
            "selected_modality": selected,
            "expected_jobs": 4,
            "jobs": jobs,
        }
    if pilot_summary is None:
        raise ValueError("--pilot-summary is required for the full matrix.")
    pilot = read_json(pilot_summary)
    if not pilot.get("proceed_to_full"):
        raise ValueError("Stage-4 pilot gate did not authorize full Qwen.")
    jobs = _jobs(list(TIE_ORDER), [0, 1, 2, 3, 4], ratio)
    return {
        "schema_version": "turkish_oversampling_qwen_matrix.v1",
        "stage": "full",
        "selected_ratio": ratio,
        "stage4_summary": str(pilot_summary),
        "expected_jobs": 30,
        "jobs": jobs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("pilot", "full"), required=True)
    parser.add_argument("--optuna-summary", type=Path, required=True)
    parser.add_argument("--pilot-summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build(args.stage, args.optuna_summary, args.pilot_summary)
    if len(payload["jobs"]) != payload["expected_jobs"]:
        raise ValueError("Qwen matrix job count mismatch.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    print(f"Wrote {len(payload['jobs'])} jobs to {args.output}")


if __name__ == "__main__":
    main()
