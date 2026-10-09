#!/usr/bin/env python3
"""Generate Worker 3 window-cap treatment configs from canonical route configs.

The treatment adds only ``training.window_cap`` to a canonical Native standalone
audio config: enabled, fraction (25/50/75 percent), fixed sampling seed 1337,
the frozen algorithm version and the base-config identity/hash. Nothing else is
changed, so the treatment differs from its control only by the training-window
cap.

Usage:
  python scripts/build_window_cap_configs.py            # write/refresh configs
  python scripts/build_window_cap_configs.py --check    # fail on any drift
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "configs" / "experiments" / "window_cap"
ALGORITHM_VERSION = "sha256-subject-permutation-v1"
SAMPLING_SEED = 1337
ARMS = {"25": 0.25, "50": 0.5, "75": 0.75}
BASE_CONFIGS = {
    "turkish_audio_only_native": "configs/main/turkish_pooled_t17_audio_only_harmonized_selmacrof1_likelihood_v1_qwen3asr_promptcontext_v1_qwen3omni_30b_a3b.yaml",
    "turkish_audio_text_native": "configs/main/turkish_pooled_t17_audio_text_harmonized_selmacrof1_likelihood_v1_qwen3asr_promptcontext_v1_qwen3omni_30b_a3b.yaml",
    "daic_audio_only_native": "configs/main/daic_audio_only_harmonized_selmacrof1_likelihood_v1.yaml",
    "daic_audio_text_native": "configs/main/daic_audio_text_harmonized_selmacrof1_likelihood_v1.yaml",
    "d3tec_audio_only_native": "configs/main/d3tec_audio_only_harmonized_selmacrof1_likelihood_v1.yaml",
    "d3tec_audio_text_native": "configs/main/d3tec_audio_text_harmonized_selmacrof1_likelihood_v1.yaml",
    "androids_interview_audio_only_native": "configs/main/androids_audio_only_harmonized_selmacrof1_likelihood_v1.yaml",
    "androids_interview_audio_text_native": "configs/main/androids_audio_text_harmonized_selmacrof1_likelihood_v1.yaml",
    "cmdc_audio_only_native": "configs/main/cmdc_audio_only_harmonized_selmacrof1_likelihood_v1.yaml",
    "cmdc_audio_text_native": "configs/main/cmdc_audio_text_harmonized_selmacrof1_likelihood_v1.yaml",
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_treatment_config(base_path: Path, fraction: float) -> dict:
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    if not isinstance(base, dict):
        raise SystemExit(f"base config is not a mapping: {base_path}")
    training = base.setdefault("training", {})
    training["window_cap"] = {
        "enabled": True,
        "fraction": float(fraction),
        "sampling_seed": SAMPLING_SEED,
        "algorithm_version": ALGORITHM_VERSION,
        "base_config": str(base_path.relative_to(PROJECT_ROOT)),
        "base_config_sha256": sha256_file(base_path),
    }
    return base


def treatment_name(base_path: Path, pct: str) -> str:
    return f"{base_path.stem}_cap{pct}.yaml"


def dump_config(config: dict) -> str:
    return yaml.safe_dump(config, sort_keys=False, allow_unicode=True)


def expected_files() -> dict[str, str]:
    expected: dict[str, str] = {}
    for route_id, rel in BASE_CONFIGS.items():
        base_path = PROJECT_ROOT / rel
        if not base_path.is_file():
            raise SystemExit(f"missing canonical base config for {route_id}: {rel}")
        for pct, fraction in ARMS.items():
            expected[treatment_name(base_path, pct)] = dump_config(
                build_treatment_config(base_path, fraction)
            )
    return expected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail on any drift")
    args = parser.parse_args()

    expected = expected_files()
    failed: list[str] = []
    if args.check:
        existing = sorted(p.name for p in OUTPUT_DIR.glob("*.yaml")) if OUTPUT_DIR.is_dir() else []
        if existing != sorted(expected):
            missing = sorted(set(expected) - set(existing))
            extra = sorted(set(existing) - set(expected))
            failed.append(f"file set drift: missing={missing[:5]} extra={extra[:5]}")
        for name, content in expected.items():
            path = OUTPUT_DIR / name
            if path.is_file() and path.read_text(encoding="utf-8") != content:
                failed.append(f"content drift: {name}")
        if failed:
            print("\n".join(failed), file=sys.stderr)
            return 1
        print(f"window cap configs OK: {len(expected)} files")
        return 0

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    written = 0
    for name, content in expected.items():
        path = OUTPUT_DIR / name
        if not path.is_file() or path.read_text(encoding="utf-8") != content:
            path.write_text(content, encoding="utf-8")
            written += 1
    print(f"window cap configs: {len(expected)} total, {written} written/updated -> {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
