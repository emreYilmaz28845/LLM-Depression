from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml


MODALITIES = ("audio_text", "audio_only", "text_only")


def main() -> None:
    output_dir = PROJECT_ROOT / "configs" / "experiments" / "turkish_oversampling"
    output_dir.mkdir(parents=True, exist_ok=True)
    for modality in MODALITIES:
        source = (
            PROJECT_ROOT
            / "configs"
            / "main"
            / f"turkish_t17_{modality}_selposf1_tf_qwen3asr.yaml"
        )
        config = yaml.safe_load(source.read_text(encoding="utf-8"))
        config["output_dirs"]["run_root"] = (
            "${PROJECT_ROOT}/output_model/experiments/"
            f"turkish_t17_qwen3asr_selmacro/{modality}"
        )
        config["training"]["selection_metric"] = "inner_val_macro_f1"
        config["training"]["selection_metric_mode"] = "max"
        config["training"]["early_stopping"]["metric"] = "inner_val_macro_f1"
        config["training"]["oversampling_ratio"] = None
        config["training"]["oversampling_seed"] = 1337
        target = output_dir / f"turkish_t17_{modality}_selmacro_tf_qwen3asr.yaml"
        header = (
            "# Generated from the reported Qwen3ASR T17 configuration. "
            "Only run root and checkpoint-selection fields differ; sampling "
            "is supplied explicitly by the experiment matrix.\n"
        )
        target.write_text(
            header + yaml.safe_dump(config, sort_keys=False),
            encoding="utf-8",
        )
        print(target)


if __name__ == "__main__":
    main()
