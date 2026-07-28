#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the three-job D3TEC smoke gate.")
    parser.add_argument("--run", action="append", required=True, help="CONFIG_ID=RUN_ROOT")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    expected = {"audio_only_rotary", "audio_text_normalized", "text_only"}
    runs = {}
    for value in args.run:
        config_id, separator, path = value.partition("=")
        if not separator:
            raise ValueError(f"Invalid --run value: {value}")
        runs[config_id] = Path(path)
    if set(runs) != expected:
        raise ValueError(f"Expected smoke configs {sorted(expected)}; found {sorted(runs)}")
    results = {}
    for config_id, run_root in sorted(runs.items()):
        fold = run_root / "fold_0"
        eval_dir = fold / "eval" / "best_checkpoint"
        required = [
            fold / "run_config.yaml",
            fold / "logs" / "split_used.json",
            fold / "logs" / "selected_checkpoint_selection_metrics.json",
            eval_dir / "metrics_original_teacher_forced.json",
            eval_dir / "predictions_subject_level.csv",
            eval_dir / "predictions_sample_level.csv",
        ]
        if config_id != "text_only":
            required.extend(
                [
                    fold / "logs" / "d3tec_training_schedule_audit.json",
                    eval_dir / "predictions_response_level.csv",
                ]
            )
        missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
        if missing:
            raise FileNotFoundError(f"{config_id} smoke missing artifacts: {missing}")
        selection = json.loads(
            (fold / "logs" / "selected_checkpoint_selection_metrics.json").read_text()
        )
        selected_epoch = int(selection["selected_epoch"])
        if selected_epoch not in {1, 2}:
            raise ValueError(f"{config_id}: selected smoke epoch must be 1 or 2.")
        item = {"run_root": str(run_root), "selected_epoch": selected_epoch}
        if config_id != "text_only":
            schedule = json.loads(
                (fold / "logs" / "d3tec_training_schedule_audit.json").read_text()
            )
            if int(schedule["virtual_epochs"]) != 2:
                raise ValueError(f"{config_id}: smoke did not exercise two virtual epochs.")
            if config_id == "audio_text_normalized" and abs(
                float(schedule["mean_loss_weight"]) - 1.0
            ) > 1e-9:
                raise ValueError("audio_text_normalized: schedule weights do not have mean one.")
            item["schedule_policy"] = schedule["policy"]
            item["examples_per_virtual_epoch"] = schedule["examples_per_virtual_epoch"]
        results[config_id] = item
    payload = {"status": "passed", "configs": results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
