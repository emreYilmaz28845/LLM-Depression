from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.daic_derived_views import materialize_cached_views
from src.data.runtime import build_examples, filter_rows_by_subjects, load_manifest_rows
from src.data.split_utils import CV_PROTOCOL_TRAIN_VAL, resolve_cv_protocol, resolve_split_mode
from src.evaluate import _load_metadata_or_build, _resolve_final_eval_subject_ids, evaluate_examples
from src.model.runtime import load_model_for_inference, load_processor
from src.utils import (
    load_yaml_with_overrides,
    read_json,
    resolve_model_name_or_path,
    save_json,
    save_yaml,
)


VIEW_OVERRIDES: dict[str, dict[str, Any]] = {
    "fixed4": {"data.eval_chunk_policy": "fixed_k", "data.eval_chunks_per_subject": 4},
    "mincover4": {"data.eval_chunk_policy": "balanced_joint_cover", "data.eval_chunks_per_subject": 4},
    "all": {"data.eval_chunk_policy": "all", "data.eval_chunks_per_subject": "all"},
}
DERIVED_VIEWS = {"fixed15", "matched10_even", "matched10_resampled"}


def _set(config: dict[str, Any], dotted: str, value: Any) -> None:
    cursor = config
    parts = dotted.split(".")
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = value


def _override_args(overrides: dict[str, Any]) -> list[str]:
    values = []
    for key, value in overrides.items():
        if isinstance(value, bool):
            rendered = "true" if value else "false"
        elif isinstance(value, (list, dict)):
            rendered = json.dumps(value, separators=(",", ":"))
        else:
            rendered = str(value)
        values.extend(["--set", f"{key}={rendered}"])
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--views", required=True)
    parser.add_argument("--overrides-json", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    overrides = json.loads(args.overrides_json)
    override_cli = _override_args(overrides)
    base_config = load_yaml_with_overrides(args.config, override_cli)
    views = [value for value in args.views.split(",") if value]
    inferred_views = [view for view in views if view not in DERIVED_VIEWS]
    unsupported = set(inferred_views) - set(VIEW_OVERRIDES)
    if unsupported:
        raise ValueError(f"Unsupported inferred comprehensive views: {sorted(unsupported)}")

    first_config = copy.deepcopy(base_config)
    for key, value in VIEW_OVERRIDES[inferred_views[0]].items():
        _set(first_config, key, value)
    first_cli = override_cli + _override_args(VIEW_OVERRIDES[inferred_views[0]])
    metadata = _load_metadata_or_build(args.config, first_config, first_cli)
    manifest_rows = load_manifest_rows(metadata["manifest_path"])
    split_mode = resolve_split_mode(first_config, metadata)
    cv_protocol = resolve_cv_protocol(first_config) if split_mode == "cv" else None

    model_name = resolve_model_name_or_path(None, base_config)
    processor = load_processor(args.checkpoint_dir, base_config)
    model = load_model_for_inference(model_name, args.checkpoint_dir, base_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    model_load_count = 1

    for view in inferred_views:
        output_dir = args.output_root / view
        metrics_path = output_dir / "metrics_original_teacher_forced.json"
        if args.resume and metrics_path.exists():
            print(f"RESUME skip completed inferred view {view}", flush=True)
            continue
        if output_dir.exists() and any(output_dir.iterdir()) and not args.resume:
            raise ValueError(f"Collision: evaluation output exists: {output_dir}")
        config = copy.deepcopy(base_config)
        for key, value in VIEW_OVERRIDES[view].items():
            _set(config, key, value)
        final_ids = _resolve_final_eval_subject_ids(config, metadata, args.fold)
        if int(config.get("split", {}).get("smoke_subject_limit", 0) or 0) > 0:
            saved_split = read_json(args.checkpoint_dir.parent / "logs" / "split_used.json")
            final_ids = sorted(saved_split["final_eval_subject_ids"])
        rows = filter_rows_by_subjects(manifest_rows, final_ids)
        role = "fold_validation" if cv_protocol == CV_PROTOCOL_TRAIN_VAL else "final_eval"
        examples = build_examples(rows, config, partition_name=role, truncation_log_path=None)
        save_yaml(
            {
                "base_config_path": str(args.config), "config_overrides": overrides,
                "view_overrides": VIEW_OVERRIDES[view], "view": view,
                "inference_dtype": config.get("evaluation", {}).get("inference_dtype", "fp32"),
                "candidate_batching": config.get("evaluation", {}).get("candidate_batching", "sequential"),
                "model_load_count_for_cell": model_load_count, "config": config,
            },
            output_dir / "eval_config.yaml",
        )
        started = time.monotonic()
        result = evaluate_examples(
            model, processor, examples, config, output_dir,
            checkpoint_name=args.checkpoint_dir.name,
        )
        sample_rows = result["sample_rows"]
        save_json(
            {
                "view": view, "derived_without_model_inference": False,
                "logical_samples": len(sample_rows),
                "actual_model_forwards": sum(int(row.get("inference_model_forwards", 0)) for row in sample_rows),
                "actual_audio_file_loads": sum(int(row.get("inference_audio_file_loads", 0)) for row in sample_rows),
                "model_load_count_for_cell": model_load_count,
                "inference_dtype": str(next(model.parameters()).dtype),
                "candidate_batching": config.get("evaluation", {}).get("candidate_batching", "sequential"),
                "elapsed_seconds": time.monotonic() - started,
            },
            output_dir / "inference_metadata.json",
        )

    if bool(base_config.get("evaluation", {}).get("reuse_derived_views", False)):
        materialize_cached_views(
            args.output_root, views, checkpoint_name=args.checkpoint_dir.name,
            resample_seed=int(base_config.get("split", {}).get("seed", 1337)),
        )
    elif any(view in DERIVED_VIEWS for view in views):
        raise ValueError("Derived comprehensive views require evaluation.reuse_derived_views=true")


if __name__ == "__main__":
    main()
