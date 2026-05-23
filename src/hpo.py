from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import optuna

from src.utils import (
    configure_logging,
    ensure_dir,
    get_logger,
    load_yaml_with_overrides,
    read_json,
    resolve_model_name_or_path,
    save_json,
    save_json_atomic,
)


LOGGER = get_logger(__name__)
TRIAL_TABLE_COLUMNS = [
    "trial_number",
    "value",
    "state",
    "lr",
    "lora_r",
    "lora_alpha",
    "lora_dropout",
    "weight_decay",
    "warmup_ratio",
    "max_grad_norm",
    "gradient_accumulation_steps",
]


def _parse_int_list(raw_value: str) -> list[int]:
    values = [item.strip() for item in str(raw_value).split(",") if item.strip()]
    if not values:
        raise ValueError(f"Expected at least one integer in {raw_value!r}.")
    return [int(item) for item in values]


def _parse_float_list(raw_value: str) -> list[float]:
    values = [item.strip() for item in str(raw_value).split(",") if item.strip()]
    if not values:
        raise ValueError(f"Expected at least one float in {raw_value!r}.")
    return [float(item) for item in values]


def _storage_url_from_path(storage_path: Path) -> str:
    return f"sqlite:///{storage_path.resolve()}"


def _read_progress(progress_path: Path) -> dict[str, Any] | None:
    if not progress_path.exists():
        return None
    try:
        payload = read_json(progress_path)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    if not isinstance(payload.get("step"), int):
        return None
    if not isinstance(payload.get("metric"), (int, float)):
        return None
    return payload


def _terminate_process_group(process: subprocess.Popen[bytes | str]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except OSError:
        process.terminate()

    deadline = time.time() + 30
    while time.time() < deadline:
        if process.poll() is not None:
            return
        time.sleep(0.2)

    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    except OSError:
        process.kill()


def _sample_trial_params(trial: optuna.Trial, args: argparse.Namespace) -> dict[str, Any]:
    params: dict[str, Any] = {
        "lr": trial.suggest_float("lr", args.lr_min, args.lr_max, log=True),
        "lora_r": trial.suggest_categorical("lora_r", args.lora_r_choices),
        "lora_alpha": trial.suggest_categorical("lora_alpha", args.lora_alpha_choices),
    }
    if args.search_lora_dropout:
        params["lora_dropout"] = trial.suggest_float("lora_dropout", args.lora_dropout_min, args.lora_dropout_max)
    if args.search_weight_decay:
        params["weight_decay"] = trial.suggest_float("weight_decay", args.weight_decay_min, args.weight_decay_max)
    if args.search_warmup_ratio:
        params["warmup_ratio"] = trial.suggest_float("warmup_ratio", args.warmup_ratio_min, args.warmup_ratio_max)
    if args.search_max_grad_norm:
        params["max_grad_norm"] = trial.suggest_categorical("max_grad_norm", args.max_grad_norm_choices)
    if args.search_gradient_accumulation_steps:
        params["gradient_accumulation_steps"] = trial.suggest_categorical(
            "gradient_accumulation_steps",
            args.gradient_accumulation_choices,
        )
    return params


def _trial_overrides(params: dict[str, Any]) -> list[str]:
    overrides = [
        f"training.learning_rate={params['lr']}",
        f"lora.rank={params['lora_r']}",
        f"lora.alpha={params['lora_alpha']}",
    ]
    if "lora_dropout" in params:
        overrides.append(f"lora.dropout={params['lora_dropout']}")
    if "weight_decay" in params:
        overrides.append(f"training.weight_decay={params['weight_decay']}")
    if "warmup_ratio" in params:
        overrides.append(f"training.warmup_ratio={params['warmup_ratio']}")
    if "max_grad_norm" in params:
        overrides.append(f"training.max_grad_norm={params['max_grad_norm']}")
    if "gradient_accumulation_steps" in params:
        overrides.append(f"training.gradient_accumulation_steps={params['gradient_accumulation_steps']}")
    return overrides


def _launch_trial(
    trial: optuna.Trial,
    args: argparse.Namespace,
    config: dict[str, Any],
    study_dir: Path,
) -> float:
    params = _sample_trial_params(trial, args)
    trial_name = f"{args.run_name_prefix}_trial_{trial.number:03d}"
    progress_path = study_dir / "trial_runtime" / f"trial_{trial.number:03d}_progress.json"
    result_path = study_dir / "trial_runtime" / f"trial_{trial.number:03d}_result.json"
    for path in (progress_path, result_path):
        if path.exists():
            path.unlink()

    command = [
        "torchrun",
        f"--nproc_per_node={args.nproc_per_node}",
        str(PROJECT_ROOT / "src" / "train.py"),
        "--config",
        args.config,
        "--fold",
        str(args.fold),
        "--run_name",
        trial_name,
        "--save_strategy",
        args.save_strategy,
        "--trial-progress-file",
        str(progress_path),
        "--trial-result-file",
        str(result_path),
    ]
    model_name_or_path = resolve_model_name_or_path(args.model_name_or_path, config)
    if args.model_name_or_path:
        command.extend(["--model_name_or_path", args.model_name_or_path])
    for override in args.config_overrides:
        command.extend(["--set", override])
    command.extend(["--set", f"training.num_train_epochs={args.trial_train_epochs}"])
    for override in _trial_overrides(params):
        command.extend(["--set", override])

    LOGGER.info(
        "Launching trial=%s run_name=%s lr=%.3e lora_r=%s lora_alpha=%s",
        trial.number,
        trial_name,
        float(params["lr"]),
        params["lora_r"],
        params["lora_alpha"],
    )
    LOGGER.info("Trial model path: %s", model_name_or_path)

    process = subprocess.Popen(command, cwd=str(PROJECT_ROOT), start_new_session=True)
    last_reported_step = -1

    try:
        while True:
            progress = _read_progress(progress_path)
            if progress is not None and progress["step"] > last_reported_step:
                last_reported_step = int(progress["step"])
                metric_value = float(progress["best_metric"])
                trial.report(metric_value, step=last_reported_step)
                if args.pruning and trial.should_prune():
                    LOGGER.info(
                        "Pruning trial=%s at epoch=%s best_metric=%.6f",
                        trial.number,
                        last_reported_step,
                        metric_value,
                    )
                    _terminate_process_group(process)
                    raise optuna.TrialPruned(
                        f"Trial {trial.number} pruned at epoch {last_reported_step} with metric {metric_value:.6f}"
                    )

            return_code = process.poll()
            if return_code is not None:
                if return_code != 0:
                    raise RuntimeError(f"torchrun exited with code {return_code} for trial {trial.number}")
                break
            time.sleep(5)

        result = read_json(result_path)
        metric_value = float(result["best_metric"])
        trial.set_user_attr("run_name", trial_name)
        trial.set_user_attr("run_root", result["run_root"])
        trial.set_user_attr("result_path", str(result_path))
        trial.set_user_attr("history_path", result.get("history_path"))
        trial.set_user_attr("save_strategy", result.get("save_strategy"))
        trial.set_user_attr("best_epoch", result.get("best_epoch"))
        trial.set_user_attr("sample_prediction_mode", result.get("sample_prediction_mode"))
        trial.set_user_attr("best_model_dir", result.get("best_model_dir"))
        trial.set_user_attr("last_model_dir", result.get("last_model_dir"))
        return metric_value
    finally:
        if process.poll() is None:
            _terminate_process_group(process)


def _trial_row(trial: optuna.trial.FrozenTrial) -> dict[str, Any]:
    row = {
        "trial_number": trial.number,
        "value": None if trial.value is None else float(trial.value),
        "state": trial.state.name,
        "lr": trial.params.get("lr"),
        "lora_r": trial.params.get("lora_r"),
        "lora_alpha": trial.params.get("lora_alpha"),
        "lora_dropout": trial.params.get("lora_dropout"),
        "weight_decay": trial.params.get("weight_decay"),
        "warmup_ratio": trial.params.get("warmup_ratio"),
        "max_grad_norm": trial.params.get("max_grad_norm"),
        "gradient_accumulation_steps": trial.params.get("gradient_accumulation_steps"),
    }
    return row


def _write_study_csv(path: Path, metadata_rows: list[tuple[str, Any]], trial_rows: list[dict[str, Any]]) -> None:
    lines: list[str] = []
    for key, value in metadata_rows:
        lines.append(f"# {key},{value}")
    lines.append("")
    lines.append(",".join(TRIAL_TABLE_COLUMNS))
    for row in trial_rows:
        lines.append(",".join("" if row.get(column) is None else str(row.get(column)) for column in TRIAL_TABLE_COLUMNS))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _study_metadata_rows(
    args: argparse.Namespace,
    config: dict[str, Any],
    study: optuna.Study,
    storage_path: Path,
) -> list[tuple[str, Any]]:
    completed_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    best_trial = study.best_trial if completed_trials else None
    return [
        ("study_name", study.study_name),
        ("dataset_name", config.get("dataset")),
        ("config_path", args.config),
        ("fold", args.fold),
        ("storage_path", str(storage_path)),
        ("n_trials", len(study.trials)),
        ("n_completed", len(completed_trials)),
        ("trial_train_epochs", args.trial_train_epochs),
        ("best_trial_number", None if best_trial is None else best_trial.number),
        ("best_f1", None if best_trial is None else float(best_trial.value)),
        ("save_strategy", args.save_strategy),
        ("pruning_enabled", args.pruning),
        ("pruner_startup_trials", args.pruner_startup_trials),
        ("pruner_warmup_steps", args.pruner_warmup_steps),
        ("pruner_interval_steps", args.pruner_interval_steps),
    ]


def _write_study_summary(
    args: argparse.Namespace,
    config: dict[str, Any],
    study: optuna.Study,
    study_dir: Path,
    storage_path: Path,
) -> None:
    completed_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    best_trial = study.best_trial if completed_trials else None
    trial_rows = [_trial_row(trial) for trial in study.trials]
    summary_payload = {
        "study_name": study.study_name,
        "dataset_name": config.get("dataset"),
        "config_path": args.config,
        "fold": int(args.fold),
        "storage_path": str(storage_path),
        "trial_train_epochs": int(args.trial_train_epochs),
        "save_strategy": args.save_strategy,
        "pruning_enabled": bool(args.pruning),
        "pruner_startup_trials": int(args.pruner_startup_trials),
        "pruner_warmup_steps": int(args.pruner_warmup_steps),
        "pruner_interval_steps": int(args.pruner_interval_steps),
        "n_trials": len(study.trials),
        "n_completed": len(completed_trials),
        "best_trial_number": None if best_trial is None else best_trial.number,
        "best_f1": None if best_trial is None else float(best_trial.value),
        "best_params": None if best_trial is None else dict(best_trial.params),
        "all_trials": trial_rows,
    }
    save_json(summary_payload, study_dir / "study_results.json")
    _write_study_csv(
        study_dir / "study_results_table.csv",
        _study_metadata_rows(args, config, study, storage_path),
        trial_rows,
    )


def _materialize_best_trial(
    args: argparse.Namespace,
    config: dict[str, Any],
    study: optuna.Study,
    study_dir: Path,
) -> None:
    completed_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if not completed_trials:
        LOGGER.warning("Skipping best-trial materialization because there are no completed trials.")
        return
    best_trial = study.best_trial
    materialize_run_name = f"{args.run_name_prefix}_best_trial_{best_trial.number:03d}"
    result_path = study_dir / "materialized_best_trial_result.json"
    progress_path = study_dir / "materialized_best_trial_progress.json"
    for path in (result_path, progress_path):
        if path.exists():
            path.unlink()

    command = [
        "torchrun",
        f"--nproc_per_node={args.nproc_per_node}",
        str(PROJECT_ROOT / "src" / "train.py"),
        "--config",
        args.config,
        "--fold",
        str(args.fold),
        "--run_name",
        materialize_run_name,
        "--save_strategy",
        "full",
        "--trial-progress-file",
        str(progress_path),
        "--trial-result-file",
        str(result_path),
    ]
    if args.model_name_or_path:
        command.extend(["--model_name_or_path", args.model_name_or_path])
    for override in args.config_overrides:
        command.extend(["--set", override])
    command.extend(["--set", f"training.num_train_epochs={args.trial_train_epochs}"])
    for override in _trial_overrides(dict(best_trial.params)):
        command.extend(["--set", override])

    LOGGER.info("Materializing best trial #%s as full training run: %s", best_trial.number, materialize_run_name)
    subprocess.run(command, cwd=str(PROJECT_ROOT), check=True)
    if result_path.exists():
        result_payload = read_json(result_path)
        save_json(result_payload, study_dir / "materialized_best_trial_summary.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Optuna HPO for LLM-Depression training.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--model_name_or_path", default=None)
    parser.add_argument("--study-name", default=None)
    parser.add_argument("--study-dir", default=None)
    parser.add_argument("--storage-path", default=None, help="SQLite DB path for Optuna state.")
    parser.add_argument("--n-trials", type=int, default=40)
    parser.add_argument("--nproc-per-node", type=int, default=4)
    parser.add_argument("--trial-train-epochs", type=int, default=10)
    parser.add_argument("--run-name-prefix", default="optuna")
    parser.add_argument("--save_strategy", choices=("full", "best_only", "hpo_minimal"), default="hpo_minimal")
    parser.add_argument("--materialize-best-trial", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pruning", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pruner-startup-trials", type=int, default=5)
    parser.add_argument("--pruner-warmup-steps", type=int, default=1)
    parser.add_argument("--pruner-interval-steps", type=int, default=1)
    parser.add_argument("--lr-min", type=float, default=5e-6)
    parser.add_argument("--lr-max", type=float, default=5e-5)
    parser.add_argument("--lora-r-choices", type=_parse_int_list, default=[2, 4, 8])
    parser.add_argument("--lora-alpha-choices", type=_parse_int_list, default=[4, 8, 16])
    parser.add_argument("--search-lora-dropout", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lora-dropout-min", type=float, default=0.1)
    parser.add_argument("--lora-dropout-max", type=float, default=0.3)
    parser.add_argument("--search-weight-decay", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--weight-decay-min", type=float, default=0.01)
    parser.add_argument("--weight-decay-max", type=float, default=0.1)
    parser.add_argument("--search-warmup-ratio", action="store_true")
    parser.add_argument("--warmup-ratio-min", type=float, default=0.0)
    parser.add_argument("--warmup-ratio-max", type=float, default=0.1)
    parser.add_argument("--search-max-grad-norm", action="store_true")
    parser.add_argument("--max-grad-norm-choices", type=_parse_float_list, default=[0.5, 1.0, 2.0])
    parser.add_argument("--search-gradient-accumulation-steps", action="store_true")
    parser.add_argument("--gradient-accumulation-choices", type=_parse_int_list, default=[4, 8, 16])
    parser.add_argument(
        "--set",
        dest="config_overrides",
        action="append",
        default=[],
        help="Override config values with KEY=VALUE, using dot paths for nested keys.",
    )
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    config = load_yaml_with_overrides(args.config, args.config_overrides)

    dataset_name = str(config["dataset"])
    study_name = args.study_name or f"{dataset_name}_fold{args.fold}_optuna"
    default_study_root = PROJECT_ROOT / "outputs" / "optuna" / dataset_name / study_name
    study_dir = ensure_dir(args.study_dir or default_study_root)
    storage_path = Path(args.storage_path) if args.storage_path else (study_dir / f"{study_name}.db")
    ensure_dir(storage_path.parent)

    pruner = optuna.pruners.NopPruner()
    if args.pruning:
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=args.pruner_startup_trials,
            n_warmup_steps=args.pruner_warmup_steps,
            interval_steps=args.pruner_interval_steps,
        )

    study = optuna.create_study(
        study_name=study_name,
        storage=_storage_url_from_path(storage_path),
        load_if_exists=True,
        direction="maximize",
        pruner=pruner,
    )

    study_metadata = {
        "study_name": study_name,
        "dataset_name": dataset_name,
        "config_path": args.config,
        "fold": int(args.fold),
        "storage_path": str(storage_path.resolve()),
        "n_trials_requested": int(args.n_trials),
        "trial_train_epochs": int(args.trial_train_epochs),
        "save_strategy": args.save_strategy,
        "pruning_enabled": bool(args.pruning),
        "nproc_per_node": int(args.nproc_per_node),
        "search_space": {
            "lr_min": args.lr_min,
            "lr_max": args.lr_max,
            "lora_r_choices": args.lora_r_choices,
            "lora_alpha_choices": args.lora_alpha_choices,
            "search_lora_dropout": args.search_lora_dropout,
            "search_weight_decay": args.search_weight_decay,
            "search_warmup_ratio": args.search_warmup_ratio,
            "search_max_grad_norm": args.search_max_grad_norm,
            "search_gradient_accumulation_steps": args.search_gradient_accumulation_steps,
        },
    }
    save_json_atomic(study_metadata, study_dir / "study_config.json")

    def objective(trial: optuna.Trial) -> float:
        return _launch_trial(trial, args, config, study_dir)

    LOGGER.info("Starting Optuna study=%s dataset=%s fold=%s trials=%s", study_name, dataset_name, args.fold, args.n_trials)
    try:
        study.optimize(objective, n_trials=args.n_trials, n_jobs=1)
    finally:
        _write_study_summary(args, config, study, study_dir, storage_path)

    if args.materialize_best_trial and args.save_strategy == "hpo_minimal":
        _materialize_best_trial(args, config, study, study_dir)
        _write_study_summary(args, config, study, study_dir, storage_path)

    LOGGER.info(
        "Study complete: name=%s completed=%s best_trial=%s best_value=%s",
        study.study_name,
        len([trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]),
        None if not any(trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials) else study.best_trial.number,
        None if not any(trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials) else f"{study.best_trial.value:.6f}",
    )


if __name__ == "__main__":
    main()
