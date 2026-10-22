"""Training diagnostics from per-epoch training history.

Pure and deterministic (stdlib only). Reads ``logs/training_history.json`` (and the optional
``logs/selected_checkpoint_selection_metrics.json``) of a fold and returns numbers plus flags.
Flags are diagnostics: they never change a run's lifecycle state.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "audiollm.training_diagnostics.v1"

# One place for every threshold. Raw numbers are always reported next to each flag,
# so a reader can re-judge a run if these are moved.
DIAGNOSTIC_THRESHOLDS: dict[str, float] = {
    "overfit_min_rising_epochs": 2,  # val_loss rises this many epochs in a row after its minimum
    "late_degradation_drop": 0.10,  # best macro-F1 minus final macro-F1
    "collapse_macro_f1_max": 0.55,  # a post-best epoch at or below this ...
    "collapse_recall_min": 0.98,  # ... with recall at or above this (predicts all positive)
    "collapse_positive_f1_max": 0.05,  # ... or positive-F1 below this (predicts all negative)
    "no_signal_macro_f1_max": 0.55,  # even the best epoch is at or below this
    "fold_best_epoch_range_max": 3,  # across folds
    "fold_macro_f1_std_max": 0.10,  # across folds (sample std of best macro-F1)
    "min_epochs_for_trend": 3,
    "spike_ratio": 5.0,  # step loss above this multiple of the median of the previous logged losses
    "spike_window": 20,
    "spike_min_history": 10,
    "high_clip_rate": 0.5,  # share of logged steps where gradient clipping fired
}

HISTORY_RELPATH = Path("logs") / "training_history.json"
SELECTION_RELPATH = Path("logs") / "selected_checkpoint_selection_metrics.json"
STEP_LOG_RELPATH = Path("logs") / "train_step_log.jsonl"


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _metric(row: dict[str, Any], name: str) -> float | None:
    """`inner_val_<name>` first, then the older/newer `selection_<name>` spelling."""
    for prefix in ("inner_val_", "selection_"):
        value = _num(row.get(prefix + name))
        if value is not None:
            return value
    return None


def _round(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(value, digits)


def _has_nonfinite(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    found = []
    for row in history:
        for key, value in row.items():
            if isinstance(value, float) and not math.isfinite(value):
                found.append({"epoch": row.get("epoch"), "field": key})
    return found


def analyze_history(
    history: list[dict[str, Any]],
    *,
    selected_epoch: int | None = None,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Analyze one fold's per-epoch history."""
    th = {**DIAGNOSTIC_THRESHOLDS, **(thresholds or {})}
    rows = [row for row in history if isinstance(row, dict) and _num(row.get("epoch")) is not None]
    rows.sort(key=lambda row: row["epoch"])
    result: dict[str, Any] = {"n_epochs": len(rows), "flags": [], "not_recorded": []}
    if not rows:
        result["status"] = "n/a"
        result["not_recorded"].append("training_history")
        return result

    epochs = [int(row["epoch"]) for row in rows]
    train_loss = [_num(row.get("train_loss")) for row in rows]
    val_loss = [_metric(row, "loss") for row in rows]
    macro_f1 = [_metric(row, "macro_f1") for row in rows]
    pos_f1 = [_metric(row, "positive_f1") for row in rows]
    recall = [_metric(row, "recall") for row in rows]

    for name, series in (("val_loss", val_loss), ("macro_f1", macro_f1), ("positive_f1", pos_f1), ("recall", recall)):
        if all(v is None for v in series):
            result["not_recorded"].append(name)

    # Best epoch: the recorded selection if present, else argmax of validation macro-F1.
    if selected_epoch is not None and int(selected_epoch) in epochs:
        best_epoch, best_source = int(selected_epoch), "selected_checkpoint_selection_metrics"
    elif any(v is not None for v in macro_f1):
        best_idx = max((i for i, v in enumerate(macro_f1) if v is not None), key=lambda i: macro_f1[i])
        best_epoch, best_source = epochs[best_idx], "derived_argmax_inner_val_macro_f1"
    else:
        best_epoch, best_source = None, None
    best_idx = epochs.index(best_epoch) if best_epoch is not None else None

    result.update(
        best_epoch=best_epoch,
        best_epoch_source=best_source,
        last_epoch=epochs[-1],
        train_loss_first=_round(train_loss[0]),
        train_loss_last=_round(train_loss[-1]),
        val_loss_min=None,
        val_loss_min_epoch=None,
        val_loss_last=_round(val_loss[-1]),
        macro_f1_best=_round(macro_f1[best_idx]) if best_idx is not None else None,
        macro_f1_last=_round(macro_f1[-1]),
    )
    flags = result["flags"]

    nonfinite = _has_nonfinite(rows)
    if nonfinite:
        flags.append({"flag": "nonfinite", "detail": {"occurrences": nonfinite[:10], "count": len(nonfinite)}})

    if best_idx is not None and best_idx == len(rows) - 1:
        flags.append({"flag": "best_is_last", "detail": {"epoch": best_epoch, "note": "may still be improving"}})

    if len(rows) < th["min_epochs_for_trend"]:
        result["status"] = "short_history"
        result["not_recorded"].append(f"trend_flags_need_{int(th['min_epochs_for_trend'])}_epochs")
        return result
    result["status"] = "ok"

    # overfit: validation loss rises for N epochs in a row after its minimum while train loss keeps falling.
    known_val = [(i, v) for i, v in enumerate(val_loss) if v is not None]
    if known_val:
        min_i, min_v = min(known_val, key=lambda pair: pair[1])
        result["val_loss_min"] = _round(min_v)
        result["val_loss_min_epoch"] = epochs[min_i]
        tail = val_loss[min_i:]
        rising = len(tail) - 1
        if rising >= th["overfit_min_rising_epochs"] and all(
            tail[k + 1] is not None and tail[k] is not None and tail[k + 1] > tail[k] for k in range(rising)
        ):
            t_min, t_last = train_loss[min_i], train_loss[-1]
            if t_min is not None and t_last is not None and t_last < t_min:
                flags.append(
                    {
                        "flag": "overfit",
                        "detail": {
                            "val_loss_min_epoch": epochs[min_i],
                            "val_loss_min": _round(min_v),
                            "val_loss_last": _round(val_loss[-1]),
                            "train_loss_at_min": _round(t_min),
                            "train_loss_last": _round(t_last),
                            "rising_epochs": rising,
                        },
                    }
                )

    # late degradation: validation macro-F1 at the end is far below its best.
    if best_idx is not None and macro_f1[-1] is not None and macro_f1[best_idx] is not None:
        drop = macro_f1[best_idx] - macro_f1[-1]
        if drop >= th["late_degradation_drop"]:
            flags.append(
                {
                    "flag": "late_degradation",
                    "detail": {
                        "best_epoch": best_epoch,
                        "best_macro_f1": _round(macro_f1[best_idx]),
                        "last_epoch": epochs[-1],
                        "last_macro_f1": _round(macro_f1[-1]),
                        "drop": _round(drop),
                    },
                }
            )

    # collapse: after the best epoch, a weak epoch whose predictions sit on one class.
    if best_idx is not None:
        hits = []
        for i in range(best_idx + 1, len(rows)):
            f1 = macro_f1[i]
            if f1 is None or f1 > th["collapse_macro_f1_max"]:
                continue
            if recall[i] is not None and recall[i] >= th["collapse_recall_min"]:
                hits.append({"epoch": epochs[i], "macro_f1": _round(f1), "recall": _round(recall[i]), "side": "all_positive"})
            elif pos_f1[i] is not None and pos_f1[i] < th["collapse_positive_f1_max"]:
                hits.append({"epoch": epochs[i], "macro_f1": _round(f1), "positive_f1": _round(pos_f1[i]), "side": "all_negative"})
        if hits:
            flags.append({"flag": "collapse", "detail": {"after_best_epoch": best_epoch, "epochs": hits}})

    if best_idx is not None and macro_f1[best_idx] is not None and macro_f1[best_idx] <= th["no_signal_macro_f1_max"]:
        flags.append({"flag": "no_signal", "detail": {"best_macro_f1": _round(macro_f1[best_idx]), "best_epoch": best_epoch}})

    return result


def analyze_step_log(rows: list[dict[str, Any]], thresholds: dict[str, float] | None = None) -> dict[str, Any]:
    """Analyze `train_step_log.jsonl` rows (loss, lr, grad_norm, clipped per logged optimizer step)."""
    th = {**DIAGNOSTIC_THRESHOLDS, **(thresholds or {})}
    result: dict[str, Any] = {"n_logged_steps": len(rows), "flags": [], "not_recorded": []}
    if not rows:
        result["not_recorded"].append("step_log_rows")
        return result
    losses = [_num(r.get("loss")) for r in rows]
    lrs = [_num(r.get("lr")) for r in rows]
    norms = [_num(r.get("grad_norm")) for r in rows]
    steps = [r.get("optimizer_step") for r in rows]
    flags = result["flags"]

    known_lr = [(i, v) for i, v in enumerate(lrs) if v is not None]
    if known_lr:
        peak_i, peak = max(known_lr, key=lambda pair: pair[1])
        result["lr"] = {
            "first": lrs[known_lr[0][0]], "peak": peak, "peak_optimizer_step": steps[peak_i],
            "last": lrs[known_lr[-1][0]], "zero_after_peak": any(v == 0.0 for _, v in known_lr if _ > peak_i),
        }
    else:
        result["not_recorded"].append("lr")

    finite_norms = [v for v in norms if v is not None and math.isfinite(v)]
    if finite_norms:
        clipped = [r.get("clipped") for r, v in zip(rows, norms) if v is not None and math.isfinite(v)]
        rate = sum(1 for c in clipped if c) / len(clipped)
        result["grad_norm"] = {
            "median": _round(statistics.median(finite_norms)), "max": _round(max(finite_norms)), "clip_rate": _round(rate),
        }
        if rate >= th["high_clip_rate"]:
            flags.append({"flag": "high_clip_rate", "detail": {"clip_rate": _round(rate), "n_logged_steps": len(finite_norms)}})
    else:
        result["not_recorded"].append("grad_norm")

    bad = [{"optimizer_step": steps[i], "field": f} for i in range(len(rows))
           for f, series in (("loss", losses), ("grad_norm", norms))
           if isinstance(rows[i].get(f), float) and not math.isfinite(rows[i][f])]
    if bad:
        flags.append({"flag": "nonfinite", "detail": {"count": len(bad), "occurrences": bad[:10]}})

    spikes = []
    window, min_hist = int(th["spike_window"]), int(th["spike_min_history"])
    for i, loss in enumerate(losses):
        if loss is None or not math.isfinite(loss):
            continue
        prev = [v for v in losses[max(0, i - window):i] if v is not None and math.isfinite(v)]
        if len(prev) < min_hist:
            continue
        med = statistics.median(prev)
        if med > 0 and loss > th["spike_ratio"] * med:
            spikes.append({"optimizer_step": steps[i], "epoch": rows[i].get("epoch"), "loss": _round(loss), "rolling_median": _round(med)})
    if spikes:
        flags.append({"flag": "loss_spike", "detail": {"count": len(spikes), "first": spikes[:3]}})
    return result


def load_step_log(fold_dir: Path) -> dict[str, Any] | None:
    """Analyze `<fold_dir>/logs/train_step_log.jsonl`; None when the run has no step log."""
    path = Path(fold_dir) / STEP_LOG_RELPATH
    if not path.is_file():
        return None
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, ValueError) as exc:
        return {"status": "unreadable", "reason": f"{type(exc).__name__}: {exc}", "flags": [], "not_recorded": []}
    return {"status": "ok", "path": str(path), **analyze_step_log(rows)}


def load_fold(fold_dir: Path) -> dict[str, Any]:
    """Analyze `<fold_dir>/logs/training_history.json`; never raises on missing or broken files."""
    fold_dir = Path(fold_dir)
    history_path = fold_dir / HISTORY_RELPATH
    out: dict[str, Any] = {"fold_dir": str(fold_dir), "history_path": str(history_path)}
    try:
        history = json.loads(history_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {**out, "status": "n/a", "reason": "training_history.json not present locally", "flags": []}
    except (OSError, ValueError) as exc:
        return {**out, "status": "unreadable", "reason": f"{type(exc).__name__}: {exc}", "flags": []}
    if not isinstance(history, list):
        return {**out, "status": "unreadable", "reason": "history is not a list", "flags": []}
    selected = None
    try:
        sel = json.loads((fold_dir / SELECTION_RELPATH).read_text(encoding="utf-8"))
        selected = int(sel["selected_epoch"])
    except (OSError, ValueError, KeyError, TypeError):
        selected = None
    result = {**out, **analyze_history(history, selected_epoch=selected)}
    step = load_step_log(fold_dir)
    if step is None:
        result["step_log"] = None
        result["not_recorded"] = [*result.get("not_recorded", []), "step_log (lr, grad_norm, loss spikes)"]
    else:
        result["step_log"] = step
        result["flags"] = [*result["flags"], *step.get("flags", [])]
    return result


def analyze_run(fold_results: dict[str, dict[str, Any]], thresholds: dict[str, float] | None = None) -> dict[str, Any]:
    """Cross-fold view: `fold_results` maps fold name to a `load_fold` result."""
    th = {**DIAGNOSTIC_THRESHOLDS, **(thresholds or {})}
    usable = {k: v for k, v in fold_results.items() if v.get("status") in ("ok", "short_history") and v.get("best_epoch") is not None}
    flags: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"n_folds": len(fold_results), "n_folds_with_history": len(usable)}
    if len(usable) >= 2:
        best_epochs = [v["best_epoch"] for v in usable.values()]
        spread = max(best_epochs) - min(best_epochs)
        summary["best_epochs"] = {k: v["best_epoch"] for k, v in sorted(usable.items())}
        if spread > th["fold_best_epoch_range_max"]:
            flags.append({"flag": "unstable_selection", "detail": {"reason": "best_epoch_range", "range": spread, "best_epochs": best_epochs}})
        scores = [v["macro_f1_best"] for v in usable.values() if v.get("macro_f1_best") is not None]
        if len(scores) >= 2:
            std = statistics.stdev(scores)
            summary["best_macro_f1_std"] = _round(std)
            if std > th["fold_macro_f1_std_max"]:
                flags.append({"flag": "unstable_selection", "detail": {"reason": "best_macro_f1_std", "std": _round(std), "scores": [_round(s) for s in scores]}})
    summary["flags"] = flags
    return summary


def run_flag_names(fold_results: dict[str, dict[str, Any]], run_summary: dict[str, Any]) -> list[str]:
    names = {f["flag"] for r in fold_results.values() for f in r.get("flags", [])}
    names |= {f["flag"] for f in run_summary.get("flags", [])}
    return sorted(names)


# ---------------------------------------------------------------- discovery


def discover_runs(scan_root: Path) -> dict[Path, list[Path]]:
    """Group every `fold_*` dir that has a `logs/` dir by its run dir (the fold's parent)."""
    runs: dict[Path, list[Path]] = {}
    for history in sorted(Path(scan_root).rglob("fold_*/logs/training_history.json")):
        fold_dir = history.parent.parent
        runs.setdefault(fold_dir.parent, []).append(fold_dir)
    return runs


def describe_run_path(run_dir: Path, scan_root: Path) -> dict[str, str]:
    """Best-effort campaign/modality/dataset/run from the path under `scan_root` (`layout` says managed, legacy or unrecognized)."""
    parts = Path(run_dir).resolve().relative_to(Path(scan_root).resolve()).parts
    if len(parts) >= 4:
        return {"campaign": parts[-4], "modality": parts[-3], "dataset": parts[-2], "run": parts[-1], "layout": "managed"}
    if len(parts) == 3:
        return {"campaign": "", "modality": parts[0], "dataset": parts[1], "run": parts[2], "layout": "legacy"}
    return {"campaign": "", "modality": "", "dataset": parts[-2] if len(parts) >= 2 else "", "run": parts[-1], "layout": "unrecognized"}


def diagnose_run(run_dir: Path, fold_dirs: list[Path] | None = None) -> dict[str, Any]:
    run_dir = Path(run_dir)
    folds = fold_dirs if fold_dirs is not None else sorted(p for p in run_dir.glob("fold_*") if p.is_dir())
    fold_results = {p.name: load_fold(p) for p in sorted(folds)}
    summary = analyze_run(fold_results)
    return {
        "schema_version": SCHEMA_VERSION,
        "run_dir": str(run_dir),
        "thresholds": dict(DIAGNOSTIC_THRESHOLDS),
        "folds": fold_results,
        "run": summary,
        "flag_names": run_flag_names(fold_results, summary),
    }


# ---------------------------------------------------------------- rendering


def _fmt(value: Any) -> str:
    return "not recorded" if value is None else str(value)


def render_run_markdown(report: dict[str, Any]) -> str:
    lines = ["## Training observations", ""]
    lines.append(f"Run: `{report['run_dir']}`")
    lines.append("Diagnostics only: flags never change a run's lifecycle state. Thresholds are in the JSON.")
    lines.append("")
    lines.append("| Fold | Epochs | Best epoch | Best / last macro-F1 | Train loss first→last | Val loss min (epoch) → last | Flags |")
    lines.append("|---|---|---|---|---|---|---|")
    for name, res in report["folds"].items():
        if res.get("status") not in ("ok", "short_history"):
            lines.append(f"| {name} | n/a | n/a | n/a | n/a | n/a | {res.get('status')}: {res.get('reason', '')} |")
            continue
        flags = ", ".join(f["flag"] for f in res["flags"]) or "none"
        lines.append(
            f"| {name} | {res['n_epochs']} | {_fmt(res.get('best_epoch'))} "
            f"| {_fmt(res.get('macro_f1_best'))} / {_fmt(res.get('macro_f1_last'))} "
            f"| {_fmt(res.get('train_loss_first'))}→{_fmt(res.get('train_loss_last'))} "
            f"| {_fmt(res.get('val_loss_min'))} ({_fmt(res.get('val_loss_min_epoch'))}) → {_fmt(res.get('val_loss_last'))} "
            f"| {flags} |"
        )
    detail_lines = []
    for name, res in report["folds"].items():
        for flag in res.get("flags", []):
            detail_lines.append(f"- {name} `{flag['flag']}`: {json.dumps(flag['detail'], sort_keys=True)}")
    for flag in report["run"].get("flags", []):
        detail_lines.append(f"- run `{flag['flag']}`: {json.dumps(flag['detail'], sort_keys=True)}")
    if detail_lines:
        lines += ["", "Flag details:", *detail_lines]
    step_lines = []
    for name, res in report["folds"].items():
        sl = res.get("step_log")
        if not sl or sl.get("status") != "ok":
            continue
        lr, gn = sl.get("lr"), sl.get("grad_norm")
        parts = [f"{sl['n_logged_steps']} logged steps"]
        if lr:
            parts.append(f"lr first {lr['first']:.3g}, peak {lr['peak']:.3g} at step {lr['peak_optimizer_step']}, last {lr['last']:.3g}")
        if gn:
            parts.append(f"grad norm median {gn['median']}, max {gn['max']}, clip rate {gn['clip_rate']}")
        step_lines.append(f"- {name}: " + "; ".join(parts))
    if step_lines:
        lines += ["", "Step log:", *step_lines]
    missing = sorted({m for res in report["folds"].values() for m in res.get("not_recorded", [])})
    if missing:
        lines += ["", "Not recorded: " + ", ".join(missing)]
    return "\n".join(lines) + "\n"
