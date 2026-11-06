"""Domain-adversarial (gradient reversal) encoder, then the usual L2 head on its representation.

Two stages per outer fold, on one standardized frozen vector per subject.

Stage 1, encoder training (DANN, Ganin and Lempitsky, 2015)::

    x -> encoder (``layers`` x [Linear, ReLU, Dropout]) -> z
    z -> temporary depression head (Linear -> 1 logit)          loss 1: binary cross entropy
    z -> gradient reversal -> dataset head (Linear, ReLU, Linear -> datasets)
                                                                 loss 2: cross entropy

The gradient reversal layer is the identity on the forward pass and multiplies
the gradient by ``-lambda`` on the backward pass: the dataset head learns to
recognise the dataset, the encoder learns to make that hard while keeping
depression detectable. ``lambda`` ramps from 0 to ``adv_lambda`` with the usual
``2 / (1 + exp(-10 p)) - 1`` schedule, where ``p`` is progress through the first
``ramp_epochs`` epochs (fixed, so an encoder snapshot at epoch ``e`` of a long
run saw the same schedule as a run of ``e`` epochs).
``adv_lambda = 0`` is the same network with no adversarial pressure.

Batches are drawn from rows oversampled so that every dataset x class cell has
the size of the largest cell, so the dataset label says nothing about
depression and the adversary cannot remove depression signal through
prevalence differences.

Stage 2, head: the encoder is frozen, both stage-1 heads are dropped, and the
L2 logistic head of ``merged_subject_mean_heads`` (equal dataset mass, balanced
classes inside each dataset, ``fit_logistic``) is fit on the train-standardized
``z``.

Selection (outer-train subjects only): for each inner fold an encoder is
trained on the inner-train subjects (snapshots at ``EPOCH_GRID``), the L2 head
is fit on its inner-train ``z`` for every penalty and scored on the inner-val
``z``. Per ``lambda`` the (epochs, penalty) pair with the best mean per-dataset
inner macro-F1 (then lower BCE, then larger penalty, then fewer epochs) is
kept. Across ``lambda`` the same rule (then smaller ``lambda``) gives the
inner-selected variant. Evaluation is deterministic (``eval()`` mode).
"""

from __future__ import annotations

import math
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.subject_mean_logreg_head import DEFAULT_L2_GRID, fit_logistic, score_metrics  # noqa: E402

EPOCH_GRID = (5, 10, 25, 50)
LAMBDA_GRID = (0.0, 0.3, 1.0, 3.0)
SELECTION_RULE = ("max mean per-dataset inner macro-F1, then min BCE, then larger L2, then fewer epochs "
                  "(across lambda: then smaller lambda)")


@dataclass(frozen=True)
class DannConfig:
    adv_lambda: float = 0.0
    seed: int = 0
    layers: int = 1
    hidden: int = 256
    dataset_hidden: int = 128
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 1e-2
    batch_size: int = 64
    epochs: int = max(EPOCH_GRID)
    ramp_epochs: int = 10


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lam * grad, None


def grad_reverse(x: torch.Tensor, lam: float) -> torch.Tensor:
    return _GradReverse.apply(x, lam)


class DannNet(nn.Module):
    def __init__(self, dim: int, n_datasets: int, cfg: DannConfig):
        super().__init__()
        blocks: list[nn.Module] = []
        width = dim
        for _ in range(cfg.layers):
            blocks += [nn.Linear(width, cfg.hidden), nn.ReLU(), nn.Dropout(cfg.dropout)]
            width = cfg.hidden
        self.encoder = nn.Sequential(*blocks)
        self.depression = nn.Linear(width, 1)
        self.dataset = nn.Sequential(nn.Linear(width, cfg.dataset_hidden), nn.ReLU(),
                                     nn.Linear(cfg.dataset_hidden, n_datasets))

    def forward(self, x: torch.Tensor, lam: float = 0.0):
        z = self.encoder(x)
        return self.depression(z).squeeze(-1), self.dataset(grad_reverse(z, lam)), z


def oversample_cells(groups: list[str], labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Row indices with every (group, class) cell copied up to the largest cell."""
    cells: dict[tuple[str, int], list[int]] = defaultdict(list)
    for i, (g, label) in enumerate(zip(groups, labels.tolist())):
        cells[(g, int(label))].append(i)
    for g in set(groups):
        if not cells.get((g, 0)) or not cells.get((g, 1)):
            raise ValueError(f"Dataset {g} lacks a class in this training set.")
    target = max(len(v) for v in cells.values())
    out: list[int] = []
    for key in sorted(cells):
        members = cells[key]
        whole, remainder = divmod(target, len(members))
        out += members * whole
        out += [members[i] for i in sorted(rng.choice(len(members), size=remainder, replace=False))]
    return np.asarray(out)


def _ramp(progress: float) -> float:
    return 2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0


@torch.no_grad()
def predict(model: DannNet, x: np.ndarray) -> dict[str, np.ndarray]:
    model.eval()
    dep, dataset, z = model(torch.as_tensor(x, dtype=torch.float32))
    return {"logit": dep.numpy().astype(np.float64), "dataset_logit": dataset.numpy(),
            "z": z.numpy().astype(np.float64)}


def train_dann(x: np.ndarray, y: np.ndarray, groups: list[str], cfg: DannConfig,
               snapshot_epochs: tuple[int, ...] = (), snapshot_inputs: tuple[np.ndarray, ...] = ()) -> dict[str, Any]:
    """Stage 1 on oversampled rows. ``snapshots[epoch]`` = encoder outputs for each of ``snapshot_inputs``."""
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng([cfg.seed, 7])
    levels = sorted(set(groups))
    rows = oversample_cells(groups, y, rng)
    xt = torch.as_tensor(x, dtype=torch.float32)
    yt = torch.as_tensor(y, dtype=torch.float32)
    gt = torch.as_tensor([levels.index(g) for g in groups], dtype=torch.long)
    model = DannNet(x.shape[1], len(levels), cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    bce, ce = nn.BCEWithLogitsLoss(), nn.CrossEntropyLoss()
    steps_per_epoch = math.ceil(len(rows) / cfg.batch_size)
    ramp_steps = cfg.ramp_epochs * steps_per_epoch
    step, history, snapshots = 0, [], {}
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        order = rows[rng.permutation(len(rows))]
        sums = np.zeros(4)
        for start in range(0, len(order), cfg.batch_size):
            batch = torch.as_tensor(order[start:start + cfg.batch_size])
            lam = cfg.adv_lambda * _ramp(min(1.0, step / max(ramp_steps, 1)))
            dep_logit, dataset_logit, _ = model(xt[batch], lam)
            loss_dep = bce(dep_logit, yt[batch])
            loss_dataset = ce(dataset_logit, gt[batch])
            optimizer.zero_grad()
            (loss_dep + loss_dataset).backward()
            optimizer.step()
            accuracy = (dataset_logit.argmax(1) == gt[batch]).float().mean().item()
            sums += [loss_dep.item(), loss_dataset.item(), accuracy, lam]
            step += 1
        mean = sums / steps_per_epoch
        history.append({"epoch": epoch, "loss_depression": mean[0], "loss_dataset": mean[1],
                        "dataset_head_train_accuracy": mean[2], "lambda": mean[3]})
        if epoch in snapshot_epochs:
            snapshots[epoch] = [predict(model, xi)["z"] for xi in snapshot_inputs]
    return {"model": model, "levels": levels, "history": history, "snapshots": snapshots,
            "fit_rows": int(len(rows))}


def head_weights(groups: list[str], y: np.ndarray) -> np.ndarray:
    """Equal dataset mass, balanced classes inside each dataset (``dataset_class_weights``)."""
    counts: dict[tuple[str, int], int] = defaultdict(int)
    for g, label in zip(groups, y.tolist()):
        counts[(g, int(label))] += 1
    n = len(set(groups))
    return np.asarray([1.0 / (n * 2 * counts[(g, int(label))]) for g, label in zip(groups, y.tolist())])


def l2_head_logits(z_train: np.ndarray, y: np.ndarray, groups: list[str], z_eval: list[np.ndarray],
                   penalty: float) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Fit the L2 head on train-standardized ``z``; logits for every array in ``z_eval``."""
    mean, scale = z_train.mean(0), z_train.std(0)
    scale[scale < 1e-6] = 1.0
    weight, bias, info = fit_logistic((z_train - mean) / scale, y, head_weights(groups, y), penalty)
    return [((z - mean) / scale) @ weight + bias for z in z_eval], info


def _criterion(y: np.ndarray, logits: np.ndarray, groups: list[str]) -> tuple[float, float]:
    g = np.asarray(groups)
    f1 = float(np.mean([score_metrics(y[g == d], logits[g == d])["macro_f1"] for d in sorted(set(groups))]))
    return f1, float(np.mean(np.logaddexp(0.0, logits) - y * logits))


def dataset_probe(z_train: np.ndarray, g_train: list[str], z_test: np.ndarray, g_test: list[str]) -> float:
    """Balanced accuracy of a linear probe (train z -> dataset) on held-out z."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(z_train)
    probe = LogisticRegression(max_iter=3000, class_weight="balanced").fit(scaler.transform(z_train), g_train)
    return float(balanced_accuracy_score(g_test, probe.predict(scaler.transform(z_test))))


def two_stage(x_train: np.ndarray, y: np.ndarray, groups: list[str], x_holdout: np.ndarray,
              folds: list[np.ndarray], seed: int, lambdas=LAMBDA_GRID, epoch_grid=EPOCH_GRID,
              l2_grid=DEFAULT_L2_GRID, base: DannConfig = DannConfig()) -> dict[str, Any]:
    """Inner-fold selection of (epochs, L2) per lambda, then the final encoder + L2 head per lambda."""
    n = len(y)
    out: dict[str, Any] = {"per_lambda": {}}
    for lam in lambdas:
        oof = {(e, p): np.zeros(n) for e in epoch_grid for p in l2_grid}
        for k, held in enumerate(folds):
            keep = np.setdiff1d(np.arange(n), held)
            g_keep = [groups[i] for i in keep]
            run = train_dann(x_train[keep], y[keep], g_keep,
                             replace(base, adv_lambda=lam, seed=seed * 100 + k + 1, epochs=max(epoch_grid)),
                             snapshot_epochs=epoch_grid, snapshot_inputs=(x_train[keep], x_train[held]))
            for e in epoch_grid:
                z_keep, z_held = run["snapshots"][e]
                for p in l2_grid:
                    (logits,), _ = l2_head_logits(z_keep, y[keep], g_keep, [z_held], p)
                    oof[(e, p)][held] = logits
        candidates = []
        for (e, p), logits in oof.items():
            f1, bce = _criterion(y, logits, groups)
            candidates.append({"epochs": e, "l2_penalty": p, "mean_dataset_macro_f1": f1, "binary_cross_entropy": bce})
        best = max(candidates, key=lambda c: (c["mean_dataset_macro_f1"], -c["binary_cross_entropy"],
                                              c["l2_penalty"], -c["epochs"]))
        final = train_dann(x_train, y, groups, replace(base, adv_lambda=lam, seed=seed, epochs=best["epochs"]))
        z_train, z_holdout = predict(final["model"], x_train)["z"], predict(final["model"], x_holdout)["z"]
        (train_logits, holdout_logits), head_info = l2_head_logits(z_train, y, groups, [z_train, z_holdout],
                                                                   best["l2_penalty"])
        out["per_lambda"][lam] = {
            "config": asdict(replace(base, adv_lambda=lam, seed=seed, epochs=best["epochs"])),
            "selection": {"selected_epochs": best["epochs"], "selected_l2_penalty": best["l2_penalty"],
                          "inner_mean_dataset_macro_f1": best["mean_dataset_macro_f1"],
                          "inner_binary_cross_entropy": best["binary_cross_entropy"], "candidates": candidates},
            "stage1": {"fit_rows": final["fit_rows"], "history": final["history"]},
            "head_fit_info": head_info,
            "holdout_logits": holdout_logits,
            "z_train": z_train, "z_holdout": z_holdout,
            "train_macro_f1": score_metrics(y, train_logits)["macro_f1"],
        }
    chosen = max(out["per_lambda"], key=lambda lam: (out["per_lambda"][lam]["selection"]["inner_mean_dataset_macro_f1"],
                                                    -out["per_lambda"][lam]["selection"]["inner_binary_cross_entropy"],
                                                    -lam))
    out["selected_lambda"] = chosen
    out["selection_rule"] = SELECTION_RULE
    return out


def fit_merged_fold(x_train, train_rows, x_holdout, holdout_rows, seed: int, lambdas=LAMBDA_GRID) -> dict[str, Any]:
    """Merged protocol: global train scaler and subject means as the L2 head (B), grouped inner folds."""
    from baselines.merged_subject_mean_heads import INNER_FOLDS, INNER_SEED, SubjectUnits, _fit_scalers, _standardize
    from src.merged.protocol import build_grouped_inner_folds

    tr, ho = SubjectUnits(train_rows), SubjectUnits(holdout_rows)
    train, holdout = sorted(tr.units), sorted(ho.units)
    scalers = _fit_scalers(x_train, tr, train, "global")
    xs_train = tr.means(_standardize(x_train, tr, scalers), train)
    xs_holdout = ho.means(_standardize(x_holdout, ho, scalers), holdout)
    y_train, g_train = tr.y(train), [tr.dataset[s] for s in train]
    y_holdout, g_holdout = ho.y(holdout), [ho.dataset[s] for s in holdout]
    inner = build_grouped_inner_folds(train_rows, inner_folds=INNER_FOLDS, seed=INNER_SEED)
    position = {s: i for i, s in enumerate(train)}
    folds = [np.asarray(sorted(position[s] for s in f["validation_subject_ids"])) for f in inner["folds"]]
    result = two_stage(xs_train, y_train, g_train, xs_holdout, folds, seed, lambdas)
    variants = {}
    for lam, r in result["per_lambda"].items():
        logits = r.pop("holdout_logits")
        z_train, z_holdout = r.pop("z_train"), r.pop("z_holdout")
        r["dataset_readout"] = {"linear_probe_z_holdout_balanced_accuracy":
                                dataset_probe(z_train, g_train, z_holdout, g_holdout),
                                "chance": 1.0 / len(set(g_train))}
        r["predictions"] = [
            {"dataset": ho.dataset[s], "subject_id": s, "label": ho.labels[s], "prediction": int(v > 0),
             "probability": 1.0 / (1.0 + math.exp(-v)), "mean_logit": v, "num_units": len(ho.units[s])}
            for s, v in zip(holdout, logits.tolist())]
        r["metrics_by_dataset"] = {d: score_metrics(y_holdout[np.asarray(g_holdout) == d],
                                                    logits[np.asarray(g_holdout) == d]) for d in sorted(set(g_holdout))}
        variants[lam] = r
    return {"per_lambda": variants, "selected_lambda": result["selected_lambda"],
            "selection_rule": result["selection_rule"], "inner_folds": INNER_FOLDS, "inner_seed": INNER_SEED,
            "inner_assignments_hash": inner.get("assignments_hash")}
