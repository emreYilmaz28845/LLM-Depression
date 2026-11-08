"""Critic-confusion encoder (uniform confusion, MMD, cross-source contrastive), then the usual L2 head.

A port of a colleague's cached-feature "reusable multilingual embedding" study
(critics + uniform confusion + multi-kernel MMD + cross-source supervised
contrastive loss) to our merged protocol. It works on one standardized frozen
vector per subject.

Stage 1, encoder training, per (method, strength)::

    x -> train-only cell-balanced standardization -> PCA 64 (components scaled)
      -> Linear 64->128, GELU, Dropout 0.1, Linear 128->width, LayerNorm (no affine) -> z
    z -> temporary depression head (Linear -> 1)                 BCE on the batch
    z -> 9 dataset critics: {linear, MLP-GELU, MLP-Tanh} x {all subjects, y=0 only, y=1 only}

Every batch holds ``per_cell`` subjects of each dataset x class cell (cycled
with replacement inside a cell, so small cells are oversampled). Per step the
critics take ``critic_steps`` ordinary cross-entropy updates on the detached
(cell-balanced standardized) ``z``; then the encoder takes one update on

    BCE + representation penalty + ramp * (strength * (confusion [+ MMD]) [+ 0.1 * SupCon])

where ``confusion`` is KL(uniform || critic) averaged over the critics (global
weight 0.5, each label 0.25), ``MMD`` is a multi-bandwidth Gaussian MMD between
every dataset pair (globally and within each label) and ``SupCon`` pulls
same-label subjects of different datasets together. ``ramp`` rises linearly
over the first 50 steps. Methods: ``task`` (none of the source terms),
``conditional`` (confusion only), ``combined`` (confusion + MMD + SupCon).
Every path starts from a fixed 100-step task-only warm-up (the colleague
selected the warm-up step on a development split; here it is fixed).

Stage 2, head: the encoder is frozen and the L2 head of ``dann_subject_head``
is fit on the train-standardized ``z``. The snapshot step (``STEP_GRID``, steps
after the warm-up) and the L2 penalty are chosen on the grouped inner folds by
mean per-dataset inner macro-F1 (then lower BCE, larger L2, earlier step).
Unlike the colleague's study, source recovery is not used for selection; it is
measured afterwards by ``fresh_attacks`` on held-out subjects. The grid has no
step 0 (end of warm-up): in a smoke run the inner F1 rule picked step 0 for
every variant, which would make all variants the same task-only warm-up encoder.
"""

from __future__ import annotations

import math
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.dann_subject_head import _criterion, l2_head_logits  # noqa: E402
from baselines.subject_mean_logreg_head import DEFAULT_L2_GRID, score_metrics  # noqa: E402

STEP_GRID = (200, 600, 1200)
VARIANT_GRID = (("task", 0.0), ("conditional", 10.0), ("conditional", 30.0), ("conditional", 100.0),
                ("combined", 10.0), ("combined", 30.0), ("combined", 100.0))
SELECTION_RULE = "max mean per-dataset inner macro-F1, then min BCE, then larger L2, then earlier step"


@dataclass(frozen=True)
class ConfusionConfig:
    method: str = "task"
    strength: float = 0.0
    seed: int = 0
    width: int = 32
    hidden: int = 128
    dropout: float = 0.1
    pca_dims: int = 64
    lr: float = 1e-3
    critic_lr: float = 3e-3
    weight_decay: float = 1e-3
    per_cell: int = 8
    critic_steps: int = 5
    warmup_steps: int = 100
    steps: int = max(STEP_GRID)
    ramp_steps: int = 50
    contrastive_weight: float = 0.1
    temperature: float = 0.2


def variant_name(method: str, strength: float) -> str:
    return "task" if method == "task" else f"{method[:4]}{strength:g}"


def cell_weights(d: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Every dataset x class cell gets the same total weight (sums to 1)."""
    cells = sorted(set(zip(d.tolist(), y.tolist())))
    if len(cells) != 2 * len(set(d.tolist())):
        raise ValueError("Both classes are required in every dataset.")
    w = np.zeros(len(y))
    for a, b in cells:
        mask = (d == a) & (y == b)
        w[mask] = 1.0 / (len(cells) * mask.sum())
    return w


def fit_pca(x: np.ndarray, d: np.ndarray, y: np.ndarray, dims: int) -> dict[str, np.ndarray]:
    w = cell_weights(d, y)
    mean = w @ x
    scale = np.sqrt(w @ ((x - mean) ** 2))
    scale[scale < 1e-6] = 1.0
    a = (x - mean) / scale
    center = w @ a
    _, _, vt = np.linalg.svd((a - center) * np.sqrt(w[:, None]), full_matrices=False)
    basis = vt[:min(dims, len(x) - 1)].T
    p = (a - center) @ basis
    pmean = w @ p
    pscale = np.sqrt(w @ ((p - pmean) ** 2))
    pscale[pscale < 1e-6] = 1.0
    return {"mean": mean, "scale": scale, "center": center, "basis": basis, "pmean": pmean, "pscale": pscale}


def apply_pca(state: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    return (((x - state["mean"]) / state["scale"] - state["center"]) @ state["basis"] - state["pmean"]) / state["pscale"]


class CellSampler:
    """Each draw: ``per_cell`` subjects from every dataset x class cell, cycling a shuffled pool per cell."""

    def __init__(self, d: np.ndarray, y: np.ndarray, per_cell: int, seed: int):
        self.cells = [(a, b) for a in sorted(set(d.tolist())) for b in (0, 1)]
        self.members = {c: np.flatnonzero((d == c[0]) & (y == c[1])) for c in self.cells}
        if any(len(m) == 0 for m in self.members.values()):
            raise ValueError("Both classes are required in every dataset.")
        self.per_cell, self.rng = per_cell, np.random.default_rng(seed)
        self.pool = {c: np.array([], dtype=int) for c in self.cells}
        self.draws = np.zeros(len(y), dtype=int)

    def draw(self) -> np.ndarray:
        out = []
        for c in self.cells:
            need, parts = self.per_cell, []
            while need:
                if not len(self.pool[c]):
                    self.pool[c] = self.rng.permutation(self.members[c])
                take = self.pool[c][:need]
                self.pool[c] = self.pool[c][len(take):]
                parts.append(take)
                need -= len(take)
            out.append(np.concatenate(parts))
        ids = np.concatenate(out)
        np.add.at(self.draws, ids, 1)
        return ids


class ConfusionNet(nn.Module):
    def __init__(self, dim: int, n_datasets: int, cfg: ConfusionConfig):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(dim, cfg.hidden), nn.GELU(), nn.Dropout(cfg.dropout),
                                     nn.Linear(cfg.hidden, cfg.width), nn.LayerNorm(cfg.width, elementwise_affine=False))
        self.depression = nn.Linear(cfg.width, 1)
        self.critics = nn.ModuleList([nn.ModuleList([
            nn.Linear(cfg.width, n_datasets),
            nn.Sequential(nn.Linear(cfg.width, 64), nn.GELU(), nn.Linear(64, n_datasets)),
            nn.Sequential(nn.Linear(cfg.width, 64), nn.Tanh(), nn.Linear(64, n_datasets))]) for _ in range(3)])


def _standardize(z: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    mean = (w[:, None] * z).sum(0)
    return (z - mean) / torch.sqrt((w[:, None] * (z - mean) ** 2).sum(0) + 1e-6)


def critic_loss(model: ConfusionNet, z: torch.Tensor, d: torch.Tensor, y: torch.Tensor, confuse: bool) -> torch.Tensor:
    """Cross entropy (critic update) or KL(uniform || critic) (encoder update); global 0.5, each label 0.25."""
    values = []
    for group, heads in enumerate(model.critics):
        use = torch.ones_like(y, dtype=torch.bool) if group == 0 else y == group - 1
        terms = []
        for head in heads:
            score = head(z[use])
            terms.append((-F.log_softmax(score, 1).mean(1) - math.log(score.shape[1])).mean() if confuse
                         else F.cross_entropy(score, d[use]))
        values.append(sum(terms) / len(terms))
    return 0.5 * values[0] + 0.25 * (values[1] + values[2])


def alignment_matrix(d: np.ndarray, y: np.ndarray) -> torch.Tensor:
    """Quadratic form whose value on a kernel matrix is the summed pairwise dataset MMD (global + per label)."""
    k, n = len(set(d.tolist())), len(d)
    out = np.zeros((n, n))
    for label, weight in ((None, 0.5), (0, 0.25), (1, 0.25)):
        use = np.ones(n, dtype=bool) if label is None else y == label
        a = np.stack([((d == j) & use).astype(float) / np.sum((d == j) & use) for j in range(k)])
        for j in range(k):
            for h in range(j):
                diff = a[j] - a[h]
                out += weight / math.comb(k, 2) * np.outer(diff, diff)
    return torch.as_tensor(out, dtype=torch.float32)


def mmd_loss(z: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    dist = ((z[:, None, :] - z[None, :, :]) ** 2).mean(-1)
    kernel = sum(torch.exp(-dist / (2 * b * b)) for b in (0.1, 0.3, 1.0, 3.0, 10.0)) / 5
    return (kernel * matrix).sum()


def cross_source_contrastive(z: torch.Tensor, d: torch.Tensor, y: torch.Tensor, ids: torch.Tensor,
                             temperature: float) -> torch.Tensor:
    z = F.normalize(z, dim=1)
    score = z @ z.T / temperature
    other = ids[:, None] != ids[None, :]
    positive = (y[:, None] == y[None, :]) & (d[:, None] != d[None, :]) & other
    logprob = score - torch.logsumexp(score.masked_fill(~other, -1e9), dim=1, keepdim=True)
    return -((logprob * positive).sum(1) / positive.sum(1).clamp(min=1)).mean()


def representation_penalty(z: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    mean = (w[:, None] * z).sum(0)
    std = torch.sqrt((w[:, None] * (z - mean) ** 2).sum(0) + 1e-6)
    normalized = (z - mean) / std
    cov = normalized.T @ (normalized * w[:, None])
    off = cov - torch.diag(torch.diag(cov))
    return 0.1 * F.relu(0.2 - std).square().mean() + 1e-3 * off.square().mean()


@torch.no_grad()
def encode(model: ConfusionNet, x: np.ndarray) -> np.ndarray:
    model.eval()
    return model.encoder(torch.as_tensor(x, dtype=torch.float32)).numpy().astype(np.float64)


def train_confusion(x: np.ndarray, y: np.ndarray, groups: list[str], cfg: ConfusionConfig,
                    snapshot_steps: tuple[int, ...] = (), snapshot_inputs: tuple[np.ndarray, ...] = ()) -> dict[str, Any]:
    """Stage 1 on PCA inputs ``x``. ``snapshots[step]`` = encoder outputs for each of ``snapshot_inputs``."""
    levels = sorted(set(groups))
    d = np.asarray([levels.index(g) for g in groups])
    torch.manual_seed(cfg.seed)
    model = ConfusionNet(x.shape[1], len(levels), cfg)
    params = list(model.encoder.parameters()) + list(model.depression.parameters())
    xt, w = torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(cell_weights(d, y), dtype=torch.float32)
    yt, dt, yl = (torch.as_tensor(y, dtype=torch.float32), torch.as_tensor(d, dtype=torch.long),
                  torch.as_tensor(y, dtype=torch.long))
    bd = np.repeat(np.arange(len(levels)), 2 * cfg.per_cell)
    by = np.tile(np.repeat([0, 1], cfg.per_cell), len(levels))
    matrix = alignment_matrix(bd, by)
    history, snapshots = [], {}

    def run(steps: int, method: str, strength: float, sampler: CellSampler, phase: str):
        optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        critic_opt = torch.optim.AdamW(model.critics.parameters(), lr=cfg.critic_lr, weight_decay=cfg.weight_decay)
        for step in range(steps + 1):
            if phase == "main" and step in snapshot_steps:
                snapshots[step] = [encode(model, xi) for xi in snapshot_inputs]
            if step == steps:
                break
            ids = torch.as_tensor(sampler.draw())
            dd, yy = dt[ids], yl[ids]
            critic_ce = torch.zeros(())
            if method != "task":
                for p in model.critics.parameters():
                    p.requires_grad_(True)
                model.eval()
                with torch.no_grad():
                    detached = _standardize(model.encoder(xt), w)[ids]
                for _ in range(cfg.critic_steps):
                    critic_opt.zero_grad(set_to_none=True)
                    critic_ce = critic_loss(model, detached, dd, yy, confuse=False)
                    critic_ce.backward()
                    torch.nn.utils.clip_grad_norm_(model.critics.parameters(), 5.0)
                    critic_opt.step()
            for p in model.critics.parameters():
                p.requires_grad_(False)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            z = model.encoder(xt)
            task = F.binary_cross_entropy_with_logits(model.depression(z[ids]).squeeze(-1), yt[ids])
            penalty = representation_penalty(z, w)
            zero = z.sum() * 0
            confusion, mmd, supcon = zero, zero, zero
            if method != "task":
                normalized = _standardize(z, w)
                confusion = critic_loss(model, normalized[ids], dd, yy, confuse=True)
                if method == "combined":
                    mmd = mmd_loss(normalized[ids], matrix)
                    supcon = cross_source_contrastive(z[ids], dd, yy, ids, cfg.temperature)
            ramp = min(1.0, (step + 1) / cfg.ramp_steps)
            loss = task + penalty + ramp * (strength * (confusion + mmd) + cfg.contrastive_weight * supcon)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite encoder loss.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            optimizer.step()
            if (step + 1) % 100 == 0:
                history.append({"phase": phase, "step": step + 1, "bce": float(task), "critic_ce": float(critic_ce),
                                "confusion": float(confusion), "mmd": float(mmd), "supcon": float(supcon),
                                "representation_penalty": float(penalty), "total": float(loss)})

    warm_sampler = CellSampler(d, y, cfg.per_cell, cfg.seed + 4831)
    run(cfg.warmup_steps, "task", 0.0, warm_sampler, "warmup")
    sampler = CellSampler(d, y, cfg.per_cell, cfg.seed + 9173)
    run(cfg.steps, cfg.method, cfg.strength, sampler, "main")
    return {"model": model, "levels": levels, "history": history, "snapshots": snapshots,
            "draws_per_subject": {"min": int(sampler.draws.min()), "max": int(sampler.draws.max())}}


def two_stage(x_train: np.ndarray, y: np.ndarray, groups: list[str], x_holdout: np.ndarray, folds: list[np.ndarray],
              seed: int, variants=VARIANT_GRID, step_grid=STEP_GRID, l2_grid=DEFAULT_L2_GRID,
              base: ConfusionConfig = ConfusionConfig()) -> dict[str, Any]:
    """Inner-fold selection of (step, L2) per variant, then the final encoder + L2 head per variant.

    ``x_train`` / ``x_holdout`` are the globally standardized subject vectors; the PCA is refit on every
    encoder's own training subjects.
    """
    n = len(y)
    g = np.asarray(groups)
    out: dict[str, Any] = {}
    for method, strength in variants:
        cfg = replace(base, method=method, strength=strength, steps=max(step_grid))
        oof = {(s, p): np.zeros(n) for s in step_grid for p in l2_grid}
        for k, held in enumerate(folds):
            keep = np.setdiff1d(np.arange(n), held)
            pca = fit_pca(x_train[keep], g[keep], y[keep], cfg.pca_dims)
            run = train_confusion(apply_pca(pca, x_train[keep]), y[keep], list(g[keep]),
                                  replace(cfg, seed=seed * 100 + k + 1), step_grid,
                                  (apply_pca(pca, x_train[keep]), apply_pca(pca, x_train[held])))
            for s in step_grid:
                z_keep, z_held = run["snapshots"][s]
                for p in l2_grid:
                    (logits,), _ = l2_head_logits(z_keep, y[keep], list(g[keep]), [z_held], p)
                    oof[(s, p)][held] = logits
        candidates = []
        for (s, p), logits in oof.items():
            f1, bce = _criterion(y, logits, groups)
            candidates.append({"step": s, "l2_penalty": p, "mean_dataset_macro_f1": f1, "binary_cross_entropy": bce})
        best = max(candidates, key=lambda c: (c["mean_dataset_macro_f1"], -c["binary_cross_entropy"],
                                              c["l2_penalty"], -c["step"]))
        pca = fit_pca(x_train, g, y, cfg.pca_dims)
        p_train, p_holdout = apply_pca(pca, x_train), apply_pca(pca, x_holdout)
        final = train_confusion(p_train, y, groups, replace(cfg, seed=seed), (best["step"],), (p_train, p_holdout))
        z_train, z_holdout = final["snapshots"][best["step"]]
        (train_logits, holdout_logits), head_info = l2_head_logits(z_train, y, groups, [z_train, z_holdout],
                                                                   best["l2_penalty"])
        out[variant_name(method, strength)] = {
            "config": asdict(replace(cfg, seed=seed)),
            "selection": {"selected_step": best["step"], "selected_l2_penalty": best["l2_penalty"],
                          "inner_mean_dataset_macro_f1": best["mean_dataset_macro_f1"],
                          "inner_binary_cross_entropy": best["binary_cross_entropy"], "candidates": candidates,
                          "rule": SELECTION_RULE},
            "stage1": {"history": final["history"], "draws_per_subject": final["draws_per_subject"],
                       "pca_dims": int(pca["basis"].shape[1])},
            "head_fit_info": head_info,
            "holdout_logits": holdout_logits, "z_train": z_train, "z_holdout": z_holdout,
            "train_macro_f1": score_metrics(y, train_logits)["macro_f1"],
        }
    return out


def fresh_attacks(z_train: np.ndarray, g_train: list[str], y_train: np.ndarray, z_test: np.ndarray,
                  g_test: list[str], y_test: np.ndarray, seed: int = 808) -> dict[str, Any]:
    """Dataset recovery from ``z`` by fresh attackers fit on train subjects, scored on held-out subjects.

    Families: L2 logistic, RBF SVM, MLP (64 GELU, 300 AdamW steps). Groups: all subjects (dataset x class
    cell weights), then y=0 only and y=1 only (equal dataset weights). C in {0.01, 1, 100} for the logistic
    and SVM is chosen by stratified 3-fold CV on the train subjects. Score: balanced accuracy over datasets.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    levels = sorted(set(g_train))
    dtr = np.asarray([levels.index(v) for v in g_train])
    dte = np.asarray([levels.index(v) for v in g_test])
    out: dict[str, Any] = {}
    for group in ("all", "y0", "y1"):
        a = np.ones(len(dtr), bool) if group == "all" else y_train == int(group[1])
        b = np.ones(len(dte), bool) if group == "all" else y_test == int(group[1])
        if len(set(dte[b].tolist())) < len(levels):
            continue
        w = cell_weights(dtr[a], y_train[a]) if group == "all" else \
            np.asarray([1.0 / (len(levels) * np.sum(dtr[a] == v)) for v in dtr[a]])
        scaler = StandardScaler().fit(z_train[a], sample_weight=w)
        xa, xb = scaler.transform(z_train[a]), scaler.transform(z_test[b])
        for family in ("linear", "rbf", "mlp"):
            if family == "mlp":
                torch.manual_seed(seed)
                net = nn.Sequential(nn.Linear(xa.shape[1], 64), nn.GELU(), nn.Linear(64, len(levels)))
                opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-2)
                xt, dt_, wt = (torch.as_tensor(xa, dtype=torch.float32), torch.as_tensor(dtr[a]),
                               torch.as_tensor(w, dtype=torch.float32))
                for _ in range(300):
                    opt.zero_grad()
                    (wt * F.cross_entropy(net(xt), dt_, reduction="none")).sum().backward()
                    opt.step()
                with torch.no_grad():
                    pred = net(torch.as_tensor(xb, dtype=torch.float32)).argmax(1).numpy()
                chosen = None
            else:
                def make(c):
                    return (LogisticRegression(C=c, max_iter=5000) if family == "linear"
                            else SVC(C=c, kernel="rbf", gamma="scale"))
                cv_scores = {}
                for c in (0.01, 1.0, 100.0):
                    preds = np.zeros(int(a.sum()), dtype=int)
                    for i_tr, i_va in StratifiedKFold(3, shuffle=True, random_state=seed).split(xa, dtr[a]):
                        preds[i_va] = make(c).fit(xa[i_tr], dtr[a][i_tr], sample_weight=w[i_tr] * len(i_tr)).predict(xa[i_va])
                    cv_scores[c] = balanced_accuracy_score(dtr[a], preds)
                chosen = max(cv_scores, key=lambda c: (cv_scores[c], -c))
                pred = make(chosen).fit(xa, dtr[a], sample_weight=w * len(w)).predict(xb)
            out[f"{group}_{family}"] = {"balanced_accuracy": float(balanced_accuracy_score(dte[b], pred)), "C": chosen}
    out["max"] = max(v["balanced_accuracy"] for v in out.values())
    out["chance"] = 1.0 / len(levels)
    return out


def fit_merged_fold(x_train, train_rows, x_holdout, holdout_rows, seed: int, variants=VARIANT_GRID,
                    attacks: bool = True) -> dict[str, Any]:
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
    result = two_stage(xs_train, y_train, g_train, xs_holdout, folds, seed, variants)
    can_attack = attacks and len(set(g_holdout)) == len(set(g_train))
    reference = None
    if can_attack:
        pca = fit_pca(xs_train, np.asarray(g_train), y_train, ConfusionConfig().pca_dims)
        reference = {"raw_standardized": fresh_attacks(xs_train, g_train, y_train, xs_holdout, g_holdout, y_holdout),
                     "pca64_input": fresh_attacks(apply_pca(pca, xs_train), g_train, y_train,
                                                  apply_pca(pca, xs_holdout), g_holdout, y_holdout)}
    variants_out = {}
    for name, r in result.items():
        logits = r.pop("holdout_logits")
        z_train, z_holdout = r.pop("z_train"), r.pop("z_holdout")
        r["dataset_attacks"] = (fresh_attacks(z_train, g_train, y_train, z_holdout, g_holdout, y_holdout)
                                if can_attack else None)
        r["predictions"] = [
            {"dataset": ho.dataset[s], "subject_id": s, "label": ho.labels[s], "prediction": int(v > 0),
             "probability": 1.0 / (1.0 + math.exp(-v)), "mean_logit": v, "num_units": len(ho.units[s])}
            for s, v in zip(holdout, logits.tolist())]
        r["metrics_by_dataset"] = {d: score_metrics(y_holdout[np.asarray(g_holdout) == d],
                                                    logits[np.asarray(g_holdout) == d]) for d in sorted(set(g_holdout))}
        variants_out[name] = r
    return {"variants": variants_out, "reference_attacks": reference, "inner_folds": INNER_FOLDS,
            "inner_seed": INNER_SEED, "inner_assignments_hash": inner.get("assignments_hash")}
