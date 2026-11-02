"""Closed-form linear concept erasure (LEACE, Belrose et al., 2023) for small-n, high-d features.

Given features ``x`` (n x d), a categorical concept ``groups`` (n,) and sample
weights, LEACE finds the affine edit ``r(x) = x - A B^T (x - mean)`` with the
smallest expected change (in the whitened metric) after which the weighted
concept-conditional means of ``r(x)`` are equal. No linear classifier can then
beat a constant predictor of the concept on the fitting distribution.

With n < d the covariance is singular, so the fit works in the span of the
weighted, centered training data (thin SVD ``sqrt(w) X_c = U S V^T``)::

    W     = V S^-1 V^T       (pseudo-inverse square root of the covariance)
    W^+   = V S V^T
    P     = projection onto colspan(W Sigma_xz) = colspan(V Q),  Q = orth(U^T Z_w)
    W^+ P W = (V S Q)(V S^-1 Q)^T  =:  A B^T

Directions outside the training span are left unchanged. The eraser is fit on
training rows only and needs no concept label at prediction time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LeaceEraser:
    mean: np.ndarray       # (d,)
    a: np.ndarray          # (d, q)  = V S Q
    b: np.ndarray          # (d, q)  = V S^-1 Q
    levels: tuple[str, ...]
    rank: int              # rank of the training covariance kept
    erased_dims: int       # q

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        return x - ((x - self.mean) @ self.b) @ self.a.T


def fit_leace(x: np.ndarray, groups, weights=None, rel_tol: float = 1e-10,
              concept_tol: float = 1e-8) -> LeaceEraser:
    x = np.asarray(x, dtype=np.float64)
    groups = np.asarray([str(g) for g in groups])
    n, d = x.shape
    if len(groups) != n:
        raise ValueError("groups must have one entry per row.")
    levels = tuple(sorted(set(groups.tolist())))
    if len(levels) < 2:
        raise ValueError("LEACE needs at least two concept levels.")
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=np.float64)
    if w.shape != (n,) or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("weights must be non-negative with a positive sum.")
    w = w / w.sum()
    mean = w @ x
    z = (groups[:, None] == np.asarray(levels)[None, :]).astype(np.float64)
    z_c = z - w @ z
    sw = np.sqrt(w)[:, None]
    u, s, vt = np.linalg.svd(sw * (x - mean), full_matrices=False)
    keep = s > rel_tol * (s[0] if len(s) else 0.0)
    u, s, v = u[:, keep], s[keep], vt[keep].T
    m = u.T @ (sw * z_c)                      # (r, k): whitened cross-covariance, in V coordinates
    mu, ms, _ = np.linalg.svd(m, full_matrices=False)
    # m is a whitened cross-covariance (entries bounded by the concept's std), so an absolute
    # threshold is meaningful; a relative one would erase pure noise once the concept is gone.
    q = mu[:, ms > concept_tol]
    a = v @ (s[:, None] * q)
    b = v @ (q / s[:, None])
    return LeaceEraser(mean=mean, a=a, b=b, levels=levels, rank=int(keep.sum()), erased_dims=int(q.shape[1]))


def max_group_mean_gap(x: np.ndarray, groups, weights=None) -> float:
    """Largest absolute difference between a weighted group mean and the weighted overall mean."""
    x = np.asarray(x, dtype=np.float64)
    groups = np.asarray([str(g) for g in groups])
    w = np.ones(len(x)) if weights is None else np.asarray(weights, dtype=np.float64)
    overall = (w @ x) / w.sum()
    gaps = [np.max(np.abs((w[groups == g] @ x[groups == g]) / w[groups == g].sum() - overall))
            for g in sorted(set(groups.tolist()))]
    return float(max(gaps))
