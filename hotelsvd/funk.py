"""FunkSVD — matrix factorization by SGD over the observed ratings only.

Plain SVD has to invent values for every empty cell before it can factorize.
FunkSVD sidesteps that: it only ever looks at ratings that actually exist, and
learns user/hotel factors (plus biases) that reconstruct them. That single
change is what fixes the original notebook, where filling blanks with 0 pulled
every prediction toward zero.

Model:   r̂(u, i) = μ + b_u + b_i + pᵤ · qᵢ
Loss:    Σ_observed (r − r̂)² + λ(‖p‖² + ‖q‖² + b_u² + b_i²)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class FunkSVD:
    n_factors: int = 8
    lr: float = 0.01
    reg: float = 0.05
    epochs: int = 120
    seed: int = 0

    # learned parameters (filled by fit)
    P: np.ndarray = field(default=None, init=False)      # (n_users, n_factors)
    Q: np.ndarray = field(default=None, init=False)      # (n_hotels, n_factors)
    bu: np.ndarray = field(default=None, init=False)
    bi: np.ndarray = field(default=None, init=False)
    mu: float = field(default=0.0, init=False)
    history: list = field(default_factory=list, init=False)   # per-epoch RMSE

    def fit(self, R: np.ndarray, train_mask: np.ndarray,
            val_mask: np.ndarray | None = None) -> "FunkSVD":
        rng = np.random.default_rng(self.seed)
        n_users, n_items = R.shape

        self.P = rng.normal(0, 0.1, (n_users, self.n_factors))
        self.Q = rng.normal(0, 0.1, (n_items, self.n_factors))
        self.bu = np.zeros(n_users)
        self.bi = np.zeros(n_items)
        self.mu = float(R[train_mask].mean())

        rows, cols = np.nonzero(train_mask)
        idx = np.arange(len(rows))

        for _ in range(self.epochs):
            rng.shuffle(idx)
            for n in idx:
                u, i = rows[n], cols[n]
                pred = self.mu + self.bu[u] + self.bi[i] + self.P[u] @ self.Q[i]
                err = R[u, i] - pred

                self.bu[u] += self.lr * (err - self.reg * self.bu[u])
                self.bi[i] += self.lr * (err - self.reg * self.bi[i])
                pu, qi = self.P[u].copy(), self.Q[i]
                self.P[u] += self.lr * (err * qi - self.reg * pu)
                self.Q[i] += self.lr * (err * pu - self.reg * qi)

            entry = {"epoch": len(self.history) + 1,
                     "train_rmse": self._rmse(R, train_mask)}
            if val_mask is not None and val_mask.any():
                entry["val_rmse"] = self._rmse(R, val_mask)
            self.history.append(entry)

        return self

    def predict_matrix(self) -> np.ndarray:
        """Full predicted-rating matrix r̂(u, i)."""
        return self.mu + self.bu[:, None] + self.bi[None, :] + self.P @ self.Q.T

    def _rmse(self, R: np.ndarray, mask: np.ndarray) -> float:
        pred = self.predict_matrix()
        diff = (R - pred)[mask]
        return float(np.sqrt(np.mean(diff ** 2))) if diff.size else float("nan")
