"""Two ways to factorize the ratings matrix: SVD from scratch, and FunkSVD."""
import numpy as np


def power_iteration(gram, tol=1e-10, max_iter=10_000, rng=None):
    rng = rng or np.random.default_rng(0)
    v = rng.random(gram.shape[0])
    v /= np.linalg.norm(v)
    for _ in range(max_iter):
        w = gram @ v
        w /= np.linalg.norm(w)
        if np.linalg.norm(v - w) < tol:
            return w
        v = w
    return v


def power_svd(A, k, seed=0):
    rng = np.random.default_rng(seed)
    residual = A.astype(float).copy()
    U, S, Vt = [], [], []
    for _ in range(min(k, min(A.shape))):
        v = power_iteration(residual.T @ residual, rng=rng)
        sigma = np.linalg.norm(residual @ v)
        if sigma < 1e-12:
            break
        u = residual @ v / sigma
        residual -= sigma * np.outer(u, v)
        U.append(u), S.append(sigma), Vt.append(v)
    return np.array(U).T, np.array(S), np.array(Vt)


def reconstruct(U, S, Vt, k=None):
    k = k or len(S)
    return (U[:, :k] * S[:k]) @ Vt[:k]


def svd_predict(R, mask, k, use_numpy=False):
    """Truncated-SVD ratings, missing entries filled with the hotel mean."""
    col_mean = (R * mask).sum(0) / np.maximum(mask.sum(0), 1)
    centered = np.where(mask, R, col_mean) - col_mean
    if use_numpy:
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        approx = reconstruct(U, S, Vt, k)
    else:
        approx = reconstruct(*power_svd(centered, k))
    return approx + col_mean


class FunkSVD:
    """Latent factors learned by SGD over the observed ratings only."""

    def __init__(self, n_factors=8, lr=0.01, reg=0.05, epochs=120, seed=0):
        self.n_factors, self.lr, self.reg, self.epochs, self.seed = n_factors, lr, reg, epochs, seed
        self.history = []

    def fit(self, R, train, val=None):
        rng = np.random.default_rng(self.seed)
        n_users, n_items = R.shape
        self.P = rng.normal(0, 0.1, (n_users, self.n_factors))
        self.Q = rng.normal(0, 0.1, (n_items, self.n_factors))
        self.bu, self.bi = np.zeros(n_users), np.zeros(n_items)
        self.mu = R[train].mean()
        rows, cols = np.nonzero(train)

        for _ in range(self.epochs):
            for n in rng.permutation(len(rows)):
                u, i = rows[n], cols[n]
                err = R[u, i] - self.predict(u, i)
                self.bu[u] += self.lr * (err - self.reg * self.bu[u])
                self.bi[i] += self.lr * (err - self.reg * self.bi[i])
                pu = self.P[u].copy()
                self.P[u] += self.lr * (err * self.Q[i] - self.reg * pu)
                self.Q[i] += self.lr * (err * pu - self.reg * self.Q[i])
            entry = {"epoch": len(self.history) + 1, "train_rmse": self._rmse(R, train)}
            if val is not None:
                entry["val_rmse"] = self._rmse(R, val)
            self.history.append(entry)
        return self

    def predict(self, u, i):
        return self.mu + self.bu[u] + self.bi[i] + self.P[u] @ self.Q[i]

    def predict_matrix(self):
        return self.mu + self.bu[:, None] + self.bi[None, :] + self.P @ self.Q.T

    def _rmse(self, R, mask):
        return np.sqrt(((R - self.predict_matrix())[mask] ** 2).mean())
