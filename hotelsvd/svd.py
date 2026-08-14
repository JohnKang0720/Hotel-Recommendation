"""SVD from scratch via power iteration + deflation.

This is my original notebook idea, cleaned up. The method:

1. Find the top right-singular vector of ``A`` by running power iteration on the
   Gram matrix ``AᵀA`` (its leading eigenvector).
2. Recover the matching singular value and left vector: ``σ = ‖Av‖``, ``u = Av/σ``.
3. **Deflate** — subtract that rank-1 piece (``A ← A − σ u vᵀ``) and repeat for
   the next component.

Running it on the small ``AᵀA`` (hotels × hotels) keeps each iteration cheap.
``power_svd`` is validated against ``numpy.linalg.svd`` in the tests.
"""
from __future__ import annotations

import numpy as np


def initialize_vector(n: int, rng: np.random.Generator) -> np.ndarray:
    """A random unit vector of length ``n`` (the power-iteration seed)."""
    v = rng.random(n)
    return v / np.linalg.norm(v)


def power_iteration(gram: np.ndarray, tol: float = 1e-10, max_iter: int = 10_000,
                    rng: np.random.Generator | None = None) -> np.ndarray:
    """Leading eigenvector of a symmetric PSD matrix via power iteration."""
    rng = rng or np.random.default_rng(0)
    v = initialize_vector(gram.shape[0], rng)
    for _ in range(max_iter):
        w = gram @ v
        w_norm = np.linalg.norm(w)
        if w_norm == 0:
            break
        w = w / w_norm
        if np.linalg.norm(v - w) < tol:
            v = w
            break
        v = w
    return v


def power_svd(A: np.ndarray, k: int, tol: float = 1e-10, seed: int = 0
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Truncated SVD ``A ≈ U diag(S) Vᵀ`` from scratch.

    Returns ``U`` (m×k), ``S`` (k,), ``Vt`` (k×n), largest singular value first.
    """
    rng = np.random.default_rng(seed)
    residual = A.astype(float).copy()
    k = min(k, min(A.shape))

    us, ss, vts = [], [], []
    for _ in range(k):
        gram = residual.T @ residual            # (n_hotels, n_hotels) — small
        v = power_iteration(gram, tol=tol, rng=rng)   # right singular vector
        Av = residual @ v
        sigma = np.linalg.norm(Av)
        if sigma < 1e-12:
            break                                # residual is (numerically) rank-exhausted
        u = Av / sigma
        residual = residual - sigma * np.outer(u, v)   # deflate

        us.append(u)
        ss.append(sigma)
        vts.append(v)

    U = np.array(us).T if us else np.zeros((A.shape[0], 0))
    S = np.array(ss)
    Vt = np.array(vts) if vts else np.zeros((0, A.shape[1]))
    return U, S, Vt


def reconstruct(U: np.ndarray, S: np.ndarray, Vt: np.ndarray, k: int | None = None) -> np.ndarray:
    """Rank-``k`` reconstruction ``U diag(S) Vᵀ`` (all components if ``k`` is None)."""
    if k is None:
        k = len(S)
    return (U[:, :k] * S[:k]) @ Vt[:k]


def svd_predict(R: np.ndarray, mask: np.ndarray, k: int, use_numpy: bool = False
               ) -> np.ndarray:
    """Rating predictions from a truncated SVD of the mean-imputed matrix.

    Missing entries are filled with the hotel (column) mean before factorizing —
    the standard way to make plain SVD usable on sparse ratings. This is the
    baseline that FunkSVD (observed-only) is meant to beat on held-out ranking.
    """
    col_mean = np.where(mask.any(axis=0), (R * mask).sum(0) / np.maximum(mask.sum(0), 1), R[mask].mean())
    filled = np.where(mask, R, col_mean)
    centered = filled - col_mean

    if use_numpy:
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        approx = reconstruct(U, S, Vt, k)
    else:
        U, S, Vt = power_svd(centered, k)
        approx = reconstruct(U, S, Vt)

    return approx + col_mean
