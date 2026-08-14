"""Figures. Every plot here is the honest output of the models in this repo.

The star is ``latent_map`` — the fix for the original notebook's broken
"two straight lines" plot. That one collapsed because it fed a hyper-sparse,
zero-filled matrix straight into a scatter with buggy axes. Here we plot the
*learned* hotel factors, projected to 2-D, colored by cluster — real structure.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

INK = "#141414"
MUTE = "#8a8a8a"
GRID = "#e6e6e6"
# Muted, distinguishable categorical palette (used only where color encodes a group).
CLUSTER_COLORS = ["#4c566a", "#5e81ac", "#8fa876", "#b4869f", "#c08457", "#7a8a99"]


def _style(ax):
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTE)
    ax.tick_params(colors=MUTE, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _save(fig, path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def scree(S: np.ndarray, path: str | Path):
    """Singular-value spectrum — how many latent factors actually matter."""
    fig, ax = plt.subplots(figsize=(7, 4))
    _style(ax)
    x = np.arange(1, len(S) + 1)
    ax.plot(x, S, marker="o", color=INK, lw=1.6, ms=5)
    ax.set_xlabel("Component"); ax.set_ylabel("Singular value")
    ax.set_title("Scree plot — the signal lives in a few factors", color=INK, fontsize=12, loc="left")
    _save(fig, path)


def latent_map(item_factors: np.ndarray, hotels: np.ndarray, clusters: np.ndarray,
               path: str | Path, n_labels: int = 8):
    """Hotels projected to 2-D latent space, colored by cluster (the real fix)."""
    coords = PCA(n_components=2, random_state=0).fit_transform(item_factors) \
        if item_factors.shape[1] > 2 else item_factors[:, :2]

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    _style(ax)
    for c in range(clusters.max() + 1):
        m = clusters == c
        ax.scatter(coords[m, 0], coords[m, 1], s=75,
                   color=CLUSTER_COLORS[c % len(CLUSTER_COLORS)],
                   edgecolor="white", linewidth=0.8, label=f"Cluster {c}")

    rng = np.random.default_rng(0)
    for i in rng.choice(len(hotels), size=min(n_labels, len(hotels)), replace=False):
        ax.annotate(str(hotels[i]), (coords[i, 0], coords[i, 1]),
                    fontsize=7.5, color=INK, xytext=(4, 4), textcoords="offset points")

    ax.set_xlabel("Latent factor 1"); ax.set_ylabel("Latent factor 2")
    ax.set_title("Hotels in latent space — similar hotels sit together", color=INK, fontsize=12, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="best")
    _save(fig, path)


def completion(R: np.ndarray, mask: np.ndarray, pred: np.ndarray, path: str | Path,
               n: int = 40):
    """Sparse observed ratings vs the model's fully-filled reconstruction."""
    R_obs = np.where(mask, R, np.nan)[:n, :n]
    R_hat = pred[:n, :n]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, mat, title in ((axes[0], R_obs, "Observed (sparse)"),
                           (axes[1], R_hat, "Reconstructed (filled)")):
        im = ax.imshow(mat, cmap="Greys", vmin=1, vmax=5, aspect="auto")
        ax.set_title(title, color=INK, fontsize=11, loc="left")
        ax.set_xlabel("Hotels"); ax.set_ylabel("Users")
        ax.tick_params(colors=MUTE, labelsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Matrix completion — SVD fills the blanks", color=INK, fontsize=12, x=0.12, ha="left")
    _save(fig, path)


def reconstruction_error(A: np.ndarray, path: str | Path, max_k: int = 25):
    """Rank-k reconstruction error — how fast low-rank captures the matrix."""
    from .svd import power_svd, reconstruct
    max_k = min(max_k, min(A.shape))
    U, S, Vt = power_svd(A, max_k)
    errs = [np.linalg.norm(A - reconstruct(U, S, Vt, k)) / np.linalg.norm(A)
            for k in range(1, len(S) + 1)]

    fig, ax = plt.subplots(figsize=(7, 4))
    _style(ax)
    ax.plot(range(1, len(errs) + 1), errs, marker="o", color=INK, lw=1.6, ms=5)
    ax.set_xlabel("Rank k"); ax.set_ylabel("Relative Frobenius error")
    ax.set_title("Rank-k reconstruction error", color=INK, fontsize=12, loc="left")
    _save(fig, path)


def convergence(A: np.ndarray, path: str | Path, max_k: int = 20):
    """Our from-scratch singular values vs NumPy's — the correctness check."""
    from .svd import power_svd
    max_k = min(max_k, min(A.shape))
    _, S_ours, _ = power_svd(A, max_k)
    S_np = np.linalg.svd(A, compute_uv=False)[:len(S_ours)]

    fig, ax = plt.subplots(figsize=(7, 4))
    _style(ax)
    x = np.arange(1, len(S_ours) + 1)
    ax.plot(x, S_np, marker="o", color=INK, lw=1.6, ms=6, label="NumPy SVD")
    ax.plot(x, S_ours, marker="x", color=MUTE, lw=1.4, ms=7, label="Power iteration (ours)")
    ax.set_xlabel("Component"); ax.set_ylabel("Singular value")
    ax.set_title("From-scratch SVD matches NumPy", color=INK, fontsize=12, loc="left")
    ax.legend(frameon=False, fontsize=9)
    _save(fig, path)


def learning_curve(history: list, path: str | Path):
    """FunkSVD train/validation RMSE per epoch."""
    epochs = [h["epoch"] for h in history]
    fig, ax = plt.subplots(figsize=(7, 4))
    _style(ax)
    ax.plot(epochs, [h["train_rmse"] for h in history], color=INK, lw=1.8, label="Train RMSE")
    if "val_rmse" in history[0]:
        ax.plot(epochs, [h["val_rmse"] for h in history], color=MUTE, lw=1.8, ls="--", label="Held-out RMSE")
    ax.set_xlabel("Epoch"); ax.set_ylabel("RMSE")
    ax.set_title("FunkSVD learning curve", color=INK, fontsize=12, loc="left")
    ax.legend(frameon=False, fontsize=9)
    _save(fig, path)
