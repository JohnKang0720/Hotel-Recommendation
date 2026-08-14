from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

INK, MUTE, GRID = "#141414", "#8a8a8a", "#e6e6e6"
CLUSTERS = ["#4c566a", "#5e81ac", "#8fa876", "#b4869f", "#c08457", "#7a8a99"]


def _ax(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTE)
    ax.tick_params(colors=MUTE, labelsize=9)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def _save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def scree(S, path):
    fig, ax = plt.subplots(figsize=(7, 4)); _ax(ax)
    ax.plot(range(1, len(S) + 1), S, marker="o", color=INK, lw=1.6, ms=5)
    ax.set_xlabel("Component"); ax.set_ylabel("Singular value")
    ax.set_title("Scree plot — the signal lives in a few factors", color=INK, fontsize=12, loc="left")
    _save(fig, path)


def latent_map(item_factors, hotels, clusters, path, n_labels=8):
    coords = PCA(2, random_state=0).fit_transform(item_factors)
    fig, ax = plt.subplots(figsize=(7.5, 5.5)); _ax(ax)
    for c in range(clusters.max() + 1):
        m = clusters == c
        ax.scatter(coords[m, 0], coords[m, 1], s=75, color=CLUSTERS[c % len(CLUSTERS)],
                   edgecolor="white", linewidth=0.8, label=f"Cluster {c}")
    for i in np.random.default_rng(0).choice(len(hotels), n_labels, replace=False):
        ax.annotate(hotels[i], (coords[i, 0], coords[i, 1]), fontsize=7.5, color=INK,
                    xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Latent factor 1"); ax.set_ylabel("Latent factor 2")
    ax.set_title("Hotels in latent space — similar hotels sit together", color=INK, fontsize=12, loc="left")
    ax.legend(frameon=False, fontsize=8)
    _save(fig, path)


def completion(R, mask, pred, path, n=40):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, mat, title in ((axes[0], np.where(mask, R, np.nan)[:n, :n], "Observed (sparse)"),
                           (axes[1], pred[:n, :n], "Reconstructed (filled)")):
        im = ax.imshow(mat, cmap="Greys", vmin=1, vmax=5, aspect="auto")
        ax.set_title(title, color=INK, fontsize=11, loc="left")
        ax.set_xlabel("Hotels"); ax.set_ylabel("Users"); ax.tick_params(colors=MUTE, labelsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Matrix completion — SVD fills the blanks", color=INK, fontsize=12, x=0.12, ha="left")
    _save(fig, path)


def reconstruction_error(A, path, max_k=25):
    from .svd import power_svd, reconstruct
    U, S, Vt = power_svd(A, min(max_k, min(A.shape)))
    errs = [np.linalg.norm(A - reconstruct(U, S, Vt, k)) / np.linalg.norm(A) for k in range(1, len(S) + 1)]
    fig, ax = plt.subplots(figsize=(7, 4)); _ax(ax)
    ax.plot(range(1, len(errs) + 1), errs, marker="o", color=INK, lw=1.6, ms=5)
    ax.set_xlabel("Rank k"); ax.set_ylabel("Relative Frobenius error")
    ax.set_title("Rank-k reconstruction error", color=INK, fontsize=12, loc="left")
    _save(fig, path)


def convergence(A, path, max_k=20):
    from .svd import power_svd
    _, ours, _ = power_svd(A, min(max_k, min(A.shape)))
    numpy_s = np.linalg.svd(A, compute_uv=False)[:len(ours)]
    fig, ax = plt.subplots(figsize=(7, 4)); _ax(ax)
    x = range(1, len(ours) + 1)
    ax.plot(x, numpy_s, marker="o", color=INK, lw=1.6, ms=6, label="NumPy SVD")
    ax.plot(x, ours, marker="x", color=MUTE, lw=1.4, ms=7, label="Power iteration (ours)")
    ax.set_xlabel("Component"); ax.set_ylabel("Singular value")
    ax.set_title("From-scratch SVD matches NumPy", color=INK, fontsize=12, loc="left")
    ax.legend(frameon=False, fontsize=9)
    _save(fig, path)


def learning_curve(history, path):
    epochs = [h["epoch"] for h in history]
    fig, ax = plt.subplots(figsize=(7, 4)); _ax(ax)
    ax.plot(epochs, [h["train_rmse"] for h in history], color=INK, lw=1.8, label="Train RMSE")
    if "val_rmse" in history[0]:
        ax.plot(epochs, [h["val_rmse"] for h in history], color=MUTE, lw=1.8, ls="--", label="Held-out RMSE")
    ax.set_xlabel("Epoch"); ax.set_ylabel("RMSE")
    ax.set_title("FunkSVD learning curve", color=INK, fontsize=12, loc="left")
    ax.legend(frameon=False, fontsize=9)
    _save(fig, path)
