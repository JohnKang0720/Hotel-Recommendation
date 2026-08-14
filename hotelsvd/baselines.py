"""Baselines to measure the factorizations against.

* ``item_based_cf`` — neighborhood collaborative filtering (my original
  item-based idea: cosine similarity between hotels, predict by similar hotels).
* ``cluster_hotels`` — K-Means over learned hotel factors. This is the "group
  similar hotels, recommend within the group" baseline, and it doubles as the
  coloring for the latent-space map.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans


def item_based_cf(R: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    """Predict ratings from cosine similarity between hotels (item-item CF)."""
    # Mean-center each hotel over its observed ratings, then cosine-similarity.
    item_mean = np.where(train_mask.any(0), (R * train_mask).sum(0) / np.maximum(train_mask.sum(0), 1), 0.0)
    centered = np.where(train_mask, R - item_mean, 0.0)

    norms = np.linalg.norm(centered, axis=0)
    denom = np.outer(norms, norms)
    sim = (centered.T @ centered) / np.where(denom == 0, 1, denom)
    np.fill_diagonal(sim, 0.0)

    # Predicted deviation = similarity-weighted average of the user's observed deviations.
    user_dev = np.where(train_mask, R - item_mean, 0.0)
    weight = train_mask.astype(float)
    num = user_dev @ sim
    den = weight @ np.abs(sim)
    pred_dev = np.divide(num, den, out=np.zeros_like(num), where=den != 0)
    return item_mean + pred_dev


def cluster_hotels(item_factors: np.ndarray, n_clusters: int = 5, seed: int = 0) -> np.ndarray:
    """K-Means labels for hotels in latent-factor space."""
    n_clusters = min(n_clusters, item_factors.shape[0])
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    return km.fit_predict(item_factors)
