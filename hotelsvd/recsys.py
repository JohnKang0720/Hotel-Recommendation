"""Data, baselines, metrics, and recommendations."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

DATA = Path(__file__).resolve().parent.parent / "data" / "hotel_ratings.csv"


@dataclass
class Ratings:
    R: np.ndarray
    mask: np.ndarray
    users: np.ndarray
    hotels: np.ndarray
    cities: np.ndarray

    @property
    def shape(self):
        return self.R.shape

    @property
    def density(self):
        return self.mask.mean()


def load(path=DATA):
    df = pd.read_csv(path)
    pivot = df.pivot_table(index="user", columns="hotel", values="rating")
    hotels = pivot.columns.to_numpy()
    cities = df.drop_duplicates("hotel").set_index("hotel")["city"].reindex(hotels).to_numpy()
    return Ratings(np.nan_to_num(pivot.to_numpy()), pivot.notna().to_numpy(),
                   pivot.index.to_numpy(), hotels, cities)


def split(mask, test_frac=0.2, seed=0):
    """Hide a fraction of each user's observed ratings for testing."""
    rng = np.random.default_rng(seed)
    train, test = mask.copy(), np.zeros_like(mask)
    for u in range(mask.shape[0]):
        rated = np.flatnonzero(mask[u])
        if len(rated) < 3:
            continue
        held = rng.choice(rated, max(1, round(len(rated) * test_frac)), replace=False)
        train[u, held], test[u, held] = False, True
    return train, test


def item_cf(R, train):
    mean = (R * train).sum(0) / np.maximum(train.sum(0), 1)
    centered = np.where(train, R - mean, 0)
    norms = np.linalg.norm(centered, axis=0)
    sim = centered.T @ centered / np.outer(norms, norms).clip(1e-9)
    np.fill_diagonal(sim, 0)
    return mean + (centered @ sim) / (train @ np.abs(sim)).clip(1e-9)


def cluster(item_factors, n=5):
    return KMeans(n, n_init=10, random_state=0).fit_predict(item_factors)


def rmse(pred, R, mask):
    return np.sqrt(((R - pred)[mask] ** 2).mean())


def ranking_metrics(pred, R, train, test, k=10, thr=4):
    gains = 1 / np.log2(np.arange(2, k + 2))
    recalls, ndcgs = [], []
    for u in range(R.shape[0]):
        relevant = set(np.flatnonzero(test[u] & (R[u] >= thr)))
        if not relevant:
            continue
        candidates = np.flatnonzero(~train[u])
        top = candidates[np.argsort(-pred[u, candidates])[:k]]
        hits = np.array([h in relevant for h in top], float)
        recalls.append(hits.sum() / len(relevant))
        ndcgs.append((hits * gains[:len(hits)]).sum() / gains[:min(len(relevant), k)].sum())
    return {f"recall@{k}": np.mean(recalls), f"ndcg@{k}": np.mean(ndcgs)}


def recommend(pred, seen, u, hotels, cities, k=5):
    scores = pred[u].copy()
    scores[seen[u]] = -np.inf
    top = np.argsort(-scores)[:k]
    return pd.DataFrame({"hotel": hotels[top], "city": cities[top],
                         "predicted_rating": scores[top].round(2)})
