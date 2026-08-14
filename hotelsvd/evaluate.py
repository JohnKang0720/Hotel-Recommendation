"""Metrics: how good are the predictions, and how good are the rankings.

* ``rmse`` — rating-prediction error on held-out entries.
* ``ranking_metrics`` — Recall@K and NDCG@K, which is what a recommender is
  actually judged on: did the hotels the user liked show up near the top?
"""
from __future__ import annotations

import numpy as np


def rmse(pred: np.ndarray, R: np.ndarray, mask: np.ndarray) -> float:
    diff = (R - pred)[mask]
    return float(np.sqrt(np.mean(diff ** 2))) if diff.size else float("nan")


def _dcg(relevances: np.ndarray) -> float:
    return float(np.sum(relevances / np.log2(np.arange(2, len(relevances) + 2))))


def ranking_metrics(pred: np.ndarray, R: np.ndarray, train_mask: np.ndarray,
                    test_mask: np.ndarray, k: int = 10, rel_threshold: float = 4.0
                    ) -> dict:
    """Average Recall@K and NDCG@K over users with held-out liked hotels."""
    recalls, ndcgs = [], []

    for u in range(R.shape[0]):
        # Candidates = hotels not used in training for this user.
        candidates = np.flatnonzero(~train_mask[u])
        if candidates.size == 0:
            continue
        relevant = set(np.flatnonzero(test_mask[u] & (R[u] >= rel_threshold)).tolist())
        if not relevant:
            continue

        ranked = candidates[np.argsort(-pred[u, candidates])]
        top = ranked[:k]

        hits = np.array([1.0 if h in relevant else 0.0 for h in top])
        recalls.append(hits.sum() / len(relevant))

        ideal = np.ones(min(len(relevant), k))
        ndcgs.append(_dcg(hits) / _dcg(ideal) if _dcg(ideal) > 0 else 0.0)

    return {
        f"recall@{k}": float(np.mean(recalls)) if recalls else float("nan"),
        f"ndcg@{k}": float(np.mean(ndcgs)) if ndcgs else float("nan"),
        "n_users_scored": len(recalls),
    }
