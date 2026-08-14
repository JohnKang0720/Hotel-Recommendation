"""Turn a predicted-rating matrix into actual recommendations."""
from __future__ import annotations

import numpy as np
import pandas as pd


def recommend_for_user(pred: np.ndarray, seen_mask: np.ndarray, user_idx: int,
                       hotels: np.ndarray, cities: np.ndarray | None = None,
                       k: int = 5) -> pd.DataFrame:
    """Top-``k`` hotels the user hasn't rated yet, best predicted score first."""
    scores = pred[user_idx].copy()
    scores[seen_mask[user_idx]] = -np.inf          # don't recommend what they've rated
    order = np.argsort(-scores)[:k]

    out = {"hotel": hotels[order], "predicted_rating": np.round(scores[order], 2)}
    if cities is not None:
        out["city"] = cities[order]
    return pd.DataFrame(out)
