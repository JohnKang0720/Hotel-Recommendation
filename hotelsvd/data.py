"""Load ratings and turn them into a user × hotel matrix.

The key detail: a rating that doesn't exist is *missing*, not zero. We carry a
boolean ``mask`` of observed entries everywhere so the factorizations never
confuse "didn't stay here" with "rated it 0".
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = REPO_ROOT / "data" / "hotel_ratings.csv"


@dataclass
class Ratings:
    """A dense ratings matrix plus the bookkeeping to use it safely."""

    R: np.ndarray            # (n_users, n_hotels), 0 where unobserved
    mask: np.ndarray         # (n_users, n_hotels) bool, True where observed
    users: np.ndarray        # user labels, indexes rows
    hotels: np.ndarray       # hotel labels, indexes columns
    cities: np.ndarray       # city per hotel, aligned to `hotels`

    @property
    def shape(self) -> tuple[int, int]:
        return self.R.shape

    @property
    def density(self) -> float:
        return float(self.mask.mean())


def load_ratings(path: str | Path = DEFAULT_DATA) -> pd.DataFrame:
    """Read the long-format ratings file (one row per user–hotel rating)."""
    df = pd.read_csv(path)
    expected = {"user", "hotel", "rating"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return df


def build_matrix(df: pd.DataFrame) -> Ratings:
    """Pivot the long table into a user × hotel matrix + observed mask."""
    pivot = df.pivot_table(index="user", columns="hotel", values="rating")
    mask = pivot.notna().to_numpy()
    R = np.nan_to_num(pivot.to_numpy(), nan=0.0)

    hotels = pivot.columns.to_numpy()
    if "city" in df.columns:
        city_by_hotel = df.drop_duplicates("hotel").set_index("hotel")["city"]
        cities = city_by_hotel.reindex(hotels).to_numpy()
    else:
        cities = np.array(["" for _ in hotels])

    return Ratings(R=R, mask=mask, users=pivot.index.to_numpy(), hotels=hotels, cities=cities)


def train_test_split(mask: np.ndarray, test_frac: float = 0.2, seed: int = 0
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Hide a fraction of the *observed* ratings for evaluation.

    Splitting happens over observed entries only — we never move a missing
    rating into the test set. Users keep at least one training rating.
    """
    rng = np.random.default_rng(seed)
    train = mask.copy()
    test = np.zeros_like(mask)

    for u in range(mask.shape[0]):
        rated = np.flatnonzero(mask[u])
        if len(rated) < 3:
            continue  # too few to spare one for testing
        n_test = max(1, int(round(len(rated) * test_frac)))
        chosen = rng.choice(rated, size=n_test, replace=False)
        train[u, chosen] = False
        test[u, chosen] = True

    return train, test
