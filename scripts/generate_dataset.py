"""Generate a small, reproducible hotel-ratings dataset.

Why synthetic: the original notebook ran on the external TripAdvisor dump, which
isn't redistributable and isn't in this repo. So the demo ships with a compact,
seeded dataset built from a *known* low-rank structure — 4 hidden factors
(Luxury, Location, Value, Family). That's not a shortcut: it means we can prove
the from-scratch SVD actually recovers the structure that's really there.

Real data still works: `scripts/prepare_tripadvisor.py` reproduces my original
pipeline and writes the same `data/hotel_ratings.csv` if you drop the
TripAdvisor CSVs in `dataset/`.

Run:  python scripts/generate_dataset.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
OUT_CSV = REPO / "data" / "hotel_ratings.csv"
OUT_TRUTH = REPO / "data" / "ground_truth.npz"

SEED = 7
N_USERS = 300
N_HOTELS = 60
RANK = 4
FACTORS = ["Luxury", "Location", "Value", "Family"]

BRANDS = ["Azure", "Regent", "Harbor", "Summit", "Ivy", "Meridian", "Coral",
          "Lantern", "Cascade", "Monarch", "Aspen", "Vela"]
KINDS = ["Grand Hotel", "Suites", "Inn", "Resort", "Lodge"]
CITIES = ["Vancouver", "Seattle", "Toronto", "New York", "Chicago", "Austin"]


def _hotel_catalog(rng: np.random.Generator):
    names, cities = [], []
    combos = [(b, k) for b in BRANDS for k in KINDS]
    rng.shuffle(combos)
    for b, k in combos[:N_HOTELS]:
        names.append(f"{b} {k}")
        cities.append(CITIES[rng.integers(len(CITIES))])
    return np.array(names), np.array(cities)


def main() -> None:
    rng = np.random.default_rng(SEED)
    hotels, cities = _hotel_catalog(rng)

    # Planted latent structure. Hotels specialize in one factor; users have tastes.
    H = np.abs(rng.normal(0, 1, (N_HOTELS, RANK)))
    H[np.arange(N_HOTELS), rng.integers(0, RANK, N_HOTELS)] += 1.5   # give each a lean
    W = rng.normal(0, 1, (N_USERS, RANK))

    mu = 3.6
    true = mu + 0.45 * (W @ H.T)                       # continuous "true" affinity
    true = np.clip(true, 1, 5)

    # Sparse, popularity-weighted observations.
    popularity = rng.uniform(0.05, 0.35, N_HOTELS)
    rows = []
    for u in range(N_USERS):
        n_rated = rng.integers(6, 21)
        probs = popularity / popularity.sum()
        chosen = rng.choice(N_HOTELS, size=n_rated, replace=False, p=probs)
        for i in chosen:
            noisy = np.clip(true[u, i] + rng.normal(0, 0.4), 1, 5)
            rows.append((f"user_{u:03d}", hotels[i], cities[i], int(round(noisy))))

    df = pd.DataFrame(rows, columns=["user", "hotel", "city", "rating"])
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    np.savez(OUT_TRUTH, W=W, H=H, hotels=hotels, factors=np.array(FACTORS))

    print(f"Wrote {len(df):,} ratings ({df.user.nunique()} users x {df.hotel.nunique()} hotels) "
          f"-> {OUT_CSV.relative_to(REPO)}")
    print(f"Density: {len(df) / (N_USERS * N_HOTELS):.1%}   |   ground truth -> {OUT_TRUTH.relative_to(REPO)}")


if __name__ == "__main__":
    main()
