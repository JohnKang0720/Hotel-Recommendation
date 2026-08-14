"""Generate a small hotel-ratings dataset with a known 4-factor structure."""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message=".*matmul.*", category=RuntimeWarning)

REPO = Path(__file__).resolve().parent.parent
N_USERS, N_HOTELS, RANK, SEED = 300, 60, 4, 7

BRANDS = ["Azure", "Regent", "Harbor", "Summit", "Ivy", "Meridian", "Coral",
          "Lantern", "Cascade", "Monarch", "Aspen", "Vela"]
KINDS = ["Grand Hotel", "Suites", "Inn", "Resort", "Lodge"]
CITIES = ["Vancouver", "Seattle", "Toronto", "New York", "Chicago", "Austin"]


def main():
    rng = np.random.default_rng(SEED)
    combos = [(b, k) for b in BRANDS for k in KINDS]
    rng.shuffle(combos)
    hotels = np.array([f"{b} {k}" for b, k in combos[:N_HOTELS]])
    cities = rng.choice(CITIES, N_HOTELS)

    H = np.abs(rng.normal(0, 1, (N_HOTELS, RANK)))
    H[np.arange(N_HOTELS), rng.integers(0, RANK, N_HOTELS)] += 1.5
    W = rng.normal(0, 1, (N_USERS, RANK))
    true = np.clip(3.6 + 0.45 * (W @ H.T), 1, 5)

    popularity = rng.uniform(0.05, 0.35, N_HOTELS)
    rows = []
    for u in range(N_USERS):
        chosen = rng.choice(N_HOTELS, rng.integers(6, 21), replace=False, p=popularity / popularity.sum())
        for i in chosen:
            rows.append((f"user_{u:03d}", hotels[i], cities[i],
                         int(round(np.clip(true[u, i] + rng.normal(0, 0.4), 1, 5)))))

    out = REPO / "data" / "hotel_ratings.csv"
    out.parent.mkdir(exist_ok=True)
    pd.DataFrame(rows, columns=["user", "hotel", "city", "rating"]).to_csv(out, index=False)
    print(f"Wrote {len(rows):,} ratings -> {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
