"""Export a self-contained JSON for the static web demo (GitHub Pages).

We precompute FunkSVD predictions for every user × hotel, plus each user's
already-rated hotels, so the page can build recommendations entirely in the
browser — no backend, no server, no cold start.

    python scripts/export_web.py
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*matmul.*", category=RuntimeWarning)

import numpy as np

from hotelsvd import data
from hotelsvd.funk import FunkSVD

REPO = Path(__file__).resolve().parent.parent
DOCS = REPO / "docs"


def main() -> None:
    ratings = data.build_matrix(data.load_ratings())

    # Fit on ALL observed ratings — the demo wants the best recommendations,
    # not a held-out experiment (the metrics table already covers evaluation).
    fs = FunkSVD(n_factors=8, lr=0.01, reg=0.05, epochs=120, seed=0).fit(ratings.R, ratings.mask)
    pred = np.round(fs.predict_matrix(), 2)

    rated = []
    for u in range(ratings.shape[0]):
        idx = np.flatnonzero(ratings.mask[u])
        order = idx[np.argsort(-ratings.R[u, idx])]
        rated.append([[int(i), int(ratings.R[u, i])] for i in order])

    metrics = json.loads((REPO / "figures" / "metrics.json").read_text())
    board = {
        "FunkSVD (observed-only)": metrics["FunkSVD (observed-only)"],
        "SVD (mean-imputed)": metrics["SVD (NumPy)"],
        "Item-based CF": metrics["Item-based CF"],
    }

    payload = {
        "users": ratings.users.tolist(),
        "hotels": ratings.hotels.tolist(),
        "cities": ratings.cities.tolist(),
        "pred": pred.tolist(),
        "rated": rated,
        "metrics": board,
        "n_ratings": int(ratings.mask.sum()),
        "density": round(ratings.density, 3),
    }

    DOCS.mkdir(exist_ok=True)
    out = DOCS / "data.json"
    out.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"Wrote {out.relative_to(REPO)}  ({out.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
