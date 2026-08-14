"""Run the whole study: fit every model, score them, print a table, save figures.

    python scripts/run_experiments.py

Outputs a metrics table to stdout and writes all figures to `figures/`.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

# Silence spurious "encountered in matmul" warnings from macOS Accelerate BLAS.
# Harmless: the from-scratch SVD still matches NumPy to ~1e-14 (checked below).
warnings.filterwarnings("ignore", message=".*matmul.*", category=RuntimeWarning)

from hotelsvd import data, svd, viz
from hotelsvd.baselines import cluster_hotels, item_based_cf
from hotelsvd.evaluate import ranking_metrics, rmse
from hotelsvd.funk import FunkSVD

REPO = Path(__file__).resolve().parent.parent
FIG = REPO / "figures"
K_TRUNC = 8          # latent factors for SVD/FunkSVD
TOP_K = 10           # ranking cutoff


def _row(name, rmse_val, rank):
    r = f"{rmse_val:6.3f}" if rmse_val == rmse_val else "   -- "
    return (f"{name:<22}{r:>8}   {rank[f'recall@{TOP_K}']:.3f}    "
            f"{rank[f'ndcg@{TOP_K}']:.3f}")


def main() -> None:
    ratings = data.build_matrix(data.load_ratings())
    R, mask = ratings.R, ratings.mask
    print(f"Loaded {mask.sum():,} ratings — {R.shape[0]} users x {R.shape[1]} hotels "
          f"({ratings.density:.1%} dense)\n")

    train, test = data.train_test_split(mask, test_frac=0.2, seed=1)

    results = {}

    # --- Truncated SVD: NumPy vs our power-iteration (correctness check) ---
    pred_np = svd.svd_predict(R, train, K_TRUNC, use_numpy=True)
    pred_pw = svd.svd_predict(R, train, K_TRUNC, use_numpy=False)
    _, S_ours, _ = svd.power_svd(np.where(train, R, 0.0), K_TRUNC)
    S_np = np.linalg.svd(np.where(train, R, 0.0), compute_uv=False)[:len(S_ours)]
    print(f"From-scratch SVD vs NumPy — max singular-value diff: {np.abs(S_ours - S_np).max():.2e}\n")

    results["SVD (NumPy)"] = (rmse(pred_np, R, test),
                              ranking_metrics(pred_np, R, train, test, TOP_K))
    results["SVD (power iter, ours)"] = (rmse(pred_pw, R, test),
                                         ranking_metrics(pred_pw, R, train, test, TOP_K))

    # --- Item-based collaborative filtering (my original baseline) ---
    pred_cf = item_based_cf(R, train)
    results["Item-based CF"] = (rmse(pred_cf, R, test),
                                ranking_metrics(pred_cf, R, train, test, TOP_K))

    # --- FunkSVD: observed-only SGD (the fix) ---
    # Fit on the real ratings over the training mask; monitor test RMSE for the
    # learning curve only (no early stopping, so nothing leaks into the model).
    fs = FunkSVD(n_factors=K_TRUNC, lr=0.01, reg=0.05, epochs=120, seed=0).fit(R, train, test)
    pred_fs = fs.predict_matrix()
    results["FunkSVD (observed-only)"] = (rmse(pred_fs, R, test),
                                          ranking_metrics(pred_fs, R, train, test, TOP_K))

    # --- Table ---
    print(f"{'model':<22}{'RMSE':>8}   Recall@{TOP_K}  NDCG@{TOP_K}")
    print("-" * 52)
    for name, (rm, rk) in results.items():
        print(_row(name, rm, rk))
    print()

    # --- Figures ---
    clusters = cluster_hotels(fs.Q, n_clusters=RANK_FOR_CLUSTERS)
    viz.scree(S_np, FIG / "scree.png")
    viz.convergence(np.where(train, R, 0.0), FIG / "svd_convergence.png", max_k=K_TRUNC + 6)
    viz.latent_map(fs.Q, ratings.hotels, clusters, FIG / "latent_map.png")
    viz.completion(R, train, pred_fs, FIG / "matrix_completion.png")
    viz.reconstruction_error(np.where(train, R, 0.0), FIG / "reconstruction_error.png", max_k=K_TRUNC + 6)
    viz.learning_curve(fs.history, FIG / "learning_curve.png")
    print(f"Saved 6 figures -> {FIG.relative_to(REPO)}/")

    metrics = {name: {"rmse": rm, **rk} for name, (rm, rk) in results.items()}
    (REPO / "figures" / "metrics.json").write_text(json.dumps(metrics, indent=2))


RANK_FOR_CLUSTERS = 5

if __name__ == "__main__":
    main()
