"""Train every model, print the scoreboard, save figures, export the web demo."""
import json
import warnings
from pathlib import Path

import numpy as np

from hotelsvd import recsys, svd, viz

warnings.filterwarnings("ignore", message=".*matmul.*", category=RuntimeWarning)

REPO = Path(__file__).resolve().parent.parent
FIG, DOCS = REPO / "figures", REPO / "docs"
K = 8


def main():
    data = recsys.load()
    R, mask = data.R, data.mask
    train, test = recsys.split(mask, seed=1)
    print(f"{mask.sum():,} ratings — {R.shape[0]} users x {R.shape[1]} hotels ({data.density:.0%} dense)\n")

    ours, numpy_s = svd.power_svd(np.where(train, R, 0), K)[1], np.linalg.svd(np.where(train, R, 0), compute_uv=False)[:K]
    print(f"From-scratch SVD vs NumPy — max singular-value diff: {np.abs(ours - numpy_s).max():.2e}\n")

    funk = svd.FunkSVD(n_factors=K, epochs=120).fit(R, train, test)
    preds = {
        "SVD (mean-imputed)": svd.svd_predict(R, train, K, use_numpy=True),
        "Item-based CF": recsys.item_cf(R, train),
        "FunkSVD (observed-only)": funk.predict_matrix(),
    }

    metrics = {}
    print(f"{'model':<24}{'RMSE':>7}  Recall@10  NDCG@10")
    for name, pred in preds.items():
        rank = recsys.ranking_metrics(pred, R, train, test)
        metrics[name] = {"rmse": recsys.rmse(pred, R, test), **rank}
        print(f"{name:<24}{metrics[name]['rmse']:>7.3f}    {rank['recall@10']:.3f}     {rank['ndcg@10']:.3f}")

    clusters = recsys.cluster(funk.Q)
    viz.scree(numpy_s, FIG / "scree.png")
    viz.convergence(np.where(train, R, 0), FIG / "svd_convergence.png", K + 6)
    viz.latent_map(funk.Q, data.hotels, clusters, FIG / "latent_map.png")
    viz.completion(R, train, funk.predict_matrix(), FIG / "matrix_completion.png")
    viz.reconstruction_error(np.where(train, R, 0), FIG / "reconstruction_error.png", K + 6)
    viz.learning_curve(funk.history, FIG / "learning_curve.png")
    (FIG / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved figures -> {FIG.relative_to(REPO)}/")

    export_web(data, funk)


def export_web(data, funk):
    """Precompute recommendations so the static demo runs with no backend."""
    full = svd.FunkSVD(n_factors=K, epochs=120).fit(data.R, data.mask)
    rated = [[[int(i), int(data.R[u, i])] for i in np.flatnonzero(data.mask[u])[np.argsort(-data.R[u, data.mask[u]])]]
             for u in range(data.shape[0])]
    metrics = json.loads((REPO / "figures" / "metrics.json").read_text())
    payload = {
        "users": data.users.tolist(), "hotels": data.hotels.tolist(), "cities": data.cities.tolist(),
        "pred": np.round(full.predict_matrix(), 2).tolist(), "rated": rated,
        "metrics": {"FunkSVD (observed-only)": metrics["FunkSVD (observed-only)"],
                    "SVD (mean-imputed)": metrics["SVD (mean-imputed)"],
                    "Item-based CF": metrics["Item-based CF"]},
        "n_ratings": int(data.mask.sum()), "density": round(data.density, 3),
    }
    DOCS.mkdir(exist_ok=True)
    (DOCS / "data.json").write_text(json.dumps(payload, separators=(",", ":")))
    for f in FIG.glob("*.png"):
        (DOCS / "figures").mkdir(exist_ok=True)
        (DOCS / "figures" / f.name).write_bytes(f.read_bytes())
    print(f"Exported demo -> {DOCS.relative_to(REPO)}/")


if __name__ == "__main__":
    main()
