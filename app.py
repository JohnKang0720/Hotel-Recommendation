"""Interactive tour of the hotel recommender — SVD from scratch, visualized.

Run locally:  streamlit run app.py
"""
from __future__ import annotations

import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*matmul.*", category=RuntimeWarning)

import numpy as np
import pandas as pd
import streamlit as st

from hotelsvd import data, svd
from hotelsvd.baselines import cluster_hotels, item_based_cf
from hotelsvd.evaluate import ranking_metrics, rmse
from hotelsvd.funk import FunkSVD
from hotelsvd.recommend import recommend_for_user

REPO = Path(__file__).resolve().parent
FIG = REPO / "figures"

st.set_page_config(page_title="Hotel SVD Recommender", page_icon="🏨", layout="wide")


@st.cache_resource(show_spinner="Fitting the models…")
def pipeline():
    ratings = data.build_matrix(data.load_ratings())
    train, test = data.train_test_split(ratings.mask, test_frac=0.2, seed=1)

    fs = FunkSVD(n_factors=8, lr=0.01, reg=0.05, epochs=100, seed=0).fit(ratings.R, train, test)
    pred_fs = fs.predict_matrix()
    pred_svd = svd.svd_predict(ratings.R, train, k=8)
    pred_cf = item_based_cf(ratings.R, train)
    clusters = cluster_hotels(fs.Q, n_clusters=5)

    scores = {
        "FunkSVD (observed-only)": (rmse(pred_fs, ratings.R, test), ranking_metrics(pred_fs, ratings.R, train, test)),
        "SVD (mean-imputed)": (rmse(pred_svd, ratings.R, test), ranking_metrics(pred_svd, ratings.R, train, test)),
        "Item-based CF": (rmse(pred_cf, ratings.R, test), ranking_metrics(pred_cf, ratings.R, train, test)),
    }
    return ratings, pred_fs, clusters, scores


ratings, pred_fs, clusters, scores = pipeline()

# ── Header ─────────────────────────────────────────────────────────────────
st.title("🏨 Hotel Recommender — SVD, from first principles")
st.caption(
    f"{ratings.mask.sum():,} ratings · {ratings.shape[0]} users × {ratings.shape[1]} hotels · "
    f"{ratings.density:.0%} dense.  Latent factors learned with FunkSVD; the underlying SVD is "
    "hand-written (power iteration + deflation) and matches NumPy to ~1e-14."
)

left, right = st.columns([1, 1.15], gap="large")

# ── Recommendations ────────────────────────────────────────────────────────
with left:
    st.subheader("Recommendations")
    user_label = st.selectbox("Pick a user", ratings.users, index=1)
    k = st.slider("How many", 3, 10, 5)
    u = int(np.flatnonzero(ratings.users == user_label)[0])

    recs = recommend_for_user(pred_fs, ratings.mask, u, ratings.hotels, ratings.cities, k=k)
    st.dataframe(recs, hide_index=True, width="stretch")

    rated = np.flatnonzero(ratings.mask[u])
    top = rated[np.argsort(-ratings.R[u, rated])][:5]
    st.caption("What they already rated highly:")
    st.dataframe(
        pd.DataFrame({"hotel": ratings.hotels[top], "rating": ratings.R[u, top].astype(int)}),
        hide_index=True, width="stretch",
    )

# ── Scoreboard ─────────────────────────────────────────────────────────────
with right:
    st.subheader("Which method wins?")
    board = pd.DataFrame(
        [{"model": m, "RMSE ↓": round(rm, 3),
          "Recall@10 ↑": round(rk["recall@10"], 3), "NDCG@10 ↑": round(rk["ndcg@10"], 3)}
         for m, (rm, rk) in scores.items()]
    )
    st.dataframe(board, hide_index=True, width="stretch")
    st.caption(
        "FunkSVD only trains on ratings that exist. Plain SVD has to fill blanks first — "
        "that bias is what makes it lose here, and it's the bug that broke the original notebook."
    )

# ── Figures ────────────────────────────────────────────────────────────────
st.divider()
st.subheader("How it works")
tabs = st.tabs(["Latent map", "SVD convergence", "Scree", "Matrix completion",
                "Learning curve", "Reconstruction error"])
figs = ["latent_map", "svd_convergence", "scree", "matrix_completion",
        "learning_curve", "reconstruction_error"]
blurbs = [
    "Hotels placed by their learned factors — similar hotels cluster together.",
    "The hand-written power-iteration SVD sits exactly on top of NumPy's.",
    "Most of the signal lives in the first handful of components.",
    "Sparse observed ratings on the left; the model's full reconstruction on the right.",
    "Training vs held-out RMSE as FunkSVD learns.",
    "How reconstruction error falls as we keep more components.",
]
for tab, name, blurb in zip(tabs, figs, blurbs):
    with tab:
        p = FIG / f"{name}.png"
        if p.exists():
            st.image(str(p), width="stretch")
        st.caption(blurb)
