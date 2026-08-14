"""Interactive recommender — run with: streamlit run app.py"""
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*matmul.*", category=RuntimeWarning)

import numpy as np
import pandas as pd
import streamlit as st

from hotelsvd import recsys, svd

FIG = Path(__file__).resolve().parent / "figures"
st.set_page_config(page_title="Hotel SVD Recommender", page_icon="🏨", layout="wide")


@st.cache_resource
def pipeline():
    data = recsys.load()
    train, test = recsys.split(data.mask, seed=1)
    funk = svd.FunkSVD(n_factors=8, epochs=100).fit(data.R, train, test)
    preds = {
        "FunkSVD (observed-only)": funk.predict_matrix(),
        "SVD (mean-imputed)": svd.svd_predict(data.R, train, 8),
        "Item-based CF": recsys.item_cf(data.R, train),
    }
    scores = {n: {"rmse": recsys.rmse(p, data.R, test), **recsys.ranking_metrics(p, data.R, train, test)}
              for n, p in preds.items()}
    return data, funk.predict_matrix(), scores


data, pred, scores = pipeline()

st.title("🏨 Hotel Recommender — SVD, from first principles")
st.caption(f"{data.mask.sum():,} ratings · {data.shape[0]} users × {data.shape[1]} hotels · "
           f"{data.density:.0%} dense. FunkSVD factors; the underlying SVD is hand-written and matches NumPy to ~1e-14.")

left, right = st.columns([1, 1.15], gap="large")
with left:
    st.subheader("Recommendations")
    user = st.selectbox("Pick a user", data.users, index=1)
    k = st.slider("How many", 3, 10, 5)
    u = int(np.flatnonzero(data.users == user)[0])
    st.dataframe(recsys.recommend(pred, data.mask, u, data.hotels, data.cities, k), hide_index=True, width="stretch")
    rated = np.flatnonzero(data.mask[u])
    top = rated[np.argsort(-data.R[u, rated])][:5]
    st.caption("Already rated highly:")
    st.dataframe(pd.DataFrame({"hotel": data.hotels[top], "rating": data.R[u, top].astype(int)}),
                 hide_index=True, width="stretch")

with right:
    st.subheader("Which method wins?")
    st.dataframe(pd.DataFrame([{"model": n, "RMSE ↓": round(s["rmse"], 3),
                                "Recall@10 ↑": round(s["recall@10"], 3), "NDCG@10 ↑": round(s["ndcg@10"], 3)}
                               for n, s in scores.items()]), hide_index=True, width="stretch")
    st.caption("FunkSVD only trains on ratings that exist. Plain SVD has to fill the blanks first — "
               "that bias is what makes it lose, and the bug that broke the original notebook.")

st.divider()
tabs = st.tabs(["Latent map", "SVD convergence", "Scree", "Matrix completion", "Learning curve", "Reconstruction error"])
figs = ["latent_map", "svd_convergence", "scree", "matrix_completion", "learning_curve", "reconstruction_error"]
for tab, name in zip(tabs, figs):
    with tab:
        if (FIG / f"{name}.png").exists():
            st.image(str(FIG / f"{name}.png"), width="stretch")
