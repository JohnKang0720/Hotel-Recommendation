"""Sanity checks for the metrics and the train/test split."""
import numpy as np

from hotelsvd.data import train_test_split
from hotelsvd.evaluate import ranking_metrics, rmse


def test_rmse_zero_for_perfect_predictions():
    R = np.array([[5.0, 3.0], [2.0, 4.0]])
    mask = np.ones_like(R, dtype=bool)
    assert rmse(R.copy(), R, mask) == 0.0


def test_perfect_ranking_scores_one():
    # 3 users, 4 hotels; predictions perfectly rank the held-out liked hotel first.
    R = np.array([
        [5, 1, 1, 1],
        [1, 5, 1, 1],
        [1, 1, 5, 1],
    ], dtype=float)
    train = np.array([
        [0, 1, 1, 1],
        [1, 0, 1, 1],
        [1, 1, 0, 1],
    ], dtype=bool)
    test = ~train
    pred = R.copy()  # perfect
    m = ranking_metrics(pred, R, train, test, k=2)
    assert m["recall@2"] == 1.0
    assert m["ndcg@2"] == 1.0


def test_split_is_disjoint_and_within_observed():
    rng = np.random.default_rng(0)
    mask = rng.random((50, 20)) < 0.5
    train, test = train_test_split(mask, test_frac=0.2, seed=0)
    assert not np.any(train & test)          # disjoint
    assert np.all((train | test) <= mask)    # never invents unobserved entries
