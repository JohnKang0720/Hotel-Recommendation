"""FunkSVD should recover a planted low-rank signal from sparse observations."""
import numpy as np

from hotelsvd.funk import FunkSVD


def _planted(n_users=120, n_items=40, rank=3, density=0.4, seed=0):
    rng = np.random.default_rng(seed)
    P = rng.normal(0, 1, (n_users, rank))
    Q = rng.normal(0, 1, (n_items, rank))
    true = np.clip(3.5 + 0.4 * (P @ Q.T), 1, 5)
    mask = rng.random((n_users, n_items)) < density
    R = np.where(mask, true, 0.0)
    return R, mask, true


def test_recovers_low_rank_signal():
    R, mask, true = _planted()
    rng = np.random.default_rng(1)
    test = mask & (rng.random(mask.shape) < 0.2)
    train = mask & ~test

    fs = FunkSVD(n_factors=3, lr=0.02, reg=0.02, epochs=60, seed=0).fit(R, train)
    pred = fs.predict_matrix()

    held_rmse = np.sqrt(np.mean((pred - true)[test] ** 2))
    assert held_rmse < 0.4  # close to the (noiseless) planted signal


def test_learning_curve_decreases():
    R, mask, _ = _planted()
    fs = FunkSVD(n_factors=3, epochs=30, seed=0).fit(R, mask)
    first, last = fs.history[0]["train_rmse"], fs.history[-1]["train_rmse"]
    assert last < first
