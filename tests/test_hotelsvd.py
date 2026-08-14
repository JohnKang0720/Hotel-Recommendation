import numpy as np

from hotelsvd import recsys
from hotelsvd.svd import FunkSVD, power_svd, reconstruct


def test_svd_matches_numpy():
    A = np.random.default_rng(0).normal(size=(40, 25))
    _, S, _ = power_svd(A, 25)
    assert np.allclose(S, np.linalg.svd(A, compute_uv=False), atol=1e-8)


def test_svd_reconstructs():
    A = np.random.default_rng(1).normal(size=(30, 18))
    assert np.allclose(reconstruct(*power_svd(A, 18)), A, atol=1e-6)


def test_funksvd_recovers_low_rank():
    rng = np.random.default_rng(0)
    P, Q = rng.normal(size=(120, 3)), rng.normal(size=(40, 3))
    true = np.clip(3.5 + 0.4 * (P @ Q.T), 1, 5)
    mask = rng.random((120, 40)) < 0.4
    test = mask & (rng.random(mask.shape) < 0.2)
    fs = FunkSVD(n_factors=3, epochs=60).fit(np.where(mask, true, 0), mask & ~test)
    assert np.sqrt(((fs.predict_matrix() - true)[test] ** 2).mean()) < 0.4


def test_split_is_disjoint():
    mask = np.random.default_rng(0).random((50, 20)) < 0.5
    train, test = recsys.split(mask)
    assert not np.any(train & test)
    assert np.all((train | test) <= mask)
