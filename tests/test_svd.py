"""The from-scratch SVD must agree with NumPy — that's the whole correctness claim."""
import numpy as np

from hotelsvd.svd import power_svd, reconstruct


def test_singular_values_match_numpy():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(40, 25))
    _, S_ours, _ = power_svd(A, k=25)
    S_np = np.linalg.svd(A, compute_uv=False)
    assert np.allclose(S_ours, S_np, atol=1e-8)


def test_reconstruction_recovers_matrix():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(30, 18))
    U, S, Vt = power_svd(A, k=18)
    assert np.allclose(reconstruct(U, S, Vt), A, atol=1e-6)


def test_factors_are_orthonormal():
    rng = np.random.default_rng(2)
    A = rng.normal(size=(35, 20))
    U, S, Vt = power_svd(A, k=10)
    assert np.allclose(U.T @ U, np.eye(U.shape[1]), atol=1e-6)
    assert np.allclose(Vt @ Vt.T, np.eye(Vt.shape[0]), atol=1e-6)
    assert np.all(np.diff(S) <= 1e-9)  # descending
