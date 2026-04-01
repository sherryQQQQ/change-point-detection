"""
Unit tests for PMF computation.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pmf import (
    _build_P,
    build_P_list,
    transient_distribution_piecewise,
    transient_distribution_uniformization,
)


def test_build_P_shape():
    P = _build_P(z_avg=5.0, mu=10.0, m=1, lam=15.0, N=50)
    assert P.shape == (51, 51)


def test_build_P_row_sums():
    """Each row of the transition matrix should sum to 1."""
    P = _build_P(z_avg=5.0, mu=10.0, m=1, lam=15.0, N=50)
    row_sums = np.array(P.sum(axis=1)).flatten()
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)


def test_build_P_list_length():
    Z_piece = [5.0, 10.0, 15.0]
    P_list = build_P_list(Z_piece, mu=10.0, m=1, lam=25.0, N=50)
    assert len(P_list) == 3


def test_piecewise_distribution_sums_to_one():
    """The transient distribution should sum to ~1."""
    Z_piece = np.array([5.0, 80.0])
    dt_piece = np.array([1.0, 5.0])
    p = transient_distribution_piecewise(Z_piece, dt_piece, mu=10.0, m=1, t=1.0, N=100)
    assert p is not None
    assert len(p) > 0
    np.testing.assert_allclose(np.sum(p), 1.0, atol=1e-6)


def test_piecewise_all_non_negative():
    Z_piece = np.array([5.0, 80.0])
    dt_piece = np.array([1.0, 5.0])
    p = transient_distribution_piecewise(Z_piece, dt_piece, mu=10.0, m=1, t=1.0, N=100)
    assert np.all(p >= -1e-15)


def test_piecewise_mm1_steady_state():
    """
    For constant arrival rate λ < μ, p(t→∞) should approach geometric(λ/μ).
    Use long horizon with constant Z to check convergence.
    """
    lam_rate = 5.0
    mu = 10.0
    rho = lam_rate / mu  # 0.5
    t = 50.0  # long enough to reach steady state

    Z_piece = np.array([lam_rate] * 100)
    dt_piece = np.array([t / 100] * 100)

    p = transient_distribution_piecewise(Z_piece, dt_piece, mu=mu, m=1, t=t, N=100)

    # Geometric steady state: p(n) = (1 - rho) * rho^n
    N = min(len(p), 30)
    expected = np.array([(1 - rho) * rho ** n for n in range(N)])

    # Should be close (not exact due to truncation and finite horizon)
    np.testing.assert_allclose(p[:N], expected, atol=0.05)


def test_uniformization_sums_to_one():
    Z_piece = np.array([5.0, 80.0])
    p = transient_distribution_uniformization(Z_piece, mu=10.0, m=1, t=1.0, T=6.0, N=100)
    assert p is not None
    np.testing.assert_allclose(np.sum(p), 1.0, atol=1e-6)


def test_uniformization_all_non_negative():
    Z_piece = np.array([5.0, 80.0])
    p = transient_distribution_uniformization(Z_piece, mu=10.0, m=1, t=1.0, T=6.0, N=100)
    assert np.all(p >= -1e-15)


def test_piecewise_t_out_of_range():
    Z_piece = np.array([5.0])
    dt_piece = np.array([1.0])
    with pytest.raises(AssertionError):
        transient_distribution_piecewise(Z_piece, dt_piece, mu=10.0, m=1, t=2.0, N=50)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
