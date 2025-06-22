"""
Example script showing how to properly import and use the PMF functions.
This fixes the import error you were experiencing.
"""

import sys
import os

# Add the current directory to Python path so we can import from src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Now you can import the functions correctly
from src.pmf import _build_P, transient_distribution_piecewise

# Example usage
import numpy as np

# Test the functions
Z_piece = np.array([20])
mu = 30
m = 1
lam = m * mu + np.max(Z_piece)
N = 50

# Test _build_P function
P_matrix = _build_P(Z_piece[0], mu, m, lam, N)
print(f"P matrix shape: {P_matrix.shape}")

# Test transient_distribution_piecewise function
dt_piece = np.array([5.0])
t = 1.0

p_t = transient_distribution_piecewise(
    Z_piece=Z_piece,
    dt_piece=dt_piece,
    mu=mu,
    m=m,
    t=t,
    N=N,
    p0_idx=0
)

print(f"Transient distribution at t={t}:")
print(f"Sum of probabilities: {p_t.sum():.6f}")
print(f"First few values: {p_t[:5]}") 