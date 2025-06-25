import numpy as np
import scipy.sparse as sp
import scipy.stats as st
import math

'''
build_P_list & transient_distribution_uniformization: For one step situation
_build_P & transient_distribution_piecewise: For multiple steps situation


'''


def build_P_list(Z_piece, mu, m, lam, N):
    """
    Return a list of sparse tri-diagonal matrices P^{(k)}, one for each
    piece-wise-constant interval.

    Parameters
    ----------
    Z_piece : 1-D array, length K
        Average arrival rate in each interval, Ẑ^{(k)}.
    mu : float
        Service rate of a single server.
    m : int
        Number of parallel servers. m=1 here
    lam : float
        Uniformization rate (≥ m·μ + max Z).
    N : int
        Current state-space upper bound (max number of customers).

    Returns
    -------
    list of csr_matrix, each of shape (N+1, N+1)
    """
    i = np.arange(N + 1)
    mu_i = np.minimum(i, m) * mu                # total departure rate in state i

    P_list = []
    for z in Z_piece:
        up   = np.full(N, z / lam)              # P_{i,i+1}
        down = mu_i[1:] / lam                   # P_{i,i-1}
        diag = 1.0 - (z + mu_i) / lam           # P_{i,i}
        diag[0]  = 1.0 - z / lam                # boundary at i = 0
        diag[-1] = 1.0 - mu_i[-1] / lam         # boundary at i = N

        Pk = sp.diags([up, diag, down], [1, 0, -1],
                      shape=(N + 1, N + 1), format='csr')
        P_list.append(Pk)

    return P_list

def transient_distribution_uniformization(
        Z_piece,         # length-K array of average arrival rates
        mu,              # service rate (per server)
        m=1,             # number of servers
        t=1.0,           # target time
        T=2.0,           # total horizon
        N=50,            # initial state-space size
        p0_idx=0,        # initial state
        eps_poiss=1e-8,  # Poisson tail tolerance
        eps_state=1e-9   # probability mass tolerated in the last state
    ):
    """
    Transient distribution p(t) for a (possibly non-homogeneous) M/M/m queue
    using piece-wise-constant uniformisation.
    """

    K = len(Z_piece)
    dt_full = T / K                          # length of each interval
    Kt = int(np.ceil(t / dt_full))           # number of intervals up to time t
    assert 1 <= Kt <= K, "`t` must be in (0, T]"

    lam = m * mu + np.max(Z_piece)          # uniformisation rate

    # initial transition matrices
    P_list = build_P_list(Z_piece, mu, m, lam, N)

    # initial distribution p(0)
    p = np.zeros(N + 1)
    p[p0_idx] = 1.0

    # loop over intervals 1 … K(t)
    for k in range(Kt):

        # actual length of this interval
        dt_k = dt_full if k < Kt - 1 else t - (Kt - 1) * dt_full

        # Poisson–matrix sum Φ^{(k)}(Δt_k)
        lam_dt = lam * dt_k
        A = st.poisson.isf(eps_poiss, lam_dt).astype(int) + 1  # truncation order
        weight = np.exp(-lam_dt)                               # a = 0 term
        Phi = weight * sp.eye(N + 1, format='csr')
        P_power = sp.eye(N + 1, format='csr')

        Pk = P_list[k]
        for a in range(1, A + 1):
            P_power = P_power @ Pk
            weight *= lam_dt / a
            Phi += weight * P_power

        # propagate distribution
        p = p @ Phi

        # adaptive enlargement of the state space
        tail_prob = p[N]
        if tail_prob > eps_state:
            extra = int(N * 0.5) + 10        # how many new states to add
            p = np.pad(p, (0, extra))
            N += extra
            # rebuild all P^{(k)} with the enlarged N
            P_list = build_P_list(Z_piece, mu, m, lam, N)

    return p[:N + 1]                         # return the distribution at time t


def _build_P(z_avg, mu, m, lam, N):
    """Sparse tri-diagonal transition matrix for a single block."""
    i     = np.arange(N + 1)
    mu_i  = np.minimum(i, m) * mu           # total departure in state i

    up    = np.full(N, z_avg / lam)
    down  = mu_i[1:] / lam
    diag  = 1.0 - (z_avg + mu_i) / lam
    diag[0]  = 1.0 - z_avg     / lam
    diag[-1] = 1.0 - mu_i[-1]  / lam

    return sp.diags([up, diag, down], [1, 0, -1],
                    shape=(N + 1, N + 1), format='csr')
    
def transient_distribution_piecewise(
        Z_piece,          # 1-D array of length K  (arrival rates per block)
        dt_piece,         # 1-D array of same length (block durations)
        mu,               # service rate per server
        m=1,              # number of servers
        t=1.0,            # target time
        N=50,             # initial state-space size
        p0_idx=0,         # initial state
        eps_poiss=1e-8,   # Poisson tail tolerance
        eps_state=1e-9):  # tolerated mass in last state
    """
    Transient distribution p(t) for an M/M/m queue with piece-wise *unequal*
    constant arrival rates.
    """
    Z_piece  = np.asarray(Z_piece,  dtype=float)
    dt_piece = np.asarray(dt_piece, dtype=float)
    assert Z_piece.shape == dt_piece.shape, "Z_piece and dt_piece must match"
    T = dt_piece.sum()
    assert 0.0 < t <= T, "`t` must lie in (0, total horizon]"

    lam = m * mu + np.max(Z_piece)          # uniformisation rate

    P_list = [_build_P(z, mu, m, lam, N) for z in Z_piece] # multiple piece-wise function has multiple state

    # initial distribution p(0)
    p = np.zeros(N + 1)
    p[p0_idx] = 1.0

    # iterate over blocks until reaching time t
    elapsed = 0.0
    for Pk, dt_full in zip(P_list, dt_piece):
        if elapsed >= t:
            break

        # actual delta t in this block (might be cut short by target t)
        dt_k = min(dt_full, t - elapsed)
        elapsed += dt_k

        # Poisson-weighted sum phi^{(k)}(Δt_k)
        lam_dt = lam * dt_k
        A      = st.poisson.isf(eps_poiss, lam_dt).astype(int) + 1 # Truncated A
        weight = np.exp(-lam_dt)                        # a = 0
        Phi    = weight * sp.eye(N + 1, format='csr')
        P_pow  = sp.eye(N + 1, format='csr')
        # print('A', A)
        for a in range(1, A + 1):
            
            # print(P_pow.shape, Pk.shape)
            P_pow  = P_pow @ Pk
            weight *= lam_dt / a 
            Phi    += weight * P_pow

        p = p @ Phi

    # adaptive enlargement loop (re-entered only if tail heavy)
    while p[-1] > eps_state:
        tail = p[-1]
        extra = int(N * 0.5) + 10
        N +=extra

        p = np.pad(p, (0, extra))


        P_list = [_build_P(z, mu, m, lam, N) for z in Z_piece]
        Pk     = P_list[len(P_list) - len(dt_piece)]  # same index as outside

        # ----- redo Φ^{(k)} with the larger size ----------------------------
        lam_dt = lam * dt_k
        A      = st.poisson.isf(eps_poiss, lam_dt).astype(int) + 1
        weight = np.exp(-lam_dt)
        Phi    = weight * sp.eye(N + 1, format='csr')
        P_pow  = sp.eye(N + 1, format='csr')
        for a in range(1, A + 1):
            P_pow  = P_pow @ Pk
            weight *= lam_dt / a
            Phi    += weight * P_pow

        # overwrite p for this block and re-check the tail
        p = p[:N+1] @ Phi
        
    return p[:N + 1]        # final distribution at time t


 