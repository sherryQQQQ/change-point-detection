"""
CoxM1-extract-and-plot-histograms-fast.py

Reads queue-length samples from an HDF5 file produced by
CoxM1-simulation-fast.py, computes empirical PMFs at the
requested checkpoints, saves them to the Figures folder,
and plots one subplot per checkpoint.
"""

import numpy as np
import h5py
import os
import matplotlib.pyplot as plt


# ── Parameters ────────────────────────────────────────────────────────────────

Z_initial_value = 20
service_rate = 30.0
#   5  → a=0.3, b=80,  service_rate=10  
#   80 → a=0.3, b=80,  service_rate=100
#   70 → a=0.3, b=150, service_rate=100
#   5  → a=0.3, b=80,  service_rate=10 or 30

v = 2

time_horizon = 10.0

# Checkpoints to plot — must be a subset of what was simulated
t_values = np.arange(1.0, time_horizon + 1.0, 1.0)

h5_path = os.path.join(
    './Data (2026)',
    f"CoxM1_Z0{Z_initial_value}_serv{int(service_rate)}_T{int(time_horizon)}_v{v}.h5"
)


# ── Load HDF5 and extract histograms ──────────────────────────────────────────

hist_data = {}   # t → {'pmf': array, 'max_q': int, 'n': int}

with h5py.File(h5_path, 'r') as f:
    ds          = f['queue_lengths']
    checkpoints = list(ds.attrs['checkpoints'])   # all simulated checkpoints
    n_replicas  = ds.shape[0]

    for t in t_values:
        if t not in checkpoints:
            raise ValueError(
                f"t={t} was not simulated. Available checkpoints: {checkpoints}"
            )
        col = checkpoints.index(t)

        # Read only the needed column (HDF5 doesn't load the full file)
        samples = ds[:, col].astype(np.int32)

        # Empirical PMF via bincount (exact for non-negative integers)
        max_q = int(samples.max())
        counts = np.bincount(samples, minlength=max_q + 1)
        pmf    = counts / n_replicas

        hist_data[t] = {'pmf': pmf, 'max_q': max_q, 'n': n_replicas}
        print(f"t={t}: {n_replicas} replicas, max queue length = {max_q}")

        # Save PMF to Figures folder as .npz
        save_name = (f"pmf_CoxM1_Z0{Z_initial_value}_serv{int(service_rate)}"
                     f"_T{int(time_horizon)}_t{int(t)}_v{v}.npz")
        save_path = os.path.join('./Figures', save_name)
        np.savez(save_path, pmf=pmf, t=t, n_replicas=n_replicas,
                 Z_initial_value=Z_initial_value, service_rate=service_rate)
        print(f"  Saved PMF → {save_path}")


# ── Plot ──────────────────────────────────────────────────────────────────────

n_plots  = len(t_values)
n_cols   = min(n_plots, 3)            # at most 3 columns
n_rows   = (n_plots + n_cols - 1) // n_cols

fig, axs = plt.subplots(n_rows, n_cols,
                         figsize=(6 * n_cols, 4 * n_rows),
                         squeeze=False)
axs_flat = axs.flat

bar_color = (0.12156863, 0.46666667, 0.70588235, 0.6)

for ax, t in zip(axs_flat, t_values):
    d   = hist_data[t]
    pmf = d['pmf']
    xs  = np.arange(len(pmf))

    ax.bar(xs, pmf, width=1.0, align='center', color=bar_color,
           label=f't = {t}')
    ax.set_xlabel('Number of customers in the system')
    ax.set_ylabel('Probability')
    ax.legend(fontsize=11)

    # x-ticks: every 1 if small range, else every 10 or 100
    if d['max_q'] <= 20:
        tick_step = 1
    elif d['max_q'] <= 100:
        tick_step = 10
    else:
        tick_step = 100
    ax.set_xticks(np.arange(0, d['max_q'] + tick_step, tick_step))

# Hide any unused subplot panels
for ax in list(axs_flat)[n_plots:]:
    ax.set_visible(False)

fig.suptitle(
    f'Queue-length PMF — CoxM1  (Z(0)={Z_initial_value}, '
    f'service rate={service_rate}, T={time_horizon})',
    fontsize=14
)
plt.tight_layout()

fig_name = (f"CoxM1_Z0{Z_initial_value}_serv{int(service_rate)}"
            f"_T{int(time_horizon)}_histograms_v{v}.png")
fig_path = os.path.join('./Figures', fig_name)
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"\nFigure saved → {fig_path}")
plt.show()
