"""
CoxM1-plot-histograms-fast.py

Reads pre-computed PMFs from .npz files (produced by
CoxM1-extract-and-plot-histograms-fast.py) and plots one subplot
per checkpoint.
"""

import numpy as np
import os
import matplotlib.pyplot as plt


# ── Parameters ────────────────────────────────────────────────────────────────

Z_initial_value = 5
#   5  → a=0.3, b=80,  service_rate=10
#   80 → a=0.3, b=80,  service_rate=100
#   70 → a=0.3, b=150, service_rate=100

v = 2

if Z_initial_value == 5:
    service_rate = 10
elif Z_initial_value == 80:
    service_rate = 100
elif Z_initial_value == 70:
    service_rate = 100
else:
    raise ValueError("Z_initial_value must be 5, 70, or 80")

time_horizon = 10.0

# Checkpoints to plot — each must have a corresponding .npz file
t_values = [1.0, 5.0]


# ── Load PMFs from .npz files ─────────────────────────────────────────────────

hist_data = {}   # t → {'pmf': array, 'max_q': int, 'n': int}

for t in t_values:
    npz_name = (f"pmf_CoxM1_Z0{Z_initial_value}_serv{int(service_rate)}"
                f"_T{int(time_horizon)}_t{int(t)}_v{v}.npz")
    npz_path = os.path.join('./Figures', npz_name)

    data = np.load(npz_path)
    pmf  = data['pmf']
    hist_data[t] = {
        'pmf':   pmf,
        'max_q': len(pmf) - 1,
        'n':     int(data['n_replicas']),
    }
    print(f"t={t}: loaded {int(data['n_replicas']):,} replicas, "
          f"max queue length = {len(pmf) - 1}")


# ── Plot ──────────────────────────────────────────────────────────────────────

n_plots = len(t_values)
n_cols  = min(n_plots, 3)
n_rows  = (n_plots + n_cols - 1) // n_cols

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
