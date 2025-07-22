import numpy as np
import matplotlib.pyplot as plt
import pickle
import os.path

Z_initial_value = 80 # 5 or 80 (initial value for simulation of arrival rate Z)

if Z_initial_value == 5:
    service_rate = 10
elif Z_initial_value == 80:
    service_rate = 100
else:
    raise ValueError("Z_initial_value must be either 5 or 80")


t_values = [1, 5]

plt.close('all')
# Set up the wide figure with two vertically stacked subplots
fig, axs = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

for i, t in enumerate(t_values):
    # Load histogram information
    hist_name = os.path.join('/Users/qianxinhui/Desktop/NU-Research/kellogg/change-point-detection/Simulation_histograms', "for_histogram_CoxM1_Z0{}_serv{}_t{}.pickle".format(Z_initial_value, service_rate, t))
    with open(hist_name, 'rb') as f:
        hist_data = pickle.load(f)

    counts = hist_data['counts']
    bins = hist_data['bins']

    # Plot histogram and overlay
    axs[i].bar(
        bins[:-1],
        counts,
        width=bins[1] - bins[0],
        align='center',
        color=(0.12156863, 0.46666667, 0.70588235, 0.4),
        label=f'Histogram at t={t}'
    )
    axs[i].set_ylabel('Probability')
    axs[i].legend()
    plt.close()

# Common x label and ticks
axs[-1].set_xlabel('Number of customers in the system')
axs[-1].set_xticks(bins)

fig.suptitle(f'Histograms for CoxM1 Queue (Z0: {Z_initial_value}, Service Rate: {service_rate})', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
