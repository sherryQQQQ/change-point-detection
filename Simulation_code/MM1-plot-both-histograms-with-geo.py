import numpy as np
import matplotlib.pyplot as plt
import pickle
import os.path

arrival_rate = 80
service_rate = 100
t_values = [1, 5]

plt.close('all')
# Set up the wide figure with two vertically stacked subplots
fig, axs = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

for i, t in enumerate(t_values):
    # Load histogram information
    hist_name = os.path.join(
        '/Users/danielahurtado/Documents/Simulations/Figures',
        f"for_histogram_MM1_arr{arrival_rate}_serv{service_rate}_t{t}.pickle"
    )
    with open(hist_name, 'rb') as f:
        hist_data = pickle.load(f)

    counts = hist_data['counts']
    bins = hist_data['bins']

    # Overlay Geometric distribution of queue lengths in steady state
    max_q_length = int(max(bins))
    x = np.linspace(0, max_q_length, max_q_length+1)
    p = 1 - (arrival_rate / service_rate)
    y = p * (1 - p) ** x

    # Plot histogram and overlay
    axs[i].bar(
        bins[:-1],
        counts,
        width=bins[1] - bins[0],
        align='center',
        color=(0.12156863, 0.46666667, 0.70588235, 0.4),
        label=f'Histogram at t={t}'
    )
    axs[i].plot(
        x,
        y,
        color='red',
        marker='x',
        linewidth=2,
        label='Steady state distribution'
    )
    axs[i].set_ylabel('Probability')
    axs[i].legend()
    plt.close()

# Common x label and ticks
axs[-1].set_xlabel('Number of customers in the system')
axs[-1].set_xticks(bins)

fig.suptitle(f'Histogram and Geometric Distribution for MM1 Queue (Arrival Rate: {arrival_rate}, Service Rate: {service_rate})', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
