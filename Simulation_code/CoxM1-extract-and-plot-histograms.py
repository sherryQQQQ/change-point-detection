import numpy as np
import pickle
import matplotlib.pyplot as plt
import os.path

v = 2 # Version of the simulation
    # v=1 for UB variable for arrival rate generation
    # v=2 for UB fixed (newer code)

Z_initial_value = 80 # 5 or 80 (initial value for simulation of arrival rate Z)

if Z_initial_value == 5:
    service_rate = 10
elif Z_initial_value == 80:
    service_rate = 100
else:
    raise ValueError("Z_initial_value must be either 5 or 80")

# Extract queue lengths at times t0 and t1 to generate histogram of pmf
t0 = 1
t1 = 5
t_values = [t0, t1] # Times at which we want to extract the queue length
time_horizon = 6 # Simulation time horizon
replicas = 10**8 # Number of replicas for the simulation

plt.close('all')
# Set up the wide figure with two vertically stacked subplots
fig, axs = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

for i, t in enumerate(t_values):
    X = {}  # Dictionary to store queue lengths at time t
    # Load queue lengths at time t
    if v == 1:
        filename = os.path.join(os.path.join('./Data',"CoxM1_Z0{}_serv{}_t{}.pickle".format(Z_initial_value, service_rate, t)))
    elif v == 2:
        filename = os.path.join(os.path.join('./Data',"CoxM1_Z0{}_serv{}_t{}_v2.pickle".format(Z_initial_value, service_rate, t)))
    with open(filename, 'rb') as f:
        X = pickle.load(f)
    X_plot = list(X.values())
    counts, bins, bars = plt.hist(X_plot, density=True, bins=int(max(X_plot)))
    dictionary = {'counts': counts, 'bins': bins, 'bars': bars}

    # Save histogram information
    if v == 1:
        hist_name = os.path.join('./Figures', "for_histogram_CoxM1_Z0{}_serv{}_t{}.pickle".format(Z_initial_value, service_rate, t))
    elif v == 2:
        hist_name = os.path.join('./Figures', "for_histogram_CoxM1_Z0{}_serv{}_t{}_v2.pickle".format(Z_initial_value, service_rate, t))
    with open(hist_name, 'wb') as f:
        pickle.dump(dictionary, f)
    
    print(len(X_plot), "data points for queue lengths at time t =", t)
    
    # Plot histogram for time t
    axs[i].bar(
        bins[:-1],
        counts,
        width=bins[1] - bins[0],
        align='center',
        color=(0.12156863, 0.46666667, 0.70588235, 0.4),
        label=f'Queue lengths at t={t}'
    )
    axs[i].set_ylabel('Probability')
    axs[i].legend()
    #plt.close()

# Common x label and ticks
axs[-1].set_xlabel('Number of customers in the system')
axs[-1].set_xticks(bins)
fig.suptitle(f'Histogram for CoxM1 Queue at time {t} (Z(0)= {Z_initial_value}, Service Rate: {service_rate})', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
