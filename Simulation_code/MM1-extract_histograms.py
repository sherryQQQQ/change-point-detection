import numpy as np
import matplotlib.pyplot as plt
import os.path
import pickle
import collections

# To open simulation data
arrival_rate = 80
service_rate = 100
time_horizon = 6
replicas = 10**7

# To generate histograms
t_values = [1, 5]

# Create dictionary to save histograms data
hist_X = collections.defaultdict(list)

# Load simulation data
for r in range(replicas):
    filename = os.path.join('/Users/danielahurtado/Documents/Simulations/Data',"MM1_arr{}_serv{}_T{}_rep{}.pickle".format(arrival_rate, service_rate, time_horizon, r))
    with open(filename, 'rb') as f:
        X = pickle.load(f)

    n_system = np.array(list(X.items()))

    i = 0
    for t in t_values:
        while i < len(n_system) and n_system[i][0] < t:
            i += 1
        hist_X[t].append(n_system[i-1][1])


# Create histogram for each time t
for t in t_values:
    counts, bins, bars = plt.hist(np.array(hist_X[t]), density=True, bins=int(max(hist_X[t])))
    dictionary = {'counts': counts, 'bins': bins, 'bars': bars}

    hist_name = os.path.join('/Users/danielahurtado/Documents/Simulations/Figures', "for_histogram_MM1_arr{}_serv{}_t{}.pickle".format(arrival_rate, service_rate, int(t)))
    with open(hist_name, 'wb') as f:
        pickle.dump(dictionary, f)

    plt.close()