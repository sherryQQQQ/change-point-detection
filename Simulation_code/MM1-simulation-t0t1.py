# Simulation of an M/M/1 queue where we only save the queue length at times t0 and t1

import os.path
import numpy as np
import pickle
import matplotlib.pyplot as plt


# Simulation of an M/M/1 queue with the following parameters:

arrival_rate = 80
service_rate = 100
time_horizon = 6
replicas = 10**7

t0 = 1
t1 = 5

# Dictionary of number-in-system per replica
X0 = {} # dictionary of the number-in-system at time t0
    # e.g. X0[4] = 2 means that in replica 4, the number in system at time t0 was 2
X1 = {} # dictionary of the number-in-system at time t1

for r in range(replicas):
    # One replica of simulation
    # initialization
    current_time = 0
    t_departure = 3*time_horizon
    t_arrival = current_time + np.random.exponential(1/arrival_rate)
    q_length = 0 # initial queue length

    t0_aux = True
    t1_aux = True

    # Simulation of next event
    while current_time < time_horizon:
        if t_arrival < t_departure:
            # arrival event
            q_length += 1
            current_time = t_arrival
            t_arrival = current_time + np.random.exponential(scale=1/arrival_rate)

            if q_length == 1:
                # first customer in the system
                t_departure = current_time + np.random.exponential(scale=1/service_rate)
        else:
            # departure event
            q_length -= 1
            current_time = t_departure

            if q_length == 0:
                # last customer in the system
                t_departure = 3*time_horizon
            else:
                t_departure = current_time + np.random.exponential(scale=1/service_rate)
        
        # Save queue length in t0 and t1
        next_event_time = min(t_arrival, t_departure)
        if t0_aux and next_event_time >= t0:
            X0[r] = q_length
            t0_aux = False
        if t1_aux and next_event_time >= t1:
            X1[r] = q_length
            t1_aux = False
    # end of simulation
    
    if r % 100000 == 0:
        print("Replica {} hundred thousand".format(r/100000))


# Save simulation data at times t0 and t1
filename = os.path.join('/Users/danielahurtado/Documents/Simulations/Data',"MM1_arr{}_serv{}_t{}.pickle".format(arrival_rate, service_rate, t0))
with open(filename, 'wb') as f:
   pickle.dump(X0, f)

filename = os.path.join('/Users/danielahurtado/Documents/Simulations/Data',"MM1_arr{}_serv{}_t{}.pickle".format(arrival_rate, service_rate, t1))
with open(filename, 'wb') as f:
   pickle.dump(X1, f)


# Save histograms of data at times t0 and t1
X_plot = list(X0.values())
counts, bins, bars = plt.hist(X_plot, density=True, bins=int(max(X_plot)))
dictionary = {'counts': counts, 'bins': bins, 'bars': bars}
hist_name = os.path.join('/Users/danielahurtado/Documents/Simulations/Figures', "for_histogram_MM1_arr{}_serv{}_t{}.pickle".format(arrival_rate, service_rate, t0))
with open(hist_name, 'wb') as f:
    pickle.dump(dictionary, f)

plt.close()

X_plot = list(X1.values())
counts, bins, bars = plt.hist(X_plot, density=True, bins=int(max(X_plot)))
dictionary = {'counts': counts, 'bins': bins, 'bars': bars}
hist_name = os.path.join('/Users/danielahurtado/Documents/Simulations/Figures', "for_histogram_MM1_arr{}_serv{}_t{}.pickle".format(arrival_rate, service_rate, t1))
with open(hist_name, 'wb') as f:
    pickle.dump(dictionary, f)

plt.close()
