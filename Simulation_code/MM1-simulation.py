import os.path
import numpy as np
import pickle


# Simulation of an M/M/1 queue with the following parameters:

arrival_rate = 20
service_rate = 30
time_horizon = 5
replicas = 10**7

for r in range(replicas):
    # One replica of simulation
    # initialization
    current_time = 0
    t_departure = 3*time_horizon
    t_arrival = current_time + np.random.exponential(1/arrival_rate)

    X = {} # dictionary of the number-in-system
    X[0] = 0 # initial number of customers in the system

    # Simulation of next event
    while current_time < time_horizon:
        if t_arrival < t_departure:
            # arrival event
            X[t_arrival] = X[current_time] + 1
            current_time = t_arrival
            t_arrival = current_time + np.random.exponential(scale=1/arrival_rate)

            if X[current_time] == 1:
                # first customer in the system
                t_departure = current_time + np.random.exponential(scale=1/service_rate)
        else:
            # departure event
            X[t_departure] = X[current_time] - 1
            current_time = t_departure

            if X[current_time] == 0:
                # last customer in the system
                t_departure = 3*time_horizon
            else:
                t_departure = current_time + np.random.exponential(scale=1/service_rate)
    # end of simulation

    filename = os.path.join('/Users/danielahurtado/Documents/Simulations/Data',"MM1_arr{}_serv{}_T{}_rep{}.pickle".format(arrival_rate, service_rate, time_horizon, r))
    with open(filename, 'wb') as f:
        pickle.dump(X, f)
    
    if r % 100000 == 0:
        print("Replica {} hundred thousand".format(r/1000))




