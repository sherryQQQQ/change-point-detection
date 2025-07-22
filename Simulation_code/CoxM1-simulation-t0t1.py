import numpy as np
import os.path
import pandas as pd
import pickle
import time

# Parameters
Z_initial_value = 5 # 5 (transient) or 80 (stationary)

if Z_initial_value == 5:
    service_rate = 10
elif Z_initial_value == 80:
    service_rate = 100
else:
    raise ValueError("Z_initial_value must be either 5 or 80")

# Extract queue lengths at times t0 and t1 to generate histogram of pmf
t0 = 1
t1 = 5
time_horizon = 6 # Simulation time horizon
replicas = 10**8 # Number of replicas for the simulation

# Import Cox arrival rate from csv file named initial_value_{Z_initial_value}_samples_.csv in folder Data with pandas
Z_data_path = '/Users/qianxinhui/Desktop/NU-Research/kellogg/change-point-detection/data_integrated/arrival_data'   
Z_data = pd.read_csv(os.path.join(Z_data_path, f"initial_value_{Z_initial_value}_samples_500.csv"))
Z = Z_data[['time', 'value']].to_numpy() # Column 0 has time, column 1 has arrival rate at that time
Z_T = 10 # Time horizon for simulation of Z
Z_M = 500 # Number of samples in Z

##### Define arrival generator function

def gen_next_arrival_time(clock, # Current time
                        Z, # Arrival rate process (col 0 has time, col 1 has arrival rate at that time)
                        T # Simulation time horizon
                        ):
    step_size = 0.02

    if clock < T:
        z_future = Z[int(clock/step_size):int(T/step_size),1]

        UB = 1.1*max(z_future) # Upper bound for the arrival rate
        u = UB # Uniform random variable to generate next arrival time
        arr_time = clock # Next arrival time to be generated
        
        while arr_time < T and Z[int(arr_time/step_size), 1] <= u:
            u = np.random.uniform(0, UB)
            arr_time += np.random.exponential(scale=1/UB)
        
        return arr_time
    else:
        return 3*T  # If clock is greater than T, return a large number to avoid further arrivals


def gen_next_arrival_time_2(clock, Z, T):
    step_size = 0.02
    UB = 1.1 * max(Z[int(clock / step_size):int(T / step_size), 1])  # Upper bound

    arr_time = clock
    while True:
        arr_time += np.random.exponential(scale=1 / UB)
        if arr_time >= T:
            return 3 * T

        index = min(int(arr_time / step_size), len(Z) - 1)
        lam = Z[index, 1]
        u = np.random.uniform()
        if u <= lam / UB:
            return arr_time

##### Run simulation

X0 = {}  # Dictionary of the number-in-system at time t0
X1 = {}  # Dictionary of the number-in-system at time t1

start_sim = time.time()
for r in range(replicas):
    # Initialization
    current_time = 0
    t_departure = 3 * time_horizon
    t_arrival = gen_next_arrival_time(current_time, Z, time_horizon)
    q_length = 0  # Initial queue length
    t0_aux = True # Check if we have saved the queue length at t0
    t1_aux = True # Check if we have saved the queue length at t1

    while current_time < time_horizon:
        if t_arrival < t_departure:
            # arrival event
            q_length += 1
            current_time = t_arrival
            t_arrival = gen_next_arrival_time(current_time, Z, time_horizon)

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
    # end of replica

    if r % 100000 == 0:
        print("Replica {} hundred thousand".format(r/100000))

end_sim = time.time()

save_path = '/Users/qianxinhui/Desktop/NU-Research/kellogg/change-point-detection/data_integrated/simulation_data'
os.makedirs(save_path, exist_ok=True)

# Save simulation data at times t0 and t1 in addition to the previous simulations
filename = os.path.join(save_path,"CoxM1_Z0{}_serv{}_t{}.pickle".format(Z_initial_value, service_rate, t0))
with open(filename, 'wb') as f:
    pickle.dump(X0, f)

filename = os.path.join(save_path,"CoxM1_Z0{}_serv{}_t{}.pickle".format(Z_initial_value, service_rate, t1))
with open(filename, 'wb') as f:
    pickle.dump(X1, f)

end_save = time.time()

print(f"Simulation time: {(end_sim - start_sim)/60} minutes")
print(f"Save time: {(end_save - end_sim)/60} minutes")

