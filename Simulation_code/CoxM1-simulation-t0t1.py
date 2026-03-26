import numpy as np
import os.path
import pandas as pd
import pickle
import time

# Parameters
Z_initial_value = 5 # 5 (transient) or 80 (stationary)

v = 2 # Version of the simulation: 1 (original), 2 (test new arrival rates with fixed UB)

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
replicas = 10**7 # Number of replicas for the simulation

# Import Cox arrival rate from csv file named initial_value_{Z_initial_value}_samples_.csv in folder Data with pandas
Z_data = pd.read_csv(os.path.join('./Data', f"initial_value_{Z_initial_value}_samples_500.csv"))
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
        if v == 1:
            z_future = Z[int(clock/step_size):int(T/step_size),1]
            UB = 1.1*max(z_future) # Upper bound for the arrival rate
        elif v == 2:
            UB = 1.1*max(Z[:,1])
        
        # Start thinning algorithm
        u = UB # Uniform random variable to generate next arrival time
        arr_time = clock # Next arrival time to be generated

        while arr_time < T and Z[int(arr_time/step_size), 1] <= u:
            u = np.random.uniform(0, UB)
            arr_time += np.random.exponential(scale=1/UB)
        
        return arr_time
    else:
        return 3*T  # If clock is greater than T, return a large number to avoid further arrivals



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


# Save simulation data at times t0 and t1
if v == 1: # Original simulation that yields strange histograms
    filename0 = os.path.join('./Data',"CoxM1_Z0{}_serv{}_t{}.pickle".format(Z_initial_value, service_rate, t0))
    filename1 = os.path.join('./Data',"CoxM1_Z0{}_serv{}_t{}.pickle".format(Z_initial_value, service_rate, t1))
elif v == 2: # New simulation that corrects arrival rates to always use same UB
    filename0 = os.path.join('./Data',"CoxM1_Z0{}_serv{}_t{}_v2.pickle".format(Z_initial_value, service_rate, t0))
    filename1 = os.path.join('./Data',"CoxM1_Z0{}_serv{}_t{}_v2.pickle".format(Z_initial_value, service_rate, t1))


with open(filename0, 'wb') as f:
   pickle.dump(X0, f)

with open(filename1, 'wb') as f:
   pickle.dump(X1, f)

end_save = time.time()

print(f"Simulation time: {(end_sim - start_sim)/60} minutes")
print(f"Save time: {(end_save - end_sim)/60} minutes")

