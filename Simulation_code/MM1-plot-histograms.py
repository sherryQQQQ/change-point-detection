import numpy as np
import matplotlib.pyplot as plt
import pickle
import os.path

t = 1 # 1 or 5

arrival_rate = 20
service_rate = 30

# Load histogram information
hist_name = os.path.join('/Users/danielahurtado/Documents/Simulations/Figures', "for_histogram_MM1_arr{}_serv{}_t{}.pickle".format(arrival_rate, service_rate, int(t)))
with open(hist_name, 'rb') as f:
    hist_data = pickle.load(f)

counts = hist_data['counts']
bins = hist_data['bins']
bars = hist_data['bars']

# Overlay Geometric distribution of queue lengths in steady state
max_q_length = int(max(bins))
x = np.linspace(0, max_q_length, max_q_length+1)
print(x)
p = 1 - (arrival_rate/service_rate)
y = p * (1 - p)**x

# Plot and save histogram
plt.close('all')
plt.bar(bins[:-1], counts, width=bins[1]-bins[0], align='center', color= (0.12156863, 0.46666667, 0.70588235, 0.4), label='Histogram at t={}'.format(t))
plt.plot(x, y, color='red', marker='x', label='Steady state distribution', linewidth=2)
plt.xticks(bins) # To show only integer values in x axis
plt.xlabel('Number of customers in the system')
plt.ylabel('Probability')
plt.legend()
plt.show()