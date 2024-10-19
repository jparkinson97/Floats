import numpy as np
import matplotlib.pyplot as plt
from mpmath import mp
import math

mp.dps = 50  

def ulp(x):
    if x == 0.0:
        return np.finfo(np.float64).eps
    else:
        return np.spacing(x)

epsilon = np.finfo(np.float64).eps  

x_min = 1
x_max = 1e10
num_points_x = 1000  

k_min = 1
k_max = 10  
num_points_k = 1000  

x_values = np.logspace(np.log10(x_min), np.log10(x_max), num=num_points_x)

k_values = np.linspace(k_min, k_max, num=num_points_k)

sample_size = 100  
sample_indices = np.linspace(0, num_points_x - 1, sample_size, dtype=int)
sample_x = x_values[sample_indices]

diff_list = []

for x in sample_x:
    current_ulp = ulp(x)
    
    y_values = k_values * current_ulp
    
    x_mp = mp.mpf(x)
    y_mp = [mp.mpf(y) for y in y_values]
    exact_sum = [x_mp + y for y in y_mp]
    
    float_sum = x + y_values  # Numpy performs float64 addition
    
    exact_sum_float = [float(s) for s in exact_sum]
    difference = [s - f for s, f in zip(exact_sum, float_sum)]
    
    diff_list.append(difference)

diff_array = np.array(diff_list)

# Plotting
plt.figure(figsize=(12, 8))

for i, x in enumerate(sample_x):
    if i % 10 == 0:  # Plot every 10th sample for clarity
        plt.plot(y_values, diff_array[i], '.', markersize=2, alpha=0.5, label=f'x={x:.1e}' if i == 0 else "")

plt.title('Difference Between Exact Sum and Floating-Point Sum for Various x')
plt.xlabel('y = k * ULP(x)')
plt.ylabel('Exact Sum - Floating-Point Sum')
plt.yscale('log')
plt.xscale('log')
plt.grid(True, which="both", ls="--", linewidth=0.5)
if any(i % 10 == 0 for i in range(len(sample_x))):
    plt.legend(title='Sampled x values', loc='upper right')
plt.show()
