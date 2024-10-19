import numpy as np
import matplotlib.pyplot as plt
from mpmath import mp
import math

# Set mpmath precision (in decimal places)
mp.dps = 50  # Increase as needed for higher precision

# Function to calculate ULP for a given float64 number
def ulp(x):
    if x == 0.0:
        return np.finfo(np.float64).eps
    else:
        return np.spacing(x)

# Define machine epsilon for double precision
epsilon = np.finfo(np.float64).eps  # Approximately 2.22e-16

# Define the range for x
x_min = 1
x_max = 1e10
num_points_x = 1000  # Number of x points (reduced for manageability)

# Define the multiplier range for y = k * ULP(x)
k_min = 1
k_max = 10  # Explore multiples of ULP(x)
num_points_k = 1000  # Number of k points

# Generate x values logarithmically spaced
x_values = np.logspace(np.log10(x_min), np.log10(x_max), num=num_points_x)

# Generate k values linearly spaced
k_values = np.linspace(k_min, k_max, num=num_points_k)

# Sample a subset of x values to plot
sample_size = 100  # Number of x samples to plot
sample_indices = np.linspace(0, num_points_x - 1, sample_size, dtype=int)
sample_x = x_values[sample_indices]

# Initialize lists to collect differences for sampled x
diff_list = []

for x in sample_x:
    # Calculate ULP(x)
    current_ulp = ulp(x)
    
    # Compute y values as multiples of ULP(x)
    y_values = k_values * current_ulp
    
    # Compute exact sum using mpmath
    x_mp = mp.mpf(x)
    y_mp = [mp.mpf(y) for y in y_values]
    exact_sum = [x_mp + y for y in y_mp]
    
    # Compute floating-point sum using float64
    float_sum = x + y_values  # Numpy performs float64 addition
    
    # Convert exact_sum back to float64 for comparison
    # Compute difference: exact_sum - float_sum
    # Since exact_sum is higher precision, convert float_sum to mpmath for accurate subtraction
    exact_sum_float = [float(s) for s in exact_sum]
    difference = [s - f for s, f in zip(exact_sum, float_sum)]
    
    diff_list.append(difference)

# Convert list to numpy array for plotting
diff_array = np.array(diff_list)

# Plotting
plt.figure(figsize=(12, 8))

for i, x in enumerate(sample_x):
    # To avoid overplotting, plot only a subset of lines
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
