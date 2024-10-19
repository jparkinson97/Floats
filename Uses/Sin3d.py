from Common.PerturbedFunctionAnalyser import FunctionPerturbationAnalyser
from mpmath import sin
from mpmath import mp
import numpy as np

# Define the function template
def func_template(x, a, b):
    if isinstance(x, mp.mpf):
        # Convert a and b to mpf types
        a_mp = mp.mpf(a)
        b_mp = mp.mpf(b)
        return a_mp * mp.cos(x) + b_mp
    else:
        return a * np.cos(x) + b

# Set up the analyzer
analyzer = FunctionPerturbationAnalyser(
    func_template=func_template,
    arg_types=('vary',),  # We are varying x
    x_min=1e-10,          # Set your desired x range
    x_max=1e+10,
    num_x=1000,
    k_min=1.0,
    k_max=100.0,
    num_k=1000,
    sample_size=100,
    relative=False,
    precision=50,
    a_min=0.5,
    a_max=2.0,
    num_a=10,     # Number of samples for a
    b_min=-1.0,
    b_max=1.0,
    num_b=4       # Number of samples for b
)

# Run the analysis
analyzer.run_analysis()

# Plot the results
analyzer.plot_differences()

