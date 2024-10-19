from Common.PerturbedFunctionAnalyser import FunctionPerturbationAnalyser
from mpmath import sin
from mpmath import mp
import numpy as np

def func_template(x, a, b):
    if isinstance(x, mp.mpf):
        a_mp = mp.mpf(a)
        b_mp = mp.mpf(b)
        return a_mp * mp.cos(x) + b_mp
    else:
        return a * np.cos(x) + b

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
    num_a=10,     
    b_min=-1.0,
    b_max=1.0,
    num_b=4       
)

analyzer.run_analysis()

analyzer.plot_differences()

