from Common.PerturbedFunctionAnalyser import PerturbedFunctionAnalyser
from mpmath import sin

def sin_func(x):
    return sin(x)

analyzer = PerturbedFunctionAnalyser(
    a_min=0.5,
    a_max=1.5,
    num_a=10,
    b_min=-0.5,
    b_max=0.5,
    num_b=10,
    x_min=1e-5,
    x_max=1e5,
    num_x=1000,
    k_min=1,
    k_max=10,
    num_k=1000,
    sample_size=100,
    relative=True,
    precision=50,
    turn_it_up_to=11
)

analyzer.run_analysis()
analyzer.plot_hypervolume()