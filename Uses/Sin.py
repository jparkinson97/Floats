from Common.FloatingPointAnalyser import FloatingPointAnalyser
from mpmath import sin

def sin_func(x):
    return sin(x)

analyzer = FloatingPointAnalyser(
    func=sin_func,
    arg_types=('vary',),
    x_min=1e-5,
    x_max=1e5,
    k_min=1,
    k_max=10,
    constant_values=None,
    relative=True
)

analyzer.run_analysis()
