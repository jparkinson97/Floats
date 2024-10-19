from Common.FloatingPointAnalyser import FloatingPointAnalyzer
import math

def sin_func(a , b):
    return a + b

analyzer = FloatingPointAnalyzer(
    func=sin_func,
    arg_types=('vary','vary'),
    x_min=1,
    x_max=1e10,
    k_min = 1,
    k_max = 10 ,
    constant_values=None  # No constants needed for unary functions
)

analyzer.run_analysis()
