import numpy as np
import matplotlib.pyplot as plt
from mpmath import mp, mpf, sin
from typing import Callable, Optional, Tuple
from Common.FloatingPointAnalyser import FloatingPointAnalyzer

### wip
class PerturbedFunctionAnalyer:
    def __init__(self, 
                 a_min: float, 
                 a_max: float, 
                 num_a: int, 
                 b_min: float, 
                 b_max: float, 
                 num_b: int, 
                 x_min: float, 
                 x_max: float, 
                 num_x: int = 1000, 
                 k_min: float = 1.0, 
                 k_max: float = 10.0, 
                 num_k: int = 1000, 
                 sample_size: int = 100, 
                 relative: bool = False, 
                 precision: int = 50):
        self.a_values = np.linspace(a_min, a_max, num_a)
        self.b_values = np.linspace(b_min, b_max, num_b)
        self.num_a = num_a
        self.num_b = num_b
        self.x_min = x_min
        self.x_max = x_max
        self.num_x = num_x
        self.k_min = k_min
        self.k_max = k_max
        self.num_k = num_k
        self.sample_size = sample_size
        self.relative = relative
        self.precision = precision
        mp.dps = self.precision
        self.hypervolume = np.zeros((num_a, num_b, num_k, num_x))
    
    def run_analysis(self):
        for i, a in enumerate(self.a_values):
            for j, b in enumerate(self.b_values):
                def func(x):
                    return a * sin(x) + b
                analyzer = FloatingPointAnalyzer(
                    func=func,
                    arg_types=('vary',),
                    x_min=self.x_min,
                    x_max=self.x_max,
                    num_x=self.num_x,
                    k_min=self.k_min,
                    k_max=self.k_max,
                    num_k=self.num_k,
                    sample_size=self.sample_size,
                    relative=self.relative,
                    precision=self.precision
                )
                perturbations, diffs = analyzer.compute_differences()
                self.hypervolume[i, j, :, :] = diffs
                print(f"Analyzed a={a}, b={b}")
    
    def plot_hypervolume(self, b_samples_indices: Optional[list] = None):
        if b_samples_indices is None:
            b_samples_indices = np.linspace(0, self.num_b -1, 4, dtype=int).tolist()
        fig = plt.figure(figsize=(20, 15))
        for idx, b_idx in enumerate(b_samples_indices):
            ax = fig.add_subplot(2, 2, idx+1, projection='3d')
            a_vals, k_vals = np.meshgrid(self.a_values, np.linspace(self.k_min, self.k_max, self.num_k))
            a_vals = a_vals.flatten()
            k_vals = k_vals.flatten()
            diffs = self.hypervolume[b_idx, :, :].flatten()
            scatter = ax.scatter(k_vals, a_vals, diffs, c=diffs, cmap='viridis', marker='o', s=1)
            ax.set_title(f'b={self.b_values[b_idx]:.2f}')
            ax.set_xlabel('k * ULP(x)')
            ax.set_ylabel('a')
            ax.set_zlabel('Differences' if not self.relative else 'Relative Differences')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_zscale('log')
            fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
        plt.tight_layout()
        plt.show()


