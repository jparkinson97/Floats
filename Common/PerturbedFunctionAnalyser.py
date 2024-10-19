import numpy as np
import matplotlib.pyplot as plt
from mpmath import mp
from typing import Callable, Optional, Tuple
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting
from Common.FloatingPointAnalyser import FloatingPointAnalyser

# Include your existing FloatingPointAnalyser class here

class FunctionPerturbationAnalyser:
    def __init__(self, 
                 func_template: Callable, 
                 arg_types: Tuple[str, ...],
                 x_min: float, 
                 x_max: float, 
                 num_x: int = 1000, 
                 k_min: float = 1.0, 
                 k_max: float = 100.0, 
                 num_k: int = 1000, 
                 sample_size: int = 100, 
                 relative: bool = False, 
                 precision: int = 50,
                 constant_values: Optional[Tuple[float, ...]] = None,
                 a_min: float = 1.0,
                 a_max: float = 1.0,
                 num_a: int = 1,
                 b_min: float = 0.0,
                 b_max: float = 0.0,
                 num_b: int = 1):
        self.func_template = func_template
        self.arg_types = arg_types
        self.x_min = x_min
        self.x_max = x_max
        self.num_x = num_x
        self.k_min = k_min
        self.k_max = k_max
        self.num_k = num_k
        self.sample_size = sample_size
        self.relative = relative
        self.precision = precision
        self.constant_values = constant_values
        self.a_min = a_min
        self.a_max = a_max
        self.num_a = num_a
        self.b_min = b_min
        self.b_max = b_max
        self.num_b = num_b

        # Generate values for a and b
        self.a_values = np.linspace(a_min, a_max, num=num_a)
        self.b_values = np.linspace(b_min, b_max, num=num_b)
        mp.dps = self.precision

    def run_analysis(self):
        num_samples = self.sample_size
        num_k = self.num_k
        num_a = self.num_a
        num_b = self.num_b
        perturbations_array = np.zeros((num_b, num_a, num_samples, num_k))
        differences_array = np.zeros((num_b, num_a, num_samples, num_k))
        
        for b_idx, b in enumerate(self.b_values):
            for a_idx, a in enumerate(self.a_values):
                print(f"Processing a={a}, b={b}")
                # Define a function that only takes x as input
                def func_a_b(x):
                    return self.func_template(x, a, b)
                analyser = FloatingPointAnalyser(
                    func=func_a_b,
                    arg_types=self.arg_types,
                    x_min=self.x_min,
                    x_max=self.x_max,
                    num_x=self.num_x,
                    k_min=self.k_min,
                    k_max=self.k_max,
                    num_k=self.num_k,
                    sample_size=self.sample_size,
                    relative=self.relative,
                    precision=self.precision,
                    constant_values=self.constant_values
                )
                perturbations, differences = analyser.compute_differences()
                # Store data
                perturbations_array[b_idx, a_idx, :, :] = perturbations
                differences_array[b_idx, a_idx, :, :] = differences
        self.perturbations_array = perturbations_array
        self.differences_array = differences_array

    def plot_differences(self):
        # For 4 samples of b
        num_b_samples = min(4, self.num_b)
        b_indices = np.linspace(0, self.num_b - 1, num_b_samples, dtype=int)
        for b_idx in b_indices:
            b = self.b_values[b_idx]
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            for a_idx, a in enumerate(self.a_values):
                # Sample a subset of indices to avoid overplotting
                sample_indices = np.linspace(0, self.sample_size -1, max(1, self.sample_size // 10), dtype=int)
                for idx in sample_indices:
                    perturbations = self.perturbations_array[b_idx, a_idx, idx, :]
                    differences = self.differences_array[b_idx, a_idx, idx, :]
                    non_zero = differences > 0
                    if np.any(non_zero):
                        ax.plot(
                            np.log10(perturbations[non_zero]),
                            np.full_like(perturbations[non_zero], a),
                            np.log10(differences[non_zero]),
                            '.', alpha=0.5
                        )
            ax.set_title(f'Differences for b={b}')
            ax.set_xlabel('log10(Perturbations)')
            ax.set_ylabel('Multiplier a')
            ax.set_zlabel('log10(Differences)')
            plt.show()
