import numpy as np
import matplotlib.pyplot as plt
from mpmath import mp, mpf
from typing import Callable, Optional, Tuple

class FloatingPointAnalyser:
    def __init__(self, 
                 func: Callable, 
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
                 constant_values: Optional[Tuple[float, ...]] = None):
        self.func = func
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
        mp.dps = self.precision
        if not all(arg in ('vary', 'constant') for arg in self.arg_types):
            raise ValueError("arg_types must be a tuple containing 'vary' and/or 'constant'.")
        self.num_vary = self.arg_types.count('vary')
        self.num_constant = self.arg_types.count('constant')
        if self.num_constant > 0:
            if constant_values is None:
                raise ValueError(f"{self.num_constant} constant value(s) must be provided for arg_types {self.arg_types}.")
            if len(constant_values) != self.num_constant:
                raise ValueError(f"Expected {self.num_constant} constant value(s), got {len(constant_values)}.")
            self.constant_values = constant_values
        else:
            self.constant_values = ()
        if self.num_vary > 0:
            self.x_values = np.logspace(np.log10(x_min), np.log10(x_max), num=num_x)
            self.sample_indices = np.linspace(0, num_x - 1, sample_size, dtype=int)
            self.sample_x = self.x_values[self.sample_indices]
        else:
            self.x_values = ()
            self.sample_x = ()
        if self.num_vary >1 and len(self.arg_types)==2 and self.arg_types==('vary','vary'):
            self.sample_y = np.logspace(np.log10(x_min), np.log10(x_max), num=num_x)[self.sample_indices]
        else:
            self.sample_y = ()
        self.k_values = np.linspace(k_min, k_max, num=num_k)
    
    def ulp(self, x: float) -> float:
        return np.spacing(x) if x != 0.0 else np.finfo(np.float64).eps
    
    def compute_differences(self) -> Tuple[np.ndarray, np.ndarray]:
        diff_list = []
        perturbation_list = []
        if len(self.arg_types) == 1:
            if self.arg_types[0] == 'vary':
                for idx, x in enumerate(self.sample_x):
                    current_ulp = self.ulp(x)
                    delta_x = self.k_values * current_ulp
                    exact_results = [self.func(mp.mpf(x + dx)) for dx in delta_x]
                    float_results = [self.func(float(x + dx)) for dx in delta_x]
                    if self.relative:
                        differences = [
                            abs(float(er) - fr) / abs(float(er)) if float(er) != 0 else 0
                            for er, fr in zip(exact_results, float_results)
                        ]
                    else:
                        differences = [
                            abs(float(er) - fr)
                            for er, fr in zip(exact_results, float_results)
                        ]
                    diff_list.append(differences)
                    perturbation_list.append(delta_x)
                    if (idx + 1) % max(1, (self.sample_size // 10)) == 0:
                        print(f"Processed {idx + 1}/{self.sample_size} sampled x values.")
            else:
                raise ValueError("For unary functions, arg_types should be ('vary',).")
        elif len(self.arg_types) == 2:
            if self.arg_types == ('vary', 'constant'):
                y_const = self.constant_values[0]
                for idx, x in enumerate(self.sample_x):
                    current_ulp = self.ulp(x)
                    delta_x = self.k_values * current_ulp
                    exact_results = [self.func(mp.mpf(x + dx), mp.mpf(y_const)) for dx in delta_x]
                    float_results = [self.func(float(x + dx), float(y_const)) for dx in delta_x]
                    if self.relative:
                        differences = [
                            abs(float(er) - fr) / abs(float(er)) if float(er) != 0 else 0
                            for er, fr in zip(exact_results, float_results)
                        ]
                    else:
                        differences = [
                            abs(float(er) - fr)
                            for er, fr in zip(exact_results, float_results)
                        ]
                    diff_list.append(differences)
                    perturbation_list.append(delta_x)
                    if (idx + 1) % max(1, (self.sample_size // 10)) == 0:
                        print(f"Processed {idx + 1}/{self.sample_size} sampled x values.")
            elif self.arg_types == ('constant', 'vary'):
                x_const = self.constant_values[0]
                for idx, y in enumerate(self.sample_y):
                    current_ulp = self.ulp(y)
                    delta_y = self.k_values * current_ulp
                    exact_results = [self.func(mp.mpf(x_const), mp.mpf(y + dy)) for dy in delta_y]
                    float_results = [self.func(float(x_const), float(y + dy)) for dy in delta_y]
                    if self.relative:
                        differences = [
                            abs(float(er) - fr) / abs(float(er)) if float(er) != 0 else 0
                            for er, fr in zip(exact_results, float_results)
                        ]
                    else:
                        differences = [
                            abs(float(er) - fr)
                            for er, fr in zip(exact_results, float_results)
                        ]
                    diff_list.append(differences)
                    perturbation_list.append(delta_y)
                    if (idx + 1) % max(1, (self.sample_size // 10)) == 0:
                        print(f"Processed {idx + 1}/{self.sample_size} sampled y values.")
            elif self.arg_types == ('vary', 'vary'):
                for idx, (x, y) in enumerate(zip(self.sample_x, self.sample_y)):
                    current_ulp_x = self.ulp(x)
                    current_ulp_y = self.ulp(y)
                    delta_x = self.k_values * current_ulp_x
                    delta_y = self.k_values * current_ulp_y
                    exact_results = [self.func(mp.mpf(x + dx), mp.mpf(y + dy)) for dx, dy in zip(delta_x, delta_y)]
                    float_results = [self.func(float(x), float(y)) for _ in delta_x]
                    if self.relative:
                        differences = [
                            abs(float(er) - fr) / abs(float(fr)) if float(fr) != 0 else 0
                            for er, fr in zip(exact_results, float_results)
                        ]
                    else:
                        differences = [
                            abs(float(er) - fr)
                            for er, fr in zip(exact_results, float_results)
                        ]
                    diff_list.append(differences)
                    perturbation_list.append(delta_x)
                    if (idx + 1) % max(1, (self.sample_size // 10)) == 0:
                        print(f"Processed {idx + 1}/{self.sample_size} sampled x and y values.")
            else:
                raise ValueError("Unsupported arg_types configuration.")
        else:
            raise NotImplementedError("Only unary and binary functions are supported.")
        diff_array = np.array(diff_list)
        perturbations_array = np.array(perturbation_list)
        return perturbations_array, diff_array
    
    def plot_differences(self, y_values_array: np.ndarray, diff_array: np.ndarray):
        plt.figure(figsize=(12, 8))
        if self.arg_types in [('vary', 'constant'), ('vary',)]:
            samples = self.sample_x
            label_prefix = 'x='
        elif self.arg_types == ('constant', 'vary'):
            samples = self.sample_y
            label_prefix = 'y='
        elif self.arg_types == ('vary', 'vary'):
            samples = self.sample_x
            label_prefix = 'x and y='
        else:
            samples = []
            label_prefix = ''
        for i, sample in enumerate(samples):
            if i % max(1, self.sample_size // 10) == 0:
                y_vals = y_values_array[i]
                diffs = diff_array[i]
                non_zero = diffs > 0
                label = f'{label_prefix}{sample:.1e}' if i == 0 else ""
                plt.plot(
                    y_vals[non_zero], 
                    diffs[non_zero], 
                    '.', 
                    markersize=2, 
                    alpha=0.5, 
                    label=label
                )
        plt.title('Difference Between Exact and Floating-Point Results')
        if self.arg_types == ('constant', 'vary'):
            plt.xlabel('y = k * ULP(y)')
        else:
            plt.xlabel('y = k * ULP(x)')
        plt.ylabel('Relative Difference' if self.relative else 'Absolute Difference')
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True, which="both", ls="--", linewidth=0.5)
        if any(i % max(1, self.sample_size // 10) ==0 for i in range(len(samples))):
            plt.legend(title='Sampled values', loc='upper right')
        plt.show()
    
    def run_analysis(self):
        perturbations, diffs = self.compute_differences()
        self.plot_differences(perturbations, diffs)
