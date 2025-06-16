# generate_tools_snapshots.py

import numpy as np
from void_analysis.tools import *

GENERATED_SNAPSHOTS = [
    "gaussian_log_likelihood_function_ref.npy"
]

def generate_snapshots():
    np.random.seed(42)

    # Synthetic dummy inputs
    np.random.seed(0)
    N = 5
    mu = np.random.rand(5)
    A = np.random.randn(5, 5)
    C = A @ A.T
    
    # gaussian_log_likelihood_function
    np.random.seed(1)
    data = np.random.rand(5)
    generate_regression_test_data(
        gaussian_log_likelihood_function,
        "gaussian_log_likelihood_function_ref.npy",
        data,mu,C
    )
    
    

    print("✅ Tools snapshots saved!")

if __name__ == "__main__":
    generate_snapshots()

