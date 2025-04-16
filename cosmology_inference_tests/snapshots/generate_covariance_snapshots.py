# generate_covariance_snapshots.py
import numpy as np
import os
from void_analysis.cosmology_inference import get_covariance_matrix

def generate_snapshot():
    np.random.seed(1)
    los_list = [[np.random.rand(10, 2) * 3.0 for _ in range(5)] for _ in range(2)]
    radii = [np.random.uniform(1.0, 2.0, size=5),
             np.random.uniform(1.0, 2.0, size=5)]

    spar_bins = np.linspace(0.0, 3.0, 6)
    sperp_bins = np.linspace(0.0, 3.0, 6)

    cov = get_covariance_matrix(
        los_list, radii, spar_bins, sperp_bins,
        nbar=1e-3, n_boot=500, seed=1234,
        lambda_reg=1e-10, cholesky=False, regularise=True, log_field=False
    )

    np.save("covariance_ref.npy", cov)
    print("Covariance snapshot saved.")

if __name__ == "__main__":
    generate_snapshot()
