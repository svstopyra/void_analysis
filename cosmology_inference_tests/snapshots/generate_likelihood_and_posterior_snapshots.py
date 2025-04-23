# generate_likelihood_and_posterior_snapshots.py

import numpy as np
from void_analysis.cosmology_inference import (
    log_likelihood_aptest,
    log_probability_aptest,
    generate_scoord_grid
)

GENERATED_SNAPSHOTS = [
    "log_likelihood_aptest_ref.npy",
    "log_probability_aptest_ref.npy"
]

def generate_snapshots():
    np.random.seed(42)

    # Synthetic dummy inputs
    theta = np.array([1.0, 0.5])
    data_field = np.random.rand(10)
    spar_bins = np.linspace(0,2,6)
    sperp_bins = np.linspace(0,2,3)
    scoords = generate_scoord_grid(sperp_bins, spar_bins)
    inverse_matrix = np.eye(10)
    z = 0.1
    Delta_func = lambda r: r
    delta_func = lambda r: r
    rho_real = lambda r: r

    ll_aptest = log_likelihood_aptest(theta, data_field, scoords, inverse_matrix, z, Delta_func, delta_func, rho_real)
    logp_aptest = log_probability_aptest(theta, data_field, scoords, inverse_matrix, z, Delta_func, delta_func, rho_real,theta_ranges = [[-1,1],[0,1]])

    np.save("log_likelihood_aptest_ref.npy", ll_aptest)
    np.save("log_probability_aptest_ref.npy", logp_aptest)

    print("✅ Likelihood and posterior snapshots saved!")

if __name__ == "__main__":
    generate_snapshots()

