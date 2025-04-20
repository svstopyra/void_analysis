# generate_likelihood_and_posterior_snapshots.py

import numpy as np
from void_analysis.cosmology_inference import (
    log_likelihood_aptest,
    log_likelihood_aptest_parallel,
    log_probability_aptest
)

GENERATED_SNAPSHOTS = [
    "log_likelihood_aptest_ref.npy",
    "log_likelihood_aptest_parallel_ref.npy",
    "log_probability_aptest_ref.npy"
]

def generate_snapshots():
    np.random.seed(42)

    # Synthetic dummy inputs
    theta = np.array([1.0, 0.5])
    data_field = np.zeros(10)
    scoords = np.zeros((10, 2))
    inverse_matrix = np.eye(2)
    z = 0.1
    Delta_func = lambda r: r
    delta_func = lambda r: r
    rho_real = lambda r: r

    ll_aptest = log_likelihood_aptest(theta, data_field, scoords, inverse_matrix, z, Delta_func, delta_func, rho_real)
    ll_aptest_parallel = log_likelihood_aptest_parallel(theta, data_field, scoords, inverse_matrix, z, Delta_func, delta_func, rho_real)
    logp_aptest = log_probability_aptest(theta, data_field, scoords, inverse_matrix, z, Delta_func, delta_func, rho_real)

    np.save("log_likelihood_aptest_ref.npy", ll_aptest)
    np.save("log_likelihood_aptest_parallel_ref.npy", ll_aptest_parallel)
    np.save("log_probability_aptest_ref.npy", logp_aptest)

    print("✅ Likelihood and posterior snapshots saved!")

if __name__ == "__main__":
    generate_snapshots()

