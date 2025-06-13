# generate_likelihood_and_posterior_snapshots.py

import numpy as np
from void_analysis.cosmology_inference import (
    log_likelihood_aptest,
    log_flat_prior_single,
    log_flat_prior,
    log_probability_aptest,
    log_likelihood_profile,
    get_profile_parameters_fixed,
    profile_modified_hamaus,
    generate_scoord_grid,
    void_los_velocity_ratio_1lpt,
    get_tabulated_inverse
)

from void_analysis import tools
from void_analysis.simulation_tools import gaussian_delta, gaussian_Delta

GENERATED_SNAPSHOTS = [
    "log_likelihood_aptest_ref.npy",
    "log_probability_aptest_ref.npy",
    "get_tabulated_inverse_ref.npy",
    "log_flat_prior_single_ref.npy",
    "log_flat_prior_ref.npy"
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
    spar_bins2 = np.linspace(0,2,21)
    sperp_bins2 = np.linspace(0,2,21)
    scoords2 = generate_scoord_grid(sperp_bins2,spar_bins2)
    s_par, s_perp = [scoords2[:,0], scoords2[:,1]]
    ntab = 10
    f1 = 0.53

    ll_aptest = log_likelihood_aptest(
        theta, data_field, scoords, inverse_matrix, z, Delta_func, 
        delta_func, rho_real
    )
    logp_aptest = log_probability_aptest(
        theta, data_field, scoords, inverse_matrix, z, Delta_func, 
        delta_func, rho_real,theta_ranges = [[-1,1],[0,1]]
    )

    np.save("log_likelihood_aptest_ref.npy", ll_aptest)
    np.save("log_probability_aptest_ref.npy", logp_aptest)
    # get_tabulated_inverse:
    F_inv = get_tabulated_inverse(
        s_par,s_perp,ntab,gaussian_Delta,f1,
        vel_model = void_los_velocity_ratio_1lpt,vel_params=None,
        use_iterative = True
    )
    
    tools.generate_regression_test_data(
        F_inv,"get_tabulated_inverse_ref.npy",
        s_par,s_perp
    )
    
    # log_flat_prior_single
    tools.generate_regression_test_data(
        log_flat_prior_single,"log_flat_prior_single_ref.npy",
        0.5, (0.0,1.0)
    )
    # log_flat_prior
    tools.generate_regression_test_data(
        log_flat_prior,"log_flat_prior_ref.npy",
        np.array([0.1, 0.5, 0.9]),[(0.0, 1.0)] * 3
    )
    
    

    print("✅ Likelihood and posterior snapshots saved!")

if __name__ == "__main__":
    generate_snapshots()

