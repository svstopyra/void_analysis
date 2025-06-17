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
    get_tabulated_inverse,
    get_mle_estimate,
    generate_scoord_grid,
    generate_data_filter,
    log_likelihood_profile
)

from void_analysis import tools
from void_analysis.simulation_tools import gaussian_delta, gaussian_Delta

GENERATED_SNAPSHOTS = [
    "log_likelihood_aptest_ref.npy",
    "log_probability_aptest_ref.npy",
    "get_tabulated_inverse_ref.npy",
    "log_flat_prior_single_ref.npy",
    "log_flat_prior_ref.npy",
    "get_mle_estimate_ref.npy",
    "generate_scoord_grid_ref.npy",
    "generate_data_filter_ref.npy",
    "log_likelihood_profile_ref.npy"
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
    
    # get_mle_estimate
    np.random.seed(0)
    N = 5
    mu = np.random.rand(5)
    A = np.random.randn(5, 5)
    C = A @ A.T
    np.random.seed(42)
    guess = np.random.rand(5)
    bounds = [(0,1) for _ in range(5)]
    tools.generate_regression_test_data(
        get_mle_estimate,
        "get_mle_estimate_ref.npy",
        guess,bounds,tools.gaussian_log_likelihood_function,
        mu,C
    )
    
    # generate_scoord_grid
    sperp_bins, spar_bins = np.linspace(0,2,21),np.linspace(0,3,31)
    tools.generate_regression_test_data(
        generate_scoord_grid,
        "generate_scoord_grid_ref.npy",
        sperp_bins,spar_bins
    )
    
    # generate_data_filter
    # Setup co-ordinate grid:
    sperp_bins, spar_bins = np.linspace(0,2,5),np.linspace(0,3,7)
    scoords = generate_scoord_grid(sperp_bins, spar_bins)
    N = (len(sperp_bins)-1)*(len(spar_bins)-1)
    # Mock mean and covariance matrix:
    np.random.seed(0)
    mean = np.random.rand(N)
    A = np.random.randn(N, N)
    cov = A @ A.T
    # Test:
    tools.generate_regression_test_data(
        generate_data_filter,
        "generate_data_filter_ref.npy",
        cov, mean, scoords, cov_thresh=0.1, srad_thresh=1.5
    )
    
    # run_basic_regression_test
    r = np.linspace(0.1, 2.0, 100)
    params = [1.0, 2.0, 1.0, -0.5, 0.0, 1.0]
    y_true = profile_modified_hamaus(r, *params)
    noise = 0.01
    np.random.seed(0)
    y_obs = y_true + noise * np.random.randn(len(r))
    sigma = noise * np.ones_like(r)
    tools.generate_regression_test_data(
        log_likelihood_profile,
        "log_likelihood_profile_ref.npy",
        params, r, y_obs, sigma, profile_modified_hamaus
    )
    

    print("✅ Likelihood and posterior snapshots saved!")

if __name__ == "__main__":
    generate_snapshots()

