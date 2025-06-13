# test_likelihood_and_posterior.py

import numpy as np
import pytest
import os
from void_analysis.cosmology_inference import (
    log_likelihood_aptest,
    log_flat_prior_single,
    log_flat_prior,
    log_probability_aptest,
    log_likelihood_profile,
    get_profile_parameters_fixed,
    profile_modified_hamaus,
    generate_scoord_grid,
)

SNAPSHOT_DIR = os.path.join(os.path.dirname(__file__), "snapshots")

@pytest.fixture
def synthetic_data():
    np.random.seed(42)
    # Synthetic dummy inputs
    theta = np.array([1.0, 0.5])
    data_field = np.random.rand(10)
    spar_bins = np.linspace(0,2,6)
    sperp_bins = np.linspace(0,2,3)
    scoords = generate_scoord_grid(sperp_bins, spar_bins)
    inverse_matrix = np.eye(10)
    z = 0.1
    return theta, data_field, scoords, inverse_matrix, z

@pytest.fixture
def synthetic_profile_data():
    np.random.seed(1)
    r = np.linspace(0.1, 10.0, 300)
    true_params = (1.5, 3.0, 1.0, -0.5, 0.1, 1.0)
    delta = profile_modified_hamaus(r, *true_params)
    delta_noisy = delta + 0.02 * np.random.randn(len(r))
    sigma = 0.02 * np.ones_like(r)
    return r, delta_noisy, sigma, true_params

# ---------------------- UNIT TESTS: Flat Priors ----------------------

def test_log_flat_prior_single_inside():
    val = 0.5
    bounds = (0.0, 1.0)
    assert np.isclose(log_flat_prior_single(val, bounds), 0.0)

def test_log_flat_prior_single_outside():
    val = 1.5
    bounds = (0.0, 1.0)
    assert log_flat_prior_single(val, bounds) == -np.inf

def test_log_flat_prior_batch_inside():
    params = np.array([0.5, 0.5, 0.5])
    bounds = [(0.0, 1.0)] * 3
    assert np.isclose(log_flat_prior(params, bounds), 0.0)

def test_log_flat_prior_batch_outside():
    params = np.array([0.5, 1.5, 0.5])
    bounds = [(0.0, 1.0)] * 3
    assert log_flat_prior(params, bounds) == -np.inf

# ---------------------- UNIT TESTS: Profile Likelihood ----------------------

def test_log_likelihood_profile_basic():
    r = np.linspace(0.1, 2.0, 100)
    params = [1.0, 2.0, 1.0, -0.5, 0.0, 1.0]
    y_true = profile_modified_hamaus(r, *params)
    noise = 0.01
    y_obs = y_true + noise * np.random.randn(len(r))
    sigma = noise * np.ones_like(r)

    ll = log_likelihood_profile(params, r, y_obs, sigma, profile_modified_hamaus)
    assert isinstance(ll, float)

# ---------------------- UNIT TESTS: Profile Fitting ----------------------

def test_get_profile_parameters_fixed_convergence():
    r = np.linspace(0.1, 2.0, 300)
    true_params = [1.0, 2.0, 1.0, -0.5, 0.0, 1.0]
    y_true = profile_modified_hamaus(r, *true_params)
    noise = 0.01
    y_obs = y_true + noise * np.random.randn(len(r))
    sigma = noise * np.ones_like(r)

    fitted_params = get_profile_parameters_fixed(r, y_obs, sigma)
    assert np.allclose(fitted_params, true_params, atol=0.5)


def test_log_likelihood_output(synthetic_profile_data):
    r, delta_noisy, sigma, true_params = synthetic_profile_data
    ll = log_likelihood_profile(true_params, r, delta_noisy, sigma, profile_modified_hamaus)
    assert isinstance(ll, float)
    assert np.isfinite(ll)

def test_log_likelihood_penalizes_wrong_model(synthetic_profile_data):
    r, delta_noisy, sigma, true_params = synthetic_profile_data
    wrong_params = tuple(p * 1.5 for p in true_params)
    ll_true = log_likelihood_profile(true_params, r, delta_noisy, sigma, profile_modified_hamaus)
    ll_wrong = log_likelihood_profile(wrong_params, r, delta_noisy, sigma, profile_modified_hamaus)
    assert ll_true > ll_wrong

def test_get_profile_parameters_fixed_converges(synthetic_profile_data):
    r, delta_noisy, sigma, true_params = synthetic_profile_data
    best_fit = get_profile_parameters_fixed(r, delta_noisy, sigma,
                                            model=profile_modified_hamaus)
    assert len(best_fit) == 6
    rel_errors = np.abs((np.array(best_fit) - true_params) / np.array(true_params))
    assert np.all(rel_errors < 0.1)
# -----------------------------------UNIT TESTS ------------------------------------

def test_log_probability_aptest_sanity():
    # Minimal fake inputs
    theta = [0.99, 0.5]  # [epsilon, f] (shortened for now)
    z = 0.02
    field = np.random.normal(1.0, 0.01, size=100)
    scoords = np.random.rand(100, 2)
    cov = np.eye(100)
    Delta = lambda r: 0.2 * np.exp(-r**2)
    delta = lambda r: -0.2 * np.exp(-r**2)
    rho_real = lambda r: delta(r) + 1.0
    logp = log_probability_aptest(theta, field, scoords, cov, z, Delta, delta, rho_real,
                                   cholesky=False, tabulate_inverse=False,
                                   sample_epsilon=True, theta_ranges=[[0.9, 1.1], [0.0, 1.0]],
                                   singular=False, log_density=False, F_inv=None,
                                   Umap=None, good_eig=None)
    assert np.isfinite(logp)



# ---------------------- REGRESSION TESTS ----------------------


def test_profile_fit_regression(synthetic_profile_data):
    r, delta_noisy, sigma, _ = synthetic_profile_data
    best_fit = get_profile_parameters_fixed(r, delta_noisy, sigma, model=profile_modified_hamaus)
    ref = np.load(os.path.join(SNAPSHOT_DIR, "profile_fit_params_ref.npy"))
    np.testing.assert_allclose(best_fit, ref, rtol=1e-5, atol=1e-7)



def test_log_likelihood_aptest_regression(synthetic_data):
    theta, data_field, scoords, inverse_matrix, z = synthetic_data
    Delta_func = lambda r: r
    delta_func = lambda r: r
    rho_real = lambda r: r
    theta = np.array([1.0, 0.5])
    ref = np.load(os.path.join(SNAPSHOT_DIR, "log_likelihood_aptest_ref.npy"))
    output = log_likelihood_aptest(theta, data_field, scoords, inverse_matrix, z, Delta_func, delta_func, rho_real)
    np.testing.assert_allclose(output, ref, rtol=1e-5)

def test_log_probability_aptest_regression(synthetic_data):
    theta, data_field, scoords, inverse_matrix, z = synthetic_data
    Delta_func = lambda r: r
    delta_func = lambda r: r
    rho_real = lambda r: r
    theta = np.array([1.0, 0.5])
    ref = np.load(os.path.join(SNAPSHOT_DIR, "log_probability_aptest_ref.npy"))
    output = log_probability_aptest(theta, data_field, scoords, inverse_matrix, z, Delta_func, delta_func, rho_real,theta_ranges = [[-1,1],[0,1]])
    np.testing.assert_allclose(output, ref, rtol=1e-5)

