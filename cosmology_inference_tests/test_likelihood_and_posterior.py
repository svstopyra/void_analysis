# test_likelihood_and_posterior.py

import numpy as np
import pytest
from void_analysis.cosmology_inference import (
    log_likelihood_aptest,
    log_likelihood_aptest_parallel,
    log_flat_prior_single,
    log_flat_prior,
    log_probability_aptest,
    log_likelihood_profile,
    get_profile_parameters_fixed,
    profile_modified_hamaus
)

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

# ---------------------- REGRESSION TESTS ----------------------

def test_log_likelihood_aptest_regression():
    theta = np.array([1.0, 0.5])
    ref = np.load("snapshots/log_likelihood_aptest_ref.npy")
    output = log_likelihood_aptest(theta, np.zeros(10), np.zeros((10, 2)), np.eye(2), 0.1, lambda r: r, lambda r: r, lambda r: r)
    np.testing.assert_allclose(output, ref, rtol=1e-5)

def test_log_likelihood_aptest_parallel_regression():
    theta = np.array([1.0, 0.5])
    ref = np.load("snapshots/log_likelihood_aptest_parallel_ref.npy")
    output = log_likelihood_aptest_parallel(theta, np.zeros(10), np.zeros((10, 2)), np.eye(2), 0.1, lambda r: r, lambda r: r, lambda r: r)
    np.testing.assert_allclose(output, ref, rtol=1e-5)

def test_log_probability_aptest_regression():
    theta = np.array([1.0, 0.5])
    ref = np.load("snapshots/log_probability_aptest_ref.npy")
    output = log_probability_aptest(theta, np.zeros(10), np.zeros((10, 2)), np.eye(2), 0.1, lambda r: r, lambda r: r, lambda r: r)
    np.testing.assert_allclose(output, ref, rtol=1e-5)

