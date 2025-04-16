import os
import numpy as np
import pytest
from void_analysis.cosmology_inference import (
    log_likelihood_profile,
    get_profile_parameters_fixed,
    profile_modified_hamaus,
    log_probability_aptest
)

SNAPSHOT_DIR = os.path.join(os.path.dirname(__file__), "snapshots")

@pytest.fixture
def synthetic_profile_data():
    np.random.seed(1)
    r = np.linspace(0.1, 10.0, 300)
    true_params = (1.5, 3.0, 1.0, -0.5, 0.1, 1.0)
    delta = profile_modified_hamaus(r, *true_params)
    delta_noisy = delta + 0.02 * np.random.randn(len(r))
    sigma = 0.02 * np.ones_like(r)
    return r, delta_noisy, sigma, true_params

# ---------------------- UNIT TESTS ----------------------

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

# ---------------------- REGRESSION TEST ----------------------

def test_profile_fit_regression(synthetic_profile_data):
    r, delta_noisy, sigma, _ = synthetic_profile_data
    best_fit = get_profile_parameters_fixed(r, delta_noisy, sigma, model=profile_modified_hamaus)
    ref = np.load(os.path.join(SNAPSHOT_DIR, "profile_fit_params_ref.npy"))
    np.testing.assert_allclose(best_fit, ref, rtol=1e-5, atol=1e-7)
