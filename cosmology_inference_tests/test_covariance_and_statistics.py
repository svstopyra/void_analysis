# test_covariance_and_statistics.py

import numpy as np
import pytest
import os
from void_analysis.cosmology_inference import (
    tikhonov_regularisation,
    regularise_covariance,
    get_inverse_covariance,
    range_excluding,
    get_nonsingular_subspace,
    get_solved_residuals,
    compute_normality_test_statistics,
    covariance,
    profile_jackknife_covariance,
    compute_singular_log_likelihood,
    get_covariance_matrix
)

from void_analysis import tools

SNAPSHOT_DIR = os.path.join(os.path.dirname(__file__), "snapshots")

@pytest.fixture
def mock_covariance_matrix():
    np.random.seed(0)
    A = np.random.randn(20, 20)
    return A @ A.T  # Symmetric positive semi-definite

@pytest.fixture
def mock_los_data():
    np.random.seed(1)
    los_list = [[np.random.rand(10, 2) * 3.0 for _ in range(5)] for _ in range(2)]
    radii = [np.random.uniform(1.0, 2.0, size=5),
             np.random.uniform(1.0, 2.0, size=5)]
    return los_list, radii

@pytest.fixture
def bin_edges():
    return np.linspace(0.0, 3.0, 6), np.linspace(0.0, 3.0, 6)  # spar, sperp



# ---------------------- UNIT TESTS: Regularization Functions ----------------------

def test_tikhonov_regularisation_identity():
    A = np.eye(5)
    reg_A = tikhonov_regularisation(A, lambda_reg=1e-6)
    assert np.allclose(reg_A, A + 1e-6*np.eye(5))

def test_regularise_covariance_symmetry():
    cov = np.random.rand(5, 5)
    cov = cov @ cov.T  # Make it positive semi-definite
    reg_cov = regularise_covariance(cov, lambda_reg=1e-8)
    assert np.allclose(reg_cov, reg_cov.T)

# ---------------------- UNIT TESTS: Inverse Covariance ----------------------

def test_get_inverse_covariance_consistency():
    cov = np.random.rand(5, 5)
    cov = cov @ cov.T + 1e-3*np.eye(5)
    inv_cov = get_inverse_covariance(cov)
    identity = inv_cov @ cov
    assert np.allclose(identity, np.eye(5), atol=1e-2)

# ---------------------- UNIT TESTS: Utility Functions ----------------------

def test_range_excluding_basic():
    result = range_excluding(0, 10, [2, 5, 7])
    assert np.array_equal(result, np.array([0, 1, 3, 4, 6, 8, 9]))

# ---------------------- UNIT TESTS: Statistical Functions ----------------------

def test_get_nonsingular_subspace_structure():
    cov = np.eye(5) * 2
    Umap, eig = get_nonsingular_subspace(cov, lambda_reg=1e-8)
    assert Umap.shape[1] == cov.shape[0]
    assert np.all(eig > 0)

def test_get_solved_residuals_shape():
    samples = np.random.randn(5, 100)
    covariance = np.cov(samples)
    xbar = np.mean(samples, axis=1)
    residuals = get_solved_residuals(samples, covariance, xbar)
    assert residuals.shape == (5, 100)

def test_compute_normality_statistics_output():
    samples = np.random.randn(5, 100)
    A, B = compute_normality_test_statistics(samples)
    assert isinstance(A, float)
    assert isinstance(B, float)

# ---------------------- UNIT TESTS: Covariance Functions ----------------------

def test_covariance_symmetry():
    X = np.random.randn(5, 100).T
    cov = covariance(X)
    assert cov.shape == (5, 5)
    assert np.allclose(cov, cov.T, atol=1e-8)

def test_profile_jackknife_covariance_shape():
    X = np.random.randn(5, 100).T
    cov = profile_jackknife_covariance(X,lambda x: np.mean(x,0)**2)
    assert cov.shape == (5, 5)

def test_compute_singular_log_likelihood_basic():
    np.random.seed(42)
    cov = np.random.rand(5, 5)
    samples = np.random.randn(5, 100)
    covariance_matrix = np.cov(samples)
    xbar = np.mean(samples, axis=1)
    residuals = get_solved_residuals(samples, covariance_matrix, xbar)
    xbar = np.mean(samples, axis=1)
    Umap, good_eig = get_nonsingular_subspace(cov, 1e-3)
    nll = compute_singular_log_likelihood(xbar,Umap,good_eig)
    assert isinstance(nll, float)


def test_regularise_covariance_is_symmetric(mock_covariance_matrix):
    reg = regularise_covariance(mock_covariance_matrix, lambda_reg=1e-10)
    assert np.allclose(reg, reg.T, atol=1e-10)

def test_regularise_covariance_is_positive_definite(mock_covariance_matrix):
    reg = regularise_covariance(mock_covariance_matrix, lambda_reg=1e-10)
    eigvals = np.linalg.eigvalsh(reg)
    assert np.all(eigvals > 0)

def test_inverse_covariance_sane(mock_covariance_matrix):
    inv = get_inverse_covariance(mock_covariance_matrix, lambda_reg=1e-10)
    identity = inv @ mock_covariance_matrix
    # Allow some small deviation due to regularization
    np.testing.assert_allclose(identity, np.eye(20), atol=1e-1)



# ---------------------- REGRESSION TESTS ----------------------

def test_tikhonov_regularisation_regression():
    A = np.eye(5)
    reg_A = tikhonov_regularisation(A, lambda_reg=1e-6)
    ref = np.load(os.path.join(SNAPSHOT_DIR, "tikhonov_regularisation_ref.npy"))
    np.testing.assert_allclose(reg_A, ref)

def test_regularise_covariance_regression():
    np.random.seed(42)
    cov = np.random.rand(5, 5)
    cov = cov @ cov.T
    reg_cov = regularise_covariance(cov, lambda_reg=1e-8)
    ref = np.load(os.path.join(SNAPSHOT_DIR, "regularise_covariance_ref.npy"))
    np.testing.assert_allclose(reg_cov, ref)

def test_get_inverse_covariance_regression():
    np.random.seed(42)
    cov = np.random.rand(5, 5)
    cov = cov @ cov.T + 1e-3*np.eye(5)
    inv_cov = get_inverse_covariance(cov)
    ref = np.load(os.path.join(SNAPSHOT_DIR, "inverse_covariance_ref.npy"))
    np.testing.assert_allclose(inv_cov, ref)

def test_compute_normality_statistics_regression():
    np.random.seed(42)
    cov = np.random.rand(5, 5)
    samples = np.random.randn(5, 100)
    stats = compute_normality_test_statistics(samples)
    ref = np.load(os.path.join(SNAPSHOT_DIR, "normality_statistics_ref.npy"))
    np.testing.assert_allclose(stats, ref)

def test_covariance_regression():
    np.random.seed(42)
    cov = np.random.rand(5, 5)
    X = np.random.randn(5, 100).T
    cov = covariance(X)
    ref = np.load(os.path.join(SNAPSHOT_DIR, "covariance_old_ref.npy"))
    np.testing.assert_allclose(cov, ref, rtol=1e-6)

def test_profile_jackknife_covariance_regression():
    np.random.seed(42)
    _ = np.random.rand(5, 5)
    X = np.random.randn(5, 100).T
    cov = profile_jackknife_covariance(X,lambda x: np.mean(x,0)**2)
    ref = np.load(os.path.join(SNAPSHOT_DIR, "profile_jackknife_covariance_ref.npy"))
    np.testing.assert_allclose(cov, ref, rtol=1e-6)

def test_compute_singular_log_likelihood_regression():
    np.random.seed(42)
    cov = np.random.rand(5, 5)
    cov = cov @ cov.T  # Make positive semi-definite
    samples = np.random.randn(5, 100)
    xbar = np.mean(samples, axis=1)
    Umap, good_eig = get_nonsingular_subspace(cov, 1e-3)
    nll = compute_singular_log_likelihood(xbar,Umap,good_eig)
    ref = np.load(os.path.join(SNAPSHOT_DIR, "singular_log_likelihood_ref.npy"))
    np.testing.assert_allclose(nll, ref, rtol=1e-6)

def test_covariance_matrix_regression(mock_los_data, bin_edges):
    los_list, radii = mock_los_data
    spar_bins, sperp_bins = bin_edges
    cov = get_covariance_matrix(
        los_list, radii, spar_bins, sperp_bins,
        nbar=1e-3, n_boot=500, seed=1234,
        lambda_reg=1e-10, cholesky=False, regularise=True, log_field=False
    )
    ref = np.load(os.path.join(SNAPSHOT_DIR, "covariance_ref.npy"))
    np.testing.assert_allclose(cov, ref, rtol=1e-6, atol=1e-10)

def test_get_solved_residuals():
    np.random.seed(0)
    samples = np.random.randn(5, 100)
    covariance = np.cov(samples)
    xbar = np.mean(samples, axis=1)
    tools.run_basic_regression_test(
        get_solved_residuals,
        os.path.join(SNAPSHOT_DIR, "get_solved_residuals_ref.npy"),
        samples,covariance,xbar
    )

