import os
import numpy as np
import pytest
from void_analysis.cosmology_inference import (
    regularise_covariance,
    get_inverse_covariance,
    get_covariance_matrix
)

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

# ---------------------- UNIT TESTS ----------------------

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

# ---------------------- REGRESSION TEST ----------------------

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
