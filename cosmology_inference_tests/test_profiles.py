import numpy as np
import pytest
import os
import scipy
from void_analysis.cosmology_inference import (
    profile_modified_hamaus,
    integrated_profile_modified_hamaus,
    profile_broken_power,
    profile_broken_power_log,
    rho_real,
    profile_modified_hamaus_derivative
)

from void_analysis import tools

SNAPSHOT_DIR = os.path.join(os.path.dirname(__file__), "snapshots")

# Basic radial test grid
@pytest.fixture
def r_grid():
    return np.linspace(0.1, 3.0, 100)

@pytest.fixture
def profile_broken_params():
    r = np.linspace(0.1, 3.0, 100)
    params = (1.0, 1.0, 2.0, 1.0, 0.5)
    return r, params

# ---------------------- UNIT TESTS: profile_modified_hamaus -------------------

def test_modified_hamaus_output_shape(r_grid):
    params = (1.5, 3.0, 1.0, -0.5, 0.1, 1.0)
    delta = profile_modified_hamaus(r_grid, *params)
    assert delta.shape == r_grid.shape
    assert np.all(np.isfinite(delta))

def test_modified_hamaus_zero_contrast(r_grid):
    params = (2.0, 4.0, 1.0, 0.2, 0.2, 1.0)
    delta = profile_modified_hamaus(r_grid, *params)
    assert np.allclose(delta, 0.2, atol=1e-6)  # Should reduce to delta_large

def test_integrated_profile_finite(r_grid):
    params = (1.5, 3.0, 1.0, -0.5, 0.1, 1.0)
    delta = integrated_profile_modified_hamaus(r_grid, *params)
    assert np.all(np.isfinite(delta))
    assert delta.shape == r_grid.shape

# ---------------------- UNIT TESTS: profile_broken_power_log ------------------

def test_broken_power_log_consistency(r_grid):
    params = (1.0, 1.0, 2.0, 1.0, 0.5)
    log_val = profile_broken_power_log(r_grid, *params)
    val = profile_broken_power(r_grid, *params)
    assert np.allclose(val, np.exp(log_val), rtol=1e-5)

# ---------------------- UNIT TESTS: rho_real ------------------

def test_rho_real_matches_hamaus(r_grid):
    profile_args = (1.5, 3.0, 1.0, -0.5, 0.1, 1.0)
    rho = rho_real(r_grid, *profile_args)
    expected = profile_modified_hamaus(r_grid, *profile_args)
    assert np.allclose(rho, expected, rtol=1e-5)

# ---------------------- UNIT TESTS: integrated_profile_modified_hamaus --------

def test_integrated_matches_numerical(r_grid):
    profile_args = (1.5, 3.0, 1.0, -0.5, 0.1, 1.0)
    delta_func = lambda r: profile_modified_hamaus(r,*profile_args) * r**2
    Delta_numerical = np.array([
        scipy.integrate.quad(delta_func,0,r)[0] * (3/(r**3)) for r in r_grid])
    Delta_analytic = integrated_profile_modified_hamaus(r_grid,*profile_args)
    assert np.allclose(Delta_analytic,Delta_numerical,rtol=1e-5)

# ---------------------- REGRESSION TESTS ----------------------

def test_modified_hamaus_regression():
    # Fixed test case (input + parameters)
    r = np.linspace(0.1, 3.0, 100)
    params = (1.5, 3.0, 1.0, -0.5, 0.1, 1.0)
    current_output = profile_modified_hamaus(r, *params)
    # Load saved reference data
    reference_output = np.load(os.path.join(SNAPSHOT_DIR, "modified_hamaus_ref.npy"))
    # Regression check
    np.testing.assert_allclose(current_output, reference_output, rtol=1e-6, atol=1e-10)

def test_integrated_hamaus_regression():
    r = np.linspace(0.1, 3.0, 100)
    params = (1.5, 3.0, 1.0, -0.5, 0.1, 1.0)
    out = integrated_profile_modified_hamaus(r, *params)
    ref = np.load(os.path.join(SNAPSHOT_DIR, "integrated_hamaus_ref.npy"))
    np.testing.assert_allclose(out, ref, rtol=1e-6)

def test_broken_power_regression(profile_broken_params):
    r, params = profile_broken_params
    out = profile_broken_power(r, *params)
    ref = np.load(os.path.join(SNAPSHOT_DIR, "broken_power_ref.npy"))
    np.testing.assert_allclose(out, ref, rtol=1e-6)

def test_profile_broken_power_log(profile_broken_params):
    r, params = profile_broken_params
    tools.run_basic_regression_test(
        profile_broken_power_log,
        os.path.join(SNAPSHOT_DIR,"profile_broken_power_log_ref.npy"),
        r,*params
    )

def test_profile_modified_hamaus_derivative():
    r = np.linspace(0.1, 3.0, 100)
    params = (1.5, 3.0, 1.0, -0.5, 0.1, 1.0)
    computed = lambda r, *params: (
        profile_modified_hamaus_derivative(r,0,*params),
        profile_modified_hamaus_derivative(r,1,*params),
        profile_modified_hamaus_derivative(r,2,*params)
    )
    tools.run_basic_regression_test(
        computed,
        os.path.join(SNAPSHOT_DIR,"profile_modified_hamaus_derivative_ref.npz"),
        r,*params
    )

def test_rho_real():
    r = np.linspace(0.1, 3.0, 100)
    params = (1.5, 3.0, 1.0, -0.5, 0.1, 1.0)
    tools.run_basic_regression_test(
        rho_real,
        os.path.join(SNAPSHOT_DIR,"rho_real_ref.npy"),
        r,*params
    )

