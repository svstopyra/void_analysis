import numpy as np
import pytest
from cosmology_inference import (
    profile_modified_hamaus,
    integrated_profile_modified_hamaus,
    profile_broken_power,
    profile_broken_power_log,
    rho_real
)

SNAPSHOT_PATH = os.path.join(os.path.dirname(__file__), "snapshots", "modified_hamaus_ref.npy")

# Basic radial test grid
@pytest.fixture
def r_grid():
    return np.linspace(0.1, 3.0, 100)

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

def test_broken_power_log_consistency(r_grid):
    params = (1.0, 1.0, 2.0, 1.0, 0.5)
    log_val = profile_broken_power_log(r_grid, *params)
    val = profile_broken_power(r_grid, *params)
    assert np.allclose(val, np.exp(log_val), rtol=1e-5)

def test_rho_real_matches_hamaus(r_grid):
    profile_args = (1.5, 3.0, 1.0, -0.5, 0.1, 1.0)
    rho = rho_real(r_grid, *profile_args)
    expected = profile_modified_hamaus(r_grid, *profile_args)
    assert np.allclose(rho, expected, rtol=1e-5)

def test_modified_hamaus_regression():
    # Fixed test case (input + parameters)
    r = np.linspace(0.1, 3.0, 100)
    params = (1.5, 3.0, 1.0, -0.5, 0.1, 1.0)
    current_output = profile_modified_hamaus(r, *params)
    # Load saved reference data
    reference_output = np.load(SNAPSHOT_PATH)
    # Regression check
    np.testing.assert_allclose(current_output, reference_output, rtol=1e-6, atol=1e-10)

def test_integrated_hamaus_regression():
    r = np.linspace(0.1, 3.0, 100)
    params = (1.5, 3.0, 1.0, -0.5, 0.1, 1.0)
    out = integrated_profile_modified_hamaus(r, *params)
    ref = np.load(os.path.join(SNAPSHOT_DIR, "integrated_hamaus_ref.npy"))
    np.testing.assert_allclose(out, ref, rtol=1e-6)

def test_broken_power_regression():
    r = np.linspace(0.1, 3.0, 100)
    params = (1.0, 1.0, 2.0, 1.0, 0.5)
    out = profile_broken_power(r, *params)
    ref = np.load(os.path.join(SNAPSHOT_DIR, "broken_power_ref.npy"))
    np.testing.assert_allclose(out, ref, rtol=1e-6)

