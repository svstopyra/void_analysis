import numpy as np
import os
import pytest
from void_analysis.cosmology_inference import (
    to_z_space,
    to_real_space,
    iterative_zspace_inverse,
    iterative_zspace_inverse_scalar
)

SNAPSHOT_DIR = os.path.join(os.path.dirname(__file__), "snapshots")

@pytest.fixture
def mock_transform_inputs():
    z = 0.1
    Om = 0.3
    f1 = 0.8
    r_par = np.linspace(0.1, 3.0, 100)
    r_perp = np.ones_like(r_par) * 1.0
    Delta = lambda r: 0.2 * np.exp(-r**2)  # Cumulative profile
    return r_par, r_perp, z, Om, f1, Delta

# ---------------------- UNIT TESTS ----------------------

def test_to_z_space_runs(mock_transform_inputs):
    r_par, r_perp, z, Om, f1, Delta = mock_transform_inputs
    s_par, s_perp = to_z_space(r_par, r_perp, z, Om, Delta=Delta, f1=f1)
    assert s_par.shape == r_par.shape
    assert np.all(np.isfinite(s_par))
    assert np.allclose(s_perp, r_perp)  # Perp unchanged

def test_iterative_inverse_scalar_accuracy():
    z = 0.1
    Om = 0.3
    f1 = 0.8
    Delta = lambda r: 0.2 * np.exp(-r**2)
    r_par = 1.5
    r_perp = 1.0
    # Forward transform
    s_par, s_perp = to_z_space(r_par, r_perp, z, Om, Delta=Delta, f1=f1)
    # Scalar inverse
    r_par_inv = iterative_zspace_inverse_scalar(s_par, s_perp, f1, Delta)
    # Validate
    assert np.isclose(r_par_inv, r_par, atol=1e-4)

def test_iterative_inverse_approx(mock_transform_inputs):
    r_par, r_perp, z, Om, f1, Delta = mock_transform_inputs
    s_par, s_perp = to_z_space(r_par, r_perp, z, Om, Delta=Delta, f1=f1)
    r_par_inv = iterative_zspace_inverse(s_par, s_perp, f1, Delta)
    assert np.allclose(r_par_inv, r_par, atol=1e-2)

def test_to_real_space_matches_inverse(mock_transform_inputs):
    r_par, r_perp, z, Om, f1, Delta = mock_transform_inputs
    s_par, s_perp = to_z_space(r_par, r_perp, z, Om, Delta=Delta, f1=f1)
    r_par_inv, r_perp_inv = to_real_space(
        s_par, s_perp, z=z, Om=Om, Delta=Delta, f1=f1
    )
    assert np.allclose(r_par_inv, r_par, atol=1e-2)
    assert np.allclose(r_perp_inv, r_perp)

# ---------------------- REGRESSION TESTS ----------------------

def test_to_z_space_regression(mock_transform_inputs):
    r_par, r_perp, z, Om, f1, Delta = mock_transform_inputs
    s_par, s_perp = to_z_space(r_par, r_perp, z, Om, Delta=Delta, f1=f1)
    result = np.vstack([s_par, s_perp])
    ref = np.load(os.path.join(SNAPSHOT_DIR, "zspace_transform_ref.npy"))
    np.testing.assert_allclose(result, ref, rtol=1e-6)





