# test_z_space_profile.py

import numpy as np
import os
import pytest
from void_analysis.cosmology_inference import (
    z_space_profile,
    to_real_space,
    z_space_jacobian,
    to_z_space
)

SNAPSHOT_DIR = os.path.join(os.path.dirname(__file__), "snapshots")

# ---------------------- UNIT TESTS: to_real_space ----------------------

@pytest.fixture
def synthetic_profile_data():
    r_par = np.linspace(-2.0, 2.0, 100)
    r_perp = np.linspace(0.1, 2.0, 100)
    z = 0.1
    Om = 0.3

    Delta = lambda r: 0.2 * np.exp(-r**2)
    delta = lambda r: -0.2 * np.exp(-r**2)
    rho_real_func = lambda r: delta(r) + 1.0

    return r_par, r_perp, z, Om, Delta, delta, rho_real_func

def test_to_real_space_preserves_shape(synthetic_profile_data):
    r_par, r_perp, z, Om, Delta, delta, _ = synthetic_profile_data
    s_par, s_perp = to_real_space(r_par, r_perp, z, Om, Delta=Delta, f=0.8)
    assert s_par.shape == r_par.shape
    assert s_perp.shape == r_perp.shape

def test_to_real_space_consistency(synthetic_profile_data):
    r_par, r_perp, z, Om, Delta, delta, _ = synthetic_profile_data
    s_par, s_perp = to_z_space(r_par, r_perp, z, Om, Delta=Delta, f=0.8)
    r_par_rec, r_perp_rec = to_real_space(s_par, s_perp, z, Om, Delta=Delta, f=0.8)
    assert np.allclose(np.abs(r_par), np.abs(r_par_rec), atol=1e-2)

# ---------------------- UNIT TESTS: z_space_jacobian ----------------------

def test_z_space_jacobian_positive(synthetic_profile_data):
    r_par, r_perp, z, Om, Delta, delta, _ = synthetic_profile_data
    J = z_space_jacobian(z, Delta, delta, r_par, r_perp, Om)
    assert J.shape == r_par.shape
    assert np.all(J > 0)

def test_z_space_jacobian_finite(synthetic_profile_data):
    r_par, r_perp, z, Om, Delta, delta, _ = synthetic_profile_data
    J = z_space_jacobian(z, Delta, delta, r_par, r_perp, Om)
    assert np.all(np.isfinite(J))

# ---------------------- UNIT TESTS: z_space_profile ----------------------

def test_z_space_profile_basic(synthetic_profile_data):
    r_par, r_perp, z, Om, Delta, delta, rho_real_func = synthetic_profile_data
    s_par, s_perp = to_real_space(r_par, r_perp, z, Om, Delta=Delta, f=0.8)
    density = z_space_profile(s_par, s_perp, rho_real_func, z, Om, Delta, delta)

    assert density.shape == r_par.shape
    assert np.all(np.isfinite(density))
    assert np.all(density > 0)

# ---------------------- REGRESSION TESTS ----------------------

def test_z_space_profile_regression(synthetic_profile_data):
    r_par, r_perp, z, Om, Delta, delta, rho_real_func = synthetic_profile_data
    s_par, s_perp = to_real_space(r_par, r_perp, z, Om, Delta=Delta, f=0.8)
    density = z_space_profile(s_par, s_perp, rho_real_func, z, Om, Delta, delta)

    ref = np.load(os.path.join(SNAPSHOT_DIR, "zspace_profile_ref.npy"))
    np.testing.assert_allclose(density, ref, rtol=1e-6)

def test_to_real_space_regression(synthetic_profile_data):
    r_par, r_perp, z, Om, Delta, delta, _ = synthetic_profile_data
    s_par, s_perp = to_real_space(r_par, r_perp, z, Om, Delta=Delta, f=0.8)
    result = np.vstack([s_par, s_perp])
    ref = np.load(os.path.join(SNAPSHOT_DIR, "to_real_space_ref.npy"))
    np.testing.assert_allclose(result, ref, rtol=1e-6)

def test_z_space_jacobian_regression(synthetic_profile_data):
    r_par, r_perp, z, Om, Delta, delta, _ = synthetic_profile_data
    J = z_space_jacobian(z, Delta, delta, r_par, r_perp, Om)
    ref = np.load(os.path.join(SNAPSHOT_DIR, "zspace_jacobian_ref.npy"))
    np.testing.assert_allclose(J, ref, rtol=1e-6)

def test_to_z_space_regression(synthetic_profile_data):
    r_par, r_perp, z, Om, Delta, delta, _ = synthetic_profile_data
    s_par, s_perp = to_z_space(r_par, r_perp, z, Om, Delta=Delta, f=0.8)
    result = np.vstack([s_par, s_perp])
    ref = np.load(os.path.join(SNAPSHOT_DIR, "to_z_space_ref.npy"))
    np.testing.assert_allclose(result, ref, rtol=1e-6)

