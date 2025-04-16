import numpy as np
import os
from void_analysis.cosmology_inference import geometry_correction

SNAPSHOT_DIR = os.path.join(os.path.dirname(__file__), "snapshots")

# ---------------------- UNIT TESTS ----------------------

def test_geometry_correction_identity():
    s_par = np.linspace(-2, 2, 100)
    s_perp = np.ones_like(s_par)
    epsilon = 1.0
    s_par_new, s_perp_new = geometry_correction(s_par, s_perp, epsilon)
    assert np.allclose(np.abs(s_par), np.abs(s_par_new))
    assert np.allclose(np.abs(s_perp), np.abs(s_perp_new))

def test_geometry_correction_output_shape():
    s_par = np.linspace(-2, 2, 100)
    s_perp = np.linspace(0.5, 1.5, 100)
    s_par_new, s_perp_new = geometry_correction(s_par, s_perp, epsilon=1.1)
    assert s_par_new.shape == s_par.shape
    assert s_perp_new.shape == s_perp.shape
    assert np.all(np.isfinite(s_par_new))
    assert np.all(np.isfinite(s_perp_new))

def test_geometry_correction_reversibility():
    s_par = np.linspace(-2, 2, 100)
    s_perp = np.linspace(0.5, 1.5, 100)
    eps = 1.05
    # Forward then inverse
    s_par_new, s_perp_new = geometry_correction(s_par, s_perp, epsilon=eps)
    s_par_rev, s_perp_rev = geometry_correction(s_par_new, s_perp_new, epsilon=1/eps)
    assert np.allclose(np.abs(s_par), np.abs(s_par_rev), atol=1e-4)
    assert np.allclose(np.abs(s_perp), np.abs(s_perp_rev), atol=1e-4)

# ---------------------- REGRESSION TEST ----------------------

def test_geometry_correction_regression():
    s_par = np.linspace(-2, 2, 100)
    s_perp = np.linspace(0.5, 1.5, 100)
    epsilon = 1.08
    s_par_new, s_perp_new = geometry_correction(s_par, s_perp, epsilon)
    result = np.vstack([s_par_new, s_perp_new])
    ref = np.load(os.path.join(SNAPSHOT_DIR, "geometry_correction_ref.npy"))
    np.testing.assert_allclose(result, ref, rtol=1e-6)

