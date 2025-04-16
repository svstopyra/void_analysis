import numpy as np
from cosmology_inference import geometry_correction

def test_geometry_correction_identity():
    s_par = np.array([1.0])
    s_perp = np.array([1.0])
    s_par_new, s_perp_new = geometry_correction(s_par, s_perp, epsilon=1.0)
    assert np.allclose(s_par, s_par_new)
    assert np.allclose(s_perp, s_perp_new)

