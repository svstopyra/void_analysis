import os
import numpy as np
import pytest

from void_analysis.cosmology_inference import (
    get_D_coefficients,
    get_ic_polynomial_coefficients,
    get_nonperturbative_polynomial_coefficients,
    get_taylor_polynomial_coefficients,
    get_initial_condition_non_perturbative,
    get_initial_condition,
    get_S1r,
    get_S2r,
    get_S3r,
    get_S4r,
    get_S5r,
    get_psi_n_r,
    get_delta_lpt,
    get_eulerian_ratio_from_lagrangian_ratio,
    process_radius,
    spherical_lpt_displacement,
    spherical_lpt_velocity
)

SNAPSHOT_DIR = os.path.join(os.path.dirname(__file__), "snapshots")

@pytest.fixture
def mock_profile_data():
    rvals = np.linspace(0,3,101)
    Delta = -0.85*np.exp(-rvals**2/2)
    return rvals, Delta

@pytest.fixture
def mock_lpt_order_data():
    rvals = np.linspace(0,3,101)
    Psi_n = [0.5**n*np.exp(-rvals**2/(2*n**2)) for n in range(1,6)]
    return rvals,Psi_n

# -----------------------UNIT TESTS---------------------------------------------

def test_find_suitable_solver_bounds_valid():
    f = lambda x: -x**3
    ulow, uupp = find_suitable_solver_bounds(
        f,-5,1.0,taylor_expand=False,
    )
    assert (f(ulow) + 5)*(f(uup) + 5) <= 0.0

def test_process_radius_gives_radial_ratio(mock_lpt_order_data):
    rvals, Psi_n = mock_lpt_order_data
    current_value = process_radius(
        rvals,Psi_n,quant_q = Psi_n,radial_fraction=True,eulerian_radius=True,
        taylor_expand=True,order=5,expand_denom_only=False
    )
    fully_expanded_ref, denom_only_ref = np.load(
        os.path.join(SNAPSHOT_DIR, "eulerian_ratio_data.npz")
    )
    # With eulerian_radius=True, should always return the output of 
    # get_eulerian_ratio_from_lagrangian_ratio.
    np.testing.assert_allclose(
        current_value, fully_expanded_ref, rtol=1e-6, atol=1e-10
    )
    # With eulerian_radius=False, should return the input unchanged
    current_value = process_radius(
        rvals,Psi_n,radial_fraction=True,eulerian_radius=False,
        taylor_expand=True,order=5,expand_denom_only=False
    )
    np.testing.assert_allclose(current_value,Psi_n)
    # radial_fraction = True and radial_fraction = False should differ by only
    # the radius:
    ratio_value = process_radius(
        rvals,Psi_n,radial_fraction=True,eulerian_radius=True,
        taylor_expand=False,order=5,expand_denom_only=False
    )
    full_value = process_radius(
        rvals,Psi_n,radial_fraction=False,eulerian_radius=True,
        taylor_expand=False,order=5,expand_denom_only=False
    )
    assert(isinstance(ratio_value,type(full_value)))
    if isinstance(ratio_value,list):
        for x, y in zip(ratio_value,full_value):
            np.testing.assert_allclose(x*rvals,y)
    else:
        np.testing.assert_allclose(ratio_value*rvals,full_value)
    
# ---------------------- REGRESSION TESTS---------------------------------------

def test_get_D_coefficients():
    Om = 0.3
    current_output = get_D_coefficients(Om,order=5,return_all=True)
    # Load saved reference data
    reference_output = np.load(os.path.join(SNAPSHOT_DIR, "D_coefficients.npy"))
    # Regression check
    assert len(current_output) == len(reference_output)
    np.testing.assert_allclose(
        current_output, reference_output, rtol=1e-6, atol=1e-10
    )

def test_get_ic_polynomial_coefficients():
    current_output = get_ic_polynomial_coefficients(5,Om=0.3)
    # Load saved reference data
    reference_output = np.load(
        os.path.join(SNAPSHOT_DIR, "A_poly_coefficients.npy")
    )
    # Regression check
    np.testing.assert_allclose(
        current_output, reference_output, rtol=1e-6, atol=1e-10
    )

def test_get_nonperturbative_polynomial_coefficients():
    current_output = get_nonperturbative_polynomial_coefficients(5,Om=0.3)
    # Load saved reference data
    reference_output = np.load(
        os.path.join(SNAPSHOT_DIR, "B_poly_coefficients.npy")
    )
    # Regression check
    np.testing.assert_allclose(
        current_output, reference_output, rtol=1e-6, atol=1e-10
    )

def test_get_taylor_polynomial_coefficients():
    current_output = get_taylor_polynomial_coefficients(5,Om=0.3)
    # Load saved reference data
    reference_output = np.load(
        os.path.join(SNAPSHOT_DIR, "Bp_poly_coefficients.npy")
    )
    # Regression check
    np.testing.assert_allclose(
        current_output, reference_output, rtol=1e-6, atol=1e-10
    )

def test_get_initial_condition_non_perturbative(mock_profile_data):
    rvals, Delta = mock_profile_data
    current_output = get_initial_condition_non_perturbative(
        Delta,order=5,Om=0.3,use_linear_on_fail=False,
        taylor_expand=False
    )
    reference_output = np.load(
        os.path.join(SNAPSHOT_DIR, "NonPert_initial_conditions.npy")
    )
    np.testing.assert_allclose(
        current_output, reference_output, rtol=1e-6, atol=1e-10
    )

# Test needs fixing, so skip for now
def broken_test_get_initial_condition(mock_profile_data):
    rvals, Delta = mock_profile_data
    current_output = get_initial_condition(
        Delta,order=5,Om=0.3,use_linear_on_fail=False
    )
    reference_output = np.load(
        os.path.join(SNAPSHOT_DIR, "Pert_initial_conditions.npy")
    )
    np.testing.assert_allclose(
        current_output, reference_output, rtol=1e-6, atol=1e-10
    )

def test_get_S1r(mock_profile_data):
    rvals, Delta = mock_profile_data
    current_output = get_S1r(
        Delta,rvals,0.3,order=5,perturbative_ics = False,taylor_expand=False,
        force_linear_ics = False
    )
    reference_output = np.load(
        os.path.join(SNAPSHOT_DIR, "S1r_data.npy")
    )
    np.testing.assert_allclose(
        current_output, reference_output, rtol=1e-6, atol=1e-10
    )

def test_get_S2r(mock_profile_data):
    rvals, Delta = mock_profile_data
    current_output = get_S2r(
        Delta,rvals,0.3,order=5,perturbative_ics = False,taylor_expand=False,
        force_linear_ics = False
    )
    reference_output = np.load(
        os.path.join(SNAPSHOT_DIR, "S1r_data.npy")
    )
    np.testing.assert_allclose(
        current_output, reference_output, rtol=1e-6, atol=1e-10
    )

def test_get_S3r(mock_profile_data):
    rvals, Delta = mock_profile_data
    current_output = get_S3r(
        Delta,rvals,0.3,order=5,perturbative_ics = False,taylor_expand=False,
        force_linear_ics = False
    )
    reference_output = np.load(
        os.path.join(SNAPSHOT_DIR, "S1r_data.npy")
    )
    np.testing.assert_allclose(
        current_output, reference_output, rtol=1e-6, atol=1e-10
    )

def test_get_S4r(mock_profile_data):
    rvals, Delta = mock_profile_data
    current_output = get_S4r(
        Delta,rvals,0.3,order=5,perturbative_ics = False,taylor_expand=False,
        force_linear_ics = False
    )
    reference_output = np.load(
        os.path.join(SNAPSHOT_DIR, "S1r_data.npy")
    )
    np.testing.assert_allclose(
        current_output, reference_output, rtol=1e-6, atol=1e-10
    )

def test_get_S5r(mock_profile_data):
    rvals, Delta = mock_profile_data
    current_output = get_S5r(
        Delta,rvals,0.3,order=5,perturbative_ics = False,taylor_expand=False,
        force_linear_ics = False
    )
    reference_output = np.load(
        os.path.join(SNAPSHOT_DIR, "S1r_data.npy")
    )
    np.testing.assert_allclose(
        current_output, reference_output, rtol=1e-6, atol=1e-10
    )

def test_get_psi_n_r(mock_profile_data):
    rvals, Delta = mock_profile_data
    current_output = get_psi_n_r(
        Delta,rvals,5,z=0,Om=0.3,order=5,return_all=True
    )
    reference_output = np.load(
        os.path.join(SNAPSHOT_DIR, "psi_n_r.npy")
    )
    np.testing.assert_allclose(
        current_output, reference_output, rtol=1e-6, atol=1e-10
    )

def test_get_delta_lpt(mock_profile_data):
    rvals, Delta = mock_profile_data
    current_output = get_delta_lpt(
        Delta,z=0,Om=0.3,order=5,return_all=True
    )
    reference_output = np.load(
        os.path.join(SNAPSHOT_DIR, "Delta_lpt.npy")
    )
    np.testing.assert_allclose(
        current_output, reference_output, rtol=1e-6, atol=1e-10
    )

def test_get_eulerian_ratio_from_lagrangian_ratio(mock_lpt_order_data):
    rvals, Psi_n = mock_lpt_order_data
    # Test both with and without expanding the numerator:
    fully_expanded = get_eulerian_ratio_from_lagrangian_ratio(
        Psi_n,Psi_n,5,expand_denom_only=False
    )
    denom_only = get_eulerian_ratio_from_lagrangian_ratio(
        np.sum(Psi_n,0),Psi_n,5,expand_denom_only=True
    )
    fully_expanded_ref, denom_only_ref = np.load(
        os.path.join(SNAPSHOT_DIR, "eulerian_ratio_data.npz")
    )
    # Assertions:
    np.testing.assert_allclose(
        denom_only, denom_only_ref, rtol=1e-6, atol=1e-10
    )
    np.testing.assert_allclose(
        fully_expanded, fully_expanded_ref, rtol=1e-6, atol=1e-10
    )

def test_process_radius(mock_lpt_order_data):
    rvals, Psi_n = mock_lpt_order_data
    current_output = process_radius(
        rvals,Psi_n,radial_fraction=True,eulerian_radius=True,
        taylor_expand=True,order=5,expand_denom_only=False
    )

def test_spherical_lpt_displacement(mock_profile_data):
    rvals, Delta = mock_profile_data
    current_value = spherical_lpt_displacement(
        rvals,Delta,order=5,z=0,Om=0.3,fixed_delta = True,
        radial_fraction = True,eulerian_radius=False,expand_denom_only=False,
        taylor_expand=False,expand_euler_ratio=False
    )
    reference_value = np.load(
        os.path.join(SNAPSHOT_DIR, "lpt_displacement.npy")
    )
    np.testing.assert_allclose(
        current_value, reference_value, rtol=1e-6, atol=1e-10
    )

def test_spherical_lpt_velocity(mock_profile_data):
    rvals, Delta = mock_profile_data
    current_value = spherical_lpt_velocity(
        rvals,Delta,order=5,z=0,Om=0.3,radial_fraction = True,
        fixed_delta = True,eulerian_radius=False,taylor_expand=False,
        expand_denom_only=False,expand_euler_ratio=False,return_all=False
    )
    reference_value = np.load(
        os.path.join(SNAPSHOT_DIR, "lpt_velocity.npy")
    )
    np.testing.assert_allclose(
        current_value, reference_value, rtol=1e-6, atol=1e-10
    )














