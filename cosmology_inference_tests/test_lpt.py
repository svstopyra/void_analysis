import os
import numpy as np
import pytest
import scipy

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
    spherical_lpt_velocity,
    find_suitable_solver_bounds,
    void_los_velocity_ratio_1lpt,
    void_los_velocity_ratio_derivative_1lpt,
    void_los_velocity_ratio_semi_analytic,
    void_los_velocity_ratio_derivative_semi_analytic,
    semi_analytic_model,
    Delta_theta,
    V_theta,
    get_upper_bound,
    invert_Delta_theta_scalar,
    theta_of_Delta
)

from void_analysis.simulation_tools import gaussian_delta, gaussian_Delta
from void_analysis import tools
from void_analysis.cosmology import Hz

SNAPSHOT_DIR = os.path.join(os.path.dirname(__file__), "snapshots")

@pytest.fixture
def mock_profile_data():
    rvals = np.linspace(0,3,101)
    A = 0.85
    sigma = 1
    delta_f = lambda r: gaussian_delta(r,A=A,sigma=sigma)
    Delta_f = lambda r: gaussian_Delta(r,A=A,sigma=sigma)
    Delta = delta_f(rvals)
    return rvals, Delta, delta_f, Delta_f, A, sigma

@pytest.fixture
def mock_lpt_order_data():
    rvals = np.linspace(0,3,101)
    Psi_n = [0.5**n*np.exp(-rvals**2/(2*n**2)) for n in range(1,6)]
    return rvals,Psi_n

@pytest.fixture
def development_angles():
    theta = np.linspace(0,10,101)
    return theta

# -----------------------UNIT TESTS---------------------------------------------

def test_find_suitable_solver_bounds_valid():
    f = lambda x: -x**3
    ulow, uupp = find_suitable_solver_bounds(
        f,-0.75,1.0,taylor_expand=True,
    )
    assert (f(ulow) + 0.75)*(f(uupp) + 0.75) <= 0.0
    f = lambda x: x**3
    ulow, uupp = find_suitable_solver_bounds(
        f,0.75,1.0,taylor_expand=False,
    )
    assert (f(ulow) - 0.75)*(f(uupp) - 0.75) <= 0.0

def test_process_radius_gives_radial_ratio(mock_lpt_order_data):
    rvals, Psi_n = mock_lpt_order_data
    current_value = process_radius(
        rvals,Psi_n,quant_q = Psi_n,radial_fraction=True,eulerian_radius=True,
        taylor_expand=True,order=5,expand_denom_only=False
    )
    ref_data = np.load(
        os.path.join(SNAPSHOT_DIR, "eulerian_ratio_data.npz")
    )
    fully_expanded_ref, denom_only_ref = [ref_data[key] for key in ref_data]
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
    for x, y in zip(ratio_value,full_value):
        np.testing.assert_allclose(x*rvals,y)

def test_lpt_velocity_derivative_integral(mock_profile_data):
    """
    Test whether the derivative profile integrates to the original profile,
    for the lpt model.
    """
    rvals, _, delta_f, Delta_f, _,_ = mock_profile_data
    tools.profile_derivative_test(
        rvals,delta_f,Delta_f,void_los_velocity_ratio_1lpt,
        void_los_velocity_ratio_derivative_1lpt,
        rtol=1e-5,atol=1e-5,f1=0.53,lower_lim = -10
    )

def test_semi_analytic_velocity_derivative_integral(mock_profile_data):
    """
    Test whether the derivative profile integrates to the original profile
    for the semi-analytic model.
    """
    rvals, _, delta_f, Delta_f, _,_ = mock_profile_data
    tools.profile_derivative_test(
        rvals,delta_f,Delta_f,void_los_velocity_ratio_semi_analytic,
        void_los_velocity_ratio_derivative_semi_analytic,
        rtol=1e-5,atol=1e-5,f1=0.53,lower_lim = -10
    )

def test_semi_analytic_model_functions_consistency(mock_profile_data):
    """
    Test whether the two functions, semi_analytic_model, and 
    void_los_velocity_ratio_semi_analytic are consistent with each other. These
    are technically functions of different variables, but represent the same
    velocity model, so they should agree. Note, 
    void_los_velocity_ratio_semi_analytic leaves off the factor of H(z)/(1+z),
    so we need to account for this.
    """
    rvals, Delta, delta_f, Delta_f, A, sigma = mock_profile_data
    u = 1 - np.cbrt(1 + Delta_f(rvals))
    alphas = [-0.5,0.1]
    z = 0.1
    Om = 0.3111
    Ha = Hz(z,Om,h=1)
    function1 = void_los_velocity_ratio_semi_analytic(
        rvals,Delta_f,0.53,params = [-0.5,0.1]
    )
    function2 = ((1.0 + z)/Ha)*semi_analytic_model(
        u,alphas,z=z,Om=Om,f1=0.53,h=1,nf1 = 5/9
    )
    np.testing.assert_allclose(function1,function2)

def test_get_upper_bound_basic():
    """
    Test whether the function to compute upper bounds for solving 
    Delta(theta) actually returns a valid upper bounds. Specifically, the range
    (0,theta_upper) should actually contain a solution of the equation
    Delta_theta(theta) = Delta
    
    """
    Delta = -0.85
    f = lambda x: Delta_theta(x) - Delta - 1
    theta_upper = get_upper_bound(Delta)
    assert(f(0)*f(theta_upper) < 0)

def test_invert_Delta_theta_scalar_basic():
    """
    Test that inver_Delta_theta_scalar actually inverts Delta_theta
    sufficiently closely.
    """
    Delta = -0.85
    theta = invert_Delta_theta_scalar(Delta)
    np.testing.assert_allclose(Delta,Delta_theta(theta)-1)

def test_theta_of_Delta_basic():
    """
    Test that theta_of_Delta actually inverts Delta_theta
    sufficiently closely, for a vectorised range of Delta:
    """
    Delta = np.linspace(0,-1,21)
    theta = theta_of_Delta(Delta)
    Delta2 = Delta_theta(theta) - 1
    np.testing.assert_allclose(Delta,Delta2)

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
    rvals, Delta, _,_,_,_ = mock_profile_data
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
def test_get_initial_condition(mock_profile_data):
    rvals, Delta, _,_,_,_ = mock_profile_data
    current_output = get_initial_condition(
        Delta,order=4,Om=0.3,use_linear_on_fail=False
    )
    reference_output = np.load(
        os.path.join(SNAPSHOT_DIR, "Pert_initial_conditions.npy")
    )
    np.testing.assert_allclose(
        current_output, reference_output, rtol=1e-6, atol=1e-10
    )

def test_get_S1r(mock_profile_data):
    rvals, Delta, _,_,_,_ = mock_profile_data
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
    rvals, Delta, _,_,_,_ = mock_profile_data
    current_output = get_S2r(
        Delta,rvals,0.3,order=5,perturbative_ics = False,taylor_expand=False,
        force_linear_ics = False
    )
    reference_output = np.load(
        os.path.join(SNAPSHOT_DIR, "S2r_data.npy")
    )
    np.testing.assert_allclose(
        current_output, reference_output, rtol=1e-6, atol=1e-10
    )

def test_get_S3r(mock_profile_data):
    rvals, Delta, _,_,_,_ = mock_profile_data
    current_output = get_S3r(
        Delta,rvals,0.3,order=5,perturbative_ics = False,taylor_expand=False,
        force_linear_ics = False
    )
    reference_output = np.load(
        os.path.join(SNAPSHOT_DIR, "S3r_data.npy")
    )
    np.testing.assert_allclose(
        current_output, reference_output, rtol=1e-6, atol=1e-10
    )

def test_get_S4r(mock_profile_data):
    rvals, Delta, _,_,_,_ = mock_profile_data
    current_output = get_S4r(
        Delta,rvals,0.3,order=5,perturbative_ics = False,taylor_expand=False,
        force_linear_ics = False
    )
    reference_output = np.load(
        os.path.join(SNAPSHOT_DIR, "S4r_data.npy")
    )
    np.testing.assert_allclose(
        current_output, reference_output, rtol=1e-6, atol=1e-10
    )

def test_get_S5r(mock_profile_data):
    rvals, Delta, _,_,_,_ = mock_profile_data
    current_output = get_S5r(
        Delta,rvals,0.3,order=5,perturbative_ics = False,taylor_expand=False,
        force_linear_ics = False
    )
    reference_output = np.load(
        os.path.join(SNAPSHOT_DIR, "S5r_data.npy")
    )
    np.testing.assert_allclose(
        current_output, reference_output, rtol=1e-6, atol=1e-10
    )

def test_get_psi_n_r(mock_profile_data):
    rvals, Delta, _,_,_,_ = mock_profile_data
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
    rvals, Delta, _,_,_,_ = mock_profile_data
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
    ref_data = np.load(
        os.path.join(SNAPSHOT_DIR, "eulerian_ratio_data.npz")
    )
    fully_expanded_ref, denom_only_ref = [ref_data[key] for key in ref_data]
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
    rvals, Delta, _,_,_,_ = mock_profile_data
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
    rvals, Delta, _,_,_,_ = mock_profile_data
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

def test_void_los_velocity_ratio_1lpt(mock_profile_data):
    rvals, _, _ , Delta_f, _, _ = mock_profile_data
    computed = void_los_velocity_ratio_1lpt(rvals,Delta_f,0.53)
    reference = np.load(os.path.join(
        SNAPSHOT_DIR, "test_void_los_velocity_ratio_1lpt.npy")
    )
    np.testing.assert_allclose(
        computed, reference, rtol=1e-6, atol=1e-10
    )

def test_void_los_velocity_ratio_derivative_1lpt(mock_profile_data):
    rvals, Delta, delta_f, Delta_f, A, sigma = mock_profile_data
    computed = void_los_velocity_ratio_derivative_1lpt(
        rvals,Delta_f,delta_f,0.53
    )
    reference = np.load(os.path.join(
        SNAPSHOT_DIR, "test_void_los_velocity_ratio_derivative_1lpt.npy")
    )
    np.testing.assert_allclose(
        computed, reference, rtol=1e-6, atol=1e-10
    )

def test_void_los_velocity_ratio_semi_analytic(mock_profile_data):
    rvals, Delta, delta_f, Delta_f, A, sigma = mock_profile_data
    tools.run_basic_regression_test(
        void_los_velocity_ratio_semi_analytic,
        os.path.join(
            SNAPSHOT_DIR, "test_void_los_velocity_ratio_semi_analytic.npy"
        ),
        rvals,Delta_f,0.53,rtol=1e-5,atol=1e-5,params = [-0.5,0.1]
    )

def test_void_los_velocity_ratio_derivative_semi_analytic(mock_profile_data):
    rvals, Delta, delta_f, Delta_f, A, sigma = mock_profile_data
    tools.run_basic_regression_test(
        void_los_velocity_ratio_derivative_semi_analytic,
        os.path.join(
            SNAPSHOT_DIR, 
            "test_void_los_velocity_ratio_derivative_semi_analytic.npy"
        ),
        rvals,Delta_f,delta_f,0.53,rtol=1e-5,atol=1e-5,params = [-0.5,0.1]
    )

def test_semi_analytic_model(mock_profile_data):
    rvals, Delta, delta_f, Delta_f, A, sigma = mock_profile_data
    u = 1 - np.cbrt(1 + Delta)
    alphas = [-0.5,0.1]
    tools.run_basic_regression_test(
        semi_analytic_model,
        os.path.join(
            SNAPSHOT_DIR, 
            "semi_analytic_model_ref.npy"
        ),
        u,alphas,z=0,Om=0.3111,f1=0.53,h=1,nf1 = 5/9
    )

def test_Delta_theta(development_angles):
    theta = development_angles
    tools.run_basic_regression_test(
        Delta_theta,
        os.path.join(SNAPSHOT_DIR,"Delta_theta_ref.npy"),
        theta
    )

def test_V_theta(development_angles):
    theta = development_angles
    tools.run_basic_regression_test(
        V_theta,
        os.path.join(SNAPSHOT_DIR,"V_theta_ref.npy"),
        theta
    )

def test_get_upper_bound():
    tools.run_basic_regression_test(
        get_upper_bound,
        os.path.join(SNAPSHOT_DIR,"get_upper_bound_ref.npy"),
        -0.85
    )

def test_invert_Delta_theta_scalar():
    Delta = -0.85
    tools.run_basic_regression_test(
        invert_Delta_theta_scalar,
        os.path.join(SNAPSHOT_DIR,"invert_Delta_theta_scalar_ref.npy"),
        -0.85
    )

def test_theta_of_Delta():
    Delta = np.linspace(0,-1,21)
    tools.run_basic_regression_test(
        theta_of_Delta,
        os.path.join(SNAPSHOT_DIR,"theta_of_Delta_ref.npy"),
        Delta
    )