# generate_zspace_snapshots.py

import numpy as np
from void_analysis.cosmology_inference import (
    to_real_space,
    z_space_jacobian,
    z_space_profile,
    to_z_space,
    iterative_zspace_inverse_scalar,
    iterative_zspace_inverse,
    void_los_velocity,
    void_los_velocity_derivative,
    get_dudr_hz_o1pz
)

GENERATED_SNAPSHOTS = [
    "to_real_space_ref.npy",
    "to_z_space_ref.npy",
    "zspace_jacobian_ref.npy",
    "zspace_profile_ref.npy",
    "iterative_zspace_inverse_scalar_ref.npy",
    "iterative_zspace_inverse_ref.npy",
    "void_los_velocity_ref.npy",
    "void_los_velocity_derivative_ref.npy"
]

def generate_snapshots():
    np.random.seed(42)

    r_par = np.linspace(-2.0, 2.0, 100)
    r_perp = np.linspace(0.1, 2.0, 100)
    z = 0.1
    Om = 0.3
    f1 = 0.8

    Delta = lambda r: 0.2 * np.exp(-r**2)
    delta = lambda r: -0.2 * np.exp(-r**2)
    rho_real_func = lambda r: delta(r) + 1.0

    # Save to_real_space snapshot
    s_par_real, s_perp_real = to_real_space(
        r_par, r_perp, z=z, Om=Om, Delta=Delta, f1=f1
    )
    np.save("to_real_space_ref.npy", np.vstack([s_par_real, s_perp_real]))

    # Save to_z_space snapshot
    s_par_z, s_perp_z = to_z_space(
        r_par, r_perp, z=z, Om=Om, Delta=Delta, f1=f1
    )
    np.save("to_z_space_ref.npy", np.vstack([s_par_z, s_perp_z]))

    # Save z_space_jacobian snapshot
    J = z_space_jacobian(Delta, delta, r_par, r_perp, Om=Om,z=z)
    np.save("zspace_jacobian_ref.npy", J)

    # Save z_space_profile snapshot
    density = z_space_profile(
        s_par_real, s_perp_real, rho_real_func,Delta, delta,z=z,Om=Om,f1=f1
    )
    np.save("zspace_profile_ref.npy", density)

    # Save iterative_zspace_inverse_scalar snapshot
    s_par_scalar = 1.0
    s_perp_scalar = 0.5
    f_scalar = 0.5
    r_par_estimated = iterative_zspace_inverse_scalar(
        s_par_scalar, s_perp_scalar, f_scalar, lambda r: 0.0
    )
    np.save(
        "iterative_zspace_inverse_scalar_ref.npy", np.array(r_par_estimated)
    )

    # Save iterative_zspace_inverse snapshot
    s_par_array = np.linspace(0.5, 2.0, 5)
    s_perp_array = np.linspace(0.1, 1.0, 5)
    r_par_array = iterative_zspace_inverse(
        s_par_array, s_perp_array, f_scalar, lambda r: 0.0
    )
    np.save("iterative_zspace_inverse_ref.npy", r_par_array)

    # New snapshots for LOS velocity and derivative
    r = np.linspace(0.1, 2.0, 50)
    Delta_func = lambda r: 0.2 * np.exp(-r**2)
    dDelta_func = lambda r: -2 * r * 0.2 * np.exp(-r**2)
    v_los = void_los_velocity(z, Delta_func, r_par,r_perp,Om)
    v_los_deriv = void_los_velocity_derivative(z, Delta_func, dDelta_func,
                                               r_par,r_perp,Om)
    np.save("void_los_velocity_ref.npy", v_los)
    np.save("void_los_velocity_derivative_ref.npy", v_los_deriv)

    print("Snapshots for to_real_space, to_z_space, z_space_jacobian, " + 
          "z_space_profile, iterative_zspace_inverse_scalar, and " + 
          "iterative_zspace_inverse saved.")

    # Consistent framework for regression tests:
    tools.generate_regression_test_data(
        get_dudr_hz_o1pz,
        "get_dudr_hz_o1pz_ref.npy",
        Delta_f, delta_f, r_par, r_perp, f1,
        vel_model = void_los_velocity_ratio_1lpt,
        dvel_dlogr_model = void_los_velocity_ratio_derivative_1lpt
    )

if __name__ == "__main__":
    generate_snapshots()
