import numpy as np
from void_analysis.cosmology_inference import (
    profile_modified_hamaus,
    integrated_profile_modified_hamaus,
    profile_broken_power,
    profile_broken_power_log,
    profile_modified_hamaus_derivative,
    rho_real
)

GENERATED_SNAPSHOTS = [
    "modified_hamaus_ref.npy",
    "integrated_hamaus_ref.npy",
    "broken_power_ref.npy",
    "profile_broken_power_log_ref.npy",
    "profile_modified_hamaus_derivative_ref.npz",
    "rho_real_ref.npy"
]

from void_analysis import tools

def generate_snapshots():
    r = np.linspace(0.1, 3.0, 100)

    # Modified Hamaus
    hamaus_params = (1.5, 3.0, 1.0, -0.5, 0.1, 1.0)
    modified = profile_modified_hamaus(r, *hamaus_params)
    np.save("modified_hamaus_ref.npy", modified)

    # Integrated Hamaus
    integrated = integrated_profile_modified_hamaus(r, *hamaus_params)
    np.save("integrated_hamaus_ref.npy", integrated)

    # Broken power
    broken_params = (1.0, 1.0, 2.0, 1.0, 0.5)
    broken = profile_broken_power(r, *broken_params)
    np.save("broken_power_ref.npy", broken)
    
    # profile_broken_power_log
    r = np.linspace(0.1, 3.0, 100)
    params = (1.0, 1.0, 2.0, 1.0, 0.5)
    tools.generate_regression_test_data(
        profile_broken_power_log,
        "profile_broken_power_log_ref.npy",
        r,*params
    )
    # profile_modified_hamaus_derivative
    r = np.linspace(0.1, 3.0, 100)
    params = (1.5, 3.0, 1.0, -0.5, 0.1, 1.0)
    computed = lambda r, *params: (
        profile_modified_hamaus_derivative(r,0,*params),
        profile_modified_hamaus_derivative(r,1,*params),
        profile_modified_hamaus_derivative(r,2,*params)
    )
    tools.generate_regression_test_data(
        computed,
        "profile_modified_hamaus_derivative_ref.npz",
        r,*params
    )
    
    # rho_real:
    r = np.linspace(0.1, 3.0, 100)
    params = (1.5, 3.0, 1.0, -0.5, 0.1, 1.0)
    tools.generate_regression_test_data(
        rho_real,
        "rho_real_ref.npy",
        r,*params
    )

    print("Snapshots saved.")

if __name__ == "__main__":
    generate_snapshots()
