import numpy as np
from void_analysis.cosmology_inference import (
    profile_modified_hamaus,
    integrated_profile_modified_hamaus,
    profile_broken_power
)

GENERATED_SNAPSHOTS = [
    "modified_hamaus_ref.npy",
    "integrated_hamaus_ref.npy",
    "broken_power_ref.npy"
]

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

    print("Snapshots saved.")

if __name__ == "__main__":
    generate_snapshots()
