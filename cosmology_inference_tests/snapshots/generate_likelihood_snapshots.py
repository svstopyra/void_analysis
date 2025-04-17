# generate_likelihood_snapshots.py
import numpy as np
from void_analysis.cosmology_inference import (
    get_profile_parameters_fixed,
    profile_modified_hamaus
)

GENERATED_SNAPSHOTS = [
    "profile_fit_params_ref.npy"
]

def generate_snapshots():
    np.random.seed(1)
    r = np.linspace(0.1, 10.0, 300)
    true_params = (1.5, 3.0, 1.0, -0.5, 0.1, 1.0)
    delta = profile_modified_hamaus(r, *true_params)
    delta_noisy = delta + 0.02 * np.random.randn(len(r))
    sigma = 0.02 * np.ones_like(r)

    best_fit = get_profile_parameters_fixed(r, delta_noisy, sigma, model=profile_modified_hamaus)
    np.save("profile_fit_params_ref.npy", best_fit)
    print("Saved regression snapshot for best-fit profile parameters.")

if __name__ == "__main__":
    generate_snapshots()
