# generate_logprob_snapshot.py
import numpy as np
from void_analysis.cosmology_inference import log_probability_aptest

GENERATED_SNAPSHOTS = [
    "logprob_aptest_ref.npy"
]

def generate_snapshots():
    np.random.seed(42)

    theta = [0.99, 0.5]
    z = 0.02
    field = np.random.normal(1.0, 0.01, size=100)
    scoords = np.random.rand(100, 2)
    cov = np.eye(100)

    Delta = lambda r: 0.2 * np.exp(-r**2)
    delta = lambda r: -0.2 * np.exp(-r**2)
    rho_real = lambda r: delta(r) + 1.0

    logp = log_probability_aptest(
        theta, field, scoords, cov, z, Delta, delta, rho_real,
        cholesky=False,
        tabulate_inverse=False,
        sample_epsilon=True,
        theta_ranges=[[0.9, 1.1], [0.0, 1.0]],
        singular=False,
        log_density=False,
        F_inv=None,
        Umap=None,
        good_eig=None
    )

    np.save("logprob_aptest_ref.npy", logp)
    print("Saved log-probability snapshot.")

if __name__ == "__main__":
    generate_snapshots()
