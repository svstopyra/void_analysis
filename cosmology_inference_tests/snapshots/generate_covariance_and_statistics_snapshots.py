# generate_covariance_and_statistics_snapshots.py

import numpy as np
from void_analysis.cosmology_inference import (
    tikhonov_regularisation,
    regularise_covariance,
    get_inverse_covariance,
    range_excluding,
    get_nonsingular_subspace,
    get_solved_residuals,
    compute_normality_test_statistics,
    covariance,
    profile_jackknife_covariance,
    compute_singular_log_likelihood,
    get_covariance_matrix
)

GENERATED_SNAPSHOTS = [
    "tikhonov_regularisation_ref.npy",
    "regularise_covariance_ref.npy",
    "inverse_covariance_ref.npy",
    "normality_statistics_ref.npy",
    "covariance_old_ref.npy",
    "profile_jackknife_covariance_ref.npy",
    "singular_log_likelihood_ref.npy",
    "get_solved_residuals_ref.npy"
]

from void_analysis import tools

def generate_snapshots():
    np.random.seed(42)

    # Simple matrix for regularisation tests
    A = np.eye(5)
    cov = np.random.rand(5, 5)
    cov = cov @ cov.T  # Make positive semi-definite
    noisy_cov = cov + 1e-3 * np.eye(5)

    reg_A = tikhonov_regularisation(A, lambda_reg=1e-6)
    reg_cov = regularise_covariance(cov, lambda_reg=1e-8)
    inv_cov = get_inverse_covariance(noisy_cov)

    samples = np.random.randn(5, 100)
    stats = np.array(compute_normality_test_statistics(samples))
    
    # Additional snapshots
    cov_direct = covariance(samples.T)
    jackknife_cov = profile_jackknife_covariance(samples.T, 
                                                 lambda x: np.mean(x,0)**2)
    xbar = np.mean(samples, axis=1)
    Umap, good_eig = get_nonsingular_subspace(cov, 1e-3)
    singular_log_like = compute_singular_log_likelihood(xbar,Umap,good_eig)

    np.save("tikhonov_regularisation_ref.npy", reg_A)
    np.save("regularise_covariance_ref.npy", reg_cov)
    np.save("inverse_covariance_ref.npy", inv_cov)
    np.save("normality_statistics_ref.npy", stats)
    np.save("covariance_old_ref.npy", cov_direct)
    np.save("profile_jackknife_covariance_ref.npy", jackknife_cov)
    np.save("singular_log_likelihood_ref.npy", singular_log_like)

    # get_solved_residuals
    np.random.seed(0)
    samples = np.random.randn(5, 100)
    cov = np.cov(samples)
    xbar = np.mean(samples, axis=1)
    tools.generate_regression_test_data(
        get_solved_residuals,
        "get_solved_residuals_ref.npy",
        samples,cov,xbar
    )

    print("✅ Covariance and statistical snapshots saved!")

if __name__ == "__main__":
    generate_snapshots()

