# test_inference_core.py
import numpy as np
import pytest
import os
from void_analysis.cosmology_inference import (
    run_inference,
    generate_scoord_grid,
    profile_modified_hamaus,
    integrated_profile_modified_hamaus,
    z_space_profile,
    get_nonsingular_subspace,
    void_los_velocity_ratio_1lpt,
    void_los_velocity_ratio_derivative_1lpt,
    log_probability_aptest,
    log_likelihood_aptest,
    run_inference_pipeline
)
from void_analysis import tools
from void_analysis.simulation_tools import gaussian_delta, gaussian_Delta
import scipy


SNAPSHOT_DIR = os.path.join(os.path.dirname(__file__), "snapshots")

@pytest.fixture
def mock_aptest_data():
    # Setup co-ordinate grid:
    sperp_bins, spar_bins = np.linspace(0,2,5),np.linspace(0,2,5)
    scoords = generate_scoord_grid(sperp_bins, spar_bins)
    N = (len(sperp_bins)-1)*(len(spar_bins)-1)
    # Test profile
    ri = np.linspace(0.1, 10.0, 300)
    true_params = (1.5, 3.0, 1.0, -0.85, 0.0, 1.0)
    deltai = profile_modified_hamaus(ri, *true_params)
    sigma_deltai = np.full(deltai.shape,0.05)
    delta = lambda r: profile_modified_hamaus(r, *true_params)
    Delta = lambda r: integrated_profile_modified_hamaus(r, *true_params)
    rho_real = lambda r: delta(r) + 1.0
    # True profile:
    rhoz = z_space_profile(
        scoords[:,0], scoords[:,1], rho_real, Delta, delta, f1=0.53,z=0,
        Om=0.3111,Om_fid=0.3111, epsilon=1.0, apply_geometry=False, 
        vel_params=None
    )
    # Mock Gaussian distribution:
    mean = rhoz
    np.random.seed(0)
    A = np.random.randn(N, N)*0.05
    cov = A @ A.T
    samples = scipy.stats.multivariate_normal(mean,cov,seed=42)
    # Data:
    data = samples.rvs()
    # Arguments:
    true_params = (1.5, 3.0, 1.0, -0.85, 0.0, 1.0)
    profile_ranges = [
        [0,np.inf],[0,np.inf],[0,np.inf],[-1,0],[-1,1],[0,2]
    ]
    np.random.seed(42)
    theta_ranges_list = [[0.9,1.1],[0,1]] + profile_ranges
    theta_initial = (
        np.hstack([np.array([1.0,0.53]),np.array(true_params)])
        + np.random.rand(2 + len(true_params))*0.05
    )
    filename = "temp_inference_test.h5"
    cholesky_matrix = scipy.linalg.cholesky(cov,lower=True)
    z = 0.0
    delta = profile_modified_hamaus
    Delta = integrated_profile_modified_hamaus
    rho_real = lambda *args: delta(*args) + 1.0
    Umap, good_eig = get_nonsingular_subspace(
        cov, lambda_reg=1e-27,
        lambda_cut=1e-23, normalised_cov=False,
        mu=mean)
    N_vel = 0
    # Arguments for log_probability_aptest:
    args = (data,scoords,
        cholesky_matrix,z,Delta,
        delta,rho_real)
    kwargs = {'cholesky':True,'tabulate_inverse':True,
              'sample_epsilon':True,'theta_ranges':theta_ranges_list,
              'singular':False,'Umap':Umap,'good_eig':good_eig,'F_inv':None,
              'log_density':False,'infer_profile_args':True,
              'linearise_jacobian':False,
              'vel_model':void_los_velocity_ratio_1lpt,
              'dvel_dlogr_model':void_los_velocity_ratio_derivative_1lpt,
              'N_vel':N_vel,'data_filter':None,'normalised':False,'ntab':10,
              'Om_fid':0.3,'N_prof':6}
    return sperp_bins,spar_bins,scoords,mean, cov, data, true_params,\
           ri, deltai, sigma_deltai, args, kwargs, N_vel, Umap,\
           good_eig,rho_real,Delta,delta,z,\
           cholesky_matrix, filename,theta_initial,theta_ranges_list


# ---------------------- UNIT TESTS: -------------------------------------------



def test_run_inference_basic(mock_aptest_data):
    sperp_bins,spar_bins,scoords,mean, cov, data, true_params,\
           ri, deltai, sigma_deltai, args, kwargs, N_vel, Umap,\
           good_eig,rho_real,Delta,delta,z,\
           cholesky_matrix, filename,theta_initial,theta_ranges_list\
           = mock_aptest_data
    tau, sampler = run_inference(
        data, theta_ranges_list, theta_initial, filename,
        log_probability_aptest, *args,
        redo_chain=True,
        backup_start=True,
        nwalkers=16, sample="all", n_mcmc=10,
        disp=1e-2, max_n=10, z=0.0,
        parallel=False, batch_size=10, n_batches=1,
        autocorr_file=None, **kwargs
    )
    tau_max = np.max(tau)
    flat_samples = sampler.get_chain()
    assert(len(tau) == 8)
    assert(len(tau) == flat_samples.shape[2])


def test_run_inference_pipeline_basic(mock_aptest_data):
    sperp_bins,spar_bins,scoords,mean, cov, data, true_params,\
           ri, deltai, sigma_deltai, args, kwargs, N_vel, Umap,\
           good_eig,rho_real,Delta,delta,z,\
           cholesky_matrix, filename,theta_initial,theta_ranges_list\
           = mock_aptest_data
    tau, sampler = run_inference_pipeline(
        data, cov, mean, sperp_bins, spar_bins,
        ri, deltai, sigma_deltai,log_field=False,infer_profile_args=True,
        infer_velocity_args = False,
        tabulate_inverse=True,cholesky=True,sample_epsilon=True,
        filter_data=False,z=z,lambda_cut=1e-23,lambda_ref=1e-27,
        profile_param_ranges=theta_ranges_list[2:],
        vel_profile_param_ranges = [],om_ranges=[[0.1, 0.5]],
        eps_ranges=[theta_ranges_list[0]],f_ranges=[theta_ranges_list[1]],
        Om_fid=0.3111,
        filename=filename,autocorr_filename="autocorr.npy",
        disp=1e-2,nwalkers=16,n_mcmc=10,max_n=10,batch_size=10,
        nbatch=1,redo_chain=True,backup_start=True,F_inv = None,
        delta_profile=profile_modified_hamaus,
        Delta_profile=integrated_profile_modified_hamaus,
        vel_model = void_los_velocity_ratio_1lpt,
        dvel_dlogr_model = void_los_velocity_ratio_derivative_1lpt,
        vel_params_guess = None
    )
    tau_max = np.max(tau)
    flat_samples = sampler.get_chain()
    assert(len(tau) == 8)
    assert(len(tau) == flat_samples.shape[2])


# ---------------------- REGRESSION TESTS: -------------------------------------







