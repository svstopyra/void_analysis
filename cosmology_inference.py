# CONFIGURATION
from void_analysis import plot, tools, snapedit, catalogue
from void_analysis.catalogue import *
from void_analysis.paper_plots_borg_antihalos_generate_data import *
from void_analysis.real_clusters import getClusterSkyPositions
from void_analysis import massconstraintsplot
from void_analysis.simulation_tools import ngPerLBin
from void_analysis.simulation_tools import redshift_space_positions
from void_analysis.simulation_tools import get_los_pos_for_snapshot
from void_analysis.simulation_tools import get_los_positions_for_all_catalogues
from void_analysis.plot import draw_ellipse, plot_los_void_stack
from void_analysis import context
from matplotlib import transforms
import matplotlib.ticker
from matplotlib.ticker import NullFormatter
from matplotlib import cm
from matplotlib import patches
import matplotlib.lines as mlines
import matplotlib.colors as colors
import pickle
import numpy as np
import seaborn as sns
import pandas
seabornColormap = sns.color_palette("colorblind",as_cmap=True)
import pynbody
import astropy.units as u
from astropy.coordinates import SkyCoord
import scipy
import os
import sys

#-------------------------------------------------------------------------------
# SNAPSHOT GROUP CLASS

# Load properties file for a given snapshot. Allow for backwards compatibility
# with older pickle version of the file:
def get_antihalo_properties(snap,file_suffix = "AHproperties",
                            default=".h5",low_memory_mode=True):
    filename = snap.filename + "." + file_suffix + default
    # For backwards compatibility:
    filename_old = snap.filename + "." + file_suffix + ".p"
    if os.path.isfile(filename):
        return h5py.File(filename, "r")
    elif os.path.isfile(filename_old):
        if low_memory_mode:
            return filename_old
        else:
            return tools.loadPickle(filename_old)
    else:
        raise Exception("Anti-halo properties file not found.")


# Class to organise anti-halo properties into a more convenient format
class SnapshotGroup:
    def __init__(self,snap_list,snap_list_reverse,low_memory_mode=True,
                 swapXZ = False,reverse = False,remap_centres=False):
        self.snaps = [tools.getPynbodySnap(snap) for snap in snap_list]
        self.snaps_reverse = [tools.getPynbodySnap(snap) 
                              for snap in snap_list_reverse]
        self.N = len(self.snaps)
        self.low_memory_mode = low_memory_mode
        if low_memory_mode:
            self.all_property_lists = [None for snap in snap_list]
        else:
            self.all_property_lists = [get_antihalo_properties(snap) 
                                   for snap in snap_list]
        self.property_list = ["halo_centres","halo_masses",
                             "antihalo_centres","antihalo_masses",
                             "cell_volumes","void_centres",
                             "void_volumes","void_radii",
                             "radius_bins","pair_counts",
                             "bin_volumes","delta_central",
                             "delta_average"]
        self.additional_properties = {"halos":None,"antihalos":None,
                                      "snaps":self.snaps,
                                      "snaps_reverse":self.snaps_reverse}
        self.property_map = {name:ind for name, ind in 
            zip(self.property_list,range(0,len(self.property_list)))}
        self.reverse = reverse
        self.swapXZ = swapXZ
        self.remap_centres = True
        self.boxsize = self.snaps[0].properties['boxsize'].ratio("Mpc a h**-1")
        self.all_properties = [None for prop in self.property_list]
        self.snap_filenames = [snap.filename for snap in self.snaps]
        self.snap_reverse_filenames = [snap.filename 
                                       for snap in self.snaps_reverse]
    def is_valid_property(self,prop):
        if type(prop) is int:
            return prop in range(0,len(self.property_list))
        elif type(prop) is str:
            return prop in self.property_list
        else:
            return False
    # Map named property to a property index, and check it is valid:
    def get_property_index(self,prop):
        if type(prop) is int:
            if prop in range(0,len(self.property_list)):
                return prop
            else:
                raise Exception("Property index is out of range.")
        elif type(prop) is str:
            if prop in self.property_list:
                return self.property_map[prop]
            else:
                raise Exception("Requested property does not exist.")
        else:
            raise Exception("Invalid property type")
    # Map property index to a named property and check it is valid:
    def get_property_name(self,prop):
        if type(prop) is int:
            if prop in range(0,len(self.property_list)):
                return self.property_list[prop]
            else:
                raise Exception("Property index is out of range.")
        elif type(prop) is str:
            if prop in self.property_list:
                return prop
            else:
                raise Exception("Requested property does not exist.")
        else:
            raise Exception("Invalid property type")
    # Get the selected property:
    def get_property(self,snap_index,property_name,recompute=False):
        prop_index = self.get_property_index(property_name)
        if (self.all_properties[prop_index] is not None) and (not recompute):
            # Load the cachec properties if they exist:
            return self.all_properties[prop_index][snap_index]
        else:
            # Otherwise, recompute them:
            property_list = self.all_property_lists[snap_index]
            if property_list is None:
                property_list = get_antihalo_properties(
                                    self.snaps[snap_index])
            # Load from hdf5 file:
            if type(property_list) is h5py._hl.files.File:
                return property_list[self.get_property_name(property_name)]
            # Load from pickled list:
            elif type(property_list) is list:
                return property_list[self.get_property_index(property_name)]
            # Load pickle, get property, then dispose of pickle to avoid
            # keeping it in memory (used for low_memory_mode):
            elif type(property_list) is str:
                # Only load the file for as long as necessary:
                props_list = tools.loadPickle(property_list)
                return props_list[self.get_property_index(property_name)]
            else:
                raise Exception("Invalid Property Type")
    # check if a property is something that should be remapped to the correct
    # co-ordinates
    def check_remappable(self,property_name):
        index = self.get_property_index(property_name)
        return (index == 0) or (index == 5)
    # Remap the centres to the correct co-ordiantes:
    # Get a list of all properties:
    def get_all_properties(self,property_name,cache = True,recompute=False):
        prop_index = self.get_property_index(property_name)
        if self.all_properties[prop_index] is None:
            if self.check_remappable(property_name):
                properties = [tools.remapAntiHaloCentre(self.get_property(
                              ind,property_name,recompute=recompute),
                              boxsize,swapXZ  = self.swapXZ,
                              reverse = self.reverse)
                              for ind in range(0,self.N)]
            else:
                properties = [self.get_property(ind,property_name,
                              recompute=recompute) 
                              for ind in range(0,self.N)]
            if cache:
                self.all_properties[prop_index] = properties
            return properties
        else:
            return self.all_properties[prop_index]
    def __getitem__(self,property_name):
        if self.is_valid_property(property_name):
            return self.get_all_properties(property_name)
        else:
            if type(property_name) is str:
                if property_name in self.additional_properties:
                    if self.additional_properties[property_name] is not None:
                        return self.additional_properties[property_name]
                    else:
                        # Generate the property:
                        if property_name == "halos":
                            prop = self.additional_properties["halos"] = \
                                [snap.halos() for snap in self.snaps]
                        elif property_name == "antihalos":
                            prop = self.additional_properties["antihalos"] = \
                                [snap.halos() for snap in self.snaps_reverse]
                        else:
                            raise Exception("Invalid property_name")
                        return prop
            else:
                raise Exception("Invalid property_name")


#-------------------------------------------------------------------------------
# COSMOLOGY FUNCTIONS

# E(z)^2 function:
def Ez2(z,Om,Or=0,Ok=0,Ol=None,**kwargs):
    if Ol is None:
        Ol = 1.0 - Om - Ok - Or
    return Or*(1 + z)**4 + Om*(1 + z)**3 + Ok*(1 + z)**2 + Ol

# Linear growth rate in Lambda-CDM
def f_lcdm(z,Om,gamma=0.55,Ol=None,Ok=0,Or=0,**kwargs):
    Ez2_val = Ez2(z,Om,**kwargs)
    return (Om*(1 + z)**3/Ez2_val)**gamma

# Hubble rate as a function of H:
def Hz(z,Om,h=None,Ol=None,Ok=0,Or=0,**kwargs):
    if h is None:
        # Use units of km/s/Mpc/h:
        h = 1
    return 100*h*np.sqrt(Ez2(z,Om,**kwargs))

def ap_parameter(z,Om,Om_fid,h=0.7,h_fid = 0.7,**kwargs):
    # Get cosmology
    cosmo_fid = astropy.cosmology.FlatLambdaCDM(H0=100*h_fid,Om0=Om_fid)
    cosmo_test = astropy.cosmology.FlatLambdaCDM(H0=100*h,Om0=Om)
    # Get the ratio:
    Hz = 100*h*np.sqrt(Om*(1 + z)**3 + 1.0 - Om)
    Hzfid = 100*h_fid*np.sqrt(Om_fid*(1 + z)**3 + 1.0 - Om_fid)
    Da = cosmo_test.angular_diameter_distance(z).value
    Dafid = cosmo_fid.angular_diameter_distance(z).value
    eps = Hz*Da/(Hzfid*Dafid)
    return eps

# Peculiar velocity as a function of distance from a void centre along LOS, 
# at linear level:
def void_los_velocity(z,Delta,r_par,r_perp,Om,f=None,**kwargs):
    # Assume lambda-cdm if no growth rate given:
    if f is None:
        f = f_lcdm(z,Om,**kwargs)
    # Hubble rate:
    hz = Hz(z,Om,**kwargs)
    # Distance from void centre:
    r = np.sqrt(r_par**2 + r_perp**2)
    # Cumulative density at this distance:
    Dr = Delta(r)
    return -(f/3.0)*(hz/(1.0 + z))*Dr*r_par

# Derivative of the peculiar velocity with LOS distance
def void_los_velocity_derivative(z,Delta,delta,r_par,r_perp,Om,f=None,**kwargs):
    # Distance from void centre:
    r = np.sqrt(r_par**2 + r_perp**2)
    # Cumulative density at this distance:
    Dr = Delta(r)
    # Shell density at this distance:
    dr = delta(r)
    # Hubble rate:
    hz = Hz(z,Om,**kwargs)
    # Assume lambda-cdm if no growth rate given:
    if f is None:
        f = f_lcdm(z,Om,**kwargs)
    return -(f/3.0)*(hz/(1.0 + z))*Dr - \
        f*(r_par/(r + 1e-12))**2*(hz/(1.0 + z))*(dr - Dr)

# Derivative of LOS distance in real space vs redshift space:
def z_space_jacobian(z,Delta,delta,r_par,r_perp,Om,
                     linearise_jacobian=True,**kwargs):
    # Hubble rate:
    hz = Hz(z,Om,**kwargs)
    dudr = void_los_velocity_derivative(z,Delta,delta,r_par,r_perp,Om,**kwargs)
    if linearise_jacobian:
        # Expand to first order:
        return 1.0 -((1.0 + z)/hz)*dudr
    else:
        # Just compute without expanding:
        return 1.0/(1.0 + ((1.0 + z)/hz)*dudr)

# Redshift space transformation around voids:
def to_z_space(r_par,r_perp,z,Om,Delta=None,u_par=None,f=None,**kwargs):
    if u_par is None:
        # Assume linear relationship:
        r = np.sqrt(r_par**2 + r_perp**2)
        if f is None:
            f = f_lcdm(z,om,**kwargs)
        s_par = (1.0  - (f/3.0)*Delta(r))*r_par
    else:
        # Use supplied peculiar velocity:
        # Hubble rate:
        hz = Hz(z,Om,**kwargs)
        s_par = r_par + (1.0 + z)*u_par/hz
    s_perp = r_perp
    return [s_par,s_perp]

def iterative_zspace_inverse(s_par,f,Delta,N_max):
    r_par_guess = s_par
    for k in range(0,N_max):
        # Iteratively improve the guess:
        r = np.sqrt((r_par_guess)**2 + r_perp**2)
        r_par_new = s_par/(1.0  - (f/3.0)*Delta(r))
        if (np.abs(r_par_new - r_par_guess) < atol) or \
            (np.abs(r_par_new/r_par_guess - 1.0) < rtol):
            break
        r_par_guess = r_par_new
    r_par = r_par_guess
    return r_par

# Transformation to real space, from redshift space, including geometric
# distortions from a wrong-cosmology:
def to_real_space(s_par,s_perp,z,Om,Om_fid=None,Delta=None,u_par=None,f=None,
                  N_max = 5,atol=1e-5,rtol=1e-5,F_inv=None,
                  **kwargs):
    # Perpendicular distance is easy:
    r_perp = s_perp
    # Get the parallel distance:
    if u_par is None:
        # Assume linear relationship:
        if f is None:
            f = f_lcdm(z,Om,**kwargs)
        # Need to guess at r:
        if F_inv is None:
            # Manually invert:
            if not np.isscalar(s_par):
                raise Exception("Must be a scalar to perform manual inversion")
            r_par_guess = s_par
            for k in range(0,N_max):
                # Iteratively improve the guess:
                r = np.sqrt((r_par_guess)**2 + r_perp**2)
                r_par_new = s_par/(1.0  - (f/3.0)*Delta(r))
                if (np.abs(r_par_new - r_par_guess) < atol) or \
                    (np.abs(r_par_new/r_par_guess - 1.0) < rtol):
                    break
                r_par_guess = r_par_new
            r_par = r_par_guess
        else:
            # Use the tabulated inverse:
            r_par = F_inv(s_perp,s_par,f*np.ones(s_perp.shape))
    else:
        # Use supplied peculiar velocity:
        # Hubble rate:
        hz = Hz(z,Om,**kwargs)
        r_par = (s_par - (1.0 + z)*u_par/hz)
    return [r_par,r_perp]

# Correct the geometry to account for miss-specified cosmology:
def geometry_correction(s_par,s_perp,epsilon,**kwargs):
    if epsilon is None:
        epsilon = 1.0
    s = np.sqrt(s_par**2 + s_perp**2)
    mus = s_par/s
    # Distance correction, accounting for change in void size:
    s_factor = np.sqrt(1.0 + epsilon**2*(1.0/mus**2 - 1.0))
    s_new = s*mus*epsilon**(-2.0/3.0)*s_factor
    # Angle correction:
    mus_new = np.sign(mus)/s_factor
    # New co-ordinates:
    s_par_new = mus_new*s_new
    s_perp_new = np.sign(s_perp)*s_new*np.sqrt(1.0 - mus_new**2)
    return s_par_new, s_perp_new

# Profile in redshift space:
def z_space_profile(s_par,s_perp,rho_real,z,Om,Delta,delta,Om_fid = 0.3111,
                    epsilon=None,apply_geometry=False,**kwargs):
    # Apply geometric correction:
    if apply_geometry:
        if epsilon is None:
            epsilon = ap_parameter(z,Om,Om_fid,**kwargs)
        s_par_new, s_perp_new = geometry_correction(s_par,s_perp,epsilon)
    else:
        s_par_new = s_par
        s_perp_new = s_perp
    # Get real-space co-ordinates:
    [r_par,r_perp] = to_real_space(s_par_new,s_perp_new,z,Om,Delta=Delta,
                                   **kwargs)
    # Get Jacobian from the transformation:
    jacobian = z_space_jacobian(z,Delta,delta,r_par,r_perp,Om,**kwargs)
    # Compute distance from centre of the void:
    r = np.sqrt(r_par**2 + r_perp**2)
    # Evaluate profile:
    return rho_real(r)*jacobian



#-------------------------------------------------------------------------------
# LIKELIHOOD AND POSTERIOR COMPUTATION


# Compute the covariance matrix of a set of data:
def covariance(data):
    # data should be an (N,M) array of N data elements, each with M values.
    # We seek to compute an M x M covariance matrix of this data
    N, M = data.shape
    mean = np.mean(data,0)
    diff = data.T - mean
    cov = np.matmul(diff,diff.T)/N
    return cov

# Estimate covariance of the profile using the jackknife method:
def profile_jackknife_covariance(data,profile_function,**kwargs):
    N, M = data.shape
    jacknife_data = np.zeros(N,M)
    for k in range(0,N):
        sample = np.setdiff1d(range(0,N),np.array([k]))
        jacknife_data[k,:] = profile_function(data[sample,:],**kwargs)
    return covariance(jacknife_data)/(N-1)


def compute_singular_log_likelihood(x,Umap,good_eig):
    u = np.matmul(Umap,x)
    Du = u/good_eig
    uDu = np.sum(u*Du)
    N = len(good_eig)
    return -0.5*uDu - (N/2)*np.log(2*np.pi) - 0.5*np.sum(np.log(good_eig))



# Likelihood function:
def log_likelihood_aptest(theta,data_field,scoords,inv_cov,
                          z,Delta,delta,rho_real,data_filter=None,
                          cholesky=False,normalised=False,tabulate_inverse=True,
                          ntab = 10,sample_epsilon=False,Om_fid=None,
                          singular=False,Umap=None,good_eig=None,
                          F_inv=None,log_density=False,infer_profile_args=False,
                          **kwargs):
    s_par = scoords[:,0]
    s_perp = scoords[:,1]
    if sample_epsilon:
        epsilon, f, *profile_args = theta
        if Om_fid is not None:
            Om = Om_fid
        else:
            Om = 0.3
    else:
        Om, f , *profile_args = theta
        if Om_fid is not None:
            epsilon = ap_parameter(z,Om,Om_fid,**kwargs)
        else:
            epsilon = 1.0
    M = len(s_par)
    delta_rho = np.zeros(s_par.shape)
    # Apply geometric correction to account for miss-specified cosmology:
    s_par_new, s_perp_new = geometry_correction(s_par,s_perp,epsilon)
    # Wrappers if we are inferring profile arguments:
    if infer_profile_args:
        Delta_func = lambda r: Delta(r,*profile_args)
        delta_func = lambda r: delta(r,*profile_args)
    else:
        # Fixed profile:
        Delta_func = Delta
        delta_func = delta
    # Evaluate the profile for the supplied value of the parameters:
    if tabulate_inverse:
        # Tabulate an inverse function and then evaluate an interpolated
        # inverse, rather than repeatedly inverting:
        data_val = data_field
        if F_inv is None:
            svals = np.linspace(np.min(s_par_new),np.max(s_par_new),ntab)
            rperp_vals = np.linspace(np.min(s_perp_new),np.max(s_perp_new),ntab)
            rvals = np.zeros((ntab,ntab))
            for i in range(0,ntab):
                for j in range(0,ntab):
                    F = (lambda r: r - r*(f/3)*\
                        Delta_func(np.sqrt(r**2 + rperp_vals[i]**2)) \
                        - svals[j])
                    rvals[i,j] = scipy.optimize.fsolve(F,svals[j])
            F_inv = lambda x, y: scipy.interpolate.interpn((rperp_vals,svals),
                                                           rvals,
                                                           np.vstack((x,y)).T,
                                                           method='cubic')
            theory_val = z_space_profile(s_par_new,s_perp_new,
                                         lambda r: rho_real(r,*profile_args),
                                         z,Om,Delta_func,delta_func,f=f,
                                         F_inv=F_inv,**kwargs)
        else:
            theory_val = z_space_profile(s_par_new,s_perp_new,
                                         lambda r: rho_real(r,*profile_args),
                                         z,Om,Delta_func,delta_func,f=f,
                                         F_inv=lambda x, y, z: F_inv(x,y,z),
                                         **kwargs)
        if log_density:
            # Examine the log density. Assumes that the user has already 
            # provided log-data and a relevant estimated covariance matrix
            theory_val = np.log(theory_val)
        if normalised:
            delta_rho = 1.0 - theory_val/data_val
        else:
            delta_rho = data_val - theory_val
    else:
        for k in range(0,M):
            data_val = data_field[k]
            theory_val = z_space_profile(s_par_new[k],s_perp_new[k],
                                         lambda r: rho_real(r,A,r0,c1,f1,B),
                                         z,Om,Delta_func,delta_func,f=f,
                                         **kwargs)
            if normalised:
                delta_rho[k] = 1.0 - theory_val/data_val
            else:
                delta_rho[k] = data_val - theory_val
    if cholesky:
        # We assume that the covariance is given in it's lower triangular form,
        # rather than an explicit covariance. We then solve this rather than
        # actually computing 
        x = scipy.linalg.solve_triangular(inv_cov,delta_rho,lower=True)
        #return -0.5*np.sum(x**2)  - (M/2)*np.log(2*np.pi) - \
        #     np.sum(np.log(np.diag(inv_cov)))
        return -0.5*np.sum(x**2)
    elif singular:
        if (Umap is None) or (good_eig is None):
            raise Exception("Must provide Umap and good_eigenvalues for " + 
                "handling singular covariance matrices")
        return compute_singular_log_likelihood(delta_rho,Umap,good_eig)
    else:
        return -0.5*np.matmul(np.matmul(delta_rho,inv_cov),delta_rho.T)

# Likelihood function, parallelised. Requires global variables!:
# UNUSED
def log_likelihood_aptest_parallel(theta,z,**kwargs):
    s_par = scoords[:,0]
    s_perp = scoords[:,1]
    Om, f , A = theta
    M = len(s_par)
    delta_rho = np.zeros(s_par.shape)
    # Evaluate the profile for the supplied value of the parameters:
    for k in range(0,M):
        delta_rho[k] = data_field[k] - \
            z_space_profile(s_par[k],s_perp[k],lambda r: rho_real(r,A),
                            z,Om,Delta_func,delta_func,f=f,**kwargs)
    return -0.5*np.matmul(np.matmul(delta_rho,inv_cov),delta_rho.T)


# Un-normalised log prior:

def log_flat_prior_single(theta,theta_range):
    in_range = (theta_range[0] <= theta <= theta_range[1])
    if in_range:
        return 0.0
    else:
        return -np.inf

def log_flat_prior(theta,theta_ranges):
    bool_array = np.aray([bounds[0] <= param <= bounds[1] 
        for bounds, param in zip(theta_ranges,theta)],dtype=bool)
    if np.all(bool_array):
        return 0.0
    else:
        return -np.inf

# Prior (assuming flat prior for now):
def log_prior_aptest(theta,
                     theta_ranges=[[0.1,0.5],[0,1.0],[-np.inf,np.inf],[0,2],
                                   [-np.inf,0],[0,np.inf],[-1,1]],
                    **kwargs):
    log_prior_array = np.zeros(theta.shape)
    flat_priors = [0,1,2,3,4,5,6]
    theta_flat = [theta[k] for k in flat_priors]
    theta_ranges_flat = [theta_ranges[k] for k in flat_priors]
    for k in flat_priors:
        log_prior_array[k] = log_flat_prior_single(theta[k],theta_ranges[k])
    # Amplitude prior (Jeffries):
    #log_prior_array[2] = -np.log(theta[2])
    return np.sum(log_prior_array)

# Posterior (unnormalised):
def log_probability_aptest(theta,*args,**kwargs):
    lp = log_prior_aptest(theta,**kwargs)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_aptest(theta,*args,**kwargs)

# UNUSED
def log_probability_aptest_parallel(theta,*args,**kwargs):
    lp = log_prior_aptest(theta,**kwargs)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_aptest_parallel(theta,*args,**kwargs)



#-------------------------------------------------------------------------------
# GAUSSIANITY TESTING

def tikhonov_regularisation(mat,lambda_reg=1e-10):
    return mat + lambda_reg*np.identity(mat.shape[0])

def regularise_covariance(cov,lambda_reg=1e-10):
    symmetric_cov = (cov + cov.T)/2
    regularised_cov = tikhonov_regularisation(symmetric_cov,
                                              lambda_reg=lambda_reg)
    return regularised_cov

def get_inverse_covariance(cov,lambda_reg=1e-10):
    regularised_cov = regularise_covariance(cov,lambda_reg=lambda_reg)
    L = np.linalg.cholesky(regularised_cov)
    P = np.linalg.inv(L)
    inv_cov = np.matmul(P,P.T)
    return inv_cov


def range_excluding(kmin,kmax,exclude):
    return np.setdiff1d(range(kmin,kmax),exclude)


def get_nonsingular_subspace(C,lambda_reg,lambda_cut = None,
                             normalised_cov=False,mu=None):
    reg_cov = regularise_covariance(C,lambda_reg= lambda_reg)
    if normalised_cov:
        if mu is None:
            raise Exception("Mean not provided.")
        else:
            norm_cov = C/np.outer(mu,mu)
            norm_reg_cov = regularise_covariance(norm_cov,
                                                 lambda_reg= lambda_reg)
            eig, U = scipy.linalg.eigh(norm_reg_cov)
    else:
        eig, U = scipy.linalg.eigh(reg_cov)
    if lambda_cut is None:
        lambda_cut = 10*lambda_reg
    bad_eig = np.where(eig < lambda_cut)[0]
    good_eig = np.where(eig >= lambda_cut)[0]
    #D = np.diag(eig[good_eig])
    Umap = (U.T)[good_eig,:]
    #Ctilde = np.matmul(Umap.T,np.matmul(D,Umap))
    #Ctilde = (Ctilde.T + Ctilde)/2
    return Umap, eig[good_eig]


def get_solved_residuals(samples,covariance,xbar,singular=False,
                         normalised_cov=False,L=None,lambda_cut=1e-23,
                         lambda_reg=1e-27,Umap=None,good_eig=None):
    if not singular:
        if normalised_cov:
                residual = samples/xbar[:,None] - 1.0
        else:
            residual = samples - xbar[:,None]
        if L is None:
            reg_cov = regularise_covariance(covariance,lambda_reg= lambda_reg)
            L = scipy.linalg.cholesky(reg_cov,lower=True)
        solved_residuals = np.array(
            [scipy.linalg.solve_triangular(L,residual[:,i],lower=True) 
            for i in tools.progressbar(range(0,n))]).T
    else:
        if (Umap is None) or (good_eig is None):
            Umap, good_eig = get_nonsingular_subspace(covariance,lambda_reg,
                                                      lambda_cut=lambda_cut,
                                                      normalised_cov = False,
                                                      mu=xbar)
        if normalised_cov:
            residual = np.matmul(Umap,samples/xbar[:,None] - 1.0)
        else:
            residual = np.matmul(Umap,samples - xbar[:,None])
        solved_residuals = residual/np.sqrt(good_eig[:,None])
    return solved_residuals

def compute_normality_test_statistics(samples,covariance=None,xbar=None,
                                      solved_residuals=None,
                                      low_memory_sum = False,**kwargs):
    n = samples.shape[1]
    k = samples.shape[0]
    if covariance is None:
        covariance = np.cov(samples)
    if xbar is None:
        xbar = np.mean(samples,1)
    if solved_residuals is None:
        solved_residuals = get_solved_residuals(samples, covariance,xbar,
                                                **kwargs)
    # Compute Skewness:
    if low_memory_sum:
        Ai = np.array(
            [np.sum(
            np.sum((solved_residuals[:,i][:,None]*solved_residuals),0)**3)
            for i in tools.progressbar(range(0,n))])
        A = np.sum(Ai)/(6*n)
    else:
        product = np.matmul(solved_residuals.T,solved_residuals)
        A = np.sum(product**3)/(6*n)
    B = np.sqrt(n/(8*k*(k+2)))*(np.sum(np.sum(solved_residuals**2,0)**2)/n \
                            - k*(k+2))
    return [A,B]

#-------------------------------------------------------------------------------
# STACKED DENSITY FIELDS

def get_zspace_centres(halo_indices,snap_list,snap_list_rev,hrlist=None,
                       recompute_zspace=False):
    if len(halo_indices) != len(snap_list):
        raise Exception("halo_indices list does not match snapshot list.")
    num_samples = len(halo_indices)
    centres = [np.ones((len(x),3))*np.nan for x in halo_indices]
    for ns in range(0,num_samples):
        snap = snap_list[ns]
        if os.path.isfile(snap.filename + ".snapsort.p"):
            sorted_indices = tools.loadPickle(snap.filename + ".snapsort.p")
        else:
            sorted_indices = np.argsort(snap['iord'])
        if hrlist is None:
            halos = snap_list_rev[ns].halos()
        else:
            halos = hrlist[ns]
        boxsize = snap.properties['boxsize'].ratio("Mpc a h**-1")
        positions = tools.loadOrRecompute(
                snap.filename + ".z_space_pos.p",
                simulation_tools.redshift_space_positions,
                snap,centre=np.array([boxsize/2]*3),
                _recomputeData=recompute_zspace)
        for k in tools.progressbar(range(0,len(halo_indices[ns]))):
            if halo_indices[ns][k] >= 0:
                indices = halos[halo_indices[ns][k]+1]['iord']
                centres[ns][k,:] = tools.remapAntiHaloCentre(
                    context.computePeriodicCentreWeighted(
                    positions[sorted_indices[indices],:],periodicity=boxsize),
                    boxsize,swapXZ  = False,reverse = True)
    return centres

# Function to combine lists of void los-coords in the same simulation into 
# a single list.
# UNUSED - possibly deprecate
def combine_los_lists(los_lists):
    lengths = np.array([len(x) for x in los_lists],dtype=int)
    if not np.all(lengths == lengths[0]):
        raise Exception("Cannot combine los lists with different lengths.")
    new_list = []
    num_parts = np.vstack([np.array([len(x) for x in los_list],dtype=int) 
        for los_list in los_lists]).T
    to_use = -np.ones(lengths[0],dtype=int)
    num_lists = len(los_lists)
    for k in range(0,num_lists):
        to_use[num_parts[:,k] > 0] = k
    for k in range(0,lengths[0]):
        if to_use[k] >= 0:
            new_list.append(los_lists[to_use[k]][k])
        else:
            new_list.append(np.zeros((0,2)))
    return new_list

# 2D void stacks:
def get_2d_void_stack_from_los_pos(los_pos,z_bins,d_bins,radii,stacked=True):
    voids_used = [np.array([len(x) for x in los]) > 0 for los in los_pos]
    # Filter out any unused voids as they just cause problems:
    los_pos_filtered = [ [x for x in los if len(x) > 0] for los in los_pos]
    # Cell volumes and void radii:
    cell_volumes_reff = np.outer(np.diff(z_bins),np.diff(d_bins))
    void_radii = [rad[filt] for rad, filt in zip(radii,voids_used)]
    # LOS positions in units of Reff:
    los_list_reff = [
        [np.abs(los/rad) for los, rad in zip(all_los,all_radii)] 
        for all_los, all_radii in zip(los_pos_filtered,void_radii)]
    # Stacked particles:
    if stacked:
        stacked_particles_reff = np.vstack([np.vstack(los_list) 
            for los_list in los_list_reff ])
        return stacked_particles_reff
    else:
        return los_list_reff


# Compute_volume weights:
def get_weights_for_stack(los_pos,void_radii,additional_weights = None,
                          stacked = True):
    if additional_weights is None:
        v_weight = [[(1.0/rad)**3*np.ones(len(los)) 
                    for los, rad in zip(all_los,all_radii)] 
                    for all_los, all_radii in zip(los_pos,void_radii)]
    else:
        if type(additional_weights) == list:
            v_weight = [[(1.0/rad)**3*np.ones(len(los))*weight 
                    for los, rad, weight in zip(all_los,all_radii,all_weights)] 
                    for all_los, all_radii, all_weights,
                    in zip(los_pos,void_radii,additional_weights)]
        else:
            v_weight = [[(1.0/rad)**3*np.ones(len(los))*weight 
                    for los, rad, weight 
                    in zip(all_los,all_radii,additional_weights)]
                    for all_los, all_radii
                    in zip(los_pos,void_radii)]
    if stacked:
        return np.hstack([np.hstack(rad) for rad in v_weight])
    else:
        return v_weight


# Put all particles into a single pile and then bin them:
def get_field_from_los_data(los_data,z_bins,d_bins,v_weight,void_count,
                            nbar=None):
    cell_volumes_reff = np.outer(np.diff(z_bins),np.diff(d_bins))
    hist = np.histogramdd(los_data,bins=[z_bins,d_bins],density=False,
                          weights = v_weight/(2*np.pi*los_data[:,1]))
    if nbar is not None:
        return hist[0]/(2*void_count*cell_volumes_reff*nbar)
    else:
        return hist[0]/(2*void_count*cell_volumes_reff)


def get_2d_fields_per_void(los_per_void,sperp_bins,spar_bins,
                           void_radii,nbar=None):
    cell_volumes_reff = np.outer(np.diff(spar_bins),np.diff(sperp_bins))
    hist = np.array([np.histogramdd(los,bins=[sperp_bins,spar_bins],
                                     density=False,
                                     weights = 1.0/(2*np.pi*los[:,1]))[0]
                      for los in los_per_void])
    volume_weight = (1/void_radii**3)
    if nbar is not None:
        denominator = (2*cell_volumes_reff*nbar)
    else:
        denominator = (2*cell_volumes_reff)
    density = volume_weight[:,None,None]*hist/denominator[None,:,:]
    return density

# Compute profile for each void individually, and then average them:
def get_2d_field_from_stacked_voids(los_per_void,sperp_bins,spar_bins,
                                    void_radii,weights=None,nbar=None):
    density = get_2d_fields_per_void(los_per_void,sperp_bins,spar_bins,
                                     void_radii,nbar=nbar)
    return np.average(density, axis=0,weights = weights)

def profile_broken_power_log(r,A,r0,c1,f1,B):
    return np.log(np.abs(A + B*(r/r0)**2 + (r/r0)**4)) + \
        ((c1 - 4)/f1)*np.log(1 + (r/r0)**f1)

def profile_broken_power(r,A,r0,c1,f1,B):
    return np.exp(profile_broken_power_log(r,A,r0,c1,f1,B))

# Modified Hamaus profile:
def profile_modified_hamaus(r,alpha,beta,rs,delta_c,delta_large = 0.0,rv=1.0):
    return (delta_c - delta_large)*(1.0 - (r/rs)**alpha)/(1 + (r/rv)**beta) \
        + delta_large

def integrated_profile_modified_hamaus(r,alpha,beta,rs,delta_c,
                                       delta_large = 0.0,rv=1.0):
    arg = ((r/rv)**beta)/(1 + (r/rv)**beta)
    hyp_1 = scipy.special.hyp2f1(3/beta,3/beta,1 + 3/beta,arg)
    hyp_2 = scipy.special.hyp2f1((alpha+3)/beta,
                                 (alpha+3)/beta,1 + (alpha+3)/beta,arg)
    return (delta_c - delta_large)*(
         ((1.0 + (r/rv)**beta)**(-3/beta))*hyp_1 - 
         (3/(alpha+3))*((r/rs)**alpha)*
         ((1 + (r/rv)**beta)**(-(alpha+3)/beta))*hyp_2) + delta_large


def rho_real(r,*profile_args):
    #return profile_broken_power(r,A,r0,c1,f1,B)
    return profile_modified_hamaus(r,*profile_args)



# Compute weights for each void contributing to the stack:
def get_weights(los_zspace,void_radii,additional_weights = None):
    voids_used = [np.array([len(x) for x in los]) > 0 
        for los in los_zspace]
    los_pos = [ [los[x] for x in np.where(ind)[0]] 
        for los, ind in zip(los_zspace,voids_used) ]
    if additional_weights is None:
        weights_list = None
    else:
        all_additional_weights = np.hstack([weights[used] 
                                       for weights, used in 
                                       zip(additional_weights,voids_used)])
        weights_list = [weights[used]/np.sum(all_additional_weights) 
            for weights, used in zip(additional_weights,voids_used)]
    v_weight = get_weights_for_stack(
        los_pos,[radii[used] for radii, used in zip(void_radii,voids_used)],
        additional_weights = weights_list)
    return v_weight


# This is actually not needed, because we can already do this in the 
# catalogue class!
def get_halo_indices(catalogue):
    final_cat = catalogue.get_final_catalogue(void_filter=True)
    halo_indices = [-np.ones(len(final_cat),dtype=int) 
                    for ns in range(0,snaps.N)]
    for ns in range(0,borg_snaps.N):
        have_void = final_cat[:,ns] >= 0
        halo_indices[ns][have_void] = \
            catalogue.indexListShort[ns][final_cat[have_void,ns]-1]
    return halo_indices


#void_radii = catalogue.getMeanProperty("radii",void_filter=True)[0]
#rep_scores = catalogue.property_with_filter(
#    catalogue.finalCatFrac,void_filter=True)

# Remove voids that don't contribute to the stacked field in any way, returning
# a list of LOS positions, and boolean arrays flagging which voids were kept:
def trim_los_list(los_list,spar_bins,sperp_bins,all_radii):
    los_list_trimmed = get_2d_void_stack_from_los_pos(
        los_list,spar_bins,sperp_bins,
        [all_radii[ns] for ns in range(0,len(los_list))],stacked=False)
    voids_used = [np.array([len(x) for x in los]) > 0 
        for los in los_list]
    return los_list_trimmed, voids_used

def get_trimmed_los_list_per_void(los_pos,spar_bins,sperp_bins,void_radii_list):
    los_list_trimmed, voids_used = trim_los_list(los_pos,rbins,rbins,
                                                 void_radii_lists)
    return sum(los_list_trimmed,[])

def get_borg_density_estimate(snaps,densities_file = None,dist_max=135,
                              seed = 1000,interval=0.68):
    boxsize = snaps.boxsize
    if np.min(snaps["snaps"][0]["pos"]) < 0:
        centre = np.array([0,0,0])
    else:
        centre = np.array([boxsize/2]*3)
    if densities_file is not None:
        deltaMCMCList = tools.loadPickle(densities_file)
    else:
        deltaMCMCList = np.array([simulation_tools.density_from_snapshot(
                                     snap,centre,dist_max)
                                  for snap in snaps["snaps"]])
    deltaMAPBootstrap = scipy.stats.bootstrap(
        (deltaMCMCList,),simulation_tools.get_map_from_sample,
        confidence_level = interval,vectorized=False,random_state=seed)
    deltaMAPInterval = deltaMAPBootstrap.confidence_interval
    return deltaMAPBootstrap, deltaMAPInterval

def get_lcdm_void_catalogue(snaps,delta_interval,dist_max=135,
                            radii_range=[10,20],centres_file = None,
                            nRandCentres = 10000,seed=1000,flattened=True):
    boxsize = snaps.boxsize
    # Get random centres (usually pre-computed, but we can create them
    # from scratch if necessary):
    if centres_file is not None:
        [randCentres,randOverDen] = tools.loadPickle(centres_file)
    else:
        [randCentres,randOverDen] = \
            simulation_tools.get_random_centres_and_densities(
                dist_max,snaps["snaps"],seed=seed,nRandCentres = nRandCentres)
    # Select regions with similar density contrast the local supervolume:
    comparableDensityMAP = [(delta <= deltaMAPInterval[1]) & \
        (delta > deltaMAPInterval[0]) for delta in randOverDen]
    centresToUse = [randCentres[comp] for comp in comparableDensityMAP]
    # Get non-overlapping sphere:
    rSep = 2*dist_max
    indicesUnderdenseNonOverlapping = simulation_tools.getNonOverlappingCentres(
        centresToUse,rSep,boxsize,returnIndices=True)
    centresUnderdenseNonOverlapping = [centres[ind] \
        for centres,ind in zip(centresToUse,indicesUnderdenseNonOverlapping)]
    densityListUnderdenseNonOverlapping = [density[ind] \
        for density, ind in zip(comparableDensityMAP,\
        indicesUnderdenseNonOverlapping)]
    densityUnderdenseNonOverlapping = np.hstack(
        densityListUnderdenseNonOverlapping)
    # Get the centres of the voids from their respective regions:
    distances_from_centre_lcdm_selected = [[
        np.sqrt(np.sum(snapedit.unwrap(centres - sphere_centre,boxsize)**2,1))
        for sphere_centre in selected_regions]
        for centres, selected_regions in zip(snaps["void_centres"],
        centresUnderdenseNonOverlapping)]
    # Filter voids to the radius range used in the catalogue:
    filter_list_lcdm_by_region = [[
        (dist < dist_max) & (radii > radii_range[0]) & (radii <= radii_range[1]) 
        for dist in all_dists]
        for all_dists, radii in 
        zip(distances_from_centre_lcdm_selected,snaps["void_radii"])]
    if flattened:
        # Combine regions in the same simulation:
        voids_used_lcdm = [simulation_tools.flatten_filter_list(filt_list) 
                          for filt_list in filter_list_lcdm_by_region]
    else:
        voids_used_lcdm = filter_list_lcdm_by_region
    return voids_used_lcdm



def get_stacked_void_density_field(snaps,void_radii_lists,void_centre_lists,
                                   bins_spar,bins_sperp,halo_indices=None,
                                   filter_list=None,additional_weights=None,
                                   dist_max=3,rmin=10,rmax=20,recompute=False,
                                   zspace=True,recompute_zspace=False,
                                   suffix=".lospos_all_zspace2.p",
                                   los_pos=None,**kwargs):
    boxsize = snaps.boxsize
    nbar = len(snaps["snaps"][0])/boxsize**3
    # Filter out any invalid halo indices (usually only occurs for
    # BORG catalogues, where anti-halos are missing in some samples):
    if halo_indices is not None:
        if filter_list is None:
            filter_list = [halo_indices[ns] >= 0 for ns in range(0,snaps.N)]
    # Get LOS positions:
    if los_pos is None:
        los_pos = get_los_positions_for_all_catalogues(
            snaps["snaps"],snaps["snaps_reverse"],void_centre_lists,
            void_radii_lists,all_particles=True,void_indices = halo_indices,
            filter_list=filter_list,dist_max=dist_max,rmin=rmin,rmax=rmax,
            recompute=recompute,zspace=zspace,recompute_zspace=recompute_zspace,
            suffix=suffix)
    # Trimmed los lists:
    los_list_trimmed, voids_used = trim_los_list(los_pos,bins_spar,bins_sperp,
                                                 void_radii_lists)
    los_list_per_void = sum(los_list_trimmed,[])
    # Number of voids:
    num_voids = np.sum([np.sum(x) for x in voids_used])
    # Radii of each void:
    void_radii_per_void = np.hstack([rad[used] 
        for rad, used in zip(void_radii_lists,voids_used)])
    # Weights for each void:
    if additional_weights is not None:
        additional_weights_per_void = np.hstack([weights[used] 
            for weights, used in zip(additional_weights,voids_used)])
    else:
        additional_weights_per_void = np.ones(void_radii_per_void.shape)
    # Compute stacked denstiy field:
    return get_2d_field_from_stacked_voids(los_list_per_void,sperp_bins,
                                           spar_bins,void_radii_per_void,
                                           weights=additional_weights_per_void,
                                           nbar=nbar)

def get_1d_real_space_field(snaps,void_radii_lists,void_centre_lists,rbins,
                            filter_list=None,additional_weights=None,
                            dist_max=3,rmin=10,rmax=20,suffix=".lospos_all.p",
                            los_pos=None,recompute=False,nbar=None,
                            n_boot = 10000,seed = 42):
    boxsize = snaps.boxsize
    nbar = len(snaps["snaps"][0])/boxsize**3
    # Filter out any invalid halo indices (usually only occurs for
    # BORG catalogues, where anti-halos are missing in some samples):
    if halo_indices is not None:
        if filter_list is None:
            filter_list = [halo_indices[ns] >= 0 for ns in range(0,snaps.N)]
    # Get LOS positions:
    if los_pos is None:
        los_pos = get_los_positions_for_all_catalogues(
            snaps["snaps"],snaps["snaps_reverse"],void_centre_lists,
            void_radii_lists,all_particles=True,void_indices = halo_indices,
            filter_list=filter_list,dist_max=dist_max,rmin=rmin,rmax=rmax,
            recompute=recompute,zspace=False,suffix=suffix)
    los_list_trimmed, voids_used = trim_los_list(los_pos,rbins,rbins,
                                                 void_radii_lists)
    r_list = [[np.sqrt(np.sum(x**2,1)) for x in y] for y in los_list_trimmed]
    r_list_all = sum(r_list,[])
    # Weights:
    los_list_per_void = sum(los_list_trimmed,[])
    void_radii_per_void = np.hstack([rad[used] 
        for rad, used in zip(void_radii_lists,voids_used)])
    if additional_weights is not None:
        additional_weights_per_void = np.hstack([weights[used] 
            for weights, used in zip(additional_weights,voids_used)])
    else:
        additional_weights_per_void = np.ones(void_radii_per_void.shape)
    v_weights_per_void = (np.ones(void_radii_per_void.shape)/
        (void_radii_per_void**3))
    # Binned density:
    cell_volumes = 4*np.pi*(rbins[1:]**3 - rbins[0:-1]**3)/3
    hist = np.vstack([np.histogram(rad,bins=rbins,density=False)[0] 
                      for rad in r_list_all])
    num_voids = np.sum([np.sum(x) for x in voids_used])
    density = hist*v_weights_per_void[:,None]/(cell_volumes*nbar)
    # Bootstrap to estimate the density profile  and it's uncertainty:
    np.random.seed(seed)
    bootstrap_samples = np.random.choice(num_voids,size=(num_voids,n_boot))
    bootstrap_profiles = np.array([np.average(
        density[bootstrap_samples[:,k],:],
        axis=0,weights=additional_weights_per_void[bootstrap_samples[:,k]]) 
        for k in tools.progressbar(range(0,n_boot))]).T
    rho_mean = np.mean(bootstrap_profiles,1)
    rho_std = np.std(bootstrap_profiles,1)
    return rho_mean, rho_std

# Get additional weighting factor for BORG voids based on their
# reproducibility score:
def get_additional_weights_borg(cat,voids_used=None):
    rep_scores = cat.property_with_filter(cat.finalCatFrac,void_filter=True)
    if voids_used is None:
        voids_used = [np.ones(rep_scores.shape[0],dtype=bool) 
                         for x in range(0,cat.numCats)]
    all_rep_scores = np.hstack([rep_scores[used] for used in voids_used])
    all_void_radii_borg = cat.getAllProperties("radii",void_filter=True).T
    num_voids = np.sum([np.sum(x) for x in voids_used])
    return [rep_scores[used]/np.sum(rep_scores) for used in voids_used]

#-------------------------------------------------------------------------------
# COVARIANCE MATRIX CALCULATION

# Compute final weighting for each void:
def get_void_weights(los_list_trimmed,voids_used,all_radii,
                     additional_weights=None):
    return get_weights_for_stack(los_list_trimmed,
        [rad[used] for used, rad in zip(voids_used,all_radii)],
        additional_weights = additional_weights,stacked=False)



# Covariance function:
def get_covariance_matrix(los_list,void_radii_all,bins_spar,bins_sperp,nbar,
                          additional_weights = None,n_boot=10000,seed=42,
                          lambda_reg = 1e-15,cholesky=True,regularise=True,
                          log_field=False,return_mean=False):
    # Get 2D stacked fields for each sample:
    los_list_trimmed, voids_used = trim_los_list(
        los_list,spar_bins,sperp_bins,void_radii_all)
    los_per_void = sum(los_list_trimmed,[])
    void_radii_per_void = np.hstack([rad[used] 
        for rad, used in zip(void_radii_all,voids_used)])
    num_voids = len(void_radii_per_void)
    stacked_fields = get_2d_fields_per_void(
        los_per_void,bins_sperp,bins_spar,
        void_radii_per_void,nbar=nbar).reshape(num_voids,
        (len(bins_spar) - 1)*(len(bins_sperp) - 1))
    # Weights, accounting for differing lengths of the void catalogues
    # in each sample:
    # Weights for each void:
    if additional_weights is not None:
        additional_weights_per_void = np.hstack([weights[used] 
            for weights, used in zip(additional_weights,voids_used)])
    else:
        additional_weights_per_void = np.ones(void_radii_per_void.shape)
    # Bootstrap samples of the voids:
    np.random.seed(42)
    bootstrap_samples = np.random.choice(num_voids,size=(num_voids,n_boot))
    bootstrap_stacks = np.array([np.average(
        stacked_fields[bootstrap_samples[:,k],:],
        axis=0,weights=additional_weights_per_void[bootstrap_samples[:,k]]) 
        for k in tools.progressbar(range(0,n_boot))]).T
    # Compute covariance of the bootstrap samples:
    if log_field:
        # Covariance of the log-field
        log_samples = np.log(bootstrap_stacks)
        finite_samples = np.where(np.all(np.isfinite(log_samples),0))[0]
        cov = np.cov(log_samples[:,finite_samples])
        mean =  np.mean(log_samples[:,finite_samples],1)
    else:
        cov = np.cov(bootstrap_stacks)
        mean = np.mean(bootstrap_stacks,1)
    if regularise:
        cov = regularise_covariance(cov,lambda_reg= lambda_reg)
    if cholesky:
        # If we want to use the Cholesky decomposition, instead of the
        # covariance directly.
        cov = scipy.linalg.cholesky(cov,lower=True)
    if return_mean:
        return cov, mean
    else:
        return cov

#-------------------------------------------------------------------------------
# COSMOLOGICAL INFERENCE

# Get a maximum-likelihood estimate for the initial parameters to use for
# inference:
def get_mle_estimate(initial_guess_eps,theta_ranges_epsilon,*args,**kwargs):
    nll = lambda theta: -log_likelihood_aptest(theta,*args,**kwargs)
    mle_estimate = scipy.optimize.minimize(nll,initial_guess_eps,
                                           bounds=theta_ranges_epsilon)
    return mle_estimate

# Tabulate the inverse function used for moving to redshift-space, to speed
# up the inference. Only useful if the inverse is not a function of the 
# parameters being inferred.
def get_fixed_inverse(Delta_func,ntab=100,ntaf_f=100,sval_range = [0,3],
                      rperp_range=[0,3],f_val_range = [0,1],):
    # Tabulated inverse:
    svals = np.linspace(sval_range[0],sval_range[1],ntab)
    rperp_vals = np.linspace(rperp_range[0],rperp_range[1],ntab)
    f_vals = np.linspace(f_val_range[0],f_val_range[1],ntab_f)
    F_inv_vals = np.zeros((ntab,ntab,ntab_f))
    for i in tools.progressbar(range(0,ntab)):
        for j in range(0,ntab):
            for k in range(0,ntab_f):
                F = (lambda r: r - r*(f_vals[k]/3)*\
                    Delta_func(np.sqrt(r**2 + rperp_vals[i]**2)) \
                    - svals[j])
                F_inv_vals[i,j,k] = scipy.optimize.fsolve(F,svals[j])
    F_inv = lambda x, y, z: scipy.interpolate.interpn((rperp_vals,svals,f_vals),
                                                   F_inv_vals,
                                                   np.vstack((x,y,z)).T,
                                                   method='cubic')
    return F_inv

def generate_scoord_grid(sperp_bins,spar_bins):
    spar = np.hstack([s*np.ones(len(sperp_bins)-1) 
        for s in plot.binCentres(spar_bins)])
    sperp = np.hstack([plot.binCentres(sperp_bins) 
        for s in plot.binCentres(spar_bins)])
    return np.vstack([spar,sperp]).T

# Data filter, that attempts to restrict to a region with high S/N:
def generate_data_filter(cov,mean,scoords,cov_thresh=5,
                         srad_thresh = 1.5):
    # Apply filter on the normalised covariance matrix:
    norm_cov = cov/np.outer(mean,mean)
    data_filter = np.where((1.0/np.sqrt(np.diag(norm_cov)) > cov_thresh) & \
        (np.sqrt(np.sum(scoords**2,1)) < srad_thresh) )[0]
    return data_filter

# Likelihood for the real-space profile (fixed):
def log_likelihood_profile(theta, x, y, yerr,profile_model):
    #rho0,p,C,rb = theta
    #A,r0,c1,f1,B = theta
    #model = profile_broken_power_log(x, A,r0,c1,f1,B)
    model = profile_model(x,*theta)
    sigma2 = yerr**2
    return -0.5 * np.sum( (y - model)**2/sigma2 + np.log(sigma2) )

# Get parameters for the fixed profile:
def get_profile_parameters_fixed(ri,rhoi,sigma_rhoi,
                                 model=profile_modified_hamaus,
                                 initial = np.array([1.0,1.0,1.0,-0.2,0.0,1.0]),
                                 bounds = [(0,None),(0,None),(0,None),(-1,0),
                                           (-1,1),(0,2)]):
    nll = lambda *theta: -log_likelihood_profile(*theta)
    sol = scipy.optimize.minimize(nll, initial, bounds = bounds,
                                  args=(ri, rhoi, sigma_rhoi,
                                        profile_modified_hamaus))
    return sol.x


def run_inference(data_field,theta_ranges_list,theta_initial,filename,
                  log_probability,*args,
                  redo_chain=False,
                  backup_start=True,nwalkers=64,sample="all",n_mcmc=10000,
                  disp=1e-4,Om_fid=0.3111,max_n=1000000,z=0.0225,
                  parallel=False,batch_size=100,n_batches=100,
                  data_filter=None,autocorr_file=None,**kwargs):
    if sample == "all":
        sample = np.array([True for theta in theta_ranges_list])
    ndims = np.sum(sample)
    ndata = len(data_field)
    if data_filter is None:
        data_filter = np.arange(0,ndata)
    # Setup run:
    initial = theta_initial + disp*np.random.randn(nwalkers,ndims)
    filename_initial = filename + ".old"
    if backup_start:
        os.system("cp " + filename + " " + filename_initial)
    backend = emcee.backends.HDFBackend(filename)
    if redo_chain:
        backend.reset(nwalkers, ndims)
    # Inference run:
    if parallel:
        with Pool() as pool:
            sampler = emcee.EnsembleSampler(
                nwalkers, ndims, log_probability_aptest_parallel, 
                args=(z,),
                kwargs={'Om_fid':Om_fid,'cholesky':True,'tabulate_inverse':True,
                        'sample_epsilon':True},
                backend=backend,pool=pool)
            sampler.run_mcmc(initial,n_mcmc , progress=True)
    else:
        #data_filter = np.where((1.0/np.sqrt(np.diag(reg_norm_cov)) > 5) & \
        #    (np.sqrt(np.sum(scoords**2,1)) < 1.5) )[0]
        #reg_cov_filtered = reg_cov[data_filter,:][:,data_filter]
        #cholesky_cov_filtered = scipy.linalg.cholesky(reg_cov_filtered,lower=True)
        sampler = emcee.EnsembleSampler(nwalkers, ndims, log_probability,
                                        args=args,kwargs=kwargs,backend=backend)
        if redo_chain or (autocorr_file is None):
            autocorr = np.zeros((ndims,0))
        else:
            autocorr = np.load(autocorr_file)
        old_tau = np.inf
        for k in range(0,n_batches):
            if (k == 0 and redo_chain):
                sampler.run_mcmc(initial,batch_size , progress=True)
            else:
                sampler.run_mcmc(None,batch_size,progress=True)
            tau = sampler.get_autocorr_time(tol=0)
            autocorr = np.hstack([autocorr,tau.reshape((ndims,1))])
            if autocorr_file is not None:
                np.save(autocorr_file,autocorr)
            # Check convergence
            converged = np.all(tau * 100 < sampler.iteration)
            converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
            if converged:
                break
            old_tau = tau
    return tau, sampler



def run_inference_pipeline(field,cov,sperp_bins,spar_bins,ri,delta_i,
                           sigma_delta_i,log_field=False,
                           infer_profile_args=True,tabulate_inverse=True,
                           cholesky=True,sample_epsilon=True,filter_data=False,
                           z = 0.0225,lambda_cut=1e-23,lambda_ref=1e-27,
                           profile_param_ranges = [[0,np.inf],[0,np.inf],
                                                   [0,np.inf],[-1,0],[-1,1],
                                                   [0,2]],
                           om_ranges = [[0.1,0.5]],eps_ranges = [[0.9,1.1]],
                           f_ranges = [[0,1]],Om_fid = 0.3111,
                           filename = "inference_weighted.h5",
                           autocorr_filename = "autocorr.npy",disp=1e-2,
                           nwalkers=64,n_mcmc=10000,max_n=1000000,
                           batch_size=100,nbatch=100,redo_chain=False,
                           backup_start=True):
    # Compute inverse covariance, or cholesky decomposition:
    if cholesky:
        # Instead of computing the inverse matrix, use cholesky decomposition
        # to evaluate the likelihood. Likelihood will interpret the 
        # 'inverse_matrix' argument as the cholesky decomposition, instead of
        # an inverse
        inverse_matrix = scipy.linalg.cholesky(cov,lower=True)
    else:
        inverse_matrix = get_inverse_covariance(cov,lambda_reg = 1e-23)
    # Generate a data filter:
    scoords = generate_scoord_grid(sperp_bins,spar_bins)
    if filter_data:
        data_filter = generate_data_filter(cov,mean,scoords)
    else:
        data_filter = np.ones(field.flatten().shape,dtype=bool)
    # Field to use for inference:
    data_field = field.flatten()[data_filter]
    # Profile functions:
    if infer_profile_args:
        delta_func = profile_modified_hamaus
        Delta_func = integrated_profile_modified_hamaus
        rho_real = lambda *args: profile_modified_hamaus(*args) + 1.0
    else:
        alpha,beta,rs,delta_c,delta_large, rv = get_profile_parameters_fixed(
            ri,delta_i,sigma_delta_i)
        delta_func = lambda r: profile_modified_hamaus(
                                   r,alpha,beta,rs,delta_c,
                                   delta_large = delta_large,rv=rv)
        Delta_func = lambda r: integrated_profile_modified_hamaus(
                                   r,alpha,beta,rs,delta_c,
                                   delta_large = delta_large,rv=rv)
        rho_real = lambda r: delta_func(r) + 1.0
    # Precompute inverse function. Only useful if profile parameters are fixed:
    if infer_profile_args or (not tabulate_inverse):
        F_inv = None
    else:
        # Tabulate the inverse:
        F_inv = get_fixed_inverse(Delta_func)
    # Setup parameter initial guesses:
    if sample_epsilon:
        initial_guess_MG = np.array([1.0,f_lcdm(z,Om_fid)])
    else:
        initial_guess_MG = np.array([Om_fid,f_lcdm(z,Om_fid)])
    if infer_profile_args:
        profile_params = get_profile_parameters_fixed(ri,delta_i,sigma_delta_i)
        initial_guess = np.hstack([initial_guess_MG,profile_params])
    else:
        initial_guess = initial_guess_MG
    # Parameter bounds:
    if sample_epsilon:
        theta_ranges = eps_ranges + f_ranges + profile_param_ranges
    else:
        theta_ranges = om_ranges + f_ranges + profile_param_ranges
    # Setup Umap to fitler out bad eigenvalue of the covariance matrix:
    Umap, good_eig = get_nonsingular_subspace(cov,lambda_ref,
                                              lambda_cut=lambda_cut,
                                              normalised_cov = False,mu=mean)
    # Arguments to supply to the MCMC run:
    args = (data_field,scoords[data_filter,:],
            inverse_matrix[data_filter,:][:,data_filter],z,Delta_func,
            delta_func,rho_real)
    kwargs = {'cholesky':cholesky,'tabulate_inverse':tabulate_inverse,
              'sample_epsilon':sample_epsilon,'theta_ranges':theta_ranges,
              'singular':False,'Umap':Umap,'good_eig':good_eig,'F_inv':F_inv,
              'log_density':True}
    # Run MCMC:
    tau, sampler = run_inference(data_field,theta_ranges,initial_guess,
                                 filename,log_probability_aptest,*args,
                                 redo_chain=redo_chain,
                                 backup_start=backup_start,nwalkers=nwalkers,
                                 sample="all",n_mcmc=n_mcmc,disp=disp,
                                 max_n=max_n,z=z,parallel=False,
                                 Om_fid=Om_fid,batch_size=batch_size,
                                 n_batches=n_batches,
                                 autocorr_file = autocorr_filename,**kwargs)
    return tau, sampler
















