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

def get_antihalo_properties(snap, file_suffix="AHproperties",
                            default=".h5", low_memory_mode=True):
    """
    Load anti-halo property data for a given snapshot.

    Attempts to load from an HDF5 file (.h5), and falls back to a legacy 
    pickle format (.p) if necessary. Supports a low-memory mode that avoids 
    loading the data into memory unless explicitly requested.

    Parameters:
        snap (pynbody.Snapshot): The simulation snapshot to load from.
        file_suffix (str): Suffix for the filename (default: "AHproperties").
        default (str): File extension to look for first (default: ".h5").
        low_memory_mode (bool): If True, return just a file reference 
                                instead of loading full data.

    Returns:
        If low_memory_mode is True and using legacy format:
            str: Filename for later loading.
        Else:
            h5py.File or list: Anti-halo data loaded from file.
    """
    filename = snap.filename + "." + file_suffix + default
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

class SnapshotGroup:
    """
    A container for managing forward and reverse simulation snapshots and 
    accessing their associated void/halo properties.

    This class handles:
    - Loading multiple simulation snapshots (and their time-reversed counterparts)
    - Accessing halo and void properties (e.g., positions, masses, radii)
    - Remapping coordinates to Equatorial space via configurable transformations
    - Efficient memory usage through optional lazy loading and caching

    Coordinate System Note:
    The underlying simulation snapshots may not be in Equatorial coordinates.
    To convert to Equatorial space, properties like void centres or halo positions
    are remapped via `tools.remapAntiHaloCentre`, which applies:
        - A shift to box-centred coordinates
        - Optional axis swapping (swapXZ=True swaps X and Z)
        - Optional axis flipping (reverse=True mirrors positions)

    These transformations are controlled by the `swapXZ` and `reverse` flags. 
    Their default values assume a particular snapshot orientation, but users
    should verify and adjust these flags if working with different conventions.
    """

    def __init__(self, snap_list, snap_list_reverse, low_memory_mode=True,
                 swapXZ=False, reverse=False, remap_centres=False):
        """
        Initialize a SnapshotGroup from lists of forward and reverse snapshots.

        Parameters:
            snap_list (list): Forward-time simulation snapshots
            snap_list_reverse (list): Corresponding reverse-time snapshots
            low_memory_mode (bool): If True, avoid loading all properties into memory
            swapXZ (bool): Whether to swap X and Z axes when remapping coordinates
            reverse (bool): Whether to flip coordinates around box center
            remap_centres (bool): (Not yet implemented - ignored) Whether to 
                                  remap void/halo centres to Equatorial frame.
        """
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
        self.property_list = [
            "halo_centres", "halo_masses",
            "antihalo_centres", "antihalo_masses",
            "cell_volumes", "void_centres",
            "void_volumes", "void_radii",
            "radius_bins", "pair_counts",
            "bin_volumes", "delta_central",
            "delta_average"
        ]
        self.additional_properties = {
            "halos": None,
            "antihalos": None,
            "snaps": self.snaps,
            "snaps_reverse": self.snaps_reverse
        }
        self.property_map = {name: idx for idx, name in enumerate(self.property_list)}
        self.reverse = reverse
        self.swapXZ = swapXZ
        self.remap_centres = remap_centres
        self.boxsize = self.snaps[0].properties['boxsize'].ratio("Mpc a h**-1")
        self.all_properties = [None for _ in self.property_list]
        self.snap_filenames = [snap.filename for snap in self.snaps]
        self.snap_reverse_filenames = [snap.filename for snap in self.snaps_reverse]
    def is_valid_property(self, prop):
        if isinstance(prop, int):
            return prop in range(len(self.property_list))
        elif isinstance(prop, str):
            return prop in self.property_list
        return False
    def get_property_index(self, prop):
        if isinstance(prop, int):
            if prop in range(len(self.property_list)):
                return prop
            raise Exception("Property index is out of range.")
        elif isinstance(prop, str):
            if prop in self.property_list:
                return self.property_map[prop]
            raise Exception("Requested property does not exist.")
        else:
            raise Exception("Invalid property type")
    def get_property_name(self, prop):
        if isinstance(prop, int):
            if prop in range(len(self.property_list)):
                return self.property_list[prop]
            raise Exception("Property index is out of range.")
        elif isinstance(prop, str):
            if prop in self.property_list:
                return prop
            raise Exception("Requested property does not exist.")
        else:
            raise Exception("Invalid property type")
    def get_property(self, snap_index, property_name, recompute=False):
        """
        Access a property for a single snapshot, loading from cache or disk.
        """
        prop_index = self.get_property_index(property_name)
        if self.all_properties[prop_index] is not None and not recompute:
            return self.all_properties[prop_index][snap_index]

        property_list = self.all_property_lists[snap_index]
        if property_list is None:
            property_list = get_antihalo_properties(self.snaps[snap_index])

        if isinstance(property_list, h5py._hl.files.File):
            return property_list[self.get_property_name(property_name)]
        elif isinstance(property_list, list):
            return property_list[self.get_property_index(property_name)]
        elif isinstance(property_list, str):
            props_list = tools.loadPickle(property_list)
            return props_list[self.get_property_index(property_name)]
        else:
            raise Exception("Invalid Property Type")
    def check_remappable(self, property_name):
        """
        Check if a property represents a position needing coordinate remapping.
        """
        index = self.get_property_index(property_name)
        return index in [0, 5]  # halo_centres or void_centres
    def get_all_properties(self, property_name, cache=True, recompute=False):
        """
        Get a list of a given property for all snapshots.
        Handles remapping to Equatorial coordinates if applicable.
        """
        prop_index = self.get_property_index(property_name)
        if self.all_properties[prop_index] is None:
            if self.check_remappable(property_name):
                # Remap positions to Equatorial coordinates
                properties = [
                    tools.remapAntiHaloCentre(
                        self.get_property(i, property_name, recompute=recompute),
                        boxsize=self.boxsize,
                        swapXZ=self.swapXZ,
                        reverse=self.reverse)
                    for i in range(self.N)
                ]
            else:
                properties = [
                    self.get_property(i, property_name, recompute=recompute)
                    for i in range(self.N)
                ]
            if cache:
                self.all_properties[prop_index] = properties
            return properties
        else:
            return self.all_properties[prop_index]
    def __getitem__(self, property_name):
        """
        Enable bracket-access to named or additional properties.
        Automatically returns all-snapshot versions of properties.
        """
        if self.is_valid_property(property_name):
            return self.get_all_properties(property_name)
        elif isinstance(property_name, str) and property_name in self.additional_properties:
            if self.additional_properties[property_name] is not None:
                return self.additional_properties[property_name]
            else:
                # Lazy-load derived properties
                if property_name == "halos":
                    self.additional_properties["halos"] = [snap.halos() for snap in self.snaps]
                elif property_name == "antihalos":
                    self.additional_properties["antihalos"] = [snap.halos() for snap in self.snaps_reverse]
                else:
                    raise Exception("Invalid property_name")
                return self.additional_properties[property_name]
        else:
            raise Exception("Invalid property_name")




#-------------------------------------------------------------------------------
# COSMOLOGY FUNCTIONS

def Ez2(z, Om, Or=0, Ok=0, Ol=None, **kwargs):
    """
    Compute E(z)^2, the square of the dimensionless Hubble parameter,
    for general FLRW cosmologies.

    E(z)^2 = Or*(1 + z)^4 + Om*(1 + z)^3 + Ok*(1 + z)^2 + Ol

    Parameters:
        z (float or array): Redshift(s)
        Om (float): Matter density parameter
        Or (float): Radiation density parameter (default: 0)
        Ok (float): Curvature density parameter (default: 0)
        Ol (float or None): Dark energy density parameter. If None, inferred
                            from flatness (1 - Om - Ok - Or)

    Returns:
        float or array: E(z)^2
    """
    if Ol is None:
        Ol = 1.0 - Om - Ok - Or
    return Or*(1 + z)**4 + Om*(1 + z)**3 + Ok*(1 + z)**2 + Ol

def f_lcdm(z, Om, gamma=0.55, Ol=None, Ok=0, Or=0, **kwargs):
    """
    Approximate the linear growth rate f(z) in ΛCDM cosmology.

    f(z) ≈ [Ω_m(z)]^γ, where γ ≈ 0.55 for ΛCDM

    Parameters:
        z (float or array): Redshift(s)
        Om (float): Present-day matter density
        gamma (float): Growth index (default: 0.55)
        (Other parameters as in Ez2)

    Returns:
        float or array: Linear growth rate f(z)
    """
    Ez2_val = Ez2(z, Om, Or=Or, Ok=Ok, Ol=Ol)
    return (Om * (1 + z)**3 / Ez2_val)**gamma


def Hz(z, Om, h=None, Ol=None, Ok=0, Or=0, **kwargs):
    """
    Compute the Hubble parameter H(z) in units of km/s/Mpc.

    If h is None, assumes h=1 (returns H(z) / h), yielding units km*h/(s*Mpc).
    If h is provided, computes H(z) directly in km/s/Mpc.

    Parameters:
        z (float or array): Redshift(s)
        Om (float): Matter density
        h (float or None): Dimensionless Hubble constant (H0/100). Default: None
        (Other parameters as in Ez2)

    Returns:
        float or array: H(z) in km/s/Mpc
    """
    if h is None:
        h = 1.0
    return 100 * h * np.sqrt(Ez2(z, Om, Or=Or, Ok=Ok, Ol=Ol))

def ap_parameter(z, Om, Om_fid, h=0.7, h_fid=0.7, **kwargs):
    """
    Compute the Alcock-Paczynski parameter ε(z), which quantifies geometric
    distortions due to miss-specified cosmological parameters.

    ε(z) = [H(z) D_A(z)] / [H_fid(z) D_A_fid(z)]

    Parameters:
        z (float): Redshift
        Om (float): Matter density in test cosmology
        Om_fid (float): Fiducial matter density
        h (float): Hubble constant for test cosmology
        h_fid (float): Hubble constant for fiducial cosmology

    Returns:
        float: Alcock-Paczynski distortion parameter ε(z)
    """
    # Cosmologies:
    cosmo_fid = astropy.cosmology.FlatLambdaCDM(H0=100*h_fid, Om0=Om_fid)
    cosmo_test = astropy.cosmology.FlatLambdaCDM(H0=100*h, Om0=Om)
    # Hubble rates:
    Hz = 100 * h * np.sqrt(Om * (1 + z)**3 + 1.0 - Om)
    Hz_fid = 100 * h_fid * np.sqrt(Om_fid * (1 + z)**3 + 1.0 - Om_fid)
    # Angular diameter distances:
    Da = cosmo_test.angular_diameter_distance(z).value
    Da_fid = cosmo_fid.angular_diameter_distance(z).value
    return (Hz * Da) / (Hz_fid * Da_fid)


def void_los_velocity(z, Delta, r_par, r_perp, Om, f=None, **kwargs):
    """
    Compute the line-of-sight (LOS) peculiar velocity at a position 
    relative to a void center, based on linear theory.

    Parameters:
        z (float): Redshift
        Delta (function): Cumulative density contrast profile Δ(r)
        r_par (float or array): LOS distance from void center
        r_perp (float or array): Transverse distance from void center
        Om (float): Matter density
        f (float or None): Growth rate. If None, computed via f_lcdm()

    Returns:
        float or array: LOS velocity in km/s
    """
    if f is None:
        f = f_lcdm(z, Om, **kwargs)
    hz = Hz(z, Om, **kwargs)
    r = np.sqrt(r_par**2 + r_perp**2)
    Dr = Delta(r)
    return -(f / 3.0) * (hz / (1.0 + z)) * Dr * r_par



def void_los_velocity_derivative(z, Delta, delta, r_par, r_perp, Om, f=None, **kwargs):
    """
    Compute the derivative of the LOS velocity with respect to r.

    Assumes Delta(r) is the cumulative density contrast profile (mass within 
    a sphere), and delta(r) is the local density contrast (mass in a shell).
    These are related via integration, but must both be provided as functions.

    Parameters:
        z (float): Redshift
        Delta (function): Cumulative density profile Δ(r)
        delta (function): Local density contrast δ(r)
        r_par (float): LOS distance
        r_perp (float): Transverse distance
        Om (float): Matter density
        f (float or None): Growth rate (Lambda-CDM value assumed if not provided)

    Returns:
        float: Derivative of velocity w.r.t. r
    """
    r = np.sqrt(r_par**2 + r_perp**2)
    Dr = Delta(r)
    dr = delta(r)
    hz = Hz(z, Om, **kwargs)
    if f is None:
        f = f_lcdm(z, Om, **kwargs)
    return -(f / 3.0) * (hz / (1.0 + z)) * Dr - \
           f * (r_par / (r + 1e-12))**2 * (hz / (1.0 + z)) * (dr - Dr)


def z_space_jacobian(z, Delta, delta, r_par, r_perp, Om, linearise_jacobian=True, **kwargs):
    """
    Compute the Jacobian for the transformation from real to redshift space.

    Parameters:
        z (float): Redshift
        Delta (function): Cumulative density profile
        delta (function): Local density profile
        r_par (float): LOS distance
        r_perp (float): Transverse distance
        Om (float): Matter density
        linearise_jacobian (bool): If True, return 1st-order approximation

    Returns:
        float: Jacobian of the transformation
    """
    hz = Hz(z, Om, **kwargs)
    dudr = void_los_velocity_derivative(z, Delta, delta, r_par, r_perp, Om, **kwargs)
    if linearise_jacobian:
        return 1.0 - ((1.0 + z) / hz) * dudr
    else:
        return 1.0 / (1.0 + ((1.0 + z) / hz) * dudr)


def to_z_space(r_par, r_perp, z, Om, Delta=None, u_par=None, f=None, **kwargs):
    """
    Transform real-space LOS coordinates to redshift space.

    If u_par is not supplied, a linear-theory velocity model is used
    based on the cumulative density profile Δ(r). Otherwise, uses 
    the explicitly provided velocity field u_par.

    Parameters:
        r_par (float or array): LOS coordinate in real space
        r_perp (float or array): Transverse coordinate in real space
        z (float): Redshift
        Om (float): Matter density
        Delta (function): Cumulative density profile Δ(r)
        u_par (float or array): LOS velocity (optional)
        f (float or None): Growth rate (Lambda-CDM value assumed if not provided)

    Returns:
        list: [s_par, s_perp] — redshift-space coordinates
    """
    if u_par is None:
        r = np.sqrt(r_par**2 + r_perp**2)
        if f is None:
            f = f_lcdm(z, Om, **kwargs)
        s_par = (1.0 - (f / 3.0) * Delta(r)) * r_par
    else:
        hz = Hz(z, Om, **kwargs)
        s_par = r_par + (1.0 + z) * u_par / hz
    s_perp = r_perp
    return [s_par, s_perp]


def iterative_zspace_inverse(s_par, f, Delta, N_max, atol=1e-5, rtol=1e-5):
    """
    Numerically invert the redshift-space LOS coordinate to estimate
    the real-space coordinate (r_par), assuming the linear-theory model.

    Parameters:
        s_par (float): LOS coordinate in redshift space
        f (float): Linear growth rate
        Delta (function): Cumulative density profile
        N_max (int): Maximum number of iterations

    Returns:
        float: Estimated r_par
    """
    r_par_guess = s_par
    for _ in range(N_max):
        r = np.sqrt(r_par_guess**2 + r_perp**2)
        r_par_new = s_par / (1.0 - (f / 3.0) * Delta(r))
        if np.abs(r_par_new - r_par_guess) < atol or \
           np.abs(r_par_new / r_par_guess - 1.0) < rtol:
            break
        r_par_guess = r_par_new
    return r_par_guess


def to_real_space(s_par, s_perp, z, Om, Om_fid=None, Delta=None, u_par=None, f=None,
                  N_max=5, atol=1e-5, rtol=1e-5, F_inv=None, **kwargs):
    """
    Convert redshift-space coordinates (s_par, s_perp) back into real-space 
    coordinates (r_par, r_perp), assuming either:

    - A linear-theory velocity model derived from the cumulative density profile Delta(r), or
    - An explicit peculiar velocity field u_par

    Optionally uses a precomputed inverse mapping function F_inv, or falls 
    back to an iterative inversion method.

    Parameters:
        s_par (float): LOS redshift-space coordinate (must be scalar if inverting manually)
        s_perp (float or array): Transverse redshift-space coordinate
        z (float): Redshift
        Om (float): Matter density
        Om_fid (float or None): Fiducial matter density (unused here but may be passed downstream)
        Delta (function): Cumulative density contrast profile Δ(r), required for linear inversion
        u_par (float or array or None): LOS peculiar velocity (if supplied, used directly)
        f (float or None): Growth rate; computed via f_lcdm if not supplied
        N_max (int): Max number of iterations for manual inversion
        atol (float): Absolute tolerance for iterative convergence
        rtol (float): Relative tolerance for iterative convergence
        F_inv (callable or None): Tabulated inverse mapping function

    Returns:
        list: [r_par, r_perp] — real-space coordinates
    """
    r_perp = s_perp  # Perpendicular component is unaffected
    if u_par is None:
        # Use linear velocity relation:
        if f is None:
            f = f_lcdm(z, Om, **kwargs)
        if F_inv is None:
            # Manual inversion via iterative method:
            if not np.isscalar(s_par):
                raise Exception("Must be a scalar to perform manual inversion")
            if Delta is None:
                raise ValueError("Delta profile must be supplied for linear inversion.")
            # Use helper to handle the iterative logic
            r_par = iterative_zspace_inverse(s_par, f, Delta, N_max,
                                             atol = atol, rtol = rtol)
        else:
            # Use tabulated inverse function
            r_par = F_inv(s_perp, s_par, f * np.ones(np.shape(s_perp)))
    else:
        # Use directly supplied LOS peculiar velocity
        hz = Hz(z, Om, **kwargs)
        r_par = s_par - (1.0 + z) * u_par / hz
    return [r_par, r_perp]


def geometry_correction(s_par, s_perp, epsilon, **kwargs):
    """
    Apply the Alcock-Paczynski geometric correction to redshift-space 
    coordinates (s_par, s_perp) to account for a misspecified cosmology.

    This transformation rescales both the distance and the apparent angle 
    between line-of-sight (LOS) and transverse components, correcting the 
    apparent shape of cosmological voids distorted due to miss-specification
    of the cosmological parameters

    Based on the transformation:
        - s_factor adjusts the total distance accounting for ε-dependent anisotropy.
        - mus (cosine of LOS angle) is adjusted to mus_new.
        - New coordinates are computed from these corrected values.

    Parameters:
        s_par (float or array): LOS redshift-space coordinate
        s_perp (float or array): Transverse redshift-space coordinate
        epsilon (float or None): Alcock-Paczynski distortion parameter ε
                                 (set to 1.0 if None)

    Returns:
        tuple: (s_par_new, s_perp_new) — corrected redshift-space coordinates
    """
    if epsilon is None:
        epsilon = 1.0
    # Radial distance from void center
    s = np.sqrt(s_par**2 + s_perp**2)
    # Cosine of LOS angle
    mus = s_par / s
    # Distance correction factor
    # This accounts for distortions in radial vs. transverse directions
    s_factor = np.sqrt(1.0 + epsilon**2 * (1.0 / mus**2 - 1.0))
    # Apply AP correction to total distance and angle
    s_new = s * mus * epsilon**(-2.0 / 3.0) * s_factor
    mus_new = np.sign(mus) / s_factor  # Corrected cosine of angle
    # Compute corrected coordinates
    s_par_new = mus_new * s_new
    s_perp_new = np.sign(s_perp) * s_new * np.sqrt(1.0 - mus_new**2)
    return s_par_new, s_perp_new


def z_space_profile(s_par, s_perp, rho_real, z, Om, Delta, delta,
                    Om_fid=0.3111, epsilon=None, apply_geometry=False, **kwargs):
    """
    Compute the redshift-space density profile ρ(s_par, s_perp) based on
    a real-space density profile and a model of redshift-space distortions.

    This includes:
      - Optional Alcock-Paczynski geometric correction (applied to input coords)
      - Coordinate transformation from redshift space to real space
      - Jacobian of the transformation (density scaling)
      - Evaluation of the density field at the mapped coordinates

    Parameters:
        s_par (float): LOS redshift-space coordinate
        s_perp (float): Transverse redshift-space coordinate
        rho_real (function): Real-space density profile ρ(r)
        z (float): Redshift
        Om (float): Matter density parameter of the assumed cosmology
        Delta (function): Cumulative density contrast profile Δ(r)
        delta (function): Local density contrast profile δ(r)
        Om_fid (float): Fiducial matter density (default: 0.3111)
        epsilon (float or None): Alcock-Paczynski distortion parameter ε.
                                 If None and apply_geometry is True, it is computed.
        apply_geometry (bool): Whether to apply the AP correction

    Returns:
        float: Redshift-space density ρ(s_par, s_perp)
    """
    # Step 1: Apply Alcock-Paczynski geometric correction (if requested)
    if apply_geometry:
        if epsilon is None:
            epsilon = ap_parameter(z, Om, Om_fid, **kwargs)
        s_par_new, s_perp_new = geometry_correction(s_par, s_perp, epsilon)
    else:
        s_par_new = s_par
        s_perp_new = s_perp
    # Step 2: Convert redshift-space coords back to real-space coords
    r_par, r_perp = to_real_space(s_par_new, s_perp_new, z, Om, Delta=Delta, **kwargs)
    # Step 3: Compute the Jacobian of the transformation (∂r/∂s)
    jacobian = z_space_jacobian(z, Delta, delta, r_par, r_perp, Om, **kwargs)
    # Step 4: Evaluate the real-space profile at the recovered radius
    r = np.sqrt(r_par**2 + r_perp**2)
    return rho_real(r) * jacobian

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

def log_likelihood_aptest_revised(theta, data_field, scoords, inv_cov, z,
                                   Delta, delta, rho_real,
                                   data_filter=None,
                                   cholesky=False,
                                   normalised=False,
                                   tabulate_inverse=False,
                                   ntab=10,
                                   sample_epsilon=False,
                                   Om_fid=0.3,
                                   singular=False,
                                   Umap=None,
                                   good_eig=None,
                                   F_inv=None,
                                   log_density=False,
                                   infer_profile_args=False,
                                   **kwargs):
    """
    Compute the log-likelihood for a cosmological + void profile model 
    using redshift-space distortion data and the Alcock-Paczynski test.

    Supports both:
    - Fixed profile functions (Δ(r), δ(r), ρ(r)), or
    - Parameterised profiles where shape parameters are inferred from data

    Parameters:
        theta (array): Model parameters:
            - [Om, f, ...] if sampling cosmology and profile
            - [epsilon, f, ...] if sample_epsilon=True
        data_field (array): Observed or simulated redshift-space density field
        scoords (array): (s_par, s_perp) coordinates
        inv_cov (array or Cholesky factor): Covariance inverse or lower triangle
        z (float): Redshift
        Delta, delta, rho_real (functions or constructors):
            - If infer_profile_args=True, must accept (*params)
            - Else, must be callables of r only
        data_filter (array or None): Optional data mask/filter
        cholesky (bool): If True, use Cholesky decomposition of covariance
        normalised (bool): Indicates whether to normalise the model
        tabulate_inverse (bool): If True, generate tabulated inverse map
        ntab (int): Number of samples for inverse interpolation
        sample_epsilon (bool): If True, sample ε instead of Om
        Om_fid (float): Fiducial Om used to compute ε
        singular (bool): Use projected likelihood in reduced eigenspace
        Umap (matrix): Projection matrix (for singular=True)
        good_eig (array): Eigenvalues for singular projection
        F_inv (callable or None): Precomputed inverse mapping
        log_density (bool): If True, use log(ρ) in likelihood
        infer_profile_args (bool): If True, extract profile params from θ

    Returns:
        float: Log-likelihood value
    """
    # Apply optional data filtering
    if data_filter is not None:
        data_field = data_field[data_filter]
        scoords = scoords[data_filter, :]
        if cholesky or not singular:
            inv_cov = inv_cov[data_filter, :][:, data_filter]

    s_par, s_perp = scoords[:, 0], scoords[:, 1]
    # Unpack parameter vector
    if sample_epsilon:
        epsilon, f = theta[0], theta[1]
        profile_params = theta[2:]
        Om = Om_fid
    else:
        Om, f = theta[0], theta[1]
        profile_params = theta[2:]
        epsilon = ap_parameter(z, Om, Om_fid, **kwargs)
    # Construct profile functions
    if infer_profile_args:
        Delta_func = lambda r: Delta(r, *profile_params)
        delta_func = lambda r: delta(r, *profile_params)
        rho_func = lambda r: rho_real(r, *profile_params)
    else:
        Delta_func = Delta
        delta_func = delta
        rho_func = rho_real
    # Generate a tabulated inverse mapping if needed
    if F_inv is None and tabulate_inverse:
        F_inv = tools.inverse_los_map(z, Om, Delta_func, f, ntab=ntab, **kwargs)
    # Evaluate the model at each (s_par, s_perp) coordinate
    model_field = np.array([
        z_space_profile(sp, st, rho_func, z, Om, Delta_func, delta_func,
                        Om_fid=Om_fid, epsilon=epsilon,
                        apply_geometry=sample_epsilon,
                        F_inv=F_inv, f=f,
                        **kwargs)
        for sp, st in zip(s_par, s_perp)
    ])
    if log_density:
        model_field = np.log(model_field)
    # Project into reduced space if using singular-mode filtering
    if singular:
        data_field = Umap @ data_field
        model_field = Umap @ model_field
        inv_cov = np.diag(1.0 / good_eig)
    # Compute residual
    delta_vec = data_field - model_field
    if cholesky:
        alpha = scipy.linalg.solve_triangular(inv_cov, delta_vec, lower=True)
        return -0.5 * np.dot(alpha, alpha)
    else:
        return -0.5 * np.dot(delta_vec, inv_cov @ delta_vec)


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

def log_flat_prior_single(x, bounds):
    """
    Compute the log of a flat prior over a bounded interval.

    Parameters:
        x (float): Parameter value
        bounds (tuple): (min, max) bounds of the flat prior

    Returns:
        float: 0 if within bounds, -inf if out of bounds
    """
    xmin, xmax = bounds
    return 0.0 if xmin <= x <= xmax else -np.inf

def log_flat_prior(theta, bounds):
    """
    Compute the joint log of a flat prior over a hyperrectangle.

    Parameters:
        theta (array-like): List of parameter values
        bounds (list of tuples): List of (min, max) bounds for each parameter

    Returns:
        float: 0 if all parameters are within bounds, -inf otherwise
    """
    if any(t < b[0] or t > b[1] for t, b in zip(theta, bounds)):
        return -np.inf
    return 0.0

# DEPRECATED Prior (assuming flat prior for now):
def log_prior_aptest_old(theta,
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

# DEPRECATED Posterior (unnormalised):
def log_probability_aptest_old(theta,*args,**kwargs):
    lp = log_prior_aptest(theta,**kwargs)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_aptest(theta,*args,**kwargs)

def log_probability_aptest(theta, *args, **kwargs):
    """
    Compute the log posterior for the AP test likelihood.

    Posterior is defined as:
        log_posterior = log_prior + log_likelihood

    Assumes a flat prior with hard bounds defined by 'theta_ranges' in kwargs.

    Parameters:
        theta (array): Parameter vector
        *args: Passed to log_likelihood_aptest_revised
        **kwargs:
            - theta_ranges (list of tuples): Prior bounds for each parameter
            - All other kwargs passed to log_likelihood_aptest

    Returns:
        float: Log posterior
    """
    theta_ranges = kwargs.pop("theta_ranges", None)
    if theta_ranges is None:
        raise ValueError("Missing 'theta_ranges' in kwargs for prior evaluation.")

    lp = log_flat_prior(theta, theta_ranges)
    if not np.isfinite(lp):
        return -np.inf

    return lp + log_likelihood_aptest(theta, *args, **kwargs)

# UNUSED
def log_probability_aptest_parallel(theta,*args,**kwargs):
    lp = log_prior_aptest(theta,**kwargs)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_aptest_parallel(theta,*args,**kwargs)



#-------------------------------------------------------------------------------
# GAUSSIANITY TESTING

def tikhonov_regularisation(cov_matrix, alpha=1e-10):
    """
    Apply Tikhonov regularisation to a covariance matrix.

    This adds a scaled identity matrix to the covariance, stabilising the inverse:
        cov_reg = cov + alpha * I

    Parameters:
        cov_matrix (ndarray): Original covariance matrix (NxN)
        alpha (float): Regularisation strength (default: 1e-3)

    Returns:
        ndarray: Regularised covariance matrix
    """
    return cov_matrix + alpha * np.eye(cov_matrix.shape[0])

def regularise_covariance(cov, lambda_reg=1e-10):
    """
    Symmetrise and regularise a covariance matrix using Tikhonov regularisation.

    Ensures the matrix is symmetric and numerically stable for inversion.

    Parameters:
        cov (ndarray): Input covariance matrix
        lambda_reg (float): Tikhonov regularisation strength (default: 1e-10)

    Returns:
        ndarray: Symmetrised and regularised covariance matrix
    """
    symmetric_cov = 0.5 * (cov + cov.T)
    regularised_cov = tikhonov_regularisation(symmetric_cov, lambda_reg=lambda_reg)
    return regularised_cov


def get_inverse_covariance(cov, lambda_reg=1e-10):
    """
    Compute the inverse of a (regularised) covariance matrix using Cholesky decomposition.

    This involves:
    - Symmetrising the input
    - Applying Tikhonov regularisation for numerical stability
    - Computing the inverse via Cholesky factorisation:
        C^(-1) = (L^(-1))^T @ L^(-1)

    Parameters:
        cov (ndarray): Covariance matrix (NxN)
        lambda_reg (float): Regularisation strength (default: 1e-10)

    Returns:
        ndarray: Inverse of the regularised covariance matrix
    """
    regularised_cov = regularise_covariance(cov, lambda_reg=lambda_reg)
    L = np.linalg.cholesky(regularised_cov)      # Lower triangular matrix
    P = np.linalg.inv(L)                         # Inverse of L
    inv_cov = np.matmul(P, P.T)                  # Reconstruct full inverse
    return inv_cov


def range_excluding(kmin, kmax, exclude):
    """
    Generate a list of integers in [kmin, kmax), excluding values in 'exclude'.

    Parameters:
        kmin (int): Start of range (inclusive)
        kmax (int): End of range (exclusive)
        exclude (list or array): Values to remove from the range

    Returns:
        ndarray: Array of integers not in 'exclude'
    """
    return np.setdiff1d(range(kmin, kmax), exclude)

def get_nonsingular_subspace(C, lambda_reg,
                             lambda_cut=None,
                             normalised_cov=False,
                             mu=None):
    """
    Compute a projection onto the non-singular subspace of a covariance matrix.

    The covariance is regularised and optionally normalised before eigenvalue decomposition.
    Only eigenvectors with eigenvalues above a cutoff are retained.

    Parameters:
        C (ndarray): Covariance matrix (k x k)
        lambda_reg (float): Tikhonov regularisation parameter
        lambda_cut (float or None): Minimum eigenvalue to keep (default: 10 * lambda_reg)
        normalised_cov (bool): If True, normalise C by mu.outer(mu)
        mu (ndarray or None): Mean vector (required if normalised_cov=True)

    Returns:
        tuple:
            - Umap (ndarray): Projection matrix to nonsingular eigenspace
            - good_eig (ndarray): Retained eigenvalues
    """
    reg_cov = regularise_covariance(C, lambda_reg=lambda_reg)

    if normalised_cov:
        if mu is None:
            raise ValueError("Mean 'mu' must be provided for normalised covariance.")
        norm_cov = C / np.outer(mu, mu)
        norm_reg_cov = regularise_covariance(norm_cov, lambda_reg=lambda_reg)
        eig, U = scipy.linalg.eigh(norm_reg_cov)
    else:
        eig, U = scipy.linalg.eigh(reg_cov)

    if lambda_cut is None:
        lambda_cut = 10 * lambda_reg

    good_eig = eig[eig >= lambda_cut]
    Umap = U[:, eig >= lambda_cut].T  # Each row is a retained eigenvector

    return Umap, good_eig


def get_solved_residuals(samples, covariance, xbar,
                         singular=False,
                         normalised_cov=False,
                         L=None,
                         lambda_cut=1e-23,
                         lambda_reg=1e-27,
                         Umap=None,
                         good_eig=None):
    """
    Compute whitened residuals for Gaussianity testing.

    Supports both full-rank and projected cases using a reduced eigenbasis.

    Parameters:
        samples (ndarray): Shape (k, n) — k variables, n samples
        covariance (ndarray): Covariance matrix (k x k)
        xbar (ndarray): Mean of the samples (length k)
        singular (bool): Whether to project into nonsingular subspace
        normalised_cov (bool): Whether to normalise by the mean
        L (ndarray or None): Cholesky factor of covariance (optional)
        lambda_cut (float): Cutoff for eigenvalue retention
        lambda_reg (float): Regularisation strength
        Umap (ndarray or None): Projection matrix (optional)
        good_eig (ndarray or None): Retained eigenvalues (optional)

    Returns:
        ndarray: Whitened residuals (projected or full)
    """
    k, n = samples.shape
    if not singular:
        if normalised_cov:
            residual = samples / xbar[:, None] - 1.0
        else:
            residual = samples - xbar[:, None]
        if L is None:
            reg_cov = regularise_covariance(covariance, lambda_reg=lambda_reg)
            L = scipy.linalg.cholesky(reg_cov, lower=True)
        # Solve triangular system for each sample
        solved_residuals = np.array([
            scipy.linalg.solve_triangular(L, residual[:, i], lower=True)
            for i in tools.progressbar(range(n))
        ]).T
    else:
        # Project into reduced eigenspace
        if Umap is None or good_eig is None:
            Umap, good_eig = get_nonsingular_subspace(
                covariance, lambda_reg=lambda_reg,
                lambda_cut=lambda_cut,
                normalised_cov=normalised_cov,
                mu=xbar
            )
        if normalised_cov:
            residual = Umap @ (samples / xbar[:, None] - 1.0)
        else:
            residual = Umap @ (samples - xbar[:, None])
        solved_residuals = residual / np.sqrt(good_eig[:, None])
    return solved_residuals


def compute_normality_test_statistics(samples,
                                      covariance=None,
                                      xbar=None,
                                      solved_residuals=None,
                                      low_memory_sum=False,
                                      **kwargs):
    """
    Compute test statistics to evaluate Gaussianity of residuals.

    Statistics:
      - A: Skewness-like cubic term
      - B: Kurtosis-like quartic term

    Parameters:
        samples (ndarray): (k, n) array of samples
        covariance (ndarray or None): Covariance matrix (used if residuals not provided)
        xbar (ndarray or None): Mean vector (used if residuals not provided)
        solved_residuals (ndarray or None): Precomputed whitened residuals
        low_memory_sum (bool): If True, use a lower-memory summation loop

    Returns:
        list: [A, B] — test statistics
    """
    n = samples.shape[1]
    k = samples.shape[0]
    if covariance is None:
        covariance = np.cov(samples)
    if xbar is None:
        xbar = np.mean(samples, axis=1)
    if solved_residuals is None:
        solved_residuals = get_solved_residuals(samples, covariance, xbar, **kwargs)
    if low_memory_sum:
        Ai = np.array([
            np.sum(
                np.sum(solved_residuals[:, i][:, None] * solved_residuals, axis=0) ** 3
            ) for i in tools.progressbar(range(n))
        ])
        A = np.sum(Ai) / (6 * n)
    else:
        product = solved_residuals.T @ solved_residuals
        A = np.sum(product ** 3) / (6 * n)
    B = np.sqrt(n / (8 * k * (k + 2))) * (
        np.sum(np.sum(solved_residuals ** 2, axis=0) ** 2) / n - k * (k + 2)
    )
    return [A, B]


#-------------------------------------------------------------------------------
# STACKED DENSITY FIELDS

def get_zspace_centres(halo_indices, snap_list, snap_list_rev,
                       hrlist=None, recompute_zspace=False,
                       swapXZ=False, reverse=True):
    """
    Compute redshift-space void centers from a halo catalogue (in reverse simulations).

    This function:
      - Loads the redshift-space positions of all particles in the forward snapshot
      - Uses the halo catalogue (from reverse sim) to identify void particles
      - Computes the redshift-space center for each void
      - Applies remapping to shift into the final coordinate frame

    Parameters:
        halo_indices (list of lists): Per-snapshot list of halo indices representing voids
        snap_list (list): Forward simulation snapshots (used for positions)
        snap_list_rev (list): Reverse snapshots (used for halos = voids)
        hrlist (list or None): If supplied, overrides halos loaded from snap_list_rev
        recompute_zspace (bool): If True, force recomputation of redshift-space positions
        swapXZ (bool): Whether to swap X and Z axes during coordinate remapping
        reverse (bool): Whether to reflect coordinates around box center during remapping

    Returns:
        list of arrays: Redshift-space centers of voids for each snapshot
    """
    if len(halo_indices) != len(snap_list):
        raise ValueError("halo_indices list does not match snapshot list.")
    num_samples = len(halo_indices)

    # Preallocate list of centers, shape (N_voids, 3) per snapshot
    centres = [np.full((len(halos), 3), np.nan) for halos in halo_indices]
    for ns in range(num_samples):
        snap = snap_list[ns]
        # Get particle index sort order (needed for consistent indexing)
        if os.path.isfile(snap.filename + ".snapsort.p"):
            sorted_indices = tools.loadPickle(snap.filename + ".snapsort.p")
        else:
            sorted_indices = np.argsort(snap['iord'])
        # Load halos from reverse simulation (or use override list)
        halos = hrlist[ns] if hrlist is not None else snap_list_rev[ns].halos()
        boxsize = snap.properties['boxsize'].ratio("Mpc a h**-1")
        # Get redshift-space positions of all particles in forward snapshot
        positions = tools.loadOrRecompute(
            snap.filename + ".z_space_pos.p",
            simulation_tools.redshift_space_positions,
            snap,
            centre=np.array([boxsize / 2] * 3),
            _recomputeData=recompute_zspace
        )
        for k in tools.progressbar(range(len(halo_indices[ns]))):
            index = halo_indices[ns][k]
            if index >= 0:
                # Lookup particle indices in halo
                particle_ids = halos[index + 1]['iord']
                particle_pos = positions[sorted_indices[particle_ids], :]

                # Compute center in redshift space, apply remapping
                centre = context.computePeriodicCentreWeighted(
                    particle_pos, periodicity=boxsize
                )
                centres[ns][k, :] = tools.remapAntiHaloCentre(
                    centre,
                    boxsize,
                    swapXZ=swapXZ,
                    reverse=reverse
                )
    return centres

def combine_los_lists(los_lists):
    """
    Combine multiple LOS particle lists into a single per-void list.

    Each input in `los_lists` is expected to be a list of per-void arrays.
    The combined result takes the non-empty list for each void from the 
    most recent list that has valid data.

    NOTE: This function is unused and may be deprecated. Originally used to 
    combine multiple versions of LOS data (e.g., density vs velocity space).

    Parameters:
        los_lists (list of lists): Each element is a list of LOS arrays 
                                   (same length across inputs)

    Returns:
        list: Combined list of LOS arrays (same length as inner lists)

    Raises:
        Exception: If input lists are not all the same length
    """
    lengths = np.array([len(x) for x in los_lists], dtype=int)
    if not np.all(lengths == lengths[0]):
        raise Exception("Cannot combine LOS lists with different lengths.")
    new_list = []
    # Build a matrix of particle counts for each void across the different lists
    num_parts = np.vstack([
        np.array([len(x) for x in los_list], dtype=int)
        for los_list in los_lists
    ]).T  # Shape: (N_voids, N_lists)
    to_use = -np.ones(lengths[0], dtype=int)  # Index of list to use per void
    for k in range(len(los_lists)):
        to_use[num_parts[:, k] > 0] = k  # Use the latest list with valid data
    for k in range(lengths[0]):
        if to_use[k] >= 0:
            new_list.append(los_lists[to_use[k]][k])
        else:
            new_list.append(np.zeros((0, 2)))  # Empty fallback
    return new_list


def get_2d_void_stack_from_los_pos(los_pos, spar_bins, sperp_bins, radii, stacked=True):
    """
    Construct a stacked dataset of LOS positions around void centers, 
    normalized by effective void radius (R_eff).

    Parameters:
        los_pos (list): Per-void list of LOS particle positions (usually 2D: void x LOS array)
        spar_bins (array): Bins along the line of sight (s_parallel)
        sperp_bins (array): Bins perpendicular to LOS (s_perp)
        radii (list of arrays): Effective radii (R_eff) per void
        stacked (bool): If True, return a single combined array of all particles;
                        If False, return nested list of per-void normalized LOS data

    Returns:
        array or list: 
            - If stacked=True: ndarray of shape (N_particles, 2)
            - Else: List of per-void arrays of normalized LOS positions
    """
    # Identify voids with at least one non-empty LOS
    voids_used = [np.array([len(x) for x in los]) > 0 for los in los_pos]
    # Remove empty LOS entries to avoid stacking issues
    los_pos_filtered = [[x for x in los if len(x) > 0] for los in los_pos]
    # Compute cell volumes (not used here directly, possibly for weighting later)
    # cell_volumes_reff = np.outer(np.diff(spar_bins), np.diff(sperp_bins))
    # Filter radii accordingly
    void_radii = [rad[filt] for rad, filt in zip(radii, voids_used)]
    # Normalize LOS positions by R_eff (to work in R-scaled coordinates)
    los_list_reff = [
        [np.abs(los / rad) for los, rad in zip(all_los, all_radii)]
        for all_los, all_radii in zip(los_pos_filtered, void_radii)
    ]
    if stacked:
        # Flatten all voids into one large particle stack
        stacked_particles_reff = np.vstack([np.vstack(los_list) for los_list in los_list_reff])
        return stacked_particles_reff
    else:
        # Return per-void lists of normalized LOS positions
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

def get_weights_for_stack(los_pos, void_radii, additional_weights=None, stacked=True):
    """
    Compute volume-based stacking weights for LOS particles.

    This function was originally designed to apply both:
      - A 1 / R_eff^3 scaling to account for void size (density normalization)
      - Optional signal-to-noise or custom weights

    In later refactors, these were separated into logically distinct steps.
    May be deprecated.

    Parameters:
        los_pos (list): List of LOS particle arrays per void
        void_radii (list): List of R_eff per void
        additional_weights (list or scalar or None): Optional multiplicative weights
        stacked (bool): If True, flatten the weights across all voids

    Returns:
        array or list of arrays:
            - Flattened array if stacked=True
            - Nested weight list matching los_pos structure if stacked=False
    """
    if additional_weights is None:
        v_weight = [
            [(1.0 / rad)**3 * np.ones(len(los))
             for los, rad in zip(all_los, all_radii)]
            for all_los, all_radii in zip(los_pos, void_radii)
        ]
    else:
        if isinstance(additional_weights, list):
            # Per-void weights (possibly per simulation)
            v_weight = [
                [(1.0 / rad)**3 * np.ones(len(los)) * weight
                 for los, rad, weight in zip(all_los, all_radii, all_weights)]
                for all_los, all_radii, all_weights in zip(los_pos, void_radii, additional_weights)
            ]
        else:
            # Scalar weight broadcast across all
            v_weight = [
                [(1.0 / rad)**3 * np.ones(len(los)) * additional_weights
                 for los, rad in zip(all_los, all_radii)]
                for all_los, all_radii in zip(los_pos, void_radii)
            ]
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


def get_2d_fields_per_void(los_per_void, sperp_bins, spar_bins,
                           void_radii, nbar=None):
    """
    Compute 2D density fields for a set of voids from their LOS particle positions.

    The stacking is performed in coordinates rescaled by void radius R_eff,
    so the resulting histograms are corrected to yield densities in physical units.

    This involves:
      - Histogramming each void’s LOS particles into (s_perp, s_par) bins
      - Dividing by the Jacobian factor in cylindrical coordinates (2π * s_perp)
      - Applying a 1/R_eff^3 scaling to convert back to real (Mpc^3) density
      - Optionally normalizing by mean cosmic density nbar to yield 1 + δ

    Parameters:
        los_per_void (list): Per-void list of LOS particle positions, each shape (N, 2)
        sperp_bins (array): Radial bins (perpendicular to LOS)
        spar_bins (array): Bins along LOS
        void_radii (array): R_eff per void (same length as los_per_void)
        nbar (float or None): Mean particle density. If supplied, returns dimensionless density.

    Returns:
        ndarray: 3D array of shape (N_voids, N_spar_bins, N_sperp_bins) representing 2D density fields
    """
    # Compute cell volumes in scaled (R_eff) units
    cell_volumes_reff = np.outer(np.diff(spar_bins), np.diff(sperp_bins))

    # Compute 2D density histograms for each void (weighted by Jacobian correction)
    histograms = np.array([
        np.histogramdd(los, bins=[sperp_bins, spar_bins],
                       density=False,
                       weights=1.0 / (2 * np.pi * los[:, 1]))[0]
        for los in los_per_void
    ])
    # Convert back to physical units (1 / R_eff^3 for scaling back from R_eff units)
    volume_weight = 1.0 / void_radii**3
    # Denominator accounts for volume and optional density normalization
    if nbar is not None:
        denominator = 2 * cell_volumes_reff * nbar
    else:
        denominator = 2 * cell_volumes_reff
    # Final normalized density: (1/R^3) * histogram / volume
    density = volume_weight[:, None, None] * histograms / denominator[None, :, :]
    return density


def get_2d_field_from_stacked_voids(los_per_void, sperp_bins, spar_bins,
                                    void_radii, weights=None, nbar=None):
    """
    Compute the average 2D density field from a collection of voids, 
    each represented by LOS particles.

    This function:
        - Computes a 2D density histogram for each void
        - Averages them (optionally weighted)

    Parameters:
        los_per_void (list): List of per-void particle LOS positions
        sperp_bins (array): Radial bins perpendicular to LOS
        spar_bins (array): Bins along the LOS direction
        void_radii (list): Effective radii (R_eff) for each void
        weights (array or None): Per-void weights for averaging (optional)
        nbar (float or None): Mean particle density for normalization (optional)

    Returns:
        ndarray: Averaged 2D density field
    """
    # Compute per-void 2D density histograms
    density = get_2d_fields_per_void(
        los_per_void, sperp_bins, spar_bins, void_radii, nbar=nbar
    )
    # Average across voids (optionally weighted)
    return np.average(density, axis=0, weights=weights)

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
    comparableDensityMAP = [(delta <= delta_interval[1]) & \
        (delta > delta_interval[0]) for delta in randOverDen]
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


def get_1d_real_space_field(snaps,rbins=None,filter_list=None,
                            additional_weights=None,n_boot = 10000,seed = 42,
                            halo_indices=None,use_precomputed_profiles=True):
    boxsize = snaps.boxsize
    nbar = len(snaps["snaps"][0])/boxsize**3
    # Filter out any invalid halo indices (usually only occurs for
    # BORG catalogues, where anti-halos are missing in some samples):
    if halo_indices is not None:
        if filter_list is None:
            filter_list = [halo_indices[ns] >= 0 for ns in range(0,snaps.N)]
    if filter_list is None:
        filter_list = [np.ones(len(x),dtype=bool) for x in snaps["pair_counts"]]
    # Load from snaps list:
    if use_precomputed_profiles:
        # Use stored pair counts, ignoring specified bins:
        rbins = snaps["radius_bins"][0]
        all_counts = snaps["pair_counts"]
        all_volumes = snaps["bin_volumes"]
    else:
        # Reconstruct counts and volumes from scratch:
        all_counts = []
        all_volumes = []
        if rbins is None:
            # Assume a default set of bins:
            rbins = np.linspace(0,3,31)
        for ns in range(0,borg_snaps.N):
            tree = scipy.spatial.cKDTree(borg_snaps["snaps"][ns]['pos'],
                                         boxsize=boxsize)
            [counts, volumes] = stacking.getPairCounts(\
                snaps["void_centres"][ns],snaps["void_radii"][ns],
                snaps["snaps"][ns],rbins,\
                nThreads=-1,tree=tree,method="poisson",
                vorVolumes=snaps["cell_volumes"][ns])
            all_counts.append(counts)
            all_volumes.append(volumes)
    # Indices of voids to use:
    if halo_indices is not None:
        # Use specified void indices. Note that if halo_indices is provided, 
        # we are assuming that the filter_list does not line up with the 
        # full lists of halos stored in snaps. This happens when the filter
        # list was derived from a void catalogue, for example.
        all_antihalos = [halo_indices[ns][halo_indices[ns] >= 0] - 1
                         for ns in range(0,snaps.N)]
    elif filter_list is not None:
        # Use the supplied boolean array of voids:
        all_antihalos = [np.where(filt)[0] for filt in filter_list]
    else:
        # No filtering, use the entire stack:
        all_antihalos = [np.arange(len(x)) for x in snaps["pair_counts"]]
    # Compute all individual void profiles:
    all_profiles = [counts[ind]/(vols[ind]*nbar) 
                    for counts, vols, ind in 
                    zip(all_counts,all_volumes,all_antihalos)]
    density = np.vstack(all_profiles)
    # Weights for voids:
    if additional_weights is not None:
        additional_weights_per_void = np.hstack([weights[used] 
            for weights, used in zip(additional_weights,filter_list)])
    else:
        additional_weights_per_void = np.ones(density.shape[0])
    # Bootstrap to get (weighted) mean profile and it's error:
    num_voids = np.sum([np.sum(x) for x in filter_list])
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
                                        model))
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
                           backup_start=True,
                           delta_profile = profile_modified_hamaus,
                           Delta_profile = integrated_profile_modified_hamaus):
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
        delta_func = delta_profile
        Delta_func = Delta_profile
        rho_real = lambda *args: delta_profile(*args) + 1.0
    else:
        *profile_params, = get_profile_parameters_fixed(
            ri,delta_i,sigma_delta_i,model=delta_profile)
        delta_func = lambda r: delta_profile(r,*profile_params)
        Delta_func = lambda r: Delta_profile(r,*profile_params)
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
















