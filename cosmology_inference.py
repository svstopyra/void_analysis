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
import emcee
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
            "snaps_reverse": self.snaps_reverse,
            "trees": None,
            "trees_reverse": None
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
                    self.additional_properties["halos"] = [
                        snap.halos() for snap in self.snaps
                    ]
                elif property_name == "antihalos":
                    self.additional_properties["antihalos"] = [
                        snap.halos() for snap in self.snaps_reverse
                    ]
                elif property_name == "trees":
                    self.additional_properties["trees"] = [
                        scipy.spatial.cKDTree(snap['pos'],boxsize=self.boxsize)
                        for snap in self.snaps
                    ]
                elif property_name == "trees_reverse":
                    self.additional_properties["trees_reverse"] = [
                        scipy.spatial.cKDTree(snap['pos'],boxsize=self.boxsize)
                        for snap in self.snaps_reverse
                    ]
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

def Omega_z(z,Om,Ol=None, Ok=0, Or=0,**kwargs):
    """
    Matter density parameter as a function of redshift
    
    Parameters:
        z (float or array): Redshift(s)
        Om (float): Present-day matter density
        (Other parameters as in Ez2)
    
    Returns:
        float or array: \Omega_m(z)
    """
    Ez2_val = Ez2(z, Om, Or=Or, Ok=Ok, Ol=Ol)
    return (Om * (1 + z)**3 / Ez2_val)

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
    return (Omega_z(z,Om,Ol=Ol, Ok=Ok, Or=Or))**gamma

def D1_CPT(z,Omz,Olz = None):
    """
    Carrol, Press, Turner (1992) approximation of the linear growth factor
    
    Parameters:
        z (float or array): Redshift(s)
        Omz (float or array): Omega matter at redshift z
        Oml (float or array): Omega lambda at redshift z (if not given, 
                              computed by assuming flat space)
    
    Returns:
        float: un-normalised linear growth factor.
    """
    if Olz is None:
        Olz = 1.0 - Omz
    a = 1/(1 + z)
    return a*(5*Omz/2)/(Omz**(4/7) - Olz + (1 + Omz/2)*(1 + Olz/70))

def D1(z,Om,Ol=None,Ok = 0,Or = 0,normalised=True,**kwargs):
    """
    Linear growth factor, D_1(z)
    
    Parameters:
        z (float or array): Redshift at which to compute D_1
        Om (float): Omega matter value for the cosmology:
        Ol (float): Omega lambda value (inferred if not given)
        Ok (float): Curvature density (defaults to 0)
        Or (float): Radiation density (defaults to 0)
        normalised (bool): If True, returns D_1(z)/D_1(0)
    """
    if Ol is None:
        Ol = 1 - Om - Ok - Or
    Omz = Omega_z(z,Om,Ol=Ol, Ok=Ok, Or=Or)
    Ez2_val = Ez2(z, Om, Or=Or, Ok=Ok, Ol=Ol)
    Olz = Ol/Ez2_val
    if normalised:
        denom = D1_CPT(0,Om,Ol)
    else:
        denom = 1
    return D1_CPT(z,Omz,Olz)/denom

def get_bin_edges_from_centres(bin_centres,edge_1 = None):
    """
    Computes the edges of the bins for a given list of bin centres. First
    element is computed assuming equally spaced bins, but can be over-ridden if
    this doesn't hold.
    
    Parameters:
        bin_centres (array): centres of the bins
        edge_1 (float or None): first bin edge. If None, computed assuming equal
                                spacing.

    Returns:
        array: Edges of the bins.
    """
    diffs = np.diff(bin_centres)
    bin_edges = np.zeros(len(bin_centres) + 1)
    if edge_1 is None:
        # Assume equal spacing to get the first edge:
        edge_1 = (3/2)*bin_centres[0] - bin_centres[1]/2
    bin_edges[0] = edge_1
    for k in range(len(bin_centres)):
        bin_edges[k+1] = 2*bin_centres[k] - bin_edges[k]
    return bin_edges

def estimate_delta_from_Delta(r,Delta):
    """
    Given a cumulative density, Delta(r), at points r, obtain the local
    density.
    
    Parameters:
        r (array): Points at which Delta is known.
        Delta (array): Delta at r
    
    Returns:
        array: delta(r), where Delta(r) = (3/r)\int_{0}^{r}\delta(r)r^2dr
    """
    rbins = get_bin_edges_from_centres(r,edge_1 = 0)
    Delta_diff = np.diff(rbins)
    result = np.zeros(Delta.shape)
    result[1:] = (r[0:-1]/3)*(Delta[1:] - Delta[0:-1])/Delta_diff[1:]
    result[0] = Delta[0]
    return result

def get_profile_derivative(r,Delta,order,delta=None,deltap = None,deltapp=None,
                           params = None,epsilon = 1e-10):
    """
    Computes the derivative of the function Delta(r), with respect to r.
    
    Parameters:
        r (float or array): Value of r at which to evaluate the derivative.
        Delta(float, array, or function): If float or array, the values of the 
                                          function at r (must be same type/size
                                          as r). If a function, must take r
                                          as an argument.
        order (int): Order of the derivative to compute
        delta (float, array, or function): Local shell density. 
                                           Delta=(3/r^3)\int_{0}^{r}\delta(r)r^2dr
                                           Estimated if not known. 
        deltap (float, array, or function): 1st derivative of delta
        deltapp (float, array, or function): 2nd derivative of delta

    Returns:
        float or array: Derivative of Delta at requested order.
    """
    if order not in [0,1,2,3]:
        raise Exception("Derivative order not implemented or invalid.")
    Delta_vals = Delta(r) if callable(Delta) else Delta
    # 0th derivative:
    if order == 0:
        return Delta_vals
    # 1st derivative:
    if delta is None:
        # Obtain a profile model:
        if params is None:
            params = get_profile_parameters_fixed(
                r,Delta_vals,0.05*Delta_vals,
                model = integrated_profile_modified_hamaus
            )
        delta = lambda r: profile_modified_hamaus(r,*params)
    delta_vals = delta(r)
    if order == 1:
        # Carefully treat r = 0 case:
        return -3*ratio_where_finite(Delta_vals - delta_vals,r,
                                     undefined_value = 0.0)
    # 2nd derivative:
    if deltap is None:
        if params is None:
            params = get_profile_parameters_fixed(
                r,Delta_vals,0.05*Delta_vals,
                model = integrated_profile_modified_hamaus
            )
        deltap = lambda r: profile_modified_hamaus_derivative(r,1,*params)
    if deltapp is None:
        if params is None:
            params = get_profile_parameters_fixed(
                r,Delta_vals,0.05*Delta_vals,
                model = integrated_profile_modified_hamaus
            )
        deltapp = lambda r: profile_modified_hamaus_derivative(r,2,*params)
    deltap_vals = deltap(r)
    if order == 2:
        # Carefully treat r = 0 case:
        if deltap(0) == 0:
            return (12*ratio_where_finite(Delta_vals - delta_vals,r**2,
                                          undefined_value = (3/5)*deltapp(0)) 
                    + 3*ratio_where_finite(deltap_vals,r,
                                           undefined_value = deltapp(0)))
        else:
            return (12*ratio_where_finite(Delta_vals - delta_vals,r**2,
                                          undefined_value = np.inf) 
                    + 3*ratio_where_finite(deltap_vals,r,
                                           undefined_value = np.inf))
    # 3rd derivative:
    deltapp_vals = deltapp(r)
    if order == 3:
        # Carefully treat r = 0 case:
        if deltapp(0) == 0:
            # Rough regularisation:
            Delta_vals = Delta(r + epsilon)
            delta_vals = delta(r + epsilon)
            deltap_vals = deltap(r + epsilon)
            deltapp_vals = deltapp(r + epsilon)
            return (-(60/(r + epsilon)**3)*(Delta_vals - delta_vals) 
                    -15*deltap_vals/(r + epslilon)**2
                    + 3*deltapp_vals/(r + epsilon))
        else:
            return (-60*ratio_where_finite(Delta_vals - delta_vals,r**3,
                                           undefined_value=np.inf)
                    -15*ratio_where_finite(deltap_vals,r**2,
                                           undefined_value=np.inf)
                    +4*ratio_where_finite(deltapp_vals,r,
                                          undefined_value=np.inf))


def get_initial_condition(Delta,order=1,Om=0.3,n2 = -1/143,n3a = -4/275,
                          n3b = -269/17875,use_linear_on_fail=False,**kwargs):
    """
    Solve the equation for the LPT initial conditions. At first and 2nd order
    this can be done analytically, but at 3rd order a cubic must be solved, 
    which we do numerically.
    
    Parameters:
        Delta (float or array): Densities at which to solve for the initial
                                condition.
        order (int): Perturbation order to find the solution for
        Om (float): Matter density parameter
        n2 (float): Exponent for the second roder growth function
        n3a (float): Exponent for the first 3rd-order growth function.
        n3b (float): Exponent for the second 3rd order growth function.
        use_linear_on_fail (bool): If True, defaults to the linear solution
                                   when we fail to find a solution for 2LPT.
                                   Otherwise, generates warnings and produces
                                   nan values as the solution.
    Returns:
        float or array: Solution for S_1r/r at this value of Delta
    """
    if order not in [1,2,3]:
        raise Exception("Perturbation order invalid or not implemented.")
    if order == 1:
        # Linear solution, known analytically:
        D10 = D1(0,Om,**kwargs)
        return -Delta/(3*D10)
    if order == 2:
        # Quadratic solution. This is known analytically, but has the potential
        # to be non-existant. In which case, we shouldn't return an imaginary
        # number, but instead report this to the user, since it indicates
        # that we have gone beyond the bounds where 2LPT is applicable:
        D10 = D1(0,Om,**kwargs)
        D20 = -(3/7)*(Om**n2)*D10**2
        A1 = -3
        A2 = 3*(D10 - D20/D10)
        A0 = -Delta/D10
        discriminant = A1**2 - 4*A2*A0
        # Split between vector and scalar cases:
        if np.isscalar(Delta):
            if discriminant < 0:
                if use_linear_on_fail:
                    return -Delta/(3*D10)
                else:
                    print("2LPT solution for initial conditions " + 
                          "does not exist.")
                    return np.nan
            else:
                # Two solutions exist. In general, only the negative solution
                # will actually be consistent with 3LPT, so we should pick that 
                # one:
                return (-A1 - np.sqrt(discriminant))/(2*A2)
        else:
            have_no_solution = (discriminant < 0)
            solution = np.zeros(Delta.shape)
            if use_linear_on_fail:
                solution[have_no_solution] = -Delta[have_no_solution]/(3*D10)
            else:
                if np.any(have_no_solution):
                    print("Warning: no 2LPT solution for some values of " + 
                          "input. Returning nan for these values.")
                solution[have_no_solution] = np.nan
            have_solution = np.logical_not(have_no_solution)
            solution[have_solution] = (
                -A1 - np.sqrt(discriminant[have_solution])
            )/(2*A2)
            return solution
    if order == 3:
        # Cubic solution. For even vaguely sensible parameters, 
        # there is always a unique solution. We can use the linear solution as 
        # an initial guess and then solve numerically:
        # Compute coefficient of the polynomial:
        D10 = D1(0,Om,**kwargs)
        D20 = -(3/7)*(Om**n2)*D10**2
        D3a0 = -(1/3)*(Om**n3a)*D10**3
        D3b0 = (10/21)*(Om**n3b)*D10**3
        A1 = -3
        A2 = 3*(D10 - D20/D10)
        A0 = -Delta/D10
        A3 = 3*D20 - 3*D10**2 - D3a0/D10 - 6*D3b0/D10
        # Solve numerically:
        guess = -Delta/(3*D10)
        f = lambda u: A3*u**3 + A2*u**2 + A1*u
        if np.isscalar(Delta):
            solution = scipy.optimize.fsolve(lambda u: f(u) + A0,guess)[0]
        else:
            solution = np.array(
                [scipy.optimize.fsolve(lambda u: f(u) + C,x0)[0]
                for C, x0 in zip(A0,guess)]
            )
        return solution

def get_S1r(Delta_r,rval,Om,order=1,n2 = -1/143,n3a = -4/275,n3b = -269/17875,
            correct_ics=True,perturbative_ics = False,**kwargs):
    """
    Compute the spatial part of the first order Lagrangian perturbation, by 
    matching to the provided final density field.
    
    Parameters:
        Delta_r (float or array): Final density to match, as a function of radius
        rval (float or array): Radius at which to compute, must match Delta_r
        Om (float): Matter density parameter
        order (int): Order of Lagrangian perturbation theory being used. Note 
                     that the correction can be different at different orders 
                     due to corrections to the initial conditions.
        n2 (float): Exponent for the second roder growth function
        n3a (float): Exponent for the first 3rd-order growth function.
        n3b (float): Exponent for the second 3rd order growth function.
        correct_ics (bool): If True, apply perturbative corrections to the 
                            initial conditions. If False, same correction at all
                            orders.
        perturbative_ics (bool): If True, attempt to find the ics perturbatively.
                                 If False (default), solve for them, possibly
                                 numerically.
    Returns:
        float or array: Value of S_{1r}
    """
    D10 = D1(0,Om,**kwargs)
    if correct_ics:
        if perturbative_ics:
            # Apply relevant correction terms up to specified order:
            D20 = -(3/7)*(Om**n2)*D10**2
            D3a0 = -(1/3)*(Om**n3a)*D10**3
            D3b0 = (10/21)*(Om**n3b)*D10**3
            if order >= 1:
                S1r = -rval*Delta_r/(3*D10)
            if order >= 2:
                S1r = S1r + rval*(D10**2 - D20)*Delta_r**2/(9*D10**3)
            if order >= 3:
                S1r = S1r + (rval/(27*D10**3))*(
                                   2*D20 - 8*D10**2/3 - 2*D20**2/D10**2
                                   + D3a0/(3*D10) + 2*D3b0/D10)*Delta_r**3
            if order >=4:
                raise Exception("Corrections of order " + str(order) + 
                                " not yet implemented.")
        else:
            # Solve for the initial conditions, possibly numerically:
            S1r = rval*get_initial_condition(Delta_r,order=order,Om=Om,n2 = n2,
                                             n3a = n3a,n3b = n3b,**kwargs)
    else:
        S1r = -rval*Delta_r/(3*D10)
    return S1r

def get_S2r(Delta_r,rval,Om,n2 = -1/143,n3a = -4/275,n3b = -269/17875,order=2,
            correct_ics=True,perturbative_ics = False,S1r=None,**kwargs):
    """
    Compute the spatial part of the second order Lagrangian perturbation, by 
    matching to the provided final density field.
    
    Parameters:
        Delta_r (float or array): Final density to match, as a function of radius
        rval (float or array): Radius at which to compute, must match Delta_r
        Om (float): Matter density parameter
        order (int): Order of Lagrangian perturbation theory being used. Note 
                     that the correction can be different at different orders 
                     due to corrections to the initial conditions.
        n2 (float): Exponent for the second roder growth function
        n3a (float): Exponent for the first 3rd-order growth function.
        n3b (float): Exponent for the second 3rd order growth function.
        correct_ics (bool): If True, apply perturbative corrections to the 
                            initial conditions. If False, same correction at all
                            orders.
    Returns:
        float or array: Value of S_{2r}
    """
    if correct_ics:
        D10 = D1(0,Om,**kwargs)
        if perturbative_ics:
            # Estimate the initial conditions perturbatively:
            D20 = -(3/7)*(Om**n2)*D10**2
            if order >= 2:
                S2r = rval*Delta_r**2/(9*D10**2)
            if order >= 3:
                S2r = S2r - rval*(3*D10**2 - 2*D20)*Delta_r**3/(27*D10**4)
        else:
            # Solve for initial conditions (numerically if 3rd order):
            if S1r is None:
                S1r = get_S1r(Delta_r,rval,Om,order=order,n2 = n2,n3a = n3a,
                              n3b = n3b,correct_ics=True,
                              perturbative_ics = False,**kwargs)
            S2r = ratio_where_finite(S1r**2,rval,undefined_value=0.0)
            if order == 3:
                S2r = S2r + D10*ratio_where_finite(S1r**3,rval**2,
                                                   undefined_value=0.0)
    else:
        S2r = (rval/9)*Delta_r**2
    return S2r

def get_S3r(Delta_r,rval,Om,order=3,correct_ics=True,perturbative_ics = False,
            S1r = None,**kwargs):
    """
    Compute the spatial part of the first order Lagrangian perturbation, by 
    matching to the provided final density field.
    
    Note: currently no higher order corrections, so changing order doesn't
    alter anything. However, we retain it in case we want to extend to higher
    orders, which would entail futher corrections.
    
    Parameters:
        Delta_r (float or array): Final density to match, as a function of radius
        rval (float or array): Radius at which to compute, must match Delta_r
        Om (float): Matter density parameter
        order (int): Order of Lagrangian perturbation theory being used. Note 
                     that the correction can be different at different orders 
                     due to corrections to the initial conditions.
                     
    Returns:
        S3ar (float or array): Value of S_{3ar}
        S3br (float or array): Value of S_{3br}
    """
    if order >=4:
        raise Exception("Corrections of order " + str(order) + 
                        " not yet implemented.")
    D10 = D1(0,Om,**kwargs)
    if correct_ics and not perturbative_ics:
        if S1r is None:
            S1r = get_S1r(Delta_r,rval,Om,order=order,n2 = n2,n3a = n3a,
                          n3b = n3b,correct_ics=True,
                          perturbative_ics = False,**kwargs)
        S3ar = ratio_where_finite(S1r**3,3*rval**2,undefined_value=0.0)
        S3br = ratio_where_finite(2*S1r**3,rval**2,undefined_value=0.0)
    else:
        S3ar = -(rval/(81*D10**3))*Delta_r**3
        S3br = - (2*rval/(27*D10**3))*Delta_r**3
    return S3ar, S3br

def get_psi_n_r(Delta_r,rval,n,z=0,Om=0.3,order=None,n2 = -1/143,
                n3a = -4/275,n3b = -269/17875,S1r=None,**kwargs):
    """
    Compute the nth order correction to the displacement field.
    
    Parameters:
        Delta_r (float or array): Final density to match, as a function of radius
        rval (float or array): Radius at which to compute, must match Delta_r
        n (int): Order of the correction to compute.
        z (float): Redshift at which to compute the displacement field.
        Om (float): Value of Omega_matter
        order (int): Order of perturbation theory being applied. If not 
                     supplied, assumed to be equal to n. This can matter, if we
                     enable correct_ics = True, since the perturbations change
                     with each order.
        n2 (float): Exponent of Omega_m(z) used to approximate D_2(t)
        n3a (float): Exponent of Omega_m(z) used to approximate D_{3a}(t)
        n3b (float): Exponent of Omega_m(z) used to approximate D_{3b}(t)
        kwargs: Keyword arguments passed to get_S1r, get_S2r, get_S3r functions.

    Return:
        float or array: Correction to the displacement field of order n
    """
    if order is None:
        order = n
    if order not in [1,2,3] or n not in [1,2,3]:
        raise Exception("Perturbation order invalid or not implemented.")
    if n > order:
        raise Exception("n <= order is required.")
    if n >= 1:
        D1_val = D1(z,Om,**kwargs)
        if S1r is None:
            S1r = get_S1r(Delta_r,rval,Om,order=order,n2 = n2,n3a = n3a,
                          n3b = n3b,**kwargs)
        if n == 1:
            return D1_val*S1r
    if n >= 2:
        Omz = Omega_z(z,Om,**kwargs)
        D2_val = -(3/7)*(Omz**n2)*D1_val**2
        S2r = get_S2r(Delta_r,rval,Om,order=order,n2=n2,n3a = n3a,
                      n3b = n3b,S1r=S1r,**kwargs)
        if n == 2:
            return D2_val*S2r
    if n >=3:
        D3a_val = -(1/3)*(Omz**n3a)*D1_val**3
        D3b_val = (10/21)*(Omz**n3b)*D1_val**3
        S3ar, S3br = get_S3r(Delta_r,rval,Om,order=order,S1r=S1r,**kwargs)
        if n == 3:
            return D3a_val*S3ar + D3b_val*S3br

def get_delta_lpt(Delta_r,z=0,Om=0.3,order=1,return_all=False,**kwargs):
    """
    Compute the final density field estimate in perturbation theory, given
    the input final density field. This is mainly used to aid in verifying the
    accuracy of the result
    
    Parameters:
        Delta_r (float or array): Final density to match, as a function of radius
        z (float): Redshift at which to compute the displacement field.
        Om (float): Value of Omega_matter
        order (int): Order of expansion for LPT. Only 1,2, or 3 implemented
        return_all (bool): Default False (return only the density estimate). If
                           True, return the corrections to the density
                           with for each order of pertubation theory
                           separately
    Returns:
        float or array (return_all = False): Final density field estimate
        3 floats or arrays (return_all = True): Perturbative corrections to the
                                                density field at each order
    """
    if order not in [1,2,3]:
        raise Exception("Perturbation order invalid or not implemented.")
    # Ratio of Psi_r/r:
    if not return_all:
        Psi_r_rat = spherical_lpt_displacement(1.0,Delta_r,order=order,
                                               fixed_delta=True,Om=Om,**kwargs)
        # Exact spherical result for the density field
        return -3*(Psi_r_rat - Psi_r_rat**2 + Psi_r_rat**3)
    else:
        # Displacement field corrections/r at each order:
        S1r = get_S1r(Delta_r,1.0,Om,order=order,**kwargs) # Precompute 
        Psi_r1 = get_psi_n_r(Delta_r,1.0,1,z=z,Om=Om,order=order,S1r=S1r,
                             **kwargs)
        Psi_r2 = get_psi_n_r(Delta_r,1.0,2,z=z,Om=Om,order=order,S1r=S1r,
                             **kwargs)
        Psi_r2_un = get_psi_n_r(Delta_r,1.0,2,z=z,Om=Om,order=2,S1r=S1r,
                             **kwargs)
        Psi_r3 = get_psi_n_r(Delta_r,1.0,3,z=z,Om=Om,order=order,S1r=S1r,
                             **kwargs)
        # Perturbative corrections, order by order to the density field:
        Delta_1 = -3*Psi_r1
        Delta_2 = -3*Psi_r2 + 3*Psi_r1**2
        Delta_3 = -3*Psi_r3 + 6*Psi_r1*Psi_r2_un - 3*Psi_r1**3
        return Delta_1, Delta_2, Delta_3

def spherical_lpt_displacement(r,Delta,order=1,z=0,Om=0.3,
                               n2 = -1/143,nf1 = 5/9,
                               nf2 = 6/11,n3a = -4/275,n3b = -269/17875,
                               nf3a = 13/24,nf3b = 13/24,fixed_delta = False,
                               radial_fraction = False,correct_ics = True,
                               **kwargs):
    """
    Compute the radial component of the displacement field, in Lagrangian 
    perturbation theory, assuming spherical symmetry for the density field.
    
    Parameters:
        r (float or array): Radius/radii at which to compute the displacement
        Delta (scalar function): Final density field, as a function of r
        order (int): Order of expansion for LPT. Only 1,2, or 3 implemented
        z (float): Redshift at which to compute the displacement field.
        Om (float): Value of Omega_matter
        n2 (float): Exponent of Omega_m(z) used to approximate D_2(t)
        nf1 (float): Exponent of Omega_m(z) used to approximate f_1(t)
        nf2 (float): Exponent of Omega_m(z) used to approximate f_2(t)
        n3a (float): Exponent of Omega_m(z) used to approximate D_{3a}(t)
        n3b (float): Exponent of Omega_m(z) used to approximate D_{3b}(t)
        nf3a (float): Exponent of Omega_m(z) used to approximate f_{3a}(t)
        nf3b (float): Exponent of Omega_m(z) used to approximate f_{3b}(t)
        radial_fraction (bool): If True, compute Psi_r/r, rather than Psi_r 
        fixed_delta (bool): If True, Delta is assumed to be a pre-computed
                            array, rather than a function.
        correct_ics (bool): If True, perturbative update the initial conditions
                            (used when Delta is the final density)

    Returns:
        float or array: Radial component of the displacement field.
    """
    if order not in [1,2,3]:
        raise Exception("Perturbation order invalid or not implemented.")
    if radial_fraction:
        rval = 1.0
    else:
        rval = r
    Delta_r = Delta if fixed_delta else Delta(r)
    # 1st order estimate of Psi_r:
    S1r = get_S1r(Delta_r,rval,Om,order=order,**kwargs) # Precompute 
    Psi_r1 = get_psi_n_r(Delta_r,rval,1,order=order,n2=n2,n3a=n3a,n3b=n3b,
                         Om=Om,z=z,correct_ics=correct_ics,S1r=S1r,**kwargs)
    Psi_r = Psi_r1
    if order == 1:
        return Psi_r
    # 2nd order estimate of Psi_r:
    Psi_r2 = get_psi_n_r(Delta_r,rval,2,order=order,n2=n2,n3a=n3a,n3b=n3b,
                         Om=Om,z=z,correct_ics=correct_ics,S1r=S1r,**kwargs)
    Psi_r = Psi_r + Psi_r2
    if order == 2:
        return Psi_r
    # 3rd order estimate of Psi_r:
    Psi_r3 = get_psi_n_r(Delta_r,rval,3,order=order,n2=n2,n3a=n3a,n3b=n3b,
                         Om=Om,z=z,correct_ics=correct_ics,S1r=S1r,**kwargs)
    Psi_r = Psi_r + Psi_r3
    return Psi_r

def spherical_lpt_velocity(r,Delta,order=1,z=0,Om=0.3,
                               n2 = -1/143,nf1 = 5/9,
                               nf2 = 6/11,n3a = -4/275,n3b = -269/17875,
                               nf3a = 13/24,nf3b = 13/24,h=1.0,
                               radial_fraction = False,fixed_delta = True,
                               correct_ics = True,**kwargs):
    """
    Compute the radial component of the velocity field, in Lagrangian 
    perturbation theory, assuming spherical symmetry for the density field.
    
    Parameters:
        r (float or array): Radius/radii at which to compute the velocity. If
                            h = 1, assumes units of Mpc/h
        Delta (scalar function): Final density field, as a function of r
        order (int): Order of expansion for LPT. Only 1,2, or 3 implemented
        z (float): Redshift at which to compute the velocity field.
        Om (float): Value of Omega_matter
        n2 (float): Exponent of Omega_m(z) used to approximate D_2(t)
        nf1 (float): Exponent of Omega_m(z) used to approximate f_1(t)
        nf2 (float): Exponent of Omega_m(z) used to approximate f_2(t)
        n3a (float): Exponent of Omega_m(z) used to approximate D_{3a}(t)
        n3b (float): Exponent of Omega_m(z) used to approximate D_{3b}(t)
        nf3a (float): Exponent of Omega_m(z) used to approximate f_{3a}(t)
        nf3b (float): Exponent of Omega_m(z) used to approximate f_{3b}(t)
        h (float): Dimensionless Hubble rate. Defaults to 1.0, which means we
                   assume distances are in Mpc/h. Otherwise, non-zero h means
                   distances are expected to be in Mpc.
        radial_fraction (bool): If True, compute v/r, rather than v
        fixed_delta (bool): If True, Delta is assumed to be a pre-computed
                            array, rather than a function.
        correct_ics (bool): If True, perturbative update the initial conditions
                            (used when Delta is the final density)
    
    Returns:
        float or array: Radial component of the velocity field.
    """
    if order not in [1,2,3]:
        raise Exception("Perturbation order invalid or not implemented.")
    D1_val = D1(z,Om,**kwargs)
    Delta_r = Delta if fixed_delta else Delta(r)
    a = 1/(1+z)
    H = Hz(z,Om,h=h,**kwargs)
    Omz = Omega_z(z,Om,**kwargs)
    if radial_fraction:
        rval = 1.0/h # Divide by h, so that velocity is always km/s
    else:
        rval = r
    # 1st order estimate of v_r:
    f1 = Omz**nf1
    S1r = get_S1r(Delta_r,rval,Om,order=order,n2=n2,n3a=n3a,n3b=n3b,
                  correct_ics=correct_ics,**kwargs)
    v_r = a*H*f1*D1_val*S1r
    if order == 1:
        return v_r
    # 2nd order estimate of v_r:
    D2_val = -(3/7)*(Omz**n2)*D1_val**2
    f2_val = 2*(Omz**nf2)
    S2r = get_S2r(Delta_r,rval,Om,order=order,n2=n2,n3a=n3a,n3b=n3b,
                  correct_ics=correct_ics,S1r = S1r,**kwargs)
    v_r = v_r + a*H*f2_val*D2_val*S2r
    if order == 2:
        return v_r
    # 3rd order estimate of v_r:
    f3a = 3*(Omz**nf3a)
    f3b = 3*(Omz**nf3b)
    D3a_val = -(1/3)*(Omz**n3a)*D1_val**3
    D3b_val = (10/21)*(Omz**n3b)*D1_val**3
    S3ar, S3br = get_S3r(Delta_r,rval,Om,order=order,S1r=S1r,**kwargs)
    v_r = (v_r + a*H*f3a*D3a_val*S3ar + a*H*f3b*D3b_val*S3br)
    return v_r


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
    Da = np.fmax(cosmo_test.angular_diameter_distance(z).value,1e-16)
    Da_fid = np.fmax(cosmo_fid.angular_diameter_distance(z).value,1e-16)
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


def get_dudr_hz_o1pz(Delta, delta, r_par, r_perp, f, reg_factor = 1e-12):
    """
    Compute the derivative of the los velocity with respect to r_par,
    multiplied by H(z)/(1+z). The advantage of this is that it is free from
    explicit dependence on z and Om (only depending on these via the 
    void profiles, \Delta(r) and \delta(r)).
    
    Parameters:
        Delta (function): Cumulative density profile Δ(r)
        delta (function): Local density contrast δ(r)
        r_par (float): LOS distance
        r_perp (float): Transverse distance
        f (float or None): Growth rate

    Returns:
        float: Derivative of velocity w.r.t. r_par
    """
    r = np.sqrt(r_par**2 + r_perp**2)
    Dr = Delta(r)
    dr = delta(r)
    return -(f / 3.0) * Dr - f * (r_par / (r + reg_factor))**2 * (dr - Dr)

def void_los_velocity_derivative(z, Delta, delta, r_par, r_perp, Om, f=None, 
                                 reg_factor = 1e-12, **kwargs):
    """
    Compute the derivative of the LOS velocity with respect to r_par.

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
        float: Derivative of velocity w.r.t. r_par
    """
    hz = Hz(z, Om, **kwargs)
    if f is None:
        f = f_lcdm(z, Om, **kwargs)
    return (hz / (1.0 + z))*get_dudr_hz_o1pz(Delta,delta,r_par,r_perp,f,
                                             reg_factor=reg_factor)


def z_space_jacobian(Delta, delta, r_par, r_perp, f=None, z = 0, Om=0.3,
                     Om_fid=0.3111,linearise_jacobian=True, **kwargs):
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
    if f is None:
        f = f_lcdm(z, Om, **kwargs)
    dudr_hz_o1pz = get_dudr_hz_o1pz(Delta,delta,r_par,r_perp,f,
                                    reg_factor=reg_factor)
    if linearise_jacobian:
        return 1.0 - dudr_hz_o1pz
    else:
        return 1.0 / (1.0 + dudr_hz_o1pz)


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


def iterative_zspace_inverse_scalar(s_par, s_perp, f, Delta, N_max = 5, atol=1e-5, rtol=1e-5):
    """
    Numerically invert the redshift-space LOS coordinate to estimate
    the real-space coordinate (r_par), assuming the linear-theory model.

    Parameters:
        s_par (float): LOS coordinate in redshift space
        s_perp (float): Perpendicular co-ordinate in refshift space
        f (float): Linear growth rate
        Delta (function): Cumulative density profile
        N_max (int): Maximum number of iterations
        atol (float): Absolute tolerance for convergence
        rtol (float): Relative tolerance for convergence

    Returns:
        float: Estimated r_par
    """
    r_par_guess = s_par
    r_perp = s_perp
    for _ in range(N_max):
        r = np.sqrt(r_par_guess**2 + r_perp**2)
        r_par_new = s_par / (1.0 - (f / 3.0) * Delta(r))
        if np.abs(r_par_new - r_par_guess) < atol or \
           np.abs(r_par_new / r_par_guess - 1.0) < rtol:
            break
        r_par_guess = r_par_new
    return r_par_guess

def iterative_zspace_inverse(s_par, s_perp, f, Delta, N_max=5, atol=1e-5, rtol=1e-5):
    """
    Array-compatible wrapper for iterative_zspace_inverse_scalar.

    Parameters:
        s_par (float or np.ndarray): LOS redshift-space coordinate(s)
        s_perp (float or np.ndarray): Perpendicular coordinate(s)
        f (float): Linear growth rate
        Delta (function): Cumulative density profile
        N_max (int): Max iterations
        atol (float): Absolute convergence tolerance
        rtol (float): Relative convergence tolerance

    Returns:
        np.ndarray or float: Estimated real-space LOS coordinate(s)
    """
    s_par = np.asarray(s_par)
    s_perp = np.asarray(s_perp)
    if s_par.shape != s_perp.shape:
        raise ValueError("s_par and s_perp must have the same shape")
    # Scalar input case
    if s_par.ndim == 0:
        return iterative_zspace_inverse_scalar(s_par, s_perp, f, Delta,
                                               N_max=N_max, atol=atol, rtol=rtol)
    # Vectorized case (apply element-wise)
    return np.array([
        iterative_zspace_inverse_scalar(sp, sp_perp, f, Delta,
                                        N_max=N_max, atol=atol, rtol=rtol)
        for sp, sp_perp in zip(s_par.flat, s_perp.flat)]
    ).reshape(s_par.shape)


def to_real_space(s_par, s_perp, Delta=None, u_par=None, f=None,z=0, Om=0.3, 
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
            if Delta is None:
                raise ValueError("Delta profile must be supplied for linear inversion.")
            # Use helper to handle the iterative logic
            r_par = iterative_zspace_inverse(s_par, s_perp, f, Delta, N_max,
                                             atol = atol, rtol = rtol)
        else:
            # Use tabulated inverse function
            r_par = F_inv(s_perp, s_par, f * np.ones(np.shape(s_perp)))
    else:
        # Use directly supplied LOS peculiar velocity
        hz = Hz(z, Om, **kwargs)
        r_par = s_par - (1.0 + z) * u_par / hz
    return [r_par, r_perp]

def ratio_where_finite(x,y,undefined_value = 0.0):
    """
    Return the ratio of x and y, except where y is zero, in which case
    return the specified undefined value (default 0.0)
    
    Parameters:
        x (float or array): Numerator(s)
        y (float or array): Denominator(s)
        undefined_value (default 0.0): Value to use for ratio when y is zero.
    
    Returns:
        ratio (float or array), same shape as x/y
    """
    if np.isscalar(y):
        if y == 0.0:
            return undefined_value
        else:
            return x/y
    else:
        res = np.full(y.shape,undefined_value)
        nz = (y != 0)
        if np.isscalar(x):
            res[nz] = x/y[nz]
        else:
            res[nz] = x[nz]/y[nz]
        return res

def product_where_finite(x,y,undefined_value = 0.0):
    """
    Return the product of x and y, except where x or y is infinite, in which case
    return the specified undefined value (default 0.0)
    
    Parameters:
        x, y (float or array): Values to multiply
        undefined_value (default 0.0): Value to use for product when x or y is
            not finite.
    
    Returns:
        ratio (float or array), same shape as x*y
    """
    if np.isscalar(x):
        if np.isscalar(y):
            if np.isfinite(x) and np.isfinite(y):
                return x*y
            else:
                return undefined_value
        else:
            if np.isfinite(x):
                res = np.full(y.shape,undefined_value)
                finite = np.isfinite(y)
                res[finite] = x*y[finite]
                return res
            else:
                return np.full(y.shape,undefined_value)
    else:
        if np.isscalar(y):
            if np.isfinite(y):
                res = np.full(x.shape,undefined_value)
                finite = np.isfinite(x)
                res[finite] = x[finite]*y
                return res
            else:
                return undefined_value
        else:
            finite_x = np.isfinite(x)
            finite_y = np.isfinite(y)
            res = np.full(y.shape,undefined_value)
            finite = finite_x & finite_y
            res[finite] = x[finite]*y[finite]
            return res

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
    mus = ratio_where_finite(s_par,s,undefined_value=0.0)
    # Distance correction factor
    # This accounts for distortions in radial vs. transverse directions
    mus_rec = ratio_where_finite(1.0,mus,undefined_value=np.inf)
    s_factor = np.sqrt(1.0 + epsilon**2 * (mus_rec**2 - 1.0))
    # Apply AP correction to total distance and angle
    s_new = product_where_finite(s * mus * epsilon**(-2.0 / 3.0), s_factor,
                                 undefined_value = 0.0)
    # Corrected cosine of angle
    mus_new = ratio_where_finite(np.sign(mus) , s_factor)
    # Compute corrected coordinates
    s_par_new = mus_new * s_new
    s_perp_new = np.sign(s_perp) * s_new * np.sqrt(1.0 - mus_new**2)
    return s_par_new, s_perp_new


def z_space_profile(s_par, s_perp, rho_real, Delta, delta, f=None,z=0, Om=0.3,
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
    r_par, r_perp = to_real_space(s_par_new, s_perp_new, f=f, z=z, Om=Om, 
                                  Delta=Delta, **kwargs)
    # Step 3: Compute the Jacobian of the transformation (∂r/∂s)
    jacobian = z_space_jacobian(Delta, delta, r_par, r_perp, Om=Om,z=z,f=f,
                                **kwargs)
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
    diff = data.T - mean[:,None]
    cov = np.matmul(diff,diff.T)/N
    return cov

# Estimate covariance of the profile using the jackknife method:
def profile_jackknife_covariance(data,profile_function,**kwargs):
    N, M = data.shape
    jacknife_data = np.zeros((N,M))
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
            F_inv = lambda x, y, z: scipy.interpolate.interpn(
                                        (rperp_vals,svals),rvals,
                                        np.vstack((x,y)).T,method='cubic')
            theory_val = z_space_profile(s_par_new,s_perp_new,
                                         lambda r: rho_real(r,*profile_args),
                                         Delta_func,delta_func,f=f,z=z,Om=Om,
                                         F_inv=F_inv,**kwargs)
        else:
            theory_val = z_space_profile(s_par_new,s_perp_new,
                                         lambda r: rho_real(r,*profile_args),
                                         Delta_func,delta_func,f=f,z=z,Om=Om,
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
                                         lambda r: rho_real(r,*profile_args),
                                         Delta_func,delta_func,f=f,z=z,Om=Om,
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
        z_space_profile(sp, st, rho_func, Delta_func, delta_func,f=f,
                        z=z, Om=Om, epsilon=epsilon,
                        apply_geometry=sample_epsilon,
                        F_inv=F_inv,
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

def tikhonov_regularisation(cov_matrix, lambda_reg=1e-10):
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
    return cov_matrix + lambda_reg * np.eye(cov_matrix.shape[0])

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
    inv_cov = np.matmul(P.T, P)                  # Reconstruct full inverse
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


def get_field_from_los_data(los_data, spar_bins, sperp_bins, v_weight,
                            void_count, nbar=None):
    """
    Bin all LOS particles into a 2D histogram to compute the stacked void density field.

    Includes a cylindrical Jacobian factor and optional normalization by cosmic mean density.

    Parameters:
        los_data (ndarray): Array of shape (N_particles, 2) with (s_par, s_perp) values
        spar_bins (array): Bin edges along LOS direction
        sperp_bins (array): Bin edges perpendicular to LOS
        v_weight (array): Weights for each particle (same length as los_data)
        void_count (int): Total number of contributing voids
        nbar (float or None): Mean number density. If supplied, normalizes output.

    Returns:
        ndarray: 2D stacked density field (shape: [N_spar_bins, N_sperp_bins])
    """
    cell_volumes_reff = np.outer(np.diff(spar_bins), np.diff(sperp_bins))
    hist = np.histogramdd(
        los_data,
        bins=[spar_bins, sperp_bins],
        density=False,
        weights=v_weight / (2 * np.pi * los_data[:, 1])
    )[0]
    if nbar is not None:
        return hist / (2 * void_count * cell_volumes_reff * nbar)
    else:
        return hist / (2 * void_count * cell_volumes_reff)


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

def profile_broken_power_log(r, A, r0, c1, f1, B):
    """
    Logarithmic version of a broken power-law void density profile.

    Used internally to simplify fitting and enforce positivity.

    Form:
        log(ρ/ρ̄) = log|A + B(r/r₀)² + (r/r₀)^4| + ((c₁ - 4)/f₁) * log(1 + (r/r₀)^f₁)

    Parameters:
        r (float or array): Radial distance
        A, B (float): Amplitude coefficients
        r0 (float): Characteristic radius
        c1, f1 (float): Shape parameters

    Returns:
        float or array: Logarithm of the void density profile
    """
    return np.log(np.abs(A + B * (r / r0)**2 + (r / r0)**4)) + \
           ((c1 - 4) / f1) * np.log(1 + (r / r0)**f1)

def profile_broken_power(r, A, r0, c1, f1, B):
    """
    Broken power-law profile for void density.

    This is the exponential of profile_broken_power_log, ensuring a positive density.

    Parameters:
        r (float or array): Radial distance
        A, B (float): Amplitude coefficients
        r0 (float): Characteristic radius
        c1, f1 (float): Shape parameters

    Returns:
        float or array: Density contrast profile δ(r)
    """
    return np.exp(profile_broken_power_log(r, A, r0, c1, f1, B))

def profile_modified_hamaus(r, alpha, beta, rs, delta_c, delta_large=0.0, rv=1.0):
    """
    Modified Hamaus et al. (2014) void density profile.

    Extends the Hamaus profile by adding a constant density offset (delta_large)
    to allow for large-scale compensation or uniform overdensity.

    Parameters:
        r (float or array): Radius
        alpha, beta (float): Shape parameters
        rs (float): Void scale radius
        delta_c (float): Central density contrast
        delta_large (float): Large-scale offset (default: 0)
        rv (float): Characteristic void radius (default: 1.0)

    Returns:
        float or array: Density contrast δ(r)
    """
    return ((delta_c - delta_large) * (1.0 - (r / rs)**alpha) /
            (1.0 + (r / rv)**beta)) + delta_large

def profile_modified_hamaus_derivative(r,order,alpha,beta,rs,delta_c,
                                       delta_large=0,rv=1.0):
    """
    Returns derivatives of profile_modified_hamaus
    
    Parameters:
        r (float or array): Radius
        order (int): order of the derivative to return
        Other parameters as in profile_modified_hamaus
    
    Returns:
        float or array: Derivative at r
    """
    if order not in [0,1,2]:
        raise Exception("Derivative order not valid or not implemented.")
    if order == 0:
        return profile_modified_hamaus(r,alpha,beta,rs,delta_c,
                                       delta_large=delta_large,rv=rv)
    numerator = 1 - (r/rs)**alpha
    denominator = 1 + (r/rv)**beta
    extra_factor = ( (alpha/rs)*(r/rs)**(alpha-1)/numerator
                    +(beta/rv)*(r/rv)**(beta-1)/denominator)
    if order == 1:
        delta = profile_modified_hamaus(r,alpha,beta,rs,delta_c,
                                       delta_large=delta_large,rv=rv)
        return -(delta - delta_large)*extra_factor
    if order == 2:
        deltap = profile_modified_hamaus_derivative(r,1,alpha,beta,rs,delta_c,
                                                    delta_large=delta_large,
                                                    rv=rv)
        return -deltap*extra_factor - (delta - delta_large)*(
            (alpha*(alpha-1)/rs**2)*(r/rs)**(alpha-2)/numerator
            + (beta*(beta-1)/rv**2)*(r/rv)**(beta-2)/denominator
            + (alpha/rs)**2*(r/rs)**(2*alpha - 2)/numerator**2
            - (beta/rv)**2*(r/rv)**(2*beta-2)/denominator**2)



def integrated_profile_modified_hamaus(r, alpha, beta, rs, delta_c,
                                       delta_large=0.0, rv=1.0):
    """
    Integrated (cumulative) version of the modified Hamaus profile.

    Computes the average density contrast Δ(r), i.e., the mass enclosed
    within a sphere of radius r divided by the volume.

    Uses hypergeometric functions to perform analytic integration of δ(r).

    Parameters:
        r (float or array): Radius
        alpha, beta (float): Shape parameters
        rs (float): Void scale radius
        delta_c (float): Central density contrast
        delta_large (float): Large-scale offset (default: 0)
        rv (float): Characteristic void radius (default: 1.0)

    Returns:
        float or array: Cumulative density contrast Δ(r)
    """
    arg = (r / rv)**beta
    hyp_1 = scipy.special.hyp2f1(1,3 / beta, 1 + (3 / beta), -arg)
    hyp_2 = scipy.special.hyp2f1(1,(alpha + 3) / beta, 1 + ( (alpha + 3) / beta), -arg)
    return delta_large + (delta_c - delta_large)*( hyp_1 - 
        ( 3 / (alpha + 3) ) * ( r / rs )**alpha * hyp_2 )



def rho_real(r, *profile_args):
    """
    Wrapper function for the real-space density profile.

    Currently hardcoded to use the modified Hamaus profile. Could be 
    generalized to allow switching between profile models.

    Parameters:
        r (float or array): Radius
        *profile_args: Arguments passed to profile_modified_hamaus

    Returns:
        float or array: Real-space density profile ρ(r)
    """
    return profile_modified_hamaus(r, *profile_args)


def get_weights(los_zspace, void_radii, additional_weights=None):
    """
    Compute per-void stacking weights for redshift-space LOS data.

    This combines:
      - 1 / R_eff^3 volume-based scaling (via get_weights_for_stack)
      - Optional user-supplied weights (e.g. inverse variance, reproducibility)

    Parameters:
        los_zspace (list): Per-snapshot list of LOS arrays (per void)
        void_radii (list): Per-snapshot list of void radii
        additional_weights (list or None): Optional weights per void (same structure)

    Returns:
        Array of weights for all particles in all voids in all snapshots
    """
    # Mask voids that actually contribute LOS data
    voids_used = [np.array([len(x) for x in los]) > 0 for los in los_zspace]
    # Remove empty entries
    los_pos = [[los[i] for i in np.where(ind)[0]] for los, ind in zip(los_zspace, voids_used)]
    # Prepare additional weights
    if additional_weights is None:
        weights_list = None
    else:
        all_additional_weights = np.hstack([
            weights[used] for weights, used in zip(additional_weights, voids_used)
        ])
        weights_list = [
            weights[used] / np.sum(all_additional_weights)
            for weights, used in zip(additional_weights, voids_used)
        ]
    # Volume × optional weighting
    v_weight = get_weights_for_stack(
        los_pos,
        [radii[used] for radii, used in zip(void_radii, voids_used)],
        additional_weights=weights_list
    )
    return v_weight


def get_halo_indices(catalogue):
    """
    Map a void catalogue to halo indices in simulation snapshots.

    Only needed if index mapping isn't already handled by the catalogue class.

    NOTE: Deprecated in favor of direct access through catalogue object.

    Parameters:
        catalogue: Void catalogue with finalCatFrac + indexListShort

    Returns:
        list of arrays: Per-snapshot list of halo indices
    """
    final_cat = catalogue.get_final_catalogue(void_filter=True)
    halo_indices = [-np.ones(len(final_cat), dtype=int) for _ in range(catalogue.numCats)]
    for ns in range(catalogue.numCats):
        have_void = final_cat[:, ns] >= 0
        halo_indices[ns][have_void] = catalogue.indexListShort[ns][
            final_cat[have_void, ns] - 1
        ]
    return halo_indices


#void_radii = catalogue.getMeanProperty("radii",void_filter=True)[0]
#rep_scores = catalogue.property_with_filter(
#    catalogue.finalCatFrac,void_filter=True)

def trim_los_list(los_list, spar_bins, sperp_bins, all_radii):
    """
    Remove voids that have no LOS particles within the stacking volume.

    This filters the LOS list using the (s_par, s_perp) bins and void radii,
    and returns:
      - Trimmed list of LOS particles per void
      - Boolean flags for which voids were retained

    Parameters:
        los_list (list): Per-snapshot list of LOS particle arrays (per void)
        spar_bins (array): LOS bin edges
        sperp_bins (array): Transverse bin edges
        all_radii (list): Per-snapshot list of void radii

    Returns:
        tuple:
            - los_list_trimmed (list of lists): Filtered LOS particle arrays
            - voids_used (list of bool arrays): Mask of retained voids
    """
    los_list_trimmed = get_2d_void_stack_from_los_pos(
        los_list, spar_bins, sperp_bins,
        [all_radii[ns] for ns in range(len(los_list))],
        stacked=False
    )
    voids_used = [np.array([len(x) > 0 for x in los]) for los in los_list]
    return los_list_trimmed, voids_used

def get_trimmed_los_list_per_void(los_pos, spar_bins, sperp_bins, void_radii_list):
    """
    Get a flat list of trimmed LOS particle arrays from all voids.

    This wraps `trim_los_list()` and flattens the result to a single
    per-void list, across all snapshots.

    Parameters:
        los_pos (list): Per-snapshot list of LOS arrays (per void)
        spar_bins (array): LOS bin edges
        sperp_bins (array): Transverse bin edges
        void_radii_list (list): Per-snapshot list of void radii

    Returns:
        list: Flattened list of LOS particle arrays per void (trimmed)
    """
    los_list_trimmed, _ = trim_los_list(
        los_pos, spar_bins, sperp_bins, void_radii_list
    )
    return sum(los_list_trimmed, [])

def get_borg_density_estimate(snaps, densities_file=None, dist_max=135,
                              seed=1000, interval=0.68):
    """
    Estimate the density contrast in a specified subvolume using BORG snapshots.

    If precomputed density samples are provided, loads them from file.
    Otherwise, computes them from raw snapshot data using spherical averaging.

    Returns a bootstrap estimate of the MAP (maximum a posteriori) density contrast
    and its uncertainty interval.

    Parameters:
        snaps (SnapHandler): Object containing snapshots in `snaps["snaps"]`
        densities_file (str or None): Pickle file containing precomputed delta samples
        dist_max (float): Radius (in Mpc/h) for subvolume used to compute densities
        seed (int): RNG seed for reproducibility of bootstrap
        interval (float): Confidence level for bootstrap interval (e.g., 0.68)

    Returns:
        tuple:
            - deltaMAPBootstrap (BootstrapResult): Bootstrap distribution object
            - deltaMAPInterval (ConfidenceInterval): Confidence interval of MAP estimate
    """
    boxsize = snaps.boxsize
    # Determine center of sphere based on particle positions
    if np.min(snaps["snaps"][0]["pos"]) < 0:
        centre = np.array([0, 0, 0])
    else:
        centre = np.array([boxsize / 2] * 3)
    # Load or compute density samples
    if densities_file is not None:
        deltaMCMCList = tools.loadPickle(densities_file)
    else:
        deltaMCMCList = np.array([
            simulation_tools.density_from_snapshot(snap, centre, dist_max)
            for snap in snaps["snaps"]
        ])
    # Bootstrap MAP density estimator
    deltaMAPBootstrap = scipy.stats.bootstrap(
        (deltaMCMCList,),
        simulation_tools.get_map_from_sample,
        confidence_level=interval,
        vectorized=False,
        random_state=seed
    )
    return deltaMAPBootstrap, deltaMAPBootstrap.confidence_interval

def get_lcdm_void_catalogue(snaps, delta_interval=None, dist_max=135,
                            radii_range=[10, 20], centres_file=None,
                            nRandCentres=10000, seed=1000, flattened=True):
    """
    Construct a void selection mask from a ΛCDM simulation by:
        1. Selecting random underdense regions with matching density
        2. Removing overlapping regions
        3. Filtering voids that fall in those regions and meet radius cuts

    Parameters:
        snaps (SnapHandler): Snapshot object with 'void_centres' and 'void_radii'
        delta_interval (tuple or None): Density contrast bounds to select regions
        dist_max (float): Radius of the spherical region used
        radii_range (list): Acceptable radius range for void selection
        centres_file (str or None): Pickle file path for caching random regions
        nRandCentres (int): Number of random regions to generate if no cache
        seed (int): Random seed for reproducibility
        flattened (bool): Whether to flatten per-region void mask into a single list

    Returns:
        list of arrays: Boolean masks per snapshot indicating selected voids
    """
    boxsize = snaps.boxsize
    # Load or generate random region centres and densities
    rand_centres, rand_densities = tools.loadOrRecompute(
        centres_file,
        simulation_tools.get_random_centres_and_densities,
        dist_max,
        snaps["snaps"],
        seed=seed,
        nRandCentres=nRandCentres
    )
    # Step 1: Filter regions by density (if delta bounds specified)
    region_masks, centres_to_use = _filter_regions_by_density(
        rand_centres, rand_densities, delta_interval
    )
    # Step 2: Prune overlapping regions
    nonoverlapping_indices = simulation_tools.getNonOverlappingCentres(
        centres_to_use, 2 * dist_max, boxsize, returnIndices=True
    )
    selected_region_centres = [
        centres[idx] for centres, idx in zip(centres_to_use, nonoverlapping_indices)
    ]
    selected_region_masks = [
        mask[idx] for mask, idx in zip(region_masks, nonoverlapping_indices)
    ]
    # Step 3: Compute distances from each void to selected regions
    region_void_dists = _compute_void_distances(
        snaps["void_centres"], selected_region_centres, boxsize
    )
    # Step 4: Apply radius and distance filters
    void_masks_by_region = _filter_voids_by_distance_and_radius(
        region_void_dists, snaps["void_radii"], dist_max, radii_range
    )
    # Step 5: Flatten masks if requested
    if flattened:
        return [simulation_tools.flatten_filter_list(masks)
                for masks in void_masks_by_region]
    else:
        return void_masks_by_region

def _filter_regions_by_density(rand_centres, rand_densities, delta_interval):
    """
    Select underdense regions within a delta contrast range.

    Returns:
        - region_masks: Boolean masks for selected centres
        - centres_to_use: Filtered centres as list of arrays per snapshot
    """
    if isinstance(rand_centres,list):
        centres_list = rand_centres
    elif isinstance(rand_centres,np.ndarray):
        centres_list = [rand_centres for _ in rand_densities]
    else:
        raise Exception("Invalid rand_centres")
    if delta_interval is not None:
        region_masks = [
            (deltas > delta_interval[0]) & (deltas <= delta_interval[1])
            for deltas in rand_densities
        ]
        centres_to_use = [
            centres[mask,:] for centres, mask in zip(centres_list, region_masks)
        ]
    else:
        region_masks = [np.ones_like(deltas, dtype=bool) for deltas in rand_densities]
        centres_to_use = centres_list
    return region_masks, centres_to_use

def _compute_void_distances(void_centres, region_centres, boxsize):
    """
    Compute distances from each void to every selected region.

    Returns:
        list of lists: [ [distances for region 1], [region 2], ... ] per snapshot
    """
    return [[
        np.sqrt(np.sum(snapedit.unwrap(voids - region, boxsize)**2, axis=1))
        for region in regions
    ] for voids, regions in zip(void_centres, region_centres)]

def _filter_voids_by_distance_and_radius(dist_lists, radii_lists, dist_max, radii_range):
    """
    Apply filtering to voids based on spatial and size constraints.

    Returns:
        list of lists of boolean arrays: One list per region per snapshot
    """
    return [[
        (dist < dist_max) & (radii > radii_range[0]) & (radii <= radii_range[1])
        for dist in region_dists
    ] for region_dists, radii in zip(dist_lists, radii_lists)]


def get_stacked_void_density_field(snaps, void_radii_lists, void_centre_lists,
                                   spar_bins, sperp_bins, halo_indices=None,
                                   filter_list=None, additional_weights=None,
                                   dist_max=3, rmin=10, rmax=20,
                                   recompute=False, zspace=True,
                                   recompute_zspace=False,
                                   suffix=".lospos_all_zspace2.p",
                                   los_pos=None, **kwargs):
    """
    Compute the 2D stacked density field from LOS particle data for a set of voids.

    Handles filtering, redshift-space conversion, void trimming, and weighted stacking.

    Parameters:
        snaps (SnapHandler): Simulation snapshot handler (must include boxsize, snaps, etc.)
        void_radii_lists (list): Per-snapshot list of void radii
        void_centre_lists (list): Per-snapshot list of void centers
        spar_bins, sperp_bins (array): Bin edges for LOS and transverse directions
        halo_indices (list or None): Indices for each void (optional)
        filter_list (list or None): Optional masks for void selection
        additional_weights (list or None): Optional void weights
        dist_max, (float): Stacking distance threshold
        rmin, rmax, (float) minimum and maximum void radius to consider
        recompute, zspace, recompute_zspace (bool): Control redshift-space position cache
        recompute, (bool): if true, recompute LOS positions in the cache
        zspace, (bool): if true, use redshift space positions, not real space
        recompute_zspace, (bool): if true, recompute redshift space positions.
        suffix (str): File suffix for LOS cache
        los_pos (list or None): Precomputed LOS arrays (optional)
        **kwargs: Passed to internal helper functions

    Returns:
        ndarray: 2D stacked density field
    """
    boxsize = snaps.boxsize
    nbar = len(snaps["snaps"][0]) / boxsize**3
    if halo_indices is not None and filter_list is None:
        filter_list = [halo_indices[ns] >= 0 for ns in range(snaps.N)]
    # Get LOS particle data
    if los_pos is None:
        los_pos = get_los_positions_for_all_catalogues(
            snaps["snaps"], snaps["snaps_reverse"],
            void_centre_lists, void_radii_lists,
            all_particles=True,
            void_indices=halo_indices,
            filter_list=filter_list,
            dist_max=dist_max, rmin=rmin, rmax=rmax,
            recompute=recompute,
            zspace=zspace,
            recompute_zspace=recompute_zspace,
            suffix=suffix
        )
    # Trim voids with no useful particles
    los_list_trimmed, voids_used = trim_los_list(
        los_pos, spar_bins, sperp_bins, void_radii_lists
    )
    los_list_per_void = sum(los_list_trimmed, [])
    num_voids = sum(np.sum(u) for u in voids_used)
    # Per-void radius and weights
    void_radii_per_void = np.hstack([r[used] for r, used in zip(void_radii_lists, voids_used)])
    if additional_weights is not None:
        additional_weights_per_void = np.hstack([
            weights[used] for weights, used in zip(additional_weights, voids_used)
        ])
    else:
        additional_weights_per_void = np.ones(len(void_radii_per_void))
    # Compute stacked field
    return get_2d_field_from_stacked_voids(
        los_list_per_void, sperp_bins, spar_bins,
        void_radii_per_void,
        weights=additional_weights_per_void,
        nbar=nbar
    )



def get_1d_real_space_field(snaps, rbins=None, filter_list=None,
                            additional_weights=None, n_boot=10000, seed=42,
                            halo_indices=None, use_precomputed_profiles=True):
    """
    Compute a bootstrapped real-space 1D void density profile (radial stacking).

    Parameters:
        snaps (SnapHandler): Object with snapshot data and precomputed pair counts
        rbins (array or None): Radial bin edges. If None, defaults to linspace(0, 3)
        filter_list (list or None): Per-snapshot list of boolean masks for void selection
        additional_weights (list or None): Optional void-level weights
        n_boot (int): Number of bootstrap samples
        seed (int): RNG seed for bootstrap
        halo_indices (list or None): Void indices to use (alternative to filter_list)
        use_precomputed_profiles (bool): If True, use stored pair counts

    Returns:
        tuple:
            - rho_mean (array): Bootstrapped mean profile
            - rho_std (array): Profile standard deviation
    """
    boxsize = snaps.boxsize
    nbar = len(snaps["snaps"][0]) / boxsize**3
    # Fallback filter setup
    if halo_indices is not None and filter_list is None:
        filter_list = [halo_indices[ns] >= 0 for ns in range(snaps.N)]
    if filter_list is None:
        filter_list = [np.ones(len(x), dtype=bool) for x in snaps["pair_counts"]]
    # Load pair counts
    if use_precomputed_profiles:
        rbins = snaps["radius_bins"][0]
        all_counts = snaps["pair_counts"]
        all_volumes = snaps["bin_volumes"]
    else:
        if rbins is None:
            rbins = np.linspace(0, 3, 31)
        all_counts, all_volumes = [], []
        for ns in range(snaps.N):
            tree = scipy.spatial.cKDTree(snaps["snaps"][ns]['pos'], boxsize=boxsize)
            counts, volumes = stacking.getPairCounts(
                snaps["void_centres"][ns], snaps["void_radii"][ns],
                snaps["snaps"][ns], rbins,
                nThreads=-1, tree=tree, method="poisson",
                vorVolumes=snaps["cell_volumes"][ns]
            )
            all_counts.append(counts)
            all_volumes.append(volumes)
    # Select voids
    if halo_indices is not None:
        all_antihalos = [halo_indices[ns][halo_indices[ns] >= 0] - 1 for ns in range(snaps.N)]
    elif filter_list is not None:
        all_antihalos = [np.where(filt)[0] for filt in filter_list]
    else:
        all_antihalos = [np.arange(len(x)) for x in snaps["pair_counts"]]
    # Compute all individual density profiles
    all_profiles = [counts[idx] / (vols[idx] * nbar)
                    for counts, vols, idx in zip(all_counts, all_volumes, all_antihalos)]
    density = np.vstack(all_profiles)
    # Weights
    if additional_weights is not None:
        additional_weights_per_void = np.hstack([
            weights[used] for weights, used in zip(additional_weights, filter_list)
        ])
    else:
        additional_weights_per_void = np.ones(density.shape[0])
    # Bootstrap over voids
    np.random.seed(seed)
    num_voids = len(additional_weights_per_void)
    bootstrap_samples = np.random.choice(num_voids, size=(num_voids, n_boot))
    bootstrap_profiles = np.array([
        np.average(density[bootstrap_samples[:, k], :], axis=0,
                   weights=additional_weights_per_void[bootstrap_samples[:, k]])
        for k in tools.progressbar(range(n_boot))
    ]).T
    rho_mean = np.mean(bootstrap_profiles, axis=1)
    rho_std = np.std(bootstrap_profiles, axis=1)
    return rho_mean, rho_std

def get_additional_weights_borg(cat, voids_used=None):
    """
    Get reproducibility-based weights for BORG voids.

    Uses the finalCatFrac property as a proxy for the confidence or reproducibility
    of a void detection. These are normalized across all voids used in stacking.

    Parameters:
        cat (BorgVoidCatalogue): Void catalogue object
        voids_used (list of bool arrays or None): Optional mask per snapshot

    Returns:
        list of arrays: Normalized reproducibility weights for each void (per snapshot)
    """
    rep_scores = cat.property_with_filter(cat.finalCatFrac, void_filter=True)
    if voids_used is None:
        voids_used = [np.ones(rep_scores.shape[0], dtype=bool)
                      for _ in range(cat.numCats)]
    all_rep_scores = np.hstack([rep_scores[used] for used in voids_used])
    norm_factors = np.sum(all_rep_scores)
    return [rep_scores[used] / norm_factors for used in voids_used]


#-------------------------------------------------------------------------------
# COVARIANCE MATRIX CALCULATION

def get_void_weights(los_list_trimmed, voids_used, all_radii,
                     additional_weights=None):
    """
    Compute the total stacking weight for each void in the dataset.

    This wraps get_weights_for_stack(), filtering the void radii using the
    `voids_used` boolean masks to match the trimmed LOS list.

    Weights combine:
      - A volume-based scaling (1 / R_eff^3)
      - Optional additional weights (e.g. for S/N or inverse variance)

    NOTE: This function may be deprecated if weights are split into separate
    logical operations (e.g., volume normalization vs S/N weighting).

    Parameters:
        los_list_trimmed (list): List of per-void LOS particle arrays (2D)
        voids_used (list of bool arrays): Per-snapshot mask of valid voids
        all_radii (list of arrays): R_eff for all voids (before masking)
        additional_weights (list or scalar or None): Optional multiplicative weights

    Returns:
        list of arrays: Per-void weights (unstacked)
    """
    # Apply void selection mask to radii
    filtered_radii = [rad[used] for used, rad in zip(voids_used, all_radii)]
    return get_weights_for_stack(
        los_list_trimmed,
        filtered_radii,
        additional_weights=additional_weights,
        stacked=False
    )

def get_covariance_matrix(los_list, void_radii_all,
                          spar_bins, sperp_bins, nbar,
                          additional_weights=None,
                          n_boot=10000, seed=42,
                          lambda_reg=1e-15,
                          cholesky=True, regularise=True,
                          log_field=False, return_mean=False):
    """
    Estimate the covariance matrix (or its Cholesky decomposition) of a
    2D stacked void density field using bootstrap resampling.

    Steps:
        1. Trims voids with no contributing LOS particles
        2. Computes 2D density fields for each individual void
        3. Bootstraps the mean stacked profile over voids
        4. Computes the covariance matrix (or its log)
        5. Optionally regularizes and/or returns Cholesky decomposition

    Parameters:
        los_list (list): Per-snapshot list of LOS particle arrays (per void)
        void_radii_all (list): Per-snapshot list of void radii
        spar_bins (array): LOS bin edges
        sperp_bins (array): Transverse bin edges
        nbar (float): Mean number density
        additional_weights (list or None): Optional void weights per snapshot
        n_boot (int): Number of bootstrap realizations
        seed (int): RNG seed for reproducibility
        lambda_reg (float): Regularization strength for covariance
        cholesky (bool): If True, return lower-triangular Cholesky factor
        regularise (bool): If True, apply regularization to covariance
        log_field (bool): If True, compute covariance of log-density
        return_mean (bool): If True, return the bootstrap mean as well

    Returns:
        cov (ndarray): Covariance matrix (or Cholesky factor if cholesky=True)
        mean (ndarray, optional): Mean field if return_mean=True
    """
    # --- Step 1: Trim out voids with no contributing particles
    los_list_trimmed, voids_used = trim_los_list(
        los_list, spar_bins, sperp_bins, void_radii_all
    )
    los_per_void = sum(los_list_trimmed, [])
    void_radii_per_void = np.hstack([
        radii[used] for radii, used in zip(void_radii_all, voids_used)
    ])
    num_voids = len(void_radii_per_void)
    # --- Step 2: Compute stacked 2D field per void (flattened)
    stacked_fields = get_2d_fields_per_void(
        los_per_void, sperp_bins, spar_bins,
        void_radii_per_void, nbar=nbar
    ).reshape(num_voids, (len(spar_bins) - 1) * (len(sperp_bins) - 1))
    # --- Step 3: Prepare void weights
    if additional_weights is not None:
        additional_weights_per_void = np.hstack([
            weights[used] for weights, used in zip(additional_weights, voids_used)
        ])
    else:
        additional_weights_per_void = np.ones(num_voids)
    # --- Step 4: Bootstrap stacking over voids
    np.random.seed(seed)
    bootstrap_samples = np.random.choice(num_voids, size=(num_voids, n_boot))
    bootstrap_stacks = np.array([
        np.average(
            stacked_fields[bootstrap_samples[:, k], :],
            axis=0,
            weights=additional_weights_per_void[bootstrap_samples[:, k]]
        )
        for k in tools.progressbar(range(n_boot))
    ]).T  # shape: [n_bins, n_boot]
    # --- Step 5: Compute covariance matrix
    if log_field:
        log_samples = np.log(bootstrap_stacks)
        finite_mask = np.all(np.isfinite(log_samples), axis=0)
        log_samples = log_samples[:, finite_mask]
        cov = np.cov(log_samples)
        mean = np.mean(log_samples, axis=1)
    else:
        cov = np.cov(bootstrap_stacks)
        mean = np.mean(bootstrap_stacks, axis=1)
    # --- Step 6: Regularization
    if regularise:
        cov = regularise_covariance(cov, lambda_reg=lambda_reg)
    # --- Step 7: Cholesky decomposition (if requested)
    if cholesky:
        cov = scipy.linalg.cholesky(cov, lower=True)
    if return_mean:
        return cov, mean
    else:
        return cov


# Covariance function:
def get_covariance_matrix_old(los_list,void_radii_all,spar_bins,sperp_bins,nbar,
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
        los_per_void,sperp_bins,spar_bins,
        void_radii_per_void,nbar=nbar).reshape(num_voids,
        (len(spar_bins) - 1)*(len(sperp_bins) - 1))
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

def get_mle_estimate(initial_guess_eps, theta_ranges_epsilon, *args, **kwargs):
    """
    Compute the Maximum Likelihood Estimate (MLE) for cosmological parameters,
    by minimizing the negative log-likelihood function.

    Parameters:
        initial_guess_eps (array): Initial guess for [epsilon, f] or [Omega_m, f]
        theta_ranges_epsilon (list of tuples): Parameter bounds
        *args: Arguments passed to the likelihood function
        **kwargs: Keyword args for the likelihood

    Returns:
        OptimizeResult: Output from scipy.optimize.minimize
    """
    nll = lambda theta: -log_likelihood_aptest(theta, *args, **kwargs)
    mle_estimate = scipy.optimize.minimize(
        nll,
        initial_guess_eps,
        bounds=theta_ranges_epsilon
    )
    return mle_estimate


def get_fixed_inverse(Delta_func, ntab=100, ntab_f=100,
                      sval_range=[0, 3], rperp_range=[0, 3],
                      f_val_range=[0, 1]):
    """
    Precompute and tabulate the inverse mapping from redshift-space
    (s_parallel) to real-space (r_parallel), assuming fixed profile parameters.

    This accelerates likelihood evaluations when the inverse is not dependent
    on sampled parameters.

    Parameters:
        Delta_func (function): Cumulative density profile Δ(r)
        ntab (int): Number of grid points for r_perp and s_parallel
        ntab_f (int): Number of grid points for f
        sval_range (list): Range of s_parallel values
        rperp_range (list): Range of r_perp values
        f_val_range (list): Range of f values

    Returns:
        function: Interpolated inverse mapping function F_inv(s_perp, s_par, f)
    """
    svals = np.linspace(sval_range[0], sval_range[1], ntab)
    rperp_vals = np.linspace(rperp_range[0], rperp_range[1], ntab)
    f_vals = np.linspace(f_val_range[0], f_val_range[1], ntab_f)
    F_inv_vals = np.zeros((ntab, ntab, ntab_f))
    for i in tools.progressbar(range(ntab)):
        for j in range(ntab):
            for k in range(ntab_f):
                f = f_vals[k]
                rperp = rperp_vals[i]
                spar = svals[j]
                func = lambda r: r - r * (f / 3.0) * Delta_func(np.sqrt(r**2 + rperp**2)) - spar
                F_inv_vals[i, j, k] = scipy.optimize.fsolve(func, spar)

    F_inv = lambda x, y, z: scipy.interpolate.interpn(
        (rperp_vals, svals, f_vals),
        F_inv_vals,
        np.vstack((x, y, z)).T,
        method="cubic"
    )
    return F_inv

def generate_scoord_grid(sperp_bins, spar_bins):
    """
    Generate (s_par, s_perp) bin-centre coordinate pairs for all bins
    in the 2D stacked void field.

    Parameters:
        sperp_bins (array): Bin edges in transverse direction
        spar_bins (array): Bin edges in line-of-sight direction

    Returns:
        ndarray: Array of shape (N_bins, 2), where each row is [s_par, s_perp]
    """
    spar = np.hstack([
        s * np.ones(len(sperp_bins) - 1)
        for s in plot.binCentres(spar_bins)
    ])
    sperp = np.hstack([
        plot.binCentres(sperp_bins)
        for _ in plot.binCentres(spar_bins)
    ])
    return np.vstack([spar, sperp]).T


def generate_data_filter(cov, mean, scoords, cov_thresh=5, srad_thresh=1.5):
    """
    Apply a data mask to exclude noisy or poorly constrained bins.

    Parameters:
        cov (ndarray): Covariance matrix
        mean (ndarray): Mean stacked profile
        scoords (ndarray): Grid coordinates (s_par, s_perp)
        cov_thresh (float): Inverse S/N threshold (1/sigma > cov_thresh)
        srad_thresh (float): Max radial distance to include (in units of R_eff)

    Returns:
        ndarray: Array of indices for bins that pass the filter
    """
    norm_cov = cov / np.outer(mean, mean)
    inv_sigma = 1.0 / np.sqrt(np.diag(norm_cov))
    radial_dist = np.sqrt(np.sum(scoords**2, axis=1))
    data_filter = np.where((inv_sigma > cov_thresh) & (radial_dist < srad_thresh))[0]
    return data_filter

def log_likelihood_profile(theta, x, y, yerr, profile_model):
    """
    Gaussian log-likelihood for a real-space density profile model.

    Parameters:
        theta (array): Profile parameters
        x (array): Radial coordinates
        y (array): Observed densities
        yerr (array): Uncertainties
        profile_model (function): Model function ρ(x, *theta)

    Returns:
        float: Log-likelihood
    """
    model = profile_model(x, *theta)
    sigma2 = yerr**2
    return -0.5 * np.sum((y - model)**2 / sigma2 + np.log(sigma2))


def get_profile_parameters_fixed(ri, rhoi, sigma_rhoi,
                                 model=profile_modified_hamaus,
                                 initial=np.array([1.0, 1.0, 1.0, -0.2, 0.0, 1.0]),
                                 bounds=[(0, None), (0, None), (0, None),
                                         (-1, 0), (-1, 1), (0, 2)]):
    """
    Fit a fixed-profile model to a 1D void density profile using MLE.

    Parameters:
        ri (array): Radial bin centres (in units of R_eff)
        rhoi (array): Measured density values at ri
        sigma_rhoi (array): Uncertainties on density
        model (function): Parametric profile function δ(r, *params)
        initial (array): Initial guess for profile parameters
        bounds (list of tuples): Parameter bounds for optimization

    Returns:
        array: Best-fit parameter values
    """
    # Define negative log-likelihood function for optimizer
    nll = lambda *theta: -log_likelihood_profile(*theta)
    # Perform bounded minimization
    sol = scipy.optimize.minimize(
        nll,
        initial,
        bounds=bounds,
        args=(ri, rhoi, sigma_rhoi, model)
    )
    return sol.x



def run_inference(data_field, theta_ranges_list, theta_initial, filename,
                  log_probability, *args,
                  redo_chain=False,
                  backup_start=True,
                  nwalkers=64, sample="all", n_mcmc=10000,
                  disp=1e-4, Om_fid=0.3111, max_n=1000000, z=0.0225,
                  parallel=False, batch_size=100, n_batches=100,
                  data_filter=None, autocorr_file=None, **kwargs):
    """
    Run MCMC inference using emcee to sample posterior over cosmological + profile parameters.

    Parameters:
        data_field (array): Flattened observed density field (after filtering)
        theta_ranges_list (list of tuples): Parameter bounds
        theta_initial (array): Initial guess for parameters
        filename (str): Path to HDF5 file for emcee backend
        log_probability (function): Log-posterior function to sample
        *args: Arguments passed to log_probability
        redo_chain (bool): If True, erase and restart chain
        backup_start (bool): If True, backup existing chain to .old
        nwalkers (int): Number of walkers
        sample (str or array): "all" or boolean array specifying sampled parameters
        n_mcmc (int): Number of MCMC steps
        disp (float): Walker spread around initial guess
        Om_fid (float): Fiducial Omega_m
        max_n (int): Maximum number of samples to allow
        z (float): Redshift
        parallel (bool): If True, use multiprocessing Pool
        batch_size (int): MCMC steps per batch
        n_batches (int): Max number of batches
        data_filter (array or None): Indices of used bins
        autocorr_file (str or None): Path to save autocorrelation data
        **kwargs: Passed to log_probability

    Returns:
        tau (array): Autocorrelation time estimates for each parameter
        sampler (emcee.EnsembleSampler): Final MCMC sampler object
    """
    if sample == "all":
        sample = np.array([True for _ in theta_ranges_list])
    ndims = np.sum(sample)
    ndata = len(data_field)
    if data_filter is None:
        data_filter = np.arange(ndata)
    # --- Initial walker positions
    initial = theta_initial + disp * np.random.randn(nwalkers, ndims)
    # --- Optional: backup old file
    filename_initial = filename + ".old"
    if backup_start:
        os.system(f"cp {filename} {filename_initial}")
    backend = emcee.backends.HDFBackend(filename)
    if redo_chain:
        backend.reset(nwalkers, ndims)
    # --- Run in parallel
    if parallel:
        with Pool() as pool:
            sampler = emcee.EnsembleSampler(
                nwalkers, ndims, log_probability_aptest_parallel,
                args=(z,),
                kwargs={'Om_fid': Om_fid, 'cholesky': True,
                        'tabulate_inverse': True, 'sample_epsilon': True},
                backend=backend, pool=pool
            )
            sampler.run_mcmc(initial, n_mcmc, progress=True)
    else:
        # --- Serial run with convergence monitoring
        sampler = emcee.EnsembleSampler(
            nwalkers, ndims, log_probability,
            args=args,
            kwargs=kwargs,
            backend=backend
        )
        # Load or initialize autocorrelation array
        if redo_chain or autocorr_file is None:
            autocorr = np.zeros((ndims, 0))
        else:
            autocorr = np.load(autocorr_file)
        old_tau = np.inf
        for k in range(n_batches):
            if k == 0 and redo_chain:
                sampler.run_mcmc(initial, batch_size, progress=True)
            else:
                sampler.run_mcmc(None, batch_size, progress=True)
            # --- Autocorrelation monitoring
            try:
                tau = sampler.get_autocorr_time(tol=0)
            except emcee.autocorr.AutocorrError:
                tau = old_tau  # fallback
            autocorr = np.hstack([autocorr, tau.reshape((ndims, 1))])
            if autocorr_file is not None:
                np.save(autocorr_file, autocorr)
            # --- Convergence check
            converged = np.all(tau * 100 < sampler.iteration)
            converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
            if converged:
                break
            old_tau = tau
    return tau, sampler


def run_inference_pipeline(field, cov, mean, sperp_bins, spar_bins,
                           ri, delta_i, sigma_delta_i,
                           log_field=False,
                           infer_profile_args=True,
                           tabulate_inverse=True,
                           cholesky=True,
                           sample_epsilon=True,
                           filter_data=False,
                           z=0.0225,
                           lambda_cut=1e-23,
                           lambda_ref=1e-27,
                           profile_param_ranges=[[0, np.inf], [0, np.inf], [0, np.inf],
                                                 [-1, 0], [-1, 1], [0, 2]],
                           om_ranges=[[0.1, 0.5]],
                           eps_ranges=[[0.9, 1.1]],
                           f_ranges=[[0, 1]],
                           Om_fid=0.3111,
                           filename="inference_weighted.h5",
                           autocorr_filename="autocorr.npy",
                           disp=1e-2,
                           nwalkers=64,
                           n_mcmc=10000,
                           max_n=1000000,
                           batch_size=100,
                           nbatch=100,
                           redo_chain=False,
                           backup_start=True,
                           delta_profile=profile_modified_hamaus,
                           Delta_profile=integrated_profile_modified_hamaus):
    """
    Full inference pipeline to constrain cosmological parameters from
    stacked void density fields in redshift space.

    Steps:
        1. Compute inverse covariance or Cholesky factor
        2. Filter data spatially and based on S/N
        3. Define density profile model (fixed or inferred)
        4. Precompute redshift-space inverse mapping (optional)
        5. Prepare parameter bounds and priors
        6. Run MCMC inference

    Parameters:
        field (2D array): Stacked void field
        cov (2D array): Covariance matrix for the field
        sperp_bins, spar_bins (arrays): Bin edges in LOS and transverse directions
        ri, delta_i, sigma_delta_i (arrays): Real-space profile and uncertainties
        log_field (bool): If True, use log-density field
        infer_profile_args (bool): If True, sample profile parameters
        tabulate_inverse (bool): If True, precompute redshift-space inverse mapping
        cholesky (bool): If True, use Cholesky factor of covariance
        sample_epsilon (bool): If True, sample epsilon and f instead of Om and f
        filter_data (bool): If True, apply data mask to reduce noise
        z (float): Redshift
        lambda_cut (float): Eigenvalue cutoff for filtering singular modes
        lambda_ref (float): Tikhonov regularization for nonsingular subspace
        profile_param_ranges (list): Parameter bounds for profile model
        om_ranges, eps_ranges, f_ranges (list): Bounds for Om/epsilon, f
        Om_fid (float): Fiducial Omega_m for model evaluation
        filename (str): HDF5 file to save MCMC chain
        autocorr_filename (str): Path to store autocorrelation history
        disp (float): Initial walker spread
        nwalkers (int): Number of MCMC walkers
        n_mcmc (int): Total MCMC steps
        max_n (int): Max samples (unused)
        batch_size (int): Steps per convergence-check batch
        nbatch (int): Max number of batches
        redo_chain (bool): If True, overwrite previous chain
        backup_start (bool): Backup old chain before overwrite
        delta_profile, Delta_profile (functions): Density and cumulative profile models

    Returns:
        tau (array): Autocorrelation times
        sampler (EnsembleSampler): Final MCMC sampler object
    """
    # --- Step 1: Covariance preparation
    if cholesky:
        inverse_matrix = scipy.linalg.cholesky(cov, lower=True)
    else:
        inverse_matrix = get_inverse_covariance(cov, lambda_reg=lambda_ref)
    # --- Step 2: Filter data
    scoords = generate_scoord_grid(sperp_bins, spar_bins)
    if filter_data:
        data_filter = generate_data_filter(cov, mean, scoords)
    else:
        data_filter = np.ones(field.flatten().shape, dtype=bool)
    data_field = field.flatten()[data_filter]
    # --- Step 3: Choose profile model
    if infer_profile_args:
        delta_func = delta_profile
        Delta_func = Delta_profile
        rho_real = lambda *args: delta_profile(*args) + 1.0
    else:
        profile_params = get_profile_parameters_fixed(
            ri, delta_i, sigma_delta_i, model=delta_profile
        )
        delta_func = lambda r: delta_profile(r, *profile_params)
        Delta_func = lambda r: Delta_profile(r, *profile_params)
        rho_real = lambda r: delta_func(r) + 1.0
    # --- Step 4: Tabulated inverse (optional)
    if infer_profile_args or not tabulate_inverse:
        F_inv = None
    else:
        F_inv = get_fixed_inverse(Delta_func)
    # --- Step 5: Parameter bounds and initial guess
    if sample_epsilon:
        initial_guess_MG = np.array([1.0, f_lcdm(z, Om_fid)])
        theta_ranges = eps_ranges + f_ranges + profile_param_ranges
    else:
        initial_guess_MG = np.array([Om_fid, f_lcdm(z, Om_fid)])
        theta_ranges = om_ranges + f_ranges + profile_param_ranges
    if infer_profile_args:
        profile_params = get_profile_parameters_fixed(ri, delta_i, sigma_delta_i)
        initial_guess = np.hstack([initial_guess_MG, profile_params])
    else:
        initial_guess = initial_guess_MG
    # --- Step 6: Filter singular modes
    Umap, good_eig = get_nonsingular_subspace(
        cov, lambda_reg=lambda_ref,
        lambda_cut=lambda_cut, normalised_cov=False,
        mu=mean)
    # --- Step 7: Assemble args and kwargs for MCMC
    args = (
        data_field,
        scoords[data_filter, :],
        inverse_matrix[data_filter][:, data_filter],
        z,
        Delta_func,
        delta_func,
        rho_real
    )
    kwargs = {
        'cholesky': cholesky,
        'tabulate_inverse': tabulate_inverse,
        'sample_epsilon': sample_epsilon,
        'theta_ranges': theta_ranges,
        'singular': False,
        'Umap': Umap,
        'good_eig': good_eig,
        'F_inv': F_inv,
        'log_density': log_field,
        'infer_profile_args':infer_profile_args
    }
    # --- Step 8: Run MCMC
    tau, sampler = run_inference(
        data_field, theta_ranges, initial_guess, filename,
        log_probability_aptest, *args,
        redo_chain=redo_chain,
        backup_start=backup_start,
        nwalkers=nwalkers,
        sample="all",
        n_mcmc=n_mcmc,
        disp=disp,
        max_n=max_n,
        z=z,
        parallel=False,
        Om_fid=Om_fid,
        batch_size=batch_size,
        n_batches=nbatch,
        autocorr_file=autocorr_filename,
        **kwargs
    )
    return tau, sampler















