#-------------------------------------------------------------------------------
# CONFIGURATION
from void_analysis import plot, tools, snapedit, catalogue
from void_analysis.catalogue import *
from void_analysis.paper_plots_borg_antihalos_generate_data import *
from void_analysis.real_clusters import getClusterSkyPositions
from void_analysis import massconstraintsplot
from void_analysis.simulation_tools import ngPerLBin
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

# Set global figure font:
import matplotlib.pyplot as plt
fontfamily = "serif"
#plt.rcParams["font.family"] = "serif"
#plt.rcParams['font.serif'] = ["Times New Roman"]
#plt.rcParams['font.serif'] = ["DejaVu Serif"]
#plt.rcParams["mathtext.fontset"] = "stix"


# Data export options:
save_plot_data = True
load_plot_data = True

figuresFolder = "void_applications/"
#figuresFolder = "borg-antihalos_paper_figures/all_samples/"
#figuresFolder = "borg-antihalos_paper_figures/batch5-2/"
#figuresFolder = "borg-antihalos_paper_figures/batch5-4/"
#figuresFolder = "borg-antihalos_paper_figures/batch5-1/"
#figuresFolder = "borg-antihalos_paper_figures/batch5-3/"
#figuresFolder = "borg-antihalos_paper_figures/batch10-2/"
#figuresFolder = "borg-antihalos_paper_figures/batch10-1/"

recomputeData = False
testDataFolder = figuresFolder + "tests_data/"
runTests = False

# Filename data:
unconstrainedFolderNew = "new_chain/unconstrained_samples/"
unconstrainedFolderOld = "unconstrainedSamples/"
snapnameNew = "gadget_full_forward_512/snapshot_001"
snapnameNewRev = "gadget_full_reverse_512/snapshot_001"
samplesFolder="new_chain/"
samplesFolderOld = "./"
snapnameOld = "forward_output/snapshot_006"
snapnameOldRev = "reverse_output/snapshot_006"

data_folder = figuresFolder
data_folder2 = "borg-antihalos_paper_figures/all_samples/"

fontsize = 9
legendFontsize = 9

low_memory_mode = True

#-------------------------------------------------------------------------------
# LOAD SNAPSHOT DATA:



clusterNames = np.array([['Perseus-Pisces (A426)'],
       ['Hercules B (A2147)'],
       ['Coma (A1656)'],
       ['Norma (A3627)'],
       ['Shapley (A3571)'],
       ['A548'],
       ['Hercules A (A2199)'],
       ['Hercules C (A2063)'],
       ['Leo (A1367)']], dtype='<U21')


# HMF plots data:

# Snapshots to use:
snapNumListOld = [7422,7500,8000,8500,9000,9500]
#snapNumList = [7000,7200,7400,7600,8000]
#snapNumList = [7000,7200,7400,7600,7800,8000]
#snapNumList = np.arange(7000,10300 +1,300)
#snapNumList = [8800,9100,9400,9700,10000]
snapNumList = [7300,7600,7900,8200,8500,8800,9100,9400,9700,10000,\
    10300,10600,10900,11200,11500,11800,12100,12400,12700,13000]
#snapNumList = [7300,7600,7900,8200,8500,8800,9100,9400,9700,10000,\
#    10300,10600,10900,11200,11500,11800,12100,12400,12700,13000,\
#    13300,13600,13900,14200,14500,14800,15100,15400,15700,16000]
# Batch5-1:
#snapNumList = [7300,7600,7900,8200,8500]
# Batch5-2:
#snapNumList = [8800,9100,9400,9700,10000]
# Batch5-3:
#snapNumList = [10300,10600,10900,11200,11500]
# Batch5-4:
#snapNumList = [11800,12100,12400,12700,13000]
# Batch10-1 
#snapNumList = [7300,7600,7900,8200,8500,8800,9100,9400,9700,10000]
# Batch10-2
#snapNumList = [10300,10600,10900,11200,11500,11800,12100,12400,12700,13000]
# Batch 5-5:
#snapNumList = [13000,13300,13600,13900,14200]

#snapNumListUncon = [1,2,3,4,5]
snapNumListUncon = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
#snapNumListUncon = [2,4,5,6,7,8,9,10]
snapNumListUnconOld = [1,2,3,5,6]
boxsize = 677.7

# Get profiles for the constrained voids only:
snapList =  [pynbody.load(samplesFolder + "sample" + str(snapNum) + "/" 
             + "gadget_full_forward_512/snapshot_001") 
             for snapNum in snapNumList]
snapListRev =  [pynbody.load(samplesFolder + "sample" + str(snapNum) + "/" 
                + "gadget_full_reverse_512/snapshot_001") 
                for snapNum in snapNumList]
snapNameList = [samplesFolder + "sample" + str(snapNum) + "/" 
                + "gadget_full_forward_512/snapshot_001" 
                for snapNum in snapNumList]
snapNameListRev = [samplesFolder + "sample" + str(k) + "/" + snapnameNewRev \
    for k in snapNumList]

# Unconstrained simulations:
snapListUn = [pynbody.load("new_chain/unconstrained_samples/sample"
              + str(num) + "/gadget_full_forward_512/snapshot_001") 
              for num in snapNumListUncon]
snapListRevUn = [pynbody.load("new_chain/unconstrained_samples/sample"
                 + str(num) + "/gadget_full_reverse_512/snapshot_001") \
                 for num in snapNumListUncon]


def extract_antihalo_property(snap_list,prop):
    properties = []
    for snap in snap_list:
        props_list = tools.loadPickle(snap.filename + ".AHproperties.p")
        properties.append(props_list[prop])
        del props_list
        gc.collect()
    return properties



if not low_memory_mode:
    # Properties of anti-halos:
    ahProps = [pickle.load(open(snap.filename + ".AHproperties.p","rb")) 
               for snap in snapList]
    ahPropsUn = [pickle.load(open(snap.filename + ".AHproperties.p","rb")) 
                 for snap in snapListUn]
    # Anti-halo catalogues:
    hrList = [snap.halos() for snap in snapListRev]
    hrListUn = [snap.halos() for snap in snapListRevUn]
    # Centres, radii, and masses:
    antihaloCentresUn = [tools.remapAntiHaloCentre(props[5],boxsize) \
                for props in ahPropsUn]
    antihaloMassesUn = [props[3] for props in ahPropsUn]
    antihaloRadiiUn = [props[7] for props in ahPropsUn]
    antihaloCentres = [tools.remapAntiHaloCentre(props[5],boxsize,
                                                 swapXZ  = False,
                                                 reverse = True)
                       for props in ahProps]
    antihaloMasses = [props[3] for props in ahProps]
    antihaloRadii = [props[7] for props in ahProps]
else:
    antihaloCentresUn = [tools.remapAntiHaloCentre(centres,boxsize,
                                                 swapXZ  = False,
                                                 reverse = True)
                       for centres in extract_antihalo_property(snapListUn,5)]
    antihaloMassesUn = extract_antihalo_property(snapListUn,3)
    antihaloCentres = [tools.remapAntiHaloCentre(centres,boxsize,
                                                 swapXZ  = False,
                                                 reverse = True)
                       for centres in extract_antihalo_property(snapList,5)]
    antihaloRadiiUn = extract_antihalo_property(snapListUn,7)
    antihaloCentres = extract_antihalo_property(snapList,5)
    antihaloMasses = extract_antihalo_property(snapList,3)
    antihaloRadii = extract_antihalo_property(snapList,7)

#-------------------------------------------------------------------------------
# BUILDING VOID CATALOGUES FROM SCRATCH
recomputeCatalogues = False

# Parameters:
rSphere = 300 # Radius out to which to search for voids
muOpt = 0.75 # Optimal choice of \mu_R, radius ratio
rSearchOpt = 0.5 # Optimal choice of \mu_S, search radius ratio
NWayMatch = False # Whether to use N-way matching code.
refineCentres = True # Whether to refine centres iteratively using all samples
sortBy = "radius" # Quantity to sort catalogue by
enforceExclusive = True # Whether to apply the algorithm purging duplicate
    # voids from the catalogues.
mMin = 1e11 # Minimum mass halos to include (Note, this is effectively
    # over-ridden by the minimum radius threshold)
mMax = 1e16 # Maximum mass halos to include (Note, this is effectively 
    # over-ridden by the maximum radius threshold)
rMin = 5 # Minimum radius voids to use
rMax = 30 # Maximum radius voids to use
m_unit = snapList[0]['mass'][0]*1e10

# Signal to noise information:
snrThresh=10
chainFile="chain_properties.p"
if not low_memory_mode:
    [snrFilter,snrAllCatsList] = getSNRFilterFromChainFile(
        chainFile,snrThresh,snapNameList,boxsize)

if recomputeCatalogues or (not os.path.isfile(data_folder2 + "cat300.p")):
    cat300 = catalogue.combinedCatalogue(
        snapNameList,snapNameListRev,\
        muOpt,rSearchOpt,rSphere,\
        ahProps=ahProps,hrList=hrList,max_index=None,\
        twoWayOnly=True,blockDuplicates=True,\
        massRange = [mMin,mMax],\
        NWayMatch = NWayMatch,r_min=rMin,r_max=rMax,\
        additionalFilters = snrFilter,verbose=False,\
        refineCentres=refineCentres,sortBy=sortBy,\
        enforceExclusive=enforceExclusive)
    cat300test = catalogue.combinedCatalogue(
        snapNameList,snapNameListRev,\
        muOpt,rSearchOpt,rSphere,\
        ahProps=ahProps,hrList=hrList,max_index=None,\
        twoWayOnly=True,blockDuplicates=True,\
        massRange = [m_unit*800,mMax],\
        NWayMatch = NWayMatch,r_min=10,r_max=30,\
        additionalFilters = snrFilter,verbose=False,\
        refineCentres=refineCentres,sortBy=sortBy,\
        enforceExclusive=enforceExclusive)
    finalCat300test = cat300test.constructAntihaloCatalogue()
    finalCat300 = cat300.constructAntihaloCatalogue()
    tools.savePickle(cat300,data_folder2 + "cat300.p")
else:
    cat300 = tools.loadPickle(data_folder2 + "cat300.p")

# Random catalogues:
snapNameListRand = [snap.filename for snap in snapListUn]
snapNameListRandRev = [snap.filename for snap in snapListRevUn]

if recomputeCatalogues or (not os.path.isfile(data_folder2 + "cat300Rand.p")):
    cat300Rand = catalogue.combinedCatalogue(
        snapNameListRand,snapNameListRandRev,\
        muOpt,rSearchOpt,rSphere,\
        ahProps=ahPropsUn,hrList=hrListUn,max_index=None,\
        twoWayOnly=True,blockDuplicates=True,\
        massRange = [mMin,mMax],\
        NWayMatch = NWayMatch,r_min=rMin,r_max=rMax,\
        additionalFilters = None,verbose=False,\
        refineCentres=refineCentres,sortBy=sortBy,\
        enforceExclusive=enforceExclusive)
    cat300RandTest = catalogue.combinedCatalogue(
        snapNameListRand,snapNameListRandRev,\
        muOpt,rSearchOpt,rSphere,\
        ahProps=ahPropsUn,hrList=hrListUn,max_index=None,\
        twoWayOnly=True,blockDuplicates=True,\
        massRange = [m_unit*800,mMax],\
        NWayMatch = NWayMatch,r_min=10,r_max=30,\
        additionalFilters = None,verbose=False,\
        refineCentres=refineCentres,sortBy=sortBy,\
        enforceExclusive=enforceExclusive)
    finalCat300RandTest = cat300RandTest.constructAntihaloCatalogue()
    finalCat300Rand = cat300Rand.constructAntihaloCatalogue()
    tools.savePickle(cat300Rand,data_folder2 + "cat300Rand.p")
else:
    cat300Rand = tools.loadPickle(data_folder2 + "cat300Rand.p")

# Apply Catalogue filter:
nBinEdges = 8
rLower = 10
rUpper = 20
radBins = np.linspace(rLower,rUpper,nBinEdges)
cat300.set_filter_from_random_catalogue(cat300Rand,radBins,r_min=10,r_max=25,
                                        r_sphere=135)



#-------------------------------------------------------------------------------
# CATALOGUE DATA:

ns = 0
doSky=True
snapToShow = pynbody.load(samplesFolder + "sample7000" + \
    "/gadget_full_forward_512/snapshot_001")
tools.remapBORGSimulation(snapToShow,swapXZ=False,reverse=True)

rCut = 135
ha = ['right','left','left','left','left','center','right',\
        'right','right']
va = ['center','left','bottom','bottom','top',\
        'top','center','center','center']
annotationPos = [[-1.1,0.9],\
        [1.3,0.1],[1.5,0.6],[1.3,-1.2],[1.3,-0.7],[-1,0.2],[0.8,0.6],\
        [1.0,0.1],[-1.7,0.5]]
nameList = [name[0] for name in clusterNames]
textwidth=7.1014
textheight=9.0971
scale = 1.26
width = textwidth
height = 0.55*textwidth
cropPoint = ((scale -1)/2)*np.array([width,height]) + np.array([0,0.09])
bound_box = transforms.Bbox([[cropPoint[0], cropPoint[1]],
    [cropPoint[0] + width, cropPoint[1] + height]])

# Cluster locations:
# Galaxy positions:
if doSky:
    [combinedAbellN,combinedAbellPos,abell_nums] = \
        real_clusters.getCombinedAbellCatalogue()
    abell_nums = [426,2147,1656,3627,3571,548,2197,2052,1367]
    [abell_l,abell_b,abell_n,abell_z,\
            abell_d,p_abell,coordAbell] = getClusterSkyPositions("./")
    clusterInd = [np.where(combinedAbellN == n)[0] for n in abell_nums]
    clusterIndMain = [ind[0] for ind in clusterInd]
    coordCombinedAbellCart = SkyCoord(x=combinedAbellPos[:,0]*u.Mpc,\
            y = combinedAbellPos[:,1]*u.Mpc,z = combinedAbellPos[:,2]*u.Mpc,\
            frame='icrs',representation_type='cartesian')
    equatorialRThetaPhi = np.vstack(\
        [coordCombinedAbellCart.icrs.spherical.distance.value,\
        coordCombinedAbellCart.icrs.spherical.lat.value*np.pi/180.0,\
        coordCombinedAbellCart.icrs.spherical.lon.value*np.pi/180]).transpose()
    coordCombinedAbellSphere = SkyCoord(distance=\
        coordCombinedAbellCart.icrs.spherical.distance.value*u.Mpc,\
        ra = coordCombinedAbellCart.icrs.spherical.lon.value*u.deg,\
        dec = coordCombinedAbellCart.icrs.spherical.lat.value*u.deg,\
        frame='icrs')

clusterLoc = np.array([np.array([\
    coordCombinedAbellCart[ind].x.value,\
    coordCombinedAbellCart[ind].y.value,\
    coordCombinedAbellCart[ind].z.value]) for ind in clusterIndMain])

referenceSnap = snapToShow
Om0 = referenceSnap.properties['omegaM0']
Ode0 = referenceSnap.properties['omegaL0']
H0 = referenceSnap.properties['h']*100
h = referenceSnap.properties['h']
boxsize = referenceSnap.properties['boxsize'].ratio("Mpc a h**-1")
cosmo = astropy.cosmology.LambdaCDM(H0,Om0,Ode0)


# 2M++ Data:
catFile = "./2mpp_data/2m++.txt"
catalogueData = np.loadtxt(catFile,delimiter='|',skiprows=31,
    usecols=(1,2,3,4,5,6,7,8,10,11,12,13,14,15,16))
# Filter useable galaxies:
useGalaxy = (catalogueData[:,10] == 0.0) & (catalogueData[:,5] > 0)
c = 299792.458 # Speed of light in km/s
z = catalogueData[:,5]/c # Redshift
# Cosmological parameters:

# Comoving distance to all galaxies, in Mpc/h:
dcz = cosmo.comoving_distance(z[useGalaxy]).value*cosmo.h
# Co-ordinates of the galaxies (in Mpc/h):
coord = astropy.coordinates.SkyCoord(\
    ra = catalogueData[useGalaxy,0]*astropy.units.degree,\
    dec=catalogueData[useGalaxy,1]*astropy.units.degree,\
    distance=dcz*astropy.units.Mpc)
# Cartesian positions of galaxies in equatorial, comoving co-ordinates (Mpc/h):
equatorialXYZ = np.vstack((coord.cartesian.x.value,\
    coord.cartesian.y.value,coord.cartesian.z.value)).T
# In spherical polar co-ordinates:
equatorialRThetaPhi = np.vstack((coord.icrs.spherical.distance.value,\
    coord.icrs.spherical.lon.value,\
    coord.icrs.spherical.lat.value)).T



#-------------------------------------------------------------------------------
# STACKED VOIDS

# Stack all voids relative to their barycentre:

# Get points within an ellipse of radius R, and ellipticity eps, given a z
# position and displacement from the axis
# Get and save the line of sight positions:

from void_analysis.simulation_tools import redshift_space_positions
from void_analysis.simulation_tools import get_los_pos_for_snapshot
from void_analysis.simulation_tools import get_los_positions_for_all_catalogues
from void_analysis.plot import draw_ellipse, plot_los_void_stack


#-------------------------------------------------------------------------------
# COSMOLOGY CONNECTION



#-------------------------------------------------------------------------------
# FUNCTIONS LIBRARY

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
                          F_inv=None,log_density=False,**kwargs):
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
                        Delta(np.sqrt(r**2 + rperp_vals[i]**2)) \
                        - svals[j])
                    rvals[i,j] = scipy.optimize.fsolve(F,svals[j])
            F_inv = lambda x, y: scipy.interpolate.interpn((rperp_vals,svals),
                                                           rvals,
                                                           np.vstack((x,y)).T,
                                                           method='cubic')
            theory_val = z_space_profile(s_par_new,s_perp_new,
                                         lambda r: rho_real(r,*profile_args),
                                         z,Om,Delta,delta,f=f,F_inv=F_inv,
                                         **kwargs)
        else:
            theory_val = z_space_profile(s_par_new,s_perp_new,
                                         lambda r: rho_real(r,*profile_args),
                                         z,Om,Delta,delta,f=f,
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
                                         z,Om,Delta,delta,f=f,**kwargs)
            if normalised:
                delta_rho[k] = 1.0 - theory_val/data_val
            else:
                delta_rho[k] = data_val - theory_val
    if cholesky:
        # We assume that the covariance is given in it's lower triangular form,
        # rather than an explicit covariance. We then solve this rather than
        # actually computing 
        x = scipy.linalg.solve_triangular(inv_cov,delta_rho,lower=True)
        return -0.5*np.sum(x**2)  - (M/2)*np.log(2*np.pi) - \
             np.sum(np.log(np.diag(inv_cov)))
    elif singular:
        if (Umap is None) or (good_eig is None):
            raise Exception("Must provide Umap and good_eigenvalues for " + 
                "handling singular covariance matrices")
        return compute_singular_log_likelihood(delta_rho,Umap,good_eig)
    else:
        return -0.5*np.matmul(np.matmul(delta_rho,inv_cov),delta_rho.T)

# Likelihood function, parallelised. Requires global variables!:
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

def log_probability_aptest_parallel(theta,*args,**kwargs):
    lp = log_prior_aptest(theta,**kwargs)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_aptest_parallel(theta,*args,**kwargs)

from void_analysis import context
#hrList = [snap.halos() for snap in snapListRev]
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


def get_field_from_los_data(los_data,z_bins,d_bins,v_weight,void_count,
                            nbar=None):
    cell_volumes_reff = np.outer(np.diff(z_bins),np.diff(d_bins))
    hist = np.histogramdd(los_data,bins=[z_bins,d_bins],density=False,
                          weights = v_weight/(2*np.pi*los_data[:,1]))
    if nbar is not None:
        return hist[0]/(2*void_count*cell_volumes_reff*nbar)
    else:
        return hist[0]/(2*void_count*cell_volumes_reff)


def profile_broken_power_log(r,A,r0,c1,f1,B):
    return np.log(np.abs(A + B*(r/r0)**2 + (r/r0)**4)) + \
        ((c1 - 4)/f1)*np.log(1 + (r/r0)**f1)

def profile_broken_power(r,A,r0,c1,f1,B):
    return np.exp(profile_broken_power_log(r,A,r0,c1,f1,B))

# Modified Hamaus profile:
def profile_modified_hamaus(r,alpha,beta,rs,delta_c,delta_large = 0.0,rv=1.0):
    return (delta_c - delta_large)*(1.0 - (r/rs)**alpha)/(1 + (r/rv)**beta) \
        + 1.0 + delta_large


def rho_real(r,*profile_args):
    #return profile_broken_power(r,A,r0,c1,f1,B)
    return profile_modified_hamaus(r,*profile_args)


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




#-------------------------------------------------------------------------------
# LOS LISTS

#los_list_lcdm = get_los_positions_for_all_catalogues(snapListUn,snapListRevUn,
#    antihaloCentresUn,antihaloRadiiUn,filter_list=filter_list_lcdm,
#    dist_max=60,rmin=10,rmax=20,zspace=True,recompute=False,
#    suffix=".lospos_zspace")
final_cat = cat300.get_final_catalogue(void_filter=True)
halo_indices = [-np.ones(len(final_cat),dtype=int) 
    for ns in range(0,len(snapList))]
for ns in range(0,len(snapList)):
    have_void = final_cat[:,ns] >= 0
    halo_indices[ns][have_void] = \
        cat300.indexListShort[ns][final_cat[have_void,ns]-1]

# Convert void centres to redshift space:
zcentres = tools.loadOrRecompute(data_folder + "zspace_centres.p",
                                 get_zspace_centres,halo_indices,snapList,
                                 snapListRev,hrlist=None,_recomputeData=False)

filter_list_borg = [halo_indices[ns] >= 0 for ns in range(0,len(snapList))]


# BORG particles, void only, z-space:
los_list_void_only_borg_zspace = get_los_positions_for_all_catalogues(snapList,
    snapListRev,zcentres,cat300.getAllProperties("radii",void_filter=True).T,
    all_particles=False,void_indices = halo_indices,
    filter_list=filter_list_borg,dist_max=3,rmin=10,rmax=20,recompute=False,
    zspace=True,recompute_zspace=False,suffix=".lospos_void_only_zspace2.p")

# BORG particles, all, z-space:
los_list_all_borg_zspace = get_los_positions_for_all_catalogues(snapList,
    snapListRev,zcentres,cat300.getAllProperties("radii",void_filter=True).T,
    all_particles=True,void_indices = halo_indices,
    filter_list=filter_list_borg,dist_max=3,rmin=10,rmax=20,recompute=False,
    zspace=True,recompute_zspace=False,suffix=".lospos_all_zspace2.p")

# BORG particles, Real space positions (void only):
los_list_void_only_borg = get_los_positions_for_all_catalogues(snapList,
    snapListRev,
    cat300.getAllCentres(void_filter=True),
    cat300.getAllProperties("radii",void_filter=True).T,all_particles=False,
    void_indices = halo_indices,filter_list=filter_list_borg,
    dist_max=3,rmin=10,rmax=20,recompute=False,suffix=".lospos_void_only2.p")
# BORG particles, Real space positions (all):
los_list_all_borg = get_los_positions_for_all_catalogues(snapList,
    snapListRev,
    cat300.getAllCentres(void_filter=True),
    cat300.getAllProperties("radii",void_filter=True).T,all_particles=True,
    void_indices = halo_indices,filter_list=filter_list_borg,
    dist_max=3,rmin=10,rmax=20,recompute=False,suffix=".lospos_all.p")


#-------------------------------------------------------------------------------
# FILTER LOS LISTS

# Filter the voids in unconstrained simulations to match those used in the 
# BORG catalogue.

# LCDM examples for comparison:
distances_from_centre_lcdm = [
    np.sqrt(np.sum(snapedit.unwrap(centres - np.array([boxsize/2]*3),
    boxsize)**2,1)) for centres in antihaloCentresUn]
filter_list_lcdm = [(dist < 135) & (radii > 10) & (radii <= 20) 
    for dist, radii in zip(distances_from_centre_lcdm,antihaloRadiiUn)]

# LCDM particles (centres), void only, z-space:
#los_list_void_only_lcdm_zspace = get_los_positions_for_all_catalogues(
#    snapListUn,snapListRevUn,antihaloCentresUn,antihaloRadiiUn,
#    all_particles=False,filter_list=filter_list_lcdm,dist_max=60,rmin=10,
#    rmax=20,recompute=False,zspace=True,recompute_zspace=False,
#    suffix=".lospos_void_only_zspace.p")

# LCDM particles (centres), all, z-space:
#los_list_all_lcdm_zspace = get_los_positions_for_all_catalogues(
#    snapListUn,snapListRevUn,antihaloCentresUn,antihaloRadiiUn,
#    all_particles=True,filter_list=filter_list_lcdm,dist_max=60,rmin=10,
#    rmax=20,recompute=False,zspace=True,recompute_zspace=False,
#    suffix=".lospos_all_zspace.p")

# Get LOS list for selection-effect-processed regions:

# Randomly selected regions (pre-selected) in unconstrained simulations:
[randCentres,randOverDen] = tools.loadPickle(data_folder2 + \
    "random_centres_and_densities.p")
# Density contrasts of these regions (precomputed):
deltaMCMCList = tools.loadPickle(data_folder2 + "delta_list.p")
from void_analysis.simulation_tools import get_map_from_sample

# MAP estimate of the density from these samples, using bootstrap:
deltaMAPBootstrap = scipy.stats.bootstrap(
    (deltaMCMCList,),get_map_from_sample,confidence_level = 0.68,
    vectorized=False,random_state=1000)
deltaMAPInterval = deltaMAPBootstrap.confidence_interval

# Select regions with similar density contrast the local supervolume:
comparableDensityMAP = [(delta <= deltaMAPInterval[1]) & \
    (delta > deltaMAPInterval[0]) for delta in randOverDen]
centresToUse = [randCentres[comp] for comp in comparableDensityMAP]
deltaToUse = [randOverDen[ns][comp] \
    for ns, comp in zip(range(0,len(snapList)),comparableDensityMAP)]
rSep = 2*135
# Filter out spheres which overlap with already selected spheres, to ensure
# independence:
indicesUnderdenseNonOverlapping = simulation_tools.getNonOverlappingCentres(
    centresToUse,rSep,boxsize,returnIndices=True)
centresUnderdenseNonOverlapping = [centres[ind] \
    for centres,ind in zip(centresToUse,indicesUnderdenseNonOverlapping)]
densityListUnderdenseNonOverlapping = [density[ind] \
    for density, ind in zip(comparableDensityMAP,\
    indicesUnderdenseNonOverlapping)]
densityUnderdenseNonOverlapping = np.hstack(
    densityListUnderdenseNonOverlapping)

# Get distances from the centre of each region for each of the voids in that
# region:
distances_from_centre_lcdm_selected = [[
    np.sqrt(np.sum(snapedit.unwrap(centres - sphere_centre,boxsize)**2,1))
    for sphere_centre in selected_regions]
    for centres, selected_regions in zip(antihaloCentresUn,
    centresUnderdenseNonOverlapping)]

# Filter voids to the radius range used in the catalogue:
filter_list_lcdm_by_region = [[
    (dist < 135) & (radii > 10) & (radii <= 20) 
    for dist in all_dists]
    for all_dists, radii in 
    zip(distances_from_centre_lcdm_selected,antihaloRadiiUn)]

filter_list_lcdm_selected = [np.any(np.array([
    (dist < 135) & (radii > 10) & (radii <= 20) 
    for dist in all_dists]),0)
    for all_dists, radii in 
    zip(distances_from_centre_lcdm_selected,antihaloRadiiUn)]



#-------------------------------------------------------------------------------
# LOS LISTS FOR FILTERED VOIDS


# LCDM particles, density selected, void only, real space:
los_list_void_only_selected_lcdm = get_los_positions_for_all_catalogues(
    snapListUn,snapListRevUn,antihaloCentresUn,antihaloRadiiUn,
    all_particles=False,filter_list=filter_list_lcdm_by_region,dist_max=3,
    rmin=10,rmax=20,recompute=False,suffix=".lospos_void_only_selected.p")

# LCDM particles, density selected, all, real space:
los_list_all_selected_lcdm = get_los_positions_for_all_catalogues(
    snapListUn,snapListRevUn,antihaloCentresUn,antihaloRadiiUn,
    all_particles=True,filter_list=filter_list_lcdm_by_region,dist_max=3,
    rmin=10,rmax=20,recompute=False,suffix=".lospos_all_selected.p")

# LCDM particles, density selected, void only, z-space:
los_list_void_only_lcdm_zspace_selected = get_los_positions_for_all_catalogues(
    snapListUn,snapListRevUn,antihaloCentresUn,antihaloRadiiUn,
    all_particles=False,filter_list=filter_list_lcdm_by_region,dist_max=3,
    rmin=10,rmax=20,recompute=False,zspace=True,recompute_zspace=False,
    suffix=".lospos_void_only_zspace_selected.p")

# LCDM particles, density selected, all, z-space:
los_list_all_lcdm_zspace_selected = get_los_positions_for_all_catalogues(
    snapListUn,snapListRevUn,antihaloCentresUn,antihaloRadiiUn,
    all_particles=True,filter_list=filter_list_lcdm_by_region,dist_max=3,
    rmin=10,rmax=20,recompute=False,zspace=True,recompute_zspace=False,
    suffix=".lospos_all_zspace_selected.p")

los_list_void_only_selected_lcdm_flat = [x 
    for y in los_list_void_only_selected_lcdm for x in y]

los_list_all_selected_lcdm_flat = [x 
    for y in los_list_all_selected_lcdm for x in y]


# Combined Lists
# Combine the voids from different regions of the same simulation
# so that we can use them in the same way as the BORG samples:
los_list_void_only_combined_lcdm = [combine_los_lists(los_list) 
    for los_list in los_list_void_only_selected_lcdm]

los_list_void_only_combined_lcdm_zspace = [combine_los_lists(los_list) 
    for los_list in los_list_void_only_lcdm_zspace_selected]

los_list_all_combined_lcdm = [combine_los_lists(los_list) 
    for los_list in los_list_all_selected_lcdm]

los_list_all_combined_lcdm_zspace = [combine_los_lists(los_list) 
    for los_list in los_list_all_lcdm_zspace_selected]

#dist_all_combined_lcdm_zspace = [[np.sqrt(np.sum(los**2,1))/rad 
#    for los, rad in zip(all_los,all_rad)] 
#    for all_los, all_rad in 
#    zip(los_list_all_combined_lcdm_zspace,antihaloRadiiUn)]

#los_list_void_only_lcdm = get_los_positions_for_all_catalogues(snapListUn,
#    snapListRevUn,antihaloCentresUn,antihaloRadiiUn,all_particles=False,
#    filter_list=filter_list_lcdm,dist_max=60,rmin=10,rmax=20,recompute=False,
#    suffix=".lospos_void_only.p")

#-------------------------------------------------------------------------------
# CONSTRUCT FIELD USED FOR INFERENCE

# Get relevant LOS lists:
los_lcdm_real = los_list_all_combined_lcdm
los_lcdm_zspace = los_list_all_combined_lcdm_zspace
los_borg_real = los_list_all_borg
los_borg_zspace = los_list_all_borg_zspace

# Bins in LOS co-ords:
upper_dist_reff = 2
bins_z_reff = np.linspace(0,upper_dist_reff,21)
bins_d_reff = np.linspace(0,upper_dist_reff,21)
bin_z_centres = plot.binCentres(bins_z_reff)
bin_d_centres = plot.binCentres(bins_d_reff)

# Stacked void particles in 2d (in redshift space):
stacked_particles_reff_lcdm_abs = get_2d_void_stack_from_los_pos(
    los_lcdm_zspace,bins_z_reff,bins_d_reff,
    antihaloRadiiUn)
void_radii_borg = cat300.getMeanProperty("radii",void_filter=True)[0]
void_individual_radii_borg = cat300.getAllProperties("radii",void_filter=True)
stacked_particles_reff_borg_abs = get_2d_void_stack_from_los_pos(
    los_borg_zspace,bins_z_reff,bins_d_reff,
    [void_radii_borg for rad in antihaloRadii])

# Stacked void_particles in 1d:
# We can use the real space profile for this:
stacked_particles_reff_lcdm_real = get_2d_void_stack_from_los_pos(
    los_lcdm_real,bins_z_reff,bins_d_reff,antihaloRadiiUn)
radlist = [antihaloRadiiUn[k] 
    for k in range(0,20) for x in filter_list_lcdm_by_region[k]]
stacked_particles_reff_lcdm_real_all = [get_2d_void_stack_from_los_pos(
    [los],bins_z_reff,bins_d_reff,[rad])
    for los, rad in zip(los_list_all_selected_lcdm_flat,radlist)]
#stacked_particles_reff_borg_real = get_2d_void_stack_from_los_pos(
#    los_list_void_only_borg,bins_z_reff,bins_d_reff,
#    [void_radii_borg for rad in antihaloRadii])
stacked_particles_reff_borg_real = get_2d_void_stack_from_los_pos(
    los_borg_real,bins_z_reff,bins_d_reff,
    [void_individual_radii_borg[:,k] for k in range(0,20)])
stacked_particles_reff_borg_real_all = [get_2d_void_stack_from_los_pos(
    [los_borg_real[k]],bins_z_reff,bins_d_reff,
    [void_individual_radii_borg[:,k]]) for k in range(0,len(snapList))]
stacked_1d_real_lcdm = np.sqrt(np.sum(stacked_particles_reff_lcdm_real**2,1))
stacked_1d_real_borg = np.sqrt(np.sum(stacked_particles_reff_borg_real**2,1))
stacked_1d_real_lcdm_all = [np.sqrt(np.sum(stacked_los**2,1)) 
    for stacked_los in stacked_particles_reff_lcdm_real_all]
stacked_1d_real_borg_all = [np.sqrt(np.sum(stacked_los**2,1)) 
    for stacked_los in stacked_particles_reff_borg_real_all]

# Unweighted bin counts:
[_,noInBins_lcdm] = plot_utilities.binValues(stacked_1d_real_lcdm,bins_d_reff)
[_,noInBins_borg] = plot_utilities.binValues(stacked_1d_real_borg,bins_d_reff)

noInBins_lcdm_all = [plot_utilities.binValues(stacked_dist,bins_d_reff)[1]
    for stacked_dist in stacked_1d_real_lcdm_all]
noInBins_borg_all = [plot_utilities.binValues(stacked_dist,bins_d_reff)[1]
    for stacked_dist in stacked_1d_real_borg_all]


#-------------------------------------------------------------------------------
# COMPUTE WEIGHTS FOR DATA FIELD

# Weights for each void in the stack:
voids_used_lcdm = [np.array([len(x) for x in los]) > 0 
    for los in los_lcdm_zspace]
voids_used_lcdm_all = [np.array([len(x) for x in los]) > 0 
    for los in los_list_all_selected_lcdm_flat]
voids_used_lcdm_ind = [np.where(x)[0] for x in voids_used_lcdm]
voids_used_borg = [np.array([len(x) for x in los]) > 0 
    for los in los_borg_zspace]
void_radii_lcdm = [rad[filt] 
    for rad, filt in zip(antihaloRadiiUn,voids_used_lcdm)]
void_radii_lcdm_all = [rad[filt] 
    for rad, filt in zip(radlist,voids_used_lcdm_all)]


los_pos_lcdm = [ [los[x] for x in np.where(ind)[0]] 
    for los, ind in zip(los_lcdm_zspace,voids_used_lcdm) ]
los_pos_borg = [ [los[x] for x in np.where(ind)[0]] 
    for los, ind in zip(los_borg_zspace,voids_used_borg) ]
los_pos_borg_real = [ [los[x] for x in np.where(ind)[0]] 
    for los, ind in zip(los_borg_real,voids_used_borg) ]
los_pos_lcdm_all = [ [los[x] for x in np.where(ind)[0]] 
    for los, ind in 
    zip(los_list_all_selected_lcdm_flat,voids_used_lcdm_all) ]

rep_scores = void_cat_frac = cat300.property_with_filter(
    cat300.finalCatFrac,void_filter=True)

all_rep_scores = np.hstack([rep_scores[used] for used in voids_used_borg])
v_weight_borg = get_weights_for_stack(
    los_pos_borg,[void_radii_borg[used] for used in voids_used_borg],
    additional_weights = [rep_scores[used]/np.sum(all_rep_scores) 
    for used in voids_used_borg])
v_weight_borg_unweighted = get_weights_for_stack(
    los_pos_borg,[void_radii_borg[used] for used in voids_used_borg],
    additional_weights = None)
v_weight_lcdm = get_weights_for_stack(los_pos_lcdm,void_radii_lcdm)


# Weighted bin counts:
v_weight_borg_all = [get_weights_for_stack(
    [los_pos_borg_real[k]],[void_individual_radii_borg[voids_used_borg[k],k]],
    additional_weights = [rep_scores[voids_used_borg[k]]/\
    np.sum(rep_scores[voids_used_borg[k]])]) for k in range(0,len(snapList))]
v_weight_lcdm_all = [get_weights_for_stack(
    [los_pos_lcdm_all[k]],[void_radii_lcdm_all[k]],
    additional_weights = \
    [np.ones(len(void_radii_lcdm_all[k]))/len(void_radii_lcdm_all[k])])
    for k in range(0,len(void_radii_lcdm_all))]


weighted_counts_borg = np.array([
    np.histogram(dist,bins=bins_d_reff,weights=weights)[0]
    for dist, weights in zip(stacked_1d_real_borg_all,v_weight_borg_all)])
weighted_counts_lcdm = np.array([
    np.histogram(dist,bins=bins_d_reff,weights=weights)[0]
    for dist, weights in zip(stacked_1d_real_lcdm_all,v_weight_lcdm_all)])


# Fields:
nbar = len(referenceSnap)/boxsize**3
num_voids_lcdm = np.sum([np.sum(x) for x in voids_used_lcdm]) 
cell_volumes_reff = np.outer(np.diff(bins_z_reff),np.diff(bins_d_reff))
field_lcdm = get_field_from_los_data(stacked_particles_reff_lcdm_abs,
                                     bins_z_reff,bins_d_reff,v_weight_lcdm,
                                     num_voids_lcdm,nbar=nbar)

num_voids_sample_borg = np.array([np.sum(x) for x in voids_used_borg])
num_voids_borg = np.sum(num_voids_sample_borg) # Not the actual number
    # but the effective number being stacked, so the number of voids multiplied
    # by the number of samples.
nmean = len(snapList[0])/(boxsize**3)

# Field with the reproducibility score weighting applied:
field_borg = get_field_from_los_data(stacked_particles_reff_borg_abs,
                                     bins_z_reff,bins_d_reff,v_weight_borg,1,
                                     nbar=nbar)

# Field without reproducibility score weighting
field_borg_unweighted = get_field_from_los_data(
    stacked_particles_reff_borg_abs,bins_z_reff,bins_d_reff,
    v_weight_borg_unweighted,num_voids_borg,nbar=nbar)


#-------------------------------------------------------------------------------
# REAL SPACE DENSITY FIELD

# Get the matter density fields:
# Better, just load them directly from the pre-computed profiles:
[noConstraintsDict,regionAndVoidDensityConditionDict,
     rBinStackCentres,nbar,rhoMCMCToUse,sigmaRhoMCMCToUse] = \
         tools.loadPickle(data_folder2 + "profile_plot_data.p")

# Posterior profiles:
[allPairsSample,allVolumesSample] = tools.loadPickle(
    data_folder2 + "pair_counts_mcmc_cut_samples.p")
[delta_borg, sigma_delta_borg] = [x/nbar 
    for x in stacking.get_mean_mcmc_profile(
    allPairsSample,allVolumesSample,cumulative = False)]
[Delta_borg, sigma_Delta_borg] = [x/nbar 
    for x in stacking.get_mean_mcmc_profile(allPairsSample,allVolumesSample,
    cumulative = True)]

# LCDM profiles:
all_pairs = regionAndVoidDensityConditionDict['pairs']
all_volumes = regionAndVoidDensityConditionDict['volumes']
delta_lcdm_all = stacking.get_profiles_in_regions(all_pairs,all_volumes,
                                              cumulative=False)
Delta_lcdm_all = np.array([(np.sum(np.cumsum(all_pairs[k],1),0)+1)/\
            np.sum(np.cumsum(all_volumes[k],1),0) \
            for k in range(0,len(all_volumes))])
delta_lcdm = np.mean(delta_lcdm_all,0)/nbar
Delta_lcdm = np.mean(Delta_lcdm_all,0)/nbar
sigma_delta_lcdm = np.std(delta_lcdm_all,0)/nbar
sigma_Delta_lcdm = np.std(Delta_lcdm_all,0)/nbar

# Profile function (should we use the inferred profile, or the 
# lcdm mock profiles?):
r_bin_centres = plot_utilities.binCentres(bins_d_reff)
#rho_r = noInBins_borg/np.sum(noInBins_borg)
rho_r = noInBins_lcdm/(np.sum(noInBins_lcdm)*\
    4*np.pi*(bins_d_reff[1:]**3 - bins_d_reff[0:-1]**3)/3)
rho_r_error = np.sqrt(noInBins_lcdm)/(np.sum(noInBins_lcdm)*\
    4*np.pi*(bins_d_reff[1:]**3 - bins_d_reff[0:-1]**3)/3)


#rho_r_all = np.array([n/(np.sum(n)*\
#    4*np.pi*(bins_d_reff[1:]**3 - bins_d_reff[0:-1]**3)/3)
#    for n in noInBins_lcdm_all])

rho_r_all = weighted_counts_lcdm/\
    (4*np.pi*(bins_d_reff[1:]**3 - bins_d_reff[0:-1]**3)*nbar/3)

rho_r_std = np.std(rho_r_all,0)
rho_r_mean = np.mean(rho_r_all,0)
rho_r_range_1sigma = np.percentile(rho_r_all,[16,84],axis=0)
rho_r_range_2sigma = np.percentile(rho_r_all,[2.5,97.5],axis=0)


rho_borg_r = noInBins_borg/(np.sum(noInBins_borg)*\
    4*np.pi*(bins_d_reff[1:]**3 - bins_d_reff[0:-1]**3)/3)
rho_borg_r_error = np.sqrt(noInBins_borg)/(np.sum(noInBins_borg)*\
    4*np.pi*(bins_d_reff[1:]**3 - bins_d_reff[0:-1]**3)/3)

#rho_r_borg_all = np.array([n/(np.sum(n)*\
#    4*np.pi*(bins_d_reff[1:]**3 - bins_d_reff[0:-1]**3)/3)
#    for n in noInBins_borg_all])
rho_r_borg_all =  weighted_counts_borg/\
    (4*np.pi*(bins_d_reff[1:]**3 - bins_d_reff[0:-1]**3)*nbar/3)
rho_r_borg_std = np.std(rho_r_borg_all,0)
rho_r_borg_mean = np.mean(rho_r_borg_all,0)
rho_r_borg_range_1sigma = np.percentile(rho_r_borg_all,[16,84],axis=0)
rho_r_borg_range_2sigma = np.percentile(rho_r_borg_all,[2.5,97.5],axis=0)

Delta_r = np.cumsum(noInBins_borg)/np.sum(noInBins_borg)
delta_func = scipy.interpolate.interp1d(
    rBinStackCentres,delta_borg - 1.0,kind='cubic',
    fill_value=(delta_borg[0],delta_borg[-1]),bounds_error=False)
Delta_func = scipy.interpolate.interp1d(
    rBinStackCentres,Delta_borg - 1.0,kind='cubic',
    fill_value=(Delta_borg[0],Delta_borg[-1]),bounds_error=False)
rho_func = scipy.interpolate.interp1d(
    r_bin_centres,rho_r,kind='cubic',
    fill_value=(rho_r[0],rho_r[-1]),bounds_error=False)
rho_func_borg = scipy.interpolate.interp1d(
    r_bin_centres,rho_borg_r,kind='cubic',
    fill_value=(rho_borg_r[0],rho_borg_r[-1]),bounds_error=False)
rvals = np.linspace(np.min(r_bin_centres),np.max(r_bin_centres),1000)

rho_func_lcdm_z0 = scipy.interpolate.interp1d(
    r_bin_centres,field_lcdm[0],kind='cubic',
    fill_value=(field_lcdm[0][0],field_lcdm[0][-1]),
    bounds_error=False)
rho_func_borg_z0 = scipy.interpolate.interp1d(
    r_bin_centres,field_borg[0],kind='cubic',
    fill_value=(field_borg[0][0],field_borg[0][-1]),
    bounds_error=False)

#-------------------------------------------------------------------------------
# TEST PLOTS FOR REAL SPACE FIELD

# Test Plot:
fig, ax = plt.subplots()
ax.plot(rvals,delta_func(rvals),label='$\\delta(r)$')
ax.plot(rvals,Delta_func(rvals),label='$\\Delta(r)$')
ax.set_xlabel('r [$\\mathrm{Mpc}h^{-1}$]')
ax.set_ylabel('$\\rho(r)$')
plt.legend(frameon=False)
plt.savefig(figuresFolder + "rho_real_plot.pdf")
plt.show()

start_values = np.linspace(0,0.5,101)

x = r_bin_centres
y = rho_r_mean

fit_large = np.polyfit(np.log(x[x > 1.5]),np.log(y[x > 1.5]),1)
fit_small = np.polyfit(np.log(x[x < 0.5]),np.log(y[x < 0.5]),1)

plt.clf()
fig, ax = plt.subplots()
ax.fill_between(r_bin_centres,rho_r_range_2sigma[0,:],rho_r_range_2sigma[1,:],
    alpha=0.25,color=seabornColormap[0],ec=None,
    label='$\\rho_{\\Lambda\\mathrm{CDM}}(r), 95\\%$')
ax.fill_between(r_bin_centres,rho_r_range_1sigma[0,:],rho_r_range_1sigma[1,:],
    alpha=0.5,color=seabornColormap[0],ec=None,
    label='$\\rho_{\\Lambda\\mathrm{CDM}}(r), 68\\%$')
ax.errorbar(r_bin_centres,rho_r_borg_mean,
    yerr=np.abs(rho_r_borg_range_1sigma - rho_r_borg_mean),
    label='$\\rho_{\\mathrm{borg}}(r)$',color=seabornColormap[1])
#ax.plot(r_bin_centres,field_borg[0]/np.mean(rho_func_borg_z0(start_values)),
#    label='$\\rho_{\\mathrm{2d}}(0,d)$')
ax.plot(rvals,np.exp(fit_large[0]*np.log(rvals) + fit_large[1]),
        linestyle=':',color='k',
        label="$\\rho = " + ("%.2g" % (np.exp(fit_large[1]))) + "\\cdot " + 
        "r^{" + ("%.2g" % (fit_large[0])) + "}$")
ax.plot(rvals,np.exp(fit_small[0]*np.log(rvals) + fit_small[1]),
        linestyle='--',color='k',
        label="$\\rho = " + ("%.2g" % (np.exp(fit_small[1]))) + "\\cdot " + 
        "r^{" + ("%.2g" % (fit_small[0])) + "}$")
ax.set_xlabel('$r/r_{\\mathrm{eff}}$')
ax.set_ylabel('$\\rho(r)$')
#ax.set_yscale('log')
#ax.set_xscale('log')
ax.set_ylim([1e-3,0.1])
ax.set_xlim([0,2])
plt.legend(frameon=False)
plt.savefig(figuresFolder + "rho_real_plot_void_only.pdf")
plt.show()

# 2D profile function test (zspace):
z = 0.0225
profile_2d = np.zeros((len(bins_z_reff)-1,len(bins_d_reff)-1))

Om = 0.3111
f = f_lcdm(z,Om)
A = 0.013

#-------------------------------------------------------------------------------
# DENSITY PROFILE FITS PLOTS

# Test plot:
plot_los_void_stack(\
        field_lcdm,bin_d_centres,bin_z_centres,
        cmap='Blues',
        vmin=0,vmax=1e-1,fontsize=10,
        xlabel = '$s_{\\perp}/R_{\\mathrm{eff}}$ (Perpendicular distance)',
        ylabel = '$s_{\\parallel}/R_{\\mathrm{eff}}$ (LOS distance)',
        density_unit='probability',
        savename=figuresFolder + "profile_2d_test_data_lcdm.pdf",
        title=None,colorbar=True,shrink=0.9,
        colorbar_title="$\\rho(s_{\\parallel},s_{\\perp})$")


plot_los_void_stack(\
        field_borg,bin_d_centres,bin_z_centres,
        cmap='PuOr_r',
        vmin=0,vmax=2,fontsize=10,
        xlabel = '$s_{\\perp}/R_{\\mathrm{eff}}$ (Perpendicular distance)',
        ylabel = '$s_{\\parallel}/R_{\\mathrm{eff}}$ (LOS distance)',
        fontfamily='serif',
        density_unit='probability',
        savename=figuresFolder + "profile_2d_test_data_borg.pdf",
        title=None,colorbar=True,shrink=0.9,
        colorbar_title="$\\rho(s_{\\parallel},s_{\\perp})$")

plt.clf()
fig, ax = plt.subplots(1,2,figsize=(textwidth,0.45*textwidth))
plot_los_void_stack(\
        field_borg_unweighted,bin_d_centres,bin_z_centres,
        cmap='PuOr_r',ax= ax[0],
        vmin=0,vmax=2,fontsize=10,
        xlabel = '$s_{\\perp}/R_{\\mathrm{eff}}$',
        ylabel = '$s_{\\parallel}/R_{\\mathrm{eff}}$ (LOS distance)',
        fontfamily='serif',
        density_unit='probability',
        savename=None,
        title=None,colorbar=False,shrink=0.9,
        colorbar_title="$\\rho(s_{\\parallel},s_{\\perp})$")
im = plot_los_void_stack(\
        field_borg,bin_d_centres,bin_z_centres,
        cmap='PuOr_r',ax= ax[1],
        vmin=0,vmax=2,fontsize=10,
        xlabel = '$s_{\\perp}/R_{\\mathrm{eff}}$',
        ylabel = '$s_{\\parallel}/R_{\\mathrm{eff}}$ (LOS distance)',
        fontfamily='serif',
        density_unit='probability',
        savename=None,
        title=None,colorbar=False,shrink=0.9,
        colorbar_title="$\\rho(s_{\\parallel},s_{\\perp})$")
plt.subplots_adjust(hspace=0.0,wspace=0.0,
                    left=0.12,right=0.95,bottom=0.15,top=0.95)
fig.colorbar(im,shrink=0.9,label="$\\rho(s_{\\parallel},s_{\\perp})$",
             ax=ax.ravel().tolist())
ax[1].yaxis.label.set_visible(False)
ax[1].yaxis.set_major_formatter(NullFormatter())
ax[1].yaxis.set_minor_formatter(NullFormatter())
ax[0].set_title("Unweighted Stack")
ax[1].set_title("Weighted Stack")
plt.savefig(figuresFolder + "profile_2d_weighting_comparison.pdf")
plt.show()

plt.clf()
plot_los_void_stack(\
        (field_borg - field_borg_unweighted)/field_borg_unweighted,
        bin_d_centres,bin_z_centres,
        cmap='PuOr_r',
        vmin=-0.5,vmax=0.5,fontsize=10,
        xlabel = '$s_{\\perp}/R_{\\mathrm{eff}}$',
        ylabel = '$s_{\\parallel}/R_{\\mathrm{eff}}$ (LOS distance)',
        fontfamily='serif',
        density_unit='probability',
        savename=figuresFolder + "profile_2d_weighted-unweighted.pdf",
        title=None,colorbar=True,shrink=0.9,
        colorbar_title="$\\Delta\\rho(s_{\\parallel},s_{\\perp})/" + \
        "\\rho(s_{\\parallel},s_{\\perp})$")
plt.show()

#-------------------------------------------------------------------------------
# COSMIOLOGICAL INFERENCE:

#-------------------------------------------------------------------------------
# COVARIANCE MATRIX ESTIMATION:
los_list_reff_borg = get_2d_void_stack_from_los_pos(
    los_borg_zspace,bins_z_reff,bins_d_reff,
    [void_radii_borg for rad in antihaloRadii],stacked=False)



v_weights_all_borg = get_weights_for_stack(
    los_pos_borg,[void_radii_borg[used] for used in voids_used_borg],
    additional_weights = [rep_scores[used]/np.sum(all_rep_scores) 
    for used in voids_used_borg],stacked=False)

v_weights_all_borg_scalar = np.hstack([np.array([x[0] for x in y]) 
    for y in v_weights_all_borg])

all_fields_borg = [[
    get_field_from_los_data(los,bins_z_reff,bins_d_reff,v_weight,1) 
    for los, v_weight in zip(los_vals,v_weights)] 
    for los_vals, v_weights in zip(los_list_reff_borg,v_weights_all_borg)]


f_lengths = [np.array([len(los) for los in all_los])
    for all_los in los_list_reff_borg]
f_totals = np.array([np.sum(x) for x in f_lengths])

f_weights = [length/total for length, total in zip(f_lengths,f_totals)]

cov_data = [np.cov(np.vstack([x.flatten() for x in fields]).T,
    aweights = weights) for fields, weights in zip(all_fields_borg,f_weights)]

stacked_weights = (np.hstack(f_lengths)/np.sum(f_totals))*\
    v_weights_all_borg_scalar
stacked_fields = np.vstack([np.vstack([x.flatten() for x in fields]) 
    for fields in all_fields_borg]).T
mean_fields = np.average(stacked_fields,axis=1,weights=stacked_weights)
mean_products = np.outer(mean_fields,mean_fields)
stacked_cov = np.cov(stacked_fields,aweights = stacked_weights)
symmetric_cov = (stacked_cov + stacked_cov.T)/2

lambda_reg = 1e-20
regularised_cov = symmetric_cov + lambda_reg*np.identity(symmetric_cov.shape[0])
#tools.minmax(np.real(np.linalg.eig(regularised_cov)[0]))
eigen = np.real(np.linalg.eig(regularised_cov)[0])

normalised_cov = regularised_cov/mean_products

L = np.linalg.cholesky(regularised_cov)
P = np.linalg.inv(L)
inv_cov = np.matmul(P,P.T)

# Covariance over all posterior samples:
los_list_sample_borg = [np.vstack(los) for los in los_list_reff_borg]
v_weights_sample_borg = [np.hstack(weights) for weights in v_weights_all_borg]
sample_fields_borg = np.array([
    get_field_from_los_data(los,bins_z_reff,bins_d_reff,v_weight,1) 
    for los, v_weight, count in zip(los_list_sample_borg,v_weights_sample_borg,
    num_voids_sample_borg)])

jackknife_samples = np.array([np.mean(sample_fields_borg[
    np.setdiff1d(range(0,sample_fields_borg.shape[0]),k),:,:],0).flatten() 
    for k in range(0,sample_fields_borg.shape[0])]).T

jackknife_cov = np.cov(jackknife_samples)
jackknife_mean = np.mean(jackknife_samples,1)
norm_cov = jackknife_cov/np.outer(jackknife_mean,jackknife_mean)
reg_norm_cov = regularise_covariance(norm_cov,lambda_reg= 1e-10)
reg_cov = regularise_covariance(jackknife_cov,lambda_reg= 1e-12)
cholesky_cov = scipy.linalg.cholesky(reg_cov,lower=True)

inv_cov = get_inverse_covariance(norm_cov,lambda_reg = 1e-10)
#eigen = np.real(np.linalg.eig(norm_cov)[0])

# Jackknife over all voids:
num_voids = stacked_fields.shape[1]

jackknife_samples_all = np.array([np.average(
    stacked_fields[:,range_excluding(0,num_voids,k)],axis=1,
    weights=stacked_weights[range_excluding(0,num_voids,k)]) 
    for k in range(0,num_voids)]).T
jackknife_cov = np.cov(jackknife_samples)
jackknife_mean = np.mean(jackknife_samples,1)
norm_cov = jackknife_cov/np.outer(jackknife_mean,jackknife_mean)
reg_norm_cov = regularise_covariance(norm_cov,lambda_reg= 1e-10)


# Bootstrap over all voids:
n_boot = 10000
np.random.seed(42)
bootstrap_samples = np.random.choice(num_voids,size=(num_voids,n_boot))
bootstrap_stacks = np.array([np.average(
    stacked_fields[:,bootstrap_samples[:,k]],
    axis=1,weights=stacked_weights[bootstrap_samples[:,k]]) 
    for k in tools.progressbar(range(0,n_boot))]).T
jackknife_cov = np.cov(bootstrap_stacks)
jackknife_mean = np.mean(bootstrap_stacks,1)
norm_cov = jackknife_cov/np.outer(jackknife_mean,jackknife_mean)
reg_norm_cov = regularise_covariance(norm_cov,lambda_reg= 1e-10)
reg_cov = regularise_covariance(jackknife_cov,lambda_reg= 1e-15)
cholesky_cov = scipy.linalg.cholesky(reg_cov,lower=True)

inv_cov = get_inverse_covariance(reg_cov,lambda_reg = 1e-23)

#logrho_mean = np.mean(np.log(bootstrap_stacks),1)

log_samples = np.log(bootstrap_stacks)
finite_samples = np.where(np.all(np.isfinite(log_samples),0))[0]
#if np.min(bootstrap_stacks) > 0:
logrho_mean = np.mean(log_samples[:,finite_samples],1)
logrho_cov = np.cov(log_samples[:,finite_samples])
#else:
#    logrho_mean = np.array(np.ma.mean(np.ma.masked_invalid(log_samples,1)))
#    logrho_cov = np.array(np.ma.cov(np.ma.masked_invalid(log_samples)))

logrho_reg_cov = regularise_covariance(logrho_cov,lambda_reg= 1e-7)

#logrho_mean = np.mean(log_samples,1)
#logrho_cov = np.cov(log_samples)
#logrho_reg_cov = regularise_covariance(logrho_cov,lambda_reg= 1e-27)

#-------------------------------------------------------------------------------
# TESTING EIGENVALUE DISTRIBUTION

# Eigenalue distribution:
eig, U = scipy.linalg.eigh(logrho_reg_cov)
eigen = np.real(np.linalg.eig(logrho_cov)[0])

plt.clf()
bins = np.logspace(-10,1,21)
plt.hist(eig,bins=bins,alpha=0.5,color=seabornColormap[1],cumulative=True,
         label="Cumulative eigenvalues")
plt.hist(eig,bins=bins,alpha=0.5,color=seabornColormap[0],
         label='Eigenvalues by bin')
plt.xlabel('Eigenvalue, $\\lambda$')
plt.ylabel('Number of eigenvectors')
plt.xscale('log')
plt.yscale('log')
plt.axhline(1600 - 1529,linestyle=':',color='k',label='Possibly singular d.o.f')
plt.legend(frameon=False)
plt.savefig(figuresFolder + "eigenvalue_distribution_log.pdf")
plt.show()



# Eigenalue distribution:
eig, U = scipy.linalg.eigh(reg_cov)
eigen = np.real(np.linalg.eig(jackknife_cov)[0])

plt.clf()
bins = np.logspace(-16,-7,21)
plt.hist(eig,bins=bins,alpha=0.5,color=seabornColormap[1],cumulative=True,
         label="Cumulative eigenvalues")
plt.hist(eig,bins=bins,alpha=0.5,color=seabornColormap[0],
         label='Eigenvalues by bin')
plt.xlabel('Eigenvalue, $\\lambda$')
plt.ylabel('Number of eigenvectors')
plt.xscale('log')
plt.yscale('log')
#plt.axhline(1600 - 1529,linestyle=':',color='k',
#            label='Possibly singular d.o.f')
plt.legend(frameon=False)
plt.savefig(figuresFolder + "eigenvalue_distribution.pdf")
plt.show()

# Eigenalue distribution (notmalised):
eig, U = scipy.linalg.eigh(reg_norm_cov)

plt.clf()
bins = np.logspace(-11,10,21)
plt.hist(eig,bins=bins,alpha=0.5,color=seabornColormap[1],cumulative=True,
         label="Cumulative eigenvalues")
plt.hist(eig,bins=bins,alpha=0.5,color=seabornColormap[0],
         label='Eigenvalues by bin')
plt.xlabel('Eigenvalue, $\\lambda$')
plt.ylabel('Number of eigenvectors')
plt.xscale('log')
plt.yscale('log')
#plt.axhline(1600 - 1529,linestyle=':',color='k',label='Possibly singular d.o.f')
plt.legend(frameon=False)
plt.savefig(figuresFolder + "eigenvalue_distribution_normalised.pdf")
plt.show()


#-------------------------------------------------------------------------------
# GAUSSIANITY TESTING

# Normality test:
import seaborn
i = 0
mu = jackknife_mean[i]
sigma = np.sqrt(jackknife_cov[i,i])
x_samples = np.linspace(mu - 3*sigma,mu + 3*sigma,1001)
y_samples = np.exp(-(x_samples - mu)**2/(2*sigma**2))/np.sqrt(2*np.pi*sigma**2)

plt.clf()
seaborn.kdeplot(bootstrap_stacks[i,:],color=seabornColormap[0],alpha=0.5,
                label = "")
plt.plot(x_samples,y_samples,linestyle=':',color='k',label='Gaussian fit')
plt.legend(frameon=False)
plt.xlabel('$\\rho [h^3\\mathrm{Mpc}^{-3}]$')
plt.ylabel('Probability Density')
plt.savefig(figuresFolder + "gaussian_test.pdf")
plt.show()

# Mardia's Test of normality:




# Singular Gaussian likelihood:

Umap, good_eig = get_nonsingular_subspace(jackknife_cov,1e-27,lambda_cut=1e-23,
                                          normalised_cov = False,
                                          mu=jackknife_mean)


#Umap, good_eig = get_nonsingular_subspace(logrho_cov,1e-27,lambda_cut=1e-23,
#                                          normalised_cov = False,
#                                          mu=logrho_mean)
singular = True


spar = np.hstack([s*np.ones(field_borg.shape[1]) 
    for s in plot.binCentres(bins_z_reff)])
sperp = np.hstack([plot.binCentres(bins_d_reff) 
    for s in plot.binCentres(bins_z_reff)])
scoords = np.vstack([spar,sperp]).T

data_filter = np.where((1.0/np.sqrt(np.diag(reg_norm_cov)) > 5) & \
        (np.sqrt(np.sum(scoords**2,1)) < 1.5) )[0]
#data_filter = np.array(range(0,bootstrap_stacks.shape[0]))


solved_residuals = get_solved_residuals(
    bootstrap_stacks[data_filter,:],reg_cov[data_filter,:][:,data_filter],
    jackknife_mean[data_filter],singular=True)
[A_test,B_test] = compute_normality_test_statistics(
    bootstrap_stacks[data_filter,:],
    covariance=reg_cov[data_filter,:][:,data_filter],
    xbar=jackknife_mean[data_filter],singular=True,
    solved_residuals=solved_residuals)

k, n = solved_residuals.shape
dof = k*(k+1)*(k+2)/6
pvalue = scipy.stats.chi2.sf(A_test,dof)
chi2_deviations = (A_test - dof)/np.sqrt(2*dof)

sample_filter = np.where(np.all(np.isfinite(log_samples[data_filter,:]),0))[0]
solved_residuals_log = get_solved_residuals(
    log_samples[data_filter,:][:,sample_filter],
    logrho_cov[data_filter,:][:,data_filter],
    logrho_mean[data_filter],singular=True)
[A_log,B_log] = compute_normality_test_statistics(
    log_samples[data_filter,:][:,sample_filter],
    covariance=logrho_cov[data_filter,:][:,data_filter],
    xbar=logrho_mean[data_filter],singular=True,
    solved_residuals=solved_residuals_log)
k, n = solved_residuals_log.shape
dof = k*(k+1)*(k+2)/6
pvalue = scipy.stats.chi2.sf(A_log,dof)
chi2_deviations_log = (A_log - dof)/np.sqrt(2*dof)


# Tests per dof:

test_values = np.zeros((len(data_filter),2))
for k in tools.progressbar(range(0,len(data_filter))):
    data = bootstrap_stacks[data_filter[k],:]
    n = len(data)
    xbar = np.mean(data)
    sigma = np.std(data)
    residuals = (data - xbar)/sigma
    product = np.outer(residuals,residuals)
    A = np.sqrt(n/6)*np.mean(residuals**3)
    B = np.sqrt(n/(24))*(np.sum(residuals**4)/n - 3)
    dof = 1
    test_values[k,0] = A
    test_values[k,1] = B

plt.clf()
fig, ax = plt.subplots(1,2)
xvals0 = np.linspace(-6,6,1001)
xvals1 = np.linspace(-6,6,1001)
seaborn.kdeplot(test_values[:,0],color=seabornColormap[0],ax=ax[0],
                alpha=0.5,fill=True,cut=0)
seaborn.kdeplot(test_values[:,1],color=seabornColormap[0],ax=ax[1],
                alpha=0.5,fill=True,cut=0)
ax[0].plot(xvals0,scipy.stats.norm.pdf(xvals0,0,1),linestyle='--',color='k')
ax[1].plot(xvals1,scipy.stats.norm.pdf(xvals1,0,1),linestyle='--',color='k')
ax[0].set_xlabel('$\\tilde{A}$ (Skewness $\\times \\sqrt{n/6}$)')
ax[1].set_xlabel('$\\tilde{B}$ (Kurtosis $\\times \\sqrt{n/24}$)')
#ax[0].set_yscale('log')
#ax[0].set_xscale('log')
plt.savefig(figuresFolder + "AB_distribution.pdf")
plt.show()


# Sample distribution plots:

plt.clf()
fig, ax = plt.subplots(1,2)
seaborn.kdeplot(bootstrap_stacks[data_filter,:].flatten(),
                color=seabornColormap[0],alpha=0.5,fill=True,
                label='Density distribution',ax=ax[0])
seaborn.kdeplot(log_samples[data_filter,:][:,sample_filter].flatten(),
                color=seabornColormap[0],alpha=0.5,fill=True,
                label='Log Density distribution',ax=ax[1])
ax[0].set_xlabel('$\\rho(s_{\\perp},s_{\\parallel})$')
ax[1].set_xlabel('$\\log(\\rho(s_{\\perp},s_{\\parallel}))$')
ax[0].set_ylabel('Probability Density')
ax[1].set_ylabel('Probability Density')
ax[0].set_title('Density')
ax[1].set_title('Log Density')
plt.tight_layout()
plt.savefig(figuresFolder + "density_distribution.pdf")
plt.show()


plt.clf()
nrows = 4
ncols = 4
sample = bootstrap_stacks[data_filter,:]
#selection = np.array(range(0,16),dtype=int).reshape((nrows,ncols))
np.random.seed(42)
selection = np.random.choice(sample.shape[0],
                             size=16,replace=False).reshape((nrows,ncols))
sort  = np.flip(np.argsort(test_values[np.where(test_values[:,1] > 4)[0],1]))
#selection = np.where(test_values[:,1] > 4)[0][sort][0:16].reshape((nrows,ncols))
selected_samples = sample[selection,:]
cumulative = False
width = 3e-5
fig, ax = plt.subplots(nrows,ncols)
for i in range(0,nrows):
    for j in range(0,ncols):
        axij = ax[i,j]
        xmean = np.mean(selected_samples[i,j,:])
        xstd = np.std(selected_samples[i,j,:])
        xmin = -5
        xmax = +5
        if cumulative:
            ymin = 0
            ymax = 1
        else:
            ymin = 0
            ymax = 0.5
        seaborn.kdeplot((selected_samples[i,j,:] - xmean)/xstd,
                color=seabornColormap[0],alpha=0.5,fill=False,
                ax=ax[i,j],cumulative=cumulative)
        #xvals = np.linspace(np.min(selected_samples[i,j,:]),
        #                    np.max(selected_samples[i,j,:]),101)
        xvals = np.linspace(xmin,xmax,1001)
        xbar = np.mean(selected_samples[i,j,:])
        sigma = np.std(selected_samples[i,j,:])
        axij.set_xlim([xmin,xmax])
        axij.set_ylim([1e-4,10*ymax])
        axij.set_yscale('log')
        axij.set_title("Skew. " + ("%.2g" % test_values[selection[i,j],0]) + 
                       ",Kurt. " + ("%.2g" % test_values[selection[i,j],1]),
                       x=0.5,y=0.8,fontsize=6)
        if cumulative:
            ax[i,j].plot(xvals,
                0.5 + 0.5*scipy.special.erf((xvals - xbar)/(np.sqrt(2)*sigma)),
                linestyle='--',color='k')
        else:
            ax[i,j].plot(xvals,
                np.exp(-0.5*(xvals**2))/np.sqrt(2*np.pi),
                linestyle='--',color='k')
        if j > 0:
            axij.yaxis.label.set_visible(False)
            axij.yaxis.set_major_formatter(NullFormatter())
            axij.yaxis.set_minor_formatter(NullFormatter())
        if i < nrows - 1:
            axij.xaxis.label.set_visible(False)
            axij.xaxis.set_major_formatter(NullFormatter())
            axij.xaxis.set_minor_formatter(NullFormatter())

fig.supxlabel('$(\\hat{\\rho}^s - \\bar{\\hat{\\rho}^s})/\\sigma_{\\hat{\\rho}^s}$')
#fig.supylabel('Probability Density')
plt.subplots_adjust(wspace=0.0,hspace=0.0,left=0.15,bottom=0.15,
                    top=0.95,right=0.95)
if cumulative:
    plt.savefig(figuresFolder + "gaussians_plot_cumulative.pdf")
else:
    plt.savefig(figuresFolder + "gaussians_plot.pdf")

plt.show()





# Chi2 distribution:
chi2 = np.sum(solved_residuals**2,0)
estimated_chi2_mean = int(np.round(np.mean(chi2)))
xvals = np.logspace(-4,4,10000)
yvals_1600 = scipy.stats.chi2.pdf(xvals,1600)
yvals_mean = scipy.stats.chi2.pdf(xvals,estimated_chi2_mean)

plt.clf()
seaborn.kdeplot(chi2,color=seabornColormap[0],alpha=0.5,
                label = "Chi^2 values")
#plt.plot(xvals,yvals_1600,linestyle=':',color='b',label='k=1600')
plt.plot(xvals,yvals_mean,linestyle='--',color='r',
         label='k=' + ("%.4g" % estimated_chi2_mean))
plt.xlabel('$\\chi^2$')
plt.ylabel('Probability Density')
plt.xscale('log')
plt.yscale('log')
plt.ylim([1e-8,3e-2])
#plt.xlim([1000,2000])
plt.xlim([100,1000])
plt.legend()
plt.savefig(figuresFolder + "chi_squared_plot.pdf")
plt.show()

chi2_log = np.sum(solved_residuals_log**2,0)
estimated_chi2_mean = int(np.round(np.mean(chi2_log)))
yvals_mean = scipy.stats.chi2.pdf(xvals,estimated_chi2_mean)
plt.clf()
seaborn.kdeplot(chi2_log,color=seabornColormap[0],alpha=0.5,
                label = "Chi^2 values")
#plt.plot(xvals,yvals_1600,linestyle=':',color='b',label='k=1600')
plt.plot(xvals,yvals_mean,linestyle='--',color='r',
         label='k=' + ("%.4g" % estimated_chi2_mean))
plt.xlabel('$\\chi^2$')
plt.ylabel('Probability Density')
plt.xscale('log')
plt.yscale('log')
plt.ylim([1e-8,3e-2])
#plt.xlim([1000,2000])
plt.xlim([100,1000])
plt.legend()
plt.savefig(figuresFolder + "chi_squared_plot_log.pdf")
plt.show()

# chi2 for each variable:
chi2_dist = np.mean(solved_residuals**2,1)
plt.clf()
seaborn.kdeplot(chi2_dist,color=seabornColormap[0],alpha=0.5,
                label = "Chi^2 values",log_scale=True)
plt.savefig(figuresFolder + "chi2_each_dof.pdf")
plt.show()

plt.clf()
C_diag = np.diag(reg_norm_cov).reshape((len(bin_d_centres),len(bin_z_centres)))
plt.imshow(np.sqrt(1.0/C_diag),cmap='PuOr_r',norm=colors.LogNorm(vmin=1/50,vmax=50),
           extent=(0,upper_dist_reff,0,upper_dist_reff),origin='lower')
plt.xlabel('$s_{\\mathrm{\\perp}}/R_{\\mathrm{eff}}$')
plt.ylabel('$s_{\\mathrm{\\parallel}}/R_{\\mathrm{eff}}$')
#plt.colorbar(label="$C_{ii} = \\frac{\\langle\\rho(s_{\\mathrm{\\perp},i}," + 
#    "s_{\\mathrm{\\parallel},i})\\rho(s_{\\mathrm{\\perp},i}," + 
#    "s_{\\mathrm{\\parallel},i})\\rangle}{\\langle\\rho(s_{\\mathrm{\\perp},i}," + 
#    "s_{\\mathrm{\\parallel},i})\\rangle\\langle\\rho(s_{\\mathrm{\\perp},i}," + 
#    "s_{\\mathrm{\\parallel},i})\\rangle}$")
plt.colorbar(label='S/N ratio')
plt.title("Estimated covariance of $\\rho(s_{\\mathrm{\\perp}}," + 
    "s_{\\mathrm{\\parallel}}$)")
plt.savefig(figuresFolder + "covariance_distribution.pdf")
plt.show()

#-------------------------------------------------------------------------------
# COSMOLOGICAL INFERENCE

# Eigenvalue quality filter:

Umap, good_eig = get_nonsingular_subspace(jackknife_cov,1e-27,lambda_cut=1e-23,
                                          normalised_cov = False,
                                          mu=jackknife_mean)


#Umap, good_eig = get_nonsingular_subspace(logrho_cov,1e-27,lambda_cut=1e-23,
#                                          normalised_cov = False,
#                                          mu=logrho_mean)
singular = True


spar = np.hstack([s*np.ones(field_borg.shape[1]) 
    for s in plot.binCentres(bins_z_reff)])
sperp = np.hstack([plot.binCentres(bins_d_reff) 
    for s in plot.binCentres(bins_z_reff)])
scoords = np.vstack([spar,sperp]).T

data_filter = np.where((1.0/np.sqrt(np.diag(reg_norm_cov)) > 5) & \
        (np.sqrt(np.sum(scoords**2,1)) < 1.5) )[0]


# Data to compare to:
data_field = field_borg.flatten()
spar = np.hstack([s*np.ones(field_borg.shape[1]) 
    for s in plot.binCentres(bins_z_reff)])
sperp = np.hstack([plot.binCentres(bins_d_reff) 
    for s in plot.binCentres(bins_z_reff)])
scoords = np.vstack([spar,sperp]).T
z = 0.0225

# Covariances plot:
plt.clf()
plt.imshow(reg_norm_cov,vmin=-1e-3,vmax=1e-3,cmap='PuOr_r')
plt.savefig(figuresFolder + "covariance_plot.pdf")
plt.show()

# 

theta_initial_guess = np.array([0.3,f_lcdm(z,0.3),0.01])
test = False
import time

if test:
    t0 = time.time()
    for k in tools.progressbar(range(0,100)):
        logp = log_probability_aptest(theta_initial_guess,data_field,scoords,
                                      cholesky_cov,z,Delta_func,delta_func,
                                      rho_real,Om_fid = 0.3111,cholesky=True,
                                      tabulate_inverse=True)
    t1 = time.time()
    average_time = (t1 - t0)/100


# Start with a MLE guess of the 1d profile parameters:

def log_likelihood(theta, x, y, yerr,profile_model):
    #rho0,p,C,rb = theta
    #A,r0,c1,f1,B = theta
    #model = profile_broken_power_log(x, A,r0,c1,f1,B)
    model = profile_model(x,*theta)
    sigma2 = yerr**2
    return -0.5 * np.sum( (y - model)**2/sigma2 + np.log(sigma2) )


nll1 = lambda *theta: -log_likelihood(*theta)
#initial = np.array([0.1,0.9,-9.5,3.5,0.01])
initial = np.array([1.0,1.0,1.0,-0.2,0.0,1.0])
bounds = [(0,None),(0,None),(0,None),(-1,0),(-1,1),(0,2)]

sol2_old = scipy.optimize.minimize(nll1, initial, 
    bounds = bounds,
    args=(r_bin_centres, rho_r_borg_mean, rho_r_borg_std,
          profile_modified_hamaus))

sol2 = scipy.optimize.minimize(nll1, initial, 
    bounds = bounds,
    args=(rBinStackCentres, delta_borg, sigma_delta_borg,
          profile_modified_hamaus))

sol2_lcdm = scipy.optimize.minimize(nll1, initial, 
    bounds = bounds,
    args=(rBinStackCentres, delta_lcdm, sigma_delta_lcdm,
          profile_modified_hamaus))

#A,r0,c1,f1,B = sol2.x
alpha,beta,rs,delta_c,delta_large, rv = sol2.x




# Inference run:


# Tabulated inverse:
ntab = 100
svals = np.linspace(0,3,ntab)
rperp_vals = np.linspace(0,3,ntab)
ntab_f = 100
f_vals = np.linspace(0,1,ntab_f)
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



# Run the inference:
#profile_param_ranges = [[-np.inf,np.inf],[0,2],[-np.inf,0],[0,np.inf],[-1,1]]
profile_param_ranges = [[0,np.inf],[0,np.inf],[0,np.inf],[-1,0],[-1,1],[0,2]]
om_ranges = [[0.1,0.5]]
eps_ranges = [[0.9,1.1]]
f_ranges = [[0,1]]
z = 0.0225
Om_fid = 0.3111
eps_initial_guess = np.array([1.0,f_lcdm(z,Om_fid)])
theta_initial_guess = np.array([0.3,f_lcdm(z,0.3)])

initial_guess_eps = np.hstack([eps_initial_guess,sol2.x])
initial_guess_theta = np.hstack([theta_initial_guess,sol2.x])

theta_ranges=om_ranges + f_ranges + profile_param_ranges
theta_ranges_epsilon = eps_ranges + f_ranges + profile_param_ranges

args = (data_field[data_filter],scoords[data_filter,:],
        cholesky_cov[data_filter,:][:,data_filter],z,Delta_func,
        delta_func,rho_real)
kwargs = {'cholesky':True,'tabulate_inverse':True,
          'sample_epsilon':True,'theta_ranges':theta_ranges_epsilon,
          'singular':False,'Umap':Umap,'good_eig':good_eig,'F_inv':F_inv}

# Wrapper allowing us to pass arbitrary arguments that won't be sampled over:
def posterior_wrapper(theta,additional_args,*args,**kwargs):
    theta_comb = np.hstack([theta,additional_args])
    return log_probability_aptest(theta_comb,*args,**kwargs)

nll = lambda theta: -log_likelihood_aptest(theta,*args,**kwargs)
mle_estimate = scipy.optimize.minimize(nll,initial_guess_eps,bounds=theta_ranges_epsilon)


plt.clf()
fig, ax = plt.subplots(figsize=(textwidth,0.45*textwidth))
plt.errorbar(rBinStackCentres,delta_borg,yerr=sigma_delta_borg,linestyle='-',
             color=seabornColormap[1],label="BORG profile")
plt.plot(rBinStackCentres,
         profile_modified_hamaus(rBinStackCentres,*mle_estimate.x[2:]),
         label="MLE Profile (joint inference)")
plt.plot(rBinStackCentres,
         profile_modified_hamaus(rBinStackCentres,*sol2.x),
         label="MLE Profile (Separate inference)")

plt.xlabel('$r/r_{\\mathrm{eff}}$')
plt.ylabel('$\\rho(r)$')
plt.xlim([0,10])
plt.ylim([0,1.1])
#plt.yscale('log')
#plt.xscale('log')
plt.legend(frameon=False,loc="lower right")
plt.tight_layout()
plt.savefig(figuresFolder + "profile_fit_test3.pdf")
plt.show()







import emcee

tau, sampler = run_inference(data_field,theta_ranges_epsilon,mle_estimate.x,
                             data_folder + "inference_weighted.h5",
                             log_probability_aptest,*args,redo_chain=True,
                             backup_start=True,nwalkers=64,sample="all",
                             n_mcmc=10000,disp=1e-1,
                             max_n=1000000,z=0.0225,parallel=False,Om_fid=0.3111,
                             batch_size=100,n_batches=10,data_filter=None,
                             autocorr_file = data_folder + "autocorr.npy",**kwargs)

sampler = emcee.backends.HDFBackend(data_folder + "inference_weighted.h5")
tau = sampler.get_autocorr_time(tol=0)
chain = samples.get_chain()
all_samples = chain.reshape(chain.shape[0]*chain.shape[1],chain.shape[2])
flat_samples = sampler.get_chain(discard = 300,thin=50,flat=True)

redo_chain = False
continue_chain = True
backup_start = True
import emcee
import h5py
nwalkers = 64
ndims = 2
n_mcmc = 10000
disp = 1e-4
max_n = 1000000


batch_size = 100

#data_filter = np.where(np.sqrt(np.sum(scoords**2,1)) < 1.5)[0]

data_filter = np.where((1.0/np.sqrt(np.diag(reg_norm_cov)) > 5) & \
         (np.sqrt(np.sum(scoords**2,1)) < 1.5) )[0]


# Filter the MCMC samples to account for correlation:
tau = sampler.get_autocorr_time(tol=0)
tau_max = np.max(tau)
flat_samples = sampler.get_chain(discard=int(3*tau_max), 
                                 thin=int(tau_max/2), flat=True)

flat_mean = np.mean(flat_samples,0)
flat_range = np.percentile(flat_samples,[16,84],axis=0)


import corner

plt.clf()
#fig = corner.corner(flat_samples, labels=["$\\Omega_{m}$","$f$","A"])
#fig = corner.corner(flat_samples, labels=["$\\epsilon$","$f$","$A$","$r_0$",
#    "$c_1$","$f_1$","$B$"])
#param_ranges = [[0.9,1.1],[0,1.0]]
param_ranges = None
fig = corner.corner(flat_samples,labels=["$\\epsilon$","$f$"],
                    range=param_ranges)
fig.suptitle("$\\Lambda$-CDM Inference from Void Catalogue")
plt.savefig(figuresFolder + "corner_plot_cosmo_inference_cosmo_only.pdf")
plt.show()


# Tests of the inverse function:
#X = scoords[:,0].reshape((20,20))
#Y = scoords[:,1].reshape((20,20))
X = np.hstack([np.array([x for k in range(0,100)]) 
    for x in np.linspace(0,2,100)])
Y = np.hstack([np.linspace(0,2,100) for k in range(0,100)])

inv_field = F_inv(X,Y,
                  0.25*np.ones(len(X))).reshape((100,100))


s_par_range = np.linspace(0,2,40)
s_perp_range = np.linspace(0,2,40)
f_range = np.linspace(0,1,40)
X, Y, Z = [x.flatten() for x in 
    np.meshgrid(s_par_range,s_perp_range,f_range)]

inv_field_3d = F_inv(X,Y,Z).reshape((40,40,40))

plt.clf()
im = plt.imshow(inv_field,cmap='Blues',vmin=0,vmax=2,extent=(0,2,0,2))
plt.xlabel('$s_{\\perp}/R_{\\mathrm{eff}}$')
plt.ylabel('$s_{\\parallel}/R_{\\mathrm{eff}}$')
plt.colorbar()
plt.savefig(figuresFolder + "inverse_field.pdf")
plt.show()

plt.clf()
fig, ax = plt.subplots(1,3)
slices = [inv_field_3d[:,10,20],inv_field_3d[27,:,13],inv_field_3d[20,39,:]]
ranges = [s_par_range,s_perp_range,f_range]
xnames = ['$s_{\\parallel}/R_{\\mathrm{eff}}$',
          '$s_{\\perp}/R_{\\mathrm{eff}}$','$f$']
data_points = []
for axi, y, xvals, name in zip(ax, slices,ranges,xnames):
    axi.plot(xvals,y)
    axi.set_xlabel('$s_{\\parallel}/R_{\\mathrm{eff}}$')
    axi.set_ylabel('$F^{-1}$')

plt.tight_layout()
plt.savefig(figuresFolder + "inverse_field_slice.pdf")
plt.show()


# Optimal A:
args2 = (data_field,scoords,np.identity(1600),z,Delta_func,delta_func,rho_real)
kwargs2={'Om_fid':Om_fid,'cholesky':True,'tabulate_inverse':True,
    'data_filter':data_filter,'sample_epsilon':False,'normalised':True}
sol_A = scipy.optimize.minimize_scalar(
    lambda A: -log_likelihood_aptest(np.array([Om_fid,f_lcdm(z,Om_fid),A]),
    *args2,**kwargs2))
A_opt = sol_A.x

# Fix the amplitude, and plot the likelihood:
Om_fid = 0.3111
data_filter = np.where((1.0/np.sqrt(np.diag(reg_norm_cov)) > 5) & \
        (np.sqrt(np.sum(scoords**2,1)) < 1.5) )[0]
reg_cov_filtered = reg_cov[data_filter,:][:,data_filter]
cholesky_cov_filtered = scipy.linalg.cholesky(reg_cov_filtered,lower=True)

profile_param_ranges_hamaus = [(0,np.inf),(0,np.inf),(0,np.inf),(-1,0),
                               (-1,1),(0,2)]

theta_ranges_hamaus = eps_ranges + f_ranges + profile_param_ranges_hamaus

args=(data_field[data_filter],scoords[data_filter,:],
                  cholesky_cov[data_filter,:][:,data_filter],z,Delta_func,
                  delta_func,rho_real)
kwargs={'Om_fid':Om_fid,'cholesky':True,'tabulate_inverse':True,
        'sample_epsilon':True,'theta_ranges':theta_ranges_hamaus,
        'singular':False,'Umap':Umap,'good_eig':good_eig,
        'F_inv':F_inv}
#data_filter = np.where(np.sqrt(np.sum(scoords**2,1)) < 2.0)[0]
#kwargs={'Om_fid':0.3111,'data_filter':data_filter}
#Om_fid = 0.3111
#kwargs={'Om_fid':Om_fid,'cholesky':True,'tabulate_inverse':True,
#    'sample_epsilon':True}
#params = combine_parameters(np.mean(flat_samples,0),fixed)
params = np.mean(flat_samples[:,2:],0)
eps_initial_guess = np.hstack([np.array([1.0,f_lcdm(z,Om_fid)]),params])
nll = lambda *theta: -log_likelihood_aptest(np.hstack([*theta,params]),*args,
                                            **kwargs)
nlp = lambda *theta: -log_probability_aptest(np.hstack([*theta,params]),
                                                 *args,**kwargs)
#nll = lambda *theta: -log_likelihood_aptest(np.hstack([*theta,A_opt]),
#                                            *args,**kwargs)
soln = scipy.optimize.minimize(nll, np.array([1.0,f_lcdm(z,Om_fid)]),
    bounds = [(0.9,1.1),(0,1.0)])
solnp = scipy.optimize.minimize(nlp, np.array([1.0,f_lcdm(z,Om_fid)]),
    bounds = [(0.9,1.1),(0,1.0)])
    #bounds=[(0.9,1.1),(0,1.0),(None,None),(0,2),(None,0),(0,None),(-1,1)])
#soln = scipy.optimize.minimize(nll, theta_initial_guess,
#    bounds=[(0.1,0.5),(0,1.0),(None,None)])
theta_initial_guess2 = np.array([Om_fid,f_lcdm(z,Om_fid)])
#soln = scipy.optimize.minimize(nll, theta_initial_guess2,
#    bounds=[(0.1,0.5),(0,1.0)])
#soln = scipy.optimize.minimize(nll, eps_initial_guess,
#    bounds=[(0.9,1.1),(0,1.0)])


Om_range = np.linspace(0.1,0.5,41)
f_range = np.linspace(0,1,41)
f_range_centres = plot.binCentres(f_range)
Om_range_centres = plot.binCentres(Om_range)
eps_range = np.linspace(0.9,1.1,41)
eps_centres = plot.binCentres(eps_range)

log_like_ap = np.zeros((40,40))
#A,r0,c1,f1,B = sol2.x
alpha,beta,rs,delta_c,delta_large, rv = sol2.x
#params = combine_parameters(np.mean(flat_samples1,0),fixed)
#A = A_opt
for i in tools.progressbar(range(0,len(Om_range_centres))):
    for j in range(0,len(f_range_centres)):
        if kwargs['sample_epsilon']:
            theta = np.array([eps_range[i],f_range_centres[j],*params])
        else:
            theta = np.array([Om_range_centres[i],f_range_centres[j],*params])
        #log_like_ap[i,j] = log_likelihood_aptest(theta,*args,**kwargs)
        log_like_ap[i,j] = log_probability_aptest(theta,*args,**kwargs)


if kwargs['sample_epsilon']:
    plt.clf()
    im = plt.imshow(-log_like_ap.T,
               extent=(eps_range[0],eps_range[-1],f_range[0],f_range[-1]),
               norm=colors.LogNorm(vmin=1e9,vmax=1e10),cmap='Blues',
               aspect='auto',origin='lower')
    X, Y = np.meshgrid(eps_centres,f_range_centres)
    CS = plt.contour(X,Y,-log_like_ap.T,levels = 10)
    plt.clabel(CS, inline=True, fontsize=10)
    plt.axvline(1.0,linestyle=':',color='k',label='Fiducial $\\Lambda$CDM')
    plt.axhline(f_lcdm(z,Om_fid),linestyle=':',color='k')
    plt.scatter(*soln.x,marker='x',color='k',label='MLE')
    plt.xlabel('$\\epsilon$')
    plt.ylabel('$f$')
    #plt.colorbar(im,label='Negative Log Likelihood')
    plt.colorbar(im,label='Negative Log Posterior')
    plt.legend(frameon=False,loc="upper left")
    plt.savefig(figuresFolder + "likelihood_test_plot_eps.pdf")
    plt.show()
else:
    plt.clf()
    plt.imshow(-log_like_ap.T,
               extent=(Om_range[0],Om_range[-1],f_range[0],f_range[-1]),
               norm=colors.LogNorm(vmin=1e21,vmax=3e22),cmap='Blues',
               aspect='auto',origin='lower')
    plt.xlabel('$\Omega_m$')
    plt.ylabel('$f$')
    plt.colorbar(label='Negative Log Likelihood')
    plt.savefig(figuresFolder + "likelihood_test_plot.pdf")
    plt.show()

plt.clf()
plt.plot(f_range_centres,log_like_ap[20,:])
plt.xlabel('f')
plt.ylabel('Log Likelihood')
plt.title("Likelihood at $\\epsilon = " + ("%.2g" % eps_range[20]) + "$")
plt.savefig(figuresFolder + "nll_plot_f.pdf")
plt.show()


#likelihood at mean value:
log_like_mean = np.zeros(eps_range[0:-1].shape)
for i in tools.progressbar(range(0,len(Om_range_centres))):
    if kwargs['sample_epsilon']:
            theta = np.array([eps_range[i],flat_mean[1],A,r0,c1,f1,B])
    else:
        theta = np.array([Om_range_centres[i],flat_mean[1],
                          A,r0,c1,f1,B])
    log_like_mean[i] = log_likelihood_aptest(theta,*args,**kwargs)

plt.clf()
plt.plot(eps_range[0:-1],log_like_mean)
#plt.axvline(ap_parameter(z,0.0,Om_fid),linestyle=':',color='k',
#    label='$\\Omega_m=0$')
#plt.axvline(ap_parameter(z,1.0,Om_fid),linestyle='--',color='k',
#    label='$\\Omega_m=1.0$')
plt.axvspan(ap_parameter(z,0.0,Om_fid),ap_parameter(z,1.0,Om_fid),alpha=0.5,
            color='k',label='$0 < \\Omega_m(\\epsilon) < 1$',ec=None)
plt.xlabel('$\\epsilon$')
plt.ylabel('Log Likelihood')
plt.title("Likelihood at $f = " + ("%.3g" % flat_mean[1]) + "$")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig(figuresFolder + "nll_plot_eps.pdf")
plt.show()



# Range of A values:
Om = 0.3111
f = f_lcdm(z,Om)
Arange = np.logspace(-5,1,101)
log_like_ap = np.zeros(Arange.shape)
log_like_ap2 = np.zeros(Arange.shape)

for i in tools.progressbar(range(0,len(Arange))):
    theta = np.array([Om,f,Arange[i]])
    log_like_ap[i] = log_likelihood_aptest(theta,*args,**kwargs)
    #log_like_ap2[i] = log_likelihood_aptest(theta,*args2,**kwargs2)

plt.clf()
plt.loglog(Arange,-log_like_ap,label='With Covariance')
#plt.loglog(Arange,-log_like_ap2,label='Least Squares')
plt.xlabel('A')
plt.ylabel('Negative log likelihood')
plt.title("Likelihood with $\\Omega_m = 0.3111, f = f_{\\mathrm{lcdm}}$")
plt.legend(frameon=False)
plt.savefig(figuresFolder + "A_likelihood_plot.pdf")
plt.show()



Om = soln.x[0]
eps = soln.x[0]
f = soln.x[1]
A = soln.x[2]

f_fid = f_lcdm(0.0225,0.3111)

for i in range(0,len(bins_z_reff)-1):
    for j in range(0,len(bins_d_reff)-1):
        spar = bin_z_centres[i]
        sperp = bin_d_centres[j]
        profile_2d[i,j] = z_space_profile(spar,sperp,
                                          lambda r: rho_real(r,*sol2.x),
                                          z,Om,Delta_func,delta_func,
                                          epsilon=1.0,
                                          f=f_fid,apply_geometry=True)

plot_los_void_stack(\
        profile_2d,bin_d_centres,bin_z_centres,
        cmap='Blues',vmin=0,vmax=0.06,fontsize=10,
        xlabel = '$s_{\\perp}/R_{\\mathrm{eff}}$ (Perpendicular distance)',
        ylabel = '$s_{\\parallel}/R_{\\mathrm{eff}}$ (LOS distance)',
        density_unit='probability',
        savename=figuresFolder + "profile_2d_test.pdf",
        title=None,colorbar=True,shrink=0.9,
        colorbar_title="$\\rho(s_{\\parallel},s_{\\perp})$")

plot_los_void_stack(\
        field_borg,bin_d_centres,bin_z_centres,
        cmap='Blues',vmin=0,vmax=0.06,fontsize=10,
        xlabel = '$s_{\\perp}/R_{\\mathrm{eff}}$ (Perpendicular distance)',
        ylabel = '$s_{\\parallel}/R_{\\mathrm{eff}}$ (LOS distance)',
        density_unit='probability',
        savename=figuresFolder + "profile_2d_field_borg.pdf",
        title=None,colorbar=True,shrink=0.9,
        colorbar_title="$\\rho(s_{\\parallel},s_{\\perp})$")

plot_los_void_stack(\
        (profile_2d - field_borg)/field_borg,bin_d_centres,bin_z_centres,
        cmap='PuOr_r',
        vmin=0,vmax=200,fontsize=10,norm=colors.Normalize(vmin=-1,vmax=1),
        xlabel = '$s_{\\perp}/R_{\\mathrm{eff}}$ (Perpendicular distance)',
        ylabel = '$s_{\\parallel}/R_{\\mathrm{eff}}$ (LOS distance)',
        density_unit='probability',
        savename=figuresFolder + "profile_2d_test_diff.pdf",
        title=None,colorbar=True,shrink=0.9,
        colorbar_title="$\\Delta\\rho(s_{\\parallel},s_{\\perp})" + \
        "/\\tilde{\\rho}(s_{\\parallel},s_{\\perp})$")


plot_los_void_stack(\
        profile_2d - field_borg,bin_d_centres,bin_z_centres,
        cmap='PuOr_r',
        vmin=0,vmax=200,fontsize=10,norm=colors.Normalize(vmin=-3e-3,vmax=3e-3),
        xlabel = '$s_{\\perp}/R_{\\mathrm{eff}}$ (Perpendicular distance)',
        ylabel = '$s_{\\parallel}/R_{\\mathrm{eff}}$ (LOS distance)',
        density_unit='probability',
        savename=figuresFolder + "profile_2d_test_diff_abs.pdf",
        title=None,colorbar=True,shrink=0.9,
        colorbar_title="$\\rho(s_{\\parallel},s_{\\perp})" + \
        " - \\tilde{\\rho}(s_{\\parallel},s_{\\perp})$")



plot_los_void_stack(\
        field_lcdm,bin_d_centres,bin_z_centres,
        cmap='Blues',
        vmin=0,vmax=0.06,fontsize=10,
        xlabel = '$s_{\\perp}/R_{\\mathrm{eff}}$ (Perpendicular distance)',
        ylabel = '$s_{\\parallel}/R_{\\mathrm{eff}}$ (LOS distance)',
        density_unit='probability',
        savename=figuresFolder + "profile_2d_theory_borg.pdf",
        title=None,colorbar=True,shrink=0.9,
        colorbar_title="$\\rho(s_{\\parallel},s_{\\perp})$")




# Fit a multi-parameter model to the profile:



def profile_fit(r,rho0,p,C,rb):
    #return A*(1.0 + k*r + (k*r)**2/(2.0 - k*r0))*np.exp(-k*r)
    return np.exp(np.log(rho0) + (p*C/2)*rb*r - C*r**p)

def log_profile_fit(r,rho0,p,C,rb):
    return np.log(rho0) + (p*C/2)*rb*r - C*r**p

def combine_parameters(theta,fixed):
    free_params = np.isnan(fixed)
    fixed_params = np.logical_not(free_params)
    if np.sum(free_params) != len(theta):
        raise Exception("Parameter count not consistent with free parameters.")
    else:
        theta_new = np.zeros(len(fixed))
        theta_new[free_params] = theta
        theta_new[fixed_params] = fixed[fixed_params]
        return theta_new

# MLE of the profile fit:
def log_likelihood(theta, x, y, yerr,profile_model,fixed=None):
    #rho0,p,C,rb = theta
    #A,r0,c1,f1,B = theta
    #model = profile_broken_power_log(x, A,r0,c1,f1,B)
    if fixed is None:
        fixed = np.array([np.nan for x in theta])
    model = profile_model(x,*combine_parameters(theta,fixed))
    sigma2 = yerr**2
    return -0.5 * np.sum( (y - model)**2/sigma2 + np.log(sigma2) )

# Priors:
def log_prior(theta):
    #A,r0,c1,f1,B = theta
    #alpha,beta,rs,delta_c,delta_large, rv = theta
    beta,delta_c,delta_large, rv = theta
    if (0 <= beta < np.inf) \
        and (-1 < delta_c <= 0) and (-1 <= delta_large <= 1) and (0 <= rv <= 2):
        # Jeffries priors:
        return 0
    else:
        return -np.inf

def log_probability(theta,x,y,yerr,*args,**kwargs):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, yerr,*args,**kwargs)

nll1 = lambda *theta: -log_likelihood(*theta)
initial = np.array([0.1,0.9,-9.5,3.5,0.01])
sol1 = scipy.optimize.minimize(nll1, initial, 
    bounds = [(0,None),(0,2),(None,0),(0,None),(-1,1)],
    args=(r_bin_centres, np.log(rho_r_mean),
    0.5*np.log((rho_r_mean + rho_r_std)/(rho_r_mean - rho_r_std)),
    profile_broken_power_log) )

sol2 = scipy.optimize.minimize(nll1, initial, 
    bounds = [(0,None),(0,2),(None,0),(0,None),(-1,1)],
    args=(r_bin_centres, np.log(rho_r_borg_mean),
    0.5*np.log((rho_r_borg_mean + rho_r_borg_std)/\
    (rho_r_borg_mean - rho_r_borg_std)),
    profile_broken_power_log) )

#A1,r01,c11,f11,B1 = sol1.x
#A2,r02,c12,f12,B2 = sol2.x

#initial = np.array([1.0,1.0,1.0,-0.2,0.0,1.0])
#bounds = [(0,None),(0,None),(0,None),(-1,0),(-1,1),(0,2)]

initial_free = np.array([1.0,-0.85,0.0,1.0])
bounds_free =  [(0,None),(-1,0),(-1,1),(0,2)]
fixed = np.array([np.nan for x in range(0,6)])
fixed[0] = 1
fixed[2] = np.inf

sol1 = scipy.optimize.minimize(nll1, initial_free, 
    bounds = bounds_free,
    args=(rBinStackCentres, delta_lcdm, sigma_delta_lcdm,
    profile_modified_hamaus,fixed) )

sol2 = scipy.optimize.minimize(nll1, initial_free, 
    bounds = bounds_free,
    args=(rBinStackCentres, delta_borg, sigma_delta_borg,
          profile_modified_hamaus,fixed) )

beta,delta_c,delta_large,rv = sol1.x

alpha=1
rs = np.inf

import emcee
pos = sol1.x + 1e-4*np.random.randn(32,4)
nwalkers, ndim = pos.shape

sampler1 = emcee.EnsembleSampler(
    nwalkers, ndim, log_probability, args=(rBinStackCentres, delta_lcdm, 
    sigma_delta_lcdm, profile_modified_hamaus,fixed)
)
sampler1.run_mcmc(pos, 10000, progress=True)

tau1 = sampler1.get_autocorr_time(tol=0)

flat_samples1 = sampler1.get_chain(discard=int(3*np.max(tau1)), 
                                 thin=int(np.max(tau1)), flat=True)

#A1, r01, c11,f11,B1 = np.mean(flat_samples1,0)
params = combine_parameters(np.mean(flat_samples1,0),fixed)

sampler2 = emcee.EnsembleSampler(
    nwalkers, ndim, log_probability, args=(rBinStackCentres, delta_borg, 
    sigma_delta_borg, profile_modified_hamaus,fixed)
)
sampler2.run_mcmc(pos, 10000, progress=True)
tau2 = sampler2.get_autocorr_time(tol=0)

flat_samples2 = sampler2.get_chain(discard=int(3*np.max(tau2)), 
                                 thin=int(np.max(tau2)), flat=True)


# Test effect on the inference:

nll_cosmo1 = lambda *theta: -log_likelihood_aptest(
    np.hstack([*theta,np.mean(flat_samples1,0)]),*args,**kwargs)
nll_cosmo2 = lambda *theta: -log_likelihood_aptest(
    np.hstack([*theta,np.mean(flat_samples2,0)]),*args,**kwargs)
#nll = lambda *theta: -log_likelihood_aptest(np.hstack([*theta,A_opt]),
#                                            *args,**kwargs)
soln_cosmo1 = scipy.optimize.minimize(nll_cosmo1, 
    np.array([1.0,f_lcdm(z,Om_fid)]),bounds = [(0.9,1.1),(0,1.0)])
soln_cosmo2 = scipy.optimize.minimize(nll_cosmo2, 
    np.array([1.0,f_lcdm(z,Om_fid)]),bounds = [(0.9,1.1),(0,1.0)])
soln_mle2 = scipy.optimize.minimize(nll, 
    np.array([1.0,f_lcdm(z,Om_fid)]),bounds = [(0.9,1.1),(0,1.0)])


# Corner plot:
import corner

plt.clf()
#fig = corner.corner(flat_samples1,labels=["$A$","$r_0$","$c_1$","$f_1$","$B$"])
fig = corner.corner(flat_samples1, labels=["$\\alpha$","$\\beta$","$r_s$",
    "$\\delta_c$","$\\delta_{LS}$","$r_v$"])
fig.suptitle("$\\Lambda$-CDM Simulations Contour Ellipticity")
plt.savefig(figuresFolder + "corner_plot_profile_fit.pdf")
plt.show()



#A = 0.1
#c1 = -9.5
#f1 = 3.5
#r0 = 0.9
#r1 = 1

#rho_r_fit_samples = np.vstack([
#    profile_broken_power(r_bin_centres,A,r0,c1,f1,B) 
#    for A,r0,c1,f1,B in flat_samples1])

#r_values = r_bin_centres
r_values = rBinStackCentres

rho_r_fit_samples = np.vstack([
    profile_modified_hamaus(r_values,*combine_parameters(theta,fixed)) 
    for theta in flat_samples1])

rho_r_fit_samples_mean = np.mean(rho_r_fit_samples,0)
rho_r_fit_samples_std = np.std(rho_r_fit_samples,0)


rho_r_borg_fit_samples = np.vstack([
    profile_modified_hamaus(r_values,*combine_parameters(theta,fixed))
    for theta in flat_samples2])

rho_r_borg_fit_samples_mean = np.mean(rho_r_borg_fit_samples,0)
rho_r_borg_fit_samples_std = np.std(rho_r_borg_fit_samples,0)

#field_borg_fit_samples = rho_r_borg_fit_samples = np.vstack([
#    profile_broken_power(r_values,A,r0,c1,f1,B) 
#    for A,r0,c1,f1,B in flat_samples[:,2:]])

field_borg_fit_samples = rho_r_borg_fit_samples = np.vstack([
    profile_modified_hamaus(r_values,combine_parameters(theta,fixed)) 
    for theta in flat_samples[:,2:]])

field_borg_fit_mean = np.mean(field_borg_fit_samples,0)
field_borg_fit_std = np.std(field_borg_fit_samples,0)

#rho_r_fit_vals = profile_broken_power(r_values,A1,r01,c11,f11,B1)
rho_r_fit_vals = profile_broken_power(r_values,A1,r01,c11,f11,B1)
rho_r_fit_vals_borg = profile_broken_power(r_values,A2,r02,c12,f12,B2)


plt.clf()
fig, ax = plt.subplots(figsize=(textwidth,0.45*textwidth))
plt.errorbar(rBinStackCentres,delta_lcdm,yerr=sigma_delta_lcdm,linestyle='-',
             color=seabornColormap[0],label="$\\Lambda$-CDM profile")
plt.fill_between(r_values,rho_r_fit_samples_mean-rho_r_fit_samples_std,
                 rho_r_fit_samples_mean + rho_r_fit_samples_std,alpha=0.5,
                 color=seabornColormap[0],
                 label="Fitting-formula, $\\Lambda$CDM-profile fit")
plt.errorbar(rBinStackCentres,delta_borg,yerr=sigma_delta_borg,linestyle='-',
             color=seabornColormap[1],label="BORG profile")
plt.fill_between(r_values,
                 rho_r_borg_fit_samples_mean-rho_r_borg_fit_samples_std,
                 rho_r_borg_fit_samples_mean + rho_r_borg_fit_samples_std,
                 alpha=0.5,color=seabornColormap[1],
                 label="Fitting-formula, BORG-profile fit")
#plt.plot(r_bin_centres,rho_r_fit_vals,
#         linestyle=':',color=seabornColormap[0])
#plt.plot(r_bin_centres,rho_r_fit_vals_borg,
#         linestyle=':',color=seabornColormap[1])
#plt.plot(bin_d_centres,field_borg[0],linestyle='-',color='k',
#    label="$\\rho_{\\mathrm{BORG}}(s_{\\perp},0)$")
#plt.fill_between(r_bin_centres,
#                 field_borg_fit_mean - field_borg_fit_std,
#                 field_borg_fit_mean + field_borg_fit_std,
#                 alpha=0.5,color='k',
#                 label="Fitting-formula, $\\rho(s_{\\parallel},s_{\\perp})$ " + 
#                 " fit")
#plt.plot(bin_d_centres,field_lcdm[0],linestyle='--',color='k',
#    label="$\\rho_{\\Lambda\\mathrm{CDM}}(s_{\\perp},0)$")

plt.xlabel('$r/r_{\\mathrm{eff}}$')
plt.ylabel('$\\rho(r)$')
plt.xlim([0,10])
plt.ylim([0,1.1])
#plt.yscale('log')
#plt.xscale('log')
plt.legend(frameon=False,loc="lower right")
plt.tight_layout()
plt.savefig(figuresFolder + "profile_fit_test2.pdf")
plt.show()


plt.clf()
fig, ax = plt.subplots(figsize=(textwidth,0.45*textwidth))
plt.errorbar(r_values,(delta_lcdm - rho_r_fit_samples_mean)/\
             rho_r_fit_samples_mean,
             yerr=sigma_delta_lcdm/rho_r_fit_samples_mean,linestyle='-',
             color=seabornColormap[0],
             label="$\\Lambda$-CDM profile")
plt.errorbar(r_values,
             (delta_borg - rho_r_borg_fit_samples_mean)/\
             rho_r_borg_fit_samples_mean,
             yerr=sigma_delta_borg/rho_r_borg_fit_samples_mean,linestyle='-',
             color=seabornColormap[1],label="BORG profile")
plt.xlabel('$r/r_{\\mathrm{eff}}$')
plt.ylabel('$\\Delta\\rho(r)/\\rho(r)$')
plt.axhline(0.0,linestyle=':',color='k')
plt.axhspan(-0.01,0.01,color='grey',alpha=0.5,label='1% error')
#plt.yscale('log')
#plt.xscale('log')
plt.ylim([-0.1,0.1])
plt.legend(frameon=False,loc="upper center")
plt.tight_layout()
plt.savefig(figuresFolder + "profile_fit_test_diff.pdf")
plt.show()








