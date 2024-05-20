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
            r_par = F_inv(s_perp,s_par)
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

# Likelihood function:
def log_likelihood_aptest(theta,data_field,scoords,inv_cov,
                          z,Delta,delta,rho_real,data_filter=None,
                          cholesky=False,normalised=False,tabulate_inverse=True,
                          ntab = 10,sample_epsilon=False,Om_fid=None,**kwargs):
    s_par = scoords[:,0]
    s_perp = scoords[:,1]
    if sample_epsilon:
        epsilon, f, A = theta
        if Om_fid is not None:
            Om = Om_fid
        else:
            Om = 0.3
    else:
        Om, f , A = theta
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
        svals = np.linspace(np.min(s_par_new),np.max(s_par_new),ntab)
        rperp_vals = np.linspace(np.min(s_perp_new),np.max(s_perp_new),ntab)
        rvals = np.zeros((ntab,ntab))
        for i in range(0,ntab):
            for j in range(0,ntab):
                F = (lambda r: r - r*(f/3)*\
                    Delta(np.sqrt(r**2 + rperp_vals[i]**2)) \
                    - svals[j])
                rvals[i,j] = scipy.optimize.fsolve(F,svals[j])
        F_inv = lambda x, y: scipy.interpolate.interpn((rperp_vals,svals),rvals,
                                                       np.vstack((x,y)).T,
                                                       method='cubic')
        theory_val = z_space_profile(s_par_new,s_perp_new,
                                     lambda r: rho_real(r,A),z,Om,Delta,
                                     delta,f=f,F_inv=F_inv,
                                     **kwargs)
        if normalised:
            delta_rho = 1.0 - theory_val/data_val
        else:
            delta_rho = data_val - theory_val
    else:
        for k in range(0,M):
            data_val = data_field[k]
            theory_val = z_space_profile(s_par_new[k],s_perp_new[k],
                                         lambda r: rho_real(r,A),z,Om,Delta,
                                         delta,f=f,**kwargs)
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
    in_range = (theta_range[0] <= theta < theta_range[1])
    if in_range:
        return 0.0
    else:
        return -np.inf

def log_flat_prior(theta,theta_ranges):
    bool_array = np.aray([bounds[0] <= param < bounds[1] 
        for bounds, param in zip(theta_ranges,theta)],dtype=bool)
    if np.all(bool_array):
        return 0.0
    else:
        return -np.inf

# Prior (assuming flat prior for now):
def log_prior_aptest(theta,theta_ranges=[[0.1,0.5],[0,1.0],[-np.inf,np.inf]],
        **kwargs):
    log_prior_array = np.zeros(theta.shape)
    flat_priors = [0,1]
    theta_flat = [theta[k] for k in flat_priors]
    theta_ranges_flat = [theta_ranges[k] for k in flat_priors]
    for k in flat_priors:
        log_prior_array[k] = log_flat_prior_single(theta[k],theta_ranges[k])
    # Amplitude prior (Jeffries):
    log_prior_array[2] = -np.log(theta[2])
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

filter_list_borg = [halo_indices[ns] >= 0 for ns in range(0,len(snapList))]
los_list_void_only_borg_zspace = get_los_positions_for_all_catalogues(snapList,
    snapListRev,
    [cat300.getMeanCentres(void_filter=True) for ns in range(0,len(snapList))],
    [cat300.getMeanProperty("radii",void_filter=True)[0] 
    for ns in range(0,len(snapList))],all_particles=False,
    void_indices = halo_indices,filter_list=filter_list_borg,
    dist_max=60,rmin=10,rmax=20,recompute=False,
    zspace=True,recompute_zspace=False,suffix=".lospos_void_only_zspace.p")

# LCDM examples for comparison:
distances_from_centre_lcdm = [
    np.sqrt(np.sum(snapedit.unwrap(centres - np.array([boxsize/2]*3),
    boxsize)**2,1)) for centres in antihaloCentresUn]
filter_list_lcdm = [(dist < 135) & (radii > 10) & (radii <= 20) 
    for dist, radii in zip(distances_from_centre_lcdm,antihaloRadiiUn)]

los_list_void_only_lcdm_zspace = get_los_positions_for_all_catalogues(
    snapListUn,snapListRevUn,antihaloCentresUn,antihaloRadiiUn,
    all_particles=False,filter_list=filter_list_lcdm,dist_max=60,rmin=10,
    rmax=20,recompute=False,zspace=True,recompute_zspace=False,
    suffix=".lospos_void_only_zspace.p")

# Get LOS list for selection-effect-processed regions:
[randCentres,randOverDen] = tools.loadOrRecompute(data_folder2 + \
    "random_centres_and_densities.p")
deltaMCMCList = tools.loadPickle(data_folder2 + "delta_list.p")
from void_analysis.simulation_tools import get_map_from_sample
deltaMAPBootstrap = scipy.stats.bootstrap(
    (deltaMCMCList,),get_map_from_sample,confidence_level = 0.68,
    vectorized=False,random_state=1000)
deltaMAPInterval = deltaMAPBootstrap.confidence_interval
comparableDensityMAP = [(delta <= deltaMAPInterval[1]) & \
    (delta > deltaMAPInterval[0]) for delta in randOverDen]
centresToUse = [randCentres[comp] for comp in comparableDensityMAP]
deltaToUse = [randOverDen[ns][comp] \
    for ns, comp in zip(range(0,len(snapList)),comparableDensityMAP)]
rSep = 2*135
indicesUnderdenseNonOverlapping = simulation_tools.getNonOverlappingCentres(
    centresToUse,rSep,boxsize,returnIndices=True)
centresUnderdenseNonOverlapping = [centres[ind] \
    for centres,ind in zip(centresToUse,indicesUnderdenseNonOverlapping)]
densityListUnderdenseNonOverlapping = [density[ind] \
    for density, ind in zip(comparableDensityMAP,\
    indicesUnderdenseNonOverlapping)]
densityUnderdenseNonOverlapping = np.hstack(
    densityListUnderdenseNonOverlapping)

distances_from_centre_lcdm_selected = [[
    np.sqrt(np.sum(snapedit.unwrap(centres - sphere_centre,boxsize)**2,1))
    for sphere_centre in selected_regions]
    for centres, selected_regions in zip(antihaloCentresUn,
    centresUnderdenseNonOverlapping)]

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

los_list_void_only_selected_lcdm = get_los_positions_for_all_catalogues(
    snapListUn,snapListRevUn,antihaloCentresUn,antihaloRadiiUn,
    all_particles=False,filter_list=filter_list_lcdm_by_region,dist_max=60,
    rmin=10,rmax=20,recompute=False,suffix=".lospos_void_only_selected.p")

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

los_list_void_only_combined_lcdm = [combine_los_lists(los_list) 
    for los_list in los_list_void_only_selected_lcdm]

los_list_void_only_lcdm_zspace_selected = get_los_positions_for_all_catalogues(
    snapListUn,snapListRevUn,antihaloCentresUn,antihaloRadiiUn,
    all_particles=False,filter_list=filter_list_lcdm_by_region,dist_max=60,
    rmin=10,rmax=20,recompute=False,zspace=True,recompute_zspace=False,
    suffix=".lospos_void_only_zspace_selected.p")

los_list_void_only_combined_lcdm_zspace = [combine_los_lists(los_list) 
    for los_list in los_list_void_only_lcdm_zspace_selected]

# Real space positions:
los_list_void_only_borg = get_los_positions_for_all_catalogues(snapList,
    snapListRev,
    [cat300.getMeanCentres(void_filter=True) for ns in range(0,len(snapList))],
    [cat300.getMeanProperty("radii",void_filter=True)[0] 
    for ns in range(0,len(snapList))],all_particles=False,
    void_indices = halo_indices,filter_list=filter_list_borg,
    dist_max=60,rmin=10,rmax=20,recompute=False,suffix=".lospos_void_only.p")
los_list_void_only_lcdm = get_los_positions_for_all_catalogues(snapListUn,
    snapListRevUn,antihaloCentresUn,antihaloRadiiUn,all_particles=False,
    filter_list=filter_list_lcdm,dist_max=60,rmin=10,rmax=20,recompute=False,
    suffix=".lospos_void_only.p")

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


los_lcdm = los_list_void_only_combined_lcdm
los_borg = los_list_void_only_borg

# Bins:
upper_dist_reff = 2
bins_z_reff = np.linspace(0,upper_dist_reff,41)
bins_d_reff = np.linspace(0,upper_dist_reff,41)
bin_z_centres = plot.binCentres(bins_z_reff)
bin_d_centres = plot.binCentres(bins_d_reff)

# Stacked void particles in 2d (in redshift space):
stacked_particles_reff_lcdm_abs = get_2d_void_stack_from_los_pos(
    los_list_void_only_lcdm_zspace_selected,bins_z_reff,bins_d_reff,
    antihaloRadiiUn)
void_radii_borg = cat300.getMeanProperty("radii",void_filter=True)[0]
stacked_particles_reff_borg_abs = get_2d_void_stack_from_los_pos(
    los_list_void_only_borg_zspace,bins_z_reff,bins_d_reff,
    [void_radii_borg for rad in antihaloRadii])

# Stacked void_particles in 1d:
# We can use the real space profile for this:
stacked_particles_reff_lcdm_real = get_2d_void_stack_from_los_pos(
    los_lcdm,bins_z_reff,bins_d_reff,antihaloRadiiUn)
stacked_particles_reff_borg_real = get_2d_void_stack_from_los_pos(
    los_list_void_only_borg,bins_z_reff,bins_d_reff,
    [void_radii_borg for rad in antihaloRadii])
stacked_1d_real_lcdm = np.sqrt(np.sum(stacked_particles_reff_lcdm_real**2,1))
stacked_1d_real_borg = np.sqrt(np.sum(stacked_particles_reff_borg_real**2,1))

[_,noInBins_lcdm] = plot_utilities.binValues(stacked_1d_real_lcdm,bins_d_reff)
[_,noInBins_borg] = plot_utilities.binValues(stacked_1d_real_borg,bins_d_reff)


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

# Weights for each void in the stack:
voids_used_lcdm = [np.array([len(x) for x in los]) > 0 
    for los in los_list_void_only_lcdm_zspace_selected]
voids_used_lcdm_ind = [np.where(x)[0] for x in voids_used_lcdm]
voids_used_borg = [np.array([len(x) for x in los]) > 0 
    for los in los_list_void_only_borg_zspace]
void_radii_lcdm = [rad[filt] 
    for rad, filt in zip(antihaloRadiiUn,voids_used_lcdm)]

los_pos_lcdm = [ [los[x] for x in np.where(ind)[0]] 
    for los, ind in zip(los_list_void_only_lcdm_zspace_selected,voids_used_lcdm) ]
los_pos_borg = [ [los[x] for x in np.where(ind)[0]] 
    for los, ind in zip(los_list_void_only_borg_zspace,voids_used_borg) ]

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

def get_field_from_los_data(los_data,z_bins,d_bins,v_weight,void_count):
    cell_volumes_reff = np.outer(np.diff(z_bins),np.diff(d_bins))
    hist = np.histogramdd(los_data,bins=[z_bins,d_bins],density=False,
                          weights = v_weight/(2*np.pi*los_data[:,1]))
    return hist[0]/(2*void_count*cell_volumes_reff)

# Fields:
num_voids_lcdm = np.sum([np.sum(x) for x in voids_used_lcdm]) 
cell_volumes_reff = np.outer(np.diff(bins_z_reff),np.diff(bins_d_reff))
field_lcdm = get_field_from_los_data(stacked_particles_reff_lcdm_abs,
                                     bins_z_reff,bins_d_reff,v_weight_lcdm,
                                     num_voids_lcdm)

num_voids_sample_borg = np.array([np.sum(x) for x in voids_used_borg])
num_voids_borg = np.sum(num_voids_sample_borg) # Not the actual number
    # but the effective number being stacked, so the number of voids multiplied
    # by the number of samples.
nmean = len(snapList[0])/(boxsize**3)

field_borg = get_field_from_los_data(stacked_particles_reff_borg_abs,
                                     bins_z_reff,bins_d_reff,v_weight_borg,1)


field_borg_unweighted = get_field_from_los_data(
    stacked_particles_reff_borg_abs,bins_z_reff,bins_d_reff,
    v_weight_borg_unweighted,num_voids_borg)



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


# Profile function (should we use the inferred profile, or the 
# lcdm mock profiles?):
r_bin_centres = plot_utilities.binCentres(bins_d_reff)
#rho_r = noInBins_borg/np.sum(noInBins_borg)
rho_r = noInBins_lcdm/(np.sum(noInBins_lcdm)*\
    4*np.pi*(bins_d_reff[1:]**3 - bins_d_reff[0:-1]**3)/3)
rho_r_error = np.sqrt(noInBins_lcdm)/(np.sum(noInBins_lcdm)*\
    4*np.pi*(bins_d_reff[1:]**3 - bins_d_reff[0:-1]**3)/3)
rho_borg_r = noInBins_borg/(np.sum(noInBins_borg)*\
    4*np.pi*(bins_d_reff[1:]**3 - bins_d_reff[0:-1]**3)/3)
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

fig, ax = plt.subplots()
ax.plot(rvals,rho_func(rvals)/np.mean(rho_func(start_values)),
    label='$\\rho(r)$')
ax.plot(rvals,rho_func_borg(rvals)/np.mean(rho_func_borg(start_values)),
    label='$\\rho_{\\mathrm{borg}}(r)$')
ax.plot(rvals,rho_func_borg_z0(rvals)/np.mean(rho_func_borg_z0(start_values)),
    label='$\\rho_{\\mathrm{2d}}(0,d)$')
ax.set_xlabel('$r/r_{\\mathrm{eff}}$')
ax.set_ylabel('$\\rho(r)$')
ax.set_yscale('log')
plt.legend(frameon=False)
plt.savefig(figuresFolder + "rho_real_plot_void_only.pdf")
plt.show()

# 2D profile function test (zspace):
z = 0.0225
profile_2d = np.zeros((len(bins_z_reff)-1,len(bins_d_reff)-1))

Om = 0.3111
f = f_lcdm(z,Om)
A = 0.013


def rho_real(r,A):
    return A*rho_func(r)/rho_func(0)

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
        cmap='Blues',
        vmin=0,vmax=0.1,fontsize=10,
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
        cmap='Blues',ax= ax[0],
        vmin=0,vmax=0.1,fontsize=10,
        xlabel = '$s_{\\perp}/R_{\\mathrm{eff}}$',
        ylabel = '$s_{\\parallel}/R_{\\mathrm{eff}}$ (LOS distance)',
        fontfamily='serif',
        density_unit='probability',
        savename=None,
        title=None,colorbar=False,shrink=0.9,
        colorbar_title="$\\rho(s_{\\parallel},s_{\\perp})$")
im = plot_los_void_stack(\
        field_borg,bin_d_centres,bin_z_centres,
        cmap='Blues',ax= ax[1],
        vmin=0,vmax=0.1,fontsize=10,
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

# Inference:

# Estimate BORG data covariance:

los_list_reff_borg = get_2d_void_stack_from_los_pos(
    los_list_void_only_borg_zspace,bins_z_reff,bins_d_reff,
    [void_radii_borg for rad in antihaloRadii],stacked=False)



v_weights_all_borg = get_weights_for_stack(
    los_pos_borg,[void_radii_borg[used] for used in voids_used_borg],
    additional_weights = [rep_scores[used]/np.sum(all_rep_scores) 
    for used in voids_used_borg],stacked=False)



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

stacked_weights = np.hstack(f_lengths)/np.sum(f_totals)
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
eigen = np.real(np.linalg.eig(norm_cov)[0])

# Jackknife over all voids:

def range_excluding(kmin,kmax,exclude):
    return np.setdiff1d(range(kmin,kmax),exclude)

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
reg_cov = regularise_covariance(jackknife_cov,lambda_reg= 1e-27)
cholesky_cov = scipy.linalg.cholesky(reg_cov,lower=True)

inv_cov = get_inverse_covariance(norm_cov,lambda_reg = 1e-10)

# Eigenalue distribution:
eig, U = scipy.linalg.eigh(reg_cov)
eigen = np.real(np.linalg.eig(jackknife_cov)[0])

plt.clf()
bins = np.logspace(-30,-9,21)
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

def compute_chi2_ij(samples,L,xbar):
    n = samples.shape[1]
    k = samples.shape[0]
    residual = samples - xbar[:,None]
    solved_residuals = np.array(
        [scipy.linalg.solve_triangular(L,residual[:,i],lower=True) 
        for i in tools.progressbar(range(0,n))]).T
    Ai = np.array(
        [np.sum(np.sum((solved_residuals[:,i][:,None]*solved_residuals),0)**3)
        for i in tools.progressbar(range(0,n))])
    A = np.sum(Ai)/(6*n)
    dof = k*(k+1)*(k+2)/6
    pvalue = scipy.stats.chi2.sf(A,dof)
    B = np.sqrt(n/(8*k*(k+2)))*(np.sum(np.sum(solved_residuals**2,0)**2)/n \
                                - k*(k+2))
    return [A,B]


# Singular Gaussian likelihood:

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


Umap, good_eig = get_nonsingular_subspace(jackknife_cov,1e-27,lambda_cut=1e-23,
                                          normalised_cov = False,
                                          mu=jackknife_mean)

def compute_singular_log_likelihood(x,Umap,good_eig):
    u = np.matmul(Umap,x)
    Du = u/good_eig
    uDu = np.sum(u*Du)
    N = len(good_eig)
    return -0.5*uDu - (N/2)*np.log(2*np.pi) - 0.5*np.sum(np.log(good_eig))

singular = True
normalised_cov = False
n = n_boot
samples = bootstrap_stacks
xbar = jackknife_mean
if not singular:
    if normalised_cov:
        residual = samples/xbar[:,None] - 1.0
    else:
        residual = samples - xbar[:,None]
    L = cholesky_cov
    solved_residuals = np.array(
        [scipy.linalg.solve_triangular(L,residual[:,i],lower=True) 
        for i in tools.progressbar(range(0,n))]).T
else:
    if normalised_cov:
        residual = np.matmul(Umap,samples/xbar[:,None] - 1.0)
    else:
        residual = np.matmul(Umap,samples - xbar[:,None])
    solved_residuals = residual/np.sqrt(good_eig[:,None])

k = solved_residuals.shape[0]
low_memory_sum = False

if low_memory_sum:
    Ai = np.array(
        [np.sum(np.sum((solved_residuals[:,i][:,None]*solved_residuals),0)**3)
        for i in tools.progressbar(range(0,n))])
    A = np.sum(Ai)/(6*n)
else:
    product = np.matmul(solved_residuals.T,solved_residuals)
    A = np.sum(product**3)/(6*n)


dof = k*(k+1)*(k+2)/6
pvalue = scipy.stats.chi2.sf(A,dof)
B = np.sqrt(n/(8*k*(k+2)))*(np.sum(np.sum(solved_residuals**2,0)**2)/n \
                            - k*(k+2))

# Chi2 distribution:
chi2 = np.sum(solved_residuals**2,0)
estimated_chi2_mean = int(np.round(np.mean(chi2)))
xvals = np.logspace(-4,4,10000)
yvals_1600 = scipy.stats.chi2.pdf(xvals,1600)
yvals_mean = scipy.stats.chi2.pdf(xvals,estimated_chi2_mean)

plt.clf()
seaborn.kdeplot(chi2,color=seabornColormap[0],alpha=0.5,
                label = "Chi^2 values")
plt.plot(xvals,yvals_1600,linestyle=':',color='b',label='k=1600')
plt.plot(xvals,yvals_mean,linestyle='--',color='r',
         label='k=' + ("%.4g" % estimated_chi2_mean))
plt.xlabel('$\\chi^2$')
plt.ylabel('Probability Density')
plt.xscale('log')
plt.yscale('log')
plt.ylim([1e-8,1])
plt.xlim([1000,2000])
plt.legend()
plt.savefig(figuresFolder + "chi_squared_plot.pdf")
plt.show()

# chi2 for each variable:
chi2_dist = np.mean(solved_residuals**2,1)
plt.clf()
seaborn.kdeplot(chi2_dist,color=seabornColormap[0],alpha=0.5,
                label = "Chi^2 values",log_scale=True)
plt.savefig(figuresFolder + "chi2_each_dof.pdf")
plt.show()

plt.clf()
C_diag = np.diag(reg_norm_cov).reshape((40,40))
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

# Run the inference:
theta_ranges=[[0.1,0.5],[0,1.0],[-np.inf,np.inf]]
theta_ranges_epsilon=[[0.9,1.10],[0,1.0],[-np.inf,np.inf]]
redo_chain = False
continue_chain = True
backup_start = True
import emcee
import h5py
nwalkers = 64
ndims = 3
n_mcmc = 10000
disp = 1e-4
Om_fid = 0.3111
max_n = 1000000
eps_initial_guess = [1.0,f_lcdm(z,Om_fid),0.01]
initial = eps_initial_guess + disp*np.random.randn(nwalkers,ndims)
filename = data_folder + "inference_weighted.h5"
filename_initial = data_folder + "inference_old_state.h5"
if backup_start:
    os.system("cp " + filename + " " + filename_initial)

backend = emcee.backends.HDFBackend(filename)
if redo_chain:
    backend.reset(nwalkers, ndims)

parallel = False

batch_size = 100
n_batches = 100
autocorr = np.zeros((3,n_batches*batch_size))
old_tau = np.inf
#data_filter = np.where(np.sqrt(np.sum(scoords**2,1)) < 1.5)[0]

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
    data_filter = np.where((1.0/np.sqrt(np.diag(reg_norm_cov)) > 5) & \
        (np.sqrt(np.sum(scoords**2,1)) < 1.5) )[0]
    reg_cov_filtered = reg_cov[data_filter,:][:,data_filter]
    cholesky_cov_filtered = scipy.linalg.cholesky(reg_cov_filtered,lower=True)
    sampler = emcee.EnsembleSampler(
            nwalkers, ndims, log_probability_aptest, 
            args=(data_field[data_filter],scoords[data_filter,:],
                  cholesky_cov_filtered,z,Delta_func,delta_func,rho_real),
            kwargs={'Om_fid':Om_fid,'cholesky':True,'tabulate_inverse':True,
                    'sample_epsilon':True,'theta_ranges':theta_ranges_epsilon},
                    backend=backend)
    if redo_chain:
        sampler.run_mcmc(initial,n_mcmc , progress=True)
    else:
        for k in range(0,n_batches):
            sampler.run_mcmc(None,batch_size,progress=True)
            tau = sampler.get_autocorr_time(tol=0)
            autocorr[:,k] = tau
            # Check convergence
            converged = np.all(tau * 100 < sampler.iteration)
            converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
            if converged:
                break
            old_tau = tau

# Filter the MCMC samples to account for correlation:
tau = sampler.get_autocorr_time(tol=0)
tau_max = np.max(tau)
flat_samples = sampler.get_chain(discard=int(3*tau_max), 
                                 thin=int(tau_max/2), flat=True)

import corner

plt.clf()
fig = corner.corner(flat_samples, labels=["$\\Omega_{m}$","$f$","A"])
fig.suptitle("$\\Lambda$-CDM Inference from Void Catalogue")
plt.savefig(figuresFolder + "corner_plot_cosmo_inference.pdf")
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
data_filter = np.where((1.0/np.sqrt(np.diag(reg_norm_cov)) > 5) & \
        (np.sqrt(np.sum(scoords**2,1)) < 1.5) )[0]
reg_cov_filtered = reg_cov[data_filter,:][:,data_filter]
cholesky_cov_filtered = scipy.linalg.cholesky(reg_cov_filtered,lower=True)
args = (data_field[data_filter],scoords[data_filter,:],cholesky_cov_filtered,
        z,Delta_func,delta_func,rho_real)
#data_filter = np.where(np.sqrt(np.sum(scoords**2,1)) < 2.0)[0]
#kwargs={'Om_fid':0.3111,'data_filter':data_filter}
Om_fid = 0.3111
kwargs={'Om_fid':Om_fid,'cholesky':True,'tabulate_inverse':True,
    'sample_epsilon':True}
eps_initial_guess = [1.0,f_lcdm(z,Om_fid),0.01]
nll = lambda *theta: -log_likelihood_aptest(*theta,*args,**kwargs)
#nll = lambda *theta: -log_likelihood_aptest(np.hstack([*theta,A_opt]),
#                                            *args,**kwargs)
soln = scipy.optimize.minimize(nll, eps_initial_guess,
    bounds=[(0.98,1.02),(0,1.0),(None,None)])
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

log_like_ap = np.zeros((40,40))
A = soln.x[2]
#A = A_opt
for i in tools.progressbar(range(0,len(Om_range_centres))):
    for j in range(0,len(f_range_centres)):
        if kwargs['sample_epsilon']:
            theta = np.array([eps_range[i],f_range_centres[j],A])
        else:
            theta = np.array([Om_range_centres[i],f_range_centres[j],A])
        log_like_ap[i,j] = log_likelihood_aptest(theta,*args,**kwargs)


if kwargs['sample_epsilon']:
    plt.clf()
    plt.imshow(-log_like_ap.T,
               extent=(eps_range[0],eps_range[-1],f_range[0],f_range[-1]),
               norm=colors.LogNorm(vmin=1e20,vmax=5e20),cmap='Blues',
               aspect='auto',origin='lower')
    plt.xlabel('$\\epsilon$')
    plt.ylabel('$f$')
    plt.colorbar(label='Negative Log Likelihood')
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


plt.clf()
plt.plot(eps_range[0:-1],log_like_ap[:,20])
plt.axvline(ap_parameter(z,0.0,Om_fid),linestyle=':',color='k',
    label='$\\Omega_m=0$')
plt.axvline(ap_parameter(z,1.0,Om_fid),linestyle='--',color='k',
    label='$\\Omega_m=1.0$')
plt.xlabel('$\\epsilon$')
plt.ylabel('Log Likelihood')
plt.title("Likelihood at $f = " + ("%.2g" % f_range_centres[20]) + "$")
plt.legend(frameon=False)
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
f = soln.x[1]
A = soln.x[2]

for i in range(0,len(bins_z_reff)-1):
    for j in range(0,len(bins_d_reff)-1):
        spar = bin_z_centres[i]
        sperp = bin_d_centres[j]
        profile_2d[i,j] = z_space_profile(spar,sperp,lambda r: rho_real(r,A),
                                          z,Om,Delta_func,delta_func,
                                          Om_fid=0.3111,f=f)

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
        profile_2d/field_borg,bin_d_centres,bin_z_centres,
        cmap='PuOr_r',
        vmin=0,vmax=200,fontsize=10,norm=colors.LogNorm(vmin=1e-2,vmax=1e2),
        xlabel = '$s_{\\perp}/R_{\\mathrm{eff}}$ (Perpendicular distance)',
        ylabel = '$s_{\\parallel}/R_{\\mathrm{eff}}$ (LOS distance)',
        density_unit='probability',
        savename=figuresFolder + "profile_2d_test_diff.pdf",
        title=None,colorbar=True,shrink=0.9,
        colorbar_title="$\\rho(s_{\\parallel},s_{\\perp})" + \
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
        profile_2d,bin_d_centres,bin_z_centres,
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

# MLE of the profile fit:

def log_likelihood(theta, x, y, yerr):
    rho0,p,C,rb = theta
    model = log_profile_fit(x, rho0,p,C,rb)
    sigma2 = yerr**2
    return -0.5 * np.sum( (y - model)**2/sigma2 + np.log(sigma2) )

# Priors:
def log_prior(theta):
    rho0,p,C,rb = theta
    if (0 <= r0 < 2.0) or (p <= 2):
        # Jeffries priors:
        return -np.log(A) - np.log(k)
    else:
        return -np.inf

def log_probability(theta,x,y,yerr):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, yerr)

nll1 = lambda *theta: -log_likelihood(*theta)
initial = np.array([0.1,3,3,1])
sol1 = scipy.optimize.minimize(nll1, initial, 
    bounds = [(0,None),(0,None),(2,None),(0,2)],
    args=(r_bin_centres, np.log(rho_r),
    0.5*np.log((rho_r + rho_r_error)/(rho_r - rho_r_error))) )

rho0,p,C,rb = sol1.x


import emcee
pos = sol1.x + 1e-4*np.random.randn(32,3)
nwalkers, ndim = pos.shape

sampler = emcee.EnsembleSampler(
    nwalkers, ndim, log_probability, args=(r_bin_centres, np.log(rho_r),
    0.5*np.log((rho_r + rho_r_error)/(rho_r - rho_r_error)))
)
sampler.run_mcmc(pos, 10000, progress=True)

tau = sampler.get_autocorr_time()

flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)

A, k, r0 = np.mean(flat_samples,0)



A = 0.1
k = 3.5
r0 = 0.5

rho_r_fit_vals = profile_fit(r_bin_centres,rho0,p,C,rb)
plt.clf()
plt.errorbar(r_bin_centres,rho_r,yerr=rho_r_error,linestyle='-',color='k')
plt.plot(r_bin_centres,rho_r_fit_vals,
         linestyle=':',color='b')
plt.xlabel('$r/r_{\\mathrm{eff}}$')
plt.ylabel('$\\rho(r)$')
plt.yscale('log')
plt.xscale('log')
plt.savefig(figuresFolder + "profile_fit_test.pdf")
plt.show()








