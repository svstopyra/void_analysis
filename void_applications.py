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

def ap_parameter(z,Om,Om_fid,h=0.7,h_fid = 0.7):
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

# Transformation to real space, from redshift space, including geometric
# distortions from a wrong-cosmology:
def to_real_space(s_par,s_perp,z,Om,Om_fid=None,Delta=None,u_par=None,f=None,
                  N_max = 5,atol=1e-5,rtol=1e-5,epsilon=None,
                  **kwargs):
    # Perpendicular distance is easy:
    r_perp = s_perp
    # Get the AP parameter:
    if epsilon is None:
        if (Om_fid is not None):
            epsilon = ap_parameter(z,Om,Om_fid,**kwargs)
        else:
            epsilon = 1.0
    # Get the parallel distance:
    if u_par is None:
        # Assume linear relationship:
        if f is None:
            f = f_lcdm(z,Om,**kwargs)
        # Need to guess at r:
        r_par_guess = s_par
        for k in range(0,N_max):
            # Iteratively improve the guess:
            r = np.sqrt((r_par_guess)**2 + r_perp**2)
            r_par_new = s_par/(1.0  - (f/3.0)*Delta(r))
            if (np.abs(r_par_new - r_par_guess) < atol) or \
                (np.abs(r_par_new/r_par_guess - 1.0) < rtol):
                break
            r_par_guess = r_par_new
        r_par = r_par_guess/epsilon
    else:
        # Use supplied peculiar velocity:
        # Hubble rate:
        hz = Hz(z,Om,**kwargs)
        r_par = (s_par - (1.0 + z)*u_par/hz)
    return [r_par/epsilon,r_perp]

# Profile in redshift space:
def z_space_profile(s_par,s_perp,rho_real,z,Om,Delta,delta,**kwargs):
    # Get real-space co-ordinates:
    [r_par,r_perp] = to_real_space(s_par,s_perp,z,Om,Delta=Delta,**kwargs)
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
                          z,Delta,delta,rho_real,**kwargs):
    s_par = scoords[:,0]
    s_perp = scoords[:,1]
    Om, f , A = theta
    M = len(s_par)
    delta_rho = np.zeros(s_par.shape)
    # Evaluate the profile for the supplied value of the parameters:
    for k in range(0,M):
        delta_rho[k] = data_field[k] - \
            z_space_profile(s_par[k],s_perp[k],lambda r: rho_real(r,A),
                            z,Om,Delta,delta,f=f,**kwargs)
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


los_lcdm = los_list_void_only_lcdm
los_borg = los_list_void_only_borg

# Bins:
upper_dist_reff = 2
bins_z_reff = np.linspace(0,upper_dist_reff,41)
bins_d_reff = np.linspace(0,upper_dist_reff,41)
bin_z_centres = plot.binCentres(bins_z_reff)
bin_d_centres = plot.binCentres(bins_d_reff)

# Stacked void particles in 2d (in redshift space):
stacked_particles_reff_lcdm_abs = get_2d_void_stack_from_los_pos(
    los_list_void_only_lcdm_zspace,bins_z_reff,bins_d_reff,antihaloRadiiUn)
void_radii_borg = cat300.getMeanProperty("radii",void_filter=True)[0]
stacked_particles_reff_borg_abs = get_2d_void_stack_from_los_pos(
    los_list_void_only_borg_zspace,bins_z_reff,bins_d_reff,
    [void_radii_borg for rad in antihaloRadii])

# Stacked void_particles in 1d:
# We can use the real space profile for this:
stacked_particles_reff_lcdm_real = get_2d_void_stack_from_los_pos(
    los_list_void_only_lcdm,bins_z_reff,bins_d_reff,antihaloRadiiUn)
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
        v_weight = [[rad**3*np.ones(len(los)) 
                    for los, rad in zip(all_los,all_radii)] 
                    for all_los, all_radii in zip(los_pos,void_radii)]
    else:
        if type(additional_weights) == list:
            v_weight = [[rad**3*np.ones(len(los))*weight 
                    for los, rad, weight in zip(all_los,all_radii,all_weights)] 
                    for all_los, all_radii, all_weights,
                    in zip(los_pos,void_radii,additional_weights)]
        else:
            v_weight = [[rad**3*np.ones(len(los))*weight 
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
    for los in los_list_void_only_lcdm_zspace]
voids_used_lcdm_ind = [np.where(x)[0] for x in voids_used_lcdm]
voids_used_borg = [np.array([len(x) for x in los]) > 0 
    for los in los_list_void_only_borg_zspace]
void_radii_lcdm = [rad[filt] 
    for rad, filt in zip(antihaloRadiiUn,voids_used_lcdm)]

los_pos_lcdm = [ [los[x] for x in np.where(ind)[0]] 
    for los, ind in zip(los_list_void_only_lcdm_zspace,voids_used_lcdm) ]
los_pos_borg = [ [los[x] for x in np.where(ind)[0]] 
    for los, ind in zip(los_list_void_only_borg_zspace,voids_used_borg) ]

rep_scores = void_cat_frac = cat300.property_with_filter(
    cat300.finalCatFrac,void_filter=True)

v_weight_borg = get_weights_for_stack(
    los_pos_borg,[void_radii_borg for rad in antihaloRadii],
    additional_weights = rep_scores/np.sum(rep_scores))
v_weight_lcdm = get_weights_for_stack(los_pos_lcdm,void_radii_lcdm)

def get_field_from_los_data(los_data,z_bins,d_bins,v_weight):
    cell_volumes_reff = np.outer(np.diff(z_bins),np.diff(d_bins))
    hist = np.histogramdd(los_data,bins=[z_bins,d_bins],density=False,
                          weights = 1.0/(2*np.pi*v_weight*los_data[:,1]))
    count = len(los_data)
    return hist[0]/(2*count*cell_volumes_reff)

# Fields:
cell_volumes_reff = np.outer(np.diff(bins_z_reff),np.diff(bins_d_reff))
field_lcdm = get_field_from_los_data(stacked_particles_reff_lcdm_abs,
                                     bins_z_reff,bins_d_reff,v_weight_lcdm)

num_voids_borg = np.sum([np.sum(x) for x in voids_used_borg]) # Not the actual number
    # but the effective number being stacked, so the number of voids multiplied
    # by the number of samples.
nmean = len(snapList[0])/(boxsize**3)

field_borg = get_field_from_los_data(stacked_particles_reff_borg_abs,
                                     bins_z_reff,bins_d_reff,v_weight_borg)

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
ax.plot(rvals,rho_func_lcdm_z0(rvals)/np.mean(rho_func_lcdm_z0(start_values)),
    label='$\\rho_{\\mathrm{2d}}(0,d)$')
ax.set_xlabel('r [$\\mathrm{Mpc}h^{-1}$]')
ax.set_ylabel('$\\rho(r)$')
ax.set_yscale('log')
plt.legend(frameon=False)
plt.savefig(figuresFolder + "rho_real_plot_void_only.pdf")
plt.show()

# 2D profile function test (zspace):
z = 0.0225
profile_2d = np.zeros((len(bins_z_reff)-1,len(bins_d_reff)-1))
Om = 0.3111
for i in range(0,len(bins_z_reff)-1):
    for j in range(0,len(bins_d_reff)-1):
        spar = bin_z_centres[i]
        sperp = bin_d_centres[j]
        profile_2d[i,j] = z_space_profile(spar,sperp,rho_func,z,Om,Delta_func,
                                          delta_func)

# Test plot:
plot_los_void_stack(\
        profile_2d,bin_d_centres,bin_z_centres,
        contour_list=[0.05,0.1],Rvals = [1,2],cmap='Blues',
        vmin=0,vmax=0.12,fontsize=10,
        xlabel = '$d/R_{\\mathrm{eff}}$ (Perpendicular distance)',
        ylabel = '$z/R_{\\mathrm{eff}}$ (LOS distance)',fontfamily='serif',
        density_unit='probability',
        savename=figuresFolder + "profile_2d_test.pdf",
        title=None,colorbar=True,shrink=0.9,
        colorbar_title="$\\rho(s_{\\parallel},s_{\\perp})$")

plot_los_void_stack(\
        field_lcdm,bin_d_centres,bin_z_centres,
        contour_list=[0.05,0.1],Rvals = [1,2],cmap='Blues',
        vmin=0,vmax=1e-4,fontsize=10,
        xlabel = '$d/R_{\\mathrm{eff}}$ (Perpendicular distance)',
        ylabel = '$z/R_{\\mathrm{eff}}$ (LOS distance)',fontfamily='serif',
        density_unit='probability',
        savename=figuresFolder + "profile_2d_test_data_lcdm.pdf",
        title=None,colorbar=True,shrink=0.9,
        colorbar_title="$\\rho(s_{\\parallel},s_{\\perp})$")

# Inference:

# Estimate BORG data covariance:

los_list_reff_borg = get_2d_void_stack_from_los_pos(
    los_list_void_only_borg_zspace,bins_z_reff,bins_d_reff,
    [void_radii_borg for rad in antihaloRadii],stacked=False)

v_weights_all_borg = get_weights_for_stack(
    los_pos_borg,[void_radii_borg for rad in antihaloRadii],stacked=False,
    additional_weights = rep_scores/np.sum(rep_scores))

all_fields_borg = [[
    get_field_from_los_data(los,bins_z_reff,bins_d_reff,v_weight) 
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
stacked_cov = np.cov(stacked_fields,aweights = stacked_weights)
symmetric_cov = (stacked_cov + stacked_cov.T)/2

lambda_reg = 1e-20
regularised_cov = symmetric_cov + lambda_reg*np.identity(symmetric_cov.shape[0])
#tools.minmax(np.real(np.linalg.eig(regularised_cov)[0]))
eigen = np.real(np.linalg.eig(regularised_cov)[0])

L = np.linalg.cholesky(regularised_cov)
P = np.linalg.inv(L)
inv_cov = np.matmul(P,P.T)

# Data to compare to:
data_field = field_borg.flatten()
spar = np.hstack([s*np.ones(field_borg.shape[1]) 
    for s in plot.binCentres(bins_z_reff)])
sperp = np.hstack([plot.binCentres(bins_d_reff) 
    for s in plot.binCentres(bins_z_reff)])
scoords = np.vstack([spar,sperp]).T
z = 0.0225

def rho_real(r,A):
    return A*rho_func(r)/rho_func(0)


# Covariances plot:
plt.clf()
plt.imshow(stacked_cov,vmin=-1e-9,vmax=1e-9,cmap='PuOr_r')
plt.savefig(figuresFolder + "covariance_plot.pdf")
plt.show()

# 

theta_initial_guess = np.array([0.3,f_lcdm(z,0.3),1e-5])
logp = log_probability_aptest(theta_initial_guess,data_field,scoords,inv_cov,
                          z,Delta_func,delta_func,rho_real,Om_fid = 0.3111)

# Run the inference:
import emcee
import h5py
nwalkers = 64
ndims = 3
n_mcmc = 5000
disp = 1e-4
initial = theta_initial_guess + disp*np.random.randn(nwalkers,ndims)
filename = data_folder + "inference_weighted.h5"
backend = emcee.backends.HDFBackend(filename)
backend.reset(nwalkers, ndims)
parallel = False

if parallel:
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndims, log_probability_aptest_parallel, 
            args=(z,),kwargs={'Om_fid':0.3111},backend=backend,pool=pool)
        sampler.run_mcmc(initial,n_mcmc , progress=True)
else:
    sampler = emcee.EnsembleSampler(
        nwalkers, ndims, log_probability_aptest, 
        args=(data_field,scoords,inv_cov,z,Delta_func,delta_func,rho_real),
        kwargs={'Om_fid':0.3111},backend=backend)
    sampler.run_mcmc(initial,n_mcmc , progress=True)

# Filter the MCMC samples to account for correlation:
tau = sampler.get_autocorr_time()
tau_max = np.max(tau)
flat_samples = sampler.get_chain(discard=int(3*tau_max), 
                                 thin=int(tau_max/2), flat=True)












