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

figuresFolder = "borg-antihalos_paper_figures/all_samples/"
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

fontsize = 8
legendFontsize = 8

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
antihaloCentres = [tools.remapAntiHaloCentre(props[5],boxsize) \
            for props in ahProps]
antihaloMasses = [props[3] for props in ahProps]
antihaloRadii = [props[7] for props in ahProps]



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
[snrFilter,snrAllCatsList] = getSNRFilterFromChainFile(chainFile,snrThresh,\
    snapNameList,boxsize)

if recomputeCatalogues or (not os.path.isfile(data_folder + "cat300.p")):
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
    tools.savePickle(cat300,data_folder + "cat300.p")
else:
    cat300 = tools.loadPickle(data_folder + "cat300.p")

# Random catalogues:
snapNameListRand = [snap.filename for snap in snapListUn]
snapNameListRandRev = [snap.filename for snap in snapListRevUn]

if recomputeCatalogues or (not os.path.isfile(data_folder + "cat300Rand.p")):
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
    tools.savePickle(cat300Rand,data_folder + "cat300Rand.p")
else:
    cat300Rand = tools.loadPickle(data_folder + "cat300Rand.p")

#-------------------------------------------------------------------------------
# CATALOGUE DATA:

ns = 0
doSky=True
snapToShow = pynbody.load(samplesFolder + "sample" + str(snapNumList[ns]) + \
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
# VOID PROFILES PLOTS - DATA SETUP

# Parameters:
load_trees = False


radiiListShort = cat300.radiusListShort
massListShort = cat300.massListShort
finalCentres300List = cat300.getAllCentres()
meanCentre300 = cat300.getMeanCentres()
#distArr = np.arange(0,3,0.1)
nBinEdges = 8
rLower = 10
rUpper = 20
radBins = np.linspace(rLower,rUpper,nBinEdges)
# Filter:

# Compute a filter for the voids we actually want to present:
radiiList300 = cat300.getAllProperties('radii')
massList300 = cat300.getAllProperties('mass')
[radiiMean300, radiiSigma300]  = cat300.getMeanProperty('radii')
[massMean300, massSigma300]  = cat300.getMeanProperty('mass')
distances300 = np.sqrt(np.sum(meanCentre300**2,1))
percentilesCat300 = cat300Rand.getThresholdsInBins(radBins,99)
thresholds300 = cat300.getAllThresholds(percentilesCat300[0],radBins)
filter300 = (radiiMean300 > 10) & (radiiMean300 <= 25) & \
    (distances300 < 135) & (cat300.finalCatFrac > thresholds300)

# Central and average densities:
[deltaCentralMean,deltaCentralSigma] = cat300.getMeanProperty('deltaCentral')
[deltaAverageMean,deltaAverageSigma] = cat300.getMeanProperty('deltaAverage')


# Correct centres to be in equatorial co-ordinates:
meanCentres = meanCentre300[filter300]
# OLD position IN SIMULATION (not equatorial co-ordinates!),
# with swapXZ = True, reverse = False
#meanCentresGadgetCoordOld = snapedit.wrap(np.fliplr(cat300old.getMeanCentres())
#     + boxsize/2,boxsize)
# NEW position IN SIMULATION (not equatorial co-ordinates!), 
# with swapXZ = False, reverse = True
meanCentresGadgetCoord = snapedit.wrap(-meanCentres + boxsize/2,
                                       boxsize)
meanRadii = radiiMean300[filter300]
meanMasses = massMean300[filter300]
allCentres300Gadget = np.array([snapedit.wrap(\
    -centre + boxsize/2,boxsize) \
    for centre in finalCentres300List[:,filter300,:]])
allRadii300 = radiiList300[filter300,:]
isnan = np.where(allRadii300 < 0)
allRadii300[isnan] = np.nan

pairsList = [None for snap in snapList]
volumesList = [None for snap in snapList]
conditionList = [None for snap in snapList]
rEffMin = 0.0
rEffMax = 10.0
#rEffMax = 3.0
rSphere = 135
nRadiusBins = 101
#nRadiusBins = 31
nbar = (512/boxsize)**3
rBinStack = np.linspace(rEffMin,rEffMax,nRadiusBins)
rBinStackCentres = plot.binCentres(rBinStack)
centresList = [meanCentresGadgetCoord for snap in snapList]
radiiList = [meanRadii for snap in snapList]
massList = [meanMasses for snap in snapList]


# Load trees. Only needed if we want to recompute
# the pair counts:
if load_trees:
    treeList = [tools.loadOrRecompute(\
                    snap.filename +   ".tree",\
                    scipy.spatial.cKDTree,snap['pos'],boxsize=boxsize,\
                    _recomputeData=False) for snap in snapList]
    treeListUncon = [tools.loadOrRecompute(\
                        snap.filename +   ".tree",\
                        scipy.spatial.cKDTree,snap['pos'],boxsize=boxsize,\
                        _recomputeData=False) for snap in snapListUn]
else:
    treeList = [None for snap in snapList]
    treeListUncon = [None for snap in snapListUn]

# MCMC profiles, about a centre specific to each sample:
[allPairsSample,allVolumesSample] = tools.loadOrRecompute(
    data_folder + "pair_counts_mcmc_cut_samples.p",
    stacking.get_all_pair_counts_MCMC_samples,allCentres300Gadget,allRadii300,
    rBinStackCentres,snapList,treeList,rBinStack,_recomputeData=False)

# Mean MCMC profiles:
[rhoMCMCToUse, sigmaRhoMCMCToUse] = stacking.get_mean_mcmc_profile(
    allPairsSample,allVolumesSample,cumulative = False)

# Import relevant plotting functions:
from void_analysis.plot import plotConditionedProfile, plotMCMCProfile
from void_analysis.plot import plotStackedProfileVsRandoms, plot_gif_animation


# Select random centres in the random simulations, and compute their
# density contrast:
[randCentres,randOverDen] = tools.loadOrRecompute(\
    data_folder + "random_centres_and_densities.p",\
    simulation_tools.get_random_centres_and_densities,rSphere,snapListUn,
    _recomputeData=False)


# Get MAP densities:
from void_analysis.simulation_tools import get_mcmc_supervolume_densities

deltaMCMCList = tools.loadOrRecompute(data_folder + "delta_list.p",
                                      get_mcmc_supervolume_densities,
                                      snapList,r_sphere=135)

# MAP value of the density of the local super-volume:
from void_analysis.simulation_tools import get_map_from_sample

deltaMAPBootstrap = scipy.stats.bootstrap((deltaMCMCList,),\
    get_map_from_sample,confidence_level = 0.68,vectorized=False,\
    random_state=1000)
deltaMAPInterval = deltaMAPBootstrap.confidence_interval

# Get comparable density regions:
comparableDensityMAP = [(delta <= deltaMAPInterval[1]) & \
    (delta > deltaMAPInterval[0]) for delta in randOverDen]
centresToUse = [randCentres[comp] for comp in comparableDensityMAP]
deltaToUse = [randOverDen[ns][comp] \
    for ns, comp in zip(range(0,len(snapList)),comparableDensityMAP)]

# Bins to use when building a catalogue similar to the constrained
# catalogue:
voidRadiusBinEdges = np.linspace(10,25,6)
[inRadBinsComb,noInRadBinsComb] = plot.binValues(meanRadii,voidRadiusBinEdges)

# Random centres WITHOUT a density condition:
rSep = 2*135
centresListAll = [randCentres for ns in range(0,len(snapListUn))]

indicesAllDensityNonOverlapping = simulation_tools.getNonOverlappingCentres(
    centresListAll,rSep,boxsize,returnIndices=True)

centresAllDensityNonOverlapping = [centres[ind] \
    for centres,ind in zip(centresListAll,indicesAllDensityNonOverlapping)]

densityListNonOverlapping = [density[ind] \
    for density, ind in zip(randOverDen,indicesAllDensityNonOverlapping)]

densityAllNonOverlapping = np.hstack(densityListNonOverlapping)


# Random centres WITH a density condition:
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

# Load all pair-counts data for efficient stacking:
allPairCountsList = [props[9] for props in ahPropsUn]



#-------------------------------------------------------------------------------
# CSV data output for the void catalogue table:

# Filter for the voids we actually want in the catalogue (high catalogue 
# fraction, only out to 135 Mpc/h):
void_filter = filter300
num_voids = np.sum(void_filter)
# Order in which the voids should be sorted:
void_cat_frac = cat300.property_with_filter(
    cat300.finalCatFrac,void_filter=void_filter)
sort_order = np.flip(np.argsort(void_cat_frac))

# Other properties:
void_radii_and_error = cat300.getMeanProperty('radii',void_filter=void_filter)
void_radii = void_radii_and_error[0]
void_radii_error = void_radii_and_error[1]

void_mass_and_error = cat300.getMeanProperty('mass',void_filter=void_filter)
void_mass = void_mass_and_error[0]
void_mass_error = void_mass_and_error[1]

void_centres = cat300.getMeanCentres(void_filter=void_filter)
void_dist = np.sqrt(np.sum(void_centres**2,1))
void_z = astropy.cosmology.z_at_value(
    lambda x: cosmo.comoving_distance(x).value,void_dist).value

# Get RA DEC co-ords:
coord_equatorial = SkyCoord(x=void_centres[:,0]*u.Mpc,\
            y = void_centres[:,1]*u.Mpc,z = void_centres[:,2]*u.Mpc,\
            frame='icrs',representation_type='cartesian')

void_ra = coord_equatorial.icrs.spherical.lon.value
void_dec = coord_equatorial.icrs.spherical.lat.value
void_dist = np.sqrt(np.sum(void_centres**2,1))

# Void SNR:
snrShortened = cat300.getShortenedQuantity(snrAllCatsList,
    cat300.centralAntihalos)
void_snr_and_error = cat300.getMeanProperty(snrShortened,
                                            void_filter=void_filter)
void_snr = void_snr_and_error[0]
void_snr_error = void_snr_and_error[1]
void_delta_central_and_error = cat300.getMeanProperty('deltaCentral',
                                                      void_filter=void_filter)
void_delta_average_and_error = cat300.getMeanProperty('deltaAverage',
                                                      void_filter=void_filter)

void_delta_central = void_delta_central_and_error[0]
void_delta_average = void_delta_average_and_error[0]
void_delta_central_error = void_delta_central_and_error[1]
void_delta_average_error = void_delta_average_and_error[1]

# List of dictionaries with all the relevant void properties:
void_dictionaries = [
    {'ID':str(k+1),
    'rad':("%.2g" % void_radii[sort_order[k]]),
    'rad_error':("%.1g" % void_radii_error[sort_order[k]]),
    'mass':("%.2g" % (void_mass[sort_order[k]]/1e14)),
    'mass_error':("%.1g" % (void_mass_error[sort_order[k]]/1e14)),
    'ra':("%.3g" % void_ra[sort_order[k]]),
    'dec':("%.3g" % void_dec[sort_order[k]]),
    'z':("%.2g" % void_z[sort_order[k]]),
    'dist':("%.3g" % void_dist[sort_order[k]]),
    'snr':("%.3g" % void_snr[sort_order[k]]),
    'cat_frac':("%.2g" % void_cat_frac[sort_order[k]])}
    for k in range(0,num_voids)]

table_titles = {'ID':"ID",'rad':"Radius $(h^{-1}\mathrm{Mpc})$",
                'rad_error':"Radius uncertainty $(h^{-1}\mathrm{Mpc})$",
                'mass':"Mass $(10^{14}h^{-1}M_{\odot})$",
                'mass_error':"Mass uncertainty $(10^{14}h^{-1}M_{\odot})$",
                'ra':'R.A. (deg.)','dec':"Dec. (deg.)",
                'z':"z",'dist':"Distance $(h^{-1}\\mathrm{Mpc})$",
                'snr':"SNR",'cat_frac':"Catalogue Fraction"}

save_name = data_folder + "void_catalogue.csv"
save_name_titled = data_folder + "void_catalogue_titled.csv"

def dictionary_to_csv(dictionary_list,filename,titles_dictionary=None):
    outfile = open(filename,"w")
    # Titles:
    if titles_dictionary is not None:
        for x in titles_dictionary:
            outfile.write(titles_dictionary[x] + ",")
        outfile.write("\n")
    # Data:
    for dictionary in dictionary_list:
        for x in dictionary:
            outfile.write(dictionary[x] + ",")
        outfile.write("\n")
    outfile.close()

# Save data:
dictionary_to_csv(void_dictionaries,save_name)
dictionary_to_csv(void_dictionaries,save_name_titled,
                  titles_dictionary=table_titles)


#-------------------------------------------------------------------------------
# DATA PRODUCTS

# Void catalogues. But we don't want to include everything here!

all_catalogues = [{'radii':cat300.radiusListShort[ns],
                   'mass':cat300.massListShort[ns],
                   'centre':cat300.centresListShort[ns],
                   'dist':np.sqrt(np.sum(cat300.centresListShort[ns]**2,1)),
                   'delta_central':cat300.deltaCentralListShort[ns],
                   'delta_average':cat300.deltaAverageListShort[ns],
                   'snr':snrAllCatsList[ns][cat300.indexListShort[ns]],
                   'ids':np.arange(0,cat300.ahCounts[ns])+1}
                   for ns in range(0,cat300.numCats)]

all_catalogues_rand = [{
    'radii':cat300Rand.radiusListShort[ns],
    'mass':cat300Rand.massListShort[ns],
    'centre':cat300Rand.centresListShort[ns],
    'dist':np.sqrt(np.sum(cat300Rand.centresListShort[ns]**2,1)),
    'delta_central':cat300Rand.deltaCentralListShort[ns],
    'delta_average':cat300Rand.deltaAverageListShort[ns],
    'snr':snrAllCatsList[ns][cat300Rand.indexListShort[ns]],
    'ids':np.arange(0,cat300Rand.ahCounts[ns])+1}
    for ns in range(0,cat300Rand.numCats)]

# Save each catalogue as an npz file:

for ns in range(0,cat300.numCats):
    # MCMC catalogues:
    savename = data_folder + "antihalo_catalogue_sample_" + str(ns+1) + ".npz"
    cat = all_catalogues[ns]
    np.savez(savename,radii=cat['radii'],mass=cat['mass'],centre=cat['centre'],
             dist=cat['dist'],delta_central=cat['delta_central'],
             delta_average=cat['delta_average'],snr=cat['snr'],ids=cat['ids'])
    # Random catalogues:
    savename = data_folder + "antihalo_catalogue_rand_sample_" + str(ns+1) \
        + ".npz"
    cat = all_catalogues_rand[ns]
    np.savez(savename,radii=cat['radii'],mass=cat['mass'],centre=cat['centre'],
             dist=cat['dist'],delta_central=cat['delta_central'],
             delta_average=cat['delta_average'],snr=cat['snr'],ids=cat['ids'])

# As a csv:
titles_dict = {'ID':"ID",'radii':"Radius $(h^{-1}\mathrm{Mpc})$",
               'mass':"Mass $(10^{14}h^{-1}M_{\odot})$",
               'centre_x':"X ($\mathrm{Mpc}h^{-1}$)",
               'centre_y':"Y ($\mathrm{Mpc}h^{-1}$)",
               'centre_z':"Z ($\mathrm{Mpc}h^{-1}$)",
               'dist':"Distance ($\mathrm{Mpc}h^{-1}$)",
               'delta_central':"Central Density Contrast",
               'delta_average':"Average Density Constrast",
               'snr':"Signal-to-Noise Ratio"}
for ns in range(0,cat300.numCats):
    # MCMC catalogues:
    savename = data_folder + "antihalo_catalogue_sample_" + str(ns+1) + ".csv"
    cat = all_catalogues[ns]
    dictionaries = [{'ids':str(cat['ids'][k]),
                     'radii':("%.3g" % cat['radii'][k]),
                     'mass':("%.3g" % cat['mass'][k]),
                     'centre_x':("%.3g" % cat['centre'][k,0]),
                     'centre_y':("%.3g" % cat['centre'][k,1]),
                     'centre_z':("%.3g" % cat['centre'][k,2]),
                     'dist':("%.3g" % cat['dist'][k]),
                     'delta_central':("%.3g" % cat['delta_central'][k]),
                     'delta_average':("%.3g" % cat['delta_average'][k]),
                     'snr':("%.3g" % cat['snr'][k])}
                    for k in range(0,cat300.ahCounts[ns])]
    dictionary_to_csv(dictionaries,savename,titles_dictionary=titles_dict)
    # Random catalogues:
    savename = data_folder + "antihalo_catalogue_rand_sample_" + str(ns+1) \
        + ".csv"
    cat = all_catalogues_rand[ns]
    dictionaries = [{'ids':str(cat['ids'][k]),
                     'radii':("%.3g" % cat['radii'][k]),
                     'mass':("%.3g" % cat['mass'][k]),
                     'centre_x':("%.3g" % cat['centre'][k,0]),
                     'centre_y':("%.3g" % cat['centre'][k,1]),
                     'centre_z':("%.3g" % cat['centre'][k,2]),
                     'dist':("%.3g" % cat['dist'][k]),
                     'delta_central':("%.3g" % cat['delta_central'][k]),
                     'delta_average':("%.3g" % cat['delta_average'][k]),
                     'snr':("%.3g" % cat['snr'][k])}
                    for k in range(0,cat300Rand.ahCounts[ns])]
    dictionary_to_csv(dictionaries,savename,titles_dictionary=titles_dict)

# Saving the final catalogue:

dictionaries_final_filtered = [
    {str(ns):str(cat300.finalCat[filter300][sort_order][k,ns])
     for ns in range(0,cat300.numCats)}
     for k in range(0,np.sum(filter300))]

dictionaries_final_unfiltered = [{str(ns):str(cat300.finalCat[k,ns])
                                for ns in range(0,cat300.numCats)}
                                for k in range(0,len(cat300.finalCat))]

titles_dict_final = {str(ns):"Sample " + str(ns+1) 
                     for ns in range(0,cat300.numCats)}

dictionary_to_csv(dictionaries_final_filtered,data_folder + 
                  "combined_catalogue.csv",titles_dictionary=titles_dict_final)
dictionary_to_csv(dictionaries_final_unfiltered,data_folder + 
                  "combined_catalogue_unfiltered.csv",
                  titles_dictionary=titles_dict_final)

final_cat_properties = [
    {'ID':str(k+1),
    'rad':("%.2g" % void_radii[sort_order[k]]),
    'rad_error':("%.1g" % void_radii_error[sort_order[k]]),
    'mass':("%.2g" % (void_mass[sort_order[k]]/1e14)),
    'mass_error':("%.1g" % (void_mass_error[sort_order[k]]/1e14)),
    'ra':("%.3g" % void_ra[sort_order[k]]),
    'dec':("%.3g" % void_dec[sort_order[k]]),
    'z':("%.2g" % void_z[sort_order[k]]),
    'dist':("%.3g" % void_dist[sort_order[k]]),
    'snr':("%.3g" % void_snr[sort_order[k]]),
    'cat_frac':("%.2g" % void_cat_frac[sort_order[k]]),
    'delta_central':("%.2g" % void_delta_central[sort_order[k]]),
    'delta_central_error':("%.2g" % void_delta_central_error[sort_order[k]]),
    'delta_average':("%.2g" % void_delta_average[sort_order[k]]),
    'delta_average_error':("%.2g" % void_delta_average_error[sort_order[k]])}
    for k in range(0,num_voids)]

final_cat_titles = {'ID':"ID",'rad':"Radius $(h^{-1}\mathrm{Mpc})$",
                    'rad_error':"Radius uncertainty $(h^{-1}\mathrm{Mpc})$",
                    'mass':"Mass $(10^{14}h^{-1}M_{\odot})$",
                    'mass_error':"Mass uncertainty $(10^{14}h^{-1}M_{\odot})$",
                    'ra':'R.A. (deg.)','dec':"Dec. (deg.)",
                    'z':"z",'dist':"Distance $(h^{-1}\\mathrm{Mpc})$",
                    'snr':"SNR",'cat_frac':"Catalogue Fraction",
                    'delta_central':"Central Density Contrast",
                    'delta_central_error':"Central Density Contrast Error",
                    'delta_average':"Average Density Contrast",
                    'delta_average_error':"Average Density Contrast Error"}


dictionary_to_csv(final_cat_properties,
                  data_folder+"combined_catalogue_properties.csv",
                  titles_dictionary=final_cat_titles)

np.savez(data_folder + "combined_catalogue.npz",
         catalogue=cat300.finalCat[filter300][sort_order],
         radii=void_radii[sort_order],radii_error=void_radii_error[sort_order],
         mass=void_mass[sort_order],mass_error=void_mass_error[sort_order],
         ra=void_ra[sort_order],dec=void_dec[sort_order],
         z=void_z[sort_order],dist=void_dist[sort_order],
         snr=void_snr[sort_order],cat_frac=void_cat_frac[sort_order],
         delta_central=void_delta_central[sort_order],
         delta_central_error=void_delta_central_error[sort_order],
         delta_average=void_delta_average[sort_order],
         delta_average_error=void_delta_average_error[sort_order])

np.savez(data_folder + "combined_catalogue_unfiltered.npz",
         catalogue=cat300.finalCat)




#-------------------------------------------------------------------------------
# COMPUTE VOID STACKS APPLYING DIFFERENT CONDITIONS

from void_analysis.catalogue import ProfileStack

rSphereInner = 135

# NO CONSTRAINTS:
noConstraintsStack = ProfileStack(\
    centresAllDensityNonOverlapping,\
    snapListUn,ahPropsUn,rSphereInner,\
    rBinStack,tree_list=treeListUncon,seed=1000,start=0,end=-1,\
    conditioning_quantity=None,\
    conditioning_quantity_to_match=None,\
    condition_bin_edges=None,\
    combine_random_regions=False,replace=False,\
    r_min = voidRadiusBinEdges[0],r_max = voidRadiusBinEdges[-1],\
    compute_pair_counts=True,max_sampling = 1,pair_counts = allPairCountsList)
#noConstraintsDict = noConstraintsStack.get_random_catalogue_pair_counts()
noConstraintsDict = tools.loadOrRecompute(\
    data_folder + "no_constraints_stack.p",\
    noConstraintsStack.get_random_catalogue_pair_counts,_recomputeData=False)
nsListNoConstraints = np.array([[k \
    for centre in centresAllDensityNonOverlapping[k] ] \
    for k in range(0,len(centresAllDensityNonOverlapping))]).flatten()
nsListDensityConstraint = np.hstack([[k \
    for centre in centresUnderdenseNonOverlapping[k] ] \
    for k in range(0,len(centresUnderdenseNonOverlapping))])


# REGION DENSITY CONSTRAINT ONLY:
regionDensityStack = ProfileStack(\
    centresUnderdenseNonOverlapping,\
    snapListUn,ahPropsUn,rSphereInner,\
    rBinStack,tree_list=treeListUncon,seed=1000,start=0,end=-1,\
    conditioning_quantity=None,\
    conditioning_quantity_to_match=None,\
    condition_bin_edges=None,\
    combine_random_regions=False,replace=False,\
    r_min = voidRadiusBinEdges[0],r_max = voidRadiusBinEdges[-1],\
    compute_pair_counts=True,max_sampling = 1,pair_counts = allPairCountsList)
#regionDensityDict = regionDensityStack.get_random_catalogue_pair_counts()
regionDensityDict = tools.loadOrRecompute(\
    data_folder + "regionDensity_stack.p",\
    regionDensityStack.get_random_catalogue_pair_counts,_recomputeData=False)

# REGION DENSITY AND RADIUS CONSTRAINT
conBinEdges = np.linspace(-1,-0.5,21)
conditioningQuantityUn = [antihaloRadiiUn[ns].T\
    for ns in range(0,len(snapListUn))]
conditioningQuantityMCMC = meanRadii
regionDensityAndRadiusStack = ProfileStack(\
    centresUnderdenseNonOverlapping,\
    snapListUn,ahPropsUn,rSphereInner,\
    rBinStack,tree_list=treeListUncon,seed=1000,start=0,end=-1,\
    conditioning_quantity=conditioningQuantityUn,\
    conditioning_quantity_to_match=conditioningQuantityMCMC,\
    condition_bin_edges=[voidRadiusBinEdges],\
    combine_random_regions=False,replace=False,\
    r_min = voidRadiusBinEdges[0],r_max = voidRadiusBinEdges[-1],\
    compute_pair_counts=True,max_sampling = 1,pair_counts = allPairCountsList)
#regionDensityAndRadiusDict = \
#    regionDensityAndRadiusStack.get_random_catalogue_pair_counts()
regionDensityAndRadiusDict = tools.loadOrRecompute(\
    data_folder + "regionDensityAndRadius_stack.p",\
    regionDensityAndRadiusStack.get_random_catalogue_pair_counts,\
    _recomputeData=False)

# REGION DENSITY, VOID RADIUS, AND VOID CENTRA/AVERAGE DENSITY CONSTRAINTS:
conBinEdges = np.linspace(-1,-0.5,21)
conditioningQuantityUn = [np.vstack([antihaloRadiiUn[ns],\
    ahPropsUn[ns][11],ahPropsUn[ns][12]]).T \
    for ns in range(0,len(snapListUn))]
conditioningQuantityMCMC = np.vstack([meanRadii,\
    deltaCentralMean[filter300],deltaAverageMean[filter300]]).T

regionDensityAndTripleConditionStack = ProfileStack(\
    centresUnderdenseNonOverlapping,\
    snapListUn,ahPropsUn,rSphereInner,\
    rBinStack,tree_list=treeListUncon,seed=1000,start=0,end=-1,\
    conditioning_quantity=conditioningQuantityUn,\
    conditioning_quantity_to_match=conditioningQuantityMCMC,\
    condition_bin_edges=[voidRadiusBinEdges,conBinEdges,conBinEdges],\
    combine_random_regions=False,replace=False,\
    r_min = voidRadiusBinEdges[0],r_max = voidRadiusBinEdges[-1],\
    compute_pair_counts=True,max_sampling = 1,pair_counts = allPairCountsList)
#regionDensityAndTripleConditionDict = \
#    regionDensityAndTripleConditionStack.get_random_catalogue_pair_counts()
regionDensityAndTripleConditionDict = tools.loadOrRecompute(\
    data_folder + "regionDensityAndTripleCondition_stack.p",\
    regionDensityAndTripleConditionStack.get_random_catalogue_pair_counts,\
    _recomputeData=False)

# REGION DENSITY AND VOID CENTRAL DENSITY CONSTRAINTS:
conBinEdges = np.linspace(-1,-0.5,21)
conditioningQuantityUn = [np.vstack([ahPropsUn[ns][11]]).T \
    for ns in range(0,len(snapListUn))]
conditioningQuantityMCMC = np.vstack([\
    deltaCentralMean[filter300]]).T

regionAndVoidCentralDensityConditionStack = ProfileStack(\
    centresUnderdenseNonOverlapping,\
    snapListUn,ahPropsUn,rSphereInner,\
    rBinStack,tree_list=treeListUncon,seed=1000,start=0,end=-1,\
    conditioning_quantity=conditioningQuantityUn,\
    conditioning_quantity_to_match=conditioningQuantityMCMC,\
    condition_bin_edges=[conBinEdges],\
    combine_random_regions=False,replace=False,\
    r_min = voidRadiusBinEdges[0],r_max = voidRadiusBinEdges[-1],\
    compute_pair_counts=True,max_sampling = 1,pair_counts = allPairCountsList)
regionAndVoidCentralDensityConditionDict = tools.loadOrRecompute(\
    data_folder + "regionAndVoidCentralDensityCondition_stack.p",\
    regionAndVoidCentralDensityConditionStack.get_random_catalogue_pair_counts,\
    _recomputeData=False)



# REGION DENSITY AND VOID CENTRAL/AVERAGE DENSITY CONSTRAINTS:
conBinEdges = np.linspace(-1,-0.5,21)
conditioningQuantityUn = [np.vstack([ahPropsUn[ns][11],ahPropsUn[ns][12]]).T \
    for ns in range(0,len(snapListUn))]
conditioningQuantityMCMC = np.vstack([\
    deltaCentralMean[filter300],deltaAverageMean[filter300]]).T

regionAndVoidDensityConditionStack = ProfileStack(\
    centresUnderdenseNonOverlapping,\
    snapListUn,ahPropsUn,rSphereInner,\
    rBinStack,tree_list=treeListUncon,seed=1000,start=0,end=-1,\
    conditioning_quantity=conditioningQuantityUn,\
    conditioning_quantity_to_match=conditioningQuantityMCMC,\
    condition_bin_edges=[conBinEdges,conBinEdges],\
    combine_random_regions=False,replace=False,\
    r_min = voidRadiusBinEdges[0],r_max = voidRadiusBinEdges[-1],\
    compute_pair_counts=True,max_sampling = 1,pair_counts = allPairCountsList)
regionAndVoidDensityConditionDict = tools.loadOrRecompute(\
    data_folder + "regionAndVoidDensityCondition_stack.p",\
    regionAndVoidDensityConditionStack.get_random_catalogue_pair_counts,\
    _recomputeData=False)


# REGION DENSITY, AND VOID CENTRAL/AVERAGE DENSITY CONSTRAINTS WITH POOLING:
conBinEdges = np.linspace(-1,-0.5,21)
conditioningQuantityUn = [np.vstack([\
    ahPropsUn[ns][11],ahPropsUn[ns][12]]).T \
    for ns in range(0,len(snapListUn))]
conditioningQuantityMCMC = np.vstack([\
    deltaCentralMean[filter300],deltaAverageMean[filter300]]).T

regionDensityAndAllConditionStackPooled = ProfileStack(\
    centresUnderdenseNonOverlapping,\
    snapListUn,ahPropsUn,rSphereInner,\
    rBinStack,tree_list=treeListUncon,seed=1000,start=0,end=-1,\
    conditioning_quantity=conditioningQuantityUn,\
    conditioning_quantity_to_match=conditioningQuantityMCMC,\
    condition_bin_edges=[conBinEdges,conBinEdges],\
    combine_random_regions=True,replace=False,\
    r_min = voidRadiusBinEdges[0],r_max = voidRadiusBinEdges[-1],\
    compute_pair_counts=True,max_sampling=200,pair_counts = allPairCountsList)
#regionDensityAndAllConditionPooledDict = \
#    regionDensityAndAllConditionStackPooled.get_random_catalogue_pair_counts()
regionDensityAndAllConditionPooledDict = tools.loadOrRecompute(\
    data_folder + "regionDensityAndAllConditionPooled_stack.p",\
    regionDensityAndAllConditionStackPooled.get_random_catalogue_pair_counts,\
    _recomputeData=False)


#-------------------------------------------------------------------------------
# GIF ANIMATION OF THE VOID PROFILES

dictionaries = [noConstraintsDict,regionDensityDict,\
    regionAndVoidCentralDensityConditionDict,regionAndVoidDensityConditionDict]

filenames = ["profiles_no_constraints.png","profiles_regionDensity.png",\
    "profiles_regionDensity_and_voidCentral.png",\
    "profiles_regionDensity_voidCentral_and_voidAverage.png"]
labels = ['No constraints','Region Density',\
    'Region Density + \nVoid Central Density',\
    'Region Density + \nVoid Central & Average Density']
deltaRange = np.array([1 + deltaMAPInterval[0],\
    1 + deltaMAPInterval[1]])
for profileDictionary, savename, label in zip(dictionaries,filenames,labels):
    plotStackedProfileVsRandoms(rBinStackCentres,profileDictionary,nbar,\
        rhoMCMCToUse,sigmaRhoMCMCToUse,deltaRange=deltaRange,\
        savename = figuresFolder + savename,title=label)

imgs = [figuresFolder + savename for savename in filenames]

plot_gif_animation(
    imgs,figuresFolder + 'profiles_constraint_progression_animation.gif')


#-------------------------------------------------------------------------------
# ALL VOID PROFILES IN A 4-PANEL PLOT

from void_analysis.plot import get_axis_handle, get_axis_indices

dictionaries = [noConstraintsDict,regionDensityDict,\
    regionAndVoidCentralDensityConditionDict,regionAndVoidDensityConditionDict]
labels = ['No constraints','Region Density',\
    'Region Density + \nVoid Central Density',\
    'Region Density + \nVoid Central & Average Density']
n_cols = 2
n_rows = 2
ylim = [0,1.2]

plt.clf()
fig, ax = plt.subplots(n_rows,n_cols,figsize=(textwidth,0.7*textwidth))
for k in range(0,len(dictionaries)):
    [i,j] = get_axis_indices(k,n_cols)
    axij = get_axis_handle(i,j,n_rows,n_cols,ax)
    plotConditionedProfile(rBinStackCentres,dictionaries[k],nbar,ax=axij,\
                           intervals=[68,95],alphas=[0.75,0.5])
    plotMCMCProfile(rBinStackCentres,rhoMCMCToUse,sigmaRhoMCMCToUse,nbar,
                    ax = axij)
    axij.axvline(1.0,color='grey',linestyle=':')
    axij.axhline(1.0,color='grey',linestyle=':')
    plot.formatPlotGrid(ax,i,j,0,None,0,None,n_rows,ylim,\
        fontsize=fontsize)
    axij.set_xlim([0,3])
    axij.set_title(labels[k],y=1.0,pad=-3,fontsize=8,va="top")
    axij.tick_params(axis='both',which='major',labelsize=fontsize)
    axij.tick_params(axis='both',which='minor',labelsize=fontsize)
    # Adjust the tick labels to prevent annoying overlaps:
    if i > 0:
        axij.set_yticks(axij.get_yticks()[0:-1])
    if j < n_cols - 1:
        axij.set_xticks(axij.get_xticks()[0:-1])


fig.add_subplot(111,frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, 
                left=False, right=False)
plt.xlabel('$r/r_{\\mathrm{eff}}$',fontsize=8)
plt.ylabel('$\\rho/\\bar{\\rho}$',fontsize=8)
ax[1,1].legend(prop={"size":fontsize,"family":"serif"},frameon=False,\
    loc="lower right")
#plt.tight_layout()
plt.subplots_adjust(hspace=0.0,wspace=0.0,left = 0.08,right=0.98,top=0.98,
                    bottom = 0.09)
plt.savefig(figuresFolder + "profile_constraint_progression_panels.pdf")
plt.show()



#-------------------------------------------------------------------------------
# ALL VOID PROFILES IN A 2-PANEL PLOT

labels = ['No constraints',
          'Region Density + \nVoid Central & \nAverage Density']
dictionaries = [noConstraintsDict,regionAndVoidDensityConditionDict]
n_cols = 2
n_rows = 1
ylim = [0,1.2]

plt.clf()
fig, ax = plt.subplots(n_rows,n_cols,figsize=(textwidth,0.45*textwidth))
for k in range(0,len(dictionaries)):
    [i,j] = get_axis_indices(k,n_cols)
    axij = get_axis_handle(i,j,n_rows,n_cols,ax)
    plotConditionedProfile(rBinStackCentres,dictionaries[k],nbar,ax=axij,\
                           intervals=[68],alphas=[1.0],ec='grey',color='None')
    plotMCMCProfile(rBinStackCentres,rhoMCMCToUse,sigmaRhoMCMCToUse,nbar,
                    ax = axij)
    axij.axvline(1.0,color='grey',linestyle=':')
    axij.axhline(1.0,color='grey',linestyle=':')
    plot.formatPlotGrid(ax,i,j,0,None,0,None,n_rows,ylim,\
        fontsize=fontsize)
    axij.set_xlim([0,3])
    axij.set_title(labels[k],y=1.0,pad=-3,fontsize=8,va="top")
    axij.tick_params(axis='both',which='major',labelsize=fontsize)
    axij.tick_params(axis='both',which='minor',labelsize=fontsize)
    # Adjust the tick labels to prevent annoying overlaps:
    if i > 0:
        axij.set_yticks(axij.get_yticks()[0:-1])
    if j < n_cols - 1:
        axij.set_xticks(axij.get_xticks()[0:-1])


fig.add_subplot(111,frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, 
                left=False, right=False)
plt.xlabel('$r/r_{\\mathrm{eff}}$',fontsize=8)
plt.ylabel('$\\rho/\\bar{\\rho}$',fontsize=8)
ax[1].legend(prop={"size":fontsize,"family":"serif"},frameon=False,\
    loc="lower right")
#plt.tight_layout()
plt.subplots_adjust(hspace=0.0,wspace=0.0,left = 0.08,right=0.98,top=0.98,
                    bottom = 0.15)
plt.savefig(figuresFolder + "profile_constraint_progression_2panels.pdf")
plt.show()

#-------------------------------------------------------------------------------
# CATALOGUE PERMUTATIONS


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

# Order testing:
permutations = []
randomlyOrderedCats = []
goodVoidsPerm = []
np.random.seed(1000)
nPerms = 10
numVoidsPerm = np.zeros(nPerms,dtype=int)
for k in range(0,nPerms):
    perm = np.random.permutation(len(snapList))
    catPerm = catalogue.combinedCatalogue(\
        [snapNameList[k] for k in perm],\
        [snapNameListRev[k] for k in perm],\
        muOpt,rSearchOpt,rSphere,\
        ahProps=[ahProps[k] for k in perm],\
        hrList=[hrList[k] for k in perm],max_index=None,\
        twoWayOnly=True,blockDuplicates=True,\
        massRange = [mMin,mMax],\
        NWayMatch = NWayMatch,r_min=rMin,r_max=rMax,\
        additionalFilters = [snrFilter[k] for k in perm],\
        verbose=False,\
        refineCentres=refineCentres,sortBy=sortBy,\
        enforceExclusive=enforceExclusive)
    finalCatPerm = catPerm.constructAntihaloCatalogue()
    randomlyOrderedCats.append(catPerm)
    [radiiPerm,sigmaRadiiPerm] = catPerm.getMeanProperty("radii")
    meanCentrePerm = catPerm.getMeanCentres()
    distancesPerm = np.sqrt(np.sum(meanCentrePerm**2,1))
    catPerm.set_filter_from_random_catalogue(cat300Rand,radBins,r_sphere=135,
                                             r_min=10,r_max=25)
    numVoidsPerm[k] = np.sum(catPerm.filter)
    goodVoidsPerm.append(catPerm.filter)
    permutations.append(perm)

goodVoidsPerm300 = []
radiiAllPerms = []
for k in range(0,nPerms):
    catPerm = randomlyOrderedCats[k]
    [radiiPerm,sigmaRadiiPerm] = catPerm.getMeanProperty("radii")
    meanCentrePerm = catPerm.getMeanCentres()
    distancesPerm = np.sqrt(np.sum(meanCentrePerm**2,1))
    catPerm.set_filter_from_random_catalogue(cat300Rand,radBins,r_sphere=300,
                                             r_min=10,r_max=25)
    filterPerm = catPerm.filter
    catPerm.set_filter_from_random_catalogue(cat300Rand,radBins,r_sphere=135,
                                             r_min=10,r_max=25)
    goodVoidsPerm300.append(filterPerm)
    radiiAllPerms.append(radiiPerm)

massFunctionsPerm135 = [cat.getMeanProperty("mass")[0][filt] \
    for cat, filt in zip(randomlyOrderedCats,goodVoidsPerm)]
massFunctionsPerm300 = [cat.getMeanProperty("mass")[0][filt] \
    for cat, filt in zip(randomlyOrderedCats,goodVoidsPerm300)]

vsfPerm135 = [cat.getMeanProperty("radii")[0][filt] \
    for cat, filt in zip(randomlyOrderedCats,goodVoidsPerm)]
vsfPerm300 = [cat.getMeanProperty("radii")[0][filt] \
    for cat, filt in zip(randomlyOrderedCats,goodVoidsPerm300)]















#-------------------------------------------------------------------------------
# MASS FUNCTIONS PLOT


# Mass functions:
leftFilter = (radiiMean300 > 10) & (radiiMean300 <= 25) & \
    (distances300 < 135) & (cat300.finalCatFrac > thresholds300)
rightFilter = (radiiMean300 > 10) & (radiiMean300 <= 25) & \
    (distances300 < 300) & (cat300.finalCatFrac > thresholds300)

nBins = 8
Om = referenceSnap.properties['omegaM0']
rhoc = 2.7754e11
boxsize = referenceSnap.properties['boxsize'].ratio("Mpc a h**-1")
N = int(np.cbrt(len(referenceSnap)))
mUnit = 8*Om*rhoc*(boxsize/N)**3
mLower = 100*mUnit
mUpper = 2e15
rSphere = 300
rSphereInner = 135
# Check mass functions:
volSphere135 = 4*np.pi*rSphereInner**3/3
volSphere = 4*np.pi*rSphere**3/3

# Bootstrap error bars?

def bin_count(data,low,high):
    return np.sum((data > low) & (data <= high))

mass_bins = 10**np.linspace(np.log10(mLower),np.log10(mUpper),nBins)

bin_bootstraps = [scipy.stats.bootstrap(
    (massMean300[leftFilter],),
    lambda x: bin_count(x,mass_bins[k],mass_bins[k+1]),
    confidence_level = 0.68,vectorized=False,random_state=2000) 
    for k in range(0,len(mass_bins)-1)]

errors = [x.confidence_interval for x in bin_bootstraps]

# Or, just take bootstrap samples:
def bootstrap_mass_function(masses,n_boot = 1000,seed=2000):
    n_data = len(masses)
    mass_function_samples = []
    np.random.seed(seed)
    for k in range(0,n_boot):
        mass_function_samples.append(
            np.random.choice(masses,size=(n_data),replace=True))
    return mass_function_samples

# Just using a single line (no error bar):
mass_left = cat300.getMeanProperty("mass",void_filter=leftFilter)
mass_right = cat300.getMeanProperty("mass",void_filter=rightFilter)
mass_samples_left = mass_left[0]
mass_samples_right = mass_right[0]
mass_error_left = cat300.getAllProperties('mass',void_filter=leftFilter)
mass_error_right = cat300.getAllProperties('mass',void_filter=rightFilter)
# Using bootstrap errors:
#mass_samples_left = bootstrap_mass_function(massMean300[leftFilter])
#mass_samples_right = bootstrap_mass_function(massMean300[rightFilter])
# Using permuted masses as errors:
#mass_samples_left = massFunctionsPerm135
#mass_samples_right = massFunctionsPerm300
#mass_error_left = None
#mass_error_right = None

plt.clf()
doCat = True
if doCat:
    plot.massFunctionComparison(mass_samples_left,
        mass_samples_right,4*np.pi*135**3/3,nBins=nBins,
        labelLeft = "Combined catalogue ($68\%$)" 
        + " \n(well-constrained voids only)",
        labelRight  ="Combined catalogue ($68\%$) " 
        + "\n(well-constrained voids only)",
        ylabel="Number of antihalos",savename=figuresFolder + 
        "mass_function_combined_300vs135_test.pdf",massLower=mLower,
        ylim=[1,1000],Om0 = 0.3111,h=0.6766,sigma8=0.8128,ns=0.9667,
        fontsize=8,massUpper = mUpper,
        titleLeft = "Combined catalogue, $<135\\mathrm{Mpc}h^{-1}$",
        titleRight = "Combined catalogue, $<300\\mathrm{Mpc}h^{-1}$",
        volSimRight = 4*np.pi*300**3/3,ylimRight=[1,1000],
        legendLoc="upper right",errorType="shaded",massErrors=True,
        error_type="bernoulli",hmf_interval=68,weight_model="bootstrap",
        mass_error_left = mass_error_left,mass_error_right=mass_error_right,\
        error_interval=68,poisson_interval=0.68,powerRange=2)


#-------------------------------------------------------------------------------
# VOID SIZE FUNCTION PLOT

from void_analysis.plot import plot_void_counts_radius

# Mass functions:
leftFilter = (radiiMean300 > 10) & (radiiMean300 <= 25) & \
    (distances300 < 135) & (cat300.finalCatFrac > thresholds300)
rightFilter = (radiiMean300 > 10) & (radiiMean300 <= 25) & \
    (distances300 < 300) & (cat300.finalCatFrac > thresholds300)


radius_bins = np.linspace(10,21,7)



# Actual plot:
mean_radii_mcmc = cat300.getMeanProperty('radii',void_filter=leftFilter)
all_radii_mcmc = cat300.getAllProperties('radii',void_filter=leftFilter)
#mean_radii_mcmc = cat300test.getMeanProperty('radii',
#                                             void_filter=cat300test.filter)
#mean_radii_mcmc = vsfPerm135

plt.clf()
fig, ax = plt.subplots(figsize=(0.45*textwidth,0.45*textwidth))
plot_void_counts_radius(mean_radii_mcmc[0],radius_bins,
                        noConstraintsDict['radii'],ax=ax,do_errors=True,
                        radii_errors = all_radii_mcmc,
                        label="MCMC catalogue ($68\%$)",
                        lcdm_label="$\\Lambda$-CDM ($68\%$)",
                        weight_model="bootstrap",mcmc_interval=68,
                        confidence=0.68)

ax.tick_params(axis='both',which='major',labelsize=fontsize)
ax.tick_params(axis='both',which='minor',labelsize=fontsize)
plt.subplots_adjust(left = 0.17,right=0.97,bottom = 0.15,top = 0.97)
plt.savefig(figuresFolder + "void_size_function.pdf")
#plt.savefig(figuresFolder + "void_size_function_test.pdf")
plt.show()


# Histogram bootstrap samples to check the distribution of the mean:
import seaborn
def plot_bootstrap_mean_distribution(data,savename,seed=1000,n_boot=10000,
                                     n_bins = 21,alpha=0.5,color=None,
                                     xlabel = "Mean radius " 
                                     + "[$\\mathrm{Mpc}h^{-1}$]",
                                     ylabel="Probability Density",
                                     title="Bootstrap mean distribution.",
                                     plot_type="histogram",textwidth=7.1014,
                                     fontsize=8):
    # Clean data:
    np.random.seed(seed)
    data_to_use = data[np.isfinite(data)]
    samples = np.random.choice(data_to_use,
                               size=(len(data_to_use),n_boot),replace=True)
    means = np.mean(samples,0)
    mean_std = np.std(means)
    mean_mean = np.mean(means)
    bins = np.linspace(np.min(means),np.max(means),n_bins)
    if color is None:
        color = seabornColormap[0]
    plt.clf()
    fig, ax = plt.subplots(figsize=(0.5*textwidth,0.5*textwidth))
    if plot_type == "histogram":
        hist = plt.hist(means,bins=bins,color=color,alpha=alpha,density=True,
            label='Histogram of means')
    elif plot_type == "kde":
        seaborn.kdeplot(data=means,color=color,alpha=alpha,
                        label='KDE of means')
    x = plot.binCentres(bins)
    gaussian = np.exp(-(x - mean_mean)**2/(2*mean_std**2))/\
        np.sqrt(2*np.pi*mean_std**2)
    plt.plot(x,gaussian,linestyle=':',color='k',
             label='Gaussian \napproximation')
    plt.xlabel(xlabel,fontsize=fontsize)
    plt.ylabel(ylabel,fontsize=fontsize)
    plt.title(title,fontsize=fontsize)
    plt.legend(prop={"size":fontsize,"family":"serif"},frameon=False)
    plt.tight_layout()
    plt.savefig(savename)
    plt.show()



plot_bootstrap_mean_distribution(all_radii_mcmc[0],
                                 figuresFolder + "mean_test_radius.pdf",
                                 n_bins=31,plot_type="kde")


plot_bootstrap_mean_distribution(mass_error_left[0],
                                 figuresFolder + "mean_test_mass.pdf",
                                 xlabel="Mass [$M_{\\odot}h^{-1}$]",n_bins=31,
                                 plot_type="kde")


#-------------------------------------------------------------------------------
# SKYPLOT

def get_snapsort_lists(snapList,recompute=False):
    snapsort_list = []
    for snap in snapList:
        filename = snap.filename + ".snapsort.p"
        if os.path.isfile(filename) and not recompute:
            snapsort_list.append(tools.loadPickle(filename))
        else:
            snapsort_list.append(np.argsort(snap['iord']))
    return snapsort_list


snapSortList = get_snapsort_lists(snapList,recompute=False)
snapToShow = snapList[0]

# Verify positions are in the correct co-ordinates:

for snap in snapList:
    tools.remapBORGSimulation(snap,swapXZ=False,reverse=True)

# Alpha shapes

[ahMWPos,alpha_shapes_finalCat,alpha_shapes_individual] = \
    tools.loadOrRecompute(
        data_folder + "cat300_alpha_shapes.p",cat300.get_alpha_shapes,
        snapList,snapListRev,antihaloCatalogueList=hrList,ahProps = ahProps,
        snapsortList=snapSortList,void_filter=filter300,reCentreSnaps=False,
        _recomputeData = False)



# Plots:


radiiOpt = cat300.getMeanProperty('radii',void_filter=filter300)[0]
sortedRadiiOpt = np.flip(np.argsort(radiiOpt))

catToUse = cat300.get_final_catalogue(void_filter=filter300)
cat_object = cat300
haveVoids = [np.where(catToUse[:,ns] > 0)[0] \
    for ns in range(0,len(snapNameList))]
cat_fracs = cat300.property_with_filter(cat300.finalCatFrac,
                                        void_filter=filter300)
cat_centres = cat300.getMeanCentres(void_filter=filter300)
cat_distances = np.sqrt(np.sum(cat_centres**2,1))
sorted_cat_frac = np.flip(np.argsort(cat_fracs))

#for ns in range(0,len(snapNameList)):
#    catToUse[haveVoids[ns],ns] = np.arange(1,len(haveVoids[ns])+1)



filterToUse = np.where(filter300)[0]
nVoidsToShow = 10
#nVoidsToShow = len(filterToUse)
#selection = np.intersect1d(sortedRadiiOpt,filterToUse)[:(nVoidsToShow)]
#selection = sortedRadiiOpt[cat_fracs > 0.7][np.arange(0,nVoidsToShow)]
selection = sorted_cat_frac[np.arange(0,nVoidsToShow)]
asListAll = []
colourListAll = []
laListAll = []
labelListAll = []

plotFormat='.png'
#plotFormat='.pdf'

textwidth=7.1014
textheight=9.0971
scale = 1.18
ratio = 0.58
width = textwidth
height = ratio*textwidth
cropPoint = ((scale -1)/2)*np.array([width,height]) + \
    np.array([0,0])
bound_box = transforms.Bbox([[cropPoint[0], cropPoint[1]],
    [cropPoint[0] + width, cropPoint[1] + height]])


for ns in range(0,len(snapNameList)):
    asList = []
    colourList = []
    laList = []
    labelList = []
    for k in range(0,np.min([nVoidsToShow,len(selection)])):
        if catToUse[selection[k],ns] > 0:
            listPosition = catToUse[selection[k],ns]-1
            ahNumber = cat_object.indexListShort[ns][listPosition]
            asList.append(alpha_shapes_individual[ns][selection[k]])
            laList.append(ahNumber)
            colourList.append(
                seabornColormap[np.mod(k,len(seabornColormap))])
            labelList.append(str(k+1))
        print("Done for void " + str(k+1))
    print("Done for sample " + str(ns+1))
    asListAll.append(asList)
    colourListAll.append(colourList)
    laListAll.append(laList)
    labelListAll.append(labelList)

asListTot = []
colourListTot = []
laListTot = []
labelListTot = []
for k in range(0,np.min([nVoidsToShow,len(selection)])):
    listPosition = catToUse[selection[k],ns]-1
    ahNumber = listPosition
    asListTot.append(alpha_shapes_finalCat[selection[k]])
    laListTot.append(ahNumber)
    colourListTot.append(
        seabornColormap[np.mod(k,len(seabornColormap))])
    labelListTot.append(str(k+1))
    print("Done for void " + str(k+1))

# Pre-compute density field healpix maps, so we don't have to
# recompute them everytime we want to make small changes:
Om = snapList[0].properties['omegaM0']
rhoc = 2.7754e11
rhobar = rhoc*Om
vmin = 1e-2
vmax = 1e2
rCut=135
nside = 64
ha = ['right','left','left','left','left','right','center',\
        'left','right']
va = ['center','center','bottom','bottom','top',\
        'top','center','center','center']
annotationPos = [[-1.1,0.9],\
        [1.5,0.3],[1.7,0.6],[1.3,-1.2],[1.5,-0.7],[-1.8,-0.7],[1.4,0.9],\
        [1.5,0.05],[-1.7,0.5]]
nameList = [name[0] for name in clusterNames]



hpx_map_list = [plot.sphericalSlice(
    snap,rCut/2,thickness=rCut,fillZeros=vmin*rhobar,centre=np.array([0,0,0]),
    nside=nside)/rhobar for snap in snapList]


# All anti-halos:
for ns in range(0,len(snapNumList)):
    plt.clf()
    plot.plotLocalUniverseMollweide(rCut,snapList[ns],\
        alpha_shapes = asListAll[ns],\
        largeAntihalos = laListAll[ns],hr=hrList[ns],\
        coordAbell = coordCombinedAbellSphere,\
        abellListLocation = clusterIndMain,\
        nameListLargeClusters = [name[0] for name in clusterNames],\
        ha = ha,va= va, annotationPos = annotationPos,\
        title = 'Sample ' +str(ns+1) + ': large voids (antihalos) within $' + \
        str(rCut) + "\\mathrm{\\,Mpc}h^{-1}$",\
        vmin=1e-2,vmax=1e2,legLoc = 'lower left',\
        bbox_to_anchor = (-0.1,-0.2),\
        snapsort = snapSortList[ns],antihaloCentres = None,\
        figOut = figuresFolder + "/ah_match_sample_" + \
        str(ns) + plotFormat,\
        showFig=False,figsize = (scale*textwidth,scale*ratio*textwidth),\
        voidColour = colourListAll[ns],antiHaloLabel=labelListAll[ns],\
        bbox_inches = bound_box,galaxyAngles=equatorialRThetaPhi[:,1:],\
        galaxyDistances = equatorialRThetaPhi[:,0],showGalaxies=False,\
        voidAlpha = 0.6,margins=None,hpxMap = hpx_map_list[ns],pad=0.05,
        cbar_aspect=10,shrink=0.35,cbar_y_pos=0.17)
    plt.show()

# Animation:
imgs = [
    figuresFolder + "ah_match_sample_" +str(ns) + ".png" 
    for ns in range(0,len(snapList))]

plot_gif_animation(
    imgs,figuresFolder + 'sky_plot_animation_largest_voids.gif')

# Combined outlines (plotting not working at present for unknown reasons)
ns = 0
plot.plotLocalUniverseMollweide(rCut,snapList[ns],\
    alpha_shapes = asListTot,\
    largeAntihalos = laListTot,hr=hrList[ns],\
    coordAbell = coordCombinedAbellSphere,\
    abellListLocation = clusterIndMain,\
    nameListLargeClusters = [name[0] for name in clusterNames],\
    ha = ha,va= va, annotationPos = annotationPos,\
    title = 'Local super-volume: large voids (antihalos) within $' + \
    str(rCut) + "\\mathrm{\\,Mpc}h^{-1}$",\
    vmin=1e-2,vmax=1e2,legLoc = 'lower left',\
    bbox_to_anchor = (-0.1,-0.2),\
    snapsort = snapSortList[ns],antihaloCentres = ahMWPos,\
    figOut = figuresFolder + "/ah_match_combined" + plotFormat,\
    showFig=False,figsize = (scale*textwidth,scale*0.55*textwidth),\
    voidColour = colourListTot,antiHaloLabel=labelListTot,\
    bbox_inches = bound_box,galaxyAngles=equatorialRThetaPhi[:,1:],\
    galaxyDistances = equatorialRThetaPhi[:,0],showGalaxies=False,\
    voidAlpha = 0.6,margins=None,positions = [None for x in laListTot],
    hpxMap = hpx_map_list[ns],pad=0.05,cbar_aspect=10,shrink=0.35,
    cbar_y_pos=0.2)


#-------------------------------------------------------------------------------
# VOID DENSITY DISTRIBUTION PLOT


# Details of the stack to plot:
centreListToTest = centresUnderdenseNonOverlapping
radBinEdges = voidRadiusBinEdges
meanRadiiMCMC = meanRadii
#conditionBinEdges = [voidRadiusBinEdges,conBinEdges,conBinEdges]
conditionBinEdges = [conBinEdges,conBinEdges]
combineRandomRegions = True
start = 0
end = -1
seed = 1000
constrainedDictionary = regionAndVoidDensityConditionDict
#constrainedDictionary = regionDensityAndAllConditionPooledDict
#constrainedDictionary = regionDensityAndTripleConditionDict
#constrainedStack = regionDensityAndAllConditionStackPooled
constrainedStack = regionAndVoidDensityConditionStack

conditioningVariable = np.vstack([\
    conditioningQuantityUn[ns][ind] \
    for ns, ind in zip(nsListNoConstraints,noConstraintsDict['selections'])])
conditionedMasses = np.hstack([\
    antihaloMassesUn[ns][ind] \
    for ns, ind in zip(nsListNoConstraints,noConstraintsDict['selections'])])
conditionedRadii = np.hstack([\
    antihaloRadiiUn[ns][ind] \
    for ns, ind in zip(nsListNoConstraints,noConstraintsDict['selections'])])
conditionedCentralDensity = np.hstack([\
    ahPropsUn[ns][11][ind] \
    for ns, ind in zip(nsListNoConstraints,noConstraintsDict['selections'])])
conditionedAverageDensity = np.hstack([\
    ahPropsUn[ns][12][ind] \
    for ns, ind in zip(nsListNoConstraints,noConstraintsDict['selections'])])
if not constrainedStack.combine_random_regions:
    allSelectedConditions = np.vstack(\
        constrainedDictionary['selected_conditions'])
    allSelectedMasses = np.hstack([antihaloMassesUn[ns][ind] \
        for ns, ind in zip(nsListDensityConstraint,\
        constrainedDictionary['selections'])])
    allSelectedRadii = np.hstack([antihaloRadiiUn[ns][ind] \
        for ns, ind in zip(nsListDensityConstraint,\
        constrainedDictionary['selections'])])
    allSelectedCentralDensity = np.hstack([ahPropsUn[ns][11][ind] \
        for ns, ind in zip(nsListDensityConstraint,\
        constrainedDictionary['selections'])])
    allSelectedAverageDensity = np.hstack([ahPropsUn[ns][12][ind] \
        for ns, ind in zip(nsListDensityConstraint,\
        constrainedDictionary['selections'])])
else:
    allSelectedConditions = np.vstack(\
        constrainedDictionary['selected_conditions'])
    constrainedStack.get_all_condition_variables()
    allMasses = constrainedStack.get_pooled_variable(antihaloMassesUn)
    allRadii = constrainedStack.get_pooled_variable(antihaloRadiiUn)
    allCentralDensity = constrainedStack.get_pooled_variable(
        [ahPropsUn[ns][11] for ns in range(0,len(snapListUn))])
    allAverageDensity = constrainedStack.get_pooled_variable(
        [ahPropsUn[ns][12] for ns in range(0,len(snapListUn))])
    allSelectedMasses = np.hstack(
        [allMasses[ind] for ind in constrainedDictionary['selections']])
    allSelectedRadii = np.hstack(
        [allRadii[ind] for ind in constrainedDictionary['selections']])
    allSelectedCentralDensity = np.hstack(
        [allCentralDensity[ind] 
        for ind in constrainedDictionary['selections']])
    allSelectedAverageDensity = np.hstack(
        [allAverageDensity[ind] 
        for ind in constrainedDictionary['selections']])

replace = False


massBins = 10**(np.linspace(np.log10(1e13),np.log10(1e15),nBinEdges))
[samplingMCMC,edges] = np.histogramdd(conditioningQuantityMCMC,\
    bins = conditionBinEdges)
[samplingMCMCMasses,edges] = np.histogramdd(meanMasses,bins=[massBins])
[samplingMCMCRadii,edges] = np.histogramdd(meanRadii,bins=[radBins])
[samplingMCMCCentralDensity,edges] = np.histogramdd(\
    deltaCentralMean[filter300],bins=[conBinEdges])
[samplingMCMCAverageDensity,edges] = np.histogramdd(
    deltaAverageMean[filter300],bins=[conBinEdges])
samplingMCMCLin = np.array(samplingMCMC.flatten(),dtype=int)
[samplingRand,edges] = np.histogramdd(conditioningVariable,\
    bins = conditionBinEdges)
[samplingRandMasses,edges] = np.histogramdd(conditionedMasses,bins=[massBins])
[samplingRandRadii,edges] = np.histogramdd(conditionedRadii,bins=[radBins])
[samplingRandCentralDensity,edges] = np.histogramdd(conditionedCentralDensity,
                                                    bins=[conBinEdges])
[samplingRandAverageDensity,edges] = np.histogramdd(conditionedAverageDensity,
                                                    bins=[conBinEdges])
samplingRandLin = np.array(samplingRand.flatten(),dtype=int)

#[samplingRandSelected,edges] = np.histogramdd(\
#    conditioningVariable[selectArray],bins = conditionBinEdges)
[samplingRandSelected,edges] = np.histogramdd(\
    allSelectedConditions,bins = conditionBinEdges)
[samplingRandSelectedMasses,edges] = np.histogramdd(\
    allSelectedMasses,bins=[massBins])
[samplingRandSelectedRadii,edges] = np.histogramdd(\
    allSelectedRadii,bins=[radBins])
[samplingRandSelectedCentralDensity,edges] = np.histogramdd(\
    allSelectedCentralDensity,bins=[conBinEdges])
[samplingRandSelectedAverageDensity,edges] = np.histogramdd(\
    allSelectedAverageDensity,bins=[conBinEdges])



nzMCMC = np.where(samplingMCMCLin > 0)
rat = np.zeros(samplingMCMCLin.shape,dtype=int)
rat[nzMCMC] = samplingRandLin[nzMCMC]/samplingMCMCLin[nzMCMC]
minRatio = np.min(rat[rat > 0])

#samplingRand0 = np.sum(samplingRand,(1,2))
#samplingRand1 = np.sum(samplingRand,(0,2))
#samplingRand2 = np.sum(samplingRand,(0,1))
samplingRand0 = samplingRandRadii
samplingRand1 = samplingRandCentralDensity
samplingRand2 = samplingRandAverageDensity
samplingRand3 = samplingRandMasses


#samplingMCMC0 = np.sum(samplingMCMC,(1,2))
#samplingMCMC1 = np.sum(samplingMCMC,(0,2))
#samplingMCMC2 = np.sum(samplingMCMC,(0,1))
samplingMCMC0 = samplingMCMCRadii
samplingMCMC1 = samplingMCMCCentralDensity
samplingMCMC2 = samplingMCMCAverageDensity
samplingMCMC3 = samplingMCMCMasses

#samplingRandSelected0 = np.sum(samplingRandSelected,(1,2))
#samplingRandSelected1 = np.sum(samplingRandSelected,(0,2))
#samplingRandSelected2 = np.sum(samplingRandSelected,(0,1))

samplingRandSelected0 = samplingRandSelectedRadii
samplingRandSelected1 = samplingRandSelectedCentralDensity
samplingRandSelected2 = samplingRandSelectedAverageDensity
samplingRandSelected3 = samplingRandSelectedMasses

def getBarWidths(bins):
    return (bins[1:] - bins[0:-1])

def getBarHeights(counts,bins):
    widths = getBarWidths(bins)
    return counts/(np.sum(counts)*widths)

def barAsHist(counts,bins,ax=None,density=True,**kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    centres = plot.binCentres(bins)
    widths = (bins[1:] - bins[0:-1])
    if density:
        heights = getBarHeights(counts,bins)
    else:
        heights = counts
    ax.bar(centres,heights,width=widths,**kwargs)

# All distributions:
nRows = 2
#nCols = 4
nCols = 2

#randCounts = [[samplingRand0,samplingRand1,samplingRand2,samplingRand3],\
#    [samplingRandSelected0,samplingRandSelected1,samplingRandSelected2,\
#    samplingRandSelected3]]

randCounts = [[samplingRand1,samplingRand2],\
    [samplingRandSelected1,samplingRandSelected2]]

#mcmcCounts = [[samplingMCMC0,samplingMCMC1,samplingMCMC2,samplingMCMC3],\
#    [samplingMCMC0,samplingMCMC1,samplingMCMC2,samplingMCMC3]]

mcmcCounts = [[samplingMCMC1,samplingMCMC2],\
    [samplingMCMC1,samplingMCMC2]]

#histBins = [radBins,conBinEdges,conBinEdges,massBins]
histBins = [conBinEdges,conBinEdges]

#xlabels = ['$R [\\mathrm{Mpc}h^{-1}]$','$\\delta_{\\mathrm{central}}$',\
    '$\\bar{\\delta}$','Mass [$M_{\\odot}h^{-1}$]']
xlabels = ['$\\delta_{\\mathrm{central}}$',\
    '$\\bar{\\delta}$']
#ylims = [[0,0.3],[0,20],[0,20],[0,7e-15]]
ylims = [[0,30],[0,30]]
xlims = [[-0.95,-0.7],[-0.85,-0.6]]
#titlesList = [['Void radii, \nall samples','Central Density, \nall samples',\
#    'Average Density, \nall samples','Void mass, \nall samples'],\
#    ['Void radii, \nconditioned samples',\
#    'Central Density, \nconditioned samples',\
#    'Average Density, \nconditioned samples',\
#    'Void mass, \nconditioned samples']]

titlesList = [['Central Density, \nall samples',\
    'Average Density, \nall samples'],\
    ['Central Density, \nconditioned samples',\
    'Average Density, \nconditioned samples']]

rowText = ["All samples","Conditioned Samples"]
colText = ["Central Density","Average Density"]

plt.clf()
fig, ax = plt.subplots(nRows,nCols,figsize=(textwidth,0.7*textwidth))
for i in range(0,nRows):
    for j in range(0,nCols):
        axij = get_axis_handle(i,j,nRows,nCols,ax)
        barAsHist(randCounts[i][j],histBins[j],ax=axij,\
            alpha=0.5,color=seabornColormap[0],label="Randoms")
        barAsHist(mcmcCounts[i][j],histBins[j],ax=axij,\
            alpha=0.5,color=seabornColormap[1],label="MCMC")
        if j == 3:
            axij.set_xscale('log')
            axij.set_xlim([1e13,1e15])
        plot.formatPlotGrid(ax,i,j,None,None,None,None,nRows,ylims[j],\
            fontsize=fontsize)
        axij.set_xlabel(xlabels[j],fontsize=fontsize,fontfamily = "serif")
        axij.set_ylabel('Probability Density',fontsize=fontsize,\
            fontfamily = "serif")
        axij.set_xlim(xlims[j])
        if i == 0:
            axij.set_title(colText[i],fontsize = fontsize,\
                fontfamily = "serif")
        axij.tick_params(axis='both',which='major',labelsize=fontsize)
        axij.tick_params(axis='both',which='minor',labelsize=fontsize)
        # Adjust the tick labels to prevent annoying overlaps:
        if i > 0:
            axij.set_yticks(axij.get_yticks()[0:-1])
        if j < n_cols - 1:
            axij.set_xticks(axij.get_xticks()[0:-1])

# Adjustment parameters:
left=0.1
right=0.93
top=0.93
bottom = 0.1
wspace=0.0
hspace=0.0
spacing = 1/nRows
start = 1/(2*nRows)

for l in range(0,nRows):
    fig.text(right + 0.025,top + (bottom - top)*(start + l*spacing),\
             rowText[l],\
             fontsize=fontsize,fontfamily="serif",ha='center',\
             rotation='vertical',va='center')

plt.subplots_adjust(wspace=wspace,hspace=hspace,left=left,right=right,
                    top=top,bottom=bottom)
ax[1,1].legend(prop={"size":fontsize,"family":"serif"},frameon=False)
plt.savefig(figuresFolder + "all_density_distributions.pdf")
plt.show()




