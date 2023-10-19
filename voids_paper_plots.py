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

# Signal to noise information:
snrThresh=10
chainFile="chain_properties.p"
[snrFilter,snrAllCatsList] = getSNRFilterFromChainFile(chainFile,snrThresh,\
    snapNameList,boxsize)

if recomputeCatalogues or (not os.path.isfile(data_folder + "cat300.p")):
    cat300 = tools.loadOrRecompute(
        data_folder + "cat300.p",\
        catalogue.combinedCatalogue,snapNameList,snapNameListRev,\
        muOpt,rSearchOpt,rSphere,\
        ahProps=ahProps,hrList=hrList,max_index=None,\
        twoWayOnly=True,blockDuplicates=True,\
        massRange = [mMin,mMax],\
        NWayMatch = NWayMatch,rMin=rMin,rMax=rMax,\
        additionalFilters = snrFilter,verbose=False,\
        refineCentres=refineCentres,sortBy=sortBy,\
        enforceExclusive=enforceExclusive,_recomputeData = recomputeCatalogues)
    finalCat300 = cat300.constructAntihaloCatalogue()
else:
    cat300 = tools.loadPickle(data_folder + "cat300.p")

# Random catalogues:
snapNameListRand = [snap.filename for snap in snapListUn]
snapNameListRandRev = [snap.filename for snap in snapListRevUn]

if recomputeCatalogues or (not os.path.isfile(data_folder + "cat300Rand.p")):
    cat300Rand = tools.loadOrRecompute(
        data_folder + "cat300Rand.p",\
        catalogue.combinedCatalogue,snapNameListRand,snapNameListRandRev,\
        muOpt,rSearchOpt,rSphere,\
        ahProps=ahPropsUn,hrList=hrListUn,max_index=None,\
        twoWayOnly=True,blockDuplicates=True,\
        massRange = [mMin,mMax],\
        NWayMatch = NWayMatch,rMin=rMin,rMax=rMax,\
        additionalFilters = None,verbose=False,\
        refineCentres=refineCentres,sortBy=sortBy,\
        enforceExclusive=enforceExclusive,_recomputeData = recomputeCatalogues)
    finalCat300Rand = cat300Rand.constructAntihaloCatalogue()
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
va = ['center','bottom','bottom','bottom','top',\
        'top','center','center','center']
annotationPos = [[-1.1,0.9],\
        [1.1,1.0],[1.5,0.6],[1.3,-1.2],[1.3,-0.7],[-1,0.2],[0.8,0.6],\
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
meanCentresGadgetCoord = snapedit.wrap(np.fliplr(meanCentres) + boxsize/2,
                                       boxsize)
meanRadii = radiiMean300[filter300]
meanMasses = massMean300[filter300]
allCentres300Gadget = np.array([snapedit.wrap(\
    np.fliplr(centre) + boxsize/2,boxsize) \
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
def get_mcmc_supervolume_densities(snap_list,r_sphere=135):
    boxsize = snap_list[0].properties['boxsize'].ratio("Mpc a h**-1")
    deltaMCMCList = np.array(\
        [simulation_tools.density_from_snapshot(
        snap,np.array([boxsize/2]*3),r_sphere) \
        for snap in snap_list])
    return deltaMCMCList

deltaMCMCList = tools.loadOrRecompute(data_folder + "delta_list.p",
                                      get_mcmc_supervolume_densities,
                                      snapList,r_sphere=135)

# MAP value of the density of the local super-volume:
def getMAPFromSample(sample):
    kde = scipy.stats.gaussian_kde(sample,bw_method="scott")
    return scipy.optimize.minimize(lambda x: -kde.evaluate(x),\
        np.mean(sample)).x[0]

deltaMAPBootstrap = scipy.stats.bootstrap((deltaMCMCList,),\
    getMAPFromSample,confidence_level = 0.68,vectorized=False,\
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
plt.xlabel('$R/R_{\\mathrm{eff}}$',fontsize=8)
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

labels = ['No constraints','Region Density + \nVoid Central & Average Density']
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
plt.xlabel('$R/R_{\\mathrm{eff}}$',fontsize=8)
plt.ylabel('$\\rho/\\bar{\\rho}$',fontsize=8)
ax[1].legend(prop={"size":fontsize,"family":"serif"},frameon=False,\
    loc="lower right")
#plt.tight_layout()
plt.subplots_adjust(hspace=0.0,wspace=0.0,left = 0.08,right=0.98,top=0.98,
                    bottom = 0.15)
plt.savefig(figuresFolder + "profile_constraint_progression_2panels.pdf")
plt.show()

#-------------------------------------------------------------------------------
# MASS FUNCTIONS PLOT


# Mass functions:
leftFilter = (radiiMean300 > 10) & (radiiMean300 <= 25) & \
    (distances300 < 135) & (cat300.finalCatFrac > thresholds300)
rightFilter = (radiiMean300 > 10) & (radiiMean300 <= 25) & \
    (distances300 < 300) & (cat300.finalCatFrac > thresholds300)

plt.clf()
doCat = True
if doCat:
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
    plot.massFunctionComparison(massMean300[leftFilter],\
        massMean300[rightFilter],4*np.pi*135**3/3,nBins=nBins,\
        labelLeft = "Combined catalogue \n(well-constrained voids only)",\
        labelRight  ="Combined catalogue \n(well-constrained voids only)",\
        ylabel="Number of antihalos",savename=figuresFolder + \
        "mass_function_combined_300vs135_test.pdf",massLower=mLower,\
        ylim=[1,1000],Om0 = 0.3111,h=0.6766,sigma8=0.8128,ns=0.9667,\
        fontsize=8,massUpper = mUpper,\
        titleLeft = "Combined catalogue, $<135\\mathrm{Mpc}h^{-1}$",\
        titleRight = "Combined catalogue, $<300\\mathrm{Mpc}h^{-1}$",\
        volSimRight = 4*np.pi*300**3/3,ylimRight=[1,1000],\
        legendLoc="upper right")



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

mean_radii_mcmc = cat300.getMeanProperty('radii',void_filter=filter300)

plt.clf()
fig, ax = plt.subplots(figsize=(0.45*textwidth,0.45*textwidth))
plot_void_counts_radius(mean_radii_mcmc[0],radius_bins,
                        noConstraintsDict['radii'],ax=ax)

ax.tick_params(axis='both',which='major',labelsize=fontsize)
ax.tick_params(axis='both',which='minor',labelsize=fontsize)
plt.subplots_adjust(left = 0.17,right=0.97,bottom = 0.15,top = 0.97)
plt.savefig(figuresFolder + "void_size_function.pdf")
plt.show()


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

# Function to get the CIC density:
from nbodykit.source.catalog import ArrayCatalog
def getCICDensity(positions,boxsize,mUnit,nres):
    cat = ArrayCatalog({'Position' : positions,\
        'Mass' : mUnit*np.ones(len(positions))},\
        BoxSize=(boxsize,boxsize,boxsize))
    mesh = cat.to_mesh(Nmesh=nres)
    return mesh.preview()

def coord_range_to_ind_range(coord_range,extent,N):
    indices = [int((x - extent[0])*N/(extent[1] - extent[0])) 
               for x in coord_range]
    return indices



density0 = getCICDensity(np.fliplr(snapToShow['pos']),
    boxsize,snapToShow['mass'][0],256)

density1 = getCICDensity(snapToShow['pos'],
    boxsize,snapToShow['mass'][0],256)

density2 = np.reshape(density0,(256,256,256),order='F')

coord_range = [-2.5,2.5]
ind_range = coord_range_to_ind_range(coord_range,[-boxsize/2,boxsize/2],256)
coord_filter = (equatorialXYZ[:,2] > coord_range[0]) & \
    (equatorialXYZ[:,2] <= coord_range[1])
coord_filter_snap = (snapToShow['pos'][:,2] > coord_range[0]) & \
    (snapToShow['pos'][:,2] <= coord_range[1])

positions = snapToShow['pos'][coord_filter_snap,0:-1]

counts = np.histogramdd(positions,
    bins = [np.linspace(-boxsize/2,boxsize/2,256+1)]*2)

fig, ax = plt.subplots()
den_field = counts[0]/np.mean(counts[0])
#den_field = np.mean(
#    density1[:,:,np.arange(ind_range[0],ind_range[1]+1)],2)
im = ax.imshow(den_field.T,norm=colors.LogNorm(vmin=1e-3,vmax=1e3),
        cmap="PuOr_r",extent=(-boxsize/2,boxsize/2,-boxsize/2,boxsize/2),
        origin="lower")
#ax.scatter(positions[:,0],positions[:,1],marker = '.',color='r')
ax.scatter(equatorialXYZ[coord_filter,0],equatorialXYZ[coord_filter,1],
           marker = '.',color='r')
ax.set_title("Simulation Density")
ax.set_xlabel("x [$\\mathrm{Mpc}h^{-1}$]")
ax.set_ylabel("y [$\\mathrm{Mpc}h^{-1}$]")
ax.set_xlim([-100,100])
ax.set_ylim([-100,100])
plt.show()


# Create the histogram with the correct bin edges
num_bins= 256
x_bins = np.linspace(-boxsize/2, boxsize/2, num_bins + 1)
y_bins = np.linspace(-boxsize/2, boxsize/2, num_bins + 1)
counts, _, _ = np.histogram2d(positions[:,0], positions[:,1], bins=[x_bins, y_bins])

fig, ax = plt.subplots()
den_field = counts / np.mean(counts)
im = ax.imshow(den_field.T, 
    norm=colors.LogNorm(vmin=1e-3, vmax=1e3), cmap="PuOr_r",
    extent=(-boxsize/2, boxsize/2, -boxsize/2, boxsize/2),origin="lower")
ax.scatter(positions[:,0], positions[:,1], marker='.', color='r')
ax.set_title("Simulation Density")
ax.set_xlabel("x [$\\mathrm{Mpc}h^{-1}$]")
ax.set_ylabel("y [$\\mathrm{Mpc}h^{-1}$]")
ax.set_xlim([-50, 50])
ax.set_ylim([-50, 50])
plt.show()



#den_field = np.mean(
#    density0[:,:,np.arange(ind_range[0],ind_range[1]+1)],2)
#ax.scatter(equatorialXYZ[coord_filter,0],equatorialXYZ[coord_filter,1],
#           marker = '.',color='r')


# Alpha shapes

[ahMWPos,alpha_shapes_finalCat,alpha_shapes_individual] = \
    tools.loadOrRecompute(
        data_folder + "cat300_alpha_shapes.p",cat300.get_alpha_shapes,
        snapList,snapListRev,antihaloCatalogueList=hrList,ahProps = ahProps,
        snapsortList=snapSortList,void_filter=filter300,recentreSnaps=False,
        _recomputeData = False)



# Plots:



sortedRadiiOpt = np.flip(np.argsort(
    cat300.getMeanProperty('radii',void_filter=filter300)[0]))
catToUse = cat300.get_final_catalogue(void_filter=filter300)
cat_object = cat300
haveVoids = [np.where(catToUse[:,ns] > 0)[0] \
    for ns in range(0,len(snapNameList))]

#for ns in range(0,len(snapNameList)):
#    catToUse[haveVoids[ns],ns] = np.arange(1,len(haveVoids[ns])+1)



filterToUse = np.where(filter300)[0]
nVoidsToShow = 10
#nVoidsToShow = len(filterToUse)
#selection = np.intersect1d(sortedRadiiOpt,filterToUse)[:(nVoidsToShow)]
selection = sortedRadiiOpt[np.arange(0,nVoidsToShow)]
asListAll = []
colourListAll = []
laListAll = []
labelListAll = []

plotFormat='.pdf'
#plotFormat='.pdf'

textwidth=7.1014
textheight=9.0971
scale = 1.26
width = textwidth
height = 0.6*textwidth
cropPoint = ((scale -1)/2)*np.array([width,height]) + np.array([0,-0.1])
bound_box = transforms.Bbox([[cropPoint[0], cropPoint[1]],
    [cropPoint[0] + width, cropPoint[1] + height]])

if doSky:
    #
    #plot.plotLocalUniverseMollweide(rCut,snapToShow,\
    #    alpha_shapes = alpha_shape_list[ns][1],
    #    largeAntihalos = largeAntihalos[ns],hr=antihaloCatalogueList[ns],
    #    coordAbell = coordCombinedAbellSphere,\
    #    abellListLocation = clusterIndMain,\
    #    nameListLargeClusters = [name[0] for name in clusterNames],\
    #    ha = ha,va= va, annotationPos = annotationPos,\
    #    title = 'Local super-volume: large voids (antihalos) within $' + \
    #    str(rCut) + "\\mathrm{\\,Mpc}h^{-1}$",
    #    vmin=1e-2,vmax=1e2,legLoc = 'lower left',bbox_to_anchor = (-0.1,-0.2),
    #    snapsort = snapsortList[ns],antihaloCentres = None,
    #    figOut = figuresFolder + "/antihalos_sky_plot.pdf",
    #    showFig=True,figsize = (scale*textwidth,scale*0.55*textwidth),
    #    voidColour = seabornColormap[0],antiHaloLabel='inPlot',
    #    bbox_inches = bound_box,galaxyAngles=equatorialRThetaPhi[:,1:],\
    #    galaxyDistances = equatorialRThetaPhi[:,0],showGalaxies=False)
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
        title = 'Local super-volume: large voids (antihalos) within $' + \
        str(rCut) + "\\mathrm{\\,Mpc}h^{-1}$",\
        vmin=1e-2,vmax=1e2,legLoc = 'lower left',\
        bbox_to_anchor = (-0.1,-0.2),\
        snapsort = snapSortList[ns],antihaloCentres = None,\
        figOut = figuresFolder + "/ah_match_sample_" + \
        str(ns) + plotFormat,\
        showFig=False,figsize = (scale*textwidth,scale*0.55*textwidth),\
        voidColour = colourListAll[ns],antiHaloLabel=labelListAll[ns],\
        bbox_inches = bound_box,galaxyAngles=equatorialRThetaPhi[:,1:],\
        galaxyDistances = equatorialRThetaPhi[:,0],showGalaxies=False,\
        voidAlpha = 0.6,margins=None,hpxMap = hpx_map_list[ns],pad=0.0,
        cbar_aspect=30)
    plt.show()

# Combined outlines:
ns = 0
plot.plotLocalUniverseMollweide(rCut,snapList[ns],\
    alpha_shapes = asListAll[ns],\
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
    hpxMap = hpx_map_list[ns])


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




