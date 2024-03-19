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

fontsize = 9
legendFontsize = 9

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
antihaloCentres = [tools.remapAntiHaloCentre(props[5],boxsize,
                                             swapXZ  = False,reverse = True) 
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
# ELLIPTICITY FUNCTIONS:

# Compute the inertia tensor:
def get_inertia(points,centre,weights=None):
    if weights is None:
        weights = np.ones(len(points))
    disp = points - centre
    other_inds = [np.setdiff1d(range(0,3),[i]) for i in range(0,3)]
    M_diag = [np.sum(weights[:,None]*disp[:,inds]**2) for inds in other_inds]
    M_off_diag = [-np.sum(weights*disp[:,inds[0]]*disp[:,inds[1]]) 
                  for inds in other_inds]
    M = np.diag(M_diag)
    for inds, Mij in zip(other_inds,M_off_diag):
        M[inds[0],inds[1]] = Mij
        M[inds[1],inds[0]] = Mij
    return M

# Get the ellipticity of a set of points:
def get_ellipticity(points,centre=None,weights=None):
    if centre is None:
        centre = np.mean(points,0)
    # Get inertia tensor:
    M = get_inertia(points,centre,weights=None)
    # Get eigenvalues:
    eigenvalues, eigenvectors = np.linalg.eig(M)
    # Compute ellipticity:
    J1 = np.min(eigenvalues)
    J3 = np.max(eigenvalues)
    return 1.0 - (J1/J3)**(0.25)


#-------------------------------------------------------------------------------
# ELLIPTICITY DISTRIBUTION

# Get long lists of all ellipticities:
def get_all_ellipticities(snapList,snapListRev,boxsize,antihaloCentres):
    ellipticity_list = []
    for ns in range(0,len(snapList)):
        # Load snapshot (don't use the snapshot list, because that will force
        # loading of all snapshot positions, using up a lot of memory, when 
        # we only want to load these one at a time):
        print("Doing sample " + str(ns+1) + " of " + str(len(snapList)))
        snap = pynbody.load(snapList[ns].filename)
        snap_reverse = pynbody.load(snapListRev[ns].filename)
        hr_list = snap_reverse.halos()
        # Sorted indices, to allow correct referencing of particles:
        sorted_indices = np.argsort(snap['iord'])
        reverse_indices = snap_reverse['iord'] # Force loading of reverse 
        # snapshot indices
        print("Sorting complete")
        # Remap positions into correct equatorial co-ordinates:
        positions = tools.remapAntiHaloCentre(snap['pos'],boxsize,
                                              swapXZ  = False,reverse = True)
        print("Positions computed")
        # Get relative positions of particles in each halo, remembering to 
        # account for wrapping:
        print("Computing ellipticities...")
        ellipticities = np.zeros(len(antihaloCentres[ns]))
        for k in tools.progressbar(range(0,len(antihaloCentres[ns]))):
            indices = hr_list[k+1]['iord']
            halo_pos = snapedit.unwrap(positions[sorted_indices[indices],:] - 
                                       antihaloCentres[ns][k],boxsize)
            ellipticities[k] = get_ellipticity(halo_pos,np.array([0]*3))
        ellipticity_list.append(ellipticities)
    return ellipticity_list

# Compute or load ellipticity data:
ellipticity_list = tools.loadOrRecompute(data_folder + "ellipticities.p",
                                         get_all_ellipticities,
                                         snapList,snapListRev,boxsize,
                                         antihaloCentres)

# Lambda-CDM reference:
antihaloCentresUn = [tools.remapAntiHaloCentre(props[5],boxsize,
                                               swapXZ  = False,reverse = True) \
                     for props in ahPropsUn]

ellipticity_list_lcdm = tools.loadOrRecompute(
    data_folder + "ellipticities_lcdm.p",get_all_ellipticities,snapListUn,
    snapListRevUn,boxsize,antihaloCentresUn)

rad_filters = [(rad > 10) & (rad <= 20) for rad in antihaloRadiiUn]

combined_ellipticities = np.hstack([epsilon[filt] 
    for epsilon, filt in zip(ellipticity_list_lcdm, rad_filters)])

# Pure catalogue elipticities only:
ellipticities_shortened = cat300.getShortenedQuantity(ellipticity_list,
    cat300.centralAntihalos)
[eps_mean,eps_std] = cat300.getMeanProperty(ellipticities_shortened,
    void_filter=True)


# Distribution of ellipticities:
plt.clf()
eps_bins = np.linspace(0,0.4,11)

[probLCDM,sigmaLCDM,noInBinsLCDM,inBinsLCDM] = plot.computeHistogram(
    combined_ellipticities,eps_bins,density=True,useGaussianError=True)
[probCat,sigmaCat,noInBinsCat,inBinsCat] = plot.computeHistogram(
    eps_mean,eps_bins,density=True,useGaussianError=True)

fig, ax = plt.subplots()
plt.hist(combined_ellipticities,bins=eps_bins,alpha=0.5,label="$\\Lambda$-CDM",
    density=True,color=seabornColormap[0])
#plt.hist(eps_mean,bins=eps_bins,alpha=0.5,label="Combined Catalogue",
#    density=True,color=seabornColormap[1])

#plot.histWithErrors(probLCDM,sigmaLCDM,eps_bins,ax=ax,color=seabornColormap[0],
#    label="$\\Lambda$-CDM")
plot.histWithErrors(probCat,sigmaCat,eps_bins,ax=ax,color=seabornColormap[1],
    label="Combined Catalogue")

plt.xlabel('Ellipticity')
plt.ylabel('Probability Density')
plt.legend(frameon=False)
plt.savefig(figuresFolder + "ellipticity_distribution.pdf")
plt.show()


#-------------------------------------------------------------------------------
# STACKED VOIDS

# Stack all voids relative to their barycentre:

# Get points within an ellipse of radius R, and ellipticity eps, given a z
# position and displacement from the axis

def verify_los_pos(los_pos):
    if len(los_pos.shape) != 2:
        raise Exception("los_pos must have two dimensions")
    if los_pos.shape[1] != 2:
        raise Exception("los_pos shape must have size 2")



def get_points_in_ellipse(los_pos,eps,R):
    verify_los_pos(los_pos)
    z = los_pos[:,0]
    d = los_pos[:,1]
    return (d <= R/eps) & (z**2 <= R**2 - eps**2*d**2)

def get_los_pos(pos,los,boxsize):
    los_unit = los/np.sqrt(np.sum(los**2))
    pos_rel = snapedit.unwrap(pos - los,boxsize)
    z = np.dot(pos_rel,los_unit)
    d = np.sqrt(np.sum(pos_rel**2,1) - z**2)
    return np.vstack((z,d)).T

def get_los_ellipticity(los_pos,weights=None):
    verify_los_pos(los_pos)
    z = los_pos[:,0]
    d = los_pos[:,1]
    if weights is None:
        weights = np.ones(los_pos.shape[0])
    return np.sqrt(2*np.sum(z**2*weights)/np.sum(d**2*weights))

def get_los_ellipticity_in_ellipse(los_pos,eps,R,weights=None):
    verify_los_pos(los_pos)
    if weights is None:
        weights = np.ones(los_pos.shape[0])
    filt = get_points_in_ellipse(los_pos,eps,R)
    return get_los_ellipticity(los_pos[filt],weights=weights[filt])

# Check ellipticity calculation:
ns = 0
k = 0
snap = pynbody.load(snapList[ns].filename)
snap_reverse = pynbody.load(snapListRev[ns].filename)
hr_list = snap_reverse.halos()
indices = hr_list[k+1]['iord']
positions = tools.remapAntiHaloCentre(snap['pos'],boxsize,swapXZ  = False,
    reverse = True)
pos = snapedit.unwrap(positions[sorted_indices[indices],:],boxsize)
los = antihaloCentres[ns][0]


los_pos = get_los_pos(pos,los,boxsize)

eps_list = np.linspace(0,2,101)

eps_calc = np.array([get_los_ellipticity_in_ellipse(los_pos,eps,10) 
    for eps in eps_list])

eps_calc = np.array([get_los_ellipticity_in_ellipse(los_pos,eps,20,
    weights=1.0/(2.0*np.pi*los_pos[:,1])) 
    for eps in eps_list])

plt.clf()
plt.plot(eps_list,eps_calc,linestyle='-',color=seabornColormap[0],
    label="Calculated ellipticity")
plt.plot(eps_list,eps_list,linestyle='--',color=seabornColormap[0],
    label="Equal ellipticities")
plt.xlabel('Cut ellipticity')
plt.ylabel('Calculated Ellipticity')
plt.legend(frameon=False)
plt.savefig(figuresFolder + "ellipticity_test.pdf")
plt.show()


def estimate_ellipticity(los_pos,limits=[1e-5,2],npoints=101,R=10,weights=None):
    eps_list = np.linspace(limits[0],limits[1],npoints)
    eps_calc = np.array([get_los_ellipticity_in_ellipse(los_pos,eps,R,
        weights=weights) for eps in eps_list])
    # Look for changes of sign:
    diff = eps_calc - eps_list
    signs = diff[1:]*diff[0:-1]
    sign_change = np.where(signs < 0)[0]
    if len(sign_change) == 0:
        # No changes of sign!
        return eps_list[np.argmin(diff**2)]
    else:
        # Get first change of sign:
        lower_bound = eps_list[sign_change[0]]
        upper_bound = eps_list[sign_change[0]+1]
        return (lower_bound + upper_bound)/2

def solve_ellipticity(los_pos,limits=[1e-5,2],R=10,weights=None,guess=1.0):
    func = lambda x: get_los_ellipticity_in_ellipse(los_pos,x,R,
        weights=weights) - x
    return scipy.optimize.fsolve(func,guess)


# Scatter test:
R=20
#eps_est = estimate_ellipticity(los_pos,R=R)
eps_est = solve_ellipticity(los_pos,R=R)
drange = np.linspace(0,(R/eps_est)*1.1,101)

plt.clf()
plt.scatter(los_pos[:,1],los_pos[:,0],marker='.')
plt.plot(drange,np.sqrt(R**2 - eps_est**2*drange**2),linestyle='--',color='k',
    label="Ellipse, $R =  " + ("%.2g" % R) + ", \\epsilon = " + 
    ("%.2g" % eps_est) + "$")
plt.plot(drange,-np.sqrt(R**2 - eps_est**2*drange**2),linestyle='--',color='k')
plt.xlabel('d [$\\mathrm{Mpc}h^{-1}$]')
plt.ylabel('z [$\\mathrm{Mpc}h^{-1}$]')
plt.legend(frameon=False)
plt.savefig(figuresFolder + "ellipticity_scatter.pdf")
plt.show()

# Get and save the line of sight positions:


def get_los_pos_for_snapshot(snapname_forward,snapname_reverse,centres,radii,
        dist_max=20,rmin=10,rmax=20,all_particles=True,filter_list=None,
        void_indices=None):
    snap = pynbody.load(snapname_forward)
    snap_reverse = pynbody.load(snapname_reverse)
    hr_list = snap_reverse.halos()
    boxsize = snap.properties['boxsize'].ratio("Mpc a h**-1")
    # Sorted indices, to allow correct referencing of particles:
    sorted_indices = np.argsort(snap['iord'])
    reverse_indices = snap_reverse['iord'] # Force loading of reverse 
    # snapshot indices
    print("Sorting complete")
    # Remap positions into correct equatorial co-ordinates:
    positions = tools.remapAntiHaloCentre(snap['pos'],boxsize,
                                          swapXZ  = False,reverse = True)
    print("Positions computed")
    # Get relative positions of particles in each halo, remembering to 
    # account for wrapping:
    if all_particles:
        print("Generating Tree")
        tree = scipy.spatial.cKDTree(snapedit.wrap(positions,boxsize),
            boxsize=boxsize)
    print("Computing ellipticities...")
    if void_indices is None:
        void_indices = np.range(0,len(hr_list))
    los_pos_all = []
    if filter_list is None:
        rad_filter = (radii > rmin) & (radii <= rmax)
    else:
        rad_filter = filter_list & (radii > rmin) & (radii <= rmax)
    for k in tools.progressbar(range(0,len(centres))):
        if rad_filter[k]:
            if not all_particles:
                indices = hr_list[void_indices[k]+1]['iord']
                halo_pos = snapedit.unwrap(positions[sorted_indices[indices],:],
                    boxsize)
                distances = np.sqrt(
                    np.sum(snapedit.unwrap(halo_pos - centres[k],boxsize)**2,1))
                halo_pos = halo_pos[distances < dist_max,:]
            else:
                indices = tree.query_ball_point(
                    snapedit.wrap(centres[k],boxsize),dist_max,workers=-1)
                halo_pos = snapedit.unwrap(positions[indices,:],boxsize)
            los_pos = get_los_pos(halo_pos,centres[k],boxsize)
            los_pos_all.append(los_pos)
        else:
            los_pos_all.append(np.zeros((0,2)))
    return los_pos_all

def get_los_positions_for_all_catalogues(snapList,snapListRev,
        antihaloCentres,antihaloRadii,recompute=False,filter_list=None,
        suffix=".lospos.p",void_indices=None,**kwargs):
    los_list = []
    if suffix == "":
        raise Exception("Suffix cannot be empty.")
    if void_indices is None:
        void_indices = [None for ns in range(0,len(snapList))]
    for ns in range(0,len(snapList)):
        # Load snapshot (don't use the snapshot list, because that will force
        # loading of all snapshot positions, using up a lot of memory, when 
        # we only want to load these one at a time):
        print("Doing sample " + str(ns+1) + " of " + str(len(snapList)))
        if filter_list is None:
            los_pos_all = tools.loadOrRecompute(
                snapList[ns].filename + suffix,
                get_los_pos_for_snapshot,snapList[ns].filename,
                snapListRev[ns].filename,antihaloCentres[ns],antihaloRadii[ns],
                _recomputeData=recompute,void_indices=void_indices[ns],**kwargs)
        else:
            los_pos_all = tools.loadOrRecompute(
                snapList[ns].filename + suffix,
                get_los_pos_for_snapshot,snapList[ns].filename,
                snapListRev[ns].filename,antihaloCentres[ns],
                antihaloRadii[ns],filter_list=filter_list[ns],
                _recomputeData=recompute,void_indices=void_indices[ns],**kwargs)
        los_list.append(los_pos_all)
    return los_list


#los_list_borg = get_los_positions_for_all_catalogues(snapList,snapListRev,
#    antihaloCentres,antihaloRadii,dist_max=60,rmin=10,rmax=20,recompute=False)

#los_list_lcdm = get_los_positions_for_all_catalogues(snapListUn,
#    snapListRevUn,antihaloCentresUn,antihaloRadiiUn,dist_max=60,rmin=10,
#    rmax=20,recompute=False)

los_list_borg = get_los_positions_for_all_catalogues(snapList,snapListRev,
    [cat300.getMeanCentres(void_filter=True) for ns in range(0,len(snapList))],
    [cat300.getMeanProperty("radii",void_filter=True)[0] 
    for ns in range(0,len(snapList))],
    dist_max=60,rmin=10,rmax=20,recompute=False)

final_cat = cat300.get_final_catalogue(void_filter=True)
halo_indices = [-np.ones(len(final_cat),dtype=int) 
    for ns in range(0,len(snapList))]
for ns in range(0,len(snapList)):
    have_void = final_cat[:,ns] >= 0
    halo_indices[ns][have_void] = \
        cat300.indexListShort[ns][final_cat[have_void,ns]-1]

filter_list_borg = [halo_indices[ns] >= 0 for ns in range(0,len(snapList))]
los_list_void_only_borg = get_los_positions_for_all_catalogues(snapList,
    snapListRev,
    [cat300.getMeanCentres(void_filter=True) for ns in range(0,len(snapList))],
    [cat300.getMeanProperty("radii",void_filter=True)[0] 
    for ns in range(0,len(snapList))],all_particles=False,
    void_indices = halo_indices,filter_list=filter_list_borg,
    dist_max=60,rmin=10,rmax=20,recompute=False,suffix=".lospos_void_only.p")

# LCDM examples for comparison:
distances_from_centre_lcdm = [
    np.sqrt(np.sum(snapedit.unwrap(centres - np.array([boxsize/2]*3),
    boxsize)**2,1)) for centres in antihaloCentresUn]
filter_list_lcdm = [(dist < 135) & (radii > 10) & (radii <= 20) 
    for dist, radii in zip(distances_from_centre_lcdm,antihaloRadiiUn)]

los_list_lcdm = get_los_positions_for_all_catalogues(snapListUn,snapListRevUn,
    antihaloCentresUn,antihaloRadiiUn,filter_list=filter_list_lcdm,
    dist_max=60,rmin=10,rmax=20,recompute=False)

los_list_void_only_lcdm = get_los_positions_for_all_catalogues(snapListUn,
    snapListRevUn,antihaloCentresUn,antihaloRadiiUn,all_particles=False,
    filter_list=filter_list_lcdm,dist_max=60,rmin=10,rmax=20,recompute=False,
    suffix=".lospos_void_only.p")

# Scatter of LCDM stacks:

# Test case:
#los_pos_all = los_list_lcdm[0]
#los_pos_all = tools.loadOrRecompute(snapListUn[ns].filename + ".lospos.p",
#    get_los_pos_for_snapshot,snapListUn[0].filename,snapListRevUn[0].filename,
#    antihaloCentresUn[ns],antihaloRadiiUn[ns],_recomputeData=True,dist_max=60,
#    rmin=10,rmax=20)

los_pos_all = los_list_lcdm[0]

stacked_particles_lcdm = np.vstack(los_pos_all)
stacked_particles_lcdm_abs = np.abs(stacked_particles_lcdm)
R=20
eps_est = solve_ellipticity(stacked_particles_lcdm,R=R,
    weights=1.0/(2*np.pi*stacked_particles_lcdm[:,1]))
drange = np.linspace(0,(R/eps_est)*1.1,101)

plt.clf()
fig, ax = plt.subplots()
ax.hist2d(stacked_particles_lcdm_abs[:,1],
           stacked_particles_lcdm_abs[:,0],
           bins=[np.linspace(0,60,31),np.linspace(0,60,31)],density=True,
           cmap="Blues",weights=(1.0/(2*np.pi*stacked_particles_lcdm_abs[:,1])))

#ax.plot(drange,np.sqrt(R**2 - eps_est**2*drange**2),linestyle='--',color='k',
#    label="Ellipse, $R =  " + ("%.2g" % R) + ", \\epsilon = " + 
#    ("%.2g" % eps_est) + "$")
#ax.plot(drange,-np.sqrt(R**2 - eps_est**2*drange**2),linestyle='--',color='k')
ax.set_xlabel('d [$\\mathrm{Mpc}h^{-1}$]')
ax.set_ylabel('z [$\\mathrm{Mpc}h^{-1}$]')
ax.set_xlim([0,60])
ax.set_ylim([0,60])
ax.set_aspect('equal')
ax.legend(frameon=False)
plt.savefig(figuresFolder + "ellipticity_scatter_all_lcdm.pdf")
plt.show()

stacked_particles_borg = np.vstack([np.vstack(los_list) 
    for los_list in los_list_borg])
stacked_particles_borg_abs = np.abs(stacked_particles_borg)

stacked_particles_lcdm = np.vstack([np.vstack(los_list) 
    for los_list in los_list_lcdm ])
stacked_particles_lcdm_abs = np.abs(stacked_particles_lcdm)

def draw_ellipse(ax,R,eps,theta_bounds=[np.pi/2,0],npoints=101,color='k',
        linestyle=':',label=None):
    theta_vals = np.linspace(theta_bounds[0],theta_bounds[1],npoints)
    d = R/np.sqrt(np.tan(theta_vals)**2 + eps**2)
    z = d*np.tan(theta_vals)
    ax.plot(d,z,color=color,linestyle=linestyle,label=label)

# Plot comparing LCDM with out voids:
upper_dist = 20
bins_z = np.linspace(0,upper_dist,21)
bins_d = np.linspace(0,upper_dist,21)
cell_volumes = np.outer(np.diff(bins_z),np.diff(bins_d))
hist_lcdm = np.histogramdd(stacked_particles_lcdm_abs,bins=[bins_z,bins_d],
                           density=False,weights = 1.0/\
                           (2*np.pi*stacked_particles_lcdm_abs[:,1]))
count_lcdm = len(stacked_particles_lcdm_abs)
num_voids_lcdm = np.sum([len(x) for x in los_list_lcdm])

hist_borg = np.histogramdd(stacked_particles_borg_abs,bins=[bins_z,bins_d],
                           density=False,weights = 1.0/\
                           (2*np.pi*stacked_particles_borg_abs[:,1]))
count_borg = len(stacked_particles_borg_abs)
num_voids_borg = np.sum([len(x) for x in los_list_borg]) # Not the actual number
    # but the effective number being stacked, so the number of voids multiplied
    # by the number of samples.


nmean = len(snapList[0])/(boxsize**3)

plt.clf()
fig, ax = plt.subplots(1,2,figsize=(textwidth,0.45*textwidth))
#ax[0].hist2d(stacked_particles_lcdm_abs[:,1],
#           stacked_particles_lcdm_abs[:,0],
#           bins=[np.linspace(0,60,31),np.linspace(0,60,31)],density=True,
#           cmap="Blues",
#           weights=(1.0/(2*np.pi*stacked_particles_lcdm_abs[:,1])))

#im = ax[1].hist2d(stacked_particles_borg_abs[:,1],
#           stacked_particles_borg_abs[:,0],
#           bins=[np.linspace(0,60,31),np.linspace(0,60,31)],density=True,
#           cmap="Blues",weights=
#           (1.0/((len(snapList)*2*np.pi*stacked_particles_borg_abs[:,1]))))

im1 = ax[0].imshow(hist_lcdm[0]/(2*cell_volumes*num_voids_lcdm*nmean),
                   cmap='PuOr_r',vmin=0,vmax = 2,
                   extent=(0,upper_dist,0,upper_dist),origin='lower')
im2 = ax[1].imshow(hist_borg[0]/(2*cell_volumes*num_voids_borg*nmean),
                   cmap='PuOr_r',vmin=0,vmax = 2,
                   extent=(0,upper_dist,0,upper_dist),origin='lower')

#ax.plot(drange,np.sqrt(R**2 - eps_est**2*drange**2),linestyle='--',color='k',
#    label="Ellipse, $R =  " + ("%.2g" % R) + ", \\epsilon = " + 
#    ("%.2g" % eps_est) + "$")
#ax.plot(drange,-np.sqrt(R**2 - eps_est**2*drange**2),linestyle='--',color='k')

#Rvals = [10,20,30,40,50,60]
Rvals = [5,10,15,20]
titles = ['$\\Lambda$-CDM Simulations','BORG Catalogue']
for axi, title in zip(ax,titles):
    axi.set_xlabel('d (Perpendicular distance) [$\\mathrm{Mpc}h^{-1}$]',
                   fontsize=fontsize,fontfamily=fontfamily)
    axi.set_ylabel('z (LOS distance)[$\\mathrm{Mpc}h^{-1}$]',
                   fontsize=fontsize,fontfamily=fontfamily)
    for r in Rvals:
        draw_ellipse(axi,r,1.0)
    axi.set_xlim([0,upper_dist])
    axi.set_ylim([0,upper_dist])
    axi.set_aspect('equal')
    axi.set_title(title,fontsize=fontsize,fontfamily=fontfamily)
    #axi.legend(frameon=False)

# Remove y labels on axis 2:
ax[1].yaxis.label.set_visible(False)
ax[1].yaxis.set_major_formatter(NullFormatter())
ax[1].yaxis.set_minor_formatter(NullFormatter())
ax[0].xaxis.get_major_ticks()[4].set_visible(False)

fig.colorbar(im1, ax=ax.ravel().tolist(),shrink=0.9,
    label='(Tracer density)/(Mean Density)')
plt.subplots_adjust(wspace=0.0,hspace=0.0,left=0.1,right=0.75,bottom=0.15,
                    top=0.95)
plt.savefig(figuresFolder + "ellipticity_scatter_comparison.pdf")
plt.show()



#-------------------------------------------------------------------------------

# With all particles:
#los_lcdm = los_list_lcdm
#los_borg = los_list_borg
#filename = "ellipticity_scatter_comparison_reff.pdf"
#density_unit = "relative"
# Void particles only:
los_lcdm = los_list_void_only_lcdm
los_borg = los_list_void_only_borg
filename = "ellipticity_scatter_comparison_reff_void_only.pdf"
density_unit = "probability"

voids_used_lcdm = [np.array([len(x) for x in los]) > 0 for los in los_lcdm]
voids_used_borg = [np.array([len(x) for x in los]) > 0 for los in los_borg]
# Filter out any unused voids as they just cause problems:
los_lcdm = [ [x for x in los if len(x) > 0] for los in los_lcdm]
los_borg = [ [x for x in los if len(x) > 0] for los in los_borg]

# Stacked Profile with rescaled voids:
upper_dist_reff = 2
bins_z_reff = np.linspace(0,upper_dist_reff,41)
bins_d_reff = np.linspace(0,upper_dist_reff,41)
bin_z_centres = plot.binCentres(bins_z_reff)
bin_d_centres = plot.binCentres(bins_d_reff)
cell_volumes_reff = np.outer(np.diff(bins_z_reff),np.diff(bins_d_reff))

# Get void effective radii:
void_radii_lcdm = [rad[filt] 
                   for rad, filt in zip(antihaloRadiiUn,voids_used_lcdm)]
void_radii_borg = cat300.getMeanProperty("radii",void_filter=True)[0]
# Express los co-ords in units of reff:
los_list_reff_lcdm = [
    [los/rad for los, rad in zip(all_los,all_radii)] 
    for all_los, all_radii in zip(los_lcdm,void_radii_lcdm)]
los_list_reff_borg = [
    [los/rad for los, rad in zip(all_los,void_radii_borg)] 
    for all_los in los_borg]

# Stack all the particles:
stacked_particles_lcdm_reff = np.vstack([np.vstack(los_list) 
    for los_list in los_list_reff_lcdm ])
stacked_particles_borg_reff = np.vstack([np.vstack(los_list) 
    for los_list in los_list_reff_borg ])
stacked_particles_reff_lcdm_abs = np.abs(stacked_particles_lcdm_reff)
stacked_particles_reff_borg_abs = np.abs(stacked_particles_borg_reff)

# Volume weights for each particle:
v_weight_lcdm = [
    [rad**3*np.ones(len(los)) for los, rad in zip(all_los,all_radii)] 
    for all_los, all_radii in zip(los_lcdm,void_radii_lcdm)]
v_weight_lcdm = np.hstack([np.hstack(rad) for rad in v_weight_lcdm])
v_weight_borg = [
    [rad**3*np.ones(len(los)) for los, rad in zip(all_los,void_radii_borg)] 
    for all_los in los_borg]
v_weight_borg = np.hstack([np.hstack(rad) for rad in v_weight_borg])


# Histograms to get the density:
hist_lcdm_reff = np.histogramdd(stacked_particles_reff_lcdm_abs,
                           bins=[bins_z_reff,bins_d_reff],
                           density=False,weights = 1.0/\
                           (2*np.pi*v_weight_lcdm*
                           stacked_particles_reff_lcdm_abs[:,1]))
count_lcdm = len(stacked_particles_reff_lcdm_abs)
num_voids_lcdm = np.sum([np.sum(x) for x in voids_used_lcdm])

hist_borg_reff = np.histogramdd(stacked_particles_reff_borg_abs,
                           bins=[bins_z_reff,bins_d_reff],
                           density=False,weights = 1.0/\
                           (2*v_weight_borg*np.pi*
                           stacked_particles_reff_borg_abs[:,1]))
count_borg = len(stacked_particles_reff_borg_abs)
num_voids_borg = np.sum([np.sum(x) for x in voids_used_borg]) # Not the actual number
    # but the effective number being stacked, so the number of voids multiplied
    # by the number of samples.
nmean = len(snapList[0])/(boxsize**3)


plt.clf()
fig, ax = plt.subplots(1,2,figsize=(textwidth,0.45*textwidth))
#ax[0].hist2d(stacked_particles_lcdm_abs[:,1],
#           stacked_particles_lcdm_abs[:,0],
#           bins=[np.linspace(0,60,31),np.linspace(0,60,31)],density=True,
#           cmap="Blues",
#           weights=(1.0/(2*np.pi*stacked_particles_lcdm_abs[:,1])))

#im = ax[1].hist2d(stacked_particles_borg_abs[:,1],
#           stacked_particles_borg_abs[:,0],
#           bins=[np.linspace(0,60,31),np.linspace(0,60,31)],density=True,
#           cmap="Blues",weights=
#           (1.0/((len(snapList)*2*np.pi*stacked_particles_borg_abs[:,1]))))
if density_unit == "relative":
    field_lcdm = hist_lcdm_reff[0]/(2*cell_volumes_reff*num_voids_lcdm*nmean)
    field_borg = hist_borg_reff[0]/(2*cell_volumes_reff*num_voids_borg*nmean)
    im1 = ax[0].imshow(
        field_lcdm,
        cmap='PuOr_r',vmin=0,vmax = 2,
        extent=(0,upper_dist_reff,0,upper_dist_reff),origin='lower')
    im2 = ax[1].imshow(
        field_borg,
        cmap='PuOr_r',vmin=0,vmax = 2,
        extent=(0,upper_dist_reff,0,upper_dist_reff),origin='lower')
elif density_unit == "absolute":
    field_lcdm = hist_lcdm_reff[0]/(2*cell_volumes_reff*num_voids_lcdm)
    field_borg = hist_borg_reff[0]/(2*cell_volumes_reff*num_voids_borg)
    im1 = ax[0].imshow(field_lcdm,
                       cmap='Blues',vmin=0,vmax = None,
                       extent=(0,upper_dist_reff,0,upper_dist_reff),
                       origin='lower')
    im2 = ax[1].imshow(field_borg,
                       cmap='Blues',vmin=0,vmax = None,
                       extent=(0,upper_dist_reff,0,upper_dist_reff),
                       origin='lower')
elif density_unit == "probability":
    field_lcdm = hist_lcdm_reff[0]/(2*count_lcdm*cell_volumes_reff)
    field_borg = hist_borg_reff[0]/(2*count_borg*cell_volumes_reff)
    im1 = ax[0].imshow(field_lcdm,cmap='Blues',vmin=0,vmax = 1e-4,
                       extent=(0,upper_dist_reff,0,upper_dist_reff),
                       origin='lower')
    im2 = ax[1].imshow(field_borg,cmap='Blues',vmin=0,vmax = 1e-4,
                       extent=(0,upper_dist_reff,0,upper_dist_reff),
                       origin='lower')
else:
    raise Exception("Unknown density_unit")

contours = True
countour_list = [1e-6,1e-5,2.5e-5,5e-5,6e-5,7e-5,8e-5,9e-5,1e-4,2e-4]
if contours:
    CS = ax[0].contour(bin_d_centres,bin_z_centres,field_lcdm,
        levels=countour_list)
    ax[0].clabel(CS, inline=True, fontsize=10)
    CS = ax[1].contour(bin_d_centres,bin_z_centres,field_borg,
        levels=countour_list)
    ax[1].clabel(CS, inline=True, fontsize=10)
#ax.plot(drange,np.sqrt(R**2 - eps_est**2*drange**2),linestyle='--',color='k',
#    label="Ellipse, $R =  " + ("%.2g" % R) + ", \\epsilon = " + 
#    ("%.2g" % eps_est) + "$")
#ax.plot(drange,-np.sqrt(R**2 - eps_est**2*drange**2),linestyle='--',color='k')

#Rvals = [10,20,30,40,50,60]
Rvals = [1,2]
titles = ['$\\Lambda$-CDM Simulations','BORG Catalogue']
for axi, title in zip(ax,titles):
    axi.set_xlabel('$d/R_{\\mathrm{eff}}$ (Perpendicular distance)',
                   fontsize=fontsize,fontfamily=fontfamily)
    axi.set_ylabel('$z/R_{\\mathrm{eff}}$ (LOS distance)',
                   fontsize=fontsize,fontfamily=fontfamily)
    for r in Rvals:
        draw_ellipse(axi,r,1.0)
    axi.set_xlim([0,upper_dist_reff])
    axi.set_ylim([0,upper_dist_reff])
    axi.set_aspect('equal')
    axi.set_title(title,fontsize=fontsize,fontfamily=fontfamily)
    #axi.legend(frameon=False)

# Remove y labels on axis 2:
ax[1].yaxis.label.set_visible(False)
ax[1].yaxis.set_major_formatter(NullFormatter())
ax[1].yaxis.set_minor_formatter(NullFormatter())
ax[0].xaxis.get_major_ticks()[4].set_visible(False)

if density_unit == "absolute":
    colorbar_title = "Average Tracer density [$h^{3}\\mathrm{MPc}^{-3}$]"
elif density_unit == "relative":
    colorbar_title = '(Tracer density)/(Mean Density)'
elif density_unit == "probability":
    colorbar_title = 'Probability Density [$h^{3}\\mathrm{MPc}^{-3}$]'

fig.colorbar(im1, ax=ax.ravel().tolist(),shrink=0.9,
    label=colorbar_title)
plt.subplots_adjust(wspace=0.0,hspace=0.0,left=0.1,right=0.75,bottom=0.15,top=0.95)
plt.savefig(figuresFolder + filename)
plt.show()

#-------------------------------------------------------------------------------
# FITTING CONTOURS

import contourpy

# To get the contour data:
CG = contourpy.contour_generator(bin_d_centres,bin_z_centres,field_lcdm)
contours = CG.lines(countour_list[0])

# We can then fit an ellipse to this.

def data_model(d,R,eps):
    if np.isscalar(d):
        if d >= R/eps:
            return 0.0
        else:
            return np.sqrt(R**2 - eps**2*d**2)
    else:
        nz = np.where(d <= R/eps)
        result = np.zeros(d.shape)
        result[nz] = np.sqrt(R**2 - eps**2*d[nz]**2)
        return result

def residual(z,d,params):
    R = params[0]
    eps = params[1]
    return z - data_model(d,R,eps)

zi = contours[0][:,0]
di = contours[0][:,1]
R_bounds = [0,2]
eps_bounds = [0,2]
lower_bounds = np.array([R_bounds[0],eps_bounds[0]])
upper_bounds = np.array([R_bounds[1],eps_bounds[1]])
ls_guess = scipy.optimize.least_squares(
    lambda x: residual(zi,di,x),np.array([1.0,1.0]),
    bounds=(lower_bounds,upper_bounds))

# Check plot:

plt.clf()
fig, ax = plt.subplots()
ax.scatter(di,zi,marker='x',color='k')
draw_ellipse(ax,ls_guess.x[0],ls_guess.x[1])
ax.set_xlabel('$d/R_{\\mathrm{eff}}$ (Perpendicular distance)',
               fontsize=fontsize,fontfamily=fontfamily)
ax.set_ylabel('$z/R_{\\mathrm{eff}}$ (LOS distance)',
               fontsize=fontsize,fontfamily=fontfamily)
plt.savefig(figuresFolder + "least_squares_test.pdf")
plt.show()


# Good for an estimate, but we really want to get errors on this, so we might
# need to setup a likelihood and do an MCMC on the two parameters.

# Error estimate assuming that the error is the distance to the closest point 
# along each dimension:
def get_interpolation_error_estimate(contour,di,zi,tol=1e-10):
    errors = np.zeros(contour.shape)
    num_points = contour.shape[0]
    for k in range(0,num_points):
        dist_d = np.abs(di - contour[k,0])
        dist_z = np.abs(zi - contour[k,1])
        closest_d = np.min(dist_d[dist_d > tol])
        closest_z = np.min(dist_z[dist_z > tol])
        errors[k,:] = np.array([closest_d,closest_z])
    return errors


# Now setup an MCMC:

errors = get_interpolation_error_estimate(contours[0],
                                          bin_d_centres,bin_z_centres)
yerr = errors[:,1]

# Log likelihood function:
def log_likelihood(theta, x, y, yerr):
    R, eps = theta
    model = data_model(x, R, eps)
    sigma2 = yerr**2
    return -0.5 * np.sum( (y - model)**2/sigma2 + np.log(sigma2) )

# Priors:
def log_prior(theta):
    R, eps = theta
    if (0 <= R < 2.0) & (0 <= eps < 2.0):
        return 0.0
    else:
        return -np.inf

def log_probability(theta,x,y,yerr):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, yerr)



# ML estimate:
x = contours[0][:,0]
y = contours[0][:,1]
nll = lambda *args: -log_likelihood(*args)
initial = np.array([1.0,1.0])
soln = scipy.optimize.minimize(nll, initial, 
    args=(x, y, yerr))
R_ml, eps_ml = soln.x

print("Maximum likelihood estimates:")
print("R = {0:.3f}".format(R_ml))
print("\\epsilon = {0:.3f}".format(eps_ml))

plt.clf()
x0 = np.linspace(0,2,101)
fig, ax = plt.subplots()
ax.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
ax.plot(x0, data_model(x0,ls_guess.x[0],ls_guess.x[1]), "--k", label="LS")
ax.plot(x0, data_model(x0,R_ml,eps_ml), ":k", label="ML")
ax.legend(fontsize=fontsize)
ax.set_xlim(0,2)
ax.set_ylim(0,2)
ax.set_xlabel('$d/R_{\\mathrm{eff}}$ (Perpendicular distance)',
               fontsize=fontsize,fontfamily=fontfamily)
ax.set_ylabel('$z/R_{\\mathrm{eff}}$ (LOS distance)',
               fontsize=fontsize,fontfamily=fontfamily)
plt.savefig(figuresFolder + "ml_test.pdf")
plt.show()


# MCMC run:

import emcee

pos = soln.x + 1e-4*np.random.randn(32,2)
nwalkers, ndim = pos.shape

sampler = emcee.EnsembleSampler(
    nwalkers, ndim, log_probability, args=(x, y, yerr)
)
sampler.run_mcmc(pos, 5000, progress=True)

tau = sampler.get_autocorr_time()

flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)


# Corner plot:
import corner

plt.clf()
fig = corner.corner(flat_samples, labels=["$R$","$\\epsilon$"])
fig.suptitle("$\\Lambda$-CDM Simulations Contour Ellipticity")
plt.savefig(figuresFolder + "corner_plot_R_eps.pdf")



def mcmc_contour_ellipticity(field,d_vals,z_vals,level,guess = 'ML',
                             initial=np.array([1.0,1.0]),nwalkers=32,disp=1e-4,
                             n_mcmc = 5000):
    # Fit the contour:
    CG = contourpy.contour_generator(d_vals,z_vals,field)
    contours = CG.lines(level)
    # Estimate errors for the contour:
    errors = get_interpolation_error_estimate(contours[0],d_vals,z_vals)
    # Data to fit:
    yerr = errors[:,1]
    x = contours[0][:,0]
    y = contours[0][:,1]
    ndims = 2
    # Get the initial guess:
    if guess == 'ML':
        nll = lambda *args: -log_likelihood(*args)
        soln = scipy.optimize.minimize(nll, initial, args=(x, y, yerr))
        pos = soln.x + disp*np.random.randn(nwalkers,ndims)
    elif guess == 'initial':
        pos = initial + disp*np.random.randn(nwalkers,ndims)
    else:
        raise Exception("Guess not recognised.")
    # Setup and run the sampler:
    sampler = emcee.EnsembleSampler(
        nwalkers, ndims, log_probability, args=(x, y, yerr)
    )
    sampler.run_mcmc(pos, n_mcmc, progress=True)
    # Filter the MCMC samples to account for correlation:
    tau = sampler.get_autocorr_time()
    tau_max = np.max(tau)
    flat_samples = sampler.get_chain(discard=int(3*tau_max), 
                                     thin=int(tau_max/2), flat=True)
    return flat_samples

flat_samples_borg = mcmc_contour_ellipticity(field_borg,
    bin_d_centres,bin_z_centres,1e-5)

flat_samples_lcdm = mcmc_contour_ellipticity(field_lcdm,
    bin_d_centres,bin_z_centres,1e-5)



plt.clf()
fig = corner.corner(flat_samples_lcdm, labels=["$R$","$\\epsilon$"])
fig.suptitle("$\\Lambda$-CDM Simulations Contour Ellipticity")
plt.savefig(figuresFolder + "corner_plot_R_eps.pdf")
plt.show()



plt.clf()
fig = corner.corner(flat_samples_borg, labels=["$R$","$\\epsilon$"])
fig.suptitle("Combined Catalogue Contour Ellipticity")
plt.savefig(figuresFolder + "corner_plot_R_eps_borg.pdf")
plt.show()


#-------------------------------------------------------------------------------
# WRONG COSMO SIMULATIONS

plus_sim = pynbody.load(
    "new_chain/sample10000/gadget_wrong_cosmo_forward_512/" + 
    "snapshot_domegam_plus_000")
plus_sim_reverse = pynbody.load(
    "new_chain/sample10000/gadget_wrong_cosmo_reverse_512/" + 
    "snapshot_domegam_plus_000")

ns_ref = 9

regular_sim = snapList[ns_ref]
regular_sim_reverse = snapListRev[ns_ref]

final_cat = cat300.get_final_catalogue(void_filter=True)
halo_indices = [-np.ones(len(final_cat),dtype=int) 
    for ns in range(0,len(snapList))]
for ns in range(0,len(snapList)):
    have_void = final_cat[:,ns] >= 0
    halo_indices[ns][have_void] = \
        cat300.indexListShort[ns][final_cat[have_void,ns]-1]

halo_indices = np.array(halo_indices).T

clean_indices = halo_indices[halo_indices[:,ns_ref] > 0,ns_ref]

# Need to cross-reference the halo catalogue for this to work:
antihalos_reg = regular_sim_reverse.halos()
antihalos_plus = plus_sim_reverse.halos()

bridge = pynbody.bridge.OrderBridge(plus_sim,regular_sim,monotonic=False)

bridge_reverse = pynbody.bridge.OrderBridge(
    regular_sim_reverse,plus_sim_reverse,monotonic=False)

match = bridge_reverse.match_catalog(min_index=1,max_index=15000,
                                     groups_1 = antihalos_reg,
                                     groups_2 = antihalos_plus)[1:]

ah_props_plus = pickle.load(open(plus_sim.filename + ".AHproperties.p","rb"))

antihalo_radii_plus = ah_props_plus[7]
antihalo_centres_plus = tools.remapAntiHaloCentre(
    ah_props_plus[5],boxsize,swapXZ  = False,reverse = True)

successful_match = match[clean_indices] > 0

radii_plus = antihalo_radii_plus[match[clean_indices][successful_match]]
radii_reg = antihaloRadii[ns_ref][clean_indices[successful_match]]

centres_plus = antihalo_centres_plus[match[clean_indices][successful_match],:]
centres_reg = antihaloCentres[ns_ref][clean_indices[successful_match],:]
centres_mean = cat300.getMeanCentres(void_filter=True)[halo_indices[:,ns_ref] > 0,:]




