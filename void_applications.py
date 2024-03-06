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
eps_list = np.linspace(0,2,1001)

eps_calc = np.array([get_los_ellipticity_in_ellipse(los_pos,eps,10) 
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


def estimate_ellipticity(los_pos,limits=[1e-5,2],npoints=101,R=10):
    eps_list = np.linspace(limits[0],limits[1],npoints)
    eps_calc = np.array([get_los_ellipticity_in_ellipse(los_pos,eps,R) 
        for eps in eps_list])
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

# Scatter test:
R=20
eps_est = estimate_ellipticity(los_pos,R=R)
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
        dist_max=20,rmin=10,rmax=20,all_particles=True):
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
    los_pos_all = []
    rad_filter = (radii > rmin) & (radii <= rmax)
    for k in tools.progressbar(range(0,len(centres))):
        if rad_filter[k]:
            if not all_particles:
                indices = hr_list[k+1]['iord']
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
        antihaloCentres,antihaloRadii,recompute=False,**kwargs):
    los_list = []
    for ns in range(0,len(snapList)):
        # Load snapshot (don't use the snapshot list, because that will force
        # loading of all snapshot positions, using up a lot of memory, when 
        # we only want to load these one at a time):
        print("Doing sample " + str(ns+1) + " of " + str(len(snapList)))
        los_pos_all = tools.loadOrRecompute(snapList[ns].filename + ".lospos.p",
            get_los_pos_for_snapshot,snapList[ns].filename,
            snapListRev[ns].filename,antihaloCentres[ns],antihaloRadii[ns],
            _recomputeData=recompute,**kwargs)
        los_list.append(los_pos_all)
    return los_list


los_list_borg = get_los_positions_for_all_catalogues(snapList,snapListRev,
    antihaloCentres,antihaloRadii,dist_max=60,rmin=10,rmax=20,recompute=True)

los_list_lcdm = get_los_positions_for_all_catalogues(snapListUn,
    snapListRevUn,antihaloCentresUn,antihaloRadiiUn,dist_max=60,rmin=10,
    rmax=20,recompute=True)


# Scatter of LCDM stacks:

# Test case:
#los_pos_all = los_list_lcdm[0]
los_pos_all = tools.loadOrRecompute(snapListUn[ns].filename + ".lospos.p",
    get_los_pos_for_snapshot,snapListUn[0].filename,snapListRevUn[0].filename,
    antihaloCentresUn[ns],antihaloRadiiUn[ns],_recomputeData=True,dist_max=60,
    rmin=10,rmax=20)

stacked_particles_lcdm = np.vstack(los_pos_all)
stacked_particles_lcdm_abs = np.abs(stacked_particles_lcdm)
R=12
eps_est = estimate_ellipticity(stacked_particles_lcdm,R=R)
drange = np.linspace(0,(R/eps_est)*1.1,101)

plt.clf()
fig, ax = plt.subplots()
ax.hist2d(stacked_particles_lcdm_abs[:,1],
           stacked_particles_lcdm_abs[:,0],
           bins=[np.linspace(0,20,31),np.linspace(0,20,31)],density=True,
           cmap="Blues")

ax.plot(drange,np.sqrt(R**2 - eps_est**2*drange**2),linestyle='--',color='k',
    label="Ellipse, $R =  " + ("%.2g" % R) + ", \\epsilon = " + 
    ("%.2g" % eps_est) + "$")
ax.plot(drange,-np.sqrt(R**2 - eps_est**2*drange**2),linestyle='--',color='k')
ax.set_xlabel('d [$\\mathrm{Mpc}h^{-1}$]')
ax.set_ylabel('z [$\\mathrm{Mpc}h^{-1}$]')
ax.set_xlim([0,20])
ax.set_ylim([0,20])
ax.set_aspect('equal')
ax.legend(frameon=False)
plt.savefig(figuresFolder + "ellipticity_scatter_all_lcdm.pdf")
plt.show()
















