#-------------------------------------------------------------------------------
# CONFIGURATION
from void_analysis import plot
from void_analysis.paper_plots_borg_antihalos_generate_data import *
from void_analysis.real_clusters import getClusterSkyPositions
from matplotlib import transforms
import pickle
import numpy as np
import seaborn as sns
seabornColormap = sns.color_palette("colorblind",as_cmap=True)
import pynbody
import nbodykit
from nbodykit import cosmology
import matplotlib.pylab as plt
import alphashape
from descartes import PolygonPatch
from void_analysis.plot import plotDensityComparison
from matplotlib import cm
import matplotlib.colors as colors
from matplotlib.ticker import NullFormatter

figuresFolder = "borg-antihalos_paper_figures/"

recomputeData = False

# Snapshots to use:
snapNumListOld = [7422,7500,8000,8500,9000,9500]
#snapNumList = [7000,7200,7400]
snapNumList = [7000,7200,7400,7600,8000]
#snapNumListUncon = [1,2,3,4,5,6]
snapNumListUncon = [1,2,3,4,5,6,7,8,9,10]
snapNumListUnconOld = [1,2,3,5,6]

# Filename data:
unconstrainedFolderNew = "new_chain/unconstrained_samples/"
unconstrainedFolderOld = "unconstrainedSamples/"
snapnameNew = "gadget_full_forward_512/snapshot_001"
snapnameNewRev = "gadget_full_reverse_512/snapshot_001"
samplesFolder="new_chain/"
samplesFolderOld = "./"
snapnameOld = "forward_output/snapshot_006"
snapnameOldRev = "reverse_output/snapshot_006"

#-------------------------------------------------------------------------------
# DATA FOR PLOTS

boxsize = 677.7
independentCentres = np.array([[0,0,0],[-boxsize/2,0,0],[0,-boxsize/2,0],\
    [0,0,-boxsize/2],[boxsize/4,boxsize/4,boxsize/4],\
    [-boxsize/4,boxsize/4,boxsize/4],[boxsize/4,-boxsize/4,boxsize/4],\
    [-boxsize/4,-boxsize/4,boxsize/4],[boxsize/4,boxsize/4,-boxsize/4],\
    [-boxsize/4,boxsize/4,-boxsize/4],[boxsize/4,-boxsize/4,-boxsize/4],\
    [-boxsize/4,-boxsize/4,-boxsize/4]])

# Void profile data (new run)
[rBinStackCentres,nbarjSepStack,\
        sigmaSepStack,nbarjSepStackUn,sigmaSepStackUn,\
        nbarjAllStacked,sigmaAllStacked,nbarjAllStackedUn,sigmaAllStackedUn,\
        nbar,rMin,mMin,mMax] = \
            tools.loadOrRecompute(figuresFolder + "void_profile_data.p",\
                getVoidProfilesData,snapNumList,snapNumListUncon,\
                unconstrainedFolder = unconstrainedFolderNew,\
                samplesFolder = samplesFolder,\
                snapname=snapnameNew,snapnameRev = snapnameNewRev,\
                _recomputeData = recomputeData,\
                unconstrainedCentreList = independentCentres)


# Void profile data (old run)
[rBinStackCentresOld,nbarjSepStackOld,\
        sigmaSepStackOld,nbarjSepStackUnOld,sigmaSepStackUnOld,\
        nbarjAllStackedOld,sigmaAllStackedOld,nbarjAllStackedUnOld,\
        sigmaAllStackedUnOld,nbarOld,rMinOld,mMinOld,mMaxOld] = \
            tools.loadOrRecompute(figuresFolder + "void_profile_data_old.p",\
                getVoidProfilesData,snapNumListOld,snapNumListUnconOld,\
                samplesFolder = './',\
                unconstrainedFolder = "unconstrainedSamples/",\
                snapname = "forward_output/snapshot_006",\
                snapnameRev = "reverse_output/snapshot_006",\
                _recomputeData=False)


[rBinStackCentresOld,nbarjSepStackOld,\
        sigmaSepStackOld,nbarjSepStackUnOld,sigmaSepStackUnOld,\
        nbarjAllStackedOld,sigmaAllStackedOld,nbarjAllStackedUnOld,\
        sigmaAllStackedUnOld,nbarOld,rMinOld,mMinOld,mMaxOld] = \
            tools.loadOrRecompute(figuresFolder + "void_profile_data_old.p",\
                getVoidProfilesData,snapNumListOld,snapNumListUnconOld,\
                samplesFolder = './',\
                unconstrainedFolder = "unconstrainedSamples/",\
                snapname = "forward_output/snapshot_006",\
                snapnameRev = "reverse_output/snapshot_006",\
                _recomputeData=True,\
                unconstrainedCentreList = independentCentres)


# Void profile data, voids paper:
snap200 = pynbody.load("../reversed_examples/sample1/standard200/snapshot_011")
ahProps200c = pickle.load(open(snap200.filename + ".AHproperties_M200c.p","rb"))
ahProps200m = pickle.load(open(snap200.filename + ".AHproperties_M200m.p","rb"))
ahCentresList200c = ahProps200c[5]
stackedMasses200c = ahProps200c[3]
stackedRadii200c = ahProps200c[7]
pairCountsList200c = ahProps200c[9]
volumesList200c = ahProps200c[10]
deltaCentral200c = ahProps200c[11]

ahCentresList200m = ahProps200m[5]
stackedMasses200m = ahProps200m[3]
stackedRadii200m = ahProps200m[7]
pairCountsList200m = ahProps200m[9]
volumesList200m = ahProps200m[10]
deltaCentral200m = ahProps200m[11]

nbar200 = (512/200)**3
[nbarjAllStacked200c,sigmaAllStacked200c] = stacking.stackVoidsWithFilter(\
        ahCentresList200c,stackedRadii200c,\
        np.where((stackedRadii200c > 5) & (stackedRadii200c < 25) & \
        (stackedMasses200c > 1e14) & (stackedMasses200c <= 1e15) & \
        (deltaCentral200c < 0))[0],\
        snap200,rBins = ahProps200c[8],\
        nPairsList = pairCountsList200c,\
        volumesList = volumesList200c,\
        method="poisson",errorType = "Weighted")

[nbarjAllStacked200m,sigmaAllStacked200m] = stacking.stackVoidsWithFilter(\
        ahCentresList200m,stackedRadii200m,\
        np.where((stackedRadii200m > 5) & (stackedRadii200m < 25) & \
        (stackedMasses200m > 1e14) & (stackedMasses200m <= 1e15) & \
        (deltaCentral200m < 0))[0],\
        snap200,rBins = ahProps200m[8],\
        nPairsList = pairCountsList200m,\
        volumesList = volumesList200m,\
        method="poisson",errorType = "Weighted")
#-------------------------------------------------------------------------------
# RADIAL VOID PROFILES

# Plot of void profiles for the old run, so that we can compare them with the
# same pipeline:
plot.plotConstrainedVsUnconstrainedProfiles(rBinStackCentresOld,\
    nbarjSepStackOld,sigmaSepStackOld,\
    nbarjAllStackedUnOld,sigmaAllStackedUnOld,nbarOld,rMinOld,mMinOld,\
    mMaxOld,showImmediately = True,\
    labelCon='Constrained',\
    labelRand=None,\
    meanErrorLabel = 'Unconstrained \nMean',\
    profileErrorLabel = 'Profile \nvariation \n',\
    savename=figuresFolder + "profiles1415Old.pdf",showTitle=False,\
    nbarjUnconstrainedStacks=nbarjSepStackUnOld,\
    sigmajUnconstrainedStacks = sigmaSepStackUnOld,\
    showMean = False)

plot.plotConstrainedVsUnconstrainedProfiles(rBinStackCentresOld,\
    nbarjSepStackOld,sigmaSepStackOld,\
    nbarjAllStackedUnOld,sigmaAllStackedUnOld,nbarOld,rMinOld,mMinOld,\
    mMaxOld,showImmediately = True,fontsize = 12,legendFontSize=12,\
    labelCon='Constrained',\
    labelRand='Unconstrained',\
    savename=figuresFolder + "profiles1415Old_allstacked.pdf",showTitle=False)

plot.plotConstrainedVsUnconstrainedProfiles(rBinStackCentres,nbarjSepStack,\
    sigmaSepStack,nbarjSepStackUn,sigmaSepStackUn,nbar,rMin,mMin,mMax,\
    showImmediately = True,fontsize = 12,legendFontSize=12,\
    labelCon='Constrained simulations',\
    labelRand='Unconstrained simulations',\
    savename=figuresFolder + "profiles1415_sepstack.pdf")

plot.plotConstrainedVsUnconstrainedProfiles(rBinStackCentres,nbarjSepStack,\
    sigmaSepStack,nbarjAllStackedUn,\
    sigmaAllStackedUn,nbar,rMin,mMin,mMax,\
    showImmediately = True,fontsize = 12,legendFontSize=12,\
    labelCon='Constrained simulations',\
    labelRand='Unconstrained simulations',\
    savename=figuresFolder + "profiles1415_allstacked.pdf")



# Comparing with the original standard200 runs:
comparison = "new"
if comparison == "old":
    plot.plotConstrainedVsUnconstrainedProfiles(rBinStackCentresOld,\
    nbarjSepStackOld,sigmaSepStackOld,\
    nbarjAllStackedUnOld,sigmaAllStackedUnOld,nbarOld,rMinOld,mMinOld,\
    mMaxOld,showImmediately = False,\
    labelCon='Constrained',\
    labelRand=None,\
    meanErrorLabel = 'Unconstrained \nMean',\
    profileErrorLabel = 'Profile \nvariation \n',\
    savename=figuresFolder + "profiles1415Old.pdf",showTitle=False,\
    nbarjUnconstrainedStacks=nbarjSepStackUnOld,\
    sigmajUnconstrainedStacks = sigmaSepStackUnOld,\
    showMean = False,showConstrained=False)
elif comparison == "new":
    plot.plotConstrainedVsUnconstrainedProfiles(rBinStackCentres,nbarjSepStack,\
    sigmaSepStack,nbarjAllStackedUn,sigmaAllStackedUn,nbar,rMin,mMin,mMax,\
    showImmediately = False,fontsize = fontsize,legendFontSize=legendFontsize,\
    labelCon='Constrained',\
    labelRand='Unconstrained \nmean',\
    showTitle=False,\
    meanErrorLabel = 'Unconstrained \nMean',\
    profileErrorLabel = 'Profile \nvariation \n',\
    nbarjUnconstrainedStacks=nbarjSepStackUn,\
    sigmajUnconstrainedStacks = sigmaSepStackUn,showMean=False,\
    showConstrained=False)

plt.errorbar(rBinStackCentres,nbarjAllStacked200c/nbar200,\
    yerr=sigmaAllStacked200c/nbar200,color=seabornColormap[0],linestyle='--',\
    label='Published \nsimulation \n(M200c)')
plt.errorbar(rBinStackCentres,nbarjAllStacked200m/nbar200,\
    yerr=sigmaAllStacked200m/nbar200,color=seabornColormap[1],linestyle='--',\
    label='Published \nsimulation \n(M200m)')
plt.legend(prop={"size":7,"family":'serif'})
plt.savefig(figuresFolder + "supporting_plots/standard200_comparison.pdf")
plt.show()


#-------------------------------------------------------------------------------
# POWER SPECTRUM PLOT:

psUncon = [pickle.load(open("new_chain/unconstrained_samples/sample" + \
    str(snapNumListUncon[i]) + "/gadget_full_forward_512/snapshot_001.ps.p",\
    "rb")) for i in range(0,len(snapNumListUncon))]

psCon = [pickle.load(open("new_chain/sample" + \
    str(snapNumList[i]) + "/gadget_full_forward_512/snapshot_001.ps.p",\
    "rb")) for i in range(0,len(snapNumList))]

powerUncon = [ps.power['power'].real for ps in psUncon]
cosmo = cosmology.cosmology.Cosmology(Omega_cdm = 0.3111 - 0.04897,h=0.6766,Omega0_b=0.04897,n_s=0.9665).match(sigma8=0.8102)
psLin = nbodykit.cosmology.LinearPower(cosmo,redshift=0,transfer='EisensteinHu')
psNonLin = nbodykit.cosmology.HalofitPower(cosmo,redshift=0)

def plotPS(ps,ax=None,linestyle = '-',color=None,psTheory=None,\
    label='Simulation Power Spectrum',legend=True,\
    theoryLabel = "Linear Power Spectrum",show = True):
    if ax is None:
        fig, ax = plt.subplots()
    ax.loglog(ps.power['k'],ps.power['power'].real - ps.attrs['shotnoise'],\
        linestyle=linestyle,color=color,label=label)
    if psTheory is not None:
        ax.loglog(ps.power['k'],psTheory(ps.power['k']),color='b',\
            linestyle=':',label=theoryLabel)
    ax.set_xlabel(r"$k$ [$h \ \mathrm{Mpc}^{-1}$]")
    ax.set_ylabel(r"$P(k)$ [$h^{-3}\mathrm{Mpc}^3$]")
    ax.set_xlim(0.01, 0.6)
    if legend:
        ax.legend()
    if show:
        plt.show()

fig, ax = plt.subplots()
for k in range(0,len(psUncon)):
    ps = psUncon[k]
    label = 'Sample ' + str(snapNumListUncon[k])
    ax.loglog(ps.power['k'],ps.power['power'].real - ps.attrs['shotnoise'],\
        linestyle='-',color=seabornColormap[k],label=label)

ax.loglog(ps.power['k'],psNonLin(ps.power['k']),color='b',linestyle=':',\
    label='Halofit')
ax.set_xlabel(r"$k$ [$h \ \mathrm{Mpc}^{-1}$]")
ax.set_ylabel(r"$P(k)$ [$h^{-3}\mathrm{Mpc}^3$]")
ax.legend()
ax.set_title('Unconstrained simulation power spectra vs Halofit')
plt.savefig(figuresFolder + "supporting_plots/unconstrained_ps.pdf")
plt.show()


fig, ax = plt.subplots()
for k in range(0,len(psCon)):
    ps = psCon[k]
    label = 'Sample ' + str(snapNumList[k])
    ax.loglog(ps.power['k'],ps.power['power'].real - ps.attrs['shotnoise'],\
        linestyle='-',color=seabornColormap[k],label=label)

ax.loglog(ps.power['k'],psNonLin(ps.power['k']),color='b',linestyle=':',\
    label='Halofit')
ax.set_xlabel(r"$k$ [$h \ \mathrm{Mpc}^{-1}$]")
ax.set_ylabel(r"$P(k)$ [$h^{-3}\mathrm{Mpc}^3$]")
ax.legend()
ax.set_title('Constrained simulation power spectra vs Halofit')
plt.savefig(figuresFolder + "supporting_plots/constrained_ps.pdf")
plt.show()


denFor = [np.reshape(np.fromfile("new_chain/unconstrained_samples/sample" + \
    str(k+1) + "/gadget_full_forward_512/snapshot_001.a_den",dtype=np.float32),\
    (256,256,256),order='F') for k in range(0,2)]

denRev = [np.reshape(np.fromfile("new_chain/unconstrained_samples/sample" + \
    str(k+1) + "/gadget_full_reverse_512/snapshot_001.a_den",dtype=np.float32),\
    (256,256,256),order='F') for k in range(0,2)]

den = [np.reshape(np.fromfile("new_chain/unconstrained_samples/sample" + \
    str(k+1) + "/ic/gadget_ic_512_for.gadget2.a_den",dtype=np.float32),\
    (256,256,256),order='F') for k in range(0,2)]
denr = [np.reshape(np.fromfile("new_chain/unconstrained_samples/sample" + \
    str(k+1) + "/ic/gadget_ic_512_rev.gadget2.a_den",dtype=np.float32),\
    (256,256,256),order='F') for k in range(0,2)]

den1 = np.reshape(np.fromfile("new_chain/unconstrained_samples/" + \
    "sample1/ic/gadget_ic_512_for.gadget2.a_den",dtype=np.float32),\
    (256,256,256),order='F')
den1r = np.reshape(np.fromfile("new_chain/unconstrained_samples/" + \
    "sample1/ic/gadget_ic_512_rev.gadget2.a_den",dtype=np.float32),\
    (256,256,256),order='F')

denRevZeld = [np.reshape(np.fromfile("new_chain/unconstrained_samples/" + \
    "sample" + str(k+1) + "/ic/gadget_z0_test_rev.gadget2.a_den",\
    dtype=np.float32),(256,256,256),order='F') for k in range(0,2)]

denForZeld = [np.reshape(np.fromfile("new_chain/unconstrained_samples/" + \
    "sample" + str(k+1) + "/ic/gadget_z0_test_for.gadget2.a_den",\
    dtype=np.float32),(256,256,256),order='F') for k in range(0,2)]

ns = 0

plotDensityComparison(denFor[ns],denRev[ns],N=256,\
    centre1=np.array([0]*3),\
    centre2=np.array([0]*3),width = 200,\
    textLeft = "Forward",textRight="reverse",\
    title="Forward vs Reverse simulations",vmax = 1000,\
    vmin=1/1000)

plotDensityComparison(denFor[0],denFor[1],N=256,\
    centre1=np.array([0]*3),\
    centre2=np.array([0]*3),width = 200,\
    textLeft = "Sample 1",textRight="Sample 2",\
    title="Unconstrained simulations",vmax = 1000,\
    vmin=1/1000)

plotDensityComparison(den[ns],denr[ns],N=256,\
    centre1=np.array([0]*3),\
    centre2=np.array([0]*3),width = 200,\
    textLeft = "Forward ICS",textRight="Reverse ICs",\
    title="Unconstrained simulations, Sample " + str(ns+1),\
    vmax = 1.5,vmin=2.0/3.0)

ns = 1
plotDensityComparison(denRev[ns],denRevZeld[ns],N=256,\
    centre1=np.array([0]*3),\
    centre2=np.array([0]*3),width = 400,\
    textLeft = "GADGET",textRight="Zel'dovich",\
    title="Unconstrained simulations (reverse), Sample " + str(ns+1),\
    vmax = 1000,vmin=1/1000)

ns = 1
plotDensityComparison(denFor[ns],denForZeld[ns],N=256,\
    centre1=np.array([0]*3),\
    centre2=np.array([0]*3),width = 200,\
    textLeft = "GADGET",textRight="Zel'dovich",\
    title="Unconstrained simulations (reverse), Sample " + str(ns+1),\
    vmax = 1000,vmin=1/1000)


# 2M++ Data:
catFile = "./2mpp_data/2m++.txt"
catalogue = np.loadtxt(catFile,delimiter='|',skiprows=31,
    usecols=(1,2,3,4,5,6,7,8,10,11,12,13,14,15,16))
# Filter useable galaxies:
useGalaxy = (catalogue[:,10] == 0.0) & (catalogue[:,5] > 0)
c = 299792.458 # Speed of light in km/s
z = catalogue[:,5]/c # Redshift
# Cosmological parameters:

# Comoving distance to all galaxies, in Mpc/h:
dcz = cosmo.comoving_distance(z[useGalaxy]).value*cosmo.h
# Co-ordinates of the galaxies (in Mpc/h):
coord = astropy.coordinates.SkyCoord(\
    ra = catalogue[useGalaxy,0]*astropy.units.degree,\
    dec=catalogue[useGalaxy,1]*astropy.units.degree,\
    distance=dcz*astropy.units.Mpc)
# Cartesian positions of galaxies in equatorial, comoving co-ordinates (Mpc/h):
equatorialXYZ = np.vstack((coord.cartesian.x.value,\
    coord.cartesian.y.value,coord.cartesian.z.value)).T



cl = 2
ns = 0
centre1 = clusterLoc[cl]
centre2 = -np.fliplr(clusterCentresSim[ns])[cl]
centre3 = clusterCentresSim[ns][cl]
#centre1 = -np.flip(haloCentres[0][np.array(\
#    centralHalos[0][0])[constrainedHaloSortM20[0]][0]])
#centre2 = -np.flip(haloCentres[0][np.array(\
#    centralHalos[0][0])[constrainedHaloSortM20[0]][0]])
#centre1 = truePeakLocs[0][truePeakOrder[0],:]
#centre2 = truePeakLocs[0][truePeakOrder[0],:]
losAxis = 1
sort = {0:[1,2],1:[0,2],2:[0,1]}
otherAxis = sort[losAxis]
width = 75
thickness = 8
Mlow = 5e13
Mhigh = 1e16
label = True
densityListDTFE = [np.fromfile(samplesFolder + "sample" + \
                str(snap) + "/gadget_full_forward_512/snapshot_001.a_den",
                dtype=np.float32) for snap in snapNumList]
densityListN = [np.reshape(den,(256,256,256),order='F') \
   for den in densityListDTFE]
ax = plotDensityComparison(mcmcDen_r[ns],np.flip(densityListN[ns]),\
    N=256,centre1=centre1,centre2=centre1,width = width,\
    textLeft = "Sample " + str(snapNumList[ns]) + " (Final Density)",\
    textRight="Sample " + str(snapNumList[ns]) + " (Evolved ICs)",\
    title="Density Field around " + clusterNames[cl][0],vmax = 1000,\
    vmin=1/1000,\
    markCentre=True,losAxis=losAxis,showGalaxies=True,flipCentreLeft=False,\
    flipCentreRight=False,flipRight=False,flipLeft=False,\
    invertAxisLeft=False,invertAxisRight=False,flipudLeft=False,\
    flipudRight=False,fliplrLeft=False,fliplrRight=False,\
    swapXZLeft=True,swapXZRight=True,\
    gal_position=equatorialXYZ,show=False,returnAx = True,thickness=thickness)

plt.show()

ngMCMC0 = ngPerLBin(
            biasParam,return_samples=True,mask=mask,\
            accelerate=True,\
            delta = [mcmcDenLin_r[0]],contrast=False,sampleList=[0],\
            beta=biasParam[0][:,:,1],rhog = biasParam[0][:,:,3],\
            epsg=biasParam[0][:,:,2],\
            nmean=biasParam[0][:,:,0],biasModel = biasNew)

# Saving the indices of the main clusters, for use in MCMC:
boxsize = 677.7
h = 0.6766
Om0 = 0.3111
Ode = 1 - Om0
cosmo = astropy.cosmology.LambdaCDM(100*h,Om0,Ode0)
# 2M++ catalogue data, and cluster locations:
[combinedAbellN,combinedAbellPos,abell_nums] = \
    real_clusters.getCombinedAbellCatalogue(Om0 = 0.3111,Ode0 = 0.6889,\
        h=0.6766,catFolder="")
clusterInd = [np.where(combinedAbellN == n)[0] for n in abell_nums]
clusterLoc = np.zeros((len(clusterInd),3))
for k in range(0,len(clusterInd)):
    if len(clusterInd[k]) == 1:
        clusterLoc[k,:] = combinedAbellPos[clusterInd[k][0],:]
    else:
        # Average positions:
        clusterLoc[k,:] = np.mean(combinedAbellPos[clusterInd[k],:],0)

wrappedPos = snapedit.wrap(clusterLoc + boxsize/2,boxsize)

indices5 = tree.query_ball_point(wrappedPos,5.0)
indices10 = tree.query_ball_point(wrappedPos,10.0)
indices15 = tree.query_ball_point(wrappedPos,15.0)
indices20 = tree.query_ball_point(wrappedPos,20.0)
indices2p5 = tree.query_ball_point(wrappedPos,2.5)

rList = np.array([2.5,5,10,15,20])

indicesAll = [tree.query_ball_point(wrappedPos,rad) for rad in rList]

pickle.dump([rList,indicesAll],open("cluster_indices.p","wb"))



#-------------------------------------------------------------------------------
# Compare first two unconstrained simulations:
def zobovVolumesToPhysical(zobovVolumes,snap,dtype=np.double):
    N = np.round(np.cbrt(len(snap))).astype(int)
    boxsize = snap.properties['boxsize'].ratio("Mpc a h**-1")
    if type(zobovVolumes) == type(""):
        vols = np.fromfile(zobovVolumes,dtype=dtype,offset=4)
    else:
        vols = zobovVolumes
    return vols*(boxsize/N)**3


snapNumListUncon = [1,2]
samplesFolder="new_chain/"
unconstrainedFolder="new_chain/unconstrained_samples/"
snapname = "gadget_full_forward_512/snapshot_001"
snapnameRev = "gadget_full_reverse_512/snapshot_001"
reCentreSnaps = False
N = 512
boxsize = 677.7
mMin = 1e14
mMax = 1e15
rMin = 5
rMax = 25
verbose = True

snapList =  [pynbody.load(samplesFolder + "sample" + str(snapNum) + "/" + \
    snapname) for snapNum in snapNumList]
snapListRev =  [pynbody.load(samplesFolder + "sample" + str(snapNum) + "/" \
    + snapnameRev) for snapNum in snapNumList]

if reCentreSnaps:
    for snap in snapList:
        tools.remapBORGSimulation(snap,swapXZ=False,reverse=True)
        snap.recentred = True
else:
    for snap in snapList:
        snap.recentred = False

# Load unconstrained snaps:
if verbose:
    print("Loading snapshots...")

snapListUnconstrained = [pynbody.load(unconstrainedFolder + "sample" \
    + str(snapNum) + "/" + snapname) for snapNum in snapNumListUncon]
snapListUnconstrainedRev = [pynbody.load(unconstrainedFolder + \
        "sample" + str(snapNum) + "/" + snapnameRev) \
        for snapNum in snapNumListUncon]

# recentre snaps:
if reCentreSnaps:
    for snap in snapListUnconstrained:
        tools.remapBORGSimulation(snap)
        snap.recentred = True
else:
    for snap in snapListUnconstrained:
        snap.recentred = False

ahPropsConstrained = [pickle.load(\
    open(snap.filename + ".AHproperties.p","rb")) \
    for snap in snapList]
ahPropsUnconstrained = [pickle.load(\
    open(snap.filename + ".AHproperties.p","rb")) \
    for snap in snapListUnconstrained]

nbar = (N/boxsize)**3
# Constrained antihalo properties:
if reCentreSnaps:
    ahCentresList = [tools.remapAntiHaloCentre(props[5],boxsize) \
        for props in ahPropsConstrained]
else:
    ahCentresList = [props[5] \
        for props in ahPropsConstrained]


antihaloMassesList = [props[3] for props in ahPropsConstrained]
antihaloRadii = [props[7] for props in ahPropsConstrained]
deltaCentralList = [props[11] for props in ahPropsConstrained]
deltaMeanList = [props[12] for props in ahPropsConstrained]
pairCountsList = [props[9] for props in ahPropsConstrained]
volumesList = [props[10] for props in ahPropsConstrained]
rBins = ahPropsConstrained[0][8]
rBinStackCentres = plot.binCentres(rBins)
centralAntihalosCon = [tools.getAntiHalosInSphere(hcentres,135) \
    for hcentres in ahCentresList]
# Unconstrained antihalo properties:
if reCentreSnaps:
    ahCentresListUn = [tools.remapAntiHaloCentre(props[5],boxsize) \
        for props in ahPropsUnconstrained]
else:
    ahCentresListUn = [props[5] \
        for props in ahPropsUnconstrained]


antihaloMassesListUn = [props[3] for props in ahPropsUnconstrained]
antihaloRadiiUn = [props[7] for props in ahPropsUnconstrained]
deltaCentralListUn = [props[11] for props in ahPropsUnconstrained]
deltaMeanListUn = [props[12] for props in ahPropsUnconstrained]
pairCountsListUn = [props[9] for props in ahPropsUnconstrained]
volumesListUn = [props[10] for props in ahPropsUnconstrained]
centralAntihalosUn = [tools.getAntiHalosInSphere(hcentres,135) \
    for hcentres in ahCentresListUn]
# Select antihalos in the central region:
conditionList = [(deltaCentralList[ns] < 0) & \
    (centralAntihalosCon[ns][1]) for ns in range(0,len(snapNumList))]
conditionListUn = [(deltaCentralListUn[ns] < 0) & \
    (centralAntihalosUn[ns][1]) \
    for ns in range(0,len(snapNumListUncon))]
conditionListMrange = [(deltaCentralList[ns] < 0) & \
    (centralAntihalosCon[ns][1]) & (antihaloMassesList[ns] > mMin) & \
    (antihaloMassesList[ns] <= mMax) \
    for ns in range(0,len(snapNumList))]
conditionListMrangeUn = [(deltaCentralListUn[ns] < 0) & \
    (centralAntihalosUn[ns][1]) & (antihaloMassesListUn[ns] > mMin) & \
    (antihaloMassesListUn[ns] <= mMax)
    for ns in range(0,len(snapNumListUncon))]
# Stacked profile data:
[nbarjStack,sigmaStack] = stacking.computeMeanStacks(ahCentresList,\
    antihaloRadii,antihaloMassesList,conditionList,pairCountsList,\
    volumesList,snapList,nbar,rBins,rMin,rMax,mMin,mMax)
[nbarjRandStack,sigmaRandStack] = stacking.computeMeanStacks(\
    ahCentresListUn,antihaloRadiiUn,antihaloMassesListUn,\
    conditionListUn,pairCountsListUn,volumesListUn,snapListUnconstrained,\
    nbar,rBins,rMin,rMax,mMin,mMax)

hrList = [snap.halos() for snap in snapListUnconstrainedRev]
snapsortList = [np.argsort(snap['iord']) for snap in snapListUnconstrained]
treeList = [tools.getKDTree(snap) for snap in snapListUnconstrained]
rSearch = 5.0
ahCentralParts = [treeList[k].query_ball_point(ahCentresListUn[k],rSearch,\
    workers=-1) for k in range(0,len(snapNumListUncon))]
volumesList = [zobovVolumesToPhysical(snap.filename + ".vols",snap,\
    dtype=np.double) for snap in snapListUnconstrained]
mUnit = np.array([snap['mass'][0]*1e10 for snap in snapListUnconstrained])
rhoM = 0.3111*2.7754e11

sortedVolumesList = [volumesList[k][snapsortList[k]] \
    for k in range(0,len(snapListUnconstrained))]

rhoSPHList = [snap['rho'] \
    for snap in snapListUnconstrained]

for snap in snapListUnconstrained:
    snap['rho'].convert_units("Msol h**2 Mpc**-3")


vorSPHList = [mUnit[k]/rhoSPHList[k] \
    for k in range(0,len(snapListUnconstrained))]

rhoVCentral = [np.array([len(ahCentralParts[k][l])*mUnit[k]/\
    (np.sum(volumesList[k][ahCentralParts[k][l]])*rhoM) \
    for l in range(0,len(ahCentralParts[k]))]) \
    for k in range(0,len(snapListUnconstrained))]

rhoVCentralSorted = [np.array([len(ahCentralParts[k][l])*mUnit[k]/\
    (np.sum(sortedVolumesList[k][ahCentralParts[k][l]])*rhoM) \
    for l in range(0,len(ahCentralParts[k]))]) \
    for k in range(0,len(snapListUnconstrained))]

filterList = [np.where(masses > 1e14) for masses in antihaloMassesListUn]

for k in range(0,len(snapListUnconstrainedRev)):
    plt.hist(rhoVCentralSorted[k][filterList[k]],bins = 10**(np.linspace(-1,2,21)),\
        color=seabornColormap[k],alpha=0.5,\
        label='Sample ' + str(snapNumListUncon[k]))

plt.xscale('log')
plt.yscale('log')
plt.xlabel('$\\rho(r < 5\\mathrm{Mpc}h^{-1})/\\bar{\\rho}$')
plt.ylabel('Number of antihalos')
plt.legend()
plt.show()


# Scatter a selection:
randSelection = np.random.randint(0,512**3,10000)
plt.scatter(vorSPHList[0][randSelection],volumesList[0][randSelection],\
    label='Sample 1',marker='.')
plt.scatter(vorSPHList[1][randSelection],volumesList[1][randSelection],\
    label='Sample 2',marker='.')
plt.xlabel('SPH Volume')
plt.ylabel('Voronoi Volume')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.show()

# Plot about A void:
ns = 1
nh = 9
from void_analysis import mayavi_plot
mayavi_plot.plotAboutPoint(snapListUnconstrained[ns],ahCentresListUn[ns][nh],\
    radius=20,hold=True)
mayavi_plot.subsnap_scatter(snapListUnconstrained[ns][\
    snapsortList[ns][hrList[ns][nh+1]['iord']]],color_spec=(1,0,0))


def posRemap(pos,boxsize,fliplr=False,reverseAboutCentre=True,\
        centre = None,returnWrapped = False,centreRelative=True):
    posNew = np.array(pos)
    if centre is None:
        centre = np.array([boxsize/2]*3)
    if centreRelative:
        posNew = snapedit.unwrap(posNew - centre,boxsize)
    if fliplr:
        if len(pos.shape) > 1:
            posNew = np.fliplr(posNew)
        else:
            posNew = np.flip(posNew)
    if reverseAboutCentre:
        if centreRelative:
            posNew = -posNew
        else:
            posNew = -snapedit.unwrap(posNew - centre,boxsize) + boxsize/2
    if returnWrapped:
        posNew = snapedit.wrap(posNew,boxsize)
    return posNew

# Slice through a void:
ns = 0
nh = 9
thickness = 8
width = 100
propsList = ['Xc','Yc','Zc']
centre1 = posRemap(ahCentresListUn[ns][nh],boxsize)
centre2 = posRemap(np.array([hrList[ns][nh+1].properties[propsList[k]]/1000 \
    for k in range(0,3)]),boxsize)
losAxis = 1
sort = {0:[1,2],1:[0,2],2:[0,1]}
optimise = False
# Positions of particles:
posVoid = posRemap(snapListUnconstrained[ns]['pos'][\
    snapsortList[ns][hrList[ns][nh+1]['iord']],:],boxsize)
filterSlice = np.where((posVoid[:,losAxis] >= centre1[losAxis] - thickness/2) \
    & (posVoid[:,losAxis] <= centre1[losAxis] + thickness/2))[0]
if optimise:
    alphaVal = alphashape.optimizealpha(posVoid[filterSlice,:][:,sort[losAxis]])
else:
    #alphaVal = 0.7421910513557205 # Suitable default chosen by optimiser
    alphaVal = 0.5

alphaShape = alphashape.alphashape(posVoid[filterSlice,:][:,sort[losAxis]],\
    alphaVal)

ax = plotDensityComparison(np.flip(denFor[ns]),np.flip(denRev[ns]),N=256,\
    centre1=centre1,centre2=centre2,\
    width = width,textLeft = "Forward",textRight="Reverse",\
    title="Forward vs Reverse simulations, sample " + \
    str(snapNumListUncon[ns]),vmax = 1000,\
    vmin=1/1000,thickness=thickness,returnAx=True,show=False)
ax[0].add_patch(PolygonPatch(alphaShape,fc=seabornColormap[0],ec='k',\
    alpha=0.5))
ax[1].scatter(centre2[sort[losAxis]][0],\
    centre2[sort[losAxis]][1],\
    marker='x',color='r')
plt.show()

#-------------------------------------------------------------------------------
# TEST OF THE ANTIHALO POPULATION:

def getAntihaloDensityInShell(snap,ahCentres,rMin = 2.5,rMax = 3.0,\
        relative=False,ahRadii=None,max_index = None,ahFilter = None,\
        cacheTree = False):
    if relative and (ahRadii is None):
        raise Exception("Anti-halo radii required for relative comparison")
    tree = tools.getKDTree(snap,cacheTree=cacheTree)
    if ahFilter is None:
        centres = ahCentres
        if ahRadii is not None:
            radii = ahRadii
    else:
        centres = ahCentres[ahFilter]
        if ahRadii is not None:
            radii = ahRadii[ahFilter]
    if max_index is not None:
        centres = centres[0:max_index,:]
        if ahRadii is not None:
            radii = radii[0:max_index]
    # Get number of particles within an annulus:
    if relative:
        rSearchMax = rMax*radii
        rSearchMin = rMin*radii
    else:
        rSearchMax = rMax
        rSearchMin = rMin
    npointsMax = tree.query_ball_point(centres,rSearchMax,workers=-1,\
        return_length=True)
    if rMin > 0:
        npointsMin = tree.query_ball_point(centres,rSearchMin,workers=-1,\
            return_length=True)
    else:
        npointsMin = 0
    # Compute the density:
    boxsize = snap.properties['boxsize'].ratio("Mpc a h**-1")
    nbar = len(snap)/(boxsize**3) # mean number density
    nAnnulus = (npointsMax - npointsMin)/\
        (4*np.pi*(rSearchMax**3 - rSearchMin**3)/3)
    return nAnnulus/nbar

# Plot comparing different cases:
snapList =  [pynbody.load(samplesFolder + "sample" + str(snapNum) + "/" + \
        "gadget_full_forward_512/snapshot_001") \
        for snapNum in snapNumList]
snapListUn =  [pynbody.load(unconstrainedFolderNew + "sample" + str(snapNum) + \
    "/" + "gadget_full_forward_512/snapshot_001") \
    for snapNum in snapNumListUncon]

ahProps = [pickle.load(\
        open(snap.filename + ".AHproperties.p","rb")) \
        for snap in snapList]
ahPropsUn = [pickle.load(\
        open(snap.filename + ".AHproperties.p","rb")) \
        for snap in snapListUn]

boxsize = 677.7
ahCentresList = [tools.remapAntiHaloCentre(props[5],boxsize) \
    for props in ahProps]
ahCentresListUn = [tools.remapAntiHaloCentre(props[5],boxsize) \
    for props in ahPropsUn]
ahCentresListUnmapped = [props[5] for props in ahProps]
ahCentresListUnmappedUn = [props[5] for props in ahPropsUn]
antihaloRadii = [props[7] for props in ahProps]
antihaloRadiiUn = [props[7] for props in ahPropsUn]
antihaloMassesList = [props[3] for props in ahProps]
antihaloMassesListUn = [props[3] for props in ahPropsUn]
deltaCentralList = [props[11] for props in ahProps]
deltaCentralListUn = [props[11] for props in ahPropsUn]
mMin = 1e14
mMax = 1e15
rMin = 2.5
rMax = 3.0

centralAntihalos = [tools.getAntiHalosInSphere(hcentres,135) \
        for hcentres in ahCentresList]
centralAntihalosUn = [tools.getAntiHalosInSphere(hcentres,135) \
        for hcentres in ahCentresListUn]
conditionListMrange = [(deltaCentralList[ns] < 0) & \
        (centralAntihalos[ns][1]) & (antihaloMassesList[ns] > mMin) & \
        (antihaloMassesList[ns] <= mMax) \
        for ns in range(0,len(snapNumList))]
conditionListMrangeUn = [(deltaCentralListUn[ns] < 0) & \
        (centralAntihalosUn[ns][1]) & (antihaloMassesListUn[ns] > mMin) & \
        (antihaloMassesListUn[ns] <= mMax) \
        for ns in range(0,len(snapNumListUncon))]




# Compute the densities:
ahDensityList = []
for k in range(0,len(snapList)):
    snap = pynbody.load(samplesFolder + "sample" + str(snapNumList[k]) + "/" + \
        "gadget_full_forward_512/snapshot_001")
    gc.collect()
    ahDensityList.append(getAntihaloDensityInShell(snapList[k],\
    ahCentresListUnmapped[k],rMin = rMin,rMax = rMax,relative=True,\
    ahRadii = antihaloRadii[k],ahFilter = conditionListMrange[k]))

ahDensityListUn = []
for k in range(0,len(snapListUn)):
    snap = pynbody.load(unconstrainedFolderNew + "sample" + \
        str(snapNumListUncon[k]) + "/" + "gadget_full_forward_512/snapshot_001")
    gc.collect()
    ahDensityListUn.append(getAntihaloDensityInShell(snapListUn[k],\
    ahCentresListUnmappedUn[k],rMin = rMin,rMax = rMax,relative=True,\
    ahRadii = antihaloRadiiUn[k],ahFilter = conditionListMrangeUn[k]))


# plot of the densities:

def plotDensityAnnulusHistograms(snapList,snapNumList,ahDensityList,\
        rhoBins = np.linspace(0.2,1.8,21),savename = None,relative=True,\
        alpha = 0.5,useHist = True):
    fig, ax = plt.subplots()
    for k in range(0,len(snapList)):
        if useHist:
            handle = ax.hist(ahDensityList[k],bins=rhoBins,\
                color=seabornColormap[k],\
                label = 'Simulation ' + str(k+1),alpha=alpha)
        else:
            handle = np.histogram(ahDensityList[k],bins=rhoBins)
            plt.errorbar(plot.binCentres(rhoBins),handle[0],\
                yerr=np.sqrt(handle[0]),color=seabornColormap[k],\
                linestyle='-',label = 'Simulation ' + str(k+1))
        plt.axvline(np.mean(ahDensityList[k]),color=seabornColormap[k],\
            linestyle='--',label = 'Simulation ' + str(k+1) + ' mean')
    plt.xlabel('$\\rho/\\bar{\\rho}$')
    plt.ylabel('Number of anti-halos')
    plt.axvline(1.0,color='k',linestyle=':',label='Cosmological mean')
    plt.legend()
    if relative:
        plt.title('Average density, $' + str(rMin) + \
            ' < R/R_{\\mathrm{eff}} < ' + str(rMax) + '$')
    else:
        plt.title('Average density, $' + str(rMin) + \
            ' < R/(\\mathrm{Mpc}h^{-1}) < ' + str(rMax) + '$')
    if savename is not None:
        plt.savefig(savename)
    plt.show()

def minmax(x):
    return np.array([np.min(x),np.max(x)])

plotDensityAnnulusHistograms(snapListUn,snapNumListUncon,ahDensityListUn,\
    savename = figuresFolder + 'unconstrained_density_histograms.pdf',\
    useHist = True,rhoBins = np.linspace(0.2,1.8,11))
plotDensityAnnulusHistograms(snapList,snapNumList,ahDensityList,\
    savename = figuresFolder + 'constrained_density_histograms.pdf',\
    useHist = True,rhoBins = np.linspace(0.2,1.8,11))

# Get density in 12 centres per unconstrained simulation, together with the 
# total density.
# These are spaced out so they don't intersect:
independentCentres = np.array([[0,0,0],[-boxsize/2,0,0],[0,-boxsize/2,0],\
    [0,0,-boxsize/2],[boxsize/4,boxsize/4,boxsize/4],\
    [-boxsize/4,boxsize/4,boxsize/4],[boxsize/4,-boxsize/4,boxsize/4],\
    [-boxsize/4,-boxsize/4,boxsize/4],[boxsize/4,boxsize/4,-boxsize/4],\
    [-boxsize/4,boxsize/4,-boxsize/4],[boxsize/4,-boxsize/4,-boxsize/4],\
    [-boxsize/4,-boxsize/4,-boxsize/4]])

centralAntihalosIndependent = [[tools.getAntiHalosInSphere(hcentres,135,\
    boxsize = boxsize,origin = centre) for centre in independentCentres] \
    for hcentres in ahCentresList]
centralAntihalosIndependentUn = [[tools.getAntiHalosInSphere(hcentres,135,\
    boxsize = boxsize,origin = centre) for centre in independentCentres] \
    for hcentres in ahCentresListUn]

#treeList = [tools.getKDTree(snap) for snap in snapList]
#treeListUn = [tools.getKDTree(snap) for snap in snapListUn]

remappedIndCentres = snapedit.wrap(independentCentres + boxsize/2,boxsize)

nbar = 512**3/(boxsize**3)
rSphere = 135

densityInSpheres = []
for k in range(0,len(snapList)):
    snap = pynbody.load(samplesFolder + "sample" + str(snapNumList[k]) + "/" + \
        "gadget_full_forward_512/snapshot_001")
    tree = tools.getKDTree(snap,cacheTree=False)
    gc.collect()
    densityInSpheres.append(tree.query_ball_point(remappedIndCentres,rSphere,\
        workers=-1,return_length=True)/(nbar*4*np.pi*rSphere**3/3) - 1.0)

densityInSpheresUn = []
for k in range(0,len(snapListUn)):
    snap = pynbody.load(unconstrainedFolderNew + "sample" + \
        str(snapNumListUncon[k]) + "/" + "gadget_full_forward_512/snapshot_001")
    tree = tools.getKDTree(snap,cacheTree=False)
    gc.collect()
    densityInSpheresUn.append(tree.query_ball_point(remappedIndCentres,rSphere,\
        workers=-1,return_length=True)/(nbar*4*np.pi*rSphere**3/3) - 1.0)

#densityInSpheres = [tree.query_ball_point(remappedIndCentres,rSphere,\
#        workers=-1,return_length=True)/(nbar*4*np.pi*rSphere**3/3) - 1.0 \
#        for tree in treeList]
#densityInSpheresUn = [tree.query_ball_point(remappedIndCentres,rSphere,\
#        workers=-1,return_length=True)/(nbar*4*np.pi*rSphere**3/3) - 1.0 \
#        for tree in treeListUn]

# Conditions for independent regions:
conditionListMrangeInd = [[(deltaCentralList[ns] < 0) & \
        (centralAntihalosIndependent[ns][k][1]) & \
        (antihaloMassesList[ns] > mMin) & (antihaloMassesList[ns] <= mMax) \
        for k in range(0,len(independentCentres))] \
        for ns in range(0,len(snapNumList))]
conditionListMrangeIndUn = [[(deltaCentralListUn[ns] < 0) & \
        (centralAntihalosIndependentUn[ns][k][1]) & \
        (antihaloMassesListUn[ns] > mMin) & (antihaloMassesListUn[ns] <= mMax) \
        for k in range(0,len(independentCentres))] \
        for ns in range(0,len(snapNumListUncon))]

# Density distributions in independent spheres:
#ahDensityListInd = [[getAntihaloDensityInShell(snapList[k],\
#    ahCentresListUnmapped[k],rMin = rMin,rMax = rMax,relative=True,\
#    ahRadii = antihaloRadii[k],ahFilter = conditionListMrangeInd[k][l]) \
#    for l in range(0,len(independentCentres))] \
#    for k in range(0,len(snapList))]

def getShellDensities(snapList,snapNumList,independentCentres,\
        antihaloCentresList,antihaloRadiiList,rMin,rMax,\
        filterList=None,snapFolder = "new_chain/unconstrained_samples/",\
        snapname = "gadget_full_forward_512/snapshot_001"):
    if filterList is None:
        filterList = [[None for l in range(0,len(independentCentres))] \
            for k in range(0,len(snapList))]
    ahDensityList = []
    for k in range(0,len(snapList)):
        snap = pynbody.load(snapFolder + "sample" + \
            str(snapNumList[k]) + "/" + snapname)
        gc.collect()
        ahDensityList.append([getAntihaloDensityInShell(snap,\
            antihaloCentresList[k],rMin = rMin,rMax = rMax,relative=True,\
            ahRadii = antihaloRadiiList[k],\
            ahFilter = filterList[k][l])  - 1.0 \
            for l in range(0,len(independentCentres))])
        gc.collect()
    return ahDensityList

recompute = False
ahDensityListIndUn = tools.loadOrRecompute("ah_density_lists_unconstrained.p",\
    getShellDensities,snapListUn,snapNumListUncon,independentCentres,\
    ahCentresListUnmappedUn,antihaloRadiiUn,rMin,rMax,\
    filterList = conditionListMrangeIndUn,\
    snapFolder = "new_chain/unconstrained_samples/",_recomputeData = recompute)

ahDensityListInd = tools.loadOrRecompute("ah_density_lists_constrained.p",\
    getShellDensities,snapList,snapNumList,independentCentres,\
    ahCentresListUnmapped,antihaloRadii,rMin,rMax,\
    filterList = conditionListMrangeInd,\
    snapFolder = "new_chain/",_recomputeData = recompute)


ahVoidRadiiInd = [[antihaloRadiiUn[k][conditionListMrangeIndUn[k][l]] \
    for l in range(0,len(independentCentres))] \
    for k in range(0,len(snapListUn))]

radiiListIndLin = np.hstack([np.hstack([rad for rad in ahVoidRadiiInd[k]]) \
    for k in range(0,len(snapListUn))])
ahDensityListIndLin = np.hstack([np.hstack(\
    [rad for rad in ahDensityListIndUn[k]]) \
    for k in range(0,len(snapListUn))])

#ahDensityListIndUn = [[getAntihaloDensityInShell(snapListUn[k],\
#    ahCentresListUnmappedUn[k],rMin = rMin,rMax = rMax,relative=True,\
#    ahRadii = antihaloRadii[k],ahFilter = conditionListMrangeIndUn[k][l]) \
#    for l in range(0,len(independentCentres))] \
#    for k in range(0,len(snapListUn))]

# Mean densities in independent spheres:
meanLargeRDensity = [[np.mean(ahDensityListInd[k][l]) \
    for l in range(0,len(independentCentres))] \
    for k in range(0,len(snapList))]
meanLargeRDensityUn = [[np.mean(ahDensityListIndUn[k][l]) \
    for l in range(0,len(independentCentres))] \
    for k in range(0,len(snapListUn))]

# plot of sphere density vs large-R profile density:
sphereDensities = np.reshape(densityInSpheresUn,\
    (len(independentCentres)*len(snapListUn)))
profileDensities = np.reshape(meanLargeRDensityUn,\
    (len(independentCentres)*len(snapListUn)))
fit = np.polyfit(sphereDensities,profileDensities,1)

plt.scatter(sphereDensities,profileDensities,marker='x',\
    color=seabornColormap[0])
plt.plot(minmax(sphereDensities),fit[0]*minmax(sphereDensities) + fit[1],\
    label='Fit, $\\delta_{\\mathrm{AH}} = ' + ("%.g2" % fit[0]) + \
    '\\delta_{\\mathrm{sphere}} ' + ("%.g2" % fit[1]) + '$')
plt.xlabel('Density constrast in $135\\mathrm{\\,Mpc}h^{-1}$ sphere, ' + \
    '$\\delta_{\\mathrm{sphere}}$')
plt.ylabel('Mean anti-halo density contrast, $' + str(rMin) + \
    ' < R/R_{\\mathrm{eff}} < ' + str(rMax) + ', \\delta_{\\mathrm{AH}}$')
plt.legend()
plt.savefig(figuresFolder + 'profile_vs_sphere_density.pdf')
plt.show()

constrainedLargeRDensity = np.array([np.mean(den[0]) \
    for den in ahDensityListInd])

# Histogram of the densities:
plt.hist(profileDensities,bins=np.linspace(-0.15,0.15,11),\
    color=seabornColormap[0],alpha=0.5)
plt.axvline(np.mean(profileDensities),linestyle=':',color='grey',\
    label='Mean unconstrained')
plt.axvline(np.mean(constrainedLargeRDensity),linestyle='--',color='k',\
    label='Mean constrained region')
plt.axvline(np.mean(ahDensityListUn[0])-1.0,linestyle='--',
    color=seabornColormap[0],label='Sample 1')
plt.axvline(np.mean(ahDensityListUn[1])-1.0,linestyle='--',\
    color=seabornColormap[1],label='Sample 2')
plt.xlabel('Mean anti-halo density contrast, $' + str(rMin) + \
    ' < R/R_{\\mathrm{eff}} < ' + str(rMax) + ', \\delta_{\\mathrm{AH}}$')
plt.ylabel('Number of $135\\mathrm{\\,Mpc}h^{-1}$ Regions')
plt.legend()
plt.show()



# Void Radii distribution:
conditionListMrange = [(deltaCentralList[ns] < 0) & \
    (centralAntihalosCon[ns][1]) & (antihaloMassesList[ns] > mMin) & \
    (antihaloMassesList[ns] <= mMax) \
    for ns in range(0,len(snapNumList))]
conditionListMrangeUn = [(deltaCentralListUn[ns] < 0) & \
    (centralAntihalosUn[ns][1]) & (antihaloMassesListUn[ns] > mMin) & \
    (antihaloMassesListUn[ns] <= mMax)
    for ns in range(0,len(snapNumListUncon))]
stackedRadii = np.hstack(antihaloRadii)
stackedRadiiUn = np.hstack(antihaloRadiiUn)
stackedConditionsM = np.hstack(conditionListMrange)
stackedConditionsMUn = np.hstack(conditionListMrangeUn)

plt.hist(stackedRadiiUn[stackedConditionsMUn],bins=np.linspace(0,35,21),\
    color=seabornColormap[0],alpha=0.5,density=True,label='Unconstrained radii')
plt.hist(stackedRadii[stackedConditionsM],bins=np.linspace(0,35,21),\
    color=seabornColormap[1],alpha=0.5,density=True,label='Constrained radii')
plt.xlabel('Void radii ($\\mathrm{Mpc}h^{-1}$)')
plt.ylabel('Fraction of voids')
plt.legend()
plt.show()




# STACKED VOIDS FROM ALL OVER UNCONSTRAINED SIMULATIONS:
stackedConditionsAllUn = np.hstack([(deltaCentralListUn[ns] < 0) \
    for ns in range(0,len(snapNumListUncon))])
rBins = ahProps[0][8]
filterUn = np.where((stackedRadiiUn > 5) & (stackedRadiiUn < 25) & \
        stackedConditionsAllUn & (stackedMassesUn > mMin) & \
        (stackedMassesUn <= mMax))[0]

profVariance = np.std(np.vstack(pairCountsListUn)[filterUn]/\
    np.vstack(volumesListUn)[filterUn],0)

[nbarjAllStackedFullVolUn,sigmaAllStackedFullVolUn] = \
    stacking.stackVoidsWithFilter(\
        np.vstack(ahCentresListUn),stackedRadiiUn,\
        filterUn,snapListUn[0],rBins,\
        nPairsList = np.vstack(pairCountsListUn),\
        volumesList = np.vstack(volumesListUn),\
        method="poisson",errorType="Weighted")

[nbarjAllStackedFullVolUn2,sigmaAllStackedFullVolUn2] = \
    stacking.stackVoidsWithFilter(\
        np.vstack(ahCentresListUn),stackedRadiiUn,\
        np.where((stackedRadiiUn > 5) & (stackedRadiiUn < 25) & \
        stackedConditionsAllUn & (stackedMassesUn > mMin) & \
        (stackedMassesUn <= mMax))[0],snapListUn[0],rBins,\
        nPairsList = np.vstack(pairCountsListUn),\
        volumesList = np.vstack(volumesListUn),\
        method="naive",errorType="Profile")

# Separate stack for the independent regions:
# Get all anti-halos that lie within one of the 12 independent spheres in each
# simulation:
conditionListIndUn = [np.any(np.array([boolList[1] \
    for boolList in centralAntihalosIndependentUn[k]]),axis=0) \
    for k in range(0,len(snapListUn))]

[nbarjSepStackIndUn,sigmaSepStackIndUn] = stacking.computeMeanStacks(\
        ahCentresListUn,antihaloRadiiUn,antihaloMassesListUn,\
        conditionListIndUn,pairCountsListUn,volumesListUn,snapListUn,\
        nbar,rBins,5,25,mMin,mMax)

indStack = []
for l in range(0,len(independentCentres)):
    condition = [centralAntihalosIndependentUn[k][l][1] \
        for k in range(0,len(snapListUn))]
    indStack.append(stacking.computeMeanStacks(\
        ahCentresListUn,antihaloRadiiUn,antihaloMassesListUn,\
        condition,pairCountsListUn,volumesListUn,snapListUn,\
        nbar,rBins,5,25,mMin,mMax))

nbarjSepStackIndUn = np.vstack([sim[0] for sim in indStack])
sigmaSepStackIndUn = np.vstack([sim[1] for sim in indStack])

[nbarjSepStackIndUn,sigmaSepStackIndUn]



#-------------------------------------------------------------------------------
# ANTIHALO CATALOGUE CONSTRUCTION

snapname = "gadget_full_forward_512/snapshot_001"
snapnameRev = "gadget_full_reverse_512/snapshot_001"
#snapNumList = [7000,7200,7400,7600,8000]
#snapNumList = [7000,7200,7400,7600,8000]

#snapNumList = [8800,9100,9400,9700,10000]

snapNumList = [7000,7200,7400,7600,8000,8800,9100,9400,9700,10000]

snapList =  [pynbody.load(samplesFolder + "sample" + str(snapNum) + "/" + \
    snapname) for snapNum in snapNumList]
snapListRev =  [pynbody.load(samplesFolder + "sample" + str(snapNum) + "/" \
    + snapnameRev) for snapNum in snapNumList]
hrList = [snap.halos() for snap in snapListRev]
boxsize = snapList[0].properties['boxsize'].ratio("Mpc a h**-1")
rSphere=300
max_index = 17000
#rSphere = 135
#max_index = 1500
rMin=5
rMax = 30

snapSortList = [np.argsort(snap['iord']) for snap in snapList]

ahProps = [pickle.load(\
            open(snap.filename + ".AHproperties.p","rb")) \
            for snap in snapList]
antihaloCentres = [tools.remapAntiHaloCentre(props[5],boxsize) \
    for props in ahProps]
antihaloMasses = [props[3] for props in ahProps]
vorVols = [props[4] for props in ahProps]
antihaloRadii = [props[7] for props in ahProps]
centralAntihalos = [tools.getAntiHalosInSphere(antihaloCentres[k],rSphere,\
        filterCondition = (antihaloRadii[k] > rMin) & \
        (antihaloRadii[k] <= rMax)) for k in range(0,len(snapNumList))]
centralAntihaloMasses = [\
        antihaloMasses[k][centralAntihalos[k][0]] \
        for k in range(0,len(centralAntihalos))]
sortedList = [np.flip(np.argsort(centralAntihaloMasses[k])) \
        for k in range(0,len(snapNumList))]
ahCounts = np.array([len(cahs[0]) for cahs in centralAntihalos])

centresListShort = [np.array([antihaloCentres[l][\
        centralAntihalos[l][0][sortedList[l][k]],:] \
        for k in range(0,np.min([ahCounts[l],max_index]))]) \
        for l in range(0,len(snapNumList))]

radiiListShort = [np.array([antihaloRadii[l][\
        centralAntihalos[l][0][sortedList[l][k]]] \
        for k in range(0,np.min([ahCounts[l],max_index]))]) \
        for l in range(0,len(snapNumList))]

massListShort = [np.array([antihaloMasses[l][\
        centralAntihalos[l][0][sortedList[l][k]]] \
        for k in range(0,np.min([ahCounts[l],max_index]))]) \
        for l in range(0,len(snapNumList))]

# Using a distance criteria (greatest matched ratio within 20 Mpc)
[finalCat,shortHaloList,twoWayMatchList,finalCandidates,\
    finalRatios,finalDistances,allCandidates,candidateCounts] = \
    constructAntihaloCatalogue(snapNumList,snapList=snapList,\
    snapListRev=snapListRev,ahProps=ahProps,hrList=hrList,max_index=max_index,\
    twoWayOnly=True,blockDuplicates=True)

[finalCatRatio,shortHaloListRatio,twoWayMatchListRatio,finalCandidatesRatio,\
    finalRatiosRatio,finalDistancesRatio] = \
    constructAntihaloCatalogue(snapNumList,snapList=snapList,\
    snapListRev=snapListRev,ahProps=ahProps,hrList=hrList,\
    sortMethod="ratio")

[finalCatDist,shortHaloListDist,twoWayMatchListDist,finalCandidatesDist,\
    finalRatiosDist,finalDistancesDist] = \
    constructAntihaloCatalogue(snapNumList,snapList=snapList,\
    snapListRev=snapListRev,ahProps=ahProps,hrList=hrList,\
    sortMethod="distance")

# Statistics on performance:
confusionRate = np.array([np.sum([len(x) for x in cand])/len(cand) \
    for cand in finalCandidates])

def getDistanceMatrix(ahIndices,centresList,lowerTriangleOnly=False):
    centres = [centresList[k][ahIndices[k]-1] \
        for k in range(0,len(ahIndices))]
    matrix = np.zeros((len(ahIndices),len(ahIndices)))
    for k in range(0,len(ahIndices)):
        for l in range(0,len(ahIndices)):
            if l > k and lowerTriangleOnly:
                continue
            if l == k:
                continue
            if ahIndices[k] > 0 and ahIndices[l] > 0:
                matrix[k,l] = np.sqrt(np.sum((centres[k] - centres[l])**2))
            else:
                matrix[k,l] = np.nan
    return matrix

def getBestCandidateMeanDistance(catalogue,centresList):
    meanDistances = np.zeros(len(catalogue))
    if len(catalogue) == 0:
        return np.array([])
    lowerTriInd = np.tril_indices(len(catalogue[0]),k=-1)
    for k in range(0,len(catalogue)):
        matrix = getDistanceMatrix(catalogue[k],centresList,\
            lowerTriangleOnly=True)
        meanDistances[k] = np.nanmean(matrix[lowerTriInd])
    return meanDistances

# Ratio of two radii, smallest/biggest
def getRadiusRatio(radii1,radii2):
    if np.isscalar(radii1):
        if np.isscalar(radii2):
            if radii2 > radii1:
                return radii1/radii2
            else:
                return radii2/radii1
        else:
            returnArr = radii2/radii1
            bigger = np.where(radii2 > radii1)[0]
            returnArr[bigger] = radii1/radii2[bigger]
            return returnArr
    else:
        if np.isscalar(radii2):
            returnArr = radii1/radii2
            smaller = np.where(radii2 < radii1)[0]
            returnArr[smaller] = radii2/radii1[smaller]
            return returnArr
        else:
            returnArr = radii2/radii1
            bigger = np.where(radii2 > radii1)[0]
            returnArr[bigger] = radii1[bigger]/radii2[bigger]
            return returnArr

def getRatioMatrix(ahIndices,radiiList,lowerTriangleOnly = False):
    radii = [radiiList[:,k][ahIndices[k]-1] \
        for k in range(0,len(ahIndices))]
    matrix = np.ones((len(ahIndices),len(ahIndices)))
    for k in range(0,len(ahIndices)):
        for l in range(0,len(ahIndices)):
            if l > k and lowerTriangleOnly:
                continue
            if l == k:
                continue
            if ahIndices[k] > 0 and ahIndices[l] > 0:
                matrix[k,l] = getRadiusRatio(radii[k],radii[l])
            else:
                matrix[k,l] = np.nan
    return matrix

def getBestCandidateMeanRatio(catalogue,radiiList):
    meanRatios = np.zeros(len(catalogue))
    if len(catalogue) == 0:
        return np.array([])
    lowerTriInd = np.tril_indices(len(catalogue[0]),k=-1)
    for k in range(0,len(catalogue)):
        matrix = getRatioMatrix(catalogue[k],radiiList,lowerTriangleOnly=True)
        # Exclude the diagonals:
        meanRatios[k] = np.nanmean(matrix[lowerTriInd])
    return meanRatios

# Get statistics on performance for different parameters:
def specialNanMin(x):
    if len(x) == 0:
        return np.nan
    else:
        return np.nanmin(x)

def specialNanMax(x):
    if len(x) == 0:
        return np.nan
    else:
        return np.nanmax(x)


def getRadiiFromCat(catList,radiiList):
    radiiListOut = -np.ones(catList.shape,dtype=float)
    for k in range(0,len(catList)):
        for l in range(0,len(catList[0])):
            if catList[k,l] > 0:
                radiiListOut[k,l] = radiiList[l][catList[k,l]-1]
    return radiiListOut

def getCentresFromCat(catList,centresList,ns):
    centresListOut = np.zeros((len(catList),3),dtype=float)
    for k in range(0,len(catList)):
        if catList[k,ns] > 0:
            centresListOut[k,:] = centresList[ns][catList[k,ns]-1]
        else:
            centresListOut[k,:] = np.nan
    return centresListOut

def getMeanProperty(propertyList,stripNaN=True,lowerLimit=0,stdError=True):
    meanProperty = np.zeros(len(propertyList))
    sigmaProperty = np.zeros(len(propertyList))
    for k in range(0,len(propertyList)):
        condition = propertyList[k,:] > lowerLimit
        if stripNaN:
            condition = condition & (np.isfinite(propertyList[k,:] > 0))
        haveProperty = np.where(condition)[0]
        meanProperty[k] = np.mean(propertyList[k,haveProperty])
        sigmaProperty[k] = np.std(propertyList[k,haveProperty])
        if stdError:
            sigmaProperty[k] /= np.sqrt(len(haveProperty))
    return [meanProperty,sigmaProperty]



thresholdArr = np.linspace(0.0,1.0,51)
distArr = np.arange(0,21,2.5)[1:]
# Mean number of candidates per other catalogue:
confusionMatrix = np.zeros((len(thresholdArr),len(distArr)))
# Number of antihalos in the catalogue:
lengthMatrix = np.zeros((len(thresholdArr),len(distArr)))
# Mean distances between possible candidates:
distMatrix = np.zeros((len(thresholdArr),len(distArr)))
# Distance to closest candidate:
minDistMatrix = np.zeros((len(thresholdArr),len(distArr)))
# Distance to furthest candidate:
maxDistMatrix = np.zeros((len(thresholdArr),len(distArr)))
# mean distance to the best candidate (depends on the criteria):
meanBestDistMatrix = np.zeros((len(thresholdArr),len(distArr)))
# Mean radius-ratio for the best candidate (depends on the criteria):
meanBestRatioMatrix = np.zeros((len(thresholdArr),len(distArr)))
# Mean radius ratio for candidates:
ratioMatrix = np.zeros((len(thresholdArr),len(distArr)))
# Minimum radius ratio for candidates:
minRatioMatrix = np.zeros((len(thresholdArr),len(distArr)))
# Maximum radius ratio for candidates:
maxRatioMatrix = np.zeros((len(thresholdArr),len(distArr)))
# Fraction which found a unique match:
uniqueMatchFracMatrix = np.zeros((len(thresholdArr),len(distArr)))
# Fraction which found no match:
noMatchFracMatrix = np.zeros((len(thresholdArr),len(distArr)))
# Fraction of matches that go both ways:
twoWayMatchFracMatrix = np.zeros((len(thresholdArr),len(distArr)))
# Average fraction of catalogues that anti-halos appear in:
catFractionMatrix = np.zeros((len(thresholdArr),len(distArr)))
# Average fraction of catalogues that final catalogue anti-halos appear in:
catFractionFinalMatrix = np.zeros((len(thresholdArr),len(distArr)))
# Method used to select the best candidate:
sortMethod = "ratio"
print("Scanning through parameters...")
diffMap = [np.setdiff1d(np.arange(0,len(snapNumList)),[k]) \
    for k in range(0,len(snapNumList))]
for k in range(0,len(thresholdArr)):
    for l in range(0,len(distArr)):
        [finalCatTest,shortHaloListTest,twoWayMatchListTest,\
            finalCandidatesTest,finalRatiosTest,finalDistancesTest,\
            allCandidatesTest,candidateCountsTest] = \
            constructAntihaloCatalogue(snapNumList,snapList=snapList,\
            snapListRev=snapListRev,ahProps=ahProps,hrList=hrList,\
            crossMatchThreshold = thresholdArr[k],distMax = distArr[l],\
            verbose=False,max_index=max_index,sortMethod=sortMethod)
        candCountStripped = np.array([candidateCountsTest[k][:,diffMap[k]] \
            for k in range(0,len(snapNumList))])
        #confusionRateTest = np.array(\
        #    [np.sum([len(x) for x in cand])/len(cand) \
        #    for cand in finalCandidatesTest])
        #candCountArr = np.array([[len(x) for x in cand] \
        #    for cand in finalCandidatesTest])
        #catFractionTest = np.array([len(np.where(arr > 0)[0]) \
        #    for arr in finalCatTest])/len(snapNumList)
        distTest =  np.array(\
            [np.nanmean(np.hstack([x for x in dist])) \
            for dist in finalDistancesTest])
        minDistTest = np.array(\
            [specialNanMin(np.hstack([x for x in dist])) \
            for dist in finalDistancesTest])
        maxDistTest = np.array(\
            [specialNanMax(np.hstack([x for x in dist])) \
            for dist in finalDistancesTest])
        ratioTest =  np.array(\
            [np.nanmean(np.hstack([x for x in ratios])) \
            for ratios in finalRatiosTest])
        minRatioTest =  np.array(\
            [specialNanMin(np.hstack([x for x in ratios])) \
            for ratios in finalRatiosTest])
        maxRatioTest =  np.array(\
            [specialNanMax(np.hstack([x for x in ratios])) \
            for ratios in finalRatiosTest])
        meanDistancesTest = getBestCandidateMeanDistance(finalCatTest,\
            centresListShort)
        meanRatiosTest = getBestCandidateMeanRatio(finalCatTest,radiiListShort)
        meanBestDistMatrix[k,l] = np.nanmean(meanDistancesTest)
        meanBestRatioMatrix[k,l] = np.nanmean(meanRatiosTest)
        confusionMatrix[k,l] = np.mean(candCountStripped)
        lengthMatrix[k,l] = len(finalCatTest)
        distMatrix[k,l] = np.nanmean(distTest)
        ratioMatrix[k,l] = np.nanmean(ratioTest)
        minDistMatrix[k,l] = np.nanmean(minDistTest)
        maxDistMatrix[k,l] = np.nanmean(maxDistTest)
        minRatioMatrix[k,l] = np.nanmean(minRatioTest)
        maxRatioMatrix[k,l] = np.nanmean(maxRatioTest)
        catFractionMatrix[k,l] = np.nanmean(\
            np.sum(np.array(candidateCountsTest,dtype=bool),2)/len(snapNumList))
        if len(finalCatTest) > 0:
            catFractionFinalMatrix[k,l] = np.mean(\
                np.sum(finalCatTest > 0,1)/len(snapNumList))
        uniqueMatchFracMatrix[k,l] = np.mean(\
            [[len(np.where(counts[:,k] == 1)[0]) \
            for k in range(0,len(snapNumList)-1)] \
            for counts in candCountStripped])/max_index
        noMatchFracMatrix[k,l] = np.mean(\
            [[len(np.where(counts[:,k] == 0)[0]) \
            for k in range(0,len(snapNumList)-1)] \
            for counts in candCountStripped])/max_index
        multiMatchFracMatrix[k,l] = np.mean(\
            [[len(np.where(counts[:,k] > 1)[0]) \
            for k in range(0,len(snapNumList)-1)] \
            for counts in candCountStripped])/max_index
        twoWayMatchFracMatrix[k,l] = np.mean([[np.sum(twoWayVec[:,k]) \
            for k in range(0,len(snapNumList)-1)] \
            for twoWayVec in twoWayMatchListTest])/max_index
        #if len(finalCatTest) > 0:
            # Fraction of unique matches:
            #uniqueMatchFracMatrix[k,l] = np.mean(\
            #    [len(np.where(candCountArr[:,m] == 1)[0])/\
            #    np.max([1,len(finalCandidatesTest)]) \
            #    for m in range(0,len(snapNumList)-1)])
            #noMatchFracMatrix[k,l] = np.mean(\
            #    [len(np.where(candCountArr[:,m] == 0)[0])/\
            #    np.max([1,len(finalCandidatesTest)]) \
            #    for m in range(0,len(snapNumList)-1)])
            # Get whether matches found from each catalogue are two way:
            #twoWay = [twoWayMatchListTest[m][\
            #    finalCatTest[:,m][np.where(finalCatTest[:,m] > 0)]-1] \
            #    for m in range(0,len(snapNumList))]
            # There are Ncat*(Ncat-1) columns of matches (forward
            # and reverse for each Ncat*(Ncat-1)/2 pairs). Average over all
            # to get the fraction of matches that are 2-way:
            #twoWayMatchFracMatrix[k,l] = np.sum([np.sum(tw) for tw in twoWay])/\
            #    (len(snapNumList)*(len(snapNumList)-1)*len(finalCatTest))
        #else:
            #twoWay = np.array([[] for m in range(0,len(snapNumList))])
            #twoWayMatchFracMatrix[k,l] = 0.0
            #uniqueMatchFracMatrix[k,l] = 0.0
    print(("%.3g" % (100*(k+1)/len(thresholdArr))) + "% complete")


# Plot Results:
plt.imshow(confusionMatrix,cmap='PuOr_r',origin='lower',\
    extent=(np.min(distArr),np.max(distArr),\
    np.min(thresholdArr),np.max(thresholdArr)),aspect='auto',\
    vmin=0.5,vmax=+1.5)
plt.colorbar()
plt.xlabel('Search radius ($\mathrm{Mpc}h^{-1}$)')
plt.ylabel('Radius ratio threshold')
plt.title('Mean number of candidates per other catalogue')
plt.show()

# Showing the fraction:
plt.plot(thresholdArr,confusionMatrix,\
    label=[("%.3g" % dist) + "$\\mathrm{\\,Mpc}h^{-1}$" for dist in distArr])
plt.xlabel('Radius ratio threshold')
plt.ylabel('Mean number of candidates per other catalogue')
plt.legend()
if sortMethod == "ratio":
    plt.savefig(figuresFolder + \
        "supporting_plots/mean_candidates_found_radius_ratio.pdf")
elif sortMethod == "distance":
    plt.savefig(figuresFolder + "supporting_plots/mean_candidates_found_distance.pdf")

plt.show()


# Showing the mean distance to candidates:
plt.plot(thresholdArr,distMatrix,\
    label=[("%.3g" % dist) + "$\\mathrm{\\,Mpc}h^{-1}$" for dist in distArr])
plt.xlabel('Radius ratio threshold')
plt.ylabel('Mean distance to candidates ($\\mathrm{Mpc}h^{-1}$)')
plt.legend()
if sortMethod == "ratio":
    plt.savefig(figuresFolder + \
        "supporting_plots/mean_distance_radius_ratio.pdf")
elif sortMethod == "distance":
    plt.savefig(figuresFolder + "supporting_plots/mean_distance_distance.pdf")

plt.show()

# Showing the minimum distance to candidates:
plt.plot(thresholdArr,minDistMatrix,\
    label=[("%.3g" % dist) + "$\\mathrm{\\,Mpc}h^{-1}$" for dist in distArr])
plt.xlabel('Radius ratio threshold')
plt.ylabel('Distance to closest candidate ($\\mathrm{Mpc}h^{-1}$)')
plt.legend()
if sortMethod == "ratio":
    plt.savefig(figuresFolder + \
        "supporting_plots/min_distance_radius_ratio.pdf")
elif sortMethod == "distance":
    plt.savefig(figuresFolder + "supporting_plots/min_distance_distance.pdf")

plt.show()

# Showing the maximum ratio for candidates:
plt.plot(thresholdArr,maxRatioMatrix,\
    label=[("%.3g" % dist) + "$\\mathrm{\\,Mpc}h^{-1}$" for dist in distArr])
plt.xlabel('Radius ratio threshold')
plt.ylabel('Highest radius-ratio among candidates')
plt.legend()
if sortMethod == "ratio":
    plt.savefig(figuresFolder + \
        "supporting_plots/max_radius_ratio_radius_ratio.pdf")
elif sortMethod == "distance":
    plt.savefig(figuresFolder + "supporting_plots/max_radius_ratio_distance.pdf")

plt.show()

# Showing the mean ratio for best candidates:
plt.plot(thresholdArr,meanBestRatioMatrix,\
    label=[("%.3g" % dist) + "$\\mathrm{\\,Mpc}h^{-1}$" for dist in distArr])
plt.xlabel('Radius ratio threshold')
plt.ylabel('Mean radius-ratio for best candidate')
plt.legend()
if sortMethod == "ratio":
    plt.title('Greatest radius-ratio as "best" candidate')
    plt.savefig(figuresFolder + \
        "supporting_plots/mean_best_ratio_radius_ratio.pdf")
elif sortMethod == "distance":
    plt.title('Closest as as "best" candidate')
    plt.savefig(figuresFolder + \
        "supporting_plots/mean_best_radius_ratio_distance.pdf")

plt.show()

# Showing the mean distance to the best candidates:
plt.plot(thresholdArr,meanBestDistMatrix,\
    label=[("%.3g" % dist) + "$\\mathrm{\\,Mpc}h^{-1}$" for dist in distArr])
plt.xlabel('Radius ratio threshold')
plt.ylabel('Mean distance to best candidate ($\\mathrm{Mpc}h^{-1}$)')
plt.legend(loc = 'upper left')
if sortMethod == "ratio":
    plt.title('Greatest radius-ratio as "best" candidate')
    plt.savefig(figuresFolder + \
        "supporting_plots/mean_best_distance_radius_ratio.pdf")
elif sortMethod == "distance":
    plt.title('Closest as as "best" candidate')
    plt.savefig(figuresFolder + \
        "supporting_plots/mean_best_distance_ratio_distance.pdf")

plt.show()


# Showing the two-way match fraction:
plt.plot(thresholdArr,twoWayMatchFracMatrix,\
    label=[("%.3g" % dist) + "$\\mathrm{\\,Mpc}h^{-1}$" for dist in distArr])
plt.xlabel('Radius ratio threshold')
plt.ylabel('Fraction of two-way matches')
plt.legend(loc = 'lower left')
if sortMethod == "ratio":
    plt.savefig(figuresFolder + \
        "supporting_plots/two_way_match_radius_ratio.pdf")
elif sortMethod == "distance":
    plt.savefig(figuresFolder + \
        "supporting_plots/two_way_match_distance.pdf")

plt.show()

# Showing the unique match fraction:
plt.plot(thresholdArr[0:-1],uniqueMatchFracMatrix[0:-1,:],\
    label=[("%.3g" % dist) + "$\\mathrm{\\,Mpc}h^{-1}$" for dist in distArr])
plt.xlabel('Radius ratio threshold')
plt.ylabel('Fraction of unique matches')
plt.legend(loc = 'lower left')
if sortMethod == "ratio":
    plt.savefig(figuresFolder + \
        "supporting_plots/unique_match_radius_ratio.pdf")
elif sortMethod == "distance":
    plt.savefig(figuresFolder + \
        "supporting_plots/unique_match_distance.pdf")

plt.show()


# Showing the no match fraction:
plt.plot(thresholdArr[0:-1],noMatchFracMatrix[0:-1,:],\
    label=[("%.3g" % dist) + "$\\mathrm{\\,Mpc}h^{-1}$" for dist in distArr])
plt.xlabel('Radius ratio threshold')
plt.ylabel('Fraction of failed matches')
plt.legend(loc = 'lower left')
if sortMethod == "ratio":
    plt.savefig(figuresFolder + \
        "supporting_plots/failed_match_radius_ratio.pdf")
elif sortMethod == "distance":
    plt.savefig(figuresFolder + \
        "supporting_plots/failed_match_distance.pdf")

plt.show()

# Showing the ambiguous match fraction:
plt.plot(thresholdArr[0:-1],multiMatchFracMatrix[0:-1,:],\
    label=[("%.3g" % dist) + "$\\mathrm{\\,Mpc}h^{-1}$" for dist in distArr])
plt.xlabel('Radius ratio threshold')
plt.ylabel('Fraction of ambiguous matches')
plt.legend(loc = 'lower left')
if sortMethod == "ratio":
    plt.savefig(figuresFolder + \
        "supporting_plots/multi_match_radius_ratio.pdf")
elif sortMethod == "distance":
    plt.savefig(figuresFolder + \
        "supporting_plots/multi_match_distance.pdf")

plt.show()

# Showing the average catalogue fraction:
plt.plot(thresholdArr[0:-1],catFractionMatrix[0:-1,:],\
    label=[("%.3g" % dist) + "$\\mathrm{\\,Mpc}h^{-1}$" for dist in distArr])
plt.xlabel('Radius ratio threshold')
plt.ylabel('Mean catalogue fraction')
plt.legend(loc = 'lower left')
if sortMethod == "ratio":
    plt.savefig(figuresFolder + \
        "supporting_plots/cat_fraction_radius_ratio.pdf")
elif sortMethod == "distance":
    plt.savefig(figuresFolder + \
        "supporting_plots/cat_fraction_distance.pdf")

plt.show()

# Showing the final average catalogue fraction:
plt.plot(thresholdArr[0:-1],catFractionFinalMatrix[0:-1,:],\
    label=[("%.3g" % dist) + "$\\mathrm{\\,Mpc}h^{-1}$" for dist in distArr])
plt.xlabel('Radius ratio threshold')
plt.ylabel('Mean final catalogue fraction')
plt.legend(loc = 'lower left')
if sortMethod == "ratio":
    plt.savefig(figuresFolder + \
        "supporting_plots/cat_fraction_final_radius_ratio.pdf")
elif sortMethod == "distance":
    plt.savefig(figuresFolder + \
        "supporting_plots/cat_fraction_final_distance.pdf")

plt.show()




# Overlap map:
from void_analysis.tools import minmax
def overlapMap(cat1,cat2,volumes1,volumes2,checkFirst = False,verbose=False):
    overlap = np.zeros((len(cat1),len(cat2)))
    vol1 = np.array([np.sum(volumes1[halo['iord']]) for halo in cat1])
    vol2 = np.array([np.sum(volumes2[halo['iord']]) for halo in cat2])
    if checkFirst:
        for k in range(0,len(cat1)):
            for l in range(0,len(cat2)):
                if checkOverlap(cat1[k+1]['iord'],cat2[l+1]['iord']):
                    intersection = np.intersect1d(cat1[k+1]['iord'],\
                        cat2[l+1]['iord'])
                    overlap[k,l] = np.sum(\
                        volumes1[intersection])/np.sqrt(vol1[k]*vol2[l])
    else:
        for k in range(0,len(cat1)):
            for l in range(0,len(cat2)):
                intersection = np.intersect1d(cat1[k+1]['iord'],\
                    cat2[l+1]['iord'])
                if len(intersection) > 0:
                    overlap[k,l] = np.sum(\
                        volumes1[intersection])/np.sqrt(vol1[k]*vol2[l])
            if verbose:
                print(("%.3g" % (100*(k*len(cat1) + l + 1)/\
                    (len(cat1)*len(cat2)))) + "% complete")
    return overlap

# Check whether two halos have any overlap:
def checkOverlap(list1,list2):
    [min1,max1] = minmax(list1)
    [min2,max2] = minmax(list2)
    if max1 < min2:
        return False
    elif min1 > max2:
        return False
    else:
        if len(list1) < len(list2):
            listMin = list1
            listMax = list2
        else:
            listMin = list2
            listMax = list1
        for k in range(0,len(listMin)):
            if np.isin(listMin[k],listMax):
                return True
        return False

def generateOverlapList(snapNumList,hrListCentral,volumesList):
    overlapList = []
    for k in range(0,len(snapNumList)):
        for l in range(0,len(snapNumList)):
            if k >= l:
                continue
            overlapList.append(overlapMap(hrListCentral[k],\
                hrListCentral[l],volumesList[k],volumesList[l],\
                verbose=False))
    return overlapList



overlapList = tools.loadOrRecompute("overlap_list.p",generateOverlapList,\
    snapNumList,hrListCentral,volumesList,recompute=False)

if len(overlapList) != int(len(snapNumList)*(len(snapNumList) - 1)/2):
    raise Exception("Invalid overlapList!")


# Optimising purity and completeness:

# Finding the optimal radius:
#threshList = np.array([0.0,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
#threshList = np.array([0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9])
#threshList = np.array([0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9])
#threshList = np.arange(0.1,1,0.05)
threshList = np.arange(0.5,1,0.05)
#distArr2 = np.arange(0,21,0.2)[1:]
distArr2 = np.arange(0,3,0.1)
massListBounds = np.array([1e13,1e14,5e14,1e16])
sortMethod='ratio'
uniqueMatchFracArray = np.zeros((len(threshList),len(distArr2),\
    len(massListBounds),2))
catFractionFinalMatrix = np.zeros((len(threshList),len(distArr2),\
    len(massListBounds),2))
purity_1f = np.zeros((len(threshList),len(distArr2),\
    len(massListBounds),2))
purity_2f = np.zeros((len(threshList),len(distArr2),\
    len(massListBounds),2))
completeness_1f = np.zeros((len(threshList),len(distArr2),\
    len(massListBounds),2))
completeness_2f = np.zeros((len(threshList),len(distArr2),\
    len(massListBounds),2))

massRange = [1e14,5e14]
if massRange is None:
    centralAntihalosTest = [tools.getAntiHalosInSphere(antihaloCentres[k],135,\
            filterCondition = (antihaloRadii[k] > rMin) & \
            (antihaloRadii[k] <= rMax)) for k in range(0,len(snapNumList))]
else:
    centralAntihalosTest = [tools.getAntiHalosInSphere(antihaloCentres[k],135,\
            filterCondition = (antihaloRadii[k] > rMin) & \
            (antihaloRadii[k] <= rMax) & (antihaloMasses[k] > massRange[0]) & \
            (antihaloMasses[k] <= massRange[1])) \
            for k in range(0,len(snapNumList))]

centralAntihaloMassesTest = [\
        antihaloMasses[k][centralAntihalosTest[k][0]] \
        for k in range(0,len(centralAntihalosTest))]
sortedListTest = [np.flip(np.argsort(centralAntihaloMassesTest[k])) \
        for k in range(0,len(snapNumList))]
ahCountsTest = np.array([len(cahs[0]) for cahs in centralAntihalosTest])
massListShortTest = [np.array([antihaloMasses[l][\
            centralAntihalosTest[l][0][sortedListTest[l][k]]] \
            for k in range(0,np.min([ahCountsTest[l],max_index]))]) \
            for l in range(0,len(snapNumList))]
diffMap = [np.setdiff1d(np.arange(0,len(snapNumList)),[k]) \
    for k in range(0,len(snapNumList))]
fixedMassThresh = 0.3
fixedRadThresh = 0.3
for k in range(0,len(threshList)):
    for q in range(0,2):
        #crossMatchQuantity = "both"
        sortMethod = "ratio"
        overlaps = None
        if q == 0:
            # Fixed mass threshold:
            #crossMatchThreshold = np.array([threshList[k],fixedMassThresh])
            crossMatchQuantity = 'radius'
            crossMatchThreshold = threshList[k]
        if q == 1:
            # Fixed radius threshold:
            crossMatchQuantity = 'mass'
            crossMatchThreshold = threshList[k]
        for l in range(0,len(distArr2)):
            #[finalCatTest,shortHaloListTest,twoWayMatchListTest,\
            #    finalCandidatesTest,finalRatiosTest,finalDistancesTest,\
            #    allCandidatesTest,candidateCountsTest] = \
            #    constructAntihaloCatalogue(snapNumList,snapList=snapList,\
            #    snapListRev=snapListRev,ahProps=ahProps,hrList=hrList,\
            #    crossMatchThreshold = threshList[k],distMax = distArr2[l],\
            #    verbose=False,max_index=max_index,sortMethod=sortMethod)
            massRange = [0,1e16]
            [finalCatTest,shortHaloListTest,twoWayMatchListTest,\
                finalCandidatesTest,finalRatiosTest,finalDistancesTest,\
                allCandidatesTest,candidateCountsTest] = \
                    constructAntihaloCatalogue(snapNumList,snapList=snapList,\
                    snapListRev=snapListRev,ahProps=ahProps,hrList=hrList,\
                    max_index=None,twoWayOnly=True,blockDuplicates=True,\
                    #crossMatchThreshold = np.array([0.7,threshList[k]]),\
                    crossMatchThreshold = crossMatchThreshold,\
                    #crossMatchThreshold = threshList[k],\
                    #crossMatchQuantity = "radius",\
                    crossMatchQuantity = crossMatchQuantity,\
                    massRange = massRange,\
                    distMax = distArr2[l],\
                    rSphere=135,verbose=False,sortMethod=sortMethod,\
                    snapSortList=snapSortList,overlapList = overlaps)
            centralAntihalosTest = [tools.getAntiHalosInSphere(\
                antihaloCentres[k],135,\
                filterCondition = (antihaloRadii[k] > rMin) & \
                (antihaloRadii[k] <= rMax) & \
                (antihaloMasses[k] > massRange[0]) & \
                (antihaloMasses[k] <= massRange[1])) \
                for k in range(0,len(snapNumList))]
            centralAntihaloMassesTest = [\
                    antihaloMasses[k][centralAntihalosTest[k][0]] \
                    for k in range(0,len(centralAntihalosTest))]
            sortedListTest = [\
                np.flip(np.argsort(centralAntihaloMassesTest[k])) \
                for k in range(0,len(snapNumList))]
            ahCountsTest = np.array([len(cahs[0]) \
                for cahs in centralAntihalosTest])
            massListShortTest = [np.array([antihaloMasses[l][\
                centralAntihalosTest[l][0][sortedListTest[l][k]]] \
                for k in range(0,np.min([ahCountsTest[l],max_index]))]) \
                for l in range(0,len(snapNumList))]
            massFilter = [[(massListShortTest[n] > massListBounds[m]) & \
                (massListShortTest[n] <= massListBounds[m+1]) \
                for m in range(0,len(massListBounds)-1)] \
                for n in range(0,len(massListShortTest))]
            purity_1f[k,l,len(massListBounds)-1,q] = np.mean(\
                [[np.sum(\
                candidateCountsTest[i][j] > 0)/\
                len(shortHaloListTest[i]) \
                for j in diffMap[i]] \
                for i in range(0,len(ahCountsTest))])
            completeness_1f[k,l,len(massListBounds)-1,q] = np.mean(\
                [[np.sum(\
                    candidateCountsTest[j][i] > 0)/\
                    len(shortHaloListTest[j]) \
                    for j in diffMap[i]] \
                    for i in range(0,len(ahCountsTest))])
            purity_2f[k,l,len(massListBounds)-1,q] = np.mean(\
                [[np.sum(\
                    np.array(twoWayMatchListTest[i])[:,j])/\
                    len(shortHaloListTest[i]) \
                    for i in range(0,len(ahCountsTest))] \
                    for j in range(0,len(ahCountsTest)-1)])
            completeness_2f[k,l,len(massListBounds)-1,q] = np.mean(\
                [[np.sum(\
                    np.array(twoWayMatchListTest[i])[:,j])/\
                    len(shortHaloListTest[i]) \
                    for i in range(0,len(ahCountsTest))] \
                    for j in range(0,len(ahCountsTest)-1)])
            for m in range(0,len(massListBounds)-1):
                # Recompute with bounded masses:
                #massRange = [massListBounds[m],massListBounds[m+1]]
                massRange = None
                if massRange is not None:
                    [finalCatTest,shortHaloListTest,twoWayMatchListTest,\
                    finalCandidatesTest,finalRatiosTest,finalDistancesTest,\
                    allCandidatesTest,candidateCountsTest] = \
                        constructAntihaloCatalogue(snapNumList,snapList=snapList,\
                        snapListRev=snapListRev,ahProps=ahProps,hrList=hrList,\
                        max_index=None,twoWayOnly=True,blockDuplicates=True,\
                        #crossMatchThreshold = np.array([0.7,threshList[k]]),\
                        crossMatchThreshold = crossMatchThreshold,\
                        #crossMatchThreshold = threshList[k],\
                        #crossMatchQuantity = "radius",\
                        crossMatchQuantity = crossMatchQuantity,\
                        massRange = massRange,\
                        distMax = distArr2[l],\
                        rSphere=135,verbose=False,sortMethod=sortMethod,\
                        snapSortList=snapSortList,overlapList = overlaps)
                    centralAntihalosTest = [tools.getAntiHalosInSphere(\
                        antihaloCentres[k],135,\
                        filterCondition = (antihaloRadii[k] > rMin) & \
                        (antihaloRadii[k] <= rMax) & \
                        (antihaloMasses[k] > massRange[0]) & \
                        (antihaloMasses[k] <= massRange[1])) \
                        for k in range(0,len(snapNumList))]
                    centralAntihaloMassesTest = [\
                            antihaloMasses[k][centralAntihalosTest[k][0]] \
                            for k in range(0,len(centralAntihalosTest))]
                    sortedListTest = [\
                        np.flip(np.argsort(centralAntihaloMassesTest[k])) \
                        for k in range(0,len(snapNumList))]
                    ahCountsTest = np.array([len(cahs[0]) \
                        for cahs in centralAntihalosTest])
                    massListShortTest = [np.array([antihaloMasses[l][\
                        centralAntihalosTest[l][0][sortedListTest[l][k]]] \
                        for k in range(0,np.min([ahCountsTest[l],max_index]))]) \
                        for l in range(0,len(snapNumList))]
                    massFilter = [[(massListShortTest[n] > massListBounds[m]) & \
                        (massListShortTest[n] <= massListBounds[m+1]) \
                        for m in range(0,len(massListBounds)-1)] \
                        for n in range(0,len(massListShortTest))]
                # Record purity/completeness:
                purity_1f[k,l,m,q] = np.mean(\
                    [[np.sum((candidateCountsTest[i][j] > 0) & \
                        (massFilter[i][m]))/np.sum(massFilter[i][m]) \
                        for j in diffMap[i]] \
                        for i in range(0,len(ahCountsTest))])
                completeness_1f[k,l,m,q] = np.mean(\
                    [[np.sum((candidateCountsTest[j][i] > 0) & \
                        (massFilter[j][m]))/np.sum((massFilter[j][m])) \
                        for j in diffMap[i]] \
                        for i in range(0,len(ahCountsTest))])
                purity_2f[k,l,m,q] = np.mean(\
                    [[np.sum(np.array(twoWayMatchListTest[i])[:,j] & \
                        (massFilter[i][m]))/np.sum(massFilter[i][m]) \
                        for i in range(0,len(ahCountsTest))] \
                        for j in range(0,len(ahCountsTest)-1)])
                completeness_2f[k,l,m,q] = np.mean(\
                    [[np.sum(np.array(twoWayMatchListTest[i])[:,j] & \
                        (massFilter[i][m]))/np.sum(massFilter[i][m]) \
                        for i in range(0,len(ahCountsTest))] \
                        for j in range(0,len(ahCountsTest)-1)])
                #np.sum(\
                #    [np.array(twoWayMatchListTest[1])[:,0] & \
                #    (massFilter[1][m]))/np.sum((massFilter[1][m]) \
                #    ])
    print(("%.3g" % (100*(k*len(distArr2) + l + 1)/\
        (len(threshList)*len(distArr2)))) + "% complete")

euclideanDist2 = np.sqrt((purity_2f - 1.0)**2  + (completeness_2f - 1.0)**2)

# Completeness/Purity:


pickle.dump([purity_1f,purity_2f,completeness_1f,completeness_2f,distArr2,\
    threshList],open("purity_data_" + str(fixedMassThresh) + ".p","wb"))

[purity_1f,purity_2f,completeness_1f,completeness_2f,distArr2,threshList] = \
    pickle.load(open("purity_data_" + str(fixedMassThresh) + ".p","rb"))



def imshowComparison(leftGrid,rightGrid,top=0.851,bottom=0.167,left=0.088,\
        right=0.845,hspace=0.2,wspace=0.0,cbaxPar = [0.87,0.2,0.02,0.68],\
        figsize=(8,4),cmap = 'Blues',extentLeft = None,extentRight=None,\
        vLeft = [0.0,0.6],vRight = None,aspect='auto',origin='lower',\
        xlabel='Search radius ($R_{\mathrm{search}}/\sqrt{R_1R_2}$)',\
        ylabel = 'Radius ratio threshold ($\mu_{\mathrm{rad}}$)',\
        ylabelRight = None,\
        titleLeft = 'Two-way completeness ($^2C_{\mu_{\mathrm{rad}}}$)',\
        titleRight = 'Two-way purity ($^2P_{\mu_{\mathrm{rad}}}$)',\
        cbarLabel = 'Two-way completeness/purity',show = True,savename=None,\
        scaling = 'linear',superTitle=None,supTitleSize = 14):
    if vRight is None:
        vRight = vLeft
        cbarMode = "single"
    else:
        cbarMode = "double"
    if ylabelRight is None:
        ylabelRight = ylabel
    fig, ax = plt.subplots(1,2,figsize=figsize)
    imLeft = ax[0].imshow(leftGrid,extent=extentLeft,aspect=aspect,\
        cmap=cmap,origin=origin,vmin=vLeft[0],vmax=vLeft[1])
    imRight = ax[1].imshow(rightGrid,extent=extentRight,aspect=aspect,\
        cmap=cmap,origin=origin,vmin=vRight[0],vmax=vRight[1])
    for axi in ax:
        axi.set_xlabel(xlabel)
    ax[0].set_ylabel(ylabel)
    ax[1].set_ylabel(ylabelRight)
    ax[0].set_title(titleLeft)
    ax[1].set_title(titleRight)
    ax[1].yaxis.label.set_visible(False)
    ax[1].yaxis.set_major_formatter(NullFormatter())
    ax[1].yaxis.set_minor_formatter(NullFormatter())
    if scaling == 'linear':
        normFunc = colors.Normalize
    elif scaling == 'log':
        normFunc = colors.LogNorm
    else:
        raise Exception("Unknown normFunc")
    if superTitle is not None:
        fig.suptitle(superTitle,fontsize=supTitleSize)
    if cbarMode == "single":
        sm = cm.ScalarMappable(normFunc(vmin=vLeft[0],vmax=vLeft[1]),\
            cmap=cmap)
        cbax = fig.add_axes(cbaxPar)
        cbar = plt.colorbar(sm, orientation="vertical",\
            label=cbarLabel,cax=cbax)
        plt.subplots_adjust(top=top,bottom=bottom,left=left,right=right,\
            hspace=hspace,wspace=wspace)
    else:
        imList = [imLeft,imRight]
        vList = [vLeft,vRight]
        for k in range(0,len(imList)):
            sm = cm.ScalarMappable(normFunc(\
                vmin=vList[k][0],vmax=vList[k][1]),cmap=cmap)
            plt.colorbar(sm,ax=ax[k],orientation="vertical",cmap=cmap)
        plt.tight_layout()
    if savename is not None:
        plt.savefig(savename)
    if show:
        plt.show()


#ylabel = 'Radius ratio threshold ($\mu_{\mathrm{rad}}$)'
ylabel = 'Mass ratio threshold ($\mu_{\mathrm{mass}}$)'

imshowComparison(completeness_2f[:,:,0],purity_2f[:,:,0],\
    extentLeft= (np.min(distArr2),np.max(distArr2),\
    np.min(threshList),np.max(threshList)),\
    extentRight = (np.min(distArr2),np.max(distArr2),\
    np.min(threshList),np.max(threshList)),ylabel=ylabel)

massTitles = ["$" + plot.scientificNotation(massListBounds[k]) + \
    " < M/M_{\\odot} < " + plot.scientificNotation(massListBounds[k+1]) \
    + "$"  for k in range(0,len(massListBounds)-1)]
massTitles.append("All masses")
nM = 1
imshowComparison(completeness_2f[:,:,nM,0],completeness_2f[:,:,nM,1],\
    vRight = [0.0,1.0],vLeft = [0.0,1.0],\
    titleRight = 'Fixed radius threshold ($\mu_{\mathrm{rad}} = ' + \
    str(fixedRadThresh) + '$)',\
    titleLeft = 'Fixed mass threshold ($\mu_{\mathrm{mass}} = ' + \
    str(fixedMassThresh) + '$)',\
    extentLeft= (np.min(distArr2),np.max(distArr2),\
    np.min(threshList),np.max(threshList)),\
    extentRight = (np.min(distArr2),np.max(distArr2),\
    np.min(threshList),np.max(threshList)),superTitle = massTitles[nM],\
    ylabel="Radius or Mass Threshold")

imshowComparison(completeness_2f[:,:,1,0],completeness_2f[:,:,2,0],\
    vRight = [0.0,1.0],vLeft = [0.0,1.0],\
    titleLeft = '$10^{14} < M/M_{\\odot} \\leq 5\\times 10^{14}$',\
    titleRight = '$M/M_{\\odot} > 5\\times 10^{14}$',\
    extentLeft= (np.min(distArr2),np.max(distArr2),\
    np.min(threshList),np.max(threshList)),\
    extentRight = (np.min(distArr2),np.max(distArr2),\
    np.min(threshList),np.max(threshList)),superTitle = "Two-way Purity",\
    ylabel="Radius Threshold, $\\mu_{\\mathrm{rad}}$")




imshowComparison(purity_1f,completeness_1f,vRight = [0.0,1.0],vLeft=[0.0,1.0],\
    titleRight = 'One-way completeness ($^1C_{\mu_{\mathrm{rad}}}$)',\
    titleLeft = 'One-way purity ($^1P_{\mu_{\mathrm{rad}}}$)',\
    extentLeft= (np.min(distArr2),np.max(distArr2),\
    np.min(threshList),np.max(threshList)),\
    extentRight = (np.min(distArr2),np.max(distArr2),\
    np.min(threshList),np.max(threshList)),ylabel=ylabel)


imshowComparison(completeness_2f,purity_2f)
imshowComparison(completeness_2f,purity_2f)


# Curves of completeness/purity:
#plt.clf()
fig, ax = plt.subplots(1,2,figsize=(textwidth,0.5*textwidth))
ax[0].plot(distArr2,euclideanDist2[:,:,1,0].transpose(),\
    label=['$\\mu_{\\mathrm{rad}} = $' + ("%.2g" % thresh) \
    for thresh in threshList])
ax[1].plot(distArr2,euclideanDist2[:,:,2,0].transpose(),\
    label=['$\\mu_{\\mathrm{rad}} = $' + ("%.2g" % thresh) \
    for thresh in threshList])
ax[0].set_title('$10^{14} < M/M_{\\odot} \\leq 5\\times 10^{14}$')
ax[1].set_title('$M/M_{\\odot} > 5\\times 10^{14}$')
for axi in ax:
    axi.axvline(1.0,linestyle=':',color='grey')
    axi.set_xlabel('$R_{\\mathrm{search}}/\sqrt{R_1R_2}$')
    axi.set_ylabel('Distance $\sqrt{(^2P_f - 1)^2 + (^2C_f - 1)^2}$')
    axi.legend()

plt.suptitle(massTitles[nM])
plt.tight_layout()
plt.savefig(figuresFolder + \
    "supporting_plots/optimal_distance_squared_rsearch.pdf")
plt.show()


fig, ax = plt.subplots(1,2,figsize=(2.0*textwidth,2.0*0.5*textwidth))
ax[0].plot(threshList,euclideanDist2[:,1:-1:5,1,0],\
    label=['$R_{\\mathrm{search}}/\sqrt{R_1R_2} = $' + ("%.2g" % R) \
    for R in distArr2[1:-1:5]])
ax[1].plot(threshList,euclideanDist2[:,1:-1:5,2,0],\
    label=['$R_{\\mathrm{search}}/\sqrt{R_1R_2} = $' + ("%.2g" % R) \
    for R in distArr2[1:-1:5]])
ax[0].set_title('$10^{14} < M/M_{\\odot} \\leq 5\\times 10^{14}$')
ax[1].set_title('$M/M_{\\odot} > 5\\times 10^{14}$')
for axi in ax:
    axi.axvline(1.0,linestyle=':',color='grey')
    axi.set_ylabel('Distance $\sqrt{(^2P_f - 1)^2 + (^2C_f - 1)^2}$')
    axi.legend()

ax[0].set_xlabel('$\\mu_{\\mathrm{rad}}$')
ax[1].set_xlabel('$\\mu_{\\mathrm{mass}}$')

plt.suptitle(massTitles[nM])
plt.tight_layout()
plt.savefig(figuresFolder + \
    "supporting_plots/optimal_distance_squared_mu.pdf")
plt.show()




plt.clf()
plt.plot(threshList,euclideanDist2[:,1:-1:5,nM],\
    label=['$R_{\\mathrm{search}}/\sqrt{R_1R_2} = $' + ("%.2g" % R) \
    for R in distArr2[1:-1:5]])
plt.xlabel('$\\mu_{\\mathrm{rad}}$')
plt.ylabel('Distance $\sqrt{(^2P_f - 1)^2 + (^2C_f - 1)^2}$')
plt.legend()
plt.title(massTitles[nM])
plt.savefig(figuresFolder + \
    "supporting_plots/optimal_distance_squared_mu.pdf")
plt.show()

plt.clf()
plt.plot(distArr2,purity_1f[:,:,nM].transpose(),\
    label=['$\\mu_{\\mathrm{rad}} = $' + ("%.2g" % thresh) \
    for thresh in threshList])
plt.axvline(1.0,linestyle=':',color='grey')
plt.xlabel('$R_{\\mathrm{search}}/\sqrt{R_1R_2}$')
plt.ylabel('$^1P_f$')
plt.legend()
plt.savefig(figuresFolder + \
    "supporting_plots/optimal_distance_squared.pdf")
plt.show()

plt.clf()
plt.plot(distArr2,purity_2f[:,:,nM].transpose(),\
    label=['$\\mu_{\\mathrm{rad}} = $' + ("%.2g" % thresh) \
    for thresh in threshList])
plt.axvline(1.0,linestyle=':',color='grey')
plt.xlabel('$R_{\\mathrm{search}}/\sqrt{R_1R_2}$')
plt.ylabel('$^2P_f$')
plt.legend()
plt.savefig(figuresFolder + \
    "supporting_plots/optimal_distance_squared.pdf")
plt.show()

plt.clf()
plt.plot(distArr2,(purity_2f[:,:,nM]/purity_1f[:,:,nM]).transpose(),\
    label=['$\\mu_{\\mathrm{rad}} = $' + ("%.2g" % thresh) \
    for thresh in threshList])
plt.axvline(1.0,linestyle=':',color='grey')
plt.xlabel('$R_{\\mathrm{search}}/\sqrt{R_1R_2}$')
plt.ylabel('$^2P_{\mu_{\mathrm{rad}}}/{^1P_{\mu_{\mathrm{rad}}}}$')
plt.legend()
plt.savefig(figuresFolder + \
    "supporting_plots/purity_ratio.pdf")
plt.show()

plt.clf()
plt.plot(distArr2,purity_1f[:,:,4].transpose(),\
    label=['$^1P_{' + ("%.2g" % thresh) + "}$" \
    for thresh in threshList],linestyle='-')
plt.plot(distArr2,purity_2f[:,:,4].transpose(),\
    label=['$^2P_{' + ("%.2g" % thresh) + "}$" \
    for thresh in threshList],linestyle='--')
plt.axvline(1.0,linestyle=':',color='grey')
plt.xlabel('$R_{\\mathrm{search}}/\sqrt{R_1R_2}$')
plt.ylabel('$^wP_f$')
plt.legend()
plt.savefig(figuresFolder + \
    "supporting_plots/optimal_distance_squared.pdf")
plt.show()


# Combined profiles:



# Finding the optimal radius:
threshList = np.array([0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
#distArr2 = np.arange(0,21,0.2)[1:]
distArr2 = np.arange(0,3,0.1)
sortMethod='ratio'
uniqueMatchFracArray = np.zeros((len(threshList),len(distArr2)))
catFractionFinalMatrix = np.zeros((len(threshList),len(distArr2)))
ahCountsList = 
for k in range(0,len(threshList)):
    for l in range(0,len(distArr2)):
        #[finalCatTest,shortHaloListTest,twoWayMatchListTest,\
        #    finalCandidatesTest,finalRatiosTest,finalDistancesTest,\
        #    allCandidatesTest,candidateCountsTest] = \
        #    constructAntihaloCatalogue(snapNumList,snapList=snapList,\
        #    snapListRev=snapListRev,ahProps=ahProps,hrList=hrList,\
        #    crossMatchThreshold = threshList[k],distMax = distArr2[l],\
        #    verbose=False,max_index=max_index,sortMethod=sortMethod)
        [finalCatTest,shortHaloListTest,twoWayMatchListTest,\
            finalCandidatesTest,finalRatiosTest,finalDistancesTest,\
            allCandidatesTest,candidateCountsTest] = \
            constructAntihaloCatalogue(snapNumList,snapList=snapList,\
            snapListRev=snapListRev,ahProps=ahProps,hrList=hrList,\
            max_index=None,twoWayOnly=True,blockDuplicates=True,\
            crossMatchThreshold = threshList[k],distMax = distArr2[l],\
            rSphere=135,verbose=False)
        ahCountsTest = [len(val) for val in shortHaloListTest]
        candCountStripped = [candidateCountsTest[k][diffMap[k],:] \
            for k in range(0,len(snapNumList))]
        uniqueMatchFracArray[k,l] = np.sum(\
            [[len(np.where(counts[k,:] == 1)[0]) \
            for k in range(0,len(snapNumList)-1)] \
            for counts in candCountStripped])/\
            np.sum([[ahCountsTest[l] for k in range(0,len(snapNumList)-1)] \
            for l in range(0,len(snapNumList))])
        if len(finalCatTest) > 0:
            catFractionFinalMatrix[k,l] = np.mean(\
                np.sum(finalCatTest > 0,1)/len(snapNumList))
        print(("%.3g" % (100*(k*len(distArr2) + l + 1)/\
            (len(threshList)*len(distArr2)))) + "% complete")






# Plot the unique fraction:
plt.clf()
plt.plot(distArr2,uniqueMatchFracArray.transpose(),\
    label=['$\\mu_{\\mathrm{rad}} = $' + ("%.2g" % thresh) \
    for thresh in threshList])
plt.axvline(1.0,linestyle=':',color='grey')
plt.xlabel('$R_{\\mathrm{search}}/\sqrt{R_1R_2}$')
plt.ylabel('Fraction of Unique matches')
plt.legend()
plt.savefig(figuresFolder + \
    "supporting_plots/optimal_search_radius_unique.pdf")
plt.show()


# Catalogue fraction:

plt.clf()
plt.plot(distArr2,catFractionFinalMatrix.transpose(),\
    label=['$\\mu_{\\mathrm{rad}} = $' + ("%.2g" % thresh) \
    for thresh in threshList])
plt.axvline(1.0,linestyle=':',color='grey')
plt.xlabel('$R_{\\mathrm{search}}/\sqrt{R_1R_2}$')
plt.ylabel('Mean fraction of catalogues')
plt.legend()
plt.savefig(figuresFolder + \
    "supporting_plots/optimal_search_radius_catalogue.pdf")
plt.show()



# Setup interpolating functions:
interpList = [scipy.interpolate.interp1d(distArr2,-uniqueMatchFracArray[k,:],\
    kind = 'cubic') for k in range(0,len(threshList))]

optimalRList = np.array([scipy.optimize.minimize_scalar(fun,bracket=[5,10],\
    method='Bounded',bounds=[5,10]).x for fun in interpList])
optimalR = np.mean(optimalRList)

# Find the optimal radius threshold:
# Finding the optimal radius:
#distArr3 = np.array([5,8.73,10])
distArr3 = np.arange(0,1.1,0.1)
threshList2 = np.arange(0.0,1.0,0.05)
sortMethod='ratio'
uniqueMatchFracArray2 = np.zeros((len(threshList2),len(distArr3)))
noMatchFracMatrix2 = np.zeros((len(threshList2),len(distArr3)))
multiMatchFracMatrix2 = np.zeros((len(threshList2),len(distArr3)))
#twoWayMatchFracMatrix2 = np.zeros((len(threshList2),len(distArr3)))
catFractionFinalMatrix2 = np.zeros((len(threshList2),len(distArr3)))
for k in range(0,len(threshList2)):
    for l in range(0,len(distArr3)):
        #[finalCatTest,shortHaloListTest,twoWayMatchListTest,\
        #    finalCandidatesTest,finalRatiosTest,finalDistancesTest,\
        #    allCandidatesTest,candidateCountsTest] = \
        #    constructAntihaloCatalogue(snapNumList,snapList=snapList,\
        #    snapListRev=snapListRev,ahProps=ahProps,hrList=hrList,\
        #    crossMatchThreshold = threshList2[k],distMax = distArr3[l],\
        #    verbose=False,max_index=max_index,sortMethod=sortMethod)
        [finalCatTest,shortHaloListTest,twoWayMatchListTest,\
            finalCandidatesTest,finalRatiosTest,finalDistancesTest,\
            allCandidatesTest,candidateCountsTest] = \
            constructAntihaloCatalogue(snapNumList,snapList=snapList,\
            snapListRev=snapListRev,ahProps=ahProps,hrList=hrList,\
            max_index=None,twoWayOnly=True,blockDuplicates=True,\
            crossMatchThreshold = threshList2[k],distMax = distArr3[l],\
            rSphere=135,verbose=False)
        candCountStripped = np.array([candidateCountsTest[m][:,diffMap[m]] \
            for m in range(0,len(snapNumList))])
        uniqueMatchFracArray2[k,l] = np.sum(\
            [[len(np.where(counts[:,m] == 1)[0]) \
            for m in range(0,len(snapNumList)-1)] \
            for counts in candCountStripped])/\
            np.sum([[len(haloList) \
            for m in range(0,len(snapNumList)-1)] \
            for haloList in shortHaloListTest])
        noMatchFracMatrix2[k,l] = np.sum(\
            [[len(np.where(counts[:,m] == 0)[0]) \
            for m in range(0,len(snapNumList)-1)] \
            for counts in candCountStripped])/np.sum([[len(haloList) \
            for m in range(0,len(snapNumList)-1)] \
            for haloList in shortHaloListTest])
        multiMatchFracMatrix2[k,l] = np.sum(\
            [[len(np.where(counts[:,m] > 1)[0]) \
            for m in range(0,len(snapNumList)-1)] \
            for counts in candCountStripped])/np.sum([[len(haloList) \
            for m in range(0,len(snapNumList)-1)] \
            for haloList in shortHaloListTest])
        if len(finalCatTest) > 0:
            catFractionFinalMatrix2[k,l] = np.mean(\
                np.sum(finalCatTest > 0,1)/len(snapNumList))
        print(("%.3g" % (100*(k*len(distArr3) + l + 1)/\
            (len(threshList2)*len(distArr3)))) + "% complete")

# Plot the unique fraction:
colorList = [seabornColormap[k] \
    for k in range(0,np.min([len(distArr3),len(seabornColormap)]))]
for k in range(0,np.min([len(distArr3),len(seabornColormap)])):
    plt.plot(threshList2,uniqueMatchFracArray2[:,k],\
        label='Unique fraction ($R_{search} = $' + \
        ("%.2g" % distArr3[k]) + ")",\
        linestyle='-',color=colorList[k])
    plt.plot(threshList2,noMatchFracMatrix2[:,k],\
        label='Failed fraction ($R_{search} = $' + \
        ("%.2g" % distArr3[k]) + ")",\
        linestyle='--',color=colorList[k])
    plt.plot(threshList2,multiMatchFracMatrix2[:,k],\
        label='Ambiguous fraction ($R_{search} = $' + \
        ("%.2g" % distArr3[k]) + ")",\
        linestyle=':',color=colorList[k])

plt.xlabel('Radius ratio threshold ($\\mu_{\\mathrm{rad}}$)')
plt.ylabel('Fraction of matches')
plt.legend(loc='upper left',prop={"size":8,"family":'serif'},frameon=False)
plt.savefig(figuresFolder + "supporting_plots/optimal_murad_radius.pdf")
plt.show()

# Plot the final catalogue fraction:
plt.plot(threshList2,catFractionFinalMatrix2[:,1])
plt.xlabel('Radius ratio threshold ($\\mu_{\\mathrm{rad}}$)')
plt.ylabel('Mean catalogue fraction')
plt.axhline(0.99*catFractionFinalMatrix2[0,1],linestyle='--',color='grey',\
    label='$1\%$ reduction')
plt.legend()
plt.savefig(figuresFolder + \
    "supporting_plots/mean_catalogue_fraction_optimal.pdf")
plt.show()


interpList2 = [scipy.interpolate.interp1d(threshList2,\
    -uniqueMatchFracArray2[:,k],\
    kind = 'cubic') for k in range(0,len(distArr3))]

optimalMuList = np.array([scipy.optimize.minimize_scalar(fun,bracket=[0.2,0.8],\
    method='Bounded',bounds=[0.2,0.8]).x for fun in interpList2])


interpList3 = [scipy.interpolate.interp1d(threshList2,\
    1.0 - catFractionFinalMatrix2[:,k]/catFractionFinalMatrix2[0,k] - 1e-2,\
    kind='cubic') for k in range(0,len(distArr3))]

optimalCatFracList = np.array([scipy.optimize.brentq(fun,0.4,0.8) \
    for fun in interpList3])

# Optimal value of mu (highest that avoids the catalogue fraction dropping
# by more than 1% from it's saturated limit):
optimalMu = optimalCatFracList[1]

# Percentiles:
[finalCatOpt,shortHaloListOpt,twoWayMatchListOpt,finalCandidatesOpt,\
    finalRatiosOpt,finalDistancesOpt,allCandidatesOpt,candidateCountsOpt] = \
    constructAntihaloCatalogue(snapNumList,snapList=snapList,\
    snapListRev=snapListRev,ahProps=ahProps,hrList=hrList,max_index=max_index,\
    twoWayOnly=True,blockDuplicates=True,\
    crossMatchThreshold = 0.0,distMax = 8.81)



# Distribution of ratios:
meanRatio = [np.mean(np.hstack(arr)) for arr in finalRatiosOpt]
plt.hist(meanRatio,bins=np.linspace(0,1,21))
plt.xlabel('Radius-ratio$')
plt.ylabel('Number of unique, 2-way matches')
plt.axvline(np.percentile(meanRatio,1.0),label='1st Percentile',\
    color='grey',linestyle='--')
plt.legend()
plt.savefig(figuresFolder + "supporting_plots/ratio_histogram.pdf")
plt.show()




# Mass function plot:
from void_analysis import cosmology, plot_utilities
def plotMassFunction(masses,volSim,ax=None,Om0=0.3,h=0.8,ns=1.0,\
        Delta=200,sigma8=0.8,fontsize=12,legendFontsize=10,font="serif",\
        Ob0=0.049,mass_function='Tinker',delta_wrt='SOCritical',massLower=5e13,\
        massUpper=1e15,figsize=(4,4),marker='x',linestyle='--',\
        color=seabornColormap[0],colorTheory = seabornColormap[1],\
        nBins=21,poisson_interval = 0.95,legendLoc='lower left',\
        label="Gadget Simulation",transfer_model='EH',fname=None,\
        xlabel="Mass [$M_{\odot}h^{-1}$]",ylabel="Number of halos",\
        ylim=[1e1,2e4],title="Gadget Simulation",showLegend=True,\
        tickRight=False,tickLeft=True):
    [dndm,m] = cosmology.TMF_from_hmf(massLower,massUpper,\
        h=h,Om0=Om0,Delta=Delta,delta_wrt=delta_wrt,\
        mass_function=mass_function,sigma8=sigma8,Ob0 = Ob0,\
        transfer_model=transfer_model,fname=fname,ns=ns)
    massBins = 10**np.linspace(np.log10(massLower),np.log10(massUpper),nBins)
    n = cosmology.dndm_to_n(m,dndm,massBins)
    bounds = np.array(scipy.stats.poisson(n*volSim).interval(poisson_interval))
    massBinCentres = plot_utilities.binCentres(massBins)
    alphaO2 = (1.0 - poisson_interval)/2.0
    if type(masses) == list:
        [noInBins,sigmaBins] = plot.computeMeanHMF(masses,\
            massLower=massLower,massUpper=massUpper,nBins = nBins)
    else:
        noInBins = plot_utilities.binValues(masses,massBins)[1]
        sigmaBins = np.abs(np.array([scipy.stats.chi2.ppf(\
            alphaO2,2*noInBins)/2,\
            scipy.stats.chi2.ppf(1.0 - alphaO2,2*(noInBins+1))/2]) - \
            noInBins)
    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=(8,4))
    ax.errorbar(massBinCentres,noInBins,sigmaBins,marker=marker,\
        linestyle=linestyle,label=label,color=color)
    ax.plot(massBinCentres,n*volSim,":",label=mass_function + ' prediction',\
                    color=seabornColormap[1])
    ax.fill_between(massBinCentres,
                    bounds[0],bounds[1],
                    facecolor=colorTheory,alpha=0.5,interpolate=True,\
                    label='$' + str(100*poisson_interval) + \
                    '\%$ Confidence \nInterval')
    if showLegend:
        ax.legend(prop={"size":legendFontsize,"family":font},
            loc=legendLoc,frameon=False)
    ax.set_title(title,fontsize=fontsize,fontfamily=font)
    ax.set_xlabel(xlabel,fontsize=fontsize,fontfamily=font)
    ax.set_ylabel(ylabel,fontsize=fontsize,fontfamily=font)
    ax.tick_params(axis='both',labelsize=fontsize,\
        labelright=tickRight,right=tickRight)
    ax.tick_params(axis='both',which='minor',bottom=True,labelsize=fontsize)
    ax.tick_params(axis='y',which='minor')
    ax.yaxis.grid(color='grey',linestyle=':',alpha=0.5)
    ax.set_ylim(ylim)
    ax.set_xscale('log')
    ax.set_yscale('log')

def massFunctionComparison(massesLeft,massesRight,volSim,Om0=0.3,h=0.8,\
        Delta=200,sigma8=0.8,fontsize=12,legendFontsize=10,font="serif",\
        Ob0=0.049,mass_function='Tinker',delta_wrt='SOCritical',massLower=5e13,\
        massUpper=1e15,figsize=(8,4),marker='x',linestyle='--',ax=None,\
        color=seabornColormap[0],colorTheory = seabornColormap[1],\
        nBins=21,poisson_interval = 0.95,legendLoc='lower left',\
        labelLeft = 'Gadget Simulation',labelRight='ML Simulation',\
        xlabel="Mass [$M_{\odot}h^{-1}$]",ylabel="Number of halos",\
        ylim=[1e1,2e4],savename=None,show=True,transfer_model='EH',fname=None,\
        returnAx = False,ns=1.0,rows=1,cols=2,titleLeft = "Gadget Simulation",\
        titleRight = "Gadget Simulation"):
    if ax is None:
        fig, ax = plt.subplots(rows,cols,figsize=(8,4))
    plotMassFunction(massesLeft,volSim,ax=ax[0],Om0=Om0,h=h,ns=ns,\
        Delta=Delta,sigma8=sigma8,fontsize=fontsize,\
        legendFontsize=legendFontsize,font="serif",\
        Ob0=Ob0,mass_function=mass_function,delta_wrt=delta_wrt,\
        massLower=massLower,title=titleLeft,\
        massUpper=massUpper,marker=marker,linestyle=linestyle,\
        color=color,colorTheory = colorTheory,\
        nBins=nBins,poisson_interval = poisson_interval,legendLoc='lower left',\
        label=labelLeft,transfer_model=transfer_model,ylim=ylim)
    plotMassFunction(massesRight,volSim,ax=ax[1],Om0=Om0,h=h,ns=ns,\
        Delta=Delta,sigma8=sigma8,fontsize=fontsize,\
        legendFontsize=legendFontsize,font="serif",\
        Ob0=Ob0,mass_function=mass_function,delta_wrt=delta_wrt,\
        massLower=massLower,title = titleRight,\
        massUpper=massUpper,marker=marker,linestyle=linestyle,\
        color=color,colorTheory = colorTheory,\
        nBins=nBins,poisson_interval = poisson_interval,legendLoc='lower left',\
        label=labelRight,transfer_model=transfer_model,ylim=ylim)
    plt.tight_layout()
    if savename is not None:
        plt.savefig(savename)
    if show:
        plt.show()
    if returnAx:
        return ax



# Construct the final Catalogue using optimal values:
muOpt = 0.70
rSearchOpt = 1.2
rSphere = 300
mMin = 1e11
mMax = 1e16

diffMap = [np.setdiff1d(np.arange(0,len(snapNumList)),[k]) \
    for k in range(0,len(snapNumList))]

[finalCatOpt,shortHaloListOpt,twoWayMatchListOpt,finalCandidatesOpt,\
    finalRatiosOpt,finalDistancesOpt,allCandidatesOpt,candidateCountsOpt,\
    allRatiosOpt,finalCombinatoricFracOpt,finalCatFracOpt] = \
    constructAntihaloCatalogue(snapNumList,snapList=snapList,\
    snapListRev=snapListRev,ahProps=ahProps,hrList=hrList,max_index=None,\
    twoWayOnly=True,blockDuplicates=True,\
    crossMatchThreshold = muOpt,distMax = rSearchOpt,rSphere=rSphere,\
    massRange = [mMin,mMax],NWayMatch = False,rMin=rMin,rMax=rMax)

finalCombinatoricFracOpt = np.zeros(len(finalCatOpt))
Ncats = len(snapNumList)
for k in range(0,len(finalCatOpt)):
    twoWayMatchCounts = 0
    for m in range(0,finalCatOpt.shape[1]):
        for d in diffMap[m]:
            allCands = allCandidatesOpt[m][d][finalCatOpt[k,m]-1]
            if len(allCands) > 0:
                if allCands[0] == finalCatOpt[k,d]-1:
                    twoWayMatchCounts += 1
    finalCombinatoricFracOpt[k] = twoWayMatchCounts/(Ncats*(Ncats-1))



indexMap = -np.ones((5,5),dtype=int)


# Expand into all possible candidates:
nV = 2
allCands = [[] for k in range(0,5)]
lengthsList = np.zeros(5,dtype=int)
for k in range(0,5):
    if finalCatOpt[nV][k] > -1:
        allCands[k].append(finalCatOpt[nV][k]-1)

lengthsListNew = np.array([len(cand) for cand in allCands],dtype=int)
while not np.all(lengthsListNew == lengthsList):
    lengthsList = lengthsListNew
    for k in range(0,5):
        if len(allCands[k]) > 0:
            for l in diffMap[k]:
                for m in range(0,len(allCands[k])):
                    if len(allCandidatesOpt[k][l][allCands[k][m]]) > 0:
                        if not np.isin(allCandidatesOpt[k][l][allCands[k][m]][0],allCands[l]):
                            allCands[l].append(allCandidatesOpt[k][l][allCands[k][m]][0])
    lengthsListNew = np.array([len(cand) for cand in allCands],dtype=int)

# Count two way matches:
twoWayMatchesAllCands = [[] for k in range(0,5)]
for k in range(0,5):
    for l in range(0,len(allCands[k])):
        nTW = np.sum(np.array(\
            [len(allCandidatesOpt[k][m][allCands[k][l]]) \
            for m in diffMap[k]]) > 0)
        twoWayMatchesAllCands[k].append(nTW)

# Compute the average fractions:
ratioAverages = [[] for k in range(0,5)]
for k in range(0,5):
    for l in range(0,len(allCands[k])):
        ratios = np.zeros(4)
        for m in range(0,4):
            if len(allRatiosOpt[k][diffMap[k][m]][allCands[k][l]]) > 0:
                ratios[m] = allRatiosOpt[k][diffMap[k][m]][allCands[k][l]][0]
        qR = np.mean(ratios)
        ratioAverages[k].append(qR)





# How then do we decide which void to pick if we have multiple candidates?
# First, pick the void with the greatest number of two-way matches to other
# catalogues.
# But if we still have an ambiguity, then we pick the one with the greatest 
# average radius ratio.




nTW = 0
for k in range(0,5):
    if len(allCands[k]) > 0:
        for l in range(0,k):
            if len(allCandidatesOpt[k][l][allCands[k][0]]) > 0:
                nTW += 1




for k in range(0,5):
    for l in range(0,5):
        if k != l:
            if len(allCandidatesOpt[k][l][nV]) == 1:
                indexMap[k,l] = allCandidatesOpt[k][l][nV][0]



antihaloCentres = [tools.remapAntiHaloCentre(props[5],boxsize) \
    for props in ahProps]
antihaloMasses = [props[3] for props in ahProps]
antihaloRadii = [props[7] for props in ahProps]
centralAntihalos = [tools.getAntiHalosInSphere(antihaloCentres[k],rSphere,\
        filterCondition = (antihaloRadii[k] > rMin) & \
        (antihaloRadii[k] <= rMax) & (antihaloMasses[k] > mMin) & \
        (antihaloMasses[k] <= mMax)) \
        for k in range(0,len(snapNumList))]
centralAntihaloMasses = [\
        antihaloMasses[k][centralAntihalos[k][0]] \
        for k in range(0,len(centralAntihalos))]
sortedList = [np.flip(np.argsort(centralAntihaloMasses[k])) \
        for k in range(0,len(snapNumList))]
ahCounts = np.array([len(cahs[0]) for cahs in centralAntihalos])

centresListShort = [np.array([antihaloCentres[l][\
        centralAntihalos[l][0][sortedList[l][k]],:] \
        for k in range(0,np.min([ahCounts[l],max_index]))]) \
        for l in range(0,len(snapNumList))]

radiiListShort = [np.array([antihaloRadii[l][\
        centralAntihalos[l][0][sortedList[l][k]]] \
        for k in range(0,np.min([ahCounts[l],max_index]))]) \
        for l in range(0,len(snapNumList))]

massListShort = [np.array([antihaloMasses[l][\
        centralAntihalos[l][0][sortedList[l][k]]] \
        for k in range(0,np.min([ahCounts[l],max_index]))]) \
        for l in range(0,len(snapNumList))]

deltaBarList = [props[12] for props in ahProps]
deltaCList = [props[11] for props in ahProps]

deltaBarListShort = [np.array([deltaBarList[l][\
        centralAntihalos[l][0][sortedList[l][k]]] \
        for k in range(0,np.min([ahCounts[l],max_index]))]) \
        for l in range(0,len(snapNumList))]
deltaCListShort = [np.array([deltaCList[l][\
        centralAntihalos[l][0][sortedList[l][k]]] \
        for k in range(0,np.min([ahCounts[l],max_index]))]) \
        for l in range(0,len(snapNumList))]

radiiListComb = getRadiiFromCat(finalCatOpt,radiiListShort)
massListComb = getRadiiFromCat(finalCatOpt,massListShort)
deltaBarListComb = getRadiiFromCat(finalCatOpt,deltaBarListShort)
deltaCListComb = getRadiiFromCat(finalCatOpt,deltaCListShort)


for k in range(0,5):
    plt.hist(radiiListShort[k][finalCatOpt[finalCatOpt[:,k] > 0,k] - 1],alpha=0.5)


for k in range(0,5):
    plt.hist(radiiListComb[radiiListComb[:,k] > 0,k],alpha=0.5)




[radiiListMean,radiiListSigma] = getMeanProperty(radiiListComb)
[massListMean,massListSigma] = getMeanProperty(massListComb)
[deltaBarListMean,deltaBarListSigma] = getMeanProperty(deltaBarListComb,\
    lowerLimit = -1,stdError=False)
[deltaCListMean,deltaCListSigma] = getMeanProperty(deltaCListComb,\
    lowerLimit = -1,stdError=False)

# Robustness vs Mass:
massBins = 10**np.linspace(np.log10(1e14),np.log10(1e15),16)
[binList,noInBins] = plot_utilities.binValues(massListMean,massBins)
massBinCentres = plot_utilities.binCentres(massBins)
meanRobustness = np.array([np.mean(finalCombinatoricFracOpt[ind]) \
    for ind in binList])
stdRobustness = np.array([np.std(finalCombinatoricFracOpt[ind])/\
    np.sqrt(len(ind)) for ind in binList])

meanCatFrac = np.array([np.mean(finalCatFracOpt[ind]) for ind in binList])
stdCatFrac = np.array([np.std(finalCatFracOpt[ind])/np.sqrt(len(ind)) \
    for ind in binList])

plt.errorbar(massBinCentres,meanRobustness,yerr=stdRobustness,\
    label='Combinatoric Fraction')
plt.errorbar(massBinCentres,meanCatFrac,yerr=stdCatFrac,\
    label = "Catalogue fraction")
plt.xscale('log')
plt.xlabel('Mass ($M_{\\odot}h^{-1}$)')
plt.ylabel('Mean Fraction')
plt.legend(frameon=False)
plt.show()



plt.xscale('log')
plt.xlabel('Mass ($M_{\\odot}h^{-1}$)')
plt.ylabel('Mean Combinatoric Fraction')
plt.show()



inCatalogueFinal = [finalCatOpt[:,k] > 0 for k in range(0,len(massListShort))]
deltaBarMeanFullList = np.hstack([deltaBarListMean[cond]] \
    for cond in inCatalogueFinal)[0]
deltaBarScatterFullList = np.hstack([deltaBarListComb[inCatalogueFinal[k],k]] \
    for k in range(0,len(inCatalogueFinal)))[0]

[binList2,noInBins2] = plot_utilities.binValues(deltaBarScatterFullList,deltaBins)
meanDelta = [np.mean(deltaBarScatterFullList[ind]) for ind in binList2]
sigmaDelta = [np.std(deltaBarScatterFullList[ind]) for ind in binList2]

plt.errorbar(deltaBinCentres,meanSigma,yerr=sigmaSigma,\
    label='Between Catalogues')
plt.errorbar(deltaBinCentres,sigmaDelta,yerr=sigmaSigma2,\
    label = "All matched antihalos")
plt.xlabel('Average-density contrast')
plt.ylabel('Standard Deviation')
plt.legend()
plt.show()

# Scatter about the mean density contrast
plotRange = [-0.8,-0.6]
plt.hist2d(deltaBarMeanFullList,deltaBarScatterFullList,\
    cmap='Blues',bins=np.linspace(plotRange[0],plotRange[1],21))
plt.xlabel('Mean Average-Density Constrast')
plt.ylabel('Average-Density Contrast')
plt.plot(plotRange,plotRange,'k--')
plt.title("Average-Density Contrast, scatter about mean")
plt.colorbar()
plt.tight_layout()
plt.show()

plt.scatter(massListMean,radiiListMean)
plt.xscale('log')
#plt.yscale('log')
plt.xlabel("Mass ($M_{\\odot}h^{-1})$")
plt.ylabel("Radius ($\\mathrm{Mpc}h^{-1})$")
plt.show()

# Scatter plot of average densities:
inCatalogueFinal = [finalCatOpt[:,k] > 0 for k in range(0,len(massListShort))]
inCatalogue = [np.isin(np.arange(1,len(massListShort[k])+1),finalCatOpt[:,k]) \
    for k in range(0,len(massListShort))]

#condition = (deltaBarListComb[:,0] > -1) & (deltaBarListComb[:,1] > -1)
condition = inCatalogueFinal[0] & inCatalogueFinal[1]
condition = condition & (massListComb[:,0] > 1e14) & (massListComb[:,0] <= 5e14)
condition = condition & (massListComb[:,1] > 1e14) & (massListComb[:,1] <= 5e14)

conditionHigh = inCatalogueFinal[0] & inCatalogueFinal[1]
conditionHigh = conditionHigh & (massListComb[:,0] > 5e14) & \
    (massListComb[:,0] <= 1e16)
conditionHigh = conditionHigh & (massListComb[:,1] > 5e14) & \
    (massListComb[:,1] <= 1e16)

deltaBins = np.linspace(-0.8,-0.6,21)
[binList,noInBins] = plot_utilities.binValues(deltaBarListMean[condition],deltaBins)
meanSigma = [np.mean(deltaBarListSigma[condition][ind]) for ind in binList]
sigmaSigma = [np.std(deltaBarListSigma[condition][ind])/np.sqrt(len(ind)) \
    for ind in binList]
deltaBinCentres = plot_utilities.binCentres(deltaBins)
deltaStd = np.std(deltaBarListComb[condition,0])

plt.errorbar(deltaBinCentres,meanSigma,yerr=sigmaSigma,\
    label = "Standard deviation \n of the same void " + \
    "\n between catalogues")
plt.axhline(deltaStd,color='grey',linestyle='--',\
    label = "Standard deviation \n of all voids \n over mass range")
plt.xlabel("Average Density Bin")
plt.ylabel("Standard Deviation")
plt.title("Mass bin $10^{14} < M/M_{\\odot} \\leq 5\\times 10^{14}$")
plt.ylim([0,0.06])
plt.legend(frameon = False)
plt.show()


# Average Density
fig, ax = plt.subplots(1,2,figsize=(textwidth,0.5*textwidth))

plotRange = [-0.8,-0.6]
ax[0].hist2d(deltaBarListComb[condition,0],deltaBarListComb[condition,1],\
    cmap='Blues',bins=np.linspace(plotRange[0],plotRange[1],21))
ax[0].set_xlabel('Catalogue 1, average density')
ax[0].set_ylabel('Catalogue 2, average density')
ax[0].plot(plotRange,plotRange,'k--')
ax[0].set_title("$10^{14} < M/M_{\\odot} \\leq 5\\times 10^{14}$")

plotRange = [-0.8,-0.6]
ax[1].hist2d(deltaBarListComb[conditionHigh,0],deltaBarListComb[conditionHigh,1],\
    cmap='Blues',bins=np.linspace(plotRange[0],plotRange[1],21))
ax[1].set_xlabel('Catalogue 1, average density')
ax[1].set_ylabel('Catalogue 2, average density')
ax[1].plot(plotRange,plotRange,'k--')
ax[1].set_title("$M/M_{\\odot} > 5\\times 10^{14}$")

ax[1].yaxis.label.set_visible(False)
ax[1].yaxis.set_major_formatter(NullFormatter())
ax[1].yaxis.set_minor_formatter(NullFormatter())


sm = cm.ScalarMappable(colors.LogNorm(vmin=1,vmax=2000),\
    cmap='Blues')
cbax = fig.add_axes([0.87,0.2,0.02,0.68])
cbar = plt.colorbar(sm, orientation="vertical",cax=cbax)
plt.subplots_adjust(top=0.885,bottom=0.194,left=0.122,right=0.849,hspace=0.2,\
    wspace=0.0)

plt.show()

fig, ax = plt.subplots()
plotRange = [-0.8,-0.6]
ax.hist2d(deltaBarListComb[condition,0],deltaBarListComb[condition,1],\
    cmap='Blues',bins=np.linspace(plotRange[0],plotRange[1],21))
ax.set_xlabel('Catalogue 1, average density')
ax.set_ylabel('Catalogue 2, average density')
ax.plot(plotRange,plotRange,'k--')
ax.set_title("$10^{14} < M/M_{\\odot} \\leq 5\\times 10^{14}$")
plt.colorbar()
plt.tight_layout()
plt.show()


# Central Density
plotRange = [-0.95,-0.7]
plt.hist2d(deltaCListComb[condition,0],deltaCListComb[condition,1],\
    cmap='Blues',bins=np.linspace(plotRange[0],plotRange[1],21))
plt.xlabel('Catalogue 1, central density')
plt.ylabel('Catalogue 2, central density')
plt.plot(plotRange,plotRange,'k--')
plt.title("Central Void Density")
plt.colorbar()
plt.tight_layout()
plt.show()

# Mass
plotRangeMass = [1e14,5e14]
plt.hist2d(massListComb[condition,0],massListComb[condition,1],\
    cmap='Blues',bins=10**np.linspace(np.log10(plotRangeMass[0]),\
    np.log10(plotRangeMass[1]),21))
plt.xlabel('Catalogue 1, mass')
plt.ylabel('Catalogue 2, mass')
plt.xscale('log')
plt.yscale('log')
plt.plot(plotRangeMass,plotRangeMass,'k--')
plt.title("Mass compared between catalogues")
plt.colorbar()
plt.tight_layout()
plt.show()

# Radius
plotRangeRad = [5,30]
plt.hist2d(radiiListComb[condition,0],radiiListComb[condition,1],\
    cmap='Blues',bins=10**np.linspace(np.log10(plotRangeRad[0]),\
    np.log10(plotRangeRad[1]),21))
plt.xlabel('Catalogue 1, radius ($\\mathrm{Mpc}h^{-1}$)')
plt.ylabel('Catalogue 2, radius ($\\mathrm{Mpc}h^{-1}$)')
plt.plot(plotRangeRad,plotRangeRad,'k--')
plt.xscale('log')
plt.yscale('log')
plt.title("Radius compared between catalogues")
plt.colorbar()
plt.tight_layout()
plt.show()


plt.scatter(radiiListComb[condition,0],radiiListComb[condition,1])
plt.plot([5,16],[5,16],'k--')
plt.xlim([5,16])
plt.ylim([5,16])
plt.show()


plt.hist(radiiListShort[0],alpha=0.5)
plt.hist(radiiListShort[1],alpha=0.5)
plt.show()

plt.hist(radiiListComb[condition,0],alpha=0.5)
plt.hist(radiiListComb[condition,1],alpha=0.5)
plt.show()

condition = (deltaBarListComb[:,0] > -1) & (deltaBarListComb[:,1] > -1)
condition = condition & (massListComb[:,0] > 1e14) & (massListComb[:,0] <= 5e14)
condition = condition & (massListComb[:,1] > 1e14) & (massListComb[:,1] <= 5e14)

plt.hist(radiiListComb[condition,0],alpha=0.5)
plt.hist(radiiListComb[condition,1],alpha=0.5)
plt.show()

# Spead in the catalogues with radius:
def plotDensityVsRadius(radii,density,cmap='Blues',radBins = None,\
        deltaBins = None,vmin = 1,vmax = 2000,Om0 = 0.3111,\
        mLines = [1e14,5e14,1e15],linestyleList = ['k--','k:','k-'],\
        xlabel = 'Radius ($\\mathrm{Mpc}h^{-1}$)',\
        ylabel = 'Average density contrast',xlim = [5,30],ylim = [-1,0],\
        ax = None,savename=None,show = True,showLegend = True,\
        showColorbar = True,logx = False,logy = False,showMassCurves = True):
    if ax is None:
        fig, ax = plt.subplots()
    if radBins is None:
        radBins = np.linspace(0,30,21)
    if deltaBins is None:
        deltaBins = np.linspace(-1,0,21)
    ax.hist2d(radii,density,\
        bins = [radBins,deltaBins],cmap=cmap,\
        norm = colors.LogNorm(vmin = vmin,vmax = vmax))
    if showMassCurves:
        rList = np.linspace(radBins[0] + 0.01,radBins[-1],len(radBins))
        rhoBar = 2.7754e11*Om0
        dList = [m/(rhoBar*4*np.pi*rList**3/3) - 1.0 for m in mLines]
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if logx:
        ax.set_xscale('log')
    if logy:
        ax.set_yscale('log')
    if showMassCurves:
        for m in range(0,len(mLines)):
            ax.plot(rList,dList[m],linestyleList[m],label = '$' + \
                plot.scientificNotation(mLines[m]) + 'M_{\\odot}h^{-1}$')
    if showLegend:
        ax.legend(frameon=False)
    if showColorbar:
        sm = cm.ScalarMappable(colors.LogNorm(vmin=vmin,vmax=vmax),cmap=cmap)
        cbar = plt.colorbar(sm, orientation="vertical",ax = ax)
    #plt.tight_layout()
    if savename is not None:
        plt.savefig(savename)
    if show:
        plt.show()


conditionMassRestricted = [(massListShort[k] > 1e14) & \
    (massListShort[k] <= 5e14) & inCatalogue[k] \
    for k in range(0,len(massListShort))]

fig, ax = plt.subplots(1,2,figsize = (textwidth,0.5*textwidth))

plotDensityVsRadius(radiiListShort[0][conditionMassRestricted[0]],\
    deltaBarListShort[0][conditionMassRestricted[0]],ax=ax[0],show=False,\
    showLegend = False,showColorbar = False)
plotDensityVsRadius(radiiListShort[1][conditionMassRestricted[1]],\
    deltaBarListShort[1][conditionMassRestricted[1]],ax=ax[1],show=False,\
    showColorbar = False)

sm = cm.ScalarMappable(colors.LogNorm(vmin=1,vmax=2000),\
    cmap='Blues')
cbax = fig.add_axes([0.87,0.2,0.02,0.68])
cbar = plt.colorbar(sm, orientation="vertical",cax=cbax)
plt.subplots_adjust(top=0.891,bottom=0.167,left=0.113,right=0.85,hspace=0.2,\
    wspace=0.06)
ax[1].yaxis.label.set_visible(False)
ax[1].yaxis.set_major_formatter(NullFormatter())
ax[1].yaxis.set_minor_formatter(NullFormatter())
ax[0].set_title("Catalogue 1")
ax[1].set_title("Catalogue 2")

plt.show()


# Radius mass distribution:
fig, ax = plt.subplots(1,2,figsize = (textwidth,0.5*textwidth))

plotDensityVsRadius(radiiListShort[0][conditionMassRestricted[0]],\
    massListShort[0][conditionMassRestricted[0]],ax=ax[0],show=False,\
    showLegend = False,showColorbar = False,\
    deltaBins = 10**(np.linspace(np.log10(1e14),np.log10(5e14),21)),\
    logy = True,ylabel = "Mass ($M_{\\odot}h^{-1}$)",\
    showMassCurves = False,ylim=[1e14,5e14])
plotDensityVsRadius(radiiListShort[1][conditionMassRestricted[1]],\
    massListShort[1][conditionMassRestricted[1]],ax=ax[1],show=False,\
    showColorbar = False,logy = True,ylabel = "Mass ($M_{\\odot}h^{-1}$)",\
    deltaBins = 10**(np.linspace(np.log10(1e14),np.log10(5e14),21)),\
    showMassCurves = False,ylim=[1e14,5e14])


fitList = [np.polyfit(np.log10(radiiListShort[k][conditionMassRestricted[k]]),\
    np.log10(massListShort[k][conditionMassRestricted[k]]),1) \
    for k in range(0,len(massListShort))]


radBins = np.linspace(0,30,21)
rList = np.linspace(radBins[0] + 0.01,radBins[-1],len(radBins))
ax[0].plot(rList,10**(fitList[0][0]*np.log10(rList) + fitList[0][1]),'k--',\
    label = "Fit, \nM = $" + plot.scientificNotation(10**fitList[0][1]) + \
    "M_{\\odot}$\n$\\times r^{" + plot.scientificNotation(fitList[0][0]) + "}$")
ax[1].plot(rList,10**(fitList[1][0]*np.log10(rList) + fitList[1][1]),'k--',\
    label = "Fit, \nM = $" + plot.scientificNotation(10**fitList[1][1]) + \
    "M_{\\odot}$\n$\\times r^{" + plot.scientificNotation(fitList[1][0]) + "}$")

sm = cm.ScalarMappable(colors.LogNorm(vmin=1,vmax=2000),\
    cmap='Blues')
cbax = fig.add_axes([0.87,0.2,0.02,0.68])
cbar = plt.colorbar(sm, orientation="vertical",cax=cbax)
plt.subplots_adjust(top=0.891,bottom=0.167,left=0.143,right=0.85,hspace=0.2,\
    wspace=0.06)
ax[1].yaxis.label.set_visible(False)
ax[1].yaxis.set_major_formatter(NullFormatter())
ax[1].yaxis.set_minor_formatter(NullFormatter())
ax[0].set_title("Catalogue 1")
ax[1].set_title("Catalogue 2")
ax[0].legend(frameon=False,loc="lower right",prop={"size":9,"family":"serif"})
ax[1].legend(frameon=False,loc="lower right",prop={"size":9,"family":"serif"})
plt.show()


# Filter for catalogue fraction:
catFracOpt = np.array([np.sum(finalCatOpt[k,:] > -1)/len(snapNumList) \
    for k in range(0,len(finalCatOpt))])


# Mean mass function:
mUnit = 8*0.3111*2.7754e11*(677.7/512)**3
mLower = 100*mUnit
volSphere = 4*np.pi*rSphere**3/3
combFracThresh = 0.1


massFunctionComparison(massListMean,\
    massListShort,volSphere,nBins=11,\
    labelLeft = "Combined anti-halo catalogue",\
    labelRight="Average of " + str(len(snapNumList)) + " catalogues",\
    ylabel="Number of antihalos",savename=figuresFolder + \
    "supporting_plots/mass_function_combined.pdf",massLower=mLower,\
    ylim=[1,500],Om0 = 0.3111,h=0.6766,sigma8=0.8128,ns=0.9667,\
    fontsize=10,massUpper = 2e15,\
    titleLeft = "Combined Catalogue",titleRight = "All catalogues average")

massFunctionComparison(massListMean[catFracOpt > 0.5],\
    massListShort,volSphere,nBins=11,\
    labelLeft = "Combined anti-halo catalogue",\
    labelRight="Average of " + str(len(snapNumList)) + " catalogues",\
    ylabel="Number of antihalos",savename=figuresFolder + \
    "supporting_plots/mass_function_combined.pdf",massLower=mLower,\
    ylim=[1,500],Om0 = 0.3111,h=0.6766,sigma8=0.8128,ns=0.9667,\
    fontsize=10,massUpper = 2e15,\
    titleLeft = "Combined Catalogue",titleRight = "All catalogues average")





massFunctionComparison(massListMean[finalCombinatoricFracOpt > combFracThresh],\
    massListShort,volSphere,nBins=11,\
    labelLeft = "Combined anti-halo catalogue",\
    labelRight="Average of 5 catalogues",\
    ylabel="Number of antihalos",savename=figuresFolder + \
    "supporting_plots/mass_function_combined.pdf",massLower=mLower,\
    ylim=[1,500],Om0 = 0.3111,h=0.6766,sigma8=0.8128,ns=0.9667,\
    fontsize=10,massUpper = 2e15,\
    titleLeft = "Combined Catalogue",titleRight = "All catalogues average")

additionalConditions = [np.isin(np.arange(0,len(antihaloMasses[k])),\
    np.array(centralAntihalos[k][0])[finalCatOpt[finalCatOpt[:,k] >= 0,k] - 1]) \
    for k in range(0,len(snapList))]

# Void profiles plot:

[rBinStackCentresCombined,nbarjSepStackCombined,\
        sigmaSepStackCombined,nbarjSepStackUnCombined,sigmaSepStackUnCombined,\
        nbarjAllStackedCombined,sigmaAllStackedCombined,\
        nbarjAllStackedUnCombined,sigmaAllStackedUnCombined,\
        nbar,rMin,mMin,mMax] = getVoidProfilesData(\
            snapNumList,snapNumListUncon,\
            snapList = snapList,snapListRev = snapListRev,\
            samplesFolder="new_chain/",\
            unconstrainedFolder="new_chain/unconstrained_samples/",\
            snapname = "gadget_full_forward_512/snapshot_001",\
            snapnameRev = "gadget_full_reverse_512/snapshot_001",\
            reCentreSnaps = False,N=512,boxsize=677.7,mMin = 1e14,mMax = 1e15,\
            rMin=5,rMax=25,verbose=True,combineSims=False,\
            method="poisson",errorType = "Weighted",\
            unconstrainedCentreList = np.array([[0,0,0]]),\
            additionalConditions = additionalConditions)

[rBinStackCentres,nbarjSepStack,\
        sigmaSepStack,nbarjSepStackUn,sigmaSepStackUn,\
        nbarjAllStacked,sigmaAllStacked,nbarjAllStackedUn,sigmaAllStackedUn,\
        nbar,rMin,mMin,mMax] = getVoidProfilesData(\
            snapNumList,snapNumListUncon,\
            snapList = snapList,snapListRev = snapListRev,\
            samplesFolder="new_chain/",\
            unconstrainedFolder="new_chain/unconstrained_samples/",\
            snapname = "gadget_full_forward_512/snapshot_001",\
            snapnameRev = "gadget_full_reverse_512/snapshot_001",\
            reCentreSnaps = False,N=512,boxsize=677.7,mMin = 1e14,mMax = 1e15,\
            rMin=5,rMax=25,verbose=True,combineSims=False,\
            method="poisson",errorType = "Weighted",\
            unconstrainedCentreList = np.array([[0,0,0]]),\
            additionalConditions = None)

textwidth=7.1014
fontsize = 12
legendFontsize=10
fig, ax = plt.subplots(1,2,figsize=(textwidth,0.5*textwidth))


plot.plotConstrainedVsUnconstrainedProfiles(rBinStackCentresCombined,\
    nbarjSepStackCombined,sigmaSepStackCombined,\
    nbarjAllStackedUnCombined,sigmaAllStackedUnCombined,nbar,rMin,mMin,mMax,\
    fontsize = fontsize,legendFontSize=legendFontsize,\
    labelCon='Constrained',\
    labelRand='Unconstrained \nmean',\
    savename=figuresFolder + "profiles1415.pdf",\
    showTitle=True,ax = ax[0],title="Combined Catalogue",\
    meanErrorLabel = 'Unconstrained \nMean',\
    profileErrorLabel = 'Profile \nvariation \n',\
    nbarjUnconstrainedStacks=nbarjSepStackUn,legendLoc='lower right',\
    sigmajUnconstrainedStacks = sigmaSepStackUn,showMean=True,show=False)

plot.plotConstrainedVsUnconstrainedProfiles(rBinStackCentres,nbarjSepStack,\
    sigmaSepStack,nbarjAllStackedUn,sigmaAllStackedUn,nbar,rMin,mMin,mMax,\
    fontsize = fontsize,legendFontSize=legendFontsize,\
    labelCon='Constrained',\
    labelRand='Unconstrained \nmean',\
    savename=figuresFolder + "profiles1415.pdf",\
    showTitle=True,ax = ax[1],title="All Catalogues",\
    meanErrorLabel = 'Unconstrained \nMean',\
    profileErrorLabel = 'Profile \nvariation \n',\
    nbarjUnconstrainedStacks=nbarjSepStackUn,legendLoc='lower right',\
    sigmajUnconstrainedStacks = sigmaSepStackUn,showMean=True,show=False)

plt.tight_layout()
plt.show()


# Catalogues of different sizes:

[finalCat135,shortHaloList135,twoWayMatchList135,finalCandidates135,\
    finalRatios135,finalDistances135,allCandidates135,candidateCounts135] = \
    constructAntihaloCatalogue(snapNumList,snapList=snapList,\
    snapListRev=snapListRev,ahProps=ahProps,hrList=hrList,max_index=None,\
    twoWayOnly=True,blockDuplicates=True,\
    crossMatchThreshold = muOpt,distMax = rSearchOpt,rSphere=135)

ahCounts135 = np.array([len(tools.getAntiHalosInSphere(antihaloCentres[k],\
        135,filterCondition = (antihaloRadii[k] > rMin) & \
        (antihaloRadii[k] <= rMax))[0]) for k in range(0,len(snapNumList))])

centralAntihalos135 = [tools.getAntiHalosInSphere(antihaloCentres[k],135,\
        filterCondition = (antihaloRadii[k] > rMin) & \
        (antihaloRadii[k] <= rMax)) for k in range(0,len(snapNumList))]
centralAntihaloMasses135 = [\
        antihaloMasses[k][centralAntihalos135[k][0]] \
        for k in range(0,len(centralAntihalos135))]
sortedList135 = [np.flip(np.argsort(centralAntihaloMasses135[k])) \
        for k in range(0,len(snapNumList))]
centresListShort135 = [np.array([antihaloCentres[l][\
        centralAntihalos135[l][0][sortedList135[l][k]],:] \
        for k in range(0,np.min([ahCounts135[l],max_index]))]) \
        for l in range(0,len(snapNumList))]
radiiListShort135 = [np.array([antihaloRadii[l][\
        centralAntihalos135[l][0][sortedList135[l][k]]] \
        for k in range(0,np.min([ahCounts135[l],max_index]))]) \
        for l in range(0,len(snapNumList))]
massListShort135 = [np.array([antihaloMasses[l][\
        centralAntihalos135[l][0][sortedList135[l][k]]] \
        for k in range(0,np.min([ahCounts135[l],max_index]))]) \
        for l in range(0,len(snapNumList))]



[finalCat300,shortHaloList300,twoWayMatchList300,finalCandidates300,\
    finalRatios300,finalDistances300,allCandidates300,candidateCounts300] = \
    constructAntihaloCatalogue(snapNumList,snapList=snapList,\
    snapListRev=snapListRev,ahProps=ahProps,hrList=hrList,max_index=None,\
    twoWayOnly=True,blockDuplicates=True,\
    crossMatchThreshold = muOpt,distMax = rSearchOpt,rSphere=300)

ahCounts300 = np.array([len(tools.getAntiHalosInSphere(antihaloCentres[k],\
        300,filterCondition = (antihaloRadii[k] > rMin) & \
        (antihaloRadii[k] <= rMax))[0]) for k in range(0,len(snapNumList))])

centralAntihalos300 = [tools.getAntiHalosInSphere(antihaloCentres[k],300,\
        filterCondition = (antihaloRadii[k] > rMin) & \
        (antihaloRadii[k] <= rMax)) for k in range(0,len(snapNumList))]
centralAntihaloMasses300 = [\
        antihaloMasses[k][centralAntihalos300[k][0]] \
        for k in range(0,len(centralAntihalos300))]
sortedList300 = [np.flip(np.argsort(centralAntihaloMasses300[k])) \
        for k in range(0,len(snapNumList))]
centresListShort300 = [np.array([antihaloCentres[l][\
        centralAntihalos300[l][0][sortedList300[l][k]],:] \
        for k in range(0,np.min([ahCounts300[l],max_index]))]) \
        for l in range(0,len(snapNumList))]
radiiListShort300 = [np.array([antihaloRadii[l][\
        centralAntihalos300[l][0][sortedList300[l][k]]] \
        for k in range(0,np.min([ahCounts300[l],max_index]))]) \
        for l in range(0,len(snapNumList))]
massListShort300 = [np.array([antihaloMasses[l][\
        centralAntihalos300[l][0][sortedList300[l][k]]] \
        for k in range(0,np.min([ahCounts300[l],max_index]))]) \
        for l in range(0,len(snapNumList))]

# Reference unconstrained catalogue:

snapListUnconstrained = [pynbody.load(unconstrainedFolderNew + "sample" \
     + str(snapNum) + "/" + snapname) for snapNum in snapNumListUncon]
snapListUnconstrainedRev = [pynbody.load(unconstrainedFolderNew + \
         "sample" + str(snapNum) + "/" + snapnameRev) \
         for snapNum in snapNumListUncon]
ahPropsUnconstrained = [pickle.load(\
     open(snap.filename + ".AHproperties.p","rb")) \
     for snap in snapListUnconstrained]

hrListUn = [snap.halos() for snap in snapListUnconstrainedRev]

[randomCat300,randomHaloList300,randomTwoWayMatchList300,randomCandidates300,\
    randomRatios300,randomDistances300,randomAllCandidates300,\
    randomCandidateCounts300] = \
    constructAntihaloCatalogue(snapNumListUncon,snapList=snapListUnconstrained,\
    snapListRev=snapListUnconstrainedRev,ahProps=ahPropsUnconstrained,\
    hrList=hrListUn,max_index=None,\
    twoWayOnly=True,blockDuplicates=True,\
    crossMatchThreshold = muOpt,distMax = rSearchOpt,rSphere=300)


antihaloCentresRandom = [tools.remapAntiHaloCentre(props[5],boxsize) \
    for props in ahPropsUnconstrained]
antihaloMassesRandom = [props[3] for props in ahPropsUnconstrained]
antihaloRadiiRandom = [props[7] for props in ahPropsUnconstrained]
centralAntihalosRandom = [tools.getAntiHalosInSphere(antihaloCentresRandom[k],\
    rSphere,filterCondition = (antihaloRadiiRandom[k] > rMin) & \
        (antihaloRadiiRandom[k] <= rMax)) \
        for k in range(0,len(snapListUnconstrained))]
centralAntihaloMassesRandom = [\
        antihaloMassesRandom[k][centralAntihalosRandom[k][0]] \
        for k in range(0,len(centralAntihalosRandom))]
sortedListRandom = [np.flip(np.argsort(centralAntihaloMassesRandom[k])) \
        for k in range(0,len(snapListUnconstrained))]
ahCountsRandom = np.array([len(cahs[0]) for cahs in centralAntihalosRandom])

centresListShortRandom = [np.array([antihaloCentresRandom[l][\
        centralAntihalosRandom[l][0][sortedListRandom[l][k]],:] \
        for k in range(0,np.min([ahCountsRandom[l],len(randomCat300)]))]) \
        for l in range(0,len(snapListUnconstrained))]

radiiListShortRandom = [np.array([antihaloRadiiRandom[l][\
        centralAntihalosRandom[l][0][sortedListRandom[l][k]]] \
        for k in range(0,np.min([ahCountsRandom[l],len(randomCat300)]))]) \
        for l in range(0,len(snapListUnconstrained))]

massListShortRandom = [np.array([antihaloMassesRandom[l][\
        centralAntihalosRandom[l][0][sortedListRandom[l][k]]] \
        for k in range(0,np.min([ahCountsRandom[l],len(randomCat300)]))]) \
        for l in range(0,len(snapListUnconstrained))]


# Statistics for the unconstrained catalogue:
diffMap = [np.setdiff1d(np.arange(0,len(snapNumListUncon)),[k]) \
    for k in range(0,len(snapNumListUncon))]
ahCountsRandom = np.array([len(cahs[0]) for cahs in centralAntihalosRandom])
candCountStrippedRand = [randomCandidateCounts300[k][diffMap[k],:] \
    for k in range(0,len(snapNumListUncon))]
uniqueMatchFracRand = np.sum(\
    [[len(np.where(candCountStrippedRand[l][k,:] == 1)[0]) \
    for k in range(0,len(snapNumListUncon)-1)] \
    for l in range(0,len(snapNumListUncon))])/\
    np.sum(\
    [[ahCountsRandom[diffMap[l][k]] \
    for k in range(0,len(snapNumListUncon)-1)] \
    for l in range(0,len(snapNumListUncon))])
noMatchFracMatrixRand = np.mean(\
    [[len(np.where(candCountStrippedRand[l][k,:] == 0)[0])/\
    ahCountsRandom[diffMap[l][k]]
    for k in range(0,len(snapNumListUncon)-1)] \
    for l in range(0,len(snapNumListUncon))])
multiMatchFracMatrixRand = np.mean(\
    [[len(np.where(candCountStrippedRand[l][k,:] > 1)[0])/\
    ahCountsRandom[diffMap[l][k]] \
    for k in range(0,len(snapNumListUncon)-1)] \
    for l in range(0,len(snapNumListUncon))])
catFractionFinalMatrixRand = np.mean(\
    np.sum(randomCat300 > 0,1)/len(snapNumListUncon))
catFractionFinalAllRand = np.sum(randomCat300 > 0,1)/len(snapNumListUncon)


def getFractions(catalogue,ahCounts,candidateCounts):
    numCats = catalogue.shape[1]
    diffMap = [np.setdiff1d(np.arange(0,numCats),[k]) \
        for k in range(0,numCats)]
    candCountStripped = [candidateCounts[k][diffMap[k],:] \
        for k in range(0,catalogue.shape[1])]
    uniqueMatchFrac = np.mean(\
        [[len(np.where(candCountStripped[l][k,:] == 1)[0])/\
        ahCounts[diffMap[l][k]] \
        for k in range(0,numCats-1)] \
        for l in range(0,numCats)])
    noMatchFrac = np.mean(\
        [[len(np.where(candCountStripped[l][k,:] == 0)[0])/\
        ahCounts[diffMap[l][k]] \
        for k in range(0,numCats-1)] \
        for l in range(0,numCats)])
    multiMatchFrac = np.mean(\
        [[len(np.where(candCountStripped[l][k,:] > 1)[0])/\
        ahCounts[diffMap[l][k]] \
        for k in range(0,numCats-1)] \
        for l in range(0,numCats)])
    catFractionFinal = np.mean(\
        np.sum(catalogue > 0,1)/numCats)
    return [uniqueMatchFrac,noMatchFrac,multiMatchFrac,catFractionFinal]

[unique135,failed135,ambig135,catFrac135] = getFractions(finalCat135,\
    ahCounts135,candidateCounts135)

[unique300,failed300,ambig300,catFrac300] = getFractions(finalCat300,\
    ahCounts300,candidateCounts300)

# Statistics for the catalogue:
diffMap = [np.setdiff1d(np.arange(0,len(snapNumList)),[k]) \
    for k in range(0,len(snapNumList))]
ahCounts = np.array([len(cahs[0]) for cahs in centralAntihalos])
candCountStrippedOpt = [candidateCountsOpt[k][diffMap[k],:] \
    for k in range(0,len(snapNumList))]
uniqueMatchFracOpt = np.sum(\
    [[len(np.where(candCountStrippedOpt[l][k,:] == 1)[0]) \
    for k in range(0,len(snapNumList)-1)] \
    for l in range(0,len(snapNumList))])/\
    np.sum([[ahCounts[diffMap[l][k]] for k in range(0,len(snapNumList)-1)]\
    for l in range(0,len(snapNumList))])
noMatchFracMatrixOpt = np.mean(\
    [[len(np.where(candCountStrippedOpt[l][k,:] == 0)[0])/\
    ahCounts[diffMap[l][k]]
    for k in range(0,len(snapNumList)-1)] \
    for l in range(0,len(snapNumList))])
multiMatchFracMatrixOpt = np.mean(\
    [[len(np.where(candCountStrippedOpt[l][k,:] > 1)[0])/\
    ahCounts[diffMap[l][k]] \
    for k in range(0,len(snapNumList)-1)] \
    for l in range(0,len(snapNumList))])
#twoWayMatchFracMatrixOpt = np.mean([[np.sum(twoWayMatchListOpt[l][:,k]) \
#    for k in range(0,len(snapNumList)-1)] \
#    for l in range(0,len(snapNumList))])
catFractionFinalMatrixOpt = np.mean(\
    np.sum(finalCatOpt > 0,1)/len(snapNumList))
catFractionFinalAllOpt = np.sum(finalCatOpt > 0,1)/len(snapNumList)
catFractionFinalAll135 = np.sum(finalCat135 > 0,1)/len(snapNumList)


plt.clf()
plt.hist(catFractionFinalAllRand,bins=np.linspace(0.2,1.0,10),\
    label='Random Catalogue',alpha=0.5,density=True)
plt.hist(catFractionFinalAllOpt,bins=np.linspace(0.3,1.0,6),density=True,\
    label='Constrained Catalogue ($r < 300\\mathrm{\\,Mpc}h^{-1}$)',alpha=0.5)
plt.hist(catFractionFinalAll135,bins=np.linspace(0.3,1.0,6),density=True,\
    label='Constrained Catalogue ($r < 135\\mathrm{\\,Mpc}h^{-1}$)',alpha=0.5)
plt.xlabel('Catalogue Fraction')
plt.ylabel('Relative Abundance')
plt.title('Catalogue fraction distribution')
plt.legend()
plt.savefig("cat_fraction_histogram.pdf")
plt.show()

def getRadiiFromCat(catList,radiiList):
    radiiListOut = -np.ones(catList.shape,dtype=float)
    for k in range(0,len(catList)):
        for l in range(0,len(catList[0])):
            if catList[k,l] > 0:
                radiiListOut[k,l] = radiiList[l][catList[k,l]-1]
    return radiiListOut

def getCentresFromCat(catList,centresList,ns):
    centresListOut = np.zeros((len(catList),3),dtype=float)
    for k in range(0,len(catList)):
        if catList[k,ns] > 0:
            centresListOut[k,:] = centresList[ns][catList[k,ns]-1]
        else:
            centresListOut[k,:] = np.nan
    return centresListOut

def getMeanProperty(propertyList):
    meanProperty = np.zeros(len(propertyList))
    sigmaProperty = np.zeros(len(propertyList))
    for k in range(0,len(propertyList)):
        haveProperty = np.where(propertyList[k,:] > 0)[0]
        meanProperty[k] = np.mean(propertyList[k,haveProperty])
        sigmaProperty[k] = np.std(propertyList[k,haveProperty])/\
            np.sqrt(len(haveProperty))
    return [meanProperty,sigmaProperty]


# Determining catalogue fraction limit by radius bin:
radiiListShortRand = [[antihaloRadiiRandom[l][\
        centralAntihalosRandom[l][0][sortedListRandom[l][k]]] \
        for k in range(0,ahCountsRandom[l])] \
        for l in range(0,len(snapNumListUncon))]

radiiListCombRand = getRadiiFromCat(randomCat300,radiiListShortRand)
[radiiListMeanRand,radiiListSigmaRand] = getMeanProperty(radiiListCombRand)

rBins = np.linspace(5,23,nBins+1)



# Property variation with radius:
nBins = 7
rBins = np.linspace(5,23,nBins+1)
mBins = 10**(np.linspace(13,15,nBins+1))
catFractionFinalRadius = np.zeros(nBins)
catFractionFinalMass = np.zeros(nBins)
catFractionFinalRadiusErr = np.zeros(nBins)
catFractionFinalMassErr = np.zeros(nBins)
uniqueMatchFracMass = np.zeros(nBins)
uniqueMatchFracMassErr = np.zeros((2,nBins))
uniqueMatchFracRadius = np.zeros(nBins)
uniqueMatchFracRadiusErr = np.zeros((2,nBins))
radiiListOpt = getRadiiFromCat(finalCatOpt,radiiListShort)
massListOpt = getRadiiFromCat(finalCatOpt,massListShort)
[radiiMeanOpt, radiiSigmaOpt]  = getMeanProperty(radiiListOpt)
[massMeanOpt, massSigmaOpt]  = getMeanProperty(massListOpt)

catFractionRandRadius = np.zeros(nBins)
catFractionRandMass = np.zeros(nBins)
catFractionRandRadiusErr = np.zeros(nBins)
catFractionRandMassErr = np.zeros(nBins)
catFractionRandRadius99 = np.zeros(nBins)
catFractionRandMass99 = np.zeros(nBins)
uniqueMatchFracMassRand = np.zeros(nBins)
uniqueMatchFracMassRandErr = np.zeros((2,nBins))
uniqueMatchFracRadiusRand = np.zeros(nBins)
uniqueMatchFracRadiusRandErr = np.zeros((2,nBins))
radiiListRand = getRadiiFromCat(randomCat300,radiiListShortRandom)
massListRand = getRadiiFromCat(randomCat300,massListShortRandom)
[radiiMeanRand, radiiSigmaRand]  = getMeanProperty(radiiListRand)
[massMeanRand, massSigmaRand]  = getMeanProperty(massListRand)


for k in range(0,nBins):
    inRangeRad = np.where(\
        (radiiMeanOpt > rBins[k]) & (radiiMeanOpt <= rBins[k+1]))[0]
    inRangeMass = np.where(\
        (massMeanOpt > mBins[k]) & (massMeanOpt <= mBins[k+1]))[0]
    catFractionFinalRadius[k] = np.mean(\
        np.sum(finalCatOpt[inRangeRad,:] > 0,1)/len(snapNumList))
    catFractionFinalMass[k] = np.mean(\
        np.sum(finalCatOpt[inRangeMass,:] > 0,1)/len(snapNumList))
    catFractionFinalRadiusErr[k] = np.std(\
        np.sum(finalCatOpt[inRangeRad,:] > 0,1)/len(snapNumList))/\
        np.sqrt(len(inRangeRad))
    catFractionFinalMassErr[k] = np.std(\
        np.sum(finalCatOpt[inRangeMass,:] > 0,1)/len(snapNumList))/\
        np.sqrt(len(inRangeMass))
    inRangeRadAll = [np.where(\
        (radiiListShort[l] > rBins[k]) & \
        (radiiListShort[l] <= rBins[k+1]))[0] \
            for l in range(0,len(snapNumList))]
    inRangeMassAll = [np.where(\
        (massListShort[l] > mBins[k]) & \
        (massListShort[l] <= mBins[k+1]))[0] \
            for l in range(0,len(snapNumList))]
    numerator = np.sum(\
        [[len(np.where(candCountStrippedOpt[m][\
        l,inRangeMassAll[m]] == 1)[0]) \
        for l in range(0,len(snapNumList)-1)] \
        for m in range(0,len(snapNumList))])
    denominator = np.sum(\
        [[len(inRangeMassAll[m]) \
        for l in range(0,len(snapNumList)-1)] \
        for m in range(0,len(snapNumList))])
    errorRange = scipy.stats.poisson(numerator).interval(0.67)
    uniqueMatchFracMass[k] = numerator/denominator
    uniqueMatchFracMassErr[0,k] = (numerator - errorRange[0])/denominator
    uniqueMatchFracMassErr[1,k] = (errorRange[1] - numerator)/denominator
    numerator = np.sum(\
        [[len(np.where(candCountStrippedOpt[m][\
        l,inRangeRadAll[m]] == 1)[0]) \
        for l in range(0,len(snapNumList)-1)] \
        for m in range(0,len(snapNumList))])
    denominator = np.sum(\
        [[len(inRangeRadAll[m]) \
        for l in range(0,len(snapNumList)-1)] \
        for m in range(0,len(snapNumList))])
    errorRange = scipy.stats.poisson(numerator).interval(0.67)
    uniqueMatchFracRadius[k] = numerator/denominator
    uniqueMatchFracRadiusErr[0,k] = (numerator - errorRange[0])/denominator
    uniqueMatchFracRadiusErr[1,k] = (errorRange[1] - numerator)/denominator


for k in range(0,nBins):
    inRangeRad = np.where(\
        (radiiMeanRand > rBins[k]) & (radiiMeanRand <= rBins[k+1]))[0]
    inRangeMass = np.where(\
        (massMeanRand > mBins[k]) & (massMeanRand <= mBins[k+1]))[0]
    catFractionRandRadius[k] = np.mean(\
        np.sum(randomCat300[inRangeRad,:] > 0,1)/len(snapNumListUncon))
    if len(np.sum(randomCat300[inRangeRad,:] > 0,1)/len(snapNumListUncon)) > 0:
        catFractionRandRadius99[k] = np.percentile(\
            np.sum(randomCat300[inRangeRad,:] > 0,1)/len(snapNumListUncon),99)
    else:
        catFractionRandRadius99[k] = 0.0
    catFractionRandMass[k] = np.mean(\
        np.sum(randomCat300[inRangeMass,:] > 0,1)/len(snapNumListUncon))
    catFractionRandMass99[k] = np.percentile(\
        np.sum(randomCat300[inRangeMass,:] > 0,1)/len(snapNumListUncon),99)
    catFractionRandRadiusErr[k] = np.std(\
        np.sum(randomCat300[inRangeRad,:] > 0,1)/len(snapNumListUncon))/\
        np.sqrt(len(inRangeRad))
    catFractionRandMassErr[k] = np.std(\
        np.sum(randomCat300[inRangeMass,:] > 0,1)/len(snapNumListUncon))/\
        np.sqrt(len(inRangeMass))
    inRangeRadAll = [np.where(\
        (radiiListShortRandom[l] > rBins[k]) & \
        (radiiListShortRandom[l] <= rBins[k+1]))[0] \
            for l in range(0,len(snapNumListUncon))]
    inRangeMassAll = [np.where(\
        (massListShortRandom[l] > mBins[k]) & \
        (massListShortRandom[l] <= mBins[k+1]))[0] \
            for l in range(0,len(snapNumListUncon))]
    numerator = np.sum(\
        [[len(np.where(candCountStrippedRand[m][\
        l,inRangeMassAll[m]] == 1)[0]) \
        for l in range(0,len(snapNumListUncon)-1)] \
        for m in range(0,len(snapNumListUncon))])
    denominator = np.sum(\
        [[len(inRangeMassAll[m]) \
        for l in range(0,len(snapNumList)-1)] \
        for m in range(0,len(snapNumList))])
    errorRange = scipy.stats.poisson(numerator).interval(0.67)
    uniqueMatchFracMassRand[k] = numerator/denominator
    uniqueMatchFracMassRandErr[0,k] = (numerator - errorRange[0])/denominator
    uniqueMatchFracMassRandErr[1,k] = (errorRange[1] - numerator)/denominator
    numerator = np.sum(\
        [[len(np.where(candCountStrippedRand[m][\
        l,inRangeRadAll[m]] == 1)[0]) \
        for l in range(0,len(snapNumListUncon)-1)] \
        for m in range(0,len(snapNumListUncon))])
    denominator = np.sum(\
        [[len(inRangeRadAll[m]) \
        for l in range(0,len(snapNumListUncon)-1)] \
        for m in range(0,len(snapNumListUncon))])
    errorRange = scipy.stats.poisson(numerator).interval(0.67)
    uniqueMatchFracRadiusRand[k] = numerator/denominator
    uniqueMatchFracRadiusRandErr[0,k] = (numerator - errorRange[0])/denominator
    uniqueMatchFracRadiusRandErr[1,k] = (errorRange[1] - numerator)/denominator


plt.clf()
plt.errorbar(plot.binCentres(rBins),catFractionFinalRadius,\
    yerr=catFractionFinalRadiusErr,label='Constrained catalogue',\
    color=seabornColormap[0])
plt.errorbar(plot.binCentres(rBins),catFractionRandRadius,\
    yerr=catFractionRandRadiusErr,label='Random catalogue',\
    color=seabornColormap[1])
plt.errorbar(plot.binCentres(rBins),catFractionRandRadius99,\
    linestyle='--',color=seabornColormap[1],\
    label="Random catalogue (99th percentile)")
plt.axhline(catFractionFinalMatrixRand,linestyle='--',color='grey',\
    label='Random Catalogue average')
plt.xlabel('$R_{\\mathrm{eff}}$ ($\\mathrm{Mpc}h^{-1}$)')
plt.ylabel('Mean fraction of catalogues')
plt.legend()
plt.savefig(figuresFolder + "supporting_plots/cat_fraction_radius.pdf")
plt.show()

plt.clf()
plt.errorbar(plot.binCentres(mBins),catFractionFinalMass,\
    yerr=catFractionFinalMassErr,label='Constrained catalogue',\
    color=seabornColormap[0])
plt.errorbar(plot.binCentres(mBins),catFractionRandMass,\
    yerr=catFractionRandMassErr,label='Random catalogue',\
    color=seabornColormap[1])
plt.errorbar(plot.binCentres(rBins),catFractionRandMass99,\
    linestyle='--',color=seabornColormap[1],\
    label="Random catalogue (99th percentile)")
plt.axhline(catFractionFinalMatrixRand,linestyle='--',color='grey',\
    label='Random Catalogue average')
plt.xscale('log')
plt.xlabel('Mass ($M_{\\odot}h^{-1}$)')
plt.ylabel('Mean fraction of catalogues')
plt.legend()
plt.savefig(figuresFolder + "supporting_plots/cat_fraction_mass.pdf")
plt.show()

plt.clf()
plt.errorbar(plot.binCentres(rBins),uniqueMatchFracRadius,\
    yerr=uniqueMatchFracRadiusErr,label='Constrained catalogue',\
    color=seabornColormap[0])
plt.errorbar(plot.binCentres(rBins),uniqueMatchFracRadiusRand,\
    yerr=uniqueMatchFracRadiusRandErr,label='Random catalogue',\
    color=seabornColormap[1])
plt.axhline(uniqueMatchFracRand,linestyle='--',color='grey',\
    label='Random Catalogue')
plt.xlabel('$R_{\\mathrm{eff}}$ ($\\mathrm{Mpc}h^{-1}$)')
plt.ylabel('Unique match fraction')
plt.legend()
plt.savefig(figuresFolder + "supporting_plots/unique_fraction_radius.pdf")
plt.show()

plt.clf()
plt.errorbar(plot.binCentres(mBins),uniqueMatchFracMass,\
    yerr=uniqueMatchFracMassErr,label='Constrained catalogue',\
    color=seabornColormap[0])
plt.errorbar(plot.binCentres(mBins),uniqueMatchFracMassRand,\
    yerr=uniqueMatchFracMassRandErr,label='Random catalogue',\
    color=seabornColormap[1])
plt.axhline(uniqueMatchFracRand,linestyle='--',color='grey',\
    label='Random Catalogue')
plt.xscale('log')
plt.xlabel('Mass ($M_{\\odot}h^{-1}$)')
plt.ylabel('Unique match fraction')
plt.legend()
plt.savefig(figuresFolder + "supporting_plots/unique_fraction_mass.pdf")
plt.show()


def getPoissonSamples(lam,nSamples):
    if np.isscalar(lam):
        samples = np.random.poisson(lam,nSamples)
    else:
        samples = np.zeros((len(lam),nSamples),dtype=int)
        for k in range(0,len(lam)):
            samples[k,:] = np.random.poisson(lam[k],nSamples)
    return samples

def estimatePoissonRatioErrorbarMonteCarlo(n1,n2,errorVal = 0.67,seed = None,\
        nSamples=1000,returnSamples=False):
    interval1 = scipy.stats.poisson(n1).interval(errorVal)
    interval2 = scipy.stats.poisson(n2).interval(errorVal)
    np.random.seed(seed)
    # Generate random samples
    n1Samples = getPoissonSamples(n1,nSamples)
    n2Samples = getPoissonSamples(n2,nSamples)
    nonZero = np.where(n2Samples != 0)
    ratioValues = np.zeros(n1Samples.shape)
    ratioValues[nonZero] = n1Samples[nonZero]/n2Samples[nonZero]
    if np.isscalar(n1):
        lower = np.percentile(ratioValues,100*(0.5 - errorVal/2))
        upper = np.percentile(ratioValues,100*(0.5 + errorVal/2))
    else:
        lower = np.zeros(n1.shape)
        upper = np.zeros(n1.shape)
        for k in range(0,len(n1)):
            lower[k] = np.percentile(ratioValues[k,:],0.5 - errorVal/2)
            upper[k] = np.percentile(ratioValues[k,:],100*(0.5 + errorVal/2))
    returnList = [n1/n2,lower,upper]
    if returnSamples:
        returnList.append(ratioValues)
    return returnList


def getRadialUniqueFractions(rMax,rWidth,centresListShort,catalogue,\
        candidateCounts,errorInterval = 0.67,errorType="Monte Carlo",\
        nSamples = 10000,seed=None):
    numCats = catalogue.shape[1]
    diffMap = [np.setdiff1d(np.arange(0,numCats),[k]) \
        for k in range(0,numCats)]
    candCountStripped = [candidateCounts[k][diffMap[k],:] \
        for k in range(0,numCats)]
    radialBins = np.arange(0,rMax,rWidth)
    radialBinCentres = plot.binCentres(radialBins)
    distances = [np.sqrt(np.sum(centres**2,1)) \
        for centres in centresListShort]
    uniqueMatchFracDistAll = np.zeros(len(radialBinCentres))
    if errorType == "gaussian":
        uniqueMatchFracDistAllErr = np.zeros(len(radialBinCentres))
    else:
        uniqueMatchFracDistAllErr = np.zeros((2,len(radialBinCentres)))
    for k in range(0,len(radialBins)-1):
        inRangeDistAll = [np.where(\
            (distances[ns] > radialBins[k]) & \
            (distances[ns] <= radialBins[k+1]))[0] \
            for ns in range(0,numCats)]
        if errorType == "gaussian":
            uniqueMatchFracDistAll[k] = np.mean(\
                [[len(np.where(candCountStripped[m][\
                l,inRangeDistAll[m]] == 1)[0])/\
                np.max([1,len(inRangeDistAll[diffMap[m][l]])]) \
                for l in range(0,numCats-1)] \
                for m in range(0,numCats)])
            uniqueMatchFracDistAllErr[k] = np.std(\
                [[len(np.where(candCountStripped[m][\
                l,inRangeDistAll[m]] == 1)[0])/\
                np.max([1,len(inRangeDistAll[diffMap[m][l]])]) \
                for l in range(0,numCats-1)] \
                for m in range(0,numCats)])/\
                np.sqrt(numCats*(numCats-1))
        elif errorType == "Poisson":
            numerator = np.sum(\
                [[len(np.where(candCountStripped[m][\
                l,inRangeDistAll[m]] == 1)[0]) \
                for l in range(0,numCats-1)] \
                for m in range(0,numCats)])
            denominator = np.max([1,np.sum([[\
                len(inRangeDistAll[m]) \
                for l in range(0,numCats-1)] \
                for m in range(0,numCats)])])
            interval = scipy.stats.poisson(numerator).interval(errorInterval)
            uniqueMatchFracDistAll[k] = numerator/denominator
            uniqueMatchFracDistAllErr[0,k] = (numerator - interval[0])/\
                denominator
            uniqueMatchFracDistAllErr[1,k] = (interval[1] - numerator)/\
                denominator
        elif errorType == "Monte Carlo":
            #numerators = np.array([[len(np.where(candCountStripped[m][\
            #    l,inRangeDistAll[m]] == 1)[0]) \
            #    for l in range(0,numCats-1)] \
            #    for m in range(0,numCats)]).reshape(numCats*(numCats-1))
            #denominators = np.array(\
            #    [[len(inRangeDistAll[diffMap[m][l]]) \
            #    for l in range(0,numCats-1)] \
            #    for m in range(0,numCats)]).reshape(numCats*(numCats-1))
            #[rat,lower,upper] = estimatePoissonRatioErrorbarMonteCarlo(\
            #    numerators,denominators,errorVal = errorInterval,\
            #    nSamples=nSamples,seed=seed)
            #finite = np.isfinite(rat) & (rat != 0.0)
            #meanError = (lower[finite] + upper[finite])/2
            #uniqueMatchFracDistAll[k] = stacking.weightedMean(rat[finite],\
            #    1.0/meanError**2)
            #uniqueMatchFracDistAllErr[k] = np.sqrt(\
            #    np.sum((rat[finite] - uniqueMatchFracDistAll[k])**2\
            #        /meanError**4)/\
            #        np.sum(1.0/meanError**2)**2)
            numerator = np.sum(\
                [[len(np.where(candCountStripped[m][\
                l,inRangeDistAll[m]] == 1)[0]) \
                for l in range(0,numCats-1)] \
                for m in range(0,numCats)])
            denominator = np.max([1,np.sum([[\
                len(inRangeDistAll[m]) \
                for l in range(0,numCats-1)] \
                for m in range(0,numCats)])])
            [rat,lower,upper] = estimatePoissonRatioErrorbarMonteCarlo(\
                numerator,denominator,errorVal = errorInterval,\
                nSamples=nSamples,seed=seed)
            uniqueMatchFracDistAll[k] = numerator/denominator
            uniqueMatchFracDistAllErr[0,k] = rat - lower
            uniqueMatchFracDistAllErr[1,k] = upper - rat
        else:
            raise Exception("Unrecognised error type")
    return [radialBinCentres,uniqueMatchFracDistAll,uniqueMatchFracDistAllErr]

def getRadialCatalogueFractions(rMax,rWidth,centresListShort,catalogue,\
        candidateCounts):
    numCats = catalogue.shape[1]
    diffMap = [np.setdiff1d(np.arange(0,numCats),[k]) \
        for k in range(0,numCats)]
    candCountStripped = [candidateCounts[k][diffMap[k],:] \
        for k in range(0,numCats)]
    radialBins = np.arange(0,rMax,rWidth)
    radialBinCentres = plot.binCentres(radialBins)
    distances = [np.sqrt(np.sum(centres**2,1)) \
        for centres in centresListShort]
    [distancesOpt,distErrOpt] = getMeanProperty(\
        getRadiiFromCat(catalogue,distances))
    catFractionFinalDistanceAll = np.zeros(len(radialBinCentres))
    catFractionFinalDistanceAllErr = np.zeros(len(radialBinCentres))
    for k in range(0,len(radialBins)-1):
        inRangeDist = np.where(\
            (distancesOpt > radialBins[k]) & \
            (distancesOpt <= radialBins[k+1]))[0]
        catFractionFinalDistanceAll[k] = np.mean(\
            np.sum(catalogue[inRangeDist,:] > 0,1)/numCats)
        catFractionFinalDistanceAllErr[k] = np.std(\
            np.sum(catalogue[inRangeDist,:] > 0,1)/numCats)/\
            np.sqrt(len(inRangeDist))
    return [radialBinCentres,catFractionFinalDistanceAll,\
        catFractionFinalDistanceAllErr]

[radialBinCentres135,uniqueMatchFracDistAll135,uniqueMatchFracDistAll135Err] = \
    getRadialUniqueFractions(135,20,centresListShort135,finalCat135,\
    candidateCounts135,errorType="Monte Carlo")

[radialBinCentres300,uniqueMatchFracDistAll300,uniqueMatchFracDistAll300Err] = \
    getRadialUniqueFractions(300,20,centresListShort300,finalCat300,\
    candidateCounts300,errorType="Monte Carlo")

numCats = len(snapNumList)
radialBins = np.arange(0,rSphere,10)

distances135 = [np.sqrt(np.sum(centres**2,1)) \
        for centres in centresListShort135]
[distancesOpt135,distErrOpt135] = getMeanProperty(\
        getRadiiFromCat(finalCat135,distances135))
inRangeDist135 = np.where(\
            (distancesOpt135 > radialBins[k]) & \
            (distancesOpt135 <= radialBins[k+1]))[0]
inRangeDistAll135 = [np.where(\
            (distances135[ns] > radialBins[k]) & \
            (distances135[ns] <= radialBins[k+1]))[0] \
            for ns in range(0,numCats)]


distances300 = [np.sqrt(np.sum(centres**2,1)) \
        for centres in centresListShort300]
[distancesOpt300,distErrOpt300] = getMeanProperty(\
        getRadiiFromCat(finalCat300,distances300))
inRangeDist300 = np.where(\
            (distancesOpt300 > radialBins[k]) & \
            (distancesOpt300 <= radialBins[k+1]))[0]
inRangeDistAll300 = [np.where(\
            (distances300[ns] > radialBins[k]) & \
            (distances300[ns] <= radialBins[k+1]))[0] \
            for ns in range(0,numCats)]

count300 = [len(ind) for ind in inRangeDistAll300]
count135 = [len(ind) for ind in inRangeDistAll135]

# Uniqueness as a function of distance?
radialBinCentres = plot.binCentres(radialBins)
distances = [np.sqrt(np.sum(centres**2,1)) \
    for centres in centresListShort]
[distancesOpt,distErrOpt] = getMeanProperty(\
    getRadiiFromCat(finalCatOpt,distances))

uniqueMatchFracDist = np.zeros((len(radialBinCentres),len(rBins)-1))
uniqueMatchFracDistErr = np.zeros((len(radialBinCentres),len(rBins)-1))
catFractionFinalDistance = np.zeros((len(radialBinCentres),len(rBins)-1))
catFractionFinalDistanceErr = np.zeros((len(radialBinCentres),len(rBins)-1))

uniqueMatchFracDistAll = np.zeros(len(radialBinCentres))
uniqueMatchFracDistAllErr = np.zeros(len(radialBinCentres))
catFractionFinalDistanceAll = np.zeros(len(radialBinCentres))
catFractionFinalDistanceAllErr = np.zeros(len(radialBinCentres))

for k in range(0,len(radialBins)-1):
    inRangeDist = np.where(\
            (distancesOpt > radialBins[k]) & \
            (distancesOpt <= radialBins[k+1]))[0]
    inRangeDistAll = [np.where(\
                (distances[ns] > radialBins[k]) & \
                (distances[ns] <= radialBins[k+1]))[0] \
                for ns in range(0,len(snapNumList))]
    catFractionFinalDistanceAll[k] = np.mean(\
            np.sum(finalCatOpt[inRangeDist,:] > 0,1)/len(snapNumList))
    catFractionFinalDistanceAllErr[k] = np.std(\
        np.sum(finalCatOpt[inRangeDist,:] > 0,1)/len(snapNumList))/\
        np.sqrt(len(inRangeDist))
    uniqueMatchFracDistAll[k] = np.mean(\
        [[len(np.where(candCountStrippedOpt[m][\
        l,inRangeDistAll[m]] == 1)[0])/\
        np.max([1,len(inRangeDistAll[m])]) \
        for l in range(0,len(snapNumList)-1)] \
        for m in range(0,len(snapNumList))])
    uniqueMatchFracDistAllErr[k] = np.std(\
        [[len(np.where(candCountStrippedOpt[m][\
        l,inRangeDistAll[m]] == 1)[0])/\
        np.max([1,len(inRangeDistAll[m])]) \
        for l in range(0,len(snapNumList)-1)] \
        for m in range(0,len(snapNumList))])/\
        np.sqrt(len(snapNumList)*(len(snapNumList)-1))
    for l in range(0,len(rBins)-1):
        inRangeDist = np.where(\
            (distancesOpt > radialBins[k]) & \
            (distancesOpt <= radialBins[k+1]) & \
            (radiiMeanOpt > rBins[l]) & (radiiMeanOpt <= rBins[l+1]))[0]
        inRangeDistAll = [np.where(\
                (distances[ns] > radialBins[k]) & \
                (distances[ns] <= radialBins[k+1]) & \
                (radiiListShort[ns] > rBins[l]) & \
                (radiiListShort[ns] <= rBins[l+1]))[0] \
                for ns in range(0,len(snapNumList))]
        catFractionFinalDistance[k,l] = np.mean(\
            np.sum(finalCatOpt[inRangeDist,:] > 0,1)/len(snapNumList))
        catFractionFinalDistanceErr[k,l] = np.std(\
            np.sum(finalCatOpt[inRangeDist,:] > 0,1)/len(snapNumList))/\
            np.sqrt(len(inRangeDist))
        uniqueMatchFracDist[k,l] = np.mean(\
            [[len(np.where(candCountStrippedOpt[m][\
            l,inRangeDistAll[m]] == 1)[0])/\
            np.max([1,len(inRangeDistAll[m])]) \
            for l in range(0,len(snapNumList)-1)] \
            for m in range(0,len(snapNumList))])
        uniqueMatchFracDistErr[k,l] = np.std(\
            [[len(np.where(candCountStrippedOpt[m][\
            l,inRangeDistAll[m]] == 1)[0])/\
            np.max([1,len(inRangeDistAll[m])]) \
            for l in range(0,len(snapNumList)-1)] \
            for m in range(0,len(snapNumList))])/\
            np.sqrt(len(snapNumList)*(len(snapNumList)-1))


[radialBinCentres,uniqueMatchFracDistAll,uniqueMatchFracDistAllErr] = \
    getRadialUniqueFractions(135,20,centresListShort,finalCatOpt,\
    candidateCountsOpt,errorType="Monte Carlo")

[radialBinCentreOpt,catFractionFinalDistance135,catFractionFinalDistance135Err] = \
    getRadialCatalogueFractions(135,20,centresListShort135,finalCat135,\
    candidateCounts135)

[radialBinCentre300,catFractionFinalDistance300,catFractionFinalDistance300Err] = \
    getRadialCatalogueFractions(300,20,centresListShort300,finalCat300,\
    candidateCounts300)

[radialBinCentresRand,uniqueMatchFracDistAllRand,\
    uniqueMatchFracDistAllRandErr] = \
    getRadialUniqueFractions(300,20,centresListShortRandom,randomCat300,\
    randomCandidateCounts300,errorType="Monte Carlo")

[radialBinCentreRand,catFractionFinalDistanceRand,\
    catFractionFinalDistanceRandErr] = \
    getRadialCatalogueFractions(300,20,centresListShortRandom,randomCat300,\
    randomCandidateCounts300)



plt.clf()
showBins = False
if showBins:
    for l in range(0,len(rBins)-1):
        plt.errorbar(radialBinCentres,uniqueMatchFracDist[:,l],\
            yerr=uniqueMatchFracDistErr[:,l],label="$" + ("%.2g" % rBins[l]) + \
            " < R_{\\mathrm{eff}}/(\\mathrm{Mpc}h^{-1}) \\leq " + \
            ("%.2g" % rBins[l+1]) + "$")
        plt.errorbar(radialBinCentres,uniqueMatchFracDist[:,l],\
            yerr=uniqueMatchFracDistErr[:,l],label="$" + ("%.2g" % rBins[l]) + \
            " < R_{\\mathrm{eff}}/(\\mathrm{Mpc}h^{-1}) \\leq " + \
            ("%.2g" % rBins[l+1]) + "$")
else:
    plt.errorbar(radialBinCentres,uniqueMatchFracDistAll,\
        yerr = uniqueMatchFracDistAllErr,label='Constrained Catalogue')
    plt.errorbar(radialBinCentresRand,uniqueMatchFracDistAllRand,\
        yerr = uniqueMatchFracDistAllRandErr,label = 'Random Catalogue')

plt.axhline(uniqueMatchFracRand,linestyle='--',color='grey',\
    label='Random Catalogue')
plt.xlabel('Distance from Centre ($\\mathrm{Mpc}h^{-1}$)')
plt.ylabel('Unique match fraction')
if showBins:
    plt.legend()

plt.savefig(figuresFolder + "supporting_plots/unique_fraction_distance.pdf")
plt.show()


plt.clf()
showBins = False
if showBins:
    for l in range(0,len(rBins)-1):
        plt.errorbar(radialBinCentres300,uniqueMatchFracDistAll300[:,l],\
            yerr=uniqueMatchFracDistErr[:,l],label="$" + ("%.2g" % rBins[l]) + \
            " < R_{\\mathrm{eff}}/(\\mathrm{Mpc}h^{-1}) \\leq " + \
            ("%.2g" % rBins[l+1]) + "$")
else:
    plt.errorbar(radialBinCentres300,uniqueMatchFracDistAll300,\
        yerr = uniqueMatchFracDistAll300Err,label='Constrained Catalogue')
    plt.errorbar(radialBinCentres300,uniqueMatchFracDistAllRand,\
        yerr = uniqueMatchFracDistAllRandErr,label='Random Catalogue')

plt.axhline(uniqueMatchFracRand,linestyle='--',color='grey',\
    label='Random Catalogue (average)')
plt.xlabel('Distance from Centre ($\\mathrm{Mpc}h^{-1}$)')
plt.ylabel('Unique match fraction')
plt.legend()
plt.savefig(figuresFolder + "supporting_plots/unique_fraction_distance_300.pdf")
plt.show()

plt.clf()
showBins = False
if showBins:
    for l in range(0,len(rBins)-1):
        plt.errorbar(radialBinCentres135,uniqueMatchFracDistAll135[:,l],\
            yerr=uniqueMatchFracDistErr[:,l],label="$" + ("%.2g" % rBins[l]) + \
            " < R_{\\mathrm{eff}}/(\\mathrm{Mpc}h^{-1}) \\leq " + \
            ("%.2g" % rBins[l+1]) + "$")
else:
    plt.errorbar(radialBinCentres135,uniqueMatchFracDistAll135,\
        yerr = uniqueMatchFracDistAll135Err)

plt.axhline(uniqueMatchFracRand,linestyle='--',color='grey',\
    label='Random Catalogue')
plt.xlabel('Distance from Centre ($\\mathrm{Mpc}h^{-1}$)')
plt.ylabel('Unique match fraction')

plt.legend()
plt.savefig(figuresFolder + "supporting_plots/unique_fraction_distance_135.pdf")
plt.show()


plt.clf()
plt.errorbar(radialBinCentres300,catFractionFinalDistanceAll,\
    yerr=catFractionFinalDistanceAllErr,label='Constrained Catalogue')
plt.axhline(catFractionFinalMatrixRand,linestyle='--',color='grey',\
    label='Random Catalogue')
plt.xlabel('Distance from Centre ($\\mathrm{Mpc}h^{-1}$)')
plt.ylabel('Mean catalogue fraction')
plt.legend()
plt.savefig(figuresFolder + "supporting_plots/cat_fraction_distance.pdf")
plt.show()



plt.clf()
plt.errorbar(radialBinCentres300,catFractionFinalDistance300,\
    yerr=catFractionFinalDistance300Err,label='Constrained Catalogue')
plt.errorbar(radialBinCentres300,catFractionFinalDistanceRand,\
    yerr=catFractionFinalDistanceRandErr,label='Random Catalogue')
#plt.errorbar(radialBinCentres135,catFractionFinalDistance135,\
#    yerr=catFractionFinalDistance135Err)
plt.axhline(catFractionFinalMatrixRand,linestyle='--',color='grey',\
    label='Random Catalogue')
plt.xlabel('Distance from Centre ($\\mathrm{Mpc}h^{-1}$)')
plt.ylabel('Mean catalogue fraction')
plt.legend()
plt.savefig(figuresFolder + "supporting_plots/cat_fraction_distance_300.pdf")
plt.show()



# Test of catalogue changes with N:
nCatsList = range(2,7)
nCatNums = len(nCatsList)
catNList = []
uniqueMatchFractionArray = np.zeros(nCatNums)
noMatchFracMatrixArray = np.zeros(nCatNums)
multiMatchFracMatrixArray = np.zeros(nCatNums)
catFractionFinalMatrixArray = np.zeros(nCatNums)
catFractionFinalAllArray = []
strippedCandList = []
diffMapList = []
for nc in range(0,nCatNums):
    nCats = nCatsList[nc]
    snapNumListShort = [snapNumList[k] for k in range(0,nCats)]
    snapListShort = [snapList[k] for k in range(0,nCats)]
    snapListRevShort = [snapListRev[k] for k in range(0,nCats)]
    hrListShort = [hrList[k] for k in range(0,nCats)]
    ahPropsShort = [ahProps[k] for k in range(0,nCats)]
    [finalCatShort,shortHaloListShort,twoWayMatchListShort,\
        finalCandidatesShort,finalRatiosShort,\
        finalDistancesShort,allCandidatesShort,candidateCountsShort] = \
        constructAntihaloCatalogue(snapNumListShort,snapList=snapListShort,\
        snapListRev=snapListRevShort,ahProps=ahPropsShort,hrList=hrListShort,\
        max_index=None,twoWayOnly=True,blockDuplicates=True,\
        crossMatchThreshold = muOpt,distMax = rSearchOpt,rSphere=135)
    catNList.append([finalCatShort,shortHaloListShort,twoWayMatchListShort,\
        finalCandidatesShort,finalRatiosShort,\
        finalDistancesShort,allCandidatesShort,candidateCountsShort])
    # Properties:
    antihaloCentresShort = [tools.remapAntiHaloCentre(props[5],boxsize) \
        for props in ahPropsShort]
    antihaloMassesShort = [props[3] for props in ahPropsShort]
    antihaloRadiiShort = [props[7] for props in ahPropsShort]
    centralAntihalosShort = [tools.getAntiHalosInSphere(\
            antihaloCentresShort[k],\
            135,filterCondition = (antihaloRadiiShort[k] > rMin) & \
            (antihaloRadiiShort[k] <= rMax)) \
            for k in range(0,len(snapNumListShort))]
    centralAntihaloMassesShort = [\
            antihaloMasses[k][centralAntihalosShort[k][0]] \
            for k in range(0,len(centralAntihalosShort))]
    sortedListShort = [np.flip(np.argsort(centralAntihaloMassesShort[k])) \
            for k in range(0,len(snapNumListShort))]
    ahCountsShort = np.array([len(cahs[0]) for cahs in centralAntihalosShort])
    centresListShort = [np.array([antihaloCentresShort[l][\
            centralAntihalosShort[l][0][sortedListShort[l][k]],:] \
            for k in range(0,np.min([ahCountsShort[l],max_index]))]) \
            for l in range(0,len(snapNumListShort))]
    radiiListShort = [np.array([antihaloRadiiShort[l][\
            centralAntihalosShort[l][0][sortedListShort[l][k]]] \
            for k in range(0,np.min([ahCountsShort[l],max_index]))]) \
            for l in range(0,len(snapNumListShort))]
    massListShort = [np.array([antihaloMassesShort[l][\
            centralAntihalosShort[l][0][sortedListShort[l][k]]] \
            for k in range(0,np.min([ahCountsShort[l],max_index]))]) \
            for l in range(0,len(snapNumListShort))]
    # Statistics for the catalogue:
    diffMap = [np.setdiff1d(np.arange(0,len(snapNumListShort)),[k]) \
        for k in range(0,len(snapNumListShort))]
    diffMapList.append(diffMap)
    candCountStrippedShort = [candidateCountsShort[k][diffMap[k],:] \
        for k in range(0,len(snapNumListShort))]
    strippedCandList.append(candCountStrippedShort)
    uniqueMatchFractionArray[nc] = np.sum(\
        [[len(np.where(candCountStrippedShort[l][k,:] == 1)[0]) \
        for k in range(0,len(snapNumListShort)-1)] \
        for l in range(0,len(snapNumListShort))])/\
        np.sum(\
        [[ahCountsShort[diffMap[l][k]] \
        for k in range(0,len(snapNumListShort)-1)] \
        for l in range(0,len(snapNumListShort))])
    noMatchFracMatrixArray[nc] = np.sum(\
        [[len(np.where(candCountStrippedShort[l][k,:] == 0)[0]) \
        for k in range(0,len(snapNumListShort)-1)] \
        for l in range(0,len(snapNumListShort))])/\
        np.sum(\
        [[ahCountsShort[diffMap[l][k]] \
        for k in range(0,len(snapNumListShort)-1)] \
        for l in range(0,len(snapNumListShort))])
    multiMatchFracMatrixArray[nc] = np.sum(\
        [[len(np.where(candCountStrippedShort[l][k,:] > 1)[0]) \
        for k in range(0,len(snapNumListShort)-1)] \
        for l in range(0,len(snapNumListShort))])/\
        np.sum(\
        [[ahCountsShort[diffMap[l][k]] \
        for k in range(0,len(snapNumListShort)-1)] \
        for l in range(0,len(snapNumListShort))])
    catFractionFinalMatrixArray[nc] = np.mean(\
        np.sum(finalCatShort > 0,1)/len(snapNumListShort))
    catFractionFinalAllArray.append(np.sum(finalCatShort > 0,1)\
        /len(snapNumListShort))




# Signal to noise calculation:
[mcmcArray,num,N,NCAT,no_bias_params,bias_matrix,mean_field,\
    std_field,hmc_Elh,hmc_Eprior,hades_accept_count,\
    hades_attempt_count] = pickle.load(open("chain_properties.p","rb"))

snrField = mean_field**2/std_field**2
snrWeighted = (snrField/(1.0 + mean_field))/np.sum(1.0/(1.0 + mean_field))
snrWeights = 1.0/(1.0 + mean_field)**2
snrWeightsLin = snrWeights.reshape(256**3)
mean_fieldLin = mean_field.reshape(256**3)
#snrWeighted = snrField
finalCentresOptList = np.array([getCentresFromCat(\
    finalCatOpt,centresListShort,ns) for ns in range(0,len(snapNumList))])

meanCentreOpt = np.nanmean(finalCentresOptList,0)
#catFractionsOpt = np.array([len(np.where(x > 0)[0])/len(snapNumList) \
#    for x in finalCatOpt])
#catFractionsOpt = finalCatFracOpt
catFractionsOpt = finalCombinatoricFracOpt
grid = snapedit.gridListPermutation(256,perm=(2,1,0))
centroids = grid*boxsize/256 + boxsize/(2*256)
positions = snapedit.unwrap(centroids - np.array([boxsize/2]*3),boxsize)
tree = scipy.spatial.cKDTree(snapedit.wrap(positions + boxsize/2,boxsize),\
    boxsize=boxsize)

nearestPoints = tree.query_ball_point(\
    snapedit.wrap(meanCentreOpt + boxsize/2,boxsize),radiiMeanOpt,workers=-1)
#nearestPoints = tree.query_ball_point(\
#    snapedit.wrap(meanCentreOpt + boxsize/2,boxsize),50,workers=-1)
snrFieldLin = np.reshape(snrField,256**3)
snrWeightedLin = np.reshape(snrWeighted,256**3)
snrList = np.array([np.mean(snrFieldLin[points]) for points in nearestPoints])
# Volume weighted mean:

#snrList = np.array([stacking.weightedMean(snrFieldLin[points],\
#    snrWeightsLin[points]) for points in nearestPoints])


#snrListError = np.sqrt(np.array([stacking.weightedVariance(snrFieldLin[points],\
#    snrWeightsLin[points]) for points in nearestPoints]))

# SNR distribution of voids:
mBins = [1e13,1e14,5e14,1e16]
rMaxInclude = 135
lowDistance = np.sqrt(np.sum(meanCentreOpt**2,1)) < rMaxInclude
[inMassBins,noInMassBins] = plot_utilities.binValues(massMeanOpt[lowDistance],\
    mBins)
for k in range(0,len(mBins)-1):
    plt.hist(snrList[lowDistance][inMassBins[k]],bins=10**np.linspace(-2,4,21),\
        alpha=0.5,label="$" + plot.scientificNotation(mBins[k]) + \
        " < M/M_{\\odot} \leq " + \
        plot.scientificNotation(mBins[k+1]) + "$",\
        density=True)

plt.xscale('log')
plt.xlabel('SNR')
plt.ylabel('Density')
plt.legend()
plt.title("Void SNR within $" + ("%.3g" % rMaxInclude) + \
    "\\,\\mathrm{Mpc}h^{=1}$")
plt.show()




# SNR with distance???
rBins = np.linspace(0,300,31)
rBinCentres = plot_utilities.binCentres(rBins)
indDist = tree.query_ball_point(np.tile(np.array([boxsize/2]*3),(30,1)),\
    rBins[1:])
snrDist = np.zeros(len(rBinCentres))
snrDistErr = np.zeros(len(rBinCentres))
for k in range(0,len(rBinCentres)):
    if k == 0:
        ind = indDist[k]
    else:
        ind = np.intersect1d(indDist[k],indDist[k-1])
    snrDist[k] = np.mean(snrFieldLin[ind])
    snrDistErr[k] = np.std(snrFieldLin[ind])/np.sqrt(len(ind))

plt.plot(rBinCentres,snrDist)
plt.ylabel('SNR ($\\delta^2/\\sigma_{\\delta}^2$)')
plt.xlabel('Shell distance from origin ($\\mathrm{Mpc}h^{-1}$)')
plt.show()



#snrBins = np.linspace(0,50,21)
rMaxInclude = 135
lowDistance = np.sqrt(np.sum(meanCentreOpt**2,1)) < rMaxInclude
snrBins = 10**np.linspace(-1,2,11)
snrBinCentres = plot_utilities.binCentres(snrBins)
[inListSNR,noInBinsSNR] = plot_utilities.binValues(snrList[lowDistance],snrBins)
meanCatFracSNR = np.array([np.mean(catFractionsOpt[ind]) for ind in inListSNR])
stdCatFracSNR = np.array([np.std(catFractionsOpt[ind])/np.sqrt(len(ind)) \
    for ind in inListSNR])

mBins = [1e13,1e14,5e14,1e16]
[inMassBins,noInMassBins] = plot_utilities.binValues(massMeanOpt,mBins)

meanCatFracBinsSNR = [np.array(\
    [np.mean(catFractionsOpt[np.intersect1d(ind,inMassBins[k])]) \
    for ind in inListSNR]) \
    for k in range(0,len(mBins) - 1)]
stdCatFracBinsSNR = [np.array(\
    [np.std(catFractionsOpt[np.intersect1d(ind,inMassBins[k])])/\
    np.sqrt(len(np.intersect1d(ind,inMassBins[k]))) \
    for ind in inListSNR]) \
    for k in range(0,len(mBins) - 1)]
#meanCatFracBinsSNR = np.array([np.mean(catFracOpt[ind]) for ind in inMassBins])

for k in range(0,len(mBins)-1):
    plt.errorbar(snrBinCentres,meanCatFracBinsSNR[k],yerr=stdCatFracBinsSNR[k],\
        label = "$" + plot.scientificNotation(mBins[k]) + \
        " < M/M_{\\odot} \leq " + \
        plot.scientificNotation(mBins[k+1]) + "$")

plt.xlabel('Mean SNR of voids')
plt.ylabel('Mean Catalogue Fraction')
plt.xscale('log')
plt.legend()
plt.show()


centralPortion = tree.query_ball_point(np.array([0]*3),135)

# Relationship between density and snr:
hist = plt.hist2d(1.0 + mean_fieldLin[centralPortion],\
    snrFieldLin[centralPortion],\
    bins = [10**np.linspace(-1,1,21),10**(np.linspace(-2,4,21))],cmap='Blues')
#plt.imshow(hist[0],extent = (-1,1,1e-2,1e4),aspect='auto')
plt.yscale('log')
plt.xscale('log')
plt.xlabel('$\\rho/\\bar{\\rho}$')
plt.ylabel('SNR ($\\delta^2/\\sigma_{\\delta}^2$)')
plt.title('Signal to Noise vs Density')
plt.colorbar()
plt.tight_layout()
plt.show()


plt.errorbar(snrBinCentres,meanCatFracSNR,yerr=stdCatFracSNR)
plt.xlabel('Mean SNR of void')
plt.ylabel('Mean Catalogue Fraction')
plt.show()


plt.clf()
plt.scatter(snrList,catFractionsOpt)
plt.xlabel('Signal-to-noise ratio')
plt.ylabel('Catalogue fraction')
plt.savefig(figuresFolder + "supporting_plots/snr_catFrac_scatter.pdf")
plt.show()


# Binned plot:
snrBins = np.linspace(np.max(snrList),np.min(snrList),21)
meanCatFracSNR = np.zeros(len(snrBins)-1)
stdCatFracSNR = np.zeros(len(snrBins)-1)
for k in range(0,len(snrBins)-1):
    inRange = np.where((snrList > snrBins[k+1]) & (snrList <= snrBins[k]))[0]
    meanCatFracSNR[k] = np.mean(catFractionsOpt[inRange])
    stdCatFracSNR[k] = np.std(catFractionsOpt[inRange])/np.sqrt(len(inRange))


plt.clf()
plt.errorbar(plot.binCentres(snrBins),meanCatFracSNR,yerr=stdCatFracSNR)
plt.xlabel('Signal-to-noise Ratio')
plt.ylabel('Mean catalogue fraction')
plt.savefig(figuresFolder + "supporting_plots/snr_vs_catFrac.pdf")
plt.show()

# Distribution of catalogue fraction:

finalCatFracOpt
mBins = [1e13,1e14,5e14,1e16]
[inMassBins,noInMassBins] = plot_utilities.binValues(massMeanOpt,mBins)
plt.hist(finalCatFracOpt,bins=4,alpha=0.5,label='All',density=True)

for k in range(0,len(mBins)-1):
    plt.hist(finalCatFracOpt[inMassBins[k]],bins=4,alpha=0.5,\
        label="$" + plot.scientificNotation(mBins[k]) + \
        " < M/M_{\\odot} \leq " + \
        plot.scientificNotation(mBins[k+1]) + "$",density=True)

plt.xlabel('Catalogue Fraction')
plt.ylabel('Density')
plt.legend()
plt.show()


# Bias plots:




radiiListOpt = getRadiiFromCat(finalCatOpt,radiiListShort)
massListOpt = getRadiiFromCat(finalCatOpt,massListShort)
[radiiMeanOpt, radiiSigmaOpt]  = getMeanProperty(radiiListOpt)
[massMeanOpt, massSigmaOpt]  = getMeanProperty(massListOpt)
sortedRadiiOpt = np.flip(np.argsort(radiiMeanOpt))

# Using the particle overlap matching procedure:
[finalCatPyn,shortHaloListPyn,twoWayMatchListPyn,finalCandidatesPyn] = \
    constructAntihaloCatalogue(snapNumList,matchType='pynbody')

radiiList = np.array([[antihaloRadii[l][\
                centralAntihalos[l][0][sortedList[l][k]]] \
                for k in range(0,max_index)] \
                for l in range(0,len(snapNumList))]).transpose()

centresListShort = [np.array([antihaloCentres[l][\
        centralAntihalos[l][0][sortedList[l][k]]] \
        for k in range(0,max_index)]) \
        for l in range(0,len(snapNumList))]



radiiListOrderedPyn = getRadiiFromCat(finalCatPyn,radiiList)
radiiListOrdered = getRadiiFromCat(finalCatOpt,radiiList)

[radiiListMeanPyn,radiiListSigmaPyn] = getMeanProperty(radiiListOrderedPyn)
[radiiListMean,radiiListSigma] = getMeanProperty(radiiListOrdered)

sortedRadiiPyn = np.flip(np.argsort(radiiListMeanPyn))
sortedRadii = np.flip(np.argsort(radiiListMean))

# To present this, we should plot the anti-halos that are matched:

[alpha_shape_list_all,largeAntihalos_all,\
        snapsortList_all] = tools.loadOrRecompute(figuresFolder + \
            "skyplot_data_all.p",\
            getAntihaloSkyPlotData,snapNumList,samplesFolder=samplesFolder,\
            _recomputeData = recomputeData,recomputeData = recomputeData,\
            nToPlot=200,snapList = snapList,snapListRev = snapListRev,\
            antihaloCatalogueList = hrList,snapsortList = snapSortList,\
            ahProps = ahProps,rCentre = rSphere,massRange = [mMin,mMax],\
            rRange = [5,30])

ns = 1
snapList =  [pynbody.load(samplesFolder + "sample" + str(snapNum) + "/" + \
        snapname) for snapNum in snapNumList]
for snap in snapList:
    tools.remapBORGSimulation(snap,swapXZ=False,reverse=True)




ns = 0
snapToShow = pynbody.load(samplesFolder + "sample" + str(snapNumList[ns]) + \
    "/gadget_full_forward_512/snapshot_001")
tools.remapBORGSimulation(snapToShow,swapXZ=False,reverse=True)

rCut = 135
ha = ['right','left','left','left','left','center','right',\
        'right','right']
va = ['center','bottom','bottom','bottom','top',\
        'top','center','center','center']
annotationPos = [[-1.2,0.9],\
        [1.3,0.8],[1.8,0.5],[1.5,-1.2],[1.7,-0.7],[-1,0.2],[0.8,0.6],\
        [1.0,0.1],[-1.8,0.5]]
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
[combinedAbellN,combinedAbellPos,abell_nums] = \
    real_clusters.getCombinedAbellCatalogue()

abell_nums = [426,2147,1656,3627,3545,548,2197,2052,1367]
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




nVoidsToShow = len(seabornColormap)
catToUse = finalCatOpt
#selection = np.arange(nVoidsToShow)
#selection = sortedRadiiPyn[0:nVoidsToShow]
#selection = sortedRadiiPyn[0:nVoidsToShow]
selection = sortedRadiiOpt[:(nVoidsToShow)]
#selection = sortedRadii[np.where(finalCat[sortedRadii,1] == 32)[0]]
asListAll = []
colourListAll = []
laListAll = []
labelListAll = []

for ns in range(0,len(snapNumList)):
    asList = []
    colourList = []
    laList = []
    labelList = []
    for k in range(0,np.min([nVoidsToShow,len(selection)])):
        if catToUse[selection[k],ns] > 0:
            asList.append(alpha_shape_list_all[ns][1][\
                catToUse[selection[k],ns]-1])
            laList.append(\
                largeAntihalos_all[ns][catToUse[selection[k],ns]-1])
            colourList.append(seabornColormap[k])
            labelList.append(str(catToUse[selection[k],ns]))
    asListAll.append(asList)
    colourListAll.append(colourList)
    laListAll.append(laList)
    labelListAll.append(labelList)


for ns in range(0,len(snapNumList)):
    plot.plotLocalUniverseMollweide(rCut,snapList[ns],\
        alpha_shapes = asListAll[ns],\
        largeAntihalos = laListAll[ns],hr=hrList[ns],\
        coordAbell = coordCombinedAbellSphere,abellListLocation = clusterIndMain,\
        nameListLargeClusters = [name[0] for name in clusterNames],\
        ha = ha,va= va, annotationPos = annotationPos,\
        title = 'Local super-volume: large voids (antihalos) within $' + \
        str(rCut) + "\\mathrm{\\,Mpc}h^{-1}$",\
        vmin=1e-2,vmax=1e2,legLoc = 'lower left',bbox_to_anchor = (-0.1,-0.2),\
        snapsort = snapsortList_all[ns],antihaloCentres = None,\
        figOut = figuresFolder + "/supporting_plots/ah_match_sample_" + \
        str(ns) + ".png",\
        showFig=False,figsize = (scale*textwidth,scale*0.55*textwidth),\
        voidColour = colourListAll[ns],antiHaloLabel=labelListAll[ns],\
        bbox_inches = bound_box,galaxyAngles=equatorialRThetaPhi[:,1:],\
        galaxyDistances = equatorialRThetaPhi[:,0],showGalaxies=False,\
        voidAlpha = 0.6)

plt.show()

def getMeanCentresFromCombinedCatalogue(combinedCat,centresList):
    meanCentresArray = np.zeros((len(combinedCat),3))
    for nV in range(0,len(combinedCat)):
        centresArray = []
        for l in range(0,combinedCat.shape[1]):
            if finalCatOpt[nV][l] > -1:
                centresArray.append(centresList[l][combinedCat[nV,l] - 1])
        meanCentresArray[nV,:] = np.mean(centresArray,0)
    return meanCentresArray

meanCentresArray = getMeanCentresFromCombinedCatalogue(finalCatOpt,\
    centresListShort)

distanceArray = np.sqrt(np.sum(meanCentresArray**2,1))


# Simpler plot - just create a gif of one cluster:

snapSortListRev = [np.argsort(snap['iord']) for snap in snapListRev]

indList = []
#nV = 26
nV = 33605
for l in range(0,len(snapNumList)):
    if finalCatOpt[nV][l] > -1:
        indList.append(centralAntihalos[l][0][sortedList[l][finalCatOpt[nV][l]-1]])
    else:
        indList.append(-1)

centresArray = []
haveCentreCount = 0
for l in range(0,len(snapNumList)):
    if finalCatOpt[nV][l] > -1:
        centresArray.append(centresListShort[l][finalCatOpt[nV,l] - 1])

#meanCentre = np.flip(np.mean(centresArray,0)) + boxsize/2
meanCentre = np.mean(centresArray,0)
stdCentre = np.std(centresArray,0)
meanCentre
stdCentre

# plot in a box around this void.
Lbox = 100
axis = 2
Om0 = 0.3111
rhoBar = Om0*2.7754e11
nPix = 32
ns = 0
binEdgesX = np.linspace(meanCentre[0] - Lbox/2,meanCentre[0] + Lbox/2,nPix)
binEdgesY = np.linspace(meanCentre[1] - Lbox/2,meanCentre[1] + Lbox/2,nPix)

densitiesHR = [np.fromfile("new_chain/sample" + str(snap) + \
    "/gadget_full_forward_512/snapshot_001.a_den",\
    dtype=np.float32) for snap in snapNumList]
densities256 = [np.reshape(density,(256,256,256),order='C') \
    for density in densitiesHR]
densities256F = [np.reshape(density,(256,256,256),order='F') \
    for density in densitiesHR]

#filtCube = pynbody.filt.Cuboid(meanCentre[0] - Lbox/2,meanCentre[0] + Lbox/2,\
#    meanCentre[1] - Lbox/2,meanCentre[1] + Lbox/2,\
#    meanCentre[2] - Lbox/2,meanCentre[2] + Lbox/2)

#snapFilter = (snapList[ns]['pos'][:,0] > meanCentre[0] - Lbox/2) & \
#    (snapList[ns]['pos'][:,0] <= meanCentre[0] + Lbox/2) &\
#    (snapList[ns]['pos'][:,1] > meanCentre[1] - Lbox/2) & \
#    (snapList[ns]['pos'][:,1] <= meanCentre[1] + Lbox/2) &\
#    (snapList[ns]['pos'][:,2] > meanCentre[2] - Lbox/2) & \
#    (snapList[ns]['pos'][:,2] <= meanCentre[2] + Lbox/2)

#hist = np.histogram2d(snapList[ns]['pos'][snapFilter,0],\
#    snapList[ns]['pos'][snapFilter,1],\
#    bins = [binEdgesX,binEdgesY])

cuboidVol = Lbox**3/(nPix**2)
mUnit = snapList[0]['mass'][0]*1e10

delta = mUnit*hist[0]/(cuboidVol*rhoBar) - 1.0

# plot:
#plt.imshow(delta,norm=colors.LogNorm(vmin=1/70,vmax=70),cmap='PuOr_r')

Lbox = 100
zSlice = meanCentre[2]
N = 256
alphaVal = 0.2
vmin = 1/1000
vmax = 1000
thickness = Lbox
indLow = int((zSlice + boxsize/2)*N/boxsize) - int((thickness/2)*N/(boxsize))
indUpp = int((zSlice + boxsize/2)*N/boxsize) + int((thickness/2)*N/(boxsize))

indLowX = int((meanCentre[0] + boxsize/2)*N/boxsize) - int((thickness/2)*N/(boxsize))
indUppX = int((meanCentre[0] + boxsize/2)*N/boxsize) + int((thickness/2)*N/(boxsize))
indLowY = int((meanCentre[1] + boxsize/2)*N/boxsize) - int((thickness/2)*N/(boxsize))
indUppY = int((meanCentre[1] + boxsize/2)*N/boxsize) + int((thickness/2)*N/(boxsize))

sm = cm.ScalarMappable(colors.LogNorm(vmin=1/1000,vmax=1000),\
            cmap='PuOr_r')

phi = np.linspace(0,2*np.pi,1000)
Xcirc = np.cos(phi)
Ycirc = np.sin(phi)


for ns in range(0,len(snapNumList)):
    plt.clf()
    denToPlot = np.mean(densities256[ns][indLowX:indUppX,indLowY:indUppY,\
        indLow:indUpp],2)
    plt.imshow(denToPlot,norm=colors.LogNorm(vmin=vmin,vmax=vmax),\
            cmap='PuOr_r',extent=(meanCentre[0] - Lbox/2,\
            meanCentre[0] + Lbox/2,\
            meanCentre[1] - Lbox/2,meanCentre[1] + Lbox/2))
    # Alpha shape:
    if indList[ns] > -1:
        halo = hrList[ns][indList[ns]+1]
        positions = tools.remapAntiHaloCentre(\
            snapList[ns]['pos'][snapSortList[ns][halo['iord']],:],boxsize)
        #alphaVal = alphashape.optimizealpha(\
        #    np.array([positions[:,0],positions[:,1]]).T)
        alphaShapeVoid = alphashape.alphashape(np.array([positions[:,0],\
            positions[:,1]]).T,alphaVal)
        ax = plt.gca()
        ax.add_patch(PolygonPatch(alphaShapeVoid,fc=None,ec='b',alpha=0.5,\
            fill = False))
        sampleCentre = centresListShort[ns][finalCatOpt[nV,ns] - 1]
        plt.scatter(sampleCentre[0],sampleCentre[1],marker='x',color='b',\
            label='Sample Centre')
        plt.plot(sampleCentre[0] + radiiListOpt[nV,ns]*Xcirc,\
            sampleCentre[1] + radiiListOpt[nV,ns]*Ycirc,\
            linestyle='--',color='b',label='Effective radius\n$' + \
            ("%.2g" % radiiListOpt[nV,ns]) + "\\mathrm{Mpc}^{-1}$")
    plt.scatter(meanCentre[0],meanCentre[1],marker='x',color='r',\
        label='Mean Centre')
    plt.legend(frameon=False,loc="lower right")
    plt.xlabel('x ($\\mathrm{Mpc}h^{-1}$)')
    plt.ylabel('y ($\\mathrm{Mpc}h^{-1}$)')
    plt.xlim([meanCentre[0] - Lbox/2,meanCentre[0] + Lbox/2])
    plt.ylim([meanCentre[1] - Lbox/2,meanCentre[1] + Lbox/2])
    plt.title('Sample ' + str(snapNumList[ns]))
    cbar = plt.colorbar(sm, orientation="vertical")
    plt.savefig(figuresFolder + "frame_" + str(ns) + ".png")
    #plt.show()

import glob
from PIL import Image
frames = []
imgs = glob.glob(figuresFolder + "frame_*.png")
for i in imgs:
    new_frame = Image.open(i)
    frames.append(new_frame)

# Save into a GIF file that loops forever
frames[0].save(figuresFolder + 'voids.gif', format='GIF',
               append_images=frames[1:],
               save_all=True,
               duration=300, loop=0)


# Plots of the anti-halos:
hrcentres = [props[2] for props in ahProps]
hrcentresListShort = [np.array([hrcentres[l][\
        centralAntihalos[l][0][sortedList[l][k]],:] \
        for k in range(0,np.min([ahCounts[l],max_index]))]) \
        for l in range(0,len(snapNumList))]


centresArrayRev = []
haveCentreCount = 0
for l in range(0,len(snapNumList)):
    if finalCatOpt[nV][l] > -1:
        centresArrayRev.append(hrcentresListShort[l][finalCatOpt[nV,l] - 1])

#meanCentre = np.flip(np.mean(centresArray,0)) + boxsize/2
meanCentreRev = np.mean(centresArrayRev,0)
imProj = pynbody.plot.sph.image(snapListRev[0],\
            qty='rho',width=boxsize/2,log=True,\
            #units = "Msol h**2 Mpc**-3",\
            units = "Msol h Mpc**-2",\
            resolution=256,cmap='PuOr_r',show_cbar=False,av_z=False,\
            noplot = True,ret_im = True)

showParticleDensityShare = False
showMass = False
showIndex = False
showFrac = True
vmin = 1/70
vmax = 70
Lbox = 20
showOtherHalos = True
mLower = 1e14
mUpper = 1e16
imList = []
for ns in range(0,len(snapNumList)):
    subsnap = snapListRev[ns]
    with pynbody.transformation.translate(subsnap,-meanCentreRev):
        im = pynbody.plot.sph.image(subsnap,\
            qty='rho',width=Lbox,log=True,\
            #units = "Msol h**2 Mpc**-3",\
            units = "Msol h Mpc**-2",\
            resolution=128,cmap='PuOr_r',show_cbar=False,av_z=False,\
            noplot = True,ret_im = True)
    imList.append(im)

# Get list of particles that are involved in any snapshot:
partLists = []
for ns in range(0,len(snapNumList)):
    if indList[ns] > -1:
        partLists.append(hrList[ns][indList[ns]+1]['iord'])

allPartList = partLists[0]
for k in range(1,len(partLists)):
    allPartList = np.union1d(partLists[k],allPartList)

partFracsList = np.zeros(allPartList.shape)
for k in range(0,len(partLists)):
    partFracsList += np.isin(allPartList,partLists[k])

partFracsList /= len(snapNumList)



for ns in range(0,len(snapNumList)):
    plt.clf()
    #denToPlot = np.mean(densities256[ns][indLowX:indUppX,indLowY:indUppY,\
    #    indLow:indUpp],2)
    #plt.imshow(denToPlot,norm=colors.LogNorm(vmin=vmin,vmax=vmax),\
    #        cmap='PuOr_r',extent=(meanCentre[0] - Lbox/2,\
    #        meanCentre[0] + Lbox/2,\
    #        meanCentre[1] - Lbox/2,meanCentre[1] + Lbox/2))
    if showParticleDensityShare:
        voidParticleHist = np.histogram2d(\
            snapListRev[ns]['pos'][snapSortListRev[ns][allPartList],0],\
            snapListRev[ns]['pos'][snapSortListRev[ns][allPartList],1],\
            bins = [np.linspace(meanCentreRev[0] - Lbox/2,\
                meanCentreRev[0] + Lbox/2,21),\
                np.linspace(meanCentreRev[1] - Lbox/2,\
                meanCentreRev[1] + Lbox/2,21)],weights=partFracsList)
        vminFrac = 1e-4
        vmaxFrac = np.max(voidParticleHist[0]/len(allPartList))
        plt.imshow(voidParticleHist[0]/len(allPartList),cmap='Blues',\
            extent=(meanCentreRev[0] - Lbox/2,\
            meanCentreRev[0] + Lbox/2,\
            meanCentreRev[1] - Lbox/2,meanCentreRev[1] + Lbox/2),\
            norm=colors.LogNorm(vmin=vminFrac,vmax=vmaxFrac))
        sm2 = cm.ScalarMappable(colors.LogNorm(vmin=vminFrac,vmax=vmaxFrac),\
            cmap='Blues')
        cbar = plt.colorbar(sm2, orientation="vertical",\
            label='Particle fraction')
    else:
        plt.imshow(np.flipud(imList[ns])/np.mean(imProj),\
            norm=colors.LogNorm(vmin=vmin,vmax=vmax),\
            cmap='PuOr_r',extent=(meanCentreRev[0] - Lbox/2,\
            meanCentreRev[0] + Lbox/2,\
            meanCentreRev[1] - Lbox/2,meanCentreRev[1] + Lbox/2))
        cbar = plt.colorbar(sm, orientation="vertical",\
            label='$\\rho/\\bar{\\rho}$ (Projected)')
    # Alpha shape:
    if indList[ns] > -1:
        halo = hrList[ns][indList[ns]+1]
        sampleCentre = hrcentresListShort[ns][finalCatOpt[nV,ns] - 1]
        plt.scatter(sampleCentre[0],\
            sampleCentre[1],marker='x',color='b',\
            label='Sample Centre\n$' + \
            plot.scientificNotation(massListOpt[nV,ns]) + "M_{\\odot}h^{-1}$")
        Rvir = halo.properties['Rvir']/1000
        plt.plot(sampleCentre[0] + Rvir*Xcirc,\
            sampleCentre[1] + Rvir*Ycirc,\
            linestyle='--',color='b',label='Virial radius\n$' + \
            ("%.2g" % Rvir) + \
            "\\mathrm{Mpc}^{-1}$")
    #plt.scatter(snapListRev[ns]['pos'][snapSortListRev[ns][allPartList],0],\
    #    snapListRev[ns]['pos'][snapSortListRev[ns][allPartList],1],\
    #    label='Antihalo Particles',marker='.',color=seabornColormap[0])
    if showOtherHalos:
        # Find other halos in this range:
        inRangePos = \
            (hrcentresListShort[ns][:,0] >= meanCentreRev[0] - Lbox/2) & \
            (hrcentresListShort[ns][:,0] < meanCentreRev[0] + Lbox/2) & \
            (hrcentresListShort[ns][:,1] >= meanCentreRev[1] - Lbox/2) & \
            (hrcentresListShort[ns][:,1] < meanCentreRev[1] + Lbox/2) & \
            (hrcentresListShort[ns][:,2] >= meanCentreRev[2] - Lbox/2) & \
            (hrcentresListShort[ns][:,2] < meanCentreRev[2] + Lbox/2)
        inRangeMass = (massListShort[ns] >= mLower) & \
            (massListShort[ns] < mUpper)
        ahIndices = np.array(centralAntihalos[ns][0])[sortedList[ns]]
        antihaloParts = np.unique(np.hstack([np.intersect1d(allPartList,\
            hrList[ns][otherAH+1]['iord']) \
            for otherAH in ahIndices[inRangePos]]))
        antihaloPartFrac = np.sum(partFracsList[\
            np.isin(allPartList,antihaloParts)])/np.sum(partFracsList)
        antihalosToShow = inRangeMass & inRangePos# & \
        #    (ahIndices != finalCatOpt[nV,ns] - 1)
        allMasses = massListShort[ns][antihalosToShow]
        allPositions = hrcentresListShort[ns][antihalosToShow,:]
        allIndices = ahIndices[antihalosToShow]
        #allShared = np.array([len(np.intersect1d(allPartList,\
        #    hrList[ns][otherAH+1]['iord']))/len(allPartList) \
        #    for otherAH in allIndices])
        allShared = np.array([np.sum(partFracsList[np.isin(allPartList,\
            hrList[ns][otherAH+1]['iord'])])/np.sum(partFracsList) \
            for otherAH in allIndices])
        plt.scatter(hrcentresListShort[ns][antihalosToShow,0],\
            hrcentresListShort[ns][antihalosToShow,1],marker='x',color='k')
        for k in range(0,np.sum(antihalosToShow)):
            textToShow = ""
            if showIndex:
                textToShow += str(allIndices[k]) + ", "
            if showFrac:
                textToShow += ("%.3g" % (allShared[k]*100))+ "%\n"
            if showMass:
                textToShow += "$" + plot.scientificNotation(allMasses[k]) + \
                    "M_{\\odot}h^{-1}$"
            plt.text(allPositions[k,0],allPositions[k,1],textToShow)
    plt.scatter(meanCentreRev[0],meanCentreRev[1],marker='x',color='r',\
        label='Mean Centre')
    plt.legend(frameon=False,loc="lower right")
    plt.xlabel('x ($\\mathrm{Mpc}h^{-1}$)')
    plt.ylabel('y ($\\mathrm{Mpc}h^{-1}$)')
    plt.xlim([meanCentreRev[0]- Lbox/2,meanCentreRev[0] + Lbox/2])
    plt.ylim([meanCentreRev[1] - Lbox/2,meanCentreRev[1] + Lbox/2])
    titleText = 'Sample ' + str(snapNumList[ns])
    if showOtherHalos:
        titleText += " $(" +  ("%.3g" % (antihaloPartFrac * 100)) + \
            "\%$ in anti-halos)"
    plt.title(titleText)
    plt.savefig(figuresFolder + "reverse_frame_" + str(ns) + ".png")
    #plt.show()

frames = []
imgs = glob.glob(figuresFolder + "reverse_frame_*.png")
for i in imgs:
    new_frame = Image.open(i)
    frames.append(new_frame)

# Save into a GIF file that loops forever
frames[0].save(figuresFolder + 'antihalos.gif', format='GIF',
               append_images=frames[1:],
               save_all=True,
               duration=300, loop=0)


rBins = np.linspace(0,300,21)
rBinCentres = plot_utilities.binCentres(rBins)
[inBins,noInBins] = plot_utilities.binValues(distanceArray,rBins)

meanCatFrac = np.array([np.mean(catFracOpt[ind]) for ind in inBins])
stdCatFrac = np.array([np.std(catFracOpt[ind])/np.sqrt(len(ind)) \
    for ind in inBins])

plt.errorbar(rBinCentres,meanCatFrac,yerr=stdCatFrac)
plt.xlabel('Distance from centres ($\\mathrm{Mpc}h^{-1}$)')
plt.ylabel('Mean Catalogue Fraction')
plt.show()


# Cat fraction by mass bin:
mBins = [1e13,1e14,5e14,1e16]
[inMassBins,noInMassBins] = plot_utilities.binValues(massMeanOpt,mBins)

meanCatFracBins = [np.array(\
    [np.mean(catFracOpt[np.intersect1d(ind,inMassBins[k])]) \
    for ind in inBins]) \
    for k in range(0,len(mBins) - 1)]
stdCatFracBins = [np.array(\
    [np.std(catFracOpt[np.intersect1d(ind,inMassBins[k])])/\
    np.sqrt(len(np.intersect1d(ind,inMassBins[k]))) \
    for ind in inBins]) \
    for k in range(0,len(mBins) - 1)]
#meanCatFracBins = np.array([np.mean(catFracOpt[ind]) for ind in inMassBins])

for k in range(0,len(mBins)-1):
    plt.errorbar(rBinCentres,meanCatFracBins[k],yerr=stdCatFracBins[k],\
        label = "$" + plot.scientificNotation(mBins[k]) + \
        " < M/M_{\\odot} \leq " + \
        plot.scientificNotation(mBins[k+1]) + "$")

plt.xlabel('Distance from centres ($\\mathrm{Mpc}h^{-1}$)')
plt.ylabel('Mean Catalogue Fraction')
plt.legend()
plt.show()




fig, ax = plt.subplots()
plot.plotDensitySlice(ax,densities256[0],meanCentre,50,boxsize,256,Lbox,\
        meanCentre[2],1/1000,1000,\
        'PuOr_r',markCentre=True,axesToShow = [0,1],losAxis=2,flip=False,\
        flipud=False,fliplr=False,swapXZ = False,centreMarkerColour='r')



#-------------------------------------------------------------------------------
# N BODY HEURISTICS

def fNOptimal(N,delta_c):
    return (N/np.log(N)) - 0.6*(8/np.sqrt(200))*(delta_c**(0.5))

def optimalN(delta_c,Nmax=1e5,Nmin=2):
    f1 = fNOptimal(Nmin,delta_c)
    f2 = fNOptimal(Nmax,delta_c)
    if f1*f2 > 0:
        raise Exception("Not bounded")
    sol = scipy.optimize.brentq(lambda N: fNOptimal(N,delta_c),Nmin,Nmax)
    return sol


# Get a power spectrum from CAMB:
[khFull,PkFull] = cosmology.powerSpectrum(h=0.6767,Om0=0.3111,Ob0 = 0.04896,\
    sigma8=0.8102,z=0,kmax=20,kmin=1e-4,ns=0.9665)

# sigma^2 as a function of M:
def sigma2(M,sigma8=0.8102,Om0=0.3111,Ob0=0.04896,h=0.6767,ns=0.9665,\
        kmin=1e-4,kmax=5,ki=None,Pki=None,z=0):
    if ki is None or Pki is None:
        [ki,Pki] = cosmology.powerSpectrum(h=h,Om0=Om0,Ob0 = Ob0,\
            sigma8=sigma8,z=z,kmax=kmax,kmin=kmin,ns=ns)
    rhoB = 2.7754e11*Om0
    R = np.cbrt(3.0*M/(4.0*np.pi*rhoB))
    logPkinterp = scipy.interpolate.interp1d(np.log(ki),np.log(Pki),\
        kind='cubic')
    integral = scipy.integrate.quad(lambda k: np.exp(logPkinterp(np.log(k)))*\
        cosmology.What(k,R)**2*k**2/(2*np.pi**2),kmin,kmax)
    return integral[0]

# Log derivative of sigma^2
def dlogSigmadLogM(M,sigma8=0.8102,Om0=0.3111,Ob0=0.04896,h=0.6767,ns=0.9665,\
        kmin=1e-4,kmax=5,ki=None,Pki=None,z=0,sigmaM2=None):
    if ki is None or Pki is None:
        [ki,Pki] = cosmology.powerSpectrum(h=h,Om0=Om0,Ob0 = Ob0,\
            sigma8=sigma8,z=z,kmax=kmax,kmin=kmin,ns=ns)
    rhoB = 2.7754e11*Om0
    R = np.cbrt(3.0*M/(4.0*np.pi*rhoB))
    logPkinterp = scipy.interpolate.interp1d(np.log(ki),np.log(Pki),\
        kind='cubic')
    integral1 = scipy.integrate.quad(lambda k: np.exp(logPkinterp(np.log(k)))*\
        cosmology.What(k,R)*cosmology.Whatp(k,R)*k**3/(4*np.pi**3*R**2*rhoB),\
        kmin,kmax)
    if sigmaM2 is None:
        sigmaM2 = sigma(M,sigma8=sigma8,Om0=Om0,Ob0=Ob0,h=h,ns=ns,\
            kmin=kmin,kmax=kmin,ki=ki,Pki=Pki,z=z)
    return M*integral1[0]/(2*sigmaM2)

# press schechter function:
def dndmPS(M,sigma8=0.8102,Om0=0.3111,Ob0=0.04896,h=0.6767,ns=0.9665,\
        kmin=1e-4,kmax=5,ki=None,Pki=None,z=0,deltac=1.686):
    rhoB = 2.7754e11*Om0
    if ki is None or Pki is None:
        [ki,Pki] = cosmology.powerSpectrum(h=h,Om0=Om0,Ob0 = Ob0,\
            sigma8=sigma8,z=z,kmax=kmax,kmin=kmin,ns=ns)
    M8 = 4*np.pi*rhoB*8**3/3
    sigma80 = sigma2(M8,sigma8=sigma8,Om0=Om0,Ob0=Ob0,h=h,ns=ns,\
                kmin=kmin,kmax=kmax,ki=ki,Pki=Pki,z=z)
    if np.isscalar(M):
        sigmaM2 = sigma8*sigma2(M,sigma8=sigma8,Om0=Om0,Ob0=Ob0,h=h,ns=ns,\
                kmin=kmin,kmax=kmax,ki=ki,Pki=Pki,z=z)/sigma80
        logDeriv = np.abs(dlogSigmadLogM(M,sigma8=sigma8,Om0=Om0,Ob0=Ob0,h=h,ns=ns,\
                kmin=kmin,kmax=kmax,ki=ki,Pki=Pki,z=z,sigmaM2 = sigmaM2))
        return np.sqrt(2.0/np.pi)*(rhoB/M**2)*(deltac/np.sqrt(sigmaM2))*logDeriv*\
            np.exp(-deltac**2/(2*sigmaM2))
    else:
        dndm = np.zeros(len(M))
        for k in range(0,len(M)):
            sigmaM2 = sigma8*sigma2(M[k],sigma8=sigma8,Om0=Om0,Ob0=Ob0,h=h,\
                ns=ns,kmin=kmin,kmax=kmax,ki=ki,Pki=Pki,z=z)/sigma80
            logDeriv = np.abs(dlogSigmadLogM(M[k],sigma8=sigma8,Om0=Om0,Ob0=Ob0,\
                h=h,ns=ns,kmin=kmin,kmax=kmax,ki=ki,Pki=Pki,z=z,\
                sigmaM2 = sigmaM2))
            dndm[k] = np.sqrt(2.0/np.pi)*(rhoB/M[k]**2)*\
                (deltac/np.sqrt(sigmaM2))*logDeriv*\
                np.exp(-deltac**2/(2*sigmaM2))
        return dndm

# Compute restricted power spectrum mass function:
Mrange = 10**(np.linspace(13,16,31))

Rmin = 677.7/256

pkFull = dndmPS(Mrange,ki=khFull,Pki=PkFull)
pkPartial2 = dndmPS(Mrange,kmax = 2*np.pi/Rmin)
pkPartial1 = dndmPS(Mrange,kmax = np.pi/Rmin)

plt.semilogx(Mrange,pkPartial1/pkFull,\
    label='$k < \pi/a$ cut')
plt.semilogx(Mrange,pkPartial2/pkFull,\
    label='$k < 2\pi/a$ cut')
plt.xlabel('Mass ($M_{\\odot}h^{-1}$)')
plt.ylabel('Mass function ratio')
plt.axhline(1.0,linestyle='--',color='k',label='')
plt.axhspan(0.99,1.01,color='grey',alpha=0.5,label='$1\%$ Error')
plt.legend()
plt.show()






[constrainedHaloMasses512New,constrainedAntihaloMasses512New,\
        deltaListMeanNew,deltaListErrorNew,\
        constrainedHaloMasses512Old,constrainedAntihaloMasses512Old,\
        deltaListMeanOld,deltaListErrorOld,\
        comparableHalosNew,comparableHaloMassesNew,\
        comparableAntihalosNew,comparableAntihaloMassesNew,\
        centralHalosNew,centralAntihalosNew,\
        centralHaloMassesNew,centralAntihaloMassesNew,\
        comparableHalosOld,comparableHaloMassesOld,\
        comparableAntihalosOld,comparableAntihaloMassesOld,\
        centralHalosOld,centralAntihalosOld,\
        centralHaloMassesOld,centralAntihaloMassesOld] = \
            tools.loadOrRecompute(figuresFolder + "amf_hmf_data_5511.p",\
                getHMFAMFData,snapNumList,snapNumListOld,snapNumListUncon,\
                snapNumListUnconOld,_recomputeData = recomputeData,\
                recomputeData = [True,False,False,False],\
                unconstrainedFolderNew = unconstrainedFolderNew,\
                unconstrainedFolderOld = unconstrainedFolderOld,\
                snapnameNew = snapnameNew,snapnameNewRev=snapnameNewRev,\
                snapnameOld = snapnameOld,snapnameOldRev = snapnameOldRev,\
                samplesFolder = samplesFolder,samplesFolderOld=samplesFolderOld)



plot.plotHMFAMFComparison(\
        constrainedHaloMasses512Old,deltaListMeanOld,deltaListErrorOld,\
        comparableHaloMassesOld,constrainedAntihaloMasses512Old,\
        comparableAntihaloMassesOld,\
        constrainedHaloMasses512New,deltaListMeanNew,deltaListErrorNew,\
        comparableHaloMassesNew,constrainedAntihaloMasses512New,\
        comparableAntihaloMassesNew,\
        referenceSnap,referenceSnapOld,\
        savename = figuresFolder + "hmf_amf_old_vs_new_forward_model_5511.pdf",\
        ylabelStartOld = 'Old reconstruction',\
        ylabelStartNew = 'New reconstruction',\
        fontsize=fontsize,legendFontsize=legendFontsize)

