#-------------------------------------------------------------------------------
# CONFIGURATION
from void_analysis import plot, tools, snapedit
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
seabornColormap = sns.color_palette("colorblind",as_cmap=True)
import pynbody
import astropy.units as u
from astropy.coordinates import SkyCoord
import scipy
import os
import sys

figuresFolder = "borg-antihalos_paper_figures/all_samples/"
#figuresFolder = "borg-antihalos_paper_figures/batch5-2/"

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
# DATA FOR PLOTS

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


#snapNumListUncon = [1,2,3,4,5]
snapNumListUncon = [1,2,3,4,5,6,7,8,9,10]
#snapNumListUncon = [2,4,5,6,7,8,9,10]
snapNumListUnconOld = [1,2,3,5,6]
boxsize = 677.7


snapNameList = ["new_chain/sample" + str(k) + \
    "/gadget_full_forward_512/snapshot_001" \
    for k in snapNumList]

# PPT plots data:
if runTests:
    testComputation(testDataFolder + "ppt_pipeline_test.p",getPPTPlotData,\
        snapNumList = [7000, 7200, 7400],samplesFolder = 'new_chain/',\
        recomputeData = True)

nBinsPPT = 6
rebin = False
print("Doing ppts")
doPPTs = True
if doPPTs:
    [galaxyNumberCountExp,galaxyNumberCountExpShells,\
        interval2MPPBootstrap,interval2MPPBootstrapShells,\
        galaxyNumberCountsRobust,\
        galaxyNumberCountsRobustAll,posteriorMassAll,\
        Aalpha,varianceAL,varianceALShell,\
        galaxyNumberCountsRobustAllShells,\
        galaxyNumberCountsRobustShells] = \
        tools.loadOrRecompute(\
            figuresFolder + "ppt_plots_data.p",getPPTPlotData,\
            _recomputeData = recomputeData or rebin,\
            nBins = nBinsPPT,nClust=9,\
            nMagBins = 16,N=256,\
            restartFile = 'new_chain_restart/merged_restart.h5',\
            snapNumList = snapNumList,samplesFolder = 'new_chain/',\
            surveyMaskPath = "./2mpp_data/",\
            Om0 = 0.3111,Ode0 = 0.6889,boxsize = 677.7,h=0.6766,\
            Mstarh = -23.28,mmin = 0.0,mmax = 12.5,\
            recomputeData = recomputeData,rBinMin = 0.1,rBinMax = 20,\
            abell_nums = [426,2147,1656,3627,3571,548,2197,2063,1367],\
            nside = 4,nRadialSlices=10,rmax=600,\
            tmppFile = "2mpp_data/2MPP.txt",\
            reductions = 4,iterations = 20,verbose=True,hpIndices=None,\
            centreMethod="density",data_folder = data_folder,\
            bootstrapInterval = [16,84,2.5,97.5,0.5,99.5,0.05,99.95])


# Load or recompute the HMF/AMF data:
if runTests:
    testComputation(testDataFolder + "hmf_pipeline_test.p",getHMFAMFData,\
        [7000, 7200, 7400],[7422,7500,8000],[1,3,5],\
        [1,2,3],recomputeData = True,\
        unconstrainedFolderNew = unconstrainedFolderNew,\
        unconstrainedFolderOld = unconstrainedFolderOld,\
        snapnameNew = snapnameNew,snapnameNewRev=snapnameNewRev,\
        snapnameOld = snapnameOld,snapnameOldRev = snapnameOldRev,\
        samplesFolder = samplesFolder,samplesFolderOld=samplesFolderOld)


doHMFs = True
if doHMFs:
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
                tools.loadOrRecompute(figuresFolder + "amf_hmf_data.p",\
                    getHMFAMFData,snapNumList,snapNumListOld,snapNumListUncon,\
                    snapNumListUnconOld,_recomputeData = recomputeData,\
                    recomputeData = [recomputeData,recomputeData,\
                        recomputeData,recomputeData],\
                    unconstrainedFolderNew = unconstrainedFolderNew,\
                    unconstrainedFolderOld = unconstrainedFolderOld,\
                    snapnameNew = snapnameNew,snapnameNewRev=snapnameNewRev,\
                    snapnameOld = snapnameOld,snapnameOldRev = snapnameOldRev,\
                    samplesFolder = samplesFolder,\
                    samplesFolderOld=samplesFolderOld,\
                    data_folder=data_folder)

loadOld = False
if loadOld:
    [constrainedHaloMasses512New2,constrainedAntihaloMasses512New2,\
        deltaListMeanNew2,deltaListErrorNew2,\
        constrainedHaloMasses512Old2,constrainedAntihaloMasses512Old2,\
        deltaListMeanOld2,deltaListErrorOld2,\
        comparableHalosNew2,comparableHaloMassesNew2,\
        comparableAntihalosNew2,comparableAntihaloMassesNew2,\
        centralHalosNew2,centralAntihalosNew2,\
        centralHaloMassesNew2,centralAntihaloMassesNew2,\
        comparableHalosOld2,comparableHaloMassesOld2,\
        comparableAntihalosOld2,comparableAntihaloMassesOld2,\
        centralHalosOld2,centralAntihalosOld2,\
        centralHaloMassesOld2,centralAntihaloMassesOld2] = tools.loadPickle(\
            figuresFolder + "amf_hmf_data_old.p")




# Void profile data:
if runTests:
    testComputation(testDataFolder + "void_profiles_pipeline_test.p",\
        getVoidProfilesData,\
        [7000, 7200, 7400],[1,3,5],\
        unconstrainedFolder = unconstrainedFolderNew,\
        samplesFolder = samplesFolder,\
        snapname=snapnameNew,snapnameRev = snapnameNewRev)


independentCentres = np.array([[0,0,0],[-boxsize/2,0,0],[0,-boxsize/2,0],\
    [0,0,-boxsize/2],[boxsize/4,boxsize/4,boxsize/4],\
    [-boxsize/4,boxsize/4,boxsize/4],[boxsize/4,-boxsize/4,boxsize/4],\
    [-boxsize/4,-boxsize/4,boxsize/4],[boxsize/4,boxsize/4,-boxsize/4],\
    [-boxsize/4,boxsize/4,-boxsize/4],[boxsize/4,-boxsize/4,-boxsize/4],\
    [-boxsize/4,-boxsize/4,-boxsize/4]])

#[rBinStackCentres,nbarjSepStack,\
#        sigmaSepStack,nbarjSepStackUn,sigmaSepStackUn,\
#        nbarjAllStacked,sigmaAllStacked,nbarjAllStackedUn,sigmaAllStackedUn,\
#        nbar,rMin,mMin,mMax] = \
#            tools.loadOrRecompute(figuresFolder + "void_profile_data.p",\
#                getVoidProfilesData,snapNumList,snapNumListUncon,\
#                unconstrainedFolder = unconstrainedFolderNew,\
#                samplesFolder = samplesFolder,\
#                snapname=snapnameNew,snapnameRev = snapnameNewRev,\
#                _recomputeData = recomputeData,\
#                unconstrainedCentreList = independentCentres)
# Reference snapshot:
samplesFolder = "new_chain/"
referenceSnapOld = pynbody.load("sample7500/forward_output/snapshot_006")
referenceSnap = pynbody.load(samplesFolder + \
        "/sample2791/gadget_full_forward_512/snapshot_001")
# Sorted reference snap indices:
snapsort = np.argsort(referenceSnap['iord'])
# Resolution limit (256^3):
mLimLower = referenceSnap['mass'][0]*1e10*100*8

mMin = 1e11
mMax = 1e16
rMin = 5
rMax = 30
muOpt = 0.9
rSearchOpt = 1
rSphere = 300
rSphereInner = 135


doCat = True
if doCat:
    print("Doing catalogue construction...")
    sys.stdout.flush()
    [massListMean,combinedFilter135,combinedFilter,rBinStackCentresCombined,\
        nbarjSepStackCombined,sigmaSepStackCombined,\
        nbarjAllStackedUnCombined,sigmaAllStackedUnCombined,nbar,rMin2,\
        mMin2,mMax2,nbarjSepStackUn,sigmaSepStackUn,\
        rBinStackCentres,nbarjSepStack,\
        sigmaSepStack,nbarjAllStackedUn,sigmaAllStackedUn,\
        nbarjSepStackUn,sigmaSepStackUn] = tools.loadOrRecompute(\
            data_folder + "finalCatData.p",getFinalCatalogue,snapNumList,\
            snapNumListUncon,\
            snrThresh = 10,snapname = "gadget_full_forward_512/snapshot_001",\
            snapnameRev = "gadget_full_reverse_512/snapshot_001",\
            samplesFolder="new_chain/",snapList = None,snapListRev = None,\
            snapListUnconstrained = None,snapListUnconstrainedRev=None,\
            mLower = "auto",mUpper = 2e15,nBins = 8,muOpt = muOpt,\
            rSearchOpt = rSearchOpt,rSphere = rSphere,\
            rSphereInner = rSphereInner,NWayMatch = True,rMin=rMin,rMax=rMax,\
            mMin=mMin,mMax = mMax,percThresh=99,chainFile="chain_properties.p",\
            Nden=256,recomputeUnconstrained = True,data_folder=data_folder,\
            _recomputeData = recomputeData,recomputeData=recomputeData)
    print("Finished catalogue construction...")
    sys.stdout.flush()
    # Centres with similar under-density:
    [centreListUn,densitiesInCentres,denListUn] = tools.loadPickle(\
        data_folder + "centre_list_unconstrained_data.p")
    # Catalogue data:
    [finalCatOpt,shortHaloListOpt,twoWayMatchListOpt,finalCandidatesOpt,\
        finalRatiosOpt,finalDistancesOpt,allCandidatesOpt,candidateCountsOpt,\
        allRatiosOpt,finalCombinatoricFracOpt,finalCatFracOpt,\
        alreadyMatched] = pickle.load(\
            open(data_folder + "catalogue_all_data.p","rb"))
    catData = np.load(data_folder + "catalogue_data.npz")
    # New method for void profiles:
    #[rBinStackCentres,nbarMean,sigmaMean,nbarVar,sigmaVar,nbar,\
    #    nbarjUnSameRadii,sigmaUnSameRadii] = getVoidProfilesForPaper(\
    #        finalCatOpt,combinedFilter135,snapNameList,\
    #        snapNumListUncon,centreListUn,data_folder = data_folder,\
    #        recomputeData=recomputeData)


# Timesteps data. Rerunning this from here not yet implemented 
#(see timesteps.py):
filename = "profiles_plot_data.p"
[clusterProfiles,clusterProfilesFull,densityListBORGPM,\
        densityListPM,densityListCOLA,densityList1024,densityListEPS_0p662,\
        densityListLogCOLA, densityListLogPM,\
        massList100m,massListFull100m,\
        massList200c,massListFull200c,\
        fluxList5,fluxList10,\
        fluxListFull5,fluxListFull10] = pickle.load(open(filename,"rb"))

filename2 = "profiles_plot_data_random.p"
[clusterProfiles2,clusterProfilesFull2,densityListBORGPM2,\
        densityListPM2,densityListCOLA2,densityList10242,densityListEPS_0p6622,\
        densityListLogCOLA2, densityListLogPM2,\
        massList100m2,massListFull100m2,\
        massList200c2,massListFull200c2,\
        fluxList52,fluxList102,\
        fluxListFull52,fluxListFull102] = pickle.load(open(filename2,"rb"))

# Antihalo sky plot data:
if runTests:
    testComputation(testDataFolder + "skyplot_pipeline_test.p",\
        getAntihaloSkyPlotData,\
        [7000, 7200, 7400],samplesFolder=samplesFolder,recomputeData = True)


# Centres about which to compute SNR:
Nden = 256
snrThresh=10
chainFile="chain_properties.p"

snrFilter = getSNRFilterFromChainFile(chainFile,snrThresh,snapNameList,boxsize)


[mcmcArray,num,N,NCAT,no_bias_params,bias_matrix,mean_field,\
        std_field,hmc_Elh,hmc_Eprior,hades_accept_count,\
        hades_attempt_count] = pickle.load(open(chainFile,"rb"))
Nden=256
snrField = mean_field**2/std_field**2
snrFieldLin = np.reshape(snrField,Nden**3)
grid = snapedit.gridListPermutation(Nden,perm=(2,1,0))
centroids = grid*boxsize/Nden + boxsize/(2*Nden)
positions = snapedit.unwrap(centroids - np.array([boxsize/2]*3),boxsize)
tree = scipy.spatial.cKDTree(snapedit.wrap(positions + boxsize/2,boxsize),\
    boxsize=boxsize)


allProps = [tools.loadPickle(snapname + ".AHproperties.p") \
    for snapname in snapNameList]
antihaloCentres = [tools.remapAntiHaloCentre(props[5],boxsize) \
        for props in allProps]
antihaloCentresUnmapped = [props[5] for props in allProps]
antihaloMasses = [props[3] for props in allProps]
antihaloRadii = [props[7] for props in allProps]



nearestPointsList = [tree.query_ball_point(\
        snapedit.wrap(antihaloCentres[k] + boxsize/2,boxsize),\
        antihaloRadii[k],workers=-1) \
        for k in range(0,len(antihaloCentres))]
snrAllCatsList = [np.array([np.mean(snrFieldLin[points]) \
        for points in nearestPointsList[k]]) for k in range(0,len(snapNumList))]
snrFilter2 = [snr > snrThresh for snr in snrAllCatsList]


centralAntihalos = [tools.getAntiHalosInSphere(antihaloCentres[k],rSphere,\
            filterCondition = (antihaloRadii[k] > rMin) & \
            (antihaloRadii[k] <= rMax) & (antihaloMasses[k] > mMin) & \
            (antihaloMasses[k] <= mMax) & snrFilter[k]) \
            for k in range(0,len(snapNumList))]
centralAntihaloMasses = [\
            antihaloMasses[k][centralAntihalos[k][0]] \
            for k in range(0,len(centralAntihalos))]
sortedList = [np.flip(np.argsort(centralAntihaloMasses[k])) \
        for k in range(0,len(snapNumList))]


ahCounts = np.array([len(cahs[0]) for cahs in centralAntihalos])
max_index = np.max(ahCounts)
centresListShort = [np.array([antihaloCentres[l][\
    centralAntihalos[l][0][sortedList[l][k]],:] \
    for k in range(0,np.min([ahCounts[l],max_index]))]) \
    for l in range(0,len(snapNumList))]

centresListShortUnmapped = [np.array([antihaloCentresUnmapped[l][\
    centralAntihalos[l][0][sortedList[l][k]],:] \
    for k in range(0,np.min([ahCounts[l],max_index]))]) \
    for l in range(0,len(snapNumList))]

radiiListShort = [np.array([antihaloRadii[l][\
        centralAntihalos[l][0][sortedList[l][k]]] \
        for k in range(0,np.min([ahCounts[l],max_index]))]) \
        for l in range(0,len(snapNumList))]




additionalConditions = [np.isin(np.arange(0,len(antihaloMasses[k])),\
    np.array(centralAntihalos[k][0])[\
    sortedList[k][finalCatOpt[(finalCatOpt[:,k] >= 0) & \
    combinedFilter135,k] - 1]]) for k in range(0,len(snapNameList))]

centralAntihalosAdditional = [tools.getAntiHalosInSphere(antihaloCentres[k],\
            rSphere,filterCondition = (antihaloRadii[k] > rMin) & \
            (antihaloRadii[k] <= rMax) & (antihaloMasses[k] > mMin) & \
            (antihaloMasses[k] <= mMax) & additionalConditions[k]) \
            for k in range(0,len(snapNumList))]

# Check to make sure skyplot centralAntihalos list is consistent:

rRange = [5,30]
massRange = [mMin,mMax]

rRangeCond = [(antihaloRadii[k] > rRange[0]) & \
    (antihaloRadii[k] <= rRange[1]) \
    for k in range(0,len(snapNumList))]
mRangeCond = [(antihaloMasses[k] > massRange[0]) & \
    (antihaloMasses[k] <= massRange[1]) \
    for k in range(0,len(snapNumList))]
filterCond = [rRangeCond[k] & mRangeCond[k] \
    for k in range(0,len(snapNumList))]
filterCond = [filterCond[k] & snrFilter[k] \
        for k in range(0,len(snapNumList))]
centralAntihalosTest = [tools.getAntiHalosInSphere(antihaloCentres[k],\
        rSphere,filterCondition = filterCond[k]) \
        for k in range(0,len(snapNumList))]

centralAntihalosRef = [tools.getAntiHalosInSphere(antihaloCentres[k],rSphere,\
            filterCondition = (antihaloRadii[k] > rMin) & \
            (antihaloRadii[k] <= rMax) & (antihaloMasses[k] > mMin) & \
            (antihaloMasses[k] <= mMax) & snrFilter[k]) \
            for k in range(0,len(snapNumList))]

centralAntihalosTest2 = [tools.getAntiHalosInSphere(antihaloCentres[k],rSphere,\
            filterCondition = (antihaloRadii[k] > rMin) & \
            (antihaloRadii[k] <= rMax) & (antihaloMasses[k] > mMin) & \
            (antihaloMasses[k] <= mMax) & snrFilter[k]) \
            for k in range(0,len(snapNumList))]


doSky = True
if doSky:
    #[alpha_shape_list,largeAntihalos,\
    #    snapsortList] = tools.loadOrRecompute(figuresFolder + "skyplot_data.p",
    #    getAntihaloSkyPlotData,snapNumList,samplesFolder=samplesFolder,\
    #    _recomputeData = recomputeData,recomputeData = recomputeData)
    # IMPORTANT - additionalFilters much match the filter used above for
    # centralAntihalos, otherwise we get inconsistent centralAntihalos lists
    # which leads to errors constructing the plots.
    # TODO - fix this constant reconstructing of the centralAntihalos list.
    # It makes sense to reconstruct it as needed, but we should change it so 
    # that we can provide a precomputed list and only recompute if this 
    # isn't provided.
    [alpha_shape_list_all,largeAntihalos_all,\
        snapsortList_all] = tools.loadOrRecompute(figuresFolder + \
            "skyplot_data_all_snr_filtered.p",\
            getAntihaloSkyPlotData,snapNumList,samplesFolder=samplesFolder,\
            _recomputeData = recomputeData,recomputeData = recomputeData,\
            massRange = [mMin,mMax],nToPlot = 200,\
            rRange = [5,30],reCentreSnaps = True,\
            additionalFilters=snrFilter,rSphere=rSphere,\
            centralAntihalos=centralAntihalos)

#-------------------------------------------------------------------------------
# CATALOGUE DATA:

ns = 0
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
# In spherical polar co-ordinates:
equatorialRThetaPhi = np.vstack((coord.icrs.spherical.distance.value,\
    coord.icrs.spherical.lon.value,\
    coord.icrs.spherical.lat.value)).T



#-------------------------------------------------------------------------------
# PPTs PLOT:
# Names of the clusters of interest for PPT plots:

# PPTs:
# Options:
suffix = ''
#rBins = np.logspace(np.log10(0.1),np.log10(20),31)
rBins = np.linspace(0.1,20,nBinsPPT+1)

binAbs = 0
binApp = 0
nCat = 2*binAbs + binApp

mAbs = np.linspace(-21,-25,9)
mApp = ["m<11.5","12.5 \\leq m < 12.5"]
mAppName = ["Bright catalogue", "Dim Catalogue"]

# Load amplitudes data:
N = 256
nMagBins = 16
restartFile = 'new_chain_restart/merged_restart.h5'
restart = h5py.File(restartFile)
hpIndices = restart['scalars']['colormap3d'][()]
hpIndicesLinear = hpIndices.reshape(N**3)
nside = 4
nsamples = len(snapNumList)

ng2MPP = np.reshape(tools.loadOrRecompute(data_folder + "mg2mppK3.p",\
    survey.griddedGalCountFromCatalogue,\
    cosmo,tmppFile="2mpp_data/2MPP.txt",Kcorrection = True,N=N,\
    _recomputeData=recomputeData),(nMagBins,N**3))
#ngHP = tools.loadOrRecompute(data_folder + "ngHP3.p",\
#    tools.getCountsInHealpixSlices,\
#    ng2MPP,hpIndices,nside=nside,nres=N,_recomputeData=recomputeData)

grid = snapedit.gridListPermutation(N,perm=(2,1,0))
centroids = grid*boxsize/N + boxsize/(2*N)
positions = snapedit.unwrap(centroids - np.array([boxsize/2]*3),boxsize)
tree = scipy.spatial.cKDTree(snapedit.wrap(positions + boxsize/2,boxsize),\
    boxsize=boxsize)


wrappedPos = snapedit.wrap(clusterLoc + boxsize/2,boxsize)
indices = tree.query_ball_point(wrappedPos,10)
clusterAmpsInds = [np.unique(hpIndicesLinear[ind]) for ind in indices]
nGalsList = np.array([[np.sum(ng2MPP[m][indices[l]]) \
    for m in range(0,16)] for l in range(0,9)])


if doPPTs:
    plot.plotPPTProfiles(np.sum(galaxyNumberCountExp,2),\
        np.sum(galaxyNumberCountsRobust,2),\
        savename=figuresFolder + "ppt_Ngal_robust" + suffix + ".pdf",\
        ylim=[1,1000],\
        show=True,rBins=rBins,clusterNames=clusterNames,rescale=False,\
        density=False,legLoc = [0.3,0.1],hspace=0.3,\
        ylabel='Number of galaxies $ < r$',height=0.7,fontsize=8)
    for binAbs in range(0,8):
        for binApp in range(0,2):
            nCat = 2*binAbs + binApp
            plot.plotPPTProfiles(galaxyNumberCountExp[:,:,nCat],\
                galaxyNumberCountsRobust[:,:,nCat],\
                savename=figuresFolder + "ppt_Ngal_robust_" + str(nCat) + \
                suffix + ".pdf",ylim=[1,1000],\
                show=True,rBins=rBins,clusterNames=clusterNames,rescale=False,\
                density=False,legLoc = [0.3,0.1],hspace=0.3,\
                ylabel='Number of galaxies $ < r$',height=0.7,fontsize=8,\
                title="$" + str(mAbs[binAbs]) + " \\leq M < " + \
                str(mAbs[binAbs+1]) + "$, " + mAppName[binApp],top=0.88)
    plot.plotPPTProfiles(np.sum(galaxyNumberCountExp,2),\
        np.sum(galaxyNumberCountsRobustAll,3),\
        savename=figuresFolder + "ppt_Ngal_variance" + suffix + ".pdf",\
        ylim=[1,1000],\
        show=True,rBins=rBins,clusterNames=clusterNames,rescale=False,\
        density=False,legLoc = [0.3,0.1],hspace=0.3,\
        ylabel='Number of galaxies $ < r$',height=0.7,fontsize=8,\
        showPoissonRange=False,color2='grey',showVariance=False)
    plot.plotPPTProfiles(np.sum(galaxyNumberCountExp,2),\
        np.sum(galaxyNumberCountsRobustAll,3),\
        savename=figuresFolder + "ppt_Ngal_poisson" + suffix + ".pdf",\
        ylim=[1,1000],\
        show=True,rBins=rBins,clusterNames=clusterNames,rescale=False,\
        density=False,legLoc = [0.3,0.1],hspace=0.3,\
        ylabel='Number of galaxies $ < r$',height=0.7,fontsize=8,\
        showPoissonRange=True,color2='grey',showVariance=False)

dist = np.sqrt(np.sum(clusterLoc**2,1))

# Variance of a distribution which is an integrated Poisson distribution:
def smoothedPoissonVariance(galCounts):
    expectedCounts = np.mean(galCounts,2)
    varCounts = np.var(galCounts,2)
    return expectedCounts + varCounts

variancesRobust = np.mean(galaxyNumberCountsRobustAll,2) + \
    np.var(galaxyNumberCountsRobustAll,2)

errorType = "bootstrap"

mode = "shells"

if mode == "shells":
    mcmcCounts = galaxyNumberCountsRobustShells
    mcmcCountsAll = galaxyNumberCountsRobustAllShells
    errorCounts = varianceALShell
    counts2MPP = galaxyNumberCountExpShells
    error2MPPAll = interval2MPPBootstrapShells
else:
    mcmcCounts = galaxyNumberCountsRobust
    mcmcCountsAll = galaxyNumberCountsRobustAll
    errorCounts = varianceAL
    error2MPPAll = interval2MPPBootstrap
    counts2MPP = galaxyNumberCountExp

do2MPPerrors = True
doMCMCerrors = False

# PPTs just comparing Coma and PP, in each of the magnitude bins:
nRows = 3
nCols = 2
fontfamily='serif'
fontsize = 8
rBinCentres = plot.binCentres(rBins)
ncList = [0,2]
logscale = False

# Scaling and y limits:
densityPlot = False
if densityPlot:
    if mode == "shells":
        scale = 4*np.pi*(rBins[1:]**3 - rBins[0:-1]**3)/3.0
    else:
        scale = 4*np.pi*rBins[1:]**3/3.0
    if logscale:
        ylim = [1e-4,1]
    else:
        ylim = [0,1]
else:
    scale = 1.0
    if logscale:
        ylim = [1,200]
    else:
        ylim = [0,80]

# Ticks:
xticks = np.array([0,5,10,15,20])
if densityPlot:
    if logscale:
        yticks = np.array([1e-4,1e-3,1e-2,1e-1,1])
    else:
        yticks = np.arange(0,ylim[1],20)
else:
    if logscale:
        yticks = np.array([1,10,100])
    else:
        yticks = np.arange(0,ylim[1],20)

truncateTicks = False
xlim = [0,20]
powerRange = 2
fig, ax = plt.subplots(nRows,4,figsize=(textwidth,0.7*textwidth))
for m in range(2,8):
    i = int((m-2)/nCols)
    j = m - 2 - nCols*i
    magTitle="$" + str(mAbs[m+1]) + " \\leq M_K < " + str(mAbs[m]) + "$"
    # Cluster 1:
    bright = mcmcCounts[:,ncList[0],2*m]/scale
    nz1 = np.where(bright > 0)[0]
    dim = mcmcCounts[:,ncList[0],2*m+1]/scale
    nz2 = np.where(dim > 0)[0]
    bright2Mpp = counts2MPP[:,ncList[0],2*m]/scale
    dim2Mpp = counts2MPP[:,ncList[0],2*m+1]/scale
    dim2MppError = error2MPPAll[:,ncList[0],:,2*m+1].T/scale
    bright2MppError = error2MPPAll[:,ncList[0],:,2*m].T/scale
    nz12Mpp = np.where(bright2Mpp)[0]
    nz22Mpp = np.where(dim2Mpp)[0]
    # Bright catalogue, cluster 1:
    if len(nz1) > 1:
        ax[i,j].plot(rBinCentres[nz1],bright[nz1],\
            color=seabornColormap[1],label="Posterior ($m < 11.5$)",\
            linestyle='-')
    if do2MPPerrors and (len(nz12Mpp) > 1):
        #ax[i,j].errorbar(rBinCentres[nz12Mpp],bright2Mpp[nz12Mpp],\
        #    yerr=bright2MppError[:,nz12Mpp],color=seabornColormap[1],\
        #    label="2M++ ($m < 11.5$)",linestyle='-')
        #ax[i,j].plot(rBinCentres[nz12Mpp],bright2Mpp[nz12Mpp],\
        #    color=seabornColormap[1],label="2M++ ($m < 11.5$)",linestyle='-')
        ax[i,j].fill_between(rBinCentres[nz12Mpp],bright2MppError[0,nz12Mpp],\
            bright2MppError[1,nz12Mpp],alpha=0.5,color=seabornColormap[1])
        ax[i,j].fill_between(rBinCentres[nz12Mpp],bright2MppError[2,nz12Mpp],\
            bright2MppError[3,nz12Mpp],alpha=0.25,color=seabornColormap[1])
    else:
        if len(nz12Mpp) > 1:
            ax[i,j].plot(rBinCentres[nz12Mpp],bright2Mpp[nz12Mpp],\
                color=seabornColormap[1],label="2M++ ($m < 11.5$)",\
                linestyle='-')
    if errorType == "poisson":
        bounds = scipy.stats.poisson(bright[nz1]).interval(0.95)
    elif errorType == "quadrature":
        stdRobust = np.sqrt(variancesRobust[:,ncList[0],2*m])
        bounds = (bright[nz1] - 2*stdRobust[nz1],bright[nz1] + 2*stdRobust[nz1])
    elif errorType == "bootstrap":
        bounds = (errorCounts[nz1,ncList[0],0,2*m],\
            errorCounts[nz1,ncList[0],1,2*m])
    elif errorType == "variance":
        stdDeviation = np.std(mcmcCountsAll,2)[:,ncList[0],2*m]/\
            np.sqrt(nsamples)
        bounds = (bright[nz1] - 2*stdDeviation[nz1],\
            bright[nz1] + 2*stdDeviation[nz1])
    else:
        raise Exception("Invalid errorType!")
    if doMCMCerrors and (len(nz12Mpp) > 1):
        ax[i,j].fill_between(rBinCentres[nz1],bounds[0],bounds[1],\
            color=seabornColormap[1],alpha=0.5)
    # Dim Catalogue, cluster 1:
    if len(nz2) > 1:
        ax[i,j].plot(rBinCentres[nz2],dim[nz2],\
            color=seabornColormap[0],label="Posterior ($m > 11.5$)",\
            linestyle='-')
    if do2MPPerrors and (len(nz22Mpp) > 1):
        #ax[i,j].errorbar(rBinCentres[nz22Mpp],dim2Mpp[nz22Mpp],\
        #    yerr=dim2MppError[:,nz22Mpp],color=seabornColormap[0],\
        #    label="2M++ ($m > 11.5$)",linestyle='-')
        #ax[i,j].plot(rBinCentres[nz22Mpp],dim2Mpp[nz22Mpp],\
        #    color=seabornColormap[0],label="2M++ ($m < 11.5$)",linestyle='-')
        ax[i,j].fill_between(rBinCentres[nz22Mpp],dim2MppError[0,nz22Mpp],\
            dim2MppError[1,nz22Mpp],alpha=0.5,color=seabornColormap[0])
        ax[i,j].fill_between(rBinCentres[nz22Mpp],dim2MppError[2,nz22Mpp],\
            dim2MppError[3,nz22Mpp],alpha=0.25,color=seabornColormap[0])
    else:
        if len(nz22Mpp) > 1:
            ax[i,j].plot(rBinCentres[nz22Mpp],dim2Mpp[nz22Mpp],\
                color=seabornColormap[0],label="2M++ ($m > 11.5$)",\
                linestyle='-')
    if errorType == "poisson":
        bounds = scipy.stats.poisson(dim[nz2]).interval(0.95)
    elif errorType == "quadrature":
        stdRobust = np.sqrt(variancesRobust[:,ncList[0],2*m+1])
        bounds = (dim[nz2] - 2*stdRobust[nz2],dim[nz2] + 2*stdRobust[nz2])
    elif errorType == "bootstrap":
        bounds = (errorCounts[nz2,ncList[0],0,2*m+1],\
            errorCounts[nz2,ncList[0],1,2*m+1])
    elif errorType == "variance":
        stdDeviation = \
            np.std(mcmcCountsAll,2)[:,ncList[0],2*m+1]/\
            np.sqrt(nsamples)
        bounds = (dim[nz2] - 2*stdDeviation[nz2],\
            dim[nz2] + 2*stdDeviation[nz2])
    else:
        raise Exception("Invalid errorType!")
    if doMCMCerrors and (len(nz2) > 1):
        ax[i,j].fill_between(rBinCentres[nz2],bounds[0],bounds[1],\
            color=seabornColormap[0],alpha=0.5)
    ax[i,j].set_ylim(ylim)
    ax[i,j].set_xlim(xlim)
    #ax[i,j].set_title(magTitle,fontsize=fontsize,fontfamily=fontfamily)
    if logscale:
        ax[i,j].text(0.5*(xlim[1] + xlim[0]),\
            ylim[0] + 0.5*(ylim[1] - ylim[0]),magTitle,ha='center',\
            fontfamily=fontfamily,fontsize=fontsize)
    else:
        ax[i,j].text(0.5*(xlim[1] + xlim[0]),\
            ylim[0] + 0.90*(ylim[1] - ylim[0]),magTitle,ha='center',\
            fontfamily=fontfamily,fontsize=fontsize)
    if logscale:
        ax[i,j].set_yscale('log')
    # Cluster 2:
    bright = mcmcCounts[:,ncList[1],2*m]/scale
    nz1 = np.where(bright > 0)[0]
    dim = mcmcCounts[:,ncList[1],2*m+1]/scale
    nz2 = np.where(dim > 0)[0]
    bright2Mpp = counts2MPP[:,ncList[1],2*m]/scale
    dim2Mpp = counts2MPP[:,ncList[1],2*m+1]/scale
    dim2MppError = error2MPPAll[:,ncList[1],:,2*m+1].T/scale
    bright2MppError = error2MPPAll[:,ncList[1],:,2*m].T/scale
    nz12Mpp = np.where(bright2Mpp)[0]
    nz22Mpp = np.where(dim2Mpp)[0]
    # Bright catalogue, cluster 2:
    if len(nz1) > 1:
        ax[i,j+2].plot(rBinCentres[nz1],bright[nz1],\
            color=seabornColormap[1],label="Posterior ($m < 11.5$)",\
            linestyle='-')
    if do2MPPerrors and (len(nz12Mpp) > 1):
        #ax[i,j+2].errorbar(rBinCentres[nz12Mpp],bright2Mpp[nz12Mpp],\
        #    yerr = bright2MppError[:,nz12Mpp],\
        #    color=seabornColormap[1],label="2M++ ($m < 11.5$)",linestyle='-')
        #ax[i,j+2].plot(rBinCentres[nz12Mpp],bright2Mpp[nz12Mpp],\
        #    color=seabornColormap[1],label="2M++ ($m < 11.5$)",linestyle='-')
        ax[i,j+2].fill_between(rBinCentres[nz12Mpp],bright2MppError[0,nz12Mpp],\
            bright2MppError[1,nz12Mpp],alpha=0.5,color=seabornColormap[1])
        ax[i,j+2].fill_between(rBinCentres[nz12Mpp],bright2MppError[2,nz12Mpp],\
            bright2MppError[3,nz12Mpp],alpha=0.25,color=seabornColormap[1])
    else:
        if len(nz12Mpp) > 1:
            ax[i,j+2].plot(rBinCentres[nz12Mpp],bright2Mpp[nz12Mpp],\
                color=seabornColormap[1],label="2M++ ($m < 11.5$)",\
                linestyle='-')
    if errorType == "poisson":
        bounds = scipy.stats.poisson(bright[nz1]).interval(0.95)
    elif errorType == "quadrature":
        stdRobust = np.sqrt(variancesRobust[:,ncList[1],2*m])
        bounds = (bright[nz1] - 2*stdRobust[nz1],bright[nz1] + 2*stdRobust[nz1])
    elif errorType == "bootstrap":
        bounds = (errorCounts[nz1,ncList[1],0,2*m],\
            errorCounts[nz1,ncList[1],1,2*m])
    elif errorType == "variance":
        stdDeviation = np.std(mcmcCountsAll,2)[:,ncList[1],2*m]/\
            np.sqrt(nsamples)
        bounds = (bright[nz1] - 2*stdDeviation[nz1],\
            bright[nz1] + 2*stdDeviation[nz1])
    else:
        raise Exception("Invalid errorType!")
    if doMCMCerrors and (len(nz1) > 1):
        ax[i,j+2].fill_between(rBinCentres[nz1],bounds[0],bounds[1],\
            color=seabornColormap[1],alpha=0.5)
    # Dim Catalogue, cluster 2:
    if len(nz2) > 1:
        ax[i,j+2].plot(rBinCentres[nz2],dim[nz2],\
            color=seabornColormap[0],label="Posterior ($m > 11.5$)",\
            linestyle='-')
    if do2MPPerrors and (len(nz22Mpp) > 1):
        #ax[i,j+2].errorbar(rBinCentres[nz22Mpp],dim2Mpp[nz22Mpp],\
        #    yerr = dim2MppError[:,nz22Mpp],color=seabornColormap[0],\
        #    label="2M++ ($m > 11.5$)",linestyle='-')
        #ax[i,j+2].plot(rBinCentres[nz22Mpp],dim2Mpp[nz22Mpp],\
        #    color=seabornColormap[0],label="2M++ ($m < 11.5$)",linestyle='-')
        ax[i,j+2].fill_between(rBinCentres[nz22Mpp],dim2MppError[0,nz22Mpp],\
            dim2MppError[1,nz22Mpp],alpha=0.5,color=seabornColormap[0])
        ax[i,j+2].fill_between(rBinCentres[nz22Mpp],dim2MppError[2,nz22Mpp],\
            dim2MppError[3,nz22Mpp],alpha=0.25,color=seabornColormap[0])
    else:
        if len(nz22Mpp) > 1:
            ax[i,j+2].plot(rBinCentres[nz22Mpp],dim2Mpp[nz22Mpp],\
                color=seabornColormap[0],label="2M++ ($m > 11.5$)",\
                linestyle='-')
    if errorType == "poisson":
        bounds = scipy.stats.poisson(dim[nz2]).interval(0.95)
    elif errorType == "quadrature":
        stdRobust = np.sqrt(variancesRobust[:,ncList[1],2*m+1])
        bounds = (dim[nz2] - 2*stdRobust[nz2],dim[nz2] + 2*stdRobust[nz2])
    elif errorType == "bootstrap":
        bounds = (errorCounts[nz2,ncList[1],0,2*m+1],\
            errorCounts[nz2,ncList[1],1,2*m+1])
    elif errorType == "variance":
        stdDeviation = \
            np.std(mcmcCountsAll,2)[:,ncList[1],2*m+1]/\
            np.sqrt(nsamples)
        bounds = (dim[nz2] - 2*stdDeviation[nz2],\
            dim[nz2] + 2*stdDeviation[nz2])
    else:
        raise Exception("Invalid errorType!")
    if doMCMCerrors and (len(nz2) > 1):
        ax[i,j+2].fill_between(rBinCentres[nz2],bounds[0],bounds[1],\
            color=seabornColormap[0],alpha=0.5)
    #ax[i,j+2].set_title(magTitle,fontsize=fontsize,fontfamily=fontfamily)
    ax[i,j+2].set_ylim(ylim)
    ax[i,j+2].set_xlim(xlim)
    if logscale:
        ax[i,j+2].text(0.5*(xlim[1] + xlim[0]),\
            ylim[0] + 0.5*(ylim[1] - ylim[0]),magTitle,ha='center',\
            fontfamily=fontfamily,fontsize=fontsize)
    else:
        ax[i,j+2].text(0.5*(xlim[1] + xlim[0]),\
            ylim[0] + 0.90*(ylim[1] - ylim[0]),magTitle,ha='center',\
            fontfamily=fontfamily,fontsize=fontsize)
    if logscale:
        ax[i,j+2].set_yscale('log')


# Formatting the axis:
nCols = 4
nRows = 3

for i in range(0,nRows):
    for j in range(0,nCols):
        ax[i,j].tick_params(axis='both', which='major', labelsize=fontsize)
        ax[i,j].tick_params(axis='both', which='minor', labelsize=fontsize)
        if j != 0:
            # Remove the y labels:
            ax[i,j].yaxis.set_ticklabels([])
        if i != 0 and j == 0:
            # Change tick label fonts:
            if truncateTicks:
                ax[i,j].set_yticks(yticks[0:-1])
                ylabels = ["$" + \
                    plot.scientificNotation(tick,powerRange=powerRange) + "$" \
                    for tick in yticks[0:-1]]
            else:
                ax[i,j].set_yticks(yticks)
                ylabels = ["$" + \
                    plot.scientificNotation(tick,powerRange=powerRange) + "$" \
                    for tick in yticks]
            ax[i,j].yaxis.set_ticklabels(ylabels)
        if j == 0 and i == 0:
            ax[i,j].set_yticks(yticks)
            ylabels = ["$" + \
                plot.scientificNotation(tick,powerRange=powerRange) + "$" \
                for tick in yticks]
            ax[i,j].yaxis.set_ticklabels(ylabels)
        if i != nRows - 1:
            # Remove x labels:
            ax[i,j].xaxis.set_ticklabels([])
        else:
            # Remove the last tick, from all but the last:
            if j < nCols - 1:
                ax[i,j].set_xticks(xticks[0:-1])
                xlabels = ["$" + ("%.2g" % tick) + "$" \
                    for tick in xticks[0:-1]]
                ax[i,j].xaxis.set_ticklabels(xlabels)
            else:
                ax[i,j].set_xticks(xticks)
                xlabels = ["$" + ("%.2g" % tick) + "$" \
                    for tick in xticks]
                ax[i,j].xaxis.set_ticklabels(xlabels)

legendType = "fake"
if legendType == "fake":
    # Legend with a single indicator. Colours will be explained in the caption.
    #fake2MPP = matplotlib.lines.Line2D([0],[0],color='k',\
    #    label='2M++',linestyle='-')
    fakeMCMC = matplotlib.lines.Line2D([0],[0],color='k',\
        label='Mean posterior',linestyle='-')
    fakeError1 = matplotlib.patches.Patch(color='k',alpha=0.5,\
        label='2M++ (68% CI)')
    fakeError2 = matplotlib.patches.Patch(color='k',alpha=0.25,\
        label='2M++ (95% CI)')
    ax[0,0].legend(handles = [fakeMCMC,fakeError1,fakeError2],\
        prop={"size":fontsize,"family":fontfamily},frameon=False,loc=(0.02,0.3))
else:
    # Default legend
    ax[0,0].legend(prop={"size":fontsize,"family":fontfamily},frameon=False,\
        loc=(0.02,0.3))


left = 0.095
bottom = 0.105
top = 0.92
right = 0.980
plt.subplots_adjust(top=top,bottom=bottom,left=left,right=right,\
    hspace=0.0,wspace=0.0)

# Common axis labels:
fig.text((right+left)/2.0, 0.03,'$r\\,[\\mathrm{Mpc}h^{-1}]$',ha='center',\
    fontsize=fontsize,fontfamily=fontfamily)
if densityPlot:
    fig.text(0.03,(top+bottom)/2.0,\
        'Galaxy number density [$h^3\\mathrm{Mpc}^{-3}$]',va='center',\
        rotation='vertical',fontsize=fontsize,fontfamily=fontfamily)
else:
    fig.text(0.03,(top+bottom)/2.0,'Number of galaxies',va='center',\
        rotation='vertical',fontsize=fontsize,fontfamily=fontfamily)

# Cluster names:
fig.text(left + (right - left)*0.25,0.97,clusterNames[ncList[0]][0],\
    fontsize=fontsize,fontfamily=fontfamily,ha='center')
fig.text(left + (right - left)*0.75,0.97,clusterNames[ncList[1]][0],\
    fontsize=fontsize,fontfamily=fontfamily,ha='center')


#plt.savefig(figuresFolder + "ppts_compared_" + clusterNames[ncList[0]][0] + \
#    "_vs_" + clusterNames[ncList[1]][0] + ".pdf")
plt.savefig(figuresFolder + "ppts_compared.pdf")
plt.show()




# Density profiles around each cluster:
nRows = 3
nCols = 3
rBinCentres = plot.binCentres(rBins)
Om0 = 0.3111
rhoM = Om0*2.7754e11
binVolumes = 4*np.pi*rBins[1:]**3/3
textwidth=7.1014
fontfamily = 'serif'
fig, ax = plt.subplots(nRows,nCols,figsize=(textwidth,0.7*textwidth))
for l in range(0,nRows*nCols):
    i = int(l/nCols)
    j = l - nCols*i
    if nCols == 1 and nRows == 1:
        axij = ax
    else:
        axij = ax[i,j]
    meanProfile = np.mean(posteriorMassAll[:,l,:]/(binVolumes[:,None]*rhoM),1)
    stdProfile = np.std(posteriorMassAll[:,l,:]/(binVolumes[:,None]*rhoM),1)
    h1 = axij.plot(rBinCentres,\
        posteriorMassAll[:,l,:]/(binVolumes[:,None]*rhoM),\
        linestyle=':',color='grey',label='Individual sample density')
    h2 = axij.plot(rBinCentres,\
        np.mean(posteriorMassAll[:,l,:]/(binVolumes[:,None]*rhoM),1),\
        linestyle='-',color='k',label='Mean density')
    h3 = axij.fill_between(rBinCentres,\
        meanProfile - stdProfile,meanProfile + stdProfile,\
        alpha=0.5,color='grey',label='Standard deviation')
    axij.set_xlabel('$r [\\mathrm{Mpc}h^{-1}]$')
    axij.set_ylabel('$\\rho/\\bar{\\rho}$')
    axij.set_yscale('log')
    plot.formatPlotGrid(ax,i,j,1,'$\\rho/\\bar{\\rho}$',1,\
        '$r [\\mathrm{Mpc}h^{-1}]$',nRows,[1,1e2],nCols = nCols,fontsize=8,\
        xlim=[0,20])
    axij.tick_params(axis='both', which='major', labelsize=fontsize)
    axij.tick_params(axis='both', which='minor', labelsize=fontsize)
    if i < nRows - 1:
        ax[i,j].xaxis.label.set_visible(False)
        ax[i,j].xaxis.set_major_formatter(NullFormatter())
        ax[i,j].xaxis.set_minor_formatter(NullFormatter())
    if i < nRows -1:
        ax[i,j].get_yticklabels()[0].set_visible(False)
    if j < nCols -1:
        ax[i,j].get_xticklabels()[-1].set_visible(False)
    axij.set_title(clusterNames[l][0],fontsize=8)

plt.suptitle("Cluster Density profiles (redshift space posterior)",\
    fontsize=12)
ax[2,0].legend(handles=[h1[0]],\
    prop={"size":fontsize,"family":fontfamily},frameon=False)
ax[2,1].legend(handles=[h2[0]],\
    prop={"size":fontsize,"family":fontfamily},frameon=False)
ax[2,2].legend(handles=[h3],\
    prop={"size":fontsize,"family":fontfamily},frameon=False)
plt.subplots_adjust(wspace=0.0)
plt.savefig(figuresFolder + "cluster_density_plots.pdf")
plt.show()




# Mass profiles around each cluster:
nRows = 3
nCols = 3
rBinCentres = plot.binCentres(rBins)
Om0 = 0.3111
rhoM = Om0*2.7754e11
binVolumes = 4*np.pi*rBins[1:]**3/3
textwidth=7.1014
fontfamily = 'serif'
fig, ax = plt.subplots(nRows,nCols,figsize=(textwidth,0.7*textwidth))
for l in range(0,nRows*nCols):
    i = int(l/nCols)
    j = l - nCols*i
    if nCols == 1 and nRows == 1:
        axij = ax
    else:
        axij = ax[i,j]
    meanProfile = np.mean(posteriorMassAll[:,l,:],1)
    stdProfile = np.std(posteriorMassAll[:,l,:],1)
    h1 = axij.plot(rBinCentres,\
        posteriorMassAll[:,l,:],\
        linestyle=':',color='grey',label='Individual sample density')
    h2 = axij.plot(rBinCentres,\
        np.mean(posteriorMassAll[:,l,:],1),\
        linestyle='-',color='k',label='Mean mass')
    h3 = axij.fill_between(rBinCentres,\
        meanProfile - stdProfile,meanProfile + stdProfile,\
        alpha=0.5,color='grey',label='Standard deviation')
    axij.set_yscale('log')
    plot.formatPlotGrid(ax,i,j,1,'Cumulative Mass [$M_{\\odot}h^{-1}$]',1,\
        '$r [\\mathrm{Mpc}h^{-1}]$',nRows,[1e13,1e16],nCols = nCols,fontsize=8,\
        xlim=[0,20])
    axij.tick_params(axis='both', which='major', labelsize=fontsize)
    axij.tick_params(axis='both', which='minor', labelsize=fontsize)
    if i < nRows - 1:
        ax[i,j].xaxis.label.set_visible(False)
        ax[i,j].xaxis.set_major_formatter(NullFormatter())
        ax[i,j].xaxis.set_minor_formatter(NullFormatter())
    if i < nRows -1:
        ax[i,j].get_yticklabels()[0].set_visible(False)
    if j < nCols -1:
        ax[i,j].get_xticklabels()[-1].set_visible(False)
    axij.set_title(clusterNames[l][0],fontsize=8)

plt.suptitle("Cluster Mass profiles (redshift space posterior)",\
    fontsize=12)
ax[2,0].legend(handles=[h1[0]],\
    prop={"size":fontsize,"family":fontfamily},frameon=False)
ax[2,1].legend(handles=[h2[0]],\
    prop={"size":fontsize,"family":fontfamily},frameon=False)
ax[2,2].legend(handles=[h3],\
    prop={"size":fontsize,"family":fontfamily},frameon=False)
plt.subplots_adjust(wspace=0.0)
plt.savefig(figuresFolder + "cluster_mass_plots.pdf")
plt.show()


doDenScatter = False
if doDenScatter:
    # Density scatter plot (need to load relevant data):
    plt.clf()
    ns = 0
    for cl in [2]:
        #predicted = np.sum(ngMCMC[ns,:],0)
        predicted = ngMCMC[ns,9]
        plt.scatter(mcmcDenLin_r[ns][indicesGad[ns][cl]],\
            predicted[indicesGad[ns][cl]],marker='.',\
            color=seabornColormap[cl],label=clusterNames[cl][0])
        plt.ylim([0,150])
        plt.xlim([0,60])
    plt.xlabel('$\\rho/\\bar{\\rho} = 1+\\delta$')
    plt.ylabel('$N_{\\mathrm{gal}}$')
    plt.yscale('log')
    plt.ylim([1e-3,200])
    plt.legend()
    plt.savefig(figuresFolder + "voxel_scatter.png")
    plt.show()


# Predicted vs actual:
if doDenScatter:
    plt.clf()
    for cl in [2]:
        ngPredAll = Aalpha[:,:,hpIndicesLinear[indices[cl]]]*\
            ngMCMC[:,:,indices[cl]]
        ngPred = np.mean(ngPredAll,0)
        plt.scatter(np.sum(ng2MPP,0)[indices[cl]],\
            np.sum(ngPred,0),marker='.',\
            color=seabornColormap[cl],label=clusterNames[cl][0])
        plt.xlim([0,20])
        plt.ylim([0,20])
    plt.xlabel('2M++ galaxies')
    plt.ylabel('Predicted galaxies')
    plt.legend()
    plt.savefig(figuresFolder + "voxel_scatter_predicted_vs_actual.png")
    plt.show()


# Density scatter vs actual 2M++ galaxies:
if doDenScatter:
    plt.clf()
    for cl in range(0,9):
        plt.scatter(mcmcDenLin_r[ns][indices[cl]],\
            np.sum(ng2MPP[:],0)[indices[cl]],marker='.',\
            color=seabornColormap[cl],label=clusterNames[cl][0])
    plt.xlabel('$\\rho/\\bar{\\rho} = 1+\\delta$')
    plt.ylabel('$N_{\\mathrm{gal}}$')
    plt.legend()
    plt.savefig(figuresFolder + "voxel_scatter2.png")
    plt.show()

# Mollweide plot of the amplitudes:
nCols = 4
nRows = 4
fig, ax = plt.subplots(nRows,nCols,figsize=(2*textwidth,0.8*2*textwidth))
top = 0.95
bottom = 0.0
left = 0.05
right = 0.95
wspace=0.2
hspace=0.2
plt.subplots_adjust(bottom=bottom,top = top,left=left,right=right,\
    wspace=wspace,hspace=hspace)
ampSlice = 0
nside=4
MabsList = np.linspace(-21,-25,9)
catNames = ['Bright','Dim']
cbarLabel = "$A_{\\alpha}/\\bar{A_{\\alpha}}$"
filterDist = np.where((dist >= ampSlice*60.0) & \
    (dist <= (ampSlice + 1)*60.0))[0]
for m in range(0,8):
    for l in range(0,2):
        nc = 2*m + l
        ncInd = 8*l + m # Index for referencing the axes (different order to 
        # the index for referencing the bins):
        i = int(ncInd/nCols)
        j = ncInd - nCols*i
        #title = catNames[l] + " catalogue, ($" + str(MabsList[m]) + " < M < " \
        #    + str(MabsList[m+1]) + "$, $" + str(60.0*ampSlice) + \
        #    " < D/\\mathrm{Mpc}h^{-1} < " + str(60.0*(ampSlice + 1)) + "$)"
        title = "$" + str(MabsList[m+1]) + " < M_K < " \
            + str(MabsList[m]) + "$"
        nHPPix = 12*nside**2
        dpi=300
        guideColor='grey'
        amps = Aalpha[ns,nc,ampSlice*nHPPix:(ampSlice+1)*nHPPix]
        ampsMean = np.mean(amps)
        hpxMap = amps/ampsMean
        #namesList = [name[0] for name in clusterNames]
        namesList = [str(k+1) for k in range(0,len(clusterNames))]
        plotFormat='.pdf'
        plot.plotLocalUniverseMollweide(135,snapList[ns],hpxMap=hpxMap,\
            nside=nside,alpha_shapes = None,largeAntihalos = None,hr=None,\
            coordAbell = coordCombinedAbellSphere,\
            abellListLocation = [clusterIndMain[k] for k in filterDist],\
            arrowAnnotations=False,\
            nameListLargeClusters = [namesList[k] for k in filterDist],\
            ha = [ha[k] for k in filterDist],va= [va[k] for k in filterDist],\
            annotationPos = [annotationPos[k] for k in filterDist],\
            vmin=1e-1,vmax=1e1,legLoc = 'lower left',\
            bbox_to_anchor = (-0.1,-0.2),snapsort = None,\
            antihaloCentres = None,boundaryOff=False,\
            showFig=False,figsize = (scale*textwidth,scale*0.55*textwidth),\
            voidColour = None,antiHaloLabel=None,\
            bbox_inches = bound_box,galaxyAngles=equatorialRThetaPhi[:,1:],\
            galaxyDistances = equatorialRThetaPhi[:,0],showGalaxies=False,\
            voidAlpha = 0.6,labelFontSize=8,legendFontSize=8,title=title,\
            dpi=dpi,cbarLabel=cbarLabel,ax=ax[i,j],\
            doColorbar=False,sub=(nRows,nCols,ncInd),\
            showLegend=False,reuse_axes=False,margins=(0.2,0.2,0.8,0.8),\
            haloMarker='x')
        axij = plt.gca()

#plt.colorbar(sm,location='bottom',label=cbarLabel,\
#    shrink=0.5,pad=0.05)
#cbax = fig.add_axes([textwidth/4,0.05,textwidth/2,textwidth/16])
#cbar = plt.colorbar(sm, orientation="horizontal",
#    pad=0.05,label=cbarLabel,shrink=0.5,\
#    cax=cbax)
#cbar.ax.tick_params(axis='both',labelsize=legendFontsize)
#cbar.set_label(label = cbarLabel,fontsize = legendFontsize,\
#    fontfamily = "serif")

plt.figure(fig)
#plt.sca(fig.axes)
#fig.subplots_adjust(bottom = 0.2)
im = fig.axes[0].get_images()[0]
sm = cm.ScalarMappable(colors.LogNorm(vmin=1e-1,vmax=1e1),cmap='PuOr_r')
plt.colorbar(sm,ax=fig.axes,location='bottom',label=cbarLabel,\
    shrink=0.5,pad=0.05)
fig.text(0.97,bottom + 0.80*(top-bottom),"Bright catalogue",\
    va='center',rotation='vertical',fontsize=fontsize,fontfamily=fontfamily)
fig.text(0.97,bottom + 0.4*(top-bottom),"Dim catalogue",\
    va='center',rotation='vertical',fontsize=fontsize,fontfamily=fontfamily)
plt.suptitle("Healpix Amplitudes, $" + str(60.0*ampSlice) + \
    " < D/\\mathrm{Mpc}h^{-1} < " + str(60.0*(ampSlice + 1)) + "$")
plt.savefig(figuresFolder + "voxel_amplitude_distribution_all_cats_slice_" + \
    str(ampSlice) + ".pdf",bbox_inches = 'tight')
plt.show()



# Mollweide plot of the amplitudes (individual):
top = 0.95
bottom = 0.0
left = 0.05
right = 0.95
wspace=0.2
hspace=0.2
nCols = 4
nRows = 4
top = 0.95
bottom = 0.0
left = 0.05
right = 0.95
wspace=0.2
hspace=0.2
ampSlice = 1
nside=4
MabsList = np.linspace(-21,-25,9)
catNames = ['Bright','Dim']
cbarLabel = "$A_{\\alpha}/\\bar{A_{\\alpha}}$"
filterDist = np.where((dist >= ampSlice*60.0) & \
    (dist <= (ampSlice + 1)*60.0))[0]
for m in range(0,8):
    for l in range(0,2):
        nc = 2*m + l
        ncInd = 8*l + m # Index for referencing the axes (different order to 
        # the index for referencing the bins):
        i = int(ncInd/nCols)
        j = ncInd - nCols*i
        title = catNames[l] + " catalogue, ($" + str(MabsList[m+1]) + " < M_K < " \
            + str(MabsList[m]) + "$, $" + str(60.0*ampSlice) + \
            " < D/\\mathrm{Mpc}h^{-1} < " + str(60.0*(ampSlice + 1)) + "$)"
        nHPPix = 12*nside**2
        dpi=300
        guideColor='grey'
        amps = Aalpha[ns,nc,ampSlice*nHPPix:(ampSlice+1)*nHPPix]
        ampsMean = np.mean(amps)
        hpxMap = amps/ampsMean
        namesList = [name[0] for name in clusterNames]
        #namesList = [str(k+1) for k in range(0,len(clusterNames))]
        plotFormat='.pdf'
        fig, ax = plt.subplots(figsize=(textwidth,0.55*textwidth))
        plt.subplots_adjust(bottom=bottom,top = top,left=left,right=right,\
            wspace=wspace,hspace=hspace)
        plot.plotLocalUniverseMollweide(135,snapList[ns],hpxMap=hpxMap,\
            nside=nside,alpha_shapes = None,largeAntihalos = None,hr=None,\
            coordAbell = coordCombinedAbellSphere,\
            abellListLocation = [clusterIndMain[k] for k in filterDist],\
            arrowAnnotations=True,\
            figOut = figuresFolder + "voxel_healpix_distribution_cat_" + \
            str(nc) + "_slice_" + str(ampSlice) + ".pdf",\
            nameListLargeClusters = [namesList[k] for k in filterDist],\
            ha = [ha[k] for k in filterDist],va= [va[k] for k in filterDist],\
            annotationPos = [annotationPos[k] for k in filterDist],\
            vmin=1e-1,vmax=1e1,legLoc = 'lower left',\
            bbox_to_anchor = (-0.1,-0.2),snapsort = None,\
            antihaloCentres = None,boundaryOff=False,\
            showFig=False,figsize = (scale*textwidth,scale*0.55*textwidth),\
            voidColour = None,antiHaloLabel=None,\
            bbox_inches = 'tight',galaxyAngles=equatorialRThetaPhi[:,1:],\
            galaxyDistances = equatorialRThetaPhi[:,0],showGalaxies=False,\
            voidAlpha = 0.6,labelFontSize=8,legendFontSize=8,title=title,\
            dpi=dpi,cbarLabel=cbarLabel,ax=None,\
            doColorbar=True,sub=(nRows,nCols,ncInd),\
            showLegend=True,reuse_axes=False,margins=(0.2,0.2,0.8,0.8))

plt.show()


# Histograms of A by bin:

# Cluster A values:
nc = 2
nCols = 4
nRows = 2
logMlow = 13
logMhigh = 15.2
nBins = 25
fontsize=8
textwidth=7.1014
ylim=[0,10]

#plt.clf()
fig, ax = plt.subplots(nRows,nCols,figsize=(textwidth,textwidth))
plt.subplots_adjust(hspace=0.16,wspace=0.0)
MabsList = np.linspace(-21,-25,9)
amps = np.mean(Aalpha,0)
for l in range(0,8):
    i = int(l/nCols)
    j = l - nCols*i
    ax[i,j].hist(amps[2*l,:],\
        bins=np.logspace(-4,4,nBins),alpha=0.5,\
        color=seabornColormap[0],label='Bright catalogue',density=True)
    ax[i,j].hist(amps[2*l+1,:],\
        bins=np.logspace(-3,3,nBins),alpha=0.5,\
        color=seabornColormap[1],label='Dim catalogue',density=True)
    for k in clusterAmpsInds[nc]:
        ax[i,j].axvline(np.mean(amps[2*l,k]),linestyle='--',\
            color=seabornColormap[0])
        ax[i,j].axvline(np.mean(amps[2*l+1,k]),linestyle='--',\
            color=seabornColormap[1])
    #plot.formatPlotGrid(ax,i,j,1,'Density',1,\
    #    '$A_{\\alpha}$',nRows,[1e-4,1e4],nCols = nCols,fontsize=8,\
    #    xlim=[1e-4,1e4])
    ax[i,j].set_xscale('log')
    ax[i,j].set_yscale('log')
    ax[i,j].set_xlim([1e-3,1e3])
    ax[i,j].set_ylim([1e-4,1e4])
    #title = "$" + str(MabsList[l]) + " < M \\leq" + \
    #    str(MabsList[l+1]) + "$\nBright/Dim = " + \
    #    str(int(nGalsList[nc][2*l])) + "/" + str(int(nGalsList[nc][2*l+1]))
    title = "$" + str(MabsList[l+1]) + " < M_K \\leq" + \
        str(MabsList[l]) + "$"
    ax[i,j].set_title(title,fontsize=fontsize)
    ax[i,j].set_xticks([1e-3,1e-1,1e1,1e3])
    if j > 0:
        ax[i,j].yaxis.set_ticklabels([])
    if i < nRows - 1:
        ax[i,j].xaxis.set_ticklabels([])
    if i < nRows - 1:
        ax[i,j].xaxis.label.set_visible(False)
        ax[i,j].xaxis.set_major_formatter(NullFormatter())
        ax[i,j].xaxis.set_minor_formatter(NullFormatter())
    if i < nRows -1:
        ax[i,j].get_yticklabels()[0].set_visible(False)
    if j < nCols -1:
        ax[i,j].get_xticklabels()[-1].set_visible(False)

ax[0,0].legend(prop={"size":fontsize,"family":"serif"},frameon=False,\
    loc="upper right")

top=0.88
bottom=0.11
left=0.125
right=0.9
hspace=0.16
wspace=0.0
plt.subplots_adjust(top=top,bottom=bottom,left=left,right=right,hspace=hspace,\
    wspace=wspace)

# Common axis labels:
fig.text((right+left)/2.0, 0.03,'$A_{\\alpha}$',ha='center',\
    fontsize=fontsize,fontfamily=fontfamily)
fig.text(0.03,(top+bottom)/2.0,'Density',va='center',\
    rotation='vertical',fontsize=fontsize,fontfamily=fontfamily)


plt.suptitle("All Amplitudes across 5 MCMC resimulations vs " + \
    clusterNames[nc][0] + " amplitudes")
plt.savefig(figuresFolder + "voxel_amp_distribution_" + clusterNames[nc][0] + \
    ".pdf")
plt.show()

# A different plot, showing two clusters compared:


#plt.clf()
nRows = 2
nCols = 5
nBins = 21
fig, ax = plt.subplots(nRows,nCols,figsize=(textwidth,0.45*textwidth))
plt.subplots_adjust(hspace=0.16,wspace=0.0)
MabsList = np.linspace(-21,-25,9)
amps = np.mean(Aalpha,0)
ncList = [0,2]
binsDefault = np.logspace(-3,2,nBins)
#bins = np.linspace(0,100,nBins)
logBinsDefault = np.linspace(-3,2,nBins)

# Variable bins?
variableBins = [\
    np.linspace(0,100,nBins),\
    np.linspace(0,10,nBins),\
    np.linspace(0,10,nBins),\
    np.linspace(0,0.1,nBins),\
    np.linspace(0,2,nBins)]
variableLogBins = [\
    np.linspace(-2,3,nBins),\
    np.linspace(-2,1,nBins),\
    np.linspace(-2,1,nBins),\
    np.linspace(-3,-1,nBins),\
    np.linspace(-2,1,nBins)]

pad = 0.25
variableXlims = [[0,100],[0,10],[0,10],[0,0.1],[0,1]]
variableLogXlims = [[-2-pad,3+pad],[-2-pad,1+pad],[-2-pad,1+pad],\
    [-3-pad,-1+pad],[-2 - pad,pad]]
variableLogXTicks = [[-2,-1,0,1,2],[-2,-1,0,1],[-2,-1,0,1],[-3,-2,-1],[-2,-1,0]]
yLimitsLog = [[0,1.95],[0,2.9]]
yLimitsLogDefault = [0,2.4]
yLimits = [[0,100],[0,100]]
yLimitDefault = [0,100]
yTicksLogList = [np.arange(ylim[0],ylim[1],0.5) for ylim in yLimitsLog]
yTicksLinList = [np.arange(ylim[0],ylim[1],0.5) for ylim in yLimits]

useLogBins = True
useVariableBins = True
useDifferentYlims = True
for i in range(0,2):
    nc = ncList[i]
    for l in range(3,8):
        j = l - 3
        ax[i,j].tick_params(axis='both',which='major',labelsize=8)
        ax[i,j].tick_params(axis='both',which='minor',labelsize=8)
        if useVariableBins:
            logBins = variableLogBins[l-3]
            bins = variableBins[l-3]
            xLimits = variableXlims[l-3]
            xLogLimits = variableLogXlims[l-3]
            xLogTicks = variableLogXTicks[l-3]
        else:
            logBins = logBinsDefault
            bins = binsDefault
            xLimits = [0,100]
            xLogLimits = [-3,3]
            xLogTicks = [-3,-2,-1,0,1,2,3]
        xMid = np.mean(xLimits)
        xLogMid = np.mean(xLogLimits)
        if useDifferentYlims:
            yLimLog = yLimitsLog[i]
            yLim = yLimits[i]
            yTicksLog = yTicksLogList[i]
            yTicksLin = yTicksLinList[i]
        else:
            yLimLog = yLimitsLogDefault
            yLim = yLimitDefault
            yTicksLog = np.arange(yLimitsLogDefault[0],yLimitsLogDefault[1],0.5)
            yTicksLin = np.arange(yLimitDefault[0],yLimitDefault[1],0.5)
        if nGalsList[nc][2*l] > 0:
            if useLogBins:
                bars1 = ax[i,j].hist(np.log10(amps[2*l,amps[2*l,:] > 0]),\
                    bins=logBins,alpha=0.5,\
                    color=seabornColormap[1],label='Bright\n (all\namps.)',\
                    density=True)
            else:
                bars1 = ax[i,j].hist(amps[2*l,amps[2*l,:] > 0],\
                    bins=bins,alpha=0.5,\
                    color=seabornColormap[1],label='Bright\n (all\namps.)',\
                    density=True)
        if nGalsList[nc][2*l+1] > 0:
            if useLogBins:
                bars2 = ax[i,j].hist(np.log10(amps[2*l+1,amps[2*l+1,:] > 0]),\
                    bins=logBins,alpha=0.5,\
                    color=seabornColormap[0],label='Dim\n (all\namps.)',\
                    density=True)
            else:
                bars2 = ax[i,j].hist(amps[2*l+1,amps[2*l+1,:] > 0],\
                    bins=bins,alpha=0.5,\
                    color=seabornColormap[0],label='Dim\n (all\namps.)',\
                    density=True)
        for k in clusterAmpsInds[nc]:
            #h3 = ax[i,j].axvline(np.mean(amps[2*l,k]),linestyle='--',\
            #    color=seabornColormap[0],label='Bright\n(cluster\namps.)')
            #h4 = ax[i,j].axvline(np.mean(amps[2*l+1,k]),linestyle='--',\
            #    color=seabornColormap[1],label='Dim\n(cluster\namps.)')
            if nGalsList[nc][2*l] > 0 and (amps[2*l,k] > 0):
                if useLogBins:
                    xPos = np.log10(np.mean(amps[2*l,k]))
                    inBin = np.where((logBins[0:-1] < xPos) & \
                        (logBins[1:] >= xPos))[0][0]
                    yExtent = yLimLog[1] - yLimLog[0]
                else:
                    xPos = np.mean(amps[2*l,k])
                    inBin = np.where((bins[0:-1] < xPos) & \
                        (bins[1:] >= xPos))[0][0]
                    yExtent = yLim[1] - yLim[0]
                yPos = bars1[0][inBin]
                #arrow1 = ax[i,j].scatter([xPos],[yPos],\
                #    color=seabornColormap[1],label='Bright\n(cluster\namps.)',\
                #    linestyle='None',marker='$\\downarrow$',s=100)
                arrow2 = ax[i,j].annotate("",xy=(xPos,yPos),\
                    xytext=(xPos,yPos+0.15*yExtent),\
                    arrowprops=dict(arrowstyle='->',color=seabornColormap[1]))
            if nGalsList[nc][2*l+1] > 0 and (amps[2*l+1,k] > 0):
                if useLogBins:
                    xPos = np.log10(np.mean(amps[2*l+1,k]))
                    inBin = np.where((logBins[0:-1] < xPos) & \
                        (logBins[1:] >= xPos))[0][0]
                    yExtent = yLimLog[1] - yLimLog[0]
                else:
                    xPos = np.mean(amps[2*l+1,k])
                    inBin = np.where((bins[0:-1] < xPos) & \
                        (bins[1:] >= xPos))[0][0]
                    yExtent = yLim[1] - yLim[0]
                yPos = bars2[0][inBin]
                #arrow2 = ax[i,j].scatter([xPos],[yPos],\
                #    color=seabornColormap[0],label='Dim\n(cluster\namps.)',\
                #    linestyle='None',marker='$\\downarrow$',s=100)
                arrow2 = ax[i,j].annotate("",xy=(xPos,yPos),\
                    xytext=(xPos,yPos+0.15*yExtent),\
                    arrowprops=dict(arrowstyle='->',color=seabornColormap[0]))
        #plot.formatPlotGrid(ax,i,j,1,'Density',1,\
        #    '$A_{\\alpha}$',nRows,[1e-4,1e4],nCols = nCols,fontsize=8,\
        #    xlim=[1e-4,1e4])
        #ax[i,j].set_xscale('log')
        #ax[i,j].set_yscale('log')
        if useLogBins:
            ax[i,j].set_xlim(xLogLimits)
            ax[i,j].set_ylim(yLimLog)
            ax[i,j].set_yticks(yTicksLog)
        else:
            ax[i,j].set_xlim(xLimits)
            #ax[i,j].set_ylim([1e-1,100])
            ax[i,j].set_ylim(yLim)
            ax[i,j].set_yticks(yTicksLin)
            #ax[i,j].set_yscale('log')
        #title = "$" + str(MabsList[l]) + " < M \\leq" + \
        #    str(MabsList[l+1]) + "$\nBright: " + \
        #    ("%.2g" % (100*nGalsList[nc][2*l]/np.sum(nGalsList[nc]))) + "%"\
        #    "(" + ("%.2g" % (nGalsList[nc][2*l])) + ")" + \
        #    "\n Dim: " + \
        #    ("%.2g" % (100*nGalsList[nc][2*l+1]/np.sum(nGalsList[nc]))) + \
        #    "%" + "(" + ("%.2g" % (nGalsList[nc][2*l+1])) + ")"
        title = "$" + str(MabsList[l+1]) + " \\leq M_K <" + \
            str(MabsList[l]) + "$\n$" + ("%.2g" % (nGalsList[nc][2*l] + \
            nGalsList[nc][2*l+1])) + "$ galaxies."
        if useLogBins:
            ax[i,j].text(xLogMid,0.82*yLimLog[1],title,fontsize=fontsize,\
                ha='center')
            ax[i,j].set_xticks(xLogTicks)
            xLabels = ["$" + ("%.2g" % tick) + "$" for tick in xLogTicks]
            ax[i,j].xaxis.set_ticklabels(xLabels)
        else:
            ax[i,j].text(xMid,0.82*yLim[1],title,fontsize=fontsize,ha='center')
            xLabels = ["$" + ("%.2g" % tick) + "$" \
                for tick in ax[i,j].get_xticks()]
            ax[i,j].xaxis.set_ticklabels(xLabels)
        #ax[i,j].set_xticks([-3,-2,-1,0,1,2])
        #ax[i,j].set_xticks([-3,-2,-1,0,1,2])
        if j > 0:
            ax[i,j].yaxis.set_ticklabels([])
            print("Removing labels for (" + str(i) + "," + str(j) + ")")
        else:
            stringTicks = ["$" + ("%.2g" % tick)  + "$" \
                for tick in ax[i,j].get_yticks()]
            ax[i,j].yaxis.set_ticklabels(stringTicks)
        if i < nRows - 1:
            ax[i,j].xaxis.set_ticklabels([])
            #ax[i,j].xaxis.label.set_visible(False)
            #ax[i,j].xaxis.set_major_formatter(NullFormatter())
            #ax[i,j].xaxis.set_minor_formatter(NullFormatter())
        else:
            stringTicks = ["$" + ("%.2g" % tick)  + "$" \
                for tick in ax[i,j].get_xticks()]
            ax[i,j].xaxis.set_ticklabels(stringTicks)
        #if i < nRows -1:
        #    ax[i,j].get_yticklabels()[0].set_visible(False)
        if j < nCols -1:
            if not useLogBins:
                ax[i,j].get_xticklabels()[-1].set_visible(False)

# Fake legend entries:
h1 = matplotlib.patches.Patch(color=seabornColormap[1],alpha=0.5,\
    label='Bright\n amplitudes')
h2 = matplotlib.patches.Patch(color=seabornColormap[0],alpha=0.5,\
    label='Dim\n amplitudes')
h3 = matplotlib.lines.Line2D([0],[0],color=seabornColormap[1],\
    label='Bright\n amplitudes\n at cluster',linestyle='None',\
    marker='$\\downarrow$',markersize=10)
h4 = matplotlib.lines.Line2D([0],[0],color=seabornColormap[0],\
    label='Dim\n amplitudes\n at cluster',linestyle='None',\
    marker='$\\downarrow$',markersize=10)

showLegend = False
if showLegend:
    ax[0,0].legend(handles = [h1],\
        prop={"size":fontsize,"family":"serif"},frameon=False,\
        loc=(0.1,0.6))
    ax[0,1].legend(handles = [h2],\
        prop={"size":fontsize,"family":"serif"},frameon=False,\
        loc=(0.1,0.6))
    ax[0,2].legend(handles = [h3],\
        prop={"size":fontsize,"family":"serif"},frameon=False,\
        loc=(0.1,0.65))
    ax[0,3].legend(handles = [h4],\
        prop={"size":fontsize,"family":"serif"},frameon=False,\
        loc=(0.1,0.65))

top=0.975
bottom=0.12
left=0.065
right=0.97
hspace=0.0
wspace=0.0
plt.subplots_adjust(top=top,bottom=bottom,left=left,right=right,hspace=hspace,\
    wspace=wspace)

# Common axis labels:
if useLogBins:
    fig.text((right+left)/2.0, 0.01,\
        '$\\mathrm{log}_{\\mathrm{10}}(A_{\\alpha})$',\
        ha='center',fontsize=fontsize,fontfamily=fontfamily)
else:
    fig.text((right+left)/2.0, 0.01,'$A_{\\alpha}$',\
        ha='center',fontsize=fontsize,fontfamily=fontfamily)

fig.text(0.01,(top+bottom)/2.0,'Density',va='center',\
    rotation='vertical',fontsize=fontsize,fontfamily=fontfamily)

# Cluster names:
fig.text(0.98,bottom + 0.75*(top-bottom),clusterNames[ncList[0]][0],\
    va='center',rotation='vertical',fontsize=fontsize,fontfamily=fontfamily)
fig.text(0.98,bottom + 0.25*(top-bottom),clusterNames[ncList[1]][0],\
    va='center',rotation='vertical',fontsize=fontsize,fontfamily=fontfamily)

#plt.suptitle("All amplitudes compared to " + clusterNames[ncList[0]][0] + \
#    " and " + clusterNames[ncList[1]][0])
plt.savefig(figuresFolder + "voxel_amp_distribution_compared.pdf")
plt.show()





#-------------------------------------------------------------------------------
# HMF/AMF PLOT:

savename = figuresFolder + "hmf_amf_old_vs_new_forward_model.pdf"
if doHMFs:
    plot.plotHMFAMFComparison(\
            constrainedHaloMasses512Old,deltaListMeanOld,deltaListErrorOld,\
            comparableHaloMassesOld,constrainedAntihaloMasses512Old,\
            comparableAntihaloMassesOld,\
            constrainedHaloMasses512New,deltaListMeanNew,deltaListErrorNew,\
            comparableHaloMassesNew,constrainedAntihaloMasses512New,\
            comparableAntihaloMassesNew,\
            referenceSnap,referenceSnapOld,\
            savename = figuresFolder + "test_hmf_amf.pdf",\
            ylabelStartOld = 'PM10 forward model',\
            ylabelStartNew = 'COLA20 forward model',\
            fontsize=fontsize,legendFontsize=legendFontsize,density=False,\
            xlim = (mLimLower,3e15),nMassBins=7,mLower=1e14,mUpper=2e15,\
            ylim=(1e-1,200),showTheory=False,sigma8List=[0.8288,0.8102])


# Plots of individual mass functions:

plot.singleMassFunctionPlot(constrainedHaloMasses512Old,5e13,2e15,11,\
    Om0=referenceSnapOld.properties['omegaM0'],\
    h=referenceSnapOld.properties['h'],ns=0.9611,sigma8=0.8288,\
    Ob0=0.04825,mLimLower=mLimLower,\
    comparableHaloMasses=comparableHaloMassesOld,\
    deltaListMean=deltaListMeanOld,deltaListError=deltaListErrorOld,\
    savename=figuresFolder + "pm10_mass_function.pdf",showTheory=False)


plot.singleMassFunctionPlot(constrainedHaloMasses512New,5e13,2e15,11,\
    Om0=referenceSnap.properties['omegaM0'],\
    h=referenceSnap.properties['h'],ns=0.9665,sigma8=0.8102,\
    Ob0=0.04897468161869667,mLimLower=mLimLower,\
    comparableHaloMasses=comparableHaloMassesNew,\
    deltaListMean=deltaListMeanNew,deltaListError=deltaListErrorNew,\
    savename=figuresFolder + "cola20_mass_function.pdf",\
    label="COLA20",showTheory=False)


# HMF/AMF comparison:
textwidth=7.1014
fontsize=8
fig, ax = plt.subplots(1,2,figsize=(textwidth,0.5*textwidth))
plot.singleMassFunctionPlot(constrainedHaloMasses512Old,5e13,2e15,11,\
    Om0=referenceSnapOld.properties['omegaM0'],\
    h=referenceSnapOld.properties['h'],ns=0.9611,sigma8=0.8288,\
    Ob0=0.04825,mLimLower=mLimLower,\
    fontsize=fontsize,legendFontsize=fontsize,\
    comparableHaloMasses=comparableHaloMassesOld,\
    deltaListMean=deltaListMeanOld,deltaListError=deltaListErrorOld,\
    savename=None,showTheory=False,legendLoc='lower left',\
    ax=ax[0],ylabel='Number of halos',showLegend=False,\
    title="Halos",xticks=[2e14,5e14,1e15])
handles = plot.singleMassFunctionPlot(constrainedAntihaloMasses512Old,\
    5e13,2e15,11,Om0=referenceSnapOld.properties['omegaM0'],\
    h=referenceSnapOld.properties['h'],ns=0.9611,sigma8=0.8288,\
    Ob0=0.04825,mLimLower=mLimLower,\
    fontsize=fontsize,legendFontsize=fontsize,\
    comparableHaloMasses=comparableAntihaloMassesOld,\
    deltaListMean=deltaListMeanOld,deltaListError=deltaListErrorOld,\
    savename=None,showTheory=False,\
    ax=ax[1],ylabel='Number of antihalos',returnHandles=True,showLegend=False,\
    title="Antihalos",xticks=[2e14,5e14,1e15])
plt.tight_layout()
ax[1].legend(handles=tools.flatten(handles),\
    prop={"size":fontsize,"family":"serif"},loc='upper right',frameon=False)
plt.savefig(figuresFolder + "pm10_amf_hmf.pdf")
plt.show()




# HMF/AMF comparison:
textwidth=7.1014
fontsize=8
fig, ax = plt.subplots(1,2,figsize=(textwidth,0.5*textwidth))
plot.singleMassFunctionPlot(constrainedHaloMasses512New,5e13,2e15,11,\
    Om0=referenceSnapOld.properties['omegaM0'],\
    h=referenceSnapOld.properties['h'],ns=0.9611,sigma8=0.8288,\
    Ob0=0.04825,mLimLower=mLimLower,\
    comparableHaloMasses=comparableHaloMassesNew,\
    deltaListMean=deltaListMeanNew,deltaListError=deltaListErrorNew,\
    savename=None,showTheory=False,legendLoc='lower left',\
    fontsize=fontsize,legendFontsize=fontsize,\
    ax=ax[0],ylabel='Number of halos',showLegend=False,\
    title="Halos",label="COLA20",xticks=[2e14,5e14,1e15])
handles = plot.singleMassFunctionPlot(constrainedAntihaloMasses512New,\
    5e13,2e15,11,Om0=referenceSnapOld.properties['omegaM0'],\
    h=referenceSnapOld.properties['h'],ns=0.9611,sigma8=0.8288,\
    Ob0=0.04825,mLimLower=mLimLower,\
    comparableHaloMasses=comparableAntihaloMassesNew,\
    deltaListMean=deltaListMeanNew,deltaListError=deltaListErrorNew,\
    savename=None,showTheory=False,fontsize=fontsize,legendFontsize=fontsize,\
    ax=ax[1],ylabel='Number of antihalos',returnHandles=True,showLegend=False,\
    title="Antihalos",label="COLA20",xticks=[2e14,5e14,1e15])
plt.tight_layout()
ax[1].legend(handles=tools.flatten(handles),\
    prop={"size":fontsize,"family":"serif"},loc='upper right',frameon=False)
plt.savefig(figuresFolder + "cola20_amf_hmf.pdf")
plt.show()

# Combined on one plot:

# HMF/AMF comparison:
fontsize=8
fig, ax = plt.subplots(1,2,figsize=(textwidth,0.5*textwidth))
plot.singleMassFunctionPlot(constrainedHaloMasses512Old,5e13,2e15,11,\
    Om0=referenceSnapOld.properties['omegaM0'],\
    h=referenceSnapOld.properties['h'],ns=0.9611,sigma8=0.8288,\
    Ob0=0.04825,mLimLower=mLimLower,\
    fontsize=fontsize,legendFontsize=fontsize,\
    comparableHaloMasses=None,\
    deltaListMean=deltaListMeanOld,deltaListError=deltaListErrorOld,\
    savename=None,showTheory=False,legendLoc='lower left',\
    ax=ax[0],ylabel='Number of halos',showLegend=False,\
    title="Halos",xticks=[2e14,5e14,1e15],plotColour=seabornColormap[2],\
    compColour = 'grey')
handles1 = plot.singleMassFunctionPlot(constrainedAntihaloMasses512Old,\
    5e13,2e15,11,Om0=referenceSnapOld.properties['omegaM0'],\
    h=referenceSnapOld.properties['h'],ns=0.9611,sigma8=0.8288,\
    Ob0=0.04825,mLimLower=mLimLower,\
    fontsize=fontsize,legendFontsize=fontsize,\
    comparableHaloMasses=None,\
    deltaListMean=deltaListMeanOld,deltaListError=deltaListErrorOld,\
    savename=None,showTheory=False,\
    ax=ax[1],ylabel='Number of antihalos',returnHandles=True,showLegend=False,\
    title="Antihalos",xticks=[2e14,5e14,1e15],plotColour=seabornColormap[2],\
    compColour = 'grey')
plot.singleMassFunctionPlot(constrainedHaloMasses512New,5e13,2e15,11,\
    Om0=referenceSnapOld.properties['omegaM0'],\
    h=referenceSnapOld.properties['h'],ns=0.9611,sigma8=0.8288,\
    Ob0=0.04825,mLimLower=mLimLower,\
    comparableHaloMasses=comparableHaloMassesNew,\
    deltaListMean=deltaListMeanNew,deltaListError=deltaListErrorNew,\
    savename=None,showTheory=False,legendLoc='lower left',\
    fontsize=fontsize,legendFontsize=fontsize,\
    ax=ax[0],ylabel='Number of halos',showLegend=False,\
    title="Halos",label="COLA20",xticks=[2e14,5e14,1e15],\
    plotColour=seabornColormap[4],compColour = 'grey')
handles2 = plot.singleMassFunctionPlot(constrainedAntihaloMasses512New,\
    5e13,2e15,11,Om0=referenceSnapOld.properties['omegaM0'],\
    h=referenceSnapOld.properties['h'],ns=0.9611,sigma8=0.8288,\
    Ob0=0.04825,mLimLower=mLimLower,\
    comparableHaloMasses=comparableAntihaloMassesNew,\
    deltaListMean=deltaListMeanNew,deltaListError=deltaListErrorNew,\
    savename=None,showTheory=False,fontsize=fontsize,legendFontsize=fontsize,\
    ax=ax[1],ylabel='Number of antihalos',returnHandles=True,showLegend=False,\
    title="Antihalos",label="COLA20",xticks=[2e14,5e14,1e15],\
    plotColour=seabornColormap[4],compColour = 'grey')
handles = handles1 + handles2
plt.tight_layout()
ax[1].legend(handles=tools.flatten(handles),\
    prop={"size":fontsize,"family":"serif"},loc='upper right',frameon=False)
plt.savefig(figuresFolder + "pm10_vs_cola20_amf_hmf.pdf")
plt.show()



#-------------------------------------------------------------------------------
# HMF/AMF PLOT, UNDERDENSE:
if doHMFs:
    plot.plotHMFAMFUnderdenseComparison(\
            constrainedHaloMasses512New,deltaListMeanNew,deltaListErrorNew,\
            comparableHaloMassesNew,constrainedAntihaloMasses512New,\
            comparableAntihaloMassesNew,centralHalosNew,centralAntihalosNew,\
            centralHaloMassesNew,centralAntihaloMassesNew,showTheory=False,\
            savename = figuresFolder + "hmf_amf_underdense_comparison.pdf",\
            xlim = (mLimLower,3e15),nMassBins=11,mLower=1e14,mUpper=3e15)

#-------------------------------------------------------------------------------
# RADIAL VOID PROFILES
#if doCat:
#    plot.plotConstrainedVsUnconstrainedProfiles(rBinStackCentres,nbarjSepStack,\
#        sigmaSepStack,nbarjAllStackedUn,sigmaAllStackedUn,nbar,rMin,mMin,mMax,\
#        showImmediately = True,fontsize = fontsize,\
#        legendFontSize=legendFontsize,labelCon='Constrained',\
#        labelRand='Unconstrained \nmean',\
#        savename=figuresFolder + "profiles1415.pdf",showTitle=False,\
#        meanErrorLabel = 'Unconstrained \nMean',\
#        profileErrorLabel = 'Profile \nvariation \n',\
#        nbarjUnconstrainedStacks=nbarjSepStackUn,\
#        sigmajUnconstrainedStacks = sigmaSepStackUn,showMean=True)


#-------------------------------------------------------------------------------
# BIAS FUNCTIONAL FORM


biasDataRootDir="new_chain/sample"
biasData = [h5py.File(biasDataRootDir + str(k) + "/mcmc_" + str(k) + \
    ".h5",'r') for k in snapNumList]

biasParam = [np.array([[sample['scalars']['galaxy_bias_' + str(k)][()] \
    for k in range(0,16)]]) for sample in  biasData]

def plotBiasForm(deltaRange,biasFormLowMean,biasFormHighMean,\
        sigmaBiasFormLow,sigmaBiasFormHigh,colorLow='r',colorHigh='g',\
        ax = None,savename=None,show=False,ylim=[1e-14,1],showLegend=True,\
        returnHandles=False,ylabel = '$f(\\delta,b,\\rho,\\epsilon)$',\
        xlabel='$\\rho = 1 + \\delta$',fontfamily='serif',fontsize=8,\
        label1 = "$m \\leq 11.5$",label2 = "$11.5 < m \\leq 12.5$",\
        showMean=True):
    if ax is None:
        fig, ax = plt.subplots()
    if showMean:
        h1 = ax.plot(deltaRange+1.0,biasFormLowMean,linestyle=':',\
            color=colorLow,label=label1)
    h2 = ax.fill_between(deltaRange+1.0,biasFormLowMean + sigmaBiasFormLow,\
        biasFormLowMean - sigmaBiasFormLow,color=colorLow,alpha=0.5,\
        label=label1)
    if showMean:
        h3 = ax.plot(deltaRange+1.0,biasFormHighMean,linestyle=':',\
            color=colorHigh,label=label2)
    h4 = ax.fill_between(deltaRange+1.0,\
        biasFormHighMean + sigmaBiasFormHigh,\
        biasFormHighMean - sigmaBiasFormHigh,color=colorHigh,alpha=0.5,\
        label=label2)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel(xlabel,fontsize=fontsize,fontfamily=fontfamily)
    ax.set_ylabel(ylabel,fontsize=fontsize,fontfamily=fontfamily)
    ax.tick_params(axis='both',labelsize=fontsize)
    ax.set_ylim(ylim)
    if showLegend:
        plt.legend()
    if savename is not None:
        plt.savefig(savename)
    if show:
        plt.show()
    if returnHandles:
        if showMean:
            return [h1,h2,h3,h4]
        else:
            return [h2,h4]

def biasFunctionalForm(delta,b,rho,eps,N=1,S=1,A=1,numericalOffset = 1e-6):
    prefactor = S*N*A
    logresult = np.log(prefactor) + \
        b*np.log(1.0 + delta + numericalOffset) - \
        ((1.0 + delta + numericalOffset)/rho)**(-eps)
    return np.exp(logresult)


nsamples = len(snapNumList)
rhoMin = 1e-5
rhoMax = 1000
rhoBins = 101
deltaRange = 10**np.linspace(np.log10(rhoMin),np.log10(rhoMax),rhoBins) - 1
MbinToPlot = 7
MabsList = np.linspace(-21,-25,9)

biasFormLow = np.zeros((nsamples,len(deltaRange),8))
biasFormHigh = np.zeros((nsamples,len(deltaRange),8))


for m in range(0,8):
    for k in range(0,nsamples):
        biasFormLow[k,:,m] = biasFunctionalForm(deltaRange,\
            biasParam[k][0,2*m,1],\
            biasParam[k][0,2*m,3],\
            biasParam[k][0,2*m,2])
        biasFormHigh[k,:,m] = biasFunctionalForm(deltaRange,\
            biasParam[k][0,2*m+1,1],\
            biasParam[k][0,2*m+1,3],\
            biasParam[k][0,2*m+1,2])

# Mean and standard error:
biasFormLowMean = np.mean(biasFormLow,0)
sigmaBiasFormLow = np.std(biasFormLow,0)/np.sqrt(nsamples)
biasFormHighMean = np.mean(biasFormHigh,0)
sigmaBiasFormHigh = np.std(biasFormHigh,0)/np.sqrt(nsamples)



nCols = 4
nRows = 2
ylabelRow = 1
xlabelCol = 1
ylabel = '$f(\\delta,b,\\rho,\\epsilon)$'
xlabel='$\\rho = 1 + \\delta$'
ylim = [1e-9,1e3]
fontfamily='serif'
fontsize=8
titleSize=10
useTitles = False
title = "Functional form of bias functions for all magnitude bins."
fig, ax = plt.subplots(2,4,figsize=(textwidth,0.5*textwidth))
for m in range(0,8):
    i = int(m/nCols)
    j = m - nCols*i
    [h2,h4] = plotBiasForm(deltaRange,biasFormLowMean[:,m],\
        biasFormHighMean[:,m],sigmaBiasFormLow[:,m],\
        sigmaBiasFormHigh[:,m],ax=ax[i,j],ylabel=ylabel,\
        xlabel=xlabel,ylim=ylim,showLegend=False,returnHandles=True,\
        colorLow=seabornColormap[0],colorHigh=seabornColormap[1],\
        showMean=False,fontsize=fontsize)
    ax[i,j].set_xlim([1e-2,1e2])
    plot.formatPlotGrid(ax,i,j,ylabelRow,ylabel,xlabelCol,xlabel,nRows,ylim,\
        fontsize=fontsize)
    ax[i,j].set_xticks([1e-2,1,1e2],fontsize=fontsize)
    ax[i,j].set_yticks([1e-9,1e-6,1e-3,1,1000],fontsize=fontsize)
    ax[i,j].tick_params(axis='both', which='major', labelsize=fontsize)
    ax[i,j].tick_params(axis='both', which='minor', labelsize=fontsize)
    if i < nRows -1:
        ax[i,j].get_yticklabels()[0].set_visible(False)
        #plt.setp(ax[i,j].get_yticklabels()[-1], visible=False)
    if j < nCols -1:
        ax[i,j].get_xticklabels()[-1].set_visible(False)
        #plt.setp(ax[i,j].get_xticklabels()[-1], visible=False)
    if useTitles:
        ax[i,j].set_title("$" + str(MabsList[m]) + " \\leq M < " + \
            str(MabsList[m+1]) + "$",fontsize=fontsize,fontfamily=fontfamily)

ax[1,3].legend(handles = [h2,h4],\
    prop={"size":fontsize,"family":fontfamily},frameon=False,\
    loc="upper left")
#fig.suptitle(title, fontsize=titleSize,fontfamily=fontfamily)
plt.subplots_adjust(top=0.970,bottom=0.155,left=0.1,right=0.97,\
    hspace=0.0,wspace=0.0)
plt.savefig(figuresFolder + "bias_functional_form.pdf")
plt.show()


# Cluster catalogue statistics:
totalGalaxies = np.sum(galaxyNumberCountExp[-1],1)
galaxyFractions = galaxyNumberCountExp[-1]/totalGalaxies[:,None]

# Bias model variation at each cluster:
nRows = 3
nCols = 3
rBinCentres = plot.binCentres(rBins)
Om0 = 0.3111
rhoM = Om0*2.7754e11
binVolumes = 4*np.pi*rBins[1:]**3/3
nMagBins = 16
mAbs = 3
mApp = 0
for mAbs in range(0,8):
    fig, ax = plt.subplots(nRows,nCols,figsize=(textwidth,0.7*textwidth))
    for l in range(0,nRows*nCols):
        i = int(l/nCols)
        j = l - nCols*i
        if nCols == 1 and nRows == 1:
            axij = ax
        else:
            axij = ax[i,j]
        meanProfile = np.mean(posteriorMassAll[:,l,:]/\
            (binVolumes[:,None]*rhoM),1)
        stdProfile = np.std(posteriorMassAll[:,l,:]/\
            (binVolumes[:,None]*rhoM),1)
        # Bias function calculations:
        biasForm = np.zeros((len(snapNumList),len(rBinCentres),nMagBbins))
        for m in range(0,nMagBins):
            for ns in range(0,len(snapNumList)):
                biasForm[ns,:,m] = biasFunctionalForm(meanProfile - 1.0,\
                    biasParam[ns][0,m,1],biasParam[ns][0,m,3],\
                    biasParam[ns][0,m,2])
        logbiasFormMean = np.mean(np.log(biasForm),0)
        logbiasFormStd = np.std(np.log(biasForm),0)
        h1 = axij.fill_between(rBinCentres,\
            np.exp(logbiasFormMean[:,2*mAbs] - logbiasFormStd[:,2*mAbs]),\
            np.exp(logbiasFormMean[:,2*mAbs] + logbiasFormStd[:,2*mAbs]),\
            color=seabornColormap[0],alpha=0.5,label='$m \\leq 11.5$')
        h11 = axij.plot(rBinCentres,biasForm[:,:,2*mAbs].T,linestyle=':',\
            color=seabornColormap[0],label="Individual \nSamples (Bright)")
        h2 = axij.fill_between(rBinCentres,\
            np.exp(logbiasFormMean[:,2*mAbs+1] - logbiasFormStd[:,2*mAbs+1]),\
            np.exp(logbiasFormMean[:,2*mAbs+1] + logbiasFormStd[:,2*mAbs+1]),\
            color=seabornColormap[1],alpha=0.5,label='$11.5 < m \\leq 12.5$')
        h21 = axij.plot(rBinCentres,biasForm[:,:,2*mAbs+1].T,linestyle=':',\
            color=seabornColormap[1],label="Individual \nSamples (Dim)")
        axij.set_xlabel('$r [\\mathrm{Mpc}h^{-1}]$')
        axij.set_ylabel('Bias functional form')
        axij.set_yscale('log')
        plot.formatPlotGrid(ax,i,j,1,'Bias functional form',1,\
            '$r [\\mathrm{Mpc}h^{-1}]$',nRows,[1e-5,1],nCols = nCols,\
            fontsize=8,xlim=[0,20])
        axij.tick_params(axis='both', which='major', labelsize=fontsize)
        axij.tick_params(axis='both', which='minor', labelsize=fontsize)
        if i < nRows - 1:
            ax[i,j].xaxis.label.set_visible(False)
            ax[i,j].xaxis.set_major_formatter(NullFormatter())
            ax[i,j].xaxis.set_minor_formatter(NullFormatter())
        if i < nRows -1:
            ax[i,j].get_yticklabels()[0].set_visible(False)
        if j < nCols -1:
            ax[i,j].get_xticklabels()[-1].set_visible(False)
        axij.set_title(clusterNames[l][0] + " ($N_{\\mathrm{gal}} = " + \
            str(totalGalaxies[l]) + \
            "$)",fontsize=8)
        # Galaxy fractions:
        axij.text(10,0.3,"Bright: " + \
            ("%.2g" % (100*galaxyFractions[l,2*mAbs])) + "%",fontsize=8)
        axij.text(10,0.07,"Dim: " + \
            ("%.2g" % (100*galaxyFractions[l,2*mAbs+1])) + "%",fontsize=8)
    plt.suptitle("Cluster Bias functional forms ($" + str(MabsList[mAbs]) + \
        " < M_K < " + str(MabsList[mAbs+1]) + "$)",fontsize=12)
    ax[2,1].legend(handles=[h1,h11[0]],\
        prop={"size":fontsize,"family":fontfamily},frameon=False,\
        loc="lower right")
    ax[2,2].legend(handles=[h2,h21[0]],\
        prop={"size":fontsize,"family":fontfamily},frameon=False,\
        loc="lower right")
    plt.subplots_adjust(wspace=0.0)
    plt.savefig(figuresFolder + "cluster_bias_plots_" + str(mAbs) + "_.pdf")
    plt.show()




#-------------------------------------------------------------------------------
# ANTIHALO SKY PLOT:



snapList =  [pynbody.load(samplesFolder + "sample" + str(snapNum) + "/" \
        + "gadget_full_forward_512/snapshot_001") for snapNum in snapNumList]
snapListRev =  [pynbody.load(samplesFolder + "sample" + str(snapNum) + "/" \
        + "gadget_full_reverse_512/snapshot_001") for snapNum in snapNumList]

hrList = [snap.halos() for snap in snapListRev]

for snap in snapList:
    tools.remapBORGSimulation(snap,swapXZ=False,reverse=True)


doClusterMasses = True
if doClusterMasses:
    [meanMasses,meanCentres,sigmaMasses,sigmaCentres,\
            clusterMasses,clusterCentres,clusterCounterparts] = \
                tools.loadOrRecompute(data_folder + "mean_cluster_masses.p",\
                    getBORGClusterMassEstimates,snapNameList,clusterLoc,\
                    equatorialXYZ,_recomputeData=recomputeData)


diff = clusterCentres - meanCentres
dist = np.sqrt(np.sum(diff**2,2))
meanDist = np.mean(dist,0)
stdDist = np.std(dist,0)


# Posterior centres:
posteriorCentres = []
for ns in range(0,len(snapNumList)):
    posteriorCentres.append(tools.loadPickle(snapNameList[ns] + ".clusters1"))

allCentresPost = snapedit.unwrap(np.array(posteriorCentres) + boxsize/2,boxsize)
meanCentresPost = np.mean(allCentresPost,0)
diffPost = allCentres - meanCentresPost
distPost = np.sqrt(np.sum(diffPost**2,2))
meanDistPost = np.mean(distPost,0)
stdDistPost = np.std(distPost,0)


# Plots showing the distribution of masses:


nCols = 3
nRows = 3
logMlow = 13
logMhigh = 15.2
nBins = 11
fontsize=8
textwidth=7.1014
ylim=[0,10]
fig, ax = plt.subplots(3,3,figsize=(textwidth,textwidth))
plt.subplots_adjust(hspace=0.16,wspace=0.0)
for l in range(0,nRows*nCols):
    i = int(l/nCols)
    j = l - nCols*i
    localHigh = np.log10(np.max(clusterMasses[:,l]))
    localLow = np.log10(np.min(clusterMasses[:,l]))
    ax[i,j].hist(clusterMasses[:,l],\
        bins=10**(np.linspace(logMlow,logMhigh,nBins)),alpha=0.5,\
        color=seabornColormap[0])
    plot.formatPlotGrid(ax,i,j,1,"Number of samples",1,\
        "Mass [$M_{\\odot}h^{-1}$]",nRows,ylim,nCols = nCols,\
        fontsize=fontsize,xlim=[10**logMlow,10**logMhigh])
    ax[i,j].axvline(meanMasses[l],linestyle='--',color='grey',\
        label="Mean: $" + \
        ("%.2g" % (meanMasses[l]/1e15)) + "\\times10^{15}M_{\\odot}h^{-1}$" + \
        "\nSt. dev.: $" + \
        ("%.2g" % (stdMasses[l]/1e15)) + "\\times10^{15}M_{\\odot}h^{-1}$")
    ax[i,j].set_xscale('log')
    ax[i,j].set_title(clusterNames[l][0],fontsize=fontsize)
    ax[i,j].set_xlim([10**logMlow,10**logMhigh])
    ax[i,j].set_xticks([1e13,1e14,1e15])
    if i < nRows - 1:
        ax[i,j].xaxis.label.set_visible(False)
        ax[i,j].xaxis.set_major_formatter(NullFormatter())
        ax[i,j].xaxis.set_minor_formatter(NullFormatter())
    if i < nRows -1:
        ax[i,j].get_yticklabels()[0].set_visible(False)
    if j < nCols -1:
        ax[i,j].get_xticklabels()[-1].set_visible(False)
    ax[i,j].legend(prop={"size":fontsize,"family":"serif"},frameon=False,\
        loc="upper right")

plt.suptitle("Mass distribution across 20 MCMC resimulations")
plt.savefig(figuresFolder + "mass_distribution_plot.pdf")
plt.show()


# Distance distribution:
nCols = 3
nRows = 3
distLow = 0
distHigh = 15
nBins = 7
fontsize=8
textwidth=7.1014
ylim=[0,20]
fig, ax = plt.subplots(3,3,figsize=(textwidth,textwidth))
for l in range(0,nRows*nCols):
    i = int(l/nCols)
    j = l - nCols*i
    localHigh = np.max(dist[:,l])
    localLow = np.min(dist[:,l])
    ax[i,j].hist(dist[:,l],\
        bins=np.linspace(distLow,distHigh,nBins),alpha=0.5,\
        color=seabornColormap[0])
    plot.formatPlotGrid(ax,i,j,1,"Number of samples",1,\
        "Distance from mean centre [$\\mathrm{Mpc}h^{-1}$]",nRows,ylim,\
        nCols = nCols,fontsize=fontsize,xlim=[distLow,distHigh])
    ax[i,j].axvline(meanDist[l],linestyle='--',color='grey',\
        label="Mean: $" + ("%.2g" % meanDist[l]) + \
        "\\mathrm{Mpc}h^{-1}$\nSt. dev.: $" + ("%.2g" % stdDist[l]) + \
        "\\mathrm{Mpc}h^{-1}$")
    ax[i,j].set_title(clusterNames[l][0],fontsize=fontsize)
    if i < nRows - 1:
        ax[i,j].xaxis.label.set_visible(False)
        ax[i,j].xaxis.set_major_formatter(NullFormatter())
        ax[i,j].xaxis.set_minor_formatter(NullFormatter())
    if i < nRows -1:
        ax[i,j].get_yticklabels()[0].set_visible(False)
    if j < nCols -1:
        ax[i,j].get_xticklabels()[-1].set_visible(False)
    ax[i,j].legend(prop={"size":fontsize,"family":"serif"},frameon=False,\
        loc="upper right")

plt.subplots_adjust(hspace=0.16,wspace=0.0)
plt.suptitle("Displacement from mean centre across 20 MCMC resimulations")

plt.savefig(figuresFolder + "displacement_distribution_plot.pdf")
plt.show()








sortedRadiiOpt = np.flip(np.argsort(catData['radii'][combinedFilter135]))
catToUse = finalCatOpt[combinedFilter135]
haveVoids = [np.where(catToUse[:,ns] > 0)[0] \
    for ns in range(0,len(snapNameList))]

for ns in range(0,len(snapNameList)):
    catToUse[haveVoids[ns],ns] = np.arange(1,len(haveVoids[ns])+1)



filterToUse = np.where(combinedFilter135)[0]
nVoidsToShow = 10
#selection = np.intersect1d(sortedRadiiOpt,filterToUse)[:(nVoidsToShow)]
selection = sortedRadiiOpt[np.arange(0,nVoidsToShow)]
asListAll = []
colourListAll = []
laListAll = []
labelListAll = []

plotFormat='.pdf'
#plotFormat='.pdf'

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
        fullList = np.array(centralAntihalos[ns][0])[sortedList[ns]]
        for k in range(0,np.min([nVoidsToShow,len(selection)])):
            if catToUse[selection[k],ns] > 0:
                listPosition = finalCatOpt[combinedFilter135][selection[k],ns]-1
                ahNumber = fullList[listPosition]
                if listPosition >= len(largeAntihalos_all[ns]):
                    print("Alpha shape not computed for void " + str(k+1) + \
                        "! " + "Computing on the fly for anti-halo " + \
                        str(listPosition) + " (number " + str(ahNumber) + ")")
                    [ahMWPos,alpha_shapes] = tools.computeMollweideAlphaShapes(\
                        snapList[ns],np.array([ahNumber]),\
                        hrList[ns])
                    asList.append(alpha_shapes[0])
                else:
                    asList.append(alpha_shape_list_all[ns][1][listPosition])
                laList.append(ahNumber)
                colourList.append(seabornColormap[np.mod(k,len(seabornColormap))])
                labelList.append(str(k+1))
            print("Done for void " + str(k+1))
        print("Done for sample " + str(ns+1))
        asListAll.append(asList)
        colourListAll.append(colourList)
        laListAll.append(laList)
        labelListAll.append(labelList)
    plt.clf()
    for ns in range(0,len(snapNumList)):
        plot.plotLocalUniverseMollweide(rCut,snapList[ns],\
            alpha_shapes = asListAll[ns],\
            largeAntihalos = laListAll[ns],hr=hrList[ns],\
            coordAbell = coordCombinedAbellSphere,\
            abellListLocation = clusterIndMain,\
            nameListLargeClusters = [name[0] for name in clusterNames],\
            ha = ha,va= va, annotationPos = annotationPos,\
            title = 'Local super-volume: large voids (antihalos) within $' + \
            str(rCut) + "\\mathrm{\\,Mpc}h^{-1}$",\
            vmin=1e-2,vmax=1e2,legLoc = 'lower left',bbox_to_anchor = (-0.1,-0.2),\
            snapsort = snapsortList_all[ns],antihaloCentres = None,\
            figOut = figuresFolder + "/ah_match_sample_" + \
            str(ns) + plotFormat,\
            showFig=False,figsize = (scale*textwidth,scale*0.55*textwidth),\
            voidColour = colourListAll[ns],antiHaloLabel=labelListAll[ns],\
            bbox_inches = bound_box,galaxyAngles=equatorialRThetaPhi[:,1:],\
            galaxyDistances = equatorialRThetaPhi[:,0],showGalaxies=False,\
            voidAlpha = 0.6)
    plt.show()


# Simple plot of the galaxy distribution overlaid:
ns = 0
plot.plotLocalUniverseMollweide(100,snapList[ns],\
            alpha_shapes = None,\
            largeAntihalos = None,hr=None,\
            coordAbell = coordCombinedAbellSphere,\
            abellListLocation = clusterIndMain,\
            nameListLargeClusters = [name[0] for name in clusterNames],\
            ha = ha,va= va, annotationPos = annotationPos,\
            vmin=1e-2,vmax=1e2,legLoc = 'lower left',bbox_to_anchor = (-0.1,-0.2),\
            snapsort = snapsortList_all[ns],antihaloCentres = None,\
            figOut = figuresFolder + "/mollweide_galaxies_" + \
            str(ns) + plotFormat,\
            showFig=False,figsize = (scale*textwidth,scale*0.55*textwidth),\
            voidColour = colourListAll[ns],antiHaloLabel=labelListAll[ns],\
            bbox_inches = bound_box,galaxyAngles=equatorialRThetaPhi[:,1:],\
            galaxyDistances = equatorialRThetaPhi[:,0],showGalaxies=True,\
            voidAlpha = 0.6,labelFontSize=12,legendFontSize=8,title="",dpi=600)

plt.show()

#-------------------------------------------------------------------------------
# CLUSTER MASS PLOT

# Mass comparison plot!
if doClusterMasses:
    massconstraintsplot.showClusterMassConstraints(meanMasses,sigmaMasses,\
            figOut = figuresFolder,catFolder = "./catalogues/",h=h,Om0 = Om0,\
            savename = figuresFolder + "mass_constraints_plot.pdf")

#-------------------------------------------------------------------------------
# MASS FUNCTIONS PLOT 135 VS 300
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
    plot.massFunctionComparison(massListMean[combinedFilter135],\
        massListMean[combinedFilter],volSphere135,nBins=nBins,\
        labelLeft = "Combined catalogue",\
        labelRight  ="Combined catalogue",\
        ylabel="Number of antihalos",savename=figuresFolder + \
        "mass_function_combined_300vs135.pdf",massLower=mLower,\
        ylim=[1,1000],Om0 = 0.3111,h=0.6766,sigma8=0.8128,ns=0.9667,\
        fontsize=8,massUpper = mUpper,\
        titleLeft = "Combined catalogue, $<135\\mathrm{Mpc}h^{-1}$",\
        titleRight = "Combined catalogue, $<300\\mathrm{Mpc}h^{-1}$",\
        volSimRight = volSphere,ylimRight=[1,1000],legendLoc="upper right")

#-------------------------------------------------------------------------------
# UNDERDENSE PROFILES PLOT


# Plot for the paper, using the new method:
doProfiles = False
if doCat and doProfiles:
    textwidth=7.1014
    fontsize = 8
    legendFontsize=8
    fig, ax = plt.subplots(1,1,figsize=(0.5*textwidth,0.5*textwidth))
    plot.plotVoidProfilesPaper(rBinStackCentres,nbarMean,sigmaMean,nbar,\
        stacking.weightedMean(nbarjUnSameRadii,1.0/sigmaUnSameRadii**2,axis=0),\
        np.sqrt(stacking.weightedVariance(nbarjUnSameRadii,\
        1.0/sigmaUnSameRadii**2,axis=0)/nbarjUnSameRadii.shape[0]),\
        np.sqrt(stacking.weightedVariance(nbarjUnSameRadii,\
        1.0/sigmaUnSameRadii**2,axis=0)),ax=ax,show=False,\
        title="Combined BORG Catalogue",legendFontSize=legendFontsize,\
        legendLoc='lower right',xlim=[0,3])
    #ax[0].axhline(0.95,color='grey',linestyle=':')
    #ax[1].axhline(0.95,color='grey',linestyle=':')
    plt.tight_layout()
    plt.savefig(figuresFolder + "profiles_plot_vs_underdense.pdf")
    plt.show()

#-------------------------------------------------------------------------------
# DENSITY SLICE PLOT

from void_analysis.plot import plotDensitySlice, plotDensityComparison

import glob
from PIL import Image


cl = 0 # Cluster to plot
sm = 0 # sample to plot

densitiesHR = [np.fromfile("new_chain/sample" + str(snap) + \
    "/gadget_full_forward_512/snapshot_001.a_den",\
    dtype=np.float32) for snap in snapNumList]
densities256 = [np.reshape(density,(256,256,256),order='C') \
    for density in densitiesHR]
densities256F = [np.reshape(density,(256,256,256),order='F') \
    for density in densitiesHR]


# Load data from MCMC chains:
N = 256
newChain = [h5py.File("new_chain/sample" + str(k) + "/mcmc_" + str(k) + \
    ".h5",'r') for k in snapNumList]
mcmcDen = [1.0 + sample['scalars']['BORG_final_density'][()] \
    for sample in newChain]
mcmcDenLin = [np.reshape(den,N**3) for den in mcmcDen]
mcmcDen_r = [np.reshape(den,(256,256,256),order='F') for den in mcmcDenLin]
mcmcDenLin_r = [np.reshape(den,256**3) for den in mcmcDen_r]

biasMCMC = [np.array([[sample['scalars']['galaxy_bias_' + str(k)][()] \
    for k in range(0,16)]]) for sample in  newChain]

nmeansMCMC = [np.array([[sample['scalars']['galaxy_nmean_' + str(k)][()] \
    for k in range(0,16)]]) for sample in newChain]




# Density slice comparison:
for cl in range(0,9):
    for ns in range(0,len(snapNumList)):
        plt.clf()
        ax = plotDensityComparison(np.flip(densities256[ns]),mcmcDen_r[ns],\
            N=256,centre1=clusterLoc[cl,:],centre2=clusterLoc[cl,:],\
            width = 50,thickness=20,\
            textLeft = "Resimulation Density (real space)",\
            textRight="Posterior Density (z-space)",\
            title="Sample " + str(ns+1) + ", Density Field around " + \
            clusterNames[cl][0],vmax = 1000,vmin=1/1000,\
            markCentre=True,losAxis=1,showGalaxies=False,flipCentreLeft=False,\
            flipCentreRight=False,flipLeft=True,flipRight=False,\
            invertAxisLeft=False,invertAxisRight=False,\
            flipudLeft=False,flipudRight=False,fliplrLeft=False,\
            fliplrRight=False,swapXZLeft=True,swapXZRight=False,\
            gal_position=equatorialXYZ,returnAx=True,show=False)
        ax[0].scatter(clusterCentres[ns][cl,0],clusterCentres[ns][cl,2],\
            marker='x',color='k')
        ax[1].scatter(allCentresPost[ns][cl,0],allCentresPost[ns][cl,2],\
            marker='x',color='k')
        plt.savefig(figuresFolder + "cluster_" + str(cl) + "_sample_" + \
            str(ns) + ".png")
    frames = []
    #imgs = glob.glob(figuresFolder + "cluster_" + str(cl) + "*.png")
    imgs = [figuresFolder + "cluster_" + str(cl) + "_sample_" + str(ns) + \
        ".png" for ns in range(0,len(snapNumList))]
    for i in imgs:
        new_frame = Image.open(i)
        frames.append(new_frame)
    # Save into a GIF file that loops forever
    frames[0].save(figuresFolder + 'clusters_plot_' + str(cl) + '.gif', \
        format='GIF', append_images=frames[1:],save_all=True,duration=1000,\
        loop=0)










#-------------------------------------------------------------------------------
# MASS CONVERGENCE PLOT:

stepsList = [2,3,5,10,20,30,32,64,128]
stepsListGADGET = [16,32,64,128]
stepsList1024 = [16,32,64,128]
stepsListEPS_0p662 = [32,64]
epsList = [0.662]
resList = [256,512,1024,2048]
resStepsList = [32]
logstepsList = [3,5,10,20,30,32,64,128]
sampleList = ["sample7422","sample7500","sample8000",\
    "sample8500","sample9000","sample9500"]

# Comparison between two different types of mass:
clusterFilter = np.array([2,4,6],dtype=int) # Coma, Shapley, Hercules A
doCon=True


if doCon:
    plot.plotMassTypeComparison(np.array(massList200c)[:,:,clusterFilter],\
        np.array(massListFull200c)[:,clusterFilter],\
        np.array(massList100m)[:,:,clusterFilter],\
        np.array(massListFull100m)[:,clusterFilter],\
        stepsListGADGET,stepsList,logstepsList,stepsList1024,\
        stepsListEPS_0p662,resStepsList,clusterNames[clusterFilter,:],\
        name1 = "$M_{200\\mathrm{c}}$",name2 = "$M_{100\\mathrm{m}}$",\
        show=True,save = True,colorLinear = seabornColormap[5],\
        colorLog=seabornColormap[2],colorGadget='k',colorAdaptive='grey',\
        showGadgetAdaptive = True,\
        savename = figuresFolder + "mass_convergence_comparison.pdf",\
        massName = "M",extraMasses = None,extraMassLabel = 'Extra mass scale',\
        xlabel='Number of Steps',\
        returnHandles=False,showLegend=True,nCols=3,showGADGET=False,\
        figsize=(textwidth,0.55*textwidth),showResMasses=False,logy=False,\
        ylim1=[0,2e15],ylim2=[7e14,3.2e15],top=0.931,bottom=0.114,left=0.08,\
        right=0.92,hspace=0.0,wspace=0.0,yticks1=[5e14,1e15,1.5e15,2e15],\
        yticks2=[1e15,2e15,3e15],massLabelPos=0.97,\
        colaColour=seabornColormap[5],pmColour=seabornColormap[2],\
        logStyle=':',linStyle='-',legendMethod="grid",\
        linText=[0.1,0.8],logText=[0.1,0.7],colaText=[0.5,0.9],\
        pmText=[0.75,0.9],legLoc=[0.5,0.65],secondYAxis=True,xlim=[3,256],\
        selectMarker='o',selectSize=80,selectModels=True)



if doCon:
    #clusterFilter2 = np.array([1,2,3],dtype=int)
    clusterFilter2 = np.array([1],dtype=int)
    plot.plotMassTypeComparison(np.array(massList200c2)[:,:,clusterFilter2],\
        np.array(massListFull200c2)[:,clusterFilter2],\
        np.array(massList100m2)[:,:,clusterFilter2],\
        np.array(massListFull100m2)[:,clusterFilter2],\
        stepsListGADGET,stepsList,logstepsList,stepsList1024,\
        stepsListEPS_0p662,resStepsList,None,\
        name1 = "$M_{200\\mathrm{c}}$",name2 = "$M_{100\\mathrm{m}}$",\
        show=True,save = True,colorLinear = seabornColormap[5],\
        colorLog=seabornColormap[2],colorGadget='k',colorAdaptive='grey',\
        showGadgetAdaptive = True,\
        savename = figuresFolder + "mass_convergence_comparison_other.pdf",\
        massName = "M",extraMasses = None,extraMassLabel = 'Extra mass scale',\
        xlabel='Number of Steps',\
        returnHandles=False,showLegend=True,nCols=1,showGADGET=False,\
        figsize=(0.45*textwidth,0.55*textwidth),showResMasses=False,\
        ylim1=[0,2.2e14],ylim2=[0,5.2e14],logy=False,\
        yticks1 = [0,5e13,1e14,1.5e14,2e14],yticks2 = [0,1e14,2e14,3e14,4e14],\
        top=0.97,bottom=0.11,left=0.19,right=0.93,hspace=0.0,wspace=0.0,\
        massLabelPos=0.96,\
        colaColour=seabornColormap[5],pmColour=seabornColormap[2],\
        logStyle=':',linStyle='-',legendMethod="grid",\
        linText=[0.1,0.8],logText=[0.1,0.7],colaText=[0.5,0.9],\
        pmText=[0.75,0.9],legLoc=[0.5,0.65],secondYAxis=False,xlim=[3,256],\
        selectMarker='o',selectSize=80,selectModels=True)


#-------------------------------------------------------------------------------
# COMA PROFILES TEST


def tick_function(X):
    V = 4*np.pi*X**3/(3*1e4)
    return ["%.2f" % z for z in V]



def compareDensityProfile(radii,mProf,mProfError,mProfPost,mProfPostError,\
        constraintList=None,refList = None,radiiPost = None,\
        textwidth=7.1014,textheight=9.0971,widthFactor = 0.87,heightFactor=1,\
        yscale = 1e16,color1='grey',alpha1=0.75,alpha2=0.25,ylim=[1e-2,3],\
        xlim=[0,30],showMean=True,savename = None,ax=None,show=True,\
        returnAx = False,meanColour='tab:blue',label='Gadget density',\
        label2 = 'Cola density',title=None,omegaM=0.3111,\
        color2 = seabornColormap[3],fontsize=10,fontname='serif',\
        logy=True,logx=True,showVolAx = False,legend=True,\
        meanLabel='Mean Universe Density'):
    if radiiPost is None:
        radiiPost = radii
    colorList = ['grey','y','tab:orange','k','c','r','b',seabornColormap[0],\
        seabornColormap[1],seabornColormap[4]]
    fig, ax = plt.subplots(figsize=(heightFactor*textwidth,widthFactor*textwidth))
    plt.fill_between(radii,(mProf - mProfError)/yscale,\
        (mProf + mProfError)/yscale,\
        color=color1,alpha=alpha1)
    plt.fill_between(radii,(mProf - 2*mProfError)/yscale,\
        (mProf + 2*mProfError)/yscale,color=color1,alpha=alpha2)
    plt.semilogy(radii,mProf/yscale,label=label,color=color1)
    plt.fill_between(radiiPost,(mProfPost - mProfErrorPost)/yscale,\
        (mProfPost + mProfErrorPost)/yscale,color=color2,alpha=alpha2)
    plt.fill_between(radiiPost,(mProfPost - 2*mProfErrorPost)/yscale,\
        (mProfPost + 2*mProfErrorPost)/yscale,color=color2,alpha=alpha1)
    plt.semilogy(radiiPost,mProfPost/yscale,label=label2,color=color2)
    if showMean:
        rhom = 2.7754e11*omegaM 
        plt.semilogy(radii,rhom*4*np.pi*radii**3/(3*yscale),\
            label=meanLabel,color=meanColour)
    # Add mass estimates to the plot:
    if constraintList is not None:
        if refList is None:
            raise Exception("refList must be supplied.")
        for k in range(0,len(constraintList)):
            Y = np.array([estimate.mass \
                for estimate in constraintList[k]])/(yscale)
            Yerr = np.vstack((np.array([estimate.massLow \
                for estimate in constraintList[k]]),\
                np.array([estimate.massHigh \
                for estimate in constraintList[k]])))/yscale
            X = np.array([estimate.virial for estimate in constraintList[k]])
            Xerr = np.vstack((np.array([estimate.virialLow \
                for estimate in constraintList[k]]),\
                np.array([estimate.virialHigh \
                for estimate in constraintList[k]])))
            plt.errorbar(X,Y,xerr = Xerr,yerr=Yerr,label=refList[k],\
                marker='x',color=colorList[k],linestyle='')
    plt.ylim(ylim)
    if logy:
        plt.yscale('log')
    if logx:
        plt.xscale('log')
    print(ax.get_xticks())
    plt.xlim(xlim)
    plt.xlabel('$R [\\mathrm{Mpc}h^{-1}]$',fontsize=fontsize,\
        fontfamily=fontname)
    plt.ylabel('$M(<r) [10^{16}M_{\\odot}h^{-1}]$',fontsize=fontsize,\
        fontfamily=fontname)
    if showVolAx:
        ax2 = ax.twiny()
        ax2.set_xlabel('$V [10^4(\\mathrm{\\,Mpc}h^{-3})^3]$',\
            fontsize=fontsize,fontfamily=fontname)
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(ax.get_xticks())
        ax2.set_xticklabels(tick_function(ax.get_xticks()))
    if legend:
        ax.legend(prop={"size":fontsize,"family":"serif"})
    ax.grid()
    ax.tick_params(axis='both',labelsize=fontsize)
    if showVolAx:
        ax2.tick_params(axis='both',labelsize=fontsize)
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    if savename is not None:
        plt.savefig(savename)
    if show:
        plt.show()
    if returnAx:
        return ax

# Get mass at different radii:
radii = np.linspace(0,30,31)
dataFile = "coma_profile_data.p"
comaData2 = "coma_profile_data2.p"
[mPart,mProf,mProfError,mPartPost,\
    mProfPost,mProfErrorPost,counts,countsPost] = pickle.load(open(dataFile,"rb"))
[mPart2,mProf2,mProfError2,mPartPost2,\
        mProfPost2,mProfErrorPost2,counts2,countsPost2] = pickle.load(\
        open(comaData2,"rb"))


[mPart,mProfGad,mProfGadError,\
    mProfBorg,mProfBorgError,countsGad,countsBorg] = \
    getClusterProfilesFromSnaps(snapNumList,snapPath,clusterLoc,\
        useGadCentre = True)
from void_analysis.tools import MassConstraint
h = 0.705
geller1999 = [MassConstraint(1.65e15,0.41e15,0.41e15,10,0,0,method='X-Ray'),\
    MassConstraint(1.44e15,0.29e15,0.29e15,5.5,0,0,method='X-Ray')]
kubo2007 = [MassConstraint(1.88e15*h/0.7,0.56e15*h/0.7,0.65e15*h/0.7,1.99*h/0.7,0,0,method='Weak Lensing')]
hughes1989 = [MassConstraint(3e14*h/0.5,0.4e14*h/0.5,0.4e14*h/0.5,0.5*h/0.5,0,0,method='Dynamical'),\
    MassConstraint(0.95e15*h/0.5,0.15e15*h/0.5,0.15e15*h/0.5,2.5*h/0.5,0,0,method='Dynamical')]
theWhite1986 = [MassConstraint(1.9e15,0.3*1.9e15,0.3*1.9e15,5.4,0,0,method='Dynamical')]
colless2006 = [MassConstraint(3.1e14,0.5e14,0.5e14,1,0,0,method='Dynamical'),\
    MassConstraint(6.5e14,2.5e14,2.5e14,3,0,0,method='Dynamical')]
#gavazzi2009 = [MassConstraint(5.1e14,2.1e14,4.3e14,1.8,0.3,0.6,method='Weak Lensing'),\
#    MassConstraint(9.7e14,3.5e14,6.1e14,2.2,0.2,0.3,method='Weak Lensing'),\
#    MassConstraint(6.1e14,3.5e14,12.1e14,2.5,0.5,0.8,method='Weak Lensing'),
#    MassConstraint(11.1e14,6.1e14,16.7e14,3.6,0.7,1.1,method='Weak Lensing')]
gavazzi2009 = [MassConstraint(5.1e14,2.1e14,4.3e14,1.8,0.3,0.6,method='Weak Lensing'),\
    MassConstraint(9.7e14,3.5e14,6.1e14,2.2,0.2,0.3,method='Weak Lensing')]
falco2014 = [MassConstraint(9.7e14*h,3.6e14*h,3.6e14*h,2.5*h,0,0,'Dynamical')]

constraintList = [geller1999,kubo2007,hughes1989,theWhite1986,colless2006,gavazzi2009,\
    falco2014]
refList = ['Geller (1999)','Kubo (2007)','Hughes (1989)','The & White (1986)',\
    'Colless (2006)','Gavazzi (2009)','Falco (2014)']

constraintList = [geller1999,kubo2007,hughes1989,theWhite1986,colless2006,gavazzi2009,\
    falco2014]
refList = ['Geller (1999)','Kubo (2007)','Hughes (1989)','The & White (1986)',\
    'Colless (2006)','Gavazzi (2009)','Falco (2014)']

mPiff = np.array([6.1508e14,2.4052e14,4.2846e14,2.1360e14,4.5067e14,1.3823e14,2.9626e14,\
    2.1598e14,2.1398e14],\
    dtype=np.double)
meanErrorFrac_upp = 0.08 # Assume 20% errors, because MCXC don't give any???
meanErrorFrac_low = 0.08
zPiff = np.array([0.0179,0.0353,0.0231,0.0157,0.0391,0.0420,0.0299,0.0355,0.0214],dtype=np.double)
mPiffLow = meanErrorFrac_low*mPiff
mPiffHigh = meanErrorFrac_upp*mPiff
mPiff200c = -np.ones(mPiff.shape)
mPiff200cLow = -np.ones(mPiff.shape)
mPiff200cHigh = -np.ones(mPiff.shape)
Om0 = 0.307
Ol = 1-Om0
D1 = 500
D2 = 200
for k in range(0,len(mPiff)):
    if mPiff[k] > 0:
        # Convert masses to M200c:
        mass200c = cosmology.convertCriticalMass(mPiff[k],D1,D2=D2,\
            Om=Om0,z = zPiff[k],Ol = 1 - Om0,h=h,returnError=True)
        mPiff200c[k] = mass200c[0]
        # Errors on M500c estimate converted to M200c errors:
        errorM500c_upp = cosmology.convertCriticalMass(mPiff[k] + \
            mPiffHigh[k],D1,D2) - mPiff200c[k]
        errorM500c_low = mPiff200c[k] - \
            cosmology.convertCriticalMass(mPiff[k] - mPiffLow[k],\
            D1,D2)
        # Errors from conversion to M200c (due to scatter in assumed concentration-mass
        # relationship):
        errorCon_upp = mass200c[2]
        errorCon_low = mass200c[1]
        # Combine errors in quadrature to estimate the new error:
        mPiff200cLow[k] = np.sqrt(errorCon_low**2 + errorM500c_low**2)
        mPiff200cHigh[k] = np.sqrt(errorCon_upp**2 + errorM500c_upp**2)

rhocrit = 2.7754e11
rPiff200c = -np.ones(mPiff.shape)
rPiff200c[np.where(mPiff200c > 0)] = np.cbrt(3*mPiff200c[np.where(mPiff200c > 0)]/\
    (4*np.pi*200*rhocrit))

Piffaretti2011_200c = [MassConstraint(mPiff200c[2]*h,mPiff200cLow[2]*h,\
    mPiff200cHigh[2]*h,rPiff200c[2]*h,0,0,method='X-Ray')]
Okabe2014 = [MassConstraint(6.23e14,1.58e14,2.53e14,1.75,0,0,method='Weak Lensing')]
Lokas2003 = [MassConstraint(1.4e15,0.3*1.4e15,0.3*1.4e15,2.9,0,0,method='Dynamical')]


constraintList = [geller1999,kubo2007,hughes1989,theWhite1986,colless2006,gavazzi2009,\
    falco2014,Piffaretti2011_200c,Okabe2014,Lokas2003]
refList = ['Geller (1999)','Kubo (2007)','Hughes (1989)','The & White (1986)',\
    'Colless (2006)','Gavazzi (2009)','Falco (2014)',\
    'Piffaretti (2011)','Okabe (2014)','Lokas (2014)']


rhoc = 2.7754e11
rhom = 0.307*rhoc
deltaComaGad = 3*mProf/(4*np.pi*radii[1:]**3*rhoc)
interpComaCritical = scipy.interpolate.interp1d(radii[1:],deltaComaGad)
sol200c = scipy.optimize.brentq(lambda x: interpComaCritical(x) - 200,\
    radii[1],radii[-1])
sol100m = scipy.optimize.brentq(lambda x: interpComaCritical(x) - 100*0.307,\
    radii[1],radii[-1])


fontsize = 8
ax = compareDensityProfile(radii[1:],mProf,mProfError,\
    mProfPost2,mProfErrorPost2,\
    constraintList=None,radiiPost=radii[1:],refList=refList,\
    label = "GADGET",label2="10-step PM",\
    savename=None,widthFactor=0.45,\
    heightFactor=0.45/0.87,fontsize=fontsize,show=False,returnAx=True,\
    meanLabel="Mean Universe \nDensity")

ax.axvline(sol200c,linestyle="--",color='grey',\
    label='$M_{200c}$ radius\n$' + ("%.2g" % sol200c) + \
    "\\,\\mathrm{Mpc}h^{-1}$")
ax.axvline(sol100m,linestyle=":",color='grey',\
    label='$M_{100m}$ radius\n$' + ("%.2g" % sol100m) + \
    "\\,\\mathrm{Mpc}h^{-1}$")
ax.legend(prop={"size":fontsize,"family":"serif"},loc="upper left",\
    frameon=True,ncol=2)
plt.savefig(figuresFolder + "coma_density_plot.pdf")
plt.show()










