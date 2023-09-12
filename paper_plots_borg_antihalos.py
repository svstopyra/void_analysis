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
#muOpt = 0.9
#muOpt = 0.55830868
#rSearchOpt = 1
#rSearchOpt = 1.1565106
rSphere = 300
rSphereInner = 135
# Optimum parameters with 1-way purity/completeness:
# np.array([1.21232077, 0.54490005])
# Optimum with 2-way:
# np.array([1.1565106 , 0.55830868])

# Optimal number of voids in the final catalogue:
# [0.14237877, 0.85891552]
muOpt = 0.85891552
rSearchOpt = 0.14237877

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
            mLower = "auto",mUpper = 2e15,nBinEdges = 8,muOpt = muOpt,\
            rSearchOpt = rSearchOpt,rSphere = rSphere,\
            rSphereInner = rSphereInner,NWayMatch = False,rMin=rMin,rMax=rMax,\
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

[snrFilter,snrAllCatsList] = getSNRFilterFromChainFile(chainFile,snrThresh,\
    snapNameList,boxsize)

chainFile="chain_properties.p"
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


centralAntihalos = [tools.getAntiHalosInSphere(antihaloCentres[k],rSphere,\
            filterCondition = (antihaloRadii[k] > rMin) & \
            (antihaloRadii[k] <= rMax) & (antihaloMasses[k] > mMin) & \
            (antihaloMasses[k] <= mMax) & snrFilter[k]) \
            for k in range(0,len(snapNumList))]
centralAntihaloMasses = [\
            antihaloMasses[k][centralAntihalos[k][0]] \
            for k in range(0,len(centralAntihalos))]
centralAntihaloRadii = [\
            antihaloRadii[k][centralAntihalos[k][0]] \
            for k in range(0,len(centralAntihalos))]
sortedList = [np.flip(np.argsort(centralAntihaloMasses[k])) \
        for k in range(0,len(snapNumList))]



nearestPointsList = [tree.query_ball_point(\
        snapedit.wrap(antihaloCentres[k] + boxsize/2,boxsize),\
        antihaloRadii[k],workers=-1) \
        for k in range(0,len(antihaloCentres))]
snrAllCatsList = [np.array([np.mean(snrFieldLin[points]) \
        for points in nearestPointsList[k]]) for k in range(0,len(snapNumList))]
snrFilter2 = [snr > snrThresh for snr in snrAllCatsList]


ahCounts = np.array([len(cahs[0]) for cahs in centralAntihalos])
max_index = np.max(ahCounts)
centresListShort = [np.array([antihaloCentres[l][\
    centralAntihalos[l][0][sortedList[l][k]],:] \
    for k in range(0,np.min([ahCounts[l],max_index]))]) \
    for l in range(0,len(snapNumList))]
massListShort = [np.array([antihaloMasses[l][\
            centralAntihalos[l][0][sortedList[l][k]]] \
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
            massRange = [mMin,mMax],nToPlot = 400,\
            rRange = [5,30],reCentreSnaps = True,\
            additionalFilters=snrFilter,rSphere=rSphere,\
            centralAntihalos=centralAntihalos)
    # Alpha shapes for the final catalogue alone:
    [ahMWPos,alpha_shapes_finalCat] = tools.loadOrRecompute(figuresFolder + \
            "skyplot_data_final_cat.p",\
            getFinalCatalogueAlphaShapes,snapNumList,\
            finalCatOpt[combinedFilter135],samplesFolder=samplesFolder,\
            _recomputeData = recomputeData,recomputeData = recomputeData,\
            massRange = [mMin,mMax],\
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
# THRESHOLD PLOTS


[scaleBins,percentilesCat,percentilesComb,\
        meanCatFrac,stdErrCatFrac,meanCombFrac,stdErrCombFrac,\
        radiiListMean,massListMean,massListSigma,radiiListSigma,\
        massBins,radBins,scaleFilter] = tools.loadPickle(\
        data_folder + "catalogue_scale_cut_data.p")

[finalCatOpt,shortHaloListOpt,twoWayMatchListOpt,finalCandidatesOpt,\
        finalRatiosOpt,finalDistancesOpt,allCandidatesOpt,candidateCountsOpt,\
        allRatiosOpt,finalCombinatoricFracOpt,finalCatFracOpt,\
        alreadyMatched] = pickle.load(\
            open(data_folder + "catalogue_all_data.p","rb"))

[finalCatUn,shortHaloListUn,twoWayMatchListUn,\
            finalCandidatesOUn,\
            finalRatiosUn,finalDistancesUn,allCandidatesUn,candidateCountsUn,\
            allRatiosUn,finalCombinatoricFracUn,finalCatFracUn,\
            antihaloCentresUn,antihaloMassesUn,antihaloRadiiUn,\
            centralAntihalosUn,centralAntihaloMassesUn,sortedListUn,\
            ahCountsUn,radiiListShortUn,massListShortUn] = pickle.load(open(\
                data_folder + "unconstrained_catalogue.p","rb"))
radiiListCombUn = getPropertyFromCat(finalCatUn,radiiListShortUn)
massListCombUn = getPropertyFromCat(finalCatUn,massListShortUn)
[radiiListMeanUn,radiiListSigmaUn] = getMeanProperty(radiiListCombUn)
[massListMeanUn,massListSigmaUn] = getMeanProperty(massListCombUn)
massFilterUn = [(massListMeanUn > scaleBins[k]) & \
            (massListMeanUn <= scaleBins[k+1]) \
            for k in range(0,len(scaleBins) - 1)]


#meanRadii = np.array([np.mean(radiiListMean[inds]) for inds in scaleFilter])

meanRadii = radiiMean300
radiiBins = plot_utilities.binCentres(radBins)
scaleFilter = [np.where((meanRadii > radBins[k]) & \
    (meanRadii <= radBins[k+1]))[0] for k in range(0,len(radBins)-1)]
[meanRadiiUn,_] = cat300Rand.getMeanProperty("radii")
scaleFilterUn = [np.where((meanRadiiUn > radBins[k]) & \
    (meanRadiiUn <= radBins[k+1]))[0] for k in range(0,len(radBins)-1)]
meanCatFrac = np.array([np.mean(cat300.finalCatFrac[inds]) \
    for inds in scaleFilter])

plt.plot(meanRadii,meanCombFrac,linestyle='-',color=seabornColormap[0])
plt.plot(meanRadii,percentilesComb,linestyle=':',color=seabornColormap[0])
plt.plot(meanRadii,meanCatFrac,linestyle='-',color=seabornColormap[1])
plt.plot(meanRadii,percentilesCat,linestyle=':',color=seabornColormap[1])
plt.xlabel('Radius [$\\mathrm{Mpc}h^{-1}$]')
plt.ylabel('Catalogue or Combinatoric fraction')
line1 = mlines.Line2D([],[],linestyle='-',color=seabornColormap[0],\
    label='Combinatoric Fraction')
line2 = mlines.Line2D([],[],linestyle='-',color=seabornColormap[1],\
    label='Catalogue Fraction')
line3 = mlines.Line2D([],[],linestyle='-',color='k',\
    label='Mean of posterior')
line4 = mlines.Line2D([],[],linestyle=':',color='k',\
    label='Threshold (99th percentile)')
plt.legend(handles=[line1,line2,line3,line4])
plt.savefig(figuresFolder + "catalogue_and_combinatoric_fractions.pdf")
plt.show()

# Violin plot:
binNames = [str(x) for x in range(1,len(scaleFilter))]
#violinData = [finalCombinatoricFracOpt[inds] for inds in scaleFilter]
#violinDataUn = [finalCombinatoricFracUn[inds] for inds in massFilterUn]
violinData = [cat300.finalCatFrac[inds] for inds in scaleFilter]
violinDataUn = [cat300Rand.finalCatFrac[inds] for inds in scaleFilterUn]
# New idea. Arrange things into a data frame with categories 'value', 'simtype'
# and 'radius':
combinedValues = np.hstack([np.hstack([violinData[ind],violinDataUn[ind]]) \
    for ind in range(0,len(violinData))])
combinedLabels = np.hstack([np.hstack([\
    np.full(len(violinData[ind]),'posterior'),\
    np.full(len(violinDataUn[ind]),'random')]) \
    for ind in range(0,len(violinData))])
combinedRadii = np.hstack([np.hstack([\
    np.full(len(violinData[ind]),radiiBins[ind]),\
    np.full(len(violinDataUn[ind]),radiiBins[ind])]) \
    for ind in range(0,len(violinData))])
combinedBins = np.hstack([np.hstack([\
    np.full(len(violinData[ind]),ind),\
    np.full(len(violinDataUn[ind]),ind)]) \
    for ind in range(0,len(violinData))])
combinedData = {'value':combinedValues,'simType':combinedLabels,\
    'radius':combinedRadii,'bin':combinedBins}
combinedFrame = pandas.DataFrame(data=combinedData)

dataCounts = [len(x) for x in violinData]
dataCountsUn = [len(x) for x in violinDataUn]


fig, ax = plt.subplots(figsize=(textwidth,textwidth))
sns.violinplot(data=combinedFrame,x='radius',y='value',hue='simType',\
    split=True,cut=0.0,bw="scott",ax=ax,scale='width',saturation=0.8)
ax = plt.gca()
ax.set_xticklabels([("%.3g" % r) for r in radiiBins])
ax.set_xlabel('Radius Bin [$\\mathrm{Mpc}h^{-1}$]')
ax.set_ylabel('Catalogue Fraction')
numViolins = len(ax.get_xticklabels())
plt.plot(range(0,numViolins),percentilesCat300[0:numViolins],'k:',\
    label='99th \npercentile \nrandom \ncatalogue')
plt.plot(range(0,numViolins),meanCatFrac[0:numViolins],'k-',\
    label='Mean of \nposterior \ncatalogue')
plt.legend(frameon=False,loc='upper right')
plt.title('Catalogue fraction for ' + str(len(snapNumList)) + ' samples.')
plt.ylim([0.0,1.0])
plt.text(0,0.06,"($n_{\\mathrm{MCMC}}$,$n_{\\mathrm{random}}):$",\
    horizontalalignment="center")
for k in range(0,len(dataCounts)):
    plt.text(k,0.02,"(" + str(dataCounts[k]) + "," + str(dataCountsUn[k]) + \
        ")",horizontalalignment = 'center')

plt.savefig(figuresFolder + "catalogue_fraction_violins.pdf")
plt.show()

# Plots in each bin separately:
numCols = 3
numRows = 2
catFracBins = np.arange(0.075,1.075,0.05)
plt.clf()
fig, ax = plt.subplots(numRows,numCols,figsize=(textwidth,0.6*textwidth))
for k in range(0,6):
    i = int(np.floor(k/numCols))
    j = k - i*numCols
    ax[i,j].hist(violinData[k],bins=catFracBins,alpha=0.5,\
        color=seabornColormap[0],label='Posterior',density=True)
    ax[i,j].hist(violinDataUn[k],bins=catFracBins,alpha=0.5,\
        color=seabornColormap[1],label='Random',density=True)
    ax[i,j].set_title(("%.3g" % radBins[k]) + \
        " < $R/\\mathrm{Mpc}h^{-1}$ < " + ("%.3g" % radBins[k+1]),\
        fontsize=10,fontfamily="serif")
    plot.formatPlotGrid(ax,i,j,0,None,1,'Catalogue fraction',numRows,\
        fontsize=8,fontfamily='serif',nCols = numCols,xlim=[0.1,1.0],\
        ylim=[1e-1,20],logy=True)
    ax[i,j].axvline(percentilesCat300[k],linestyle=':',color='k',\
        label = '99th \npercentile\n randoms')
    ax[i,j].text(0.8,10,"$n_{\\mathrm{MCMC}} = $" + str(dataCounts[k]),\
        ha="center",va="center",fontsize=8,fontfamily="serif")
    ax[i,j].text(0.8,6,"$n_{\\mathrm{rand}} = $" + str(dataCountsUn[k]),\
        ha="center",va="center",fontsize=8,fontfamily="serif")

ax[1,2].legend(frameon=False,loc="upper left")
fig.text(0.03,0.5,"Probability density",fontsize=8,fontfamily="serif",\
    rotation="vertical",horizontalalignment='center',\
    verticalalignment="center")
#plt.tight_layout()
plt.subplots_adjust(wspace=0.0,bottom=0.15,top=0.92,left=0.1,right=0.95)
plt.savefig(figuresFolder + "catfrac_distribution_random_vs_posterior.pdf")
plt.show()

textwidth=7.1014
textheight=9.0971
fig, ax = plt.subplots(figsize=(textwidth,textwidth))
for k in range(0,len(dfList2)):
    if len(dfList2[k]) > 0:
        sns.violinplot(data=dfList2[k],x='X',y='value',hue='variable',\
            split=True,cut=0.0,ax=ax)


sns.violinplot(data=combined,x='bin',y='value',hue='variable',split=True,\
    cut=0.0)

plt.violinplot(violinData,positions=meanRadii[:-2])
plt.violinplot(violinDataUn,positions=meanRadii[:-1])

# Anti-halo properties:
allCentralsSorted = [np.array(centralAntihalos[ns][0])[sortedList[ns]] \
    for ns in range(0,len(snapNumList))]
allFinalCatAHs = [finalCatOpt[combinedFilter135][:,ns]-1 \
    for ns in range(0,len(snapNumList))]

#-------------------------------------------------------------------------------
# TESTS WITH DIFFERENT PERCENTILES:

nBinEdges = 8
rLower = 5
rUpper = 20
Om = snapList[0].properties['omegaM0']
rhoc = 2.7754e11
boxsize = snapList[0].properties['boxsize'].ratio("Mpc a h**-1")
N = int(np.cbrt(len(snapList[0])))
mUnit = 8*Om*rhoc*(boxsize/N)**3
mLower = 100*mUnit
mUpper = 2e15
percThresh = 99
cutScale = "mass"

[percentilesCat, percentilesComb] = getThresholdsInBins(\
        nBinEdges-1,cutScale,massListMeanUn,radiiListMeanUn,\
        finalCombinatoricFracUn,finalCatFracUn,\
        rLower,rUpper,mLower,mUpper,percThresh,massBins=massBins,\
        radBins=radBins)


snrList = catData['snr']
meanCentresArray = catData['centres']
catFracCut=True
combFracCut=False
snrCut=True


# Loop:
threshList = np.array([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
voidCounts = np.zeros((7,len(threshList)),dtype=int)
for k in range(0,len(threshList)):
    percentilesCat = threshList[k]*np.ones(7)
    [combinedFilter, meanCatFrac, stdErrCatFrac, \
        meanCombFrac, stdErrCombFrac] = applyCatalogueCuts(finalCatFracOpt,\
        finalCombinatoricFracOpt,percentilesCat,percentilesComb,scaleFilter,\
        snrList,snrThresh,catFracCut,combFracCut,snrCut)
    distanceArray = np.sqrt(np.sum(meanCentresArray**2,1))
    combinedFilter135 = combinedFilter & \
        (distanceArray < rSphereInner)
    for l in range(0,7):
        voidCounts[l,k] = np.sum(scaleFilter[l] & \
            (finalCatFracOpt > percentilesCat[l]) & \
            (distanceArray < rSphereInner))

# Get theory prediction:
[dndm,m] = cosmology.TMF_from_hmf(massBins[0],massBins[-1],\
        h=h,Om0=Om0,Delta=200,delta_wrt='SOCritical',\
        mass_function='Tinker',sigma8=0.8102,Ob0 = 0.04825,\
        transfer_model='CAMB',ns=0.9611,\
        linking_length=0.2)
volSim =  4*np.pi*rSphereInner**3/3
n = cosmology.dndm_to_n(m,dndm,massBins)
bounds = np.array(scipy.stats.poisson(n*volSim).interval(0.95))
# 99th percentiles:
[percentilesCat, percentilesComb] = getThresholdsInBins(\
        nBinEdges-1,cutScale,massListMeanUn,radiiListMeanUn,\
        finalCombinatoricFracUn,finalCatFracUn,\
        rLower,rUpper,mLower,mUpper,percThresh,massBins=massBins,\
        radBins=radBins)
[combinedFilter, meanCatFrac, stdErrCatFrac, \
        meanCombFrac, stdErrCombFrac] = applyCatalogueCuts(finalCatFracOpt,\
        finalCombinatoricFracOpt,percentilesCat,percentilesComb,scaleFilter,\
        snrList,snrThresh,catFracCut,combFracCut,snrCut)
distanceArray = np.sqrt(np.sum(meanCentresArray**2,1))
combinedFilter135 = combinedFilter & \
    (distanceArray < rSphereInner)
voidCounts99 = np.zeros(7,dtype=int)
for l in range(0,7):
    voidCounts99[l] = np.sum(scaleFilter[l] & \
            (finalCatFracOpt > percentilesCat[l]) & \
            (distanceArray < rSphereInner))


# Plot:
fig, ax = plt.subplots(figsize=(0.45*textwidth,0.45*textwidth))
massBinCentres = plot.binCentres(massBins)
for k in range(0,len(threshList)):
    ax.plot(massBinCentres,voidCounts[:,k],\
        label = ("%.2g" % threshList[k]),\
        color = matplotlib.colormaps['hot'](threshList[k]))

ax.set_xlabel("Mass bin [$M_{\\odot}h%{-1}$]",fontsize=8)
ax.set_ylabel("Number of Voids",fontsize=8)
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_ylim([1,200])
ax.plot(massBinCentres,n*volSim,":",label='Tinker \nprediction',\
    color='grey')
ax.fill_between(massBinCentres,bounds[0],bounds[1],facecolor='grey',\
    alpha=0.5,interpolate=True,label='$' + str(100*0.95) + \
    '\%$ \nConfidence \nInterval')
plt.plot(massBinCentres,voidCounts99,color='k',linestyle='--',\
    label='99th \npercentile')
ax.legend(prop={"size":6,"family":"serif"},frameon=False,ncol=2,\
    loc='upper right')
plt.tight_layout()
plt.savefig(figuresFolder + "thresholds_test.pdf")
plt.show()

#-------------------------------------------------------------------------------
# PURITY PLOTS

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


def getAllThresholds(percentiles,radBins,radii):
    scaleFilter = [(radii > radBins[k]) & \
        (radii <= radBins[k+1]) \
        for k in range(0,len(radBins) - 1)]
    thresholds = np.zeros(radii.shape)
    for filt, perc in zip(scaleFilter,percentiles):
        thresholds[filt] = perc
    return thresholds


def getCentresFromCat(catList,centresList,ns):
    centresListOut = np.zeros((len(catList),3),dtype=float)
    for k in range(0,len(catList)):
        if catList[k,ns] > 0:
            centresListOut[k,:] = centresList[ns][catList[k,ns]-1]
        else:
            centresListOut[k,:] = np.nan
    return centresListOut


longCatOptGood = shortCatalogueToLongCatalogue(finalCatOpt[filterOptGood],\
    centralAntihalos,sortedList)



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


def getRadiiFromCat(catList,radiiList):
    radiiListOut = -np.ones(catList.shape,dtype=float)
    for k in range(0,len(catList)):
        for l in range(0,len(catList[0])):
            if catList[k,l] > 0:
                radiiListOut[k,l] = radiiList[l][catList[k,l]-1]
    return radiiListOut




#threshList = np.hstack((np.array([1e-2]),np.arange(0.1,1,0.10)))
#distArr = np.hstack((np.array([1e-2]),np.arange(0.25,3,0.25)))
#distArr = np.arange(0,3,0.25)
#threshList =np.arange(0.0,1,0.10)
#distArr = np.arange(0.0,1.3,0.1) # Catgrid 2
distArr = np.linspace(0.0,0.75,16) # Catgrid 3
#threshList = np.linspace(0.75,1.0,11) # Catgrid 3
threshList = np.linspace(0.5,1.0,21) # Catgrid 3
#threshList =np.arange(0.5,1.01,0.05) # Catgrid 2
#distArr = np.arange(0.5,1.6,0.1) # On the 11 x 6 grid
#threshList =np.arange(0.5,1.01,0.15) # On the 11 x 6 grid
#threshList = np.arange(0.5,1,0.05)
#distArr = np.arange(0,3,0.1)
mLower1 = 1e13
mUpper1 = 1e15
rSphere1 = 135
nBinEdges = 8
rLower = 10
rUpper = 20
percThresh = 99
percThreshList = [50,90,99]
cutScale = "radius"
catFracCut = True
combFracCut = False
snrCut = True
radBins = np.linspace(rLower,rUpper,nBinEdges)
massBins = 10**(np.linspace(np.log10(mLower1),np.log10(mUpper1),nBinEdges))
scaleBins = radBins
snapListRev =  [pynbody.load(samplesFolder + "sample" + str(snapNum) + "/" \
        + "gadget_full_reverse_512/snapshot_001") for snapNum in snapNumList]
hrList = [snap.halos() for snap in snapListRev]
diffMap = [np.setdiff1d(np.arange(0,len(snapNumList)),[k]) \
    for k in range(0,len(snapNumList))]
ahProps = [tools.loadPickle(name + ".AHproperties.p")\
            for name in snapNameList]

purity_1f = np.zeros((len(distArr),len(threshList)))
completeness_1f = np.zeros((len(distArr),len(threshList)))
purity_2f = np.zeros((len(distArr),len(threshList)))
completeness_2f = np.zeros((len(distArr),len(threshList)))
catPercentileRand = np.zeros((len(distArr),len(threshList),\
    len(percThreshList)))
catPercentile99BinsRand = np.zeros((len(distArr),len(threshList),\
    nBinEdges-1))
catPercentileMaxRand = np.zeros((len(distArr),len(threshList)))
catMeanRand = np.zeros((len(distArr),len(threshList)))
catMeanMCMC = np.zeros((len(distArr),len(threshList)))
catMeanMCMCCut = np.zeros((len(distArr),len(threshList)))
catSizeMCMC = np.zeros((len(distArr),len(threshList)),dtype=int)
catSizeMCMCCut = np.zeros((len(distArr),len(threshList)),dtype=int)
catSizeMCMCCutNoBins = np.zeros((len(distArr),len(threshList)),dtype=int)
catMeanMCMCCutNoBins = np.zeros((len(distArr),len(threshList)))
catSizeMCMCCutNoBinsInBins = np.zeros(\
    (len(distArr),len(threshList),nBinEdges-1),dtype=int)
catSizeBinsMCMC = np.zeros((len(distArr),len(threshList),nBinEdges-1),\
    dtype=int)
catSizeBinsMCMCCut = np.zeros((len(distArr),len(threshList),nBinEdges-1),
    dtype=int)

combMeanRand = np.zeros((len(distArr),len(threshList)))
combMeanMCMC = np.zeros((len(distArr),len(threshList)))
combMeanMCMCCut = np.zeros((len(distArr),len(threshList)))
combMeanMCMCCutNoBins = np.zeros((len(distArr),len(threshList)))
combPercentileRand = np.zeros((len(distArr),len(threshList),\
    len(percThreshList)))
combPercentile99BinsRand = np.zeros((len(distArr),len(threshList),\
    nBinEdges-1))
combPercentileMaxRand = np.zeros((len(distArr),len(threshList)))
catSizeBinsMCMCCutComb = np.zeros((len(distArr),len(threshList),\
    nBinEdges-1))
catSizeMCMCCutComb = np.zeros((len(distArr),len(threshList)))
combMeanMCMCCutComb = np.zeros((len(distArr),len(threshList)))
catMeanMCMCCutComb = np.zeros((len(distArr),len(threshList)))

catSizeRand = np.zeros((len(distArr),len(threshList)),dtype=int)
catSizeRandCut = np.zeros((len(distArr),len(threshList)),dtype=int)
catSizeRandNoBinsCut = np.zeros((len(distArr),len(threshList)),dtype=int)



[finalCatOptGood,longCatOptGood,centralAntihalosOld,sortedListOld] = \
    tools.loadPickle("curated_list.p")
splittingCounts = np.zeros((len(distArr),len(threshList),\
    finalCatOptGood.shape[0]),dtype=int)


# For testing these, we can use a combined list of MCMC and random catalogues:
snapList =  [pynbody.load(samplesFolder + "sample" + str(snapNum) + "/" \
        + "gadget_full_forward_512/snapshot_001") for snapNum in snapNumList]
snapListUn = [pynbody.load("new_chain/unconstrained_samples/sample" + \
    str(num) + "/gadget_full_forward_512/snapshot_001") \
    for num in snapNumListUncon]
snapListRevUn = [pynbody.load(\
    "new_chain/unconstrained_samples/sample" + \
    str(num) + "/gadget_full_reverse_512/snapshot_001") \
    for num in snapNumListUncon]

snapNameList = [snap.filename for snap in snapList]
snapNameListRev = [snap.filename for snap in snapListRev]
snapNameListUn = [snap.filename for snap in snapListUn]
snapNameListUnRev = [snap.filename for snap in snapListRevUn]

hrListUn = [snap.halos() for snap in snapListRevUn]
ahPropsUn = [pickle.load(\
            open(snap.filename + ".AHproperties.p","rb")) \
            for snap in snapListUn]
antihaloCentresUn = [tools.remapAntiHaloCentre(props[5],boxsize) \
            for props in ahPropsUn]
antihaloMassesUn = [props[3] for props in ahPropsUn]
antihaloRadiiUn = [props[7] for props in ahPropsUn]

snapNumListComb = snapNumList + snapNumListUncon
snapListCombined = snapList + snapListUn
snapListRevCombined = snapListRev + snapListRevUn
hrListComb = hrList + hrListUn
ahPropsComb = ahProps + ahPropsUn
antihaloRadiiComb = antihaloRadii + antihaloRadiiUn
antihaloMassesComb = antihaloMasses + antihaloMassesUn
antihaloCentresComb = antihaloCentres + antihaloCentresUn

# A few things needed for computing catalogue fractions:
# For MCMC samples:

[mcmcArray,num,N,NCAT,no_bias_params,bias_matrix,mean_field,\
        std_field,hmc_Elh,hmc_Eprior,hades_accept_count,\
        hades_attempt_count] = pickle.load(open(chainFile,"rb"))
snrField = mean_field**2/std_field**2
snrFieldLin = np.reshape(snrField,Nden**3)
grid = snapedit.gridListPermutation(Nden,perm=(2,1,0))
centroids = grid*boxsize/Nden + boxsize/(2*Nden)
positions = snapedit.unwrap(centroids - np.array([boxsize/2]*3),boxsize)
tree = scipy.spatial.cKDTree(snapedit.wrap(positions + boxsize/2,boxsize),\
    boxsize=boxsize)
nearestPointsList = [tree.query_ball_point(\
    snapedit.wrap(antihaloCentres[k] + boxsize/2,boxsize),\
    antihaloRadii[k],workers=-1) \
    for k in range(0,len(antihaloCentres))]
snrAllCatsList = [np.array([np.mean(snrFieldLin[points]) \
    for points in nearestPointsList[k]]) for k in range(0,len(snapNumList))]
snrFilter = [snr > snrThresh for snr in snrAllCatsList]

centralAntihalos = [tools.getAntiHalosInSphere(antihaloCentres[k],rSphere1,\
            filterCondition = (antihaloRadii[k] > rMin) & \
            (antihaloRadii[k] <= rMax) & (antihaloMasses[k] > mLower1) & \
            (antihaloMasses[k] <= mUpper1) & snrFilter[k]) \
            for k in range(0,len(snapNumList))]
centralAntihaloRadii = [\
            antihaloRadii[k][centralAntihalos[k][0]] \
            for k in range(0,len(centralAntihalos))]
sortedList = [np.flip(np.argsort(centralAntihaloRadii[k])) \
                    for k in range(0,len(snapNumList))]
ahCounts = np.array([len(cahs[0]) for cahs in centralAntihalos])
max_index = np.max(ahCounts)
radiiListShort = [np.array([antihaloRadii[l][\
            centralAntihalos[l][0][sortedList[l][k]]] \
            for k in range(0,np.min([ahCounts[l],max_index]))]) \
            for l in range(0,len(snapNumList))]
massListShort = [np.array([antihaloMasses[l][\
            centralAntihalos[l][0][sortedList[l][k]]] \
            for k in range(0,np.min([ahCounts[l],max_index]))]) \
            for l in range(0,len(snapNumList))]
centresListShort = [np.array([antihaloCentres[l][\
            centralAntihalos[l][0][sortedList[l][k]],:] \
            for k in range(0,np.min([ahCounts[l],max_index]))]) \
            for l in range(0,len(snapNumList))]
# For randoms:
centralAntihalosUn = [tools.getAntiHalosInSphere(antihaloCentresUn[k],\
            rSphere1,filterCondition = (antihaloRadiiUn[k] > rMin) & \
            (antihaloRadiiUn[k] <= rMax) & (antihaloMassesUn[k] > mLower1) & \
            (antihaloMassesUn[k] <= mUpper1)) \
            for k in range(0,len(snapNumListUncon))]
centralAntihaloRadiiUn = [\
            antihaloRadiiUn[k][centralAntihalosUn[k][0]] \
            for k in range(0,len(centralAntihalosUn))]
sortedListUn = [np.flip(np.argsort(centralAntihaloRadiiUn[k])) \
                    for k in range(0,len(snapNumListUncon))]
ahCountsUn = np.array([len(cahs[0]) for cahs in centralAntihalosUn])
max_indexUn = np.max(ahCountsUn)
radiiListShortUn = [np.array([antihaloRadiiUn[l][\
    centralAntihalosUn[l][0][sortedListUn[l][k]]] \
    for k in range(0,np.min([ahCountsUn[l],max_indexUn]))]) \
    for l in range(0,len(snapNumListUncon))]
massListShortUn = [np.array([antihaloMassesUn[l][\
    centralAntihalosUn[l][0][sortedListUn[l][k]]] \
    for k in range(0,np.min([ahCountsUn[l],max_indexUn]))]) \
    for l in range(0,len(snapNumListUncon))]

diffMapComb = [np.setdiff1d(np.arange(0,len(snapListCombined)),[k]) \
    for k in range(0,len(snapListCombined))]


diffMapMCMCToRandom = [np.setdiff1d(\
    np.arange(len(snapList),len(snapListCombined)),[k]) \
    for k in range(0,len(snapList))]
rebuildCatalogues  = False
doCrossCatalogues = False
refineCentres = True
sortBy = 'radius'
enforceExclusive = True
NWayMatch = False
# Compute the catalogue with these settings:
for k in range(0,len(distArr)):
    for l in range(0,len(threshList)):
        print(("%.2g" % (100*(k*len(threshList) + l)/\
            (len(distArr)*len(threshList)))) + "% done.")
        rSearch = distArr[k]
        muR = threshList[l]
        if doCrossCatalogues:
            [finalCatTest,shortHaloListTest,twoWayMatchListTest,\
                finalCandidatesTest,finalRatiosTest,finalDistancesTest,\
                allCandidatesTest,candidateCountsTest,allRatiosTest,\
                finalCombinatoricFracTest,finalCatFracTest,alreadyMatchedTest] = \
                tools.loadOrRecompute(data_folder + "catalogue_grid/" + \
                    "cross_catalogue_" + str(k) + "_" + str(l) + ".p",\
                    constructAntihaloCatalogue,\
                    snapNumListComb,snapList=snapListCombined,\
                    snapListRev=snapListRevCombined,\
                    ahProps=ahPropsComb,hrList=hrListComb,max_index=None,\
                    twoWayOnly=True,blockDuplicates=True,\
                    crossMatchThreshold = muR,distMax = rSearch,\
                    rSphere=rSphere1,massRange = [mLower1,mUpper1],\
                    NWayMatch = False,rMin=rMin,rMax=rMax,\
                    additionalFilters = None,verbose=False,\
                    _recomputeData=rebuildCatalogues,\
                    refineCentres = refineCentres,sortBy = sortBy)
        #[finalCatMCMC,shortHaloListMCMC,twoWayMatchListMCMC,\
        #    finalCandidatesMCMC,finalRatiosMCMC,finalDistancesMCMC,\
        #    allCandidatesMCMC,candidateCountsMCMC,allRatiosMCMC,\
        #    finalCombinatoricFracMCMC,finalCatFracMCMC,alreadyMatchedMCMC] = \
        #    tools.loadOrRecompute(data_folder + "catalogue_grid/" + \
        #        "mcmc_catalogue_" + str(k) + "_" + str(l) + ".p",\
        #        constructAntihaloCatalogue,\
        #        snapNumList,snapList=snapList,\
        #        snapListRev=snapListRev,\
        #        ahProps=ahProps,hrList=hrList,max_index=None,\
        #        twoWayOnly=True,blockDuplicates=True,\
        #        crossMatchThreshold = muR,distMax = rSearch,\
        #        rSphere=rSphere1,massRange = [mLower1,mUpper1],\
        #        NWayMatch = False,rMin=rMin,rMax=rMax,\
        #        additionalFilters = snrFilter,verbose=False,\
        #        _recomputeData=rebuildCatalogues,\
        #        refineCentres = refineCentres,sortBy = sortBy)
        #[finalCatRand,shortHaloListRand,twoWayMatchListRand,\
        #    finalCandidatesRand,finalRatiosRand,finalDistancesRand,\
        #    allCandidatesRand,candidateCountsRand,allRatiosRand,\
        #    finalCombinatoricFracRand,finalCatFracRand,alreadyMatchedRand] = \
        #    tools.loadOrRecompute(data_folder + "catalogue_grid/" + \
        #        "random_catalogue_" + str(k) + "_" + str(l) + ".p",\
        #        constructAntihaloCatalogue,\
        #        snapNumListUncon,snapList=snapListUn,\
        #        snapListRev=snapListRevUn,\
        #        ahProps=ahPropsUn,hrList=hrListUn,max_index=None,\
        #        twoWayOnly=True,blockDuplicates=True,\
        #        crossMatchThreshold = muR,distMax = rSearch,\
        #        rSphere=rSphere1,massRange = [mLower1,mUpper1],\
        #        NWayMatch = False,rMin=rMin,rMax=rMax,\
        #        additionalFilters = None,verbose=False,\
        #        _recomputeData=rebuildCatalogues,\
        #        refineCentres = refineCentres,sortBy = sortBy)
        fileMCMC = data_folder + "catalogue_grid/" + \
                "mcmc_catalogue_" + str(k) + "_" + str(l) + ".p"
        fileRand = data_folder + "catalogue_grid/" + \
                "random_catalogue_" + str(k) + "_" + str(l) + ".p"
        # MCMC catalogues:
        if os.path.isfile(fileMCMC) and (not rebuildCatalogues):
            [finalCatMCMC,\
            finalCandidatesMCMC,finalRatiosMCMC,finalDistancesMCMC,\
            allCandidatesMCMC,candidateCountsMCMC,allRatiosMCMC,\
            finalCombinatoricFracMCMC,finalCatFracMCMC,alreadyMatchedMCMC] = \
            tools.loadPickle(fileMCMC)
        else:
            catMCMC = catalogue.combinedCatalogue(\
                snapNameList,snapNameListRev,muR,rSearch,rSphere1,\
                ahProps=ahProps,hrList=hrList,max_index=None,\
                twoWayOnly=True,blockDuplicates=True,\
                massRange = [mLower1,mUpper1],\
                NWayMatch = NWayMatch,rMin=rMin,rMax=rMax,\
                additionalFilters = snrFilter,verbose=False,\
                refineCentres=refineCentres,sortBy=sortBy,\
                enforceExclusive=enforceExclusive)
            finalCatMCMC = catMCMC.constructAntihaloCatalogue()
            [finalCatMCMC,\
            finalCandidatesMCMC,finalRatiosMCMC,finalDistancesMCMC,\
            allCandidatesMCMC,candidateCountsMCMC,allRatiosMCMC,\
            finalCombinatoricFracMCMC,finalCatFracMCMC,alreadyMatchedMCMC] = \
                [catMCMC.finalCat,\
                catMCMC.finalCandidates,\
                catMCMC.finalRatios,catMCMC.finalDistances,\
                catMCMC.allCandidates,catMCMC.candidateCounts,\
                catMCMC.allRatios,catMCMC.finalCombinatoricFrac,\
                catMCMC.finalCatFrac,catMCMC.alreadyMatched]
            if rebuildCatalogues:
                tools.savePickle([catMCMC.finalCat,\
                    catMCMC.finalCandidates,\
                    catMCMC.finalRatios,catMCMC.finalDistances,\
                    catMCMC.allCandidates,catMCMC.candidateCounts,\
                    catMCMC.allRatios,catMCMC.finalCombinatoricFrac,\
                    catMCMC.finalCatFrac,catMCMC.alreadyMatched],fileMCMC)
        # Random catalogues:
        if os.path.isfile(fileRand) and (not rebuildCatalogues):
            [finalCatRand,\
            finalCandidatesRand,finalRatiosRand,finalDistancesRand,\
            allCandidatesRand,candidateCountsRand,allRatiosRand,\
            finalCombinatoricFracRand,finalCatFracRand,alreadyMatchedRand] = \
            tools.loadPickle(fileRand)
        else:
            catRand = catalogue.combinedCatalogue(\
                snapNameListUn,snapNameListUnRev,muR,rSearch,rSphere1,\
                ahProps=ahPropsUn,hrList=hrListUn,max_index=None,\
                twoWayOnly=True,blockDuplicates=True,\
                massRange = [mLower1,mUpper1],\
                NWayMatch = NWayMatch,rMin=rMin,rMax=rMax,\
                additionalFilters = None,verbose=False,\
                refineCentres=refineCentres,sortBy=sortBy,\
                enforceExclusive=enforceExclusive)
            finalCatRand = catRand.constructAntihaloCatalogue()
            [finalCatRand,\
            finalCandidatesRand,finalRatiosRand,finalDistancesRand,\
            allCandidatesRand,candidateCountsRand,allRatiosRand,\
            finalCombinatoricFracRand,finalCatFracRand,alreadyMatchedRand] = \
                [catRand.finalCat,\
                catRand.finalCandidates,\
                catRand.finalRatios,catRand.finalDistances,\
                catRand.allCandidates,catRand.candidateCounts,\
                catRand.allRatios,catRand.finalCombinatoricFrac,\
                catRand.finalCatFrac,catRand.alreadyMatched]
            if rebuildCatalogues:
                tools.savePickle([catRand.finalCat,\
                    catRand.finalCandidates,\
                    catRand.finalRatios,catRand.finalDistances,\
                    catRand.allCandidates,catRand.candidateCounts,\
                    catRand.allRatios,catRand.finalCombinatoricFrac,\
                    catRand.finalCatFrac,catRand.alreadyMatched],fileRand)
        # Compute percentiles:
        radiiListOpt = getPropertyFromCat(finalCatMCMC,radiiListShort)
        massListOpt = getPropertyFromCat(finalCatMCMC,massListShort)
        [radiiMeanOpt, radiiSigmaOpt]  = getMeanProperty(radiiListOpt)
        [massMeanOpt, massSigmaOpt]  = getMeanProperty(massListOpt)
        scaleFilter = [(radiiMeanOpt > radBins[k]) & \
            (radiiMeanOpt <= radBins[k+1]) \
            for k in range(0,len(radBins) - 1)]
        radiiListCombUn = getPropertyFromCat(finalCatRand,radiiListShortUn)
        massListCombUn = getPropertyFromCat(finalCatRand,massListShortUn)
        [radiiListMeanUn,radiiListSigmaUn] = getMeanProperty(radiiListCombUn)
        [massListMeanUn,massListSigmaUn] = getMeanProperty(massListCombUn)
        [percentilesCatTest, percentilesCombTest] = getThresholdsInBins(\
            nBinEdges-1,cutScale,massListMeanUn,radiiListMeanUn,\
            finalCombinatoricFracRand,finalCatFracRand,\
            rLower,rUpper,mLower1,mUpper1,percThresh,massBins=massBins,\
            radBins=radBins)
        finalCentresOptList = np.array([getCentresFromCat(\
            finalCatMCMC,centresListShort,ns) \
            for ns in range(0,len(snapNumList))])
        meanCentreOpt = np.nanmean(finalCentresOptList,0)
        nearestPoints = tree.query_ball_point(\
            snapedit.wrap(meanCentreOpt + boxsize/2,boxsize),radiiMeanOpt,\
            workers=-1)
        snrList = np.array([np.mean(snrFieldLin[points]) \
            for points in nearestPoints])
        [combinedFilterTest, meanCatFracTest, stdErrCatFracTest, \
            meanCombFracTest, stdErrCombFracTest] = applyCatalogueCuts(\
            finalCatFracMCMC,finalCombinatoricFracMCMC,percentilesCatTest,\
            percentilesCombTest,scaleFilter,snrList,snrThresh,True,\
            False,snrCut)
        [combinedFilterComb, meanCatFracComb, stdErrCatFracComb, \
            meanCombFracComb, stdErrCombFracComb] = applyCatalogueCuts(\
            finalCatFracMCMC,finalCombinatoricFracMCMC,percentilesCatTest,\
            percentilesCombTest,scaleFilter,snrList,snrThresh,False,\
            True,snrCut)
        # Splitting of the curated catalogue:
        if len(finalCatMCMC) > 0:
            longCatTest = shortCatalogueToLongCatalogue(finalCatMCMC,\
                centralAntihalos,sortedList)
            radiiListTest = getRadiiFromCat(finalCatMCMC,radiiListShort)
            [radiiMeanTest, radiiSigmaTest]  = getMeanProperty(radiiListTest)
            finalCentresTestList = np.array([getCentresFromCat(\
                finalCatMCMC,centresListShort,ns) \
                for ns in range(0,len(snapNumList))])
            meanCentreTest = np.nanmean(finalCentresTestList,0)
            distancesTest = np.sqrt(np.sum(meanCentreTest**2,1))
            thresholdsTest = getAllThresholds(percentilesCatTest,radBins,\
                radiiMeanTest)
            filterTest = (radiiMeanTest > 10) & (radiiMeanTest <= 25) & \
                (distancesTest < 135) & (finalCatFracMCMC > thresholdsTest)
            splitListGood = getSplitList(longCatOptGood,longCatTest[filterTest])
            numSplitGood = np.array([len(x) for x in splitListGood],dtype=int)
            splittingCounts[k,l,:] = numSplitGood
        # Get the anti-halo properties used for filtering:
        centralAntihalosTest = [tools.getAntiHalosInSphere(\
            antihaloCentresComb[ns],rSphere1,\
            filterCondition = (antihaloRadiiComb[ns] > rMin) & \
            (antihaloRadiiComb[ns] <= rMax) & \
            (antihaloMassesComb[ns] > mLower1) & \
            (antihaloMassesComb[ns] <= mUpper1) ) \
            for ns in range(0,len(snapListCombined))]
        centralAntihaloMassesTest = [\
            antihaloMassesComb[ns][centralAntihalosTest[ns][0]] \
            for ns in range(0,len(centralAntihalosTest))]
        sortedListTest = [\
            np.flip(np.argsort(centralAntihaloMassesTest[ns])) \
            for ns in range(0,len(snapListCombined))]
        ahCountsTest = np.array([len(cahs[0]) \
            for cahs in centralAntihalosTest])
        massListShortTest = [np.array([antihaloMassesComb[ns1][\
            centralAntihalosTest[ns1][0][sortedListTest[ns1][ns2]]] \
            for ns2 in range(0,np.min([ahCountsTest[ns1],max_index]))]) \
            for ns1 in range(0,len(snapListCombined))]
        massFilter = [(massListShortTest[n] > mLower1) & \
            (massListShortTest[n] <= mUpper1) \
            for n in range(0,len(massListShortTest))]
        # Catalogue fraction calculations:
        catPercentile99BinsRand[k,l,:] = percentilesCatTest
        combPercentile99BinsRand[k,l,:] = percentilesCombTest
        conditionRand = (radiiListMeanUn > radBins[0]) & \
            (radiiListMeanUn <= radBins[-1])
        allBinsFilterRand = np.where(conditionRand)[0]
        allBinsFilterMCMC = np.where((radiiMeanOpt > radBins[0]) & \
            (radiiMeanOpt <= radBins[-1]))[0]
        catSizeRand[k,l] = len(finalCatFracRand[allBinsFilterRand])
        scaleFilterRand = [(radiiListMeanUn > radBins[k]) & \
            (radiiListMeanUn <= radBins[k+1]) \
            for k in range(0,len(radBins) - 1)]
        [combinedFilterRand, meanCatFracRand, stdErrCatFracRand, \
            meanCombFracRand, stdErrCombFracRand] = applyCatalogueCuts(\
            finalCatFracRand,finalCombinatoricFracRand,percentilesCatTest,\
            percentilesCombTest,scaleFilterRand,snrList,snrThresh,True,\
            False,False)
        if len(finalCatFracRand) > 0:
            if len(allBinsFilterRand) > 0:
                catPercentileMaxRand[k,l] = np.max(\
                    finalCatFracRand[allBinsFilterRand])
                combPercentileMaxRand[k,l] = np.max(\
                    finalCombinatoricFracRand[allBinsFilterRand])
                catSizeRandNoBinsCut[k,l] = len(\
                    finalCatFracRand[conditionRand & \
                    (finalCatFracRand > np.percentile(finalCatFracRand,99))])
                for m,thresh in zip(range(0,len(percThreshList)),percThreshList):
                    catPercentileRand[k,l,m] = np.percentile(\
                        finalCatFracRand[allBinsFilterRand],thresh)
                    combPercentileRand[k,l,m] = np.percentile(\
                        finalCombinatoricFracRand[allBinsFilterRand],thresh)
            if np.sum(combinedFilterRand) > 0:
                catSizeRandCut[k,l] = len(finalCatFracRand[combinedFilterRand])
            catMeanRand[k,l] = np.mean(finalCatFracRand)
            combMeanRand[k,l] = np.mean(finalCombinatoricFracRand)
            combinedFilterNoBins = np.where(\
                (finalCatFracMCMC > catPercentileRand[k,l,-1]) & \
                (radiiMeanOpt > radBins[0]) & \
                (radiiMeanOpt <= radBins[-1]))[0]
            catSizeMCMCCutNoBins[k,l] = len(\
                finalCatFracMCMC[combinedFilterNoBins])
            catMeanMCMCCutNoBins[k,l] = np.mean(\
                finalCatFracMCMC[combinedFilterNoBins])
            combMeanMCMCCutNoBins[k,l] = np.mean(\
                finalCombinatoricFracMCMC[combinedFilterNoBins])
            [inRadBinsCut,noInRadBinsCut] = plot.binValues(\
                radiiMeanOpt[combinedFilterNoBins],radBins)
            catSizeMCMCCutNoBinsInBins[k,l,:] = noInRadBinsCut
        if len(finalCatFracMCMC) > 0:
            catMeanMCMC[k,l] = np.mean(finalCatFracMCMC)
        if len(finalCombinatoricFracMCMC) > 0:
            combMeanMCMC[k,l] = np.mean(finalCombinatoricFracMCMC)
        if np.sum(combinedFilterTest) > 0:
            catMeanMCMCCut[k,l] = np.mean(finalCatFracMCMC[combinedFilterTest])
            combMeanMCMCCut[k,l] = np.mean(\
                finalCombinatoricFracMCMC[combinedFilterTest])
        if np.sum(combinedFilterComb) > 0:
            catMeanMCMCCutComb[k,l] = np.mean(\
                finalCatFracMCMC[combinedFilterComb])
            combMeanMCMCCutComb[k,l] = np.mean(\
                finalCombinatoricFracMCMC[combinedFilterComb])
        catSizeMCMC[k,l] = len(finalCatFracMCMC)
        catSizeMCMCCut[k,l] = len(finalCatFracMCMC[combinedFilterTest])
        catSizeMCMCCutComb[k,l] = len(finalCatFracMCMC[combinedFilterComb])
        [inRadBinsCut,noInRadBinsCut] = plot.binValues(\
            radiiMeanOpt[combinedFilterTest],radBins)
        [inRadBinsCutComb,noInRadBinsCutComb] = plot.binValues(\
            radiiMeanOpt[combinedFilterComb],radBins)
        [inRadBins,noInRadBins] = plot.binValues(\
            radiiMeanOpt,radBins)
        catSizeBinsMCMC[k,l,:] = noInRadBins
        catSizeBinsMCMCCut[k,l,:] = noInRadBinsCut
        catSizeBinsMCMCCutComb[k,l,:] = noInRadBinsCutComb
        # Compute purity and completeness:
        #purity_1f[k,l] = 1.0 - np.mean([[\
        #    np.sum((candidateCountsTest[i][j] > 0) & \
        #    (massFilter[i]))/np.sum(massFilter[i]) \
        #    for j in diffMapMCMCToRandom[i]] \
        #    for i in range(0,len(snapList))])
        #completeness_1f[k,l] = np.mean(\
        #    [[np.sum((candidateCountsTest[j][i] > 0) & \
        #    (massFilter[j]))/np.sum((massFilter[j])) \
        #    for j in diffMap[i]] \
        #    for i in range(0,len(snapList))])
        #purity_2f[k,l] = 1.0 - np.mean(\
        #    [[np.sum(np.array(twoWayMatchListTest[i])[:,j] & \
        #    (massFilter[i]))/np.sum(massFilter[i]) \
        #    for i in range(0,len(snapList))] \
        #    for j in range(len(snapList)-1,len(ahCountsTest)-1)])
        #completeness_2f[k,l] = np.mean(\
        #    [[np.sum(np.array(twoWayMatchListTest[i])[:,j] & \
        #    (massFilter[i]))/np.sum(massFilter[i]) \
        #    for i in range(0,len(snapList))] \
        #    for j in range(0,len(snapList)-1)])

euclideanDist2 = np.sqrt((purity_2f - 1.0)**2  + (completeness_2f - 1.0)**2)
euclideanDist1 = np.sqrt((purity_1f - 1.0)**2  + (completeness_1f - 1.0)**2)


tools.savePickle([splittingCounts],data_folder + "splitting_counts.p")

tools.savePickle([catSizeRand,catSizeRandCut,catSizeRandNoBinsCut],\
    data_folder + "randcat_counts_data.p")

tools.savePickle([combMeanRand,combMeanMCMC,combMeanMCMCCut,\
    combMeanMCMCCutNoBins,combPercentileRand,combPercentileMaxRand,\
    combPercentile99BinsRand,catSizeBinsMCMCCutComb,\
    catSizeMCMCCutComb,combMeanMCMCCutComb,catMeanMCMCCutComb],\
    data_folder + "combinatoric_fraction_data.p")

tools.savePickle([purity_1f,purity_2f,completeness_1f,completeness_2f,\
    catPercentile99BinsRand,catPercentileRand,catMeanRand,\
    catMeanMCMC,catMeanMCMCCut,catSizeMCMC,catSizeMCMCCut,\
    catSizeBinsMCMC,catSizeBinsMCMCCut,catSizeMCMCCutNoBins,\
    catSizeMCMCCutNoBinsInBins,catPercentileMaxRand,\
    catMeanMCMCCutNoBins],\
    data_folder + "purity_completeness_data.p")

[purity_1f,purity_2f,completeness_1f,completeness_2f,\
    catPercentile99BinsRand,catPercentileRand,catMeanRand,\
    catMeanMCMC,catMeanMCMCCut,catSizeMCMC,catSizeMCMCCut,\
    catSizeBinsMCMC,catSizeBinsMCMCCut,catSizeMCMCCutNoBins,\
    catSizeMCMCCutNoBinsInBins,catPercentileMaxRand,\
    catMeanMCMCCutNoBins]= tools.loadPickle(\
    data_folder + "purity_completeness_data.p")

[catSizeRand,catSizeRandCut,catSizeRandNoBinsCut] = tools.loadPickle(\
    data_folder + "randcat_counts_data.p")

[combMeanRand,combMeanMCMC,combMeanMCMCCut,\
    combMeanMCMCCutNoBins,combPercentileRand,combPercentileMaxRand,\
    combPercentile99BinsRand,catSizeBinsMCMCCutComb,\
    catSizeMCMCCutComb,combMeanMCMCCutComb,catMeanMCMCCutComb] = \
    tools.loadPickle(data_folder + "combinatoric_fraction_data.p")

[splittingCounts] = tools.loadPickle(data_folder + "splitting_counts.p")


[purity_1f,purity_2f,completeness_1f,completeness_2f] = tools.loadPickle(\
    "borg-antihalos_paper_figures/all_samples/" + \
    "purity_completeness_data_whole_range.p")

# Catgrid 3:
distArr = np.linspace(0.0,0.75,16) # Catgrid 3
threshList = np.linspace(0.75,1.0,11) # Catgrid 3
[purity_1f,purity_2f,completeness_1f,completeness_2f,\
    catPercentile99BinsRand,catPercentileRand,catMeanRand,\
    catMeanMCMC,catMeanMCMCCut,catSizeMCMC,catSizeMCMCCut,\
    catSizeBinsMCMC,catSizeBinsMCMCCut,catSizeMCMCCutNoBins,\
    catSizeMCMCCutNoBinsInBins,catPercentileMaxRand,\
    catMeanMCMCCutNoBins] =  tools.loadPickle(\
    "borg-antihalos_paper_figures/all_samples/" + "purity_completeness_data_catgrid3.p")
[combMeanRand,combMeanMCMC,combMeanMCMCCut,\
    combMeanMCMCCutNoBins,combPercentileRand,combPercentileMaxRand,\
    combPercentile99BinsRand,catSizeBinsMCMCCutComb,\
    catSizeMCMCCutComb,combMeanMCMCCutComb,catMeanMCMCCutComb] = tools.loadPickle(\
    "borg-antihalos_paper_figures/all_samples/" + "combinatoric_fraction_data_catgrid3.p")

# Catgrid 2:
distArr = np.arange(0.0,1.3,0.1) # Catgrid 2
threshList =np.arange(0.5,1.01,0.05) # Catgrid 2
[purity_1f,purity_2f,completeness_1f,completeness_2f,\
    catPercentile99BinsRand,catPercentileRand,catMeanRand,\
    catMeanMCMC,catMeanMCMCCut,catSizeMCMC,catSizeMCMCCut,\
    catSizeBinsMCMC,catSizeBinsMCMCCut,catSizeMCMCCutNoBins,\
    catSizeMCMCCutNoBinsInBins,catPercentileMaxRand,\
    catMeanMCMCCutNoBins] =  tools.loadPickle(\
    "borg-antihalos_paper_figures/all_samples/" + "purity_completeness_data_catgrid2.p")
[combMeanRand,combMeanMCMC,combMeanMCMCCut,\
    combMeanMCMCCutNoBins,combPercentileRand,combPercentileMaxRand,\
    combPercentile99BinsRand,catSizeBinsMCMCCutComb,\
    catSizeMCMCCutComb,combMeanMCMCCutComb,catMeanMCMCCutComb] = tools.loadPickle(\
    "borg-antihalos_paper_figures/all_samples/" + "combinatoric_fraction_data_catgrid2.p")

ylabel = 'Radius ratio threshold ($\mu_{\mathrm{R}}$)'
imshowComparison(completeness_2f.T,completeness_1f.T,\
    extentLeft= (np.min(distArr),np.max(distArr),\
    np.min(threshList),np.max(threshList)),\
    extentRight = (np.min(distArr),np.max(distArr),\
    np.min(threshList),np.max(threshList)),ylabel=ylabel,vLeft=[0,1.0],\
    titleLeft='Two-way completeness ($^2C_{\mu_{\mathrm{rad}}}$)',\
    titleRight = 'One-way completeness ($^1C_{\mu_{\mathrm{rad}}}$)')


imshowComparison(purity_2f.T,completeness_2f.T,\
    extentLeft= (np.min(distArr),np.max(distArr),\
    np.min(threshList),np.max(threshList)),\
    extentRight = (np.min(distArr),np.max(distArr),\
    np.min(threshList),np.max(threshList)),ylabel=ylabel,vLeft=[0,1.0],\
    titleLeft='One-way purity ($^2C_{\mu_{\mathrm{rad}}}$)',\
    titleRight = 'One-way completeness ($^1C_{\mu_{\mathrm{rad}}}$)')


imshowComparison(euclideanDist2.T,euclideanDist1.T,\
    extentLeft= (np.min(distArr),np.max(distArr),\
    np.min(threshList),np.max(threshList)),\
    extentRight = (np.min(distArr),np.max(distArr),\
    np.min(threshList),np.max(threshList)),ylabel=ylabel,vLeft=[0,1.0],\
    titleLeft='$\\sqrt{(^2C_{\mu_{\mathrm{rad}}}-1)^2 + ' + \
    '(^2P_{\mu_{\mathrm{rad}}}-1)^2}$',\
    titleRight = '$\\sqrt{(^1C_{\mu_{\mathrm{rad}}}-1)^2 + ' + \
    '(^1P_{\mu_{\mathrm{rad}}}-1)^2}$')

plt.plot(distArr,purity_2f)

# Purity at fixed mu_R:
sm = cm.ScalarMappable(colors.Normalize(vmin=0,\
    vmax=1.0),cmap=matplotlib.colormaps['hot'])
for k in range(0,len(threshList)):
    plt.plot(distArr,purity_2f[:,k],\
        color = matplotlib.colormaps['hot'](threshList[k]))

plt.xlabel('$\\mu_S$')
plt.ylabel('Purity, $^2P_{\mu_{\mathrm{rad}}}$')
plt.colorbar(sm,label='$\\mu_R$')
plt.show()



plt.plot(distArr,euclideanDist2)


plt.plot(threshList,euclideanDist2.T)

# Euclidean distance at fixed mu_S:
sm = cm.ScalarMappable(colors.Normalize(vmin=np.min(distArr),\
    vmax=np.max(distArr)),cmap=matplotlib.colormaps['hot'])
for k in range(0,len(distArr)):
    plt.plot(threshList,euclideanDist2[k,:],\
        color = matplotlib.colormaps['hot'](distArr[k]/np.max(distArr)))

plt.xlabel('$\\mu_R$')
plt.ylabel('$\\sqrt{(^2C_{\mu_{\mathrm{rad}}}-1)^2 + ' + \
    '(^2P_{\mu_{\mathrm{rad}}}-1)^2}$')
plt.colorbar(sm,label='$\\mu_S$')
plt.show()


# Euclidean distance at fixed mu_R:
sm = cm.ScalarMappable(colors.Normalize(vmin=0,\
    vmax=1.0),cmap=matplotlib.colormaps['hot'])
for k in range(0,len(threshList)):
    plt.plot(distArr,euclideanDist2[:,k],\
        color = matplotlib.colormaps['hot'](threshList[k]))

plt.xlabel('$\\mu_S$')
plt.ylabel('$\\sqrt{(^2C_{\mu_{\mathrm{rad}}}-1)^2 + ' + \
    '(^2P_{\mu_{\mathrm{rad}}}-1)^2}$')
plt.colorbar(sm,label='$\\mu_R$')
plt.show()

# Build 2-d interpolators to use with optimising the parameters:
from scipy.interpolate import RegularGridInterpolator
interpolator = RegularGridInterpolator((distArr,threshList),\
    euclideanDist2,method='cubic')
interpolator1 = RegularGridInterpolator((distArr,threshList),\
    euclideanDist1,method='cubic')


purityInterp = scipy.interpolate.RegularGridInterpolator((distArr,threshList),\
    purity_2f,method='cubic')
completenessInterp = scipy.interpolate.RegularGridInterpolator(\
    (distArr,threshList),completeness_2f,method='cubic')


purityInterp1 = scipy.interpolate.RegularGridInterpolator((distArr,threshList),\
    purity_1f,method='cubic')
completenessInterp1 = scipy.interpolate.RegularGridInterpolator(\
    (distArr,threshList),completeness_1f,method='cubic')



# Minimise to obtain optimal parameters:
optimal = scipy.optimize.minimize(interpolator,np.array([1,0.5]),\
    bounds=(tools.minmax(distArr),tools.minmax(threshList)))
optimal1 = scipy.optimize.minimize(interpolator1,np.array([1,0.5]),\
    bounds=(tools.minmax(distArr),tools.minmax(threshList)))


optimumParams = optimal.x
optimalPurity = purityInterp(optimumParams)
optimalCompleteness = completenessInterp(optimumParams)


optimumParams1 = optimal1.x
optimalPurity1 = purityInterp1(optimumParams1)
optimalCompleteness1 = completenessInterp1(optimumParams1)



# Plot interpolated surface:
xx = np.linspace(np.min(distArr) + 1e-3,np.max(distArr)-1e-3,100)
yy = np.linspace(np.min(threshList) + 1e-3,np.max(threshList) - 1e-3,100)
X, Y = np.meshgrid(xx,yy,indexing='ij')
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_wireframe(X,Y,interpolator((X,Y)),alpha=0.5,color='grey')
ax.scatter3D(optimumParams[0],optimumParams[1],optimal.fun,\
    color=seabornColormap[0],label = "Optimal, $(\mu_R = " + \
   ("%.3g" % optimumParams[0]) + ",\mu_S = " + \
   ("%.3g" % optimumParams[1]) + ")$")
ax.set_xlabel('Search distance ratio, $\\mu_{S}$')
ax.set_ylabel('Radius ratio, $\\mu_{R}$')
ax.set_zlabel('$\\sqrt{(^2C-1)^2 + ' + \
    '(^2P-1)^2}$')
plt.legend()
plt.tight_layout()
plt.savefig(figuresFolder + 'optimal_parameters.pdf')
plt.show()


# Plot Completeness and Purity on separate wireframes:

# Plot interpolated surface:
xx = np.linspace(np.min(distArr) + 1e-3,np.max(distArr)-1e-3,100)
yy = np.linspace(np.min(threshList) + 1e-3,np.max(threshList) - 1e-3,100)
X, Y = np.meshgrid(xx,yy,indexing='ij')
fig = plt.figure()
ax1 = fig.add_subplot(1,2,1,projection='3d')
ax2 = fig.add_subplot(1,2,2,projection='3d')
ax1.plot_wireframe(X,Y,purityInterp((X,Y)),alpha=0.5,color='grey')
ax2.plot_wireframe(X,Y,completenessInterp((X,Y)),alpha=0.5,color='grey')
ax1.scatter3D(optimumParams[0],optimumParams[1],purityInterp(optimumParams),\
    color=seabornColormap[0],label = "Optimal, $(\mu_R = " + \
    ("%.3g" % optimumParams[0]) + ",\mu_S = " + \
    ("%.3g" % optimumParams[1]) + ")$")
ax2.scatter3D(optimumParams[0],optimumParams[1],\
    completenessInterp(optimumParams),\
    color=seabornColormap[0],label = "Optimal, $(\mu_R = " + \
    ("%.3g" % optimumParams[0]) + ",\mu_S = " + \
    ("%.3g" % optimumParams[1]) + ")$")
ax1.set_xlabel('Search distance ratio, $\\mu_{S}$')
ax1.set_ylabel('Radius ratio, $\\mu_{R}$')
ax1.set_zlabel('Purity, $^2P$')
ax2.set_xlabel('Search distance ratio, $\\mu_{S}$')
ax2.set_ylabel('Radius ratio, $\\mu_{R}$')
ax2.set_zlabel('Completeness, $^2C$')
plt.legend()
#plt.tight_layout()
plt.savefig(figuresFolder + 'optimal_purity_completeness.pdf')
plt.show()




# Scatter plot:
fig, ax = plt.subplots(figsize=(textwidth,textwidth))
ax.scatter(purity_1f,completeness_1f,color=seabornColormap[0],label='1-way')
ax.scatter(purity_2f,completeness_2f,color=seabornColormap[1],label='2-way')
ax.set_xlabel('Purity')
ax.set_ylabel('Completeness')
ax.set_ylim([0,1.0])
ax.set_xlim([0,1.0])
ax.legend()
ax.set_aspect('equal')
plt.savefig(figuresFolder + "purity_completeness_scatter.pdf")
plt.show()



# Plot of the catalogue fraction of survivors:
# Plot interpolated surface:
xx = np.linspace(np.min(distArr) + 1e-3,np.max(distArr)-1e-3,100)
yy = np.linspace(np.min(threshList) + 1e-3,np.max(threshList) - 1e-3,100)
X, Y = np.meshgrid(xx,yy,indexing='ij')
catPercentile99Rand_Interp = scipy.interpolate.RegularGridInterpolator(\
    (distArr,threshList),catPercentile99Rand,method='cubic')
catMeanRand_Interp = scipy.interpolate.RegularGridInterpolator(\
    (distArr,threshList),catMeanRand,method='cubic')
fig = plt.figure()
ax1 = fig.add_subplot(1,2,1,projection='3d')
ax2 = fig.add_subplot(1,2,2,projection='3d')
ax1.plot_wireframe(X,Y,catPercentile99Rand_Interp((X,Y)),alpha=0.5,color='grey')
ax2.plot_wireframe(X,Y,catMeanRand_Interp((X,Y)),alpha=0.5,color='grey')
ax1.scatter3D(Xi,Yi,catPercentile99Rand,color=seabornColormap[0],label = "Samples")
ax2.scatter3D(Xi,Yi,catMeanRand,color=seabornColormap[0],label = "Samples")
ax1.set_xlabel('Search distance ratio, $\\mu_{S}$')
ax1.set_ylabel('Radius ratio, $\\mu_{R}$')
ax1.set_zlabel('Catalogue fraction')
ax2.set_xlabel('Search distance ratio, $\\mu_{S}$')
ax2.set_ylabel('Radius ratio, $\\mu_{R}$')
ax2.set_zlabel('Catalogue fraction')
ax1.set_title('99th Percentile of Random Catalogues')
ax2.set_title('Mean of Random Catalogues')
plt.legend()
#plt.tight_layout()
plt.savefig(figuresFolder + 'randcat_fcat.pdf')
plt.show()



# Plot of the catalogue fraction of survivors:
# Plot interpolated surface:
xx = np.linspace(np.min(distArr) + 1e-3,np.max(distArr)-1e-3,100)
yy = np.linspace(np.min(threshList) + 1e-3,np.max(threshList) - 1e-3,100)
X, Y = np.meshgrid(xx,yy,indexing='ij')
catPercentile99Rand_Interp = scipy.interpolate.RegularGridInterpolator(\
    (distArr,threshList),catPercentile99Rand,method='cubic')
catMeanRand_Interp = scipy.interpolate.RegularGridInterpolator(\
    (distArr,threshList),catMeanRand,method='cubic')
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1,projection='3d')
ax1.plot_wireframe(X,Y,catPercentile99Rand_Interp((X,Y)),alpha=0.5,\
    color='grey',label='99th percentile')
ax1.plot_wireframe(X,Y,catMeanRand_Interp((X,Y)),alpha=0.5,color='k',\
    label='mean')
ax1.scatter3D(Xi,Yi,catPercentile99Rand,color=seabornColormap[0])
ax1.scatter3D(Xi,Yi,catMeanRand,color=seabornColormap[1])
ax1.set_xlabel('Search distance ratio, $\\mu_{S}$')
ax1.set_ylabel('Radius ratio, $\\mu_{R}$')
ax1.set_zlabel('Catalogue fraction')
ax1.set_title('Catalogue fractions, random catalogues')
plt.legend()
#plt.tight_layout()
plt.savefig(figuresFolder + 'randcat_fcat_combined.pdf')
plt.show()

# Heatmaps of randoms:
# Catalogue fractions:
xx = np.linspace(np.min(distArr) + 1e-3,np.max(distArr)-1e-3,100)
yy = np.linspace(np.min(threshList) + 1e-3,np.max(threshList) - 1e-3,100)
X, Y = np.meshgrid(xx,yy,indexing='ij')
cmap = 'coolwarm'
catPercentile99Rand_Interp = scipy.interpolate.RegularGridInterpolator(\
    (distArr,threshList),catPercentileRand[:,:,-1],method='cubic')
#catMaxRand_Interp = scipy.interpolate.RegularGridInterpolator(\
#    (distArr,threshList),catPercentileMaxRand,method='cubic')
combPercentile99Rand_Interp = scipy.interpolate.RegularGridInterpolator(\
    (distArr,threshList),combPercentileRand[:,:,-1],method='cubic')
fig, ax = plt.subplots(1,2,figsize=(textwidth,0.45*textwidth))
ax1 = ax[0]
ax2 = ax[1]
ax1.imshow(catPercentile99Rand_Interp((X,Y)).T,vmin=0,vmax=1.0,\
    cmap=cmap,extent=(np.min(distArr),np.max(distArr),\
    np.min(threshList),np.max(threshList)),aspect='auto',origin='lower')
ax2.imshow(combPercentile99Rand_Interp((X,Y)).T,vmin=0,vmax=0.1,cmap=cmap,\
    extent=(np.min(distArr),np.max(distArr),\
    np.min(threshList),np.max(threshList)),aspect='auto',origin='lower')
c1 = ax1.contour(X,Y,catPercentile99Rand_Interp((X,Y)),[0.1,0.2,0.3],\
    colors=['k','k','k'])
ax1.clabel(c1,inline=True)
Xi, Yi = np.meshgrid(distArr,threshList,indexing='ij')
ax1.scatter(Xi,Yi,marker='.',color='k')
ax2.scatter(Xi,Yi,marker='.',color='k')
#ax1.contour(X,Y,catPercentile99Rand_Interp((X,Y)))
c2 = ax2.contour(X,Y,combPercentile99Rand_Interp((X,Y)),[0.01,0.02,0.03],\
    colors=['k','k','k'])
ax2.clabel(c2,inline=True)
#ax2.contour(X,Y,catMeanRand_Interp((X,Y)))
ax1.set_xlabel('Search distance ratio, $\\mu_{S}$')
ax1.set_ylabel('Radius ratio, $\\mu_{R}$')
ax1.set_title('99th Percentile \nCatalogue Fraction')
ax2.set_xlabel('Search distance ratio, $\\mu_{S}$')
ax2.set_ylabel('Radius ratio, $\\mu_{R}$')
ax2.set_title('99th Percentile \nCombinatoric Fraction')
sm = cm.ScalarMappable(colors.Normalize(vmin=0.0,vmax=1.0),cmap=cmap)
plt.colorbar(sm,label='Catalogue fraction',ax=ax1)
sm = cm.ScalarMappable(colors.Normalize(vmin=0.0,vmax=0.1),cmap=cmap)
plt.colorbar(sm,label='Coombinatoric fraction',ax=ax2)
#cbar = plt.colorbar(sm, orientation="vertical",\
#    label='Catalogue fraction',cax=cbax)
plt.tight_layout()
#plt.subplots_adjust(top=0.87,bottom=0.195,left=0.099,right=0.839,hspace=0.2,\
#    wspace=0.254)
plt.savefig(figuresFolder + 'randcat_fcat_heatmap.pdf')
plt.show()



# Heatmaps of randoms:
# Catalogue fractions:
xx = np.linspace(np.min(distArr) + 1e-3,np.max(distArr)-1e-3,100)
yy = np.linspace(np.min(threshList) + 1e-3,np.max(threshList) - 1e-3,100)
X, Y = np.meshgrid(xx,yy,indexing='ij')
cmap = 'coolwarm'
catPercentile99Rand_Interp = scipy.interpolate.RegularGridInterpolator(\
    (distArr,threshList),catPercentileRand[:,:,-1],method='cubic')
catMaxRand_Interp = scipy.interpolate.RegularGridInterpolator(\
    (distArr,threshList),catPercentileMaxRand,method='cubic')
#combPercentile99Rand_Interp = scipy.interpolate.RegularGridInterpolator(\
#    (distArr,threshList),combPercentileRand[:,:,-1],method='cubic')
fig, ax = plt.subplots(1,2,figsize=(textwidth,0.45*textwidth))
ax1 = ax[0]
ax2 = ax[1]
ax1.imshow(catPercentile99Rand_Interp((X,Y)).T,vmin=0,vmax=1.0,\
    cmap=cmap,extent=(np.min(distArr),np.max(distArr),\
    np.min(threshList),np.max(threshList)),aspect='auto',origin='lower')
ax2.imshow(catMaxRand_Interp((X,Y)).T,vmin=0,vmax=1,cmap=cmap,\
    extent=(np.min(distArr),np.max(distArr),\
    np.min(threshList),np.max(threshList)),aspect='auto',origin='lower')
c1 = ax1.contour(X,Y,catPercentile99Rand_Interp((X,Y)),[0.1,0.2,0.3],\
    colors=['k','k','k'])
ax1.clabel(c1,inline=True)
Xi, Yi = np.meshgrid(distArr,threshList,indexing='ij')
ax1.scatter(Xi,Yi,marker='.',color='k')
ax2.scatter(Xi,Yi,marker='.',color='k')
#ax1.contour(X,Y,catPercentile99Rand_Interp((X,Y)))
c2 = ax2.contour(X,Y,catMaxRand_Interp((X,Y)),[0.1,0.2,0.3],\
    colors=['k','k','k'])
ax2.clabel(c2,inline=True)
#ax2.contour(X,Y,catMeanRand_Interp((X,Y)))
ax1.set_xlabel('Search distance ratio, $\\mu_{S}$')
ax1.set_ylabel('Radius ratio, $\\mu_{R}$')
ax1.set_title('99th Percentile \nCatalogue Fraction')
ax2.set_xlabel('Search distance ratio, $\\mu_{S}$')
ax2.set_ylabel('Radius ratio, $\\mu_{R}$')
ax2.set_title('Maximum \nCatalogue Fraction')
sm = cm.ScalarMappable(colors.Normalize(vmin=0.0,vmax=1),cmap=cmap)
plt.colorbar(sm,label='Catalogue fraction',ax=ax2)
#cbar = plt.colorbar(sm, orientation="vertical",\
#    label='Catalogue fraction',cax=cbax)
plt.tight_layout()
#plt.subplots_adjust(top=0.87,bottom=0.195,left=0.099,right=0.839,hspace=0.2,\
#    wspace=0.254)
plt.savefig(figuresFolder + 'randcat_fcat_max_heatmap.pdf')
plt.show()


# Heatmap of the curated splitting counts, and the random fraction:
curatedCountsMean = np.mean(splittingCounts,2)
xx = np.linspace(np.min(distArr) + 1e-3,np.max(distArr)-1e-3,100)
yy = np.linspace(np.min(threshList) + 1e-3,np.max(threshList) - 1e-3,100)
X, Y = np.meshgrid(xx,yy,indexing='ij')
cmap = 'coolwarm'
catPercentile99Rand_Interp = scipy.interpolate.RegularGridInterpolator(\
    (distArr,threshList),catPercentileRand[:,:,-1],method='cubic')
curatedCount_Interp = scipy.interpolate.RegularGridInterpolator(\
    (distArr,threshList),curatedCountsMean,method='cubic')
#combPercentile99Rand_Interp = scipy.interpolate.RegularGridInterpolator(\
#    (distArr,threshList),combPercentileRand[:,:,-1],method='cubic')
fig, ax = plt.subplots(1,2,figsize=(textwidth,0.45*textwidth))
ax1 = ax[0]
ax2 = ax[1]
ax1.imshow(catPercentile99Rand_Interp((X,Y)).T,vmin=0,vmax=1.0,\
    cmap=cmap,extent=(np.min(distArr),np.max(distArr),\
    np.min(threshList),np.max(threshList)),aspect='auto',origin='lower')
ax2.imshow(curatedCount_Interp((X,Y)).T,vmin=0,vmax=3,cmap=cmap,\
    extent=(np.min(distArr),np.max(distArr),\
    np.min(threshList),np.max(threshList)),aspect='auto',origin='lower')
c1 = ax1.contour(X,Y,catPercentile99Rand_Interp((X,Y)),[0.1,0.2,0.3],\
    colors=['k','k','k'])
ax1.clabel(c1,inline=True)
Xi, Yi = np.meshgrid(distArr,threshList,indexing='ij')
ax1.scatter(Xi,Yi,marker='.',color='k')
ax2.scatter(Xi,Yi,marker='.',color='k')
#ax1.contour(X,Y,catPercentile99Rand_Interp((X,Y)))
c2 = ax2.contour(X,Y,curatedCount_Interp((X,Y)),[0,1,5,10],\
    colors=['k','k','k'])
ax2.clabel(c2,inline=True)
sm = cm.ScalarMappable(colors.Normalize(vmin=0.0,vmax=1.0),cmap=cmap)
plt.colorbar(sm,label='Catalogue fraction',ax=ax1)
#ax2.contour(X,Y,catMeanRand_Interp((X,Y)))
ax1.set_xlabel('Search distance ratio, $\\mu_{S}$')
ax1.set_ylabel('Radius ratio, $\\mu_{R}$')
ax1.set_title('99th Percentile \nCatalogue Fraction')
ax2.set_xlabel('Search distance ratio, $\\mu_{S}$')
ax2.set_ylabel('Radius ratio, $\\mu_{R}$')
ax2.set_title('Curated Catalogue Average Split')
sm = cm.ScalarMappable(colors.Normalize(vmin=0.0,vmax=3),cmap=cmap)
plt.colorbar(sm,label='Number of Voids',ax=ax2)
#cbar = plt.colorbar(sm, orientation="vertical",\
#    label='Catalogue fraction',cax=cbax)
plt.tight_layout()
#plt.subplots_adjust(top=0.87,bottom=0.195,left=0.099,right=0.839,hspace=0.2,\
#    wspace=0.254)
plt.savefig(figuresFolder + 'curated_count_heatmap.pdf')
plt.show()


# Heatmaps of MCMC fcat and Ncat:
xx = np.linspace(np.min(distArr) + 1e-3,np.max(distArr)-1e-3,100)
yy = np.linspace(np.min(threshList) + 1e-3,np.max(threshList) - 1e-3,100)
X, Y = np.meshgrid(xx,yy,indexing='ij')
cmap = 'coolwarm'
fcat_Interp = scipy.interpolate.RegularGridInterpolator(\
    (distArr,threshList),catMeanMCMCCutNoBins,method='cubic')
Ncat_Interp = scipy.interpolate.RegularGridInterpolator(\
    (distArr,threshList),catSizeMCMCCutNoBins,method='cubic')
fig, ax = plt.subplots(1,2,figsize=(textwidth,0.43*textwidth))
ax1 = ax[0]
ax2 = ax[1]
ax1.imshow(fcat_Interp((X,Y)).T,vmin=0,vmax=1.0,\
    cmap=cmap,extent=(np.min(distArr),np.max(distArr),\
    np.min(threshList),np.max(threshList)),aspect='auto',origin='lower')
ax2.imshow(Ncat_Interp((X,Y)).T,vmin=0,vmax=np.max(catSizeMCMCCut),cmap=cmap,\
    extent=(np.min(distArr),np.max(distArr),\
    np.min(threshList),np.max(threshList)),aspect='auto',origin='lower')
c1 = ax1.contour(X,Y,fcat_Interp((X,Y)),[0.3,0.5,0.7],\
    colors=['k','k','k'])
ax1.clabel(c1,inline=True)
#ax1.contour(X,Y,catPercentile99Rand_Interp((X,Y)))
c2 = ax2.contour(X,Y,Ncat_Interp((X,Y)),[100,300,500,1000],\
    colors=['k','k','k'])
ax2.clabel(c2,inline=True)
Xi, Yi = np.meshgrid(distArr,threshList,indexing='ij')
ax1.scatter(Xi,Yi,marker='.',color='k')
ax2.scatter(Xi,Yi,marker='.',color='k')
#ax2.contour(X,Y,catMeanRand_Interp((X,Y)))
ax1.set_xlabel('Search distance ratio, $\\mu_{S}$')
ax1.set_ylabel('Radius ratio, $\\mu_{R}$')
ax1.set_title('Mean MCMC Catalogue \nFraction after cut')
sm = cm.ScalarMappable(colors.Normalize(vmin=0.0,vmax=1.0),cmap=cmap)
plt.colorbar(sm,label='Catalogue fraction',ax=ax1)
ax2.set_xlabel('Search distance ratio, $\\mu_{S}$')
ax2.set_ylabel('Radius ratio, $\\mu_{R}$')
ax2.set_title('MCMC number of voids \nafter cut')
sm = cm.ScalarMappable(colors.Normalize(vmin=0.0,vmax=np.max(catSizeMCMCCutNoBins)),\
    cmap=cmap)
plt.colorbar(sm,label='Number of Voids',ax=ax2)
#sm = cm.ScalarMappable(colors.Normalize(vmin=0.0,vmax=1.0),cmap=cmap)
#cbax = fig.add_axes([0.87,0.2,0.02,0.68])
#cbar = plt.colorbar(sm, orientation="vertical",\
#    label='Catalogue fraction Or $N_{cat}/\\mathrm{max}(N_{cat})$',cax=cbax)
#plt.tight_layout()
plt.subplots_adjust(top=0.839,bottom=0.219,left=0.099,right=0.924,hspace=0.2,\
    wspace=0.394)
plt.savefig(figuresFolder + 'mcmccat_fcat_Ncat_heatmap.pdf')
plt.show()


# Heatmaps of Randoms Ncat and NcatCut:
xx = np.linspace(np.min(distArr) + 1e-3,np.max(distArr)-1e-3,100)
yy = np.linspace(np.min(threshList) + 1e-3,np.max(threshList) - 1e-3,100)
X, Y = np.meshgrid(xx,yy,indexing='ij')
cmap = 'coolwarm'
NcatRand_Interp = scipy.interpolate.RegularGridInterpolator(\
    (distArr,threshList),catSizeRand,method='cubic')
NcatRandCut_Interp = scipy.interpolate.RegularGridInterpolator(\
    (distArr,threshList),catSizeRandCut,method='cubic')
fig, ax = plt.subplots(1,2,figsize=(textwidth,0.43*textwidth))
ax1 = ax[0]
ax2 = ax[1]
ax1.imshow(NcatRand_Interp((X,Y)).T,\
    cmap=cmap,extent=(np.min(distArr),np.max(distArr),\
    np.min(threshList),np.max(threshList)),aspect='auto',origin='lower',\
    vmin=0,vmax=np.max(catSizeRand))
ax2.imshow(NcatRandCut_Interp((X,Y)).T,vmin=0,vmax=np.max(catSizeRandCut),cmap=cmap,\
    extent=(np.min(distArr),np.max(distArr),\
    np.min(threshList),np.max(threshList)),aspect='auto',origin='lower')
c1 = ax1.contour(X,Y,NcatRand_Interp((X,Y)),[0,100,300,500],\
    colors=['k','k','k','k'])
ax1.clabel(c1,inline=True)
#ax1.contour(X,Y,catPercentile99Rand_Interp((X,Y)))
c2 = ax2.contour(X,Y,NcatRandCut_Interp((X,Y)),[1,3,5],\
    colors=['k','k','k'])
ax2.clabel(c2,inline=True)
Xi, Yi = np.meshgrid(distArr,threshList,indexing='ij')
ax1.scatter(Xi,Yi,marker='.',color='k')
ax2.scatter(Xi,Yi,marker='.',color='k')
#ax2.contour(X,Y,catMeanRand_Interp((X,Y)))
ax1.set_xlabel('Search distance ratio, $\\mu_{S}$')
ax1.set_ylabel('Radius ratio, $\\mu_{R}$')
ax1.set_title('All voids in random catalogue')
sm = cm.ScalarMappable(colors.Normalize(vmin=0.0,vmax=np.max(catSizeRand)),\
    cmap=cmap)
plt.colorbar(sm,label='Number of Voids',ax=ax1)
ax2.set_xlabel('Search distance ratio, $\\mu_{S}$')
ax2.set_ylabel('Radius ratio, $\\mu_{R}$')
ax2.set_title('Random catalogue voids \nsurviving cut')
sm = cm.ScalarMappable(colors.Normalize(vmin=0.0,vmax=np.max(catSizeRandCut)),\
    cmap=cmap)
plt.colorbar(sm,label='Number of Voids',ax=ax2)
#sm = cm.ScalarMappable(colors.Normalize(vmin=0.0,vmax=1.0),cmap=cmap)
#cbax = fig.add_axes([0.87,0.2,0.02,0.68])
#cbar = plt.colorbar(sm, orientation="vertical",\
#    label='Catalogue fraction Or $N_{cat}/\\mathrm{max}(N_{cat})$',cax=cbax)
plt.tight_layout()
plt.savefig(figuresFolder + 'randcat_Ncat_NcatCut_heatmap.pdf')
plt.show()








# Heatmaps of MCMC fcat and Ncat:
xx = np.linspace(np.min(distArr) + 1e-3,np.max(distArr)-1e-3,100)
yy = np.linspace(np.min(threshList) + 1e-3,np.max(threshList) - 1e-3,100)
X, Y = np.meshgrid(xx,yy,indexing='ij')
cmap = 'coolwarm'
fcat_Interp = scipy.interpolate.RegularGridInterpolator(\
    (distArr,threshList),combMeanMCMCCut,method='cubic')
Ncat_Interp = scipy.interpolate.RegularGridInterpolator(\
    (distArr,threshList),catSizeMCMCCutComb,method='cubic')
fig, ax = plt.subplots(1,2,figsize=(textwidth,0.43*textwidth))
ax1 = ax[0]
ax2 = ax[1]
ax1.imshow(fcat_Interp((X,Y)).T,vmin=0,vmax=1.0,\
    cmap=cmap,extent=(np.min(distArr),np.max(distArr),\
    np.min(threshList),np.max(threshList)),aspect='auto',origin='lower')
ax2.imshow(Ncat_Interp((X,Y)).T,vmin=0,vmax=np.max(catSizeMCMCCut),cmap=cmap,\
    extent=(np.min(distArr),np.max(distArr),\
    np.min(threshList),np.max(threshList)),aspect='auto',origin='lower')
c1 = ax1.contour(X,Y,fcat_Interp((X,Y)),[0.1,0.3,0.5,0.7],\
    colors=['k','k','k'])
ax1.clabel(c1,inline=True)
#ax1.contour(X,Y,catPercentile99Rand_Interp((X,Y)))
c2 = ax2.contour(X,Y,Ncat_Interp((X,Y)),[100,200,300,500],\
    colors=['k','k','k'])
ax2.clabel(c2,inline=True)
Xi, Yi = np.meshgrid(distArr,threshList,indexing='ij')
ax1.scatter(Xi,Yi,marker='.',color='k')
ax2.scatter(Xi,Yi,marker='.',color='k')
#ax2.contour(X,Y,catMeanRand_Interp((X,Y)))
ax1.set_xlabel('Search distance ratio, $\\mu_{S}$')
ax1.set_ylabel('Radius ratio, $\\mu_{R}$')
ax1.set_title('Mean MCMC Combinatoric \nFraction after cut')
sm = cm.ScalarMappable(colors.Normalize(vmin=0.0,vmax=1.0),cmap=cmap)
plt.colorbar(sm,label='Combinatoric fraction',ax=ax1)
ax2.set_xlabel('Search distance ratio, $\\mu_{S}$')
ax2.set_ylabel('Radius ratio, $\\mu_{R}$')
ax2.set_title('MCMC number of voids \nafter cut')
sm = cm.ScalarMappable(colors.Normalize(vmin=0.0,vmax=np.max(catSizeMCMCCutNoBins)),\
    cmap=cmap)
plt.colorbar(sm,label='Number of Voids',ax=ax2)
#sm = cm.ScalarMappable(colors.Normalize(vmin=0.0,vmax=1.0),cmap=cmap)
#cbax = fig.add_axes([0.87,0.2,0.02,0.68])
#cbar = plt.colorbar(sm, orientation="vertical",\
#    label='Catalogue fraction Or $N_{cat}/\\mathrm{max}(N_{cat})$',cax=cbax)
#plt.tight_layout()
plt.subplots_adjust(top=0.839,bottom=0.219,left=0.099,right=0.924,hspace=0.2,\
    wspace=0.394)
plt.savefig(figuresFolder + 'mcmccat_fcomb_Ncat_heatmap.pdf')
plt.show()



# Optimal Ncat:

optimalParamNcat = scipy.optimize.minimize(lambda x: -Ncat_Interp(x),\
    np.array([0.15,0.87]),bounds=(tools.minmax(distArr),\
    tools.minmax(threshList)))
# Optimal number of voids at [0.14237877, 0.85891552]

# Scatter plot of fcat vs Ncat:
plt.clf()
fig, ax = plt.subplots()
cmap='plasma'
Xi, Yi = np.meshgrid(distArr,threshList,indexing='ij')
sm = cm.ScalarMappable(colors.Normalize(vmin=np.min(Yi),vmax=np.max(Yi)),\
    cmap=cmap)
ax.scatter(catMeanMCMCCut[:5,:],catSizeMCMCCut[:5,:],c=Yi[:5,:],cmap=cmap,\
    norm=colors.Normalize(vmin=np.min(Yi),vmax=np.max(Yi)))
plt.colorbar(sm,label='Radius ratio, $\mu_R$')
ax.set_xlabel('Mean MCMC catalogue fraction after cut')
ax.set_ylabel('Number of voids in MCMC catalogue after cut, ' + \
    '$N_{\\mathrm{cat}}$')
#ax.set_xlim([0,1])
#ax.set_ylim([0,400])
plt.savefig(figuresFolder + "fcat_vs_Ncat_scatter_binned_filter.pdf")
plt.show()

# Plot of the mean catalogue fraction after cuts vs random fraction:
survivingFraction = np.zeros(catSizeMCMCCut.shape)
nz = np.where(np.sum(catSizeBinsMCMC,2) != 0)
survivingFraction[nz] = np.sum(catSizeBinsMCMCCut,2)[nz]/\
    np.sum(catSizeBinsMCMC,2)[nz]
euclideanDistance = np.sqrt((catMeanMCMCCut-1)**2 + \
    (survivingFraction-1)**2)/np.sqrt(2)

survivingFractionComb = np.zeros(catSizeMCMCCut.shape)
nz = np.where(np.sum(catSizeBinsMCMC,2) != 0)
survivingFractionComb[nz] = np.sum(catSizeBinsMCMCCutComb,2)[nz]/\
    np.sum(catSizeBinsMCMC,2)[nz]
euclideanDistanceComb = np.sqrt((combMeanMCMCCutComb-1)**2 + \
    (survivingFractionComb-1)**2)/np.sqrt(2)

euclidInterp = scipy.interpolate.RegularGridInterpolator(\
    (distArr,threshList),euclideanDistance,method='cubic')
sfInterp = scipy.interpolate.RegularGridInterpolator(\
    (distArr,threshList),survivingFraction,method='cubic')

euclidInterpComb = scipy.interpolate.RegularGridInterpolator(\
    (distArr,threshList),euclideanDistanceComb,method='cubic')
sfInterpComb = scipy.interpolate.RegularGridInterpolator(\
    (distArr,threshList),survivingFractionComb,method='cubic')

# Heatmaps of MCMC f_{survive} and Ncat:
xx = np.linspace(np.min(distArr) + 1e-3,np.max(distArr)-1e-3,100)
yy = np.linspace(np.min(threshList) + 1e-3,np.max(threshList) - 1e-3,100)
X, Y = np.meshgrid(xx,yy,indexing='ij')
cmap = 'coolwarm'
fcat_Interp = scipy.interpolate.RegularGridInterpolator(\
    (distArr,threshList),catMeanMCMCCutNoBins,method='cubic')
fcomb_Interp = scipy.interpolate.RegularGridInterpolator(\
    (distArr,threshList),combMeanMCMCCutNoBins,method='cubic')
fig, ax = plt.subplots(1,2,figsize=(textwidth,0.43*textwidth))
ax1 = ax[0]
ax2 = ax[1]
ax1.imshow(fcomb_Interp((X,Y)).T,vmin=0,vmax=1.0,\
    cmap=cmap,extent=(np.min(distArr),np.max(distArr),\
    np.min(threshList),np.max(threshList)),aspect='auto',origin='lower')
ax2.imshow(sfInterpComb((X,Y)).T,vmin=0,vmax=1.0,cmap=cmap,\
    extent=(np.min(distArr),np.max(distArr),\
    np.min(threshList),np.max(threshList)),aspect='auto',origin='lower')
c1 = ax1.contour(X,Y,fcomb_Interp((X,Y)),[0.1,0.2,0.3],\
    colors=['k','k','k'])
#c1 = ax1.contour(X,Y,fcat_Interp((X,Y)),[0.3,0.5,0.7],\
#    colors=['k','k','k'])
ax1.clabel(c1,inline=True)
#ax1.contour(X,Y,catPercentile99Rand_Interp((X,Y)))
c2 = ax2.contour(X,Y,sfInterpComb((X,Y)),[0.3,0.5,0.7],\
    colors=['k','k','k'])
ax2.clabel(c2,inline=True)
Xi, Yi = np.meshgrid(distArr,threshList,indexing='ij')
ax1.scatter(Xi,Yi,marker='.',color='k')
ax2.scatter(Xi,Yi,marker='.',color='k')
#ax2.contour(X,Y,catMeanRand_Interp((X,Y)))
ax1.set_xlabel('Search distance ratio, $\\mu_{S}$')
ax1.set_ylabel('Radius ratio, $\\mu_{R}$')
#ax1.set_title('Mean MCMC Catalogue \nFraction after cut')
ax1.set_title('Mean MCMC Combinatoric \nFraction after cut')
sm = cm.ScalarMappable(colors.Normalize(vmin=0.0,vmax=0.3),cmap=cmap)
plt.colorbar(sm,label='Combinatoric fraction',ax=ax1)
ax2.set_xlabel('Search distance ratio, $\\mu_{S}$')
ax2.set_ylabel('Radius ratio, $\\mu_{R}$')
ax2.set_title('Fraction of MCMC voids\n surviving cut, ' + \
    '$N_{\\mathrm{cat}}/N_{\\mathrm{cat,uncut}}$')
sm = cm.ScalarMappable(colors.Normalize(vmin=0.0,vmax=1.0),\
    cmap=cmap)
plt.colorbar(sm,label='$N_{\\mathrm{cat}}/N_{\\mathrm{cat,uncut}}$',ax=ax2)
#sm = cm.ScalarMappable(colors.Normalize(vmin=0.0,vmax=1.0),cmap=cmap)
#cbax = fig.add_axes([0.87,0.2,0.02,0.68])
#cbar = plt.colorbar(sm, orientation="vertical",\
#    label='Catalogue fraction Or $N_{cat}/\\mathrm{max}(N_{cat})$',cax=cbax)
#plt.tight_layout()
plt.subplots_adjust(top=0.839,bottom=0.219,left=0.099,right=0.924,hspace=0.2,\
    wspace=0.394)
plt.savefig(figuresFolder + 'mcmccat_fsur_fcomb_heatmap.pdf')
plt.show()




# Scatter plot of f_sur vs f_cat:
plt.clf()
fig, ax = plt.subplots()
cmap='plasma'
Xi, Yi = np.meshgrid(distArr,threshList,indexing='ij')
sm = cm.ScalarMappable(colors.Normalize(vmin=np.min(Yi),vmax=np.max(Yi)),\
    cmap=cmap)
ax.scatter(catMeanMCMCCut[:5,:],survivingFraction[:5,:],c=Yi[:5,:],cmap=cmap,\
    norm=colors.Normalize(vmin=np.min(Yi),vmax=np.max(Yi)))
plt.colorbar(sm,label='Radius ratio, $\mu_R$')
ax.set_xlabel('Mean MCMC catalogue fraction after cut')
ax.set_ylabel('Fraction of voids surviving cut, ' + \
    '$N_{\\mathrm{cat}}/N_{\\mathrm{cat,uncut}}$')
#ax.set_xlim([0,1])
#ax.set_ylim([0,400])
plt.savefig(figuresFolder + "fcat_vs_fsurv_scatter.pdf")
plt.show()




# Heatmap of Euclidean distance:
fig, ax = plt.subplots()
ax.imshow(euclidInterp((X,Y)).T,vmin=0,vmax=1.0,\
    cmap=cmap,extent=(np.min(distArr),np.max(distArr),\
    np.min(threshList),np.max(threshList)),aspect='auto',origin='lower')
c1 = ax.contour(X,Y,euclidInterp((X,Y)),[0.3,0.5,0.7],\
    colors=['k','k','k'])
ax.clabel(c1,inline=True)
ax.set_xlabel('Search distance ratio, $\\mu_{S}$')
ax.set_ylabel('Radius ratio, $\\mu_{R}$')
#ax.set_title('$\\sqrt{((f_{\\mathrm{cat,MCMC}}-1)^2 + ' + \
#    'f_{\\mathrm{cat,rand}}^2)/2}$')
ax.set_title('$\\sqrt{((f_{\\mathrm{cat,MCMC}}-1)^2 + ' + \
    '(N_{\\mathrm{cat}}/N_{\\mathrm{cat,uncut}} - 1)^2)/2}$')
sm = cm.ScalarMappable(colors.Normalize(vmin=0.0,vmax=1.0),cmap=cmap)
plt.colorbar(sm,label='Euclidean Distance',ax=ax)
plt.savefig(figuresFolder + 'euclidean_distance.pdf')
plt.show()


# Heatmap of Euclidean distance:
fig, ax = plt.subplots()
ax.imshow(euclidInterpComb((X,Y)).T,vmin=0,vmax=1.0,\
    cmap=cmap,extent=(np.min(distArr),np.max(distArr),\
    np.min(threshList),np.max(threshList)),aspect='auto',origin='lower')
c1 = ax.contour(X,Y,euclidInterpComb((X,Y)),[0.3,0.4,0.5,0.6,0.7,0.8,0.9],\
    colors=['k','k','k','k','k','k','k'])
ax.clabel(c1,inline=True)
ax.set_xlabel('Search distance ratio, $\\mu_{S}$')
ax.set_ylabel('Radius ratio, $\\mu_{R}$')
#ax.set_title('$\\sqrt{((f_{\\mathrm{cat,MCMC}}-1)^2 + ' + \
#    'f_{\\mathrm{cat,rand}}^2)/2}$')
ax.set_title('$\\sqrt{((f_{\\mathrm{cat,MCMC}}-1)^2 + ' + \
    '(N_{\\mathrm{cat}}/N_{\\mathrm{cat,uncut}} - 1)^2)/2}$')
sm = cm.ScalarMappable(colors.Normalize(vmin=0.0,vmax=1.0),cmap=cmap)
plt.colorbar(sm,label='Euclidean Distance',ax=ax)
plt.savefig(figuresFolder + 'euclidean_distance_comb.pdf')
plt.show()



# Plot of the catalogue fraction of randoms:
# Plot interpolated surface:
xx = np.linspace(np.min(distArr) + 1e-3,np.max(distArr)-1e-3,100)
yy = np.linspace(np.min(threshList) + 1e-3,np.max(threshList) - 1e-3,100)
X, Y = np.meshgrid(xx,yy,indexing='ij')
Xi, Yi = np.meshgrid(distArr,threshList,indexing='ij')
catMeanMCMCCut_Interp = scipy.interpolate.RegularGridInterpolator(\
    (distArr,threshList),catMeanMCMCCut,method='cubic')
catSizeMCMCCut_Interp = scipy.interpolate.RegularGridInterpolator(\
    (distArr,threshList),catSizeMCMCCut,method='cubic')
fig = plt.figure()
ax1 = fig.add_subplot(1,2,1,projection='3d')
ax2 = fig.add_subplot(1,2,2,projection='3d')
ax1.plot_wireframe(X,Y,catMeanMCMCCut_Interp((X,Y)),alpha=0.5,color='grey')
ax2.plot_wireframe(X,Y,catSizeMCMCCut_Interp((X,Y)),alpha=0.5,color='grey')
ax1.scatter3D(Xi,Yi,catMeanMCMCCut,color=seabornColormap[0],label = "Samples")
ax2.scatter3D(Xi,Yi,catSizeMCMCCut,color=seabornColormap[0],label = "Samples")
ax1.set_xlabel('Search distance ratio, $\\mu_{S}$')
ax1.set_ylabel('Radius ratio, $\\mu_{R}$')
ax1.set_zlabel('Catalogue fraction')
ax2.set_xlabel('Search distance ratio, $\\mu_{S}$')
ax2.set_ylabel('Radius ratio, $\\mu_{R}$')
ax2.set_zlabel('Number of voids')
ax1.set_title('Mean Catalogue fraction (after cut)')
ax2.set_title('Catalogue Size (after cut)')
plt.legend()
#plt.tight_layout()
plt.savefig(figuresFolder + 'f_cat_and_Ncat.pdf')
plt.show()





#-------------------------------------------------------------------------------
# CATALOGUE CONSISTENCY CHECKS


# Consistency check:
centresListFinal = []
antihalosListFinal = []
cataloguesFinal = []
filtersFinal = []
scaleFiltersFinal = []
divisionsList = ['batch5-2/','batch10-1/','all_samples/']
finalCatFracList = []
combinedFiltersList = []
snrLists = []
snrThresh = 10
allCatData = []
for name in divisionsList:
    base_folder = 'borg-antihalos_paper_figures/' + name
    [massListMean,combinedFilter135,combinedFilter,rBinStackCentresCombined,\
            nbarjSepStackCombined,sigmaSepStackCombined,\
            nbarjAllStackedUnCombined,sigmaAllStackedUnCombined,nbar,rMin2,\
            mMin2,mMax2,nbarjSepStackUn,sigmaSepStackUn,\
            rBinStackCentres,nbarjSepStack,\
            sigmaSepStack,nbarjAllStackedUn,sigmaAllStackedUn,\
            nbarjSepStackUn,sigmaSepStackUn] = tools.loadPickle(\
                base_folder + "finalCatData.p")
    [finalCatOpt,shortHaloListOpt,twoWayMatchListOpt,finalCandidatesOpt,\
        finalRatiosOpt,finalDistancesOpt,allCandidatesOpt,candidateCountsOpt,\
        allRatiosOpt,finalCombinatoricFracOpt,finalCatFracOpt,\
        alreadyMatched] = pickle.load(\
            open(base_folder + "catalogue_all_data.p","rb"))
    catData = np.load(base_folder + "catalogue_data.npz")
    [scaleBins,percentilesCat,percentilesComb,\
            meanCatFrac,stdErrCatFrac,meanCombFrac,stdErrCombFrac,\
            radiiListMean,massListMean,massListSigma,radiiListSigma,\
            massBins,radBins,scaleFilter] = tools.loadPickle(\
            base_folder + "catalogue_scale_cut_data.p")
    snrLists.append(catData['snr'])
    centresListFinal.append(catData['centres'][combinedFilter135])
    antihalosListFinal.append(finalCatOpt[combinedFilter135])
    cataloguesFinal.append(finalCatOpt)
    filtersFinal.append(combinedFilter135)
    scaleFiltersFinal.append(scaleFilter)
    finalCatFracList.append(finalCatFracOpt)
    allCatData.append(catData)

compareLists = [antihalosListFinal[0],antihalosListFinal[1][:,5:],\
    antihalosListFinal[2][:,5:10]]



def compareAHLists(list1,list2,enforceExact=True):
    result = np.zeros(len(list1),dtype=bool)
    for k in range(0,len(list1)):
        if enforceExact:
            result[k] = np.any(np.all(list1[k] == list2,1))
        else:
            result[k] = np.any(np.any((list1[k] == list2) & (list1[k] >= 0),1))
    return result



successValues = [compareAHLists(compareLists[0],compareLists[1],\
    enforceExact=False),
    compareAHLists(compareLists[0],compareLists[2],\
    enforceExact=False)]

backwardsValues = [compareAHLists(compareLists[1],compareLists[0],\
    enforceExact=False),
    compareAHLists(compareLists[2],compareLists[0],\
    enforceExact=False)]

successRate5 = [np.sum(val)/len(val) for val in successValues]
backwardsRate5 = [np.sum(val)/len(val) for val in backwardsValues]

successValues10 = compareAHLists(antihalosListFinal[1],\
    antihalosListFinal[2][:,0:10],enforceExact=False)
backwardsValues10 = compareAHLists(antihalosListFinal[2][:,0:10],\
    antihalosListFinal[1],enforceExact=False)

successRate10 = np.sum(successValues10)/len(successValues10)
backwardsRate10 = np.sum(backwardsValues10)/len(backwardsValues10)

# Test the failed 20->5 matches:
# Get the 20-sample voids that don't appear in the 5-sample set:
failed20in5 = compareLists[2][np.logical_not(backwardsValues[1])]
# Find the corresponding voids that made it into the pre-cut 5-sample catalogue:
findFailed20In5 = compareAHLists(cataloguesFinal[0],failed20in5)
# Compute the fraction that would have made the cut if we allowed chain-matches:
thresholdList = np.zeros(len(combinedFilter135))
for filt, thresh in zip(scaleFiltersFinal[2],percentilesCat):
    thresholdList[filt] = thresh

# Relevant thresholds for each failed 20->5 void:
thresholdListCut = thresholdList[combinedFilter135]
catFracsFailed20in5 = np.sum(failed20in5 >= 0,1)/failed20in5.shape[1]
wouldSucceed = (catFracsFailed20in5 >= \
    thresholdListCut[np.logical_not(backwardsValues[1])])

# Play the same game with 20->10:
failed20in10 = antihalosListFinal[2][:,0:10][np.logical_not(backwardsValues10)]
# Find the corresponding voids that made it into the pre-cut 5-sample catalogue:
findFailed20In10 = compareAHLists(cataloguesFinal[1],failed20in10)
catFracsFailed20in10 = np.sum(failed20in10 >= 0,1)/failed20in10.shape[1]
wouldSucceed = (catFracsFailed20in10 >= \
    thresholdListCut[np.logical_not(backwardsValues10)])

# Compute expected failure rates per void, for each void in the 20-sample
# catalogue:
failureChance = np.zeros(len(finalCatFracOpt[combinedFilter135]))
nSelectCats = 10
thresholdList = np.zeros(len(filtersFinal[2]))
for filt, thresh in zip(scaleFiltersFinal[2],percentilesCat):
    thresholdList[filt] = thresh

# Relevant thresholds for each failed 20->5 void:
thresholdListCut = thresholdList[filtersFinal[2]]
for k in range(0,len(failureChance)):
    thresh_k = thresholdListCut[k]
    cat_k = finalCatFracList[2][filtersFinal[2]][k]
    failureChance[k] = scipy.stats.binom.cdf(nSelectCats*thresh_k,nSelectCats,cat_k)

successChance = 1.0 - failureChance

# Expected fraction that would make it into the 10-sample catalogue:
survival20to10 = np.sum(1.0 - failureChance)/len(failureChance)

# Non-trivial to compute the variance here, since the sum of binomials with
# different probabilities is not binomial. We can monte-carlo it though:

def monteCarloSums(successChances,ntries=1000,seed=None,returnSamples=False):
    # Generate 1 or zero based on the success chances, ntries times:
    if seed is not None:
        np.random.seed(seed)
    samples = np.random.binomial(1,successChances,\
        size=(ntries,len(successChances)))
    sampleSums = np.sum(samples,1)
    meanSum = np.mean(sampleSums)
    varSum = np.var(sampleSums)
    if returnSamples:
        return sampleSums
    else:
        return [meanSum,varSum]


fig, axAll = plt.subplots(1,3,figsize=(1.5*textwidth,1.5*0.3*textwidth))

withChain=True
chainFraction = np.sum(failed20in10 >= 0,1)/failed20in10.shape[1]
chainSuccess = (chainFraction > \
    thresholdListCut[np.logical_not(backwardsValues10)])

sampleSums = monteCarloSums(successChance,ntries=1000,seed=1000,\
    returnSamples=True)
[meanSum,varSum] = monteCarloSums(successChance,ntries=1000,seed=1000)

axAll[0].hist(sampleSums/len(successChance),color=seabornColormap[0],alpha=0.5,\
    label='Monte carlo samples',density=False,bins=7)
axAll[0].axvline(meanSum/len(successChance),color='k',linestyle='--',\
    label='Monte-carlo mean \nsurvival fraction')
axAll[0].axvline((len(successChance) - len(failed20in10))/len(successChance),\
    color='k',linestyle=':',label='Observed survival \nfraction')
if withChain:
    axAll[0].axvline((len(successChance) - len(failed20in10) + np.sum(chainSuccess))/\
        len(successChance),color='grey',linestyle=':',\
        label='Observed survival \nfraction (with \nchain-successes)')

axAll[0].set_xlabel('Survival Fraction',fontsize=8)
axAll[0].set_ylabel('Monte-carlo samples',fontsize=8)
#axAll[0].legend()
axAll[0].set_title('20 to 10 samples.',fontsize=8)
#if withChain:
#    plt.savefig(figuresFolder + "monte_carlo_survival_20_to_10_with_chain.pdf")
#else:
#    plt.savefig(figuresFolder + "monte_carlo_survival_20_to_10.pdf")

#plt.show()


# We should do the same for the 20->5 case:
failureChance = np.zeros(len(finalCatFracList[2][filtersFinal[2]]))
nSelectCats = 5
thresholdList = np.zeros(len(filtersFinal[2]))
for filt, thresh in zip(scaleFiltersFinal[2],percentilesCat):
    thresholdList[filt] = thresh

# Relevant thresholds for each failed 20->5 void:
thresholdListCut = thresholdList[filtersFinal[2]]
for k in range(0,len(failureChance)):
    thresh_k = thresholdListCut[k]
    cat_k = finalCatFracList[2][filtersFinal[2]][k]
    failureChance[k] = scipy.stats.binom.cdf(nSelectCats*thresh_k,nSelectCats,cat_k)

successChance = 1.0 - failureChance
chainFraction = np.sum(failed20in5 >= 0,1)/failed20in5.shape[1]
chainSuccess = (chainFraction > \
    thresholdListCut[np.logical_not(backwardsValues[1])])

sampleSums = monteCarloSums(successChance,ntries=1000,seed=1000,\
    returnSamples=True)
[meanSum,varSum] = monteCarloSums(successChance,ntries=1000,seed=1000)

axAll[1].hist(sampleSums/len(successChance),color=seabornColormap[0],alpha=0.5,\
    label='Monte carlo samples',density=False,bins=10)
axAll[1].axvline(meanSum/len(successChance),color='k',linestyle='--',\
    label='Monte-carlo mean \nsurvival fraction')
axAll[1].axvline((len(successChance) - len(failed20in5))/len(successChance),\
    color='k',linestyle=':',label='Observed survival \nfraction')
if withChain:
    axAll[1].axvline((len(successChance) - len(failed20in5) + np.sum(chainSuccess))/\
        len(successChance),color='grey',linestyle=':',\
        label='Observed survival \nfraction (with \nchain-successes)')

axAll[1].set_xlabel('Survival Fraction',fontsize=8)
axAll[1].set_ylabel('Monte-carlo samples',fontsize=8)
#axAll[1].legend()
axAll[1].set_title('20 to 5 samples.',fontsize=8)
#if withChain:
#    plt.savefig(figuresFolder + "monte_carlo_survival_20_to_5_with_chain.pdf")
#else:
#    plt.savefig(figuresFolder + "monte_carlo_survival_20_to_5.pdf")

#plt.show()

# For completeness, let's do 10->5 as well?
failed10in5 = antihalosListFinal[1][:,5:10][np.logical_not(backwardsValues[0])]
failureChance = np.zeros(len(finalCatFracList[1][filtersFinal[1]]))
nSelectCats = 5
thresholdList = np.zeros(len(filtersFinal[1]))
for filt, thresh in zip(scaleFiltersFinal[1],percentilesCat):
    thresholdList[filt] = thresh

# Relevant thresholds for each failed 20->5 void:
thresholdListCut = thresholdList[filtersFinal[1]]

for k in range(0,len(failureChance)):
    thresh_k = thresholdListCut[k]
    cat_k = finalCatFracList[1][filtersFinal[1]][k]
    failureChance[k] = scipy.stats.binom.cdf(nSelectCats*thresh_k,nSelectCats,cat_k)

successChance = 1.0 - failureChance
chainFraction = np.sum(failed10in5 >= 0,1)/failed10in5.shape[1]
chainSuccess = (chainFraction > \
    thresholdListCut[np.logical_not(backwardsValues[0])])

sampleSums = monteCarloSums(successChance,ntries=1000,seed=1000,\
    returnSamples=True)
[meanSum,varSum] = monteCarloSums(successChance,ntries=1000,seed=1000)

axAll[2].hist(sampleSums/len(successChance),color=seabornColormap[0],alpha=0.5,\
    label='Monte carlo samples',density=False,bins=10)
axAll[2].axvline(meanSum/len(successChance),color='k',linestyle='--',\
    label='Monte-carlo mean \nsurvival fraction')
axAll[2].axvline((len(successChance) - len(failed10in5))/len(successChance),\
    color='k',linestyle=':',label='Observed survival \nfraction')
if withChain:
    axAll[2].axvline((len(successChance) - len(failed10in5) + np.sum(chainSuccess))/\
        len(successChance),color='grey',linestyle=':',\
        label='Observed survival \nfraction (with \nchain-successes)')

axAll[2].set_xlabel('Survival Fraction',fontsize=8)
axAll[2].set_ylabel('Monte-carlo samples',fontsize=8)
axAll[2].legend(prop={"size":8,"family":"serif"},frameon=False)
axAll[2].set_title('10 to 5 samples.',\
    fontsize=8)
#if withChain:
#    plt.savefig(figuresFolder + "monte_carlo_survival_10_to_5_with_chain.pdf")
#else:
#    plt.savefig(figuresFolder + "monte_carlo_survival_10_to_5.pdf")

plt.tight_layout()
if withChain:
    plt.savefig(figuresFolder + "monte_carlo_survival_with_chain.pdf")
else:
    plt.savefig(figuresFolder + "monte_carlo_survival.pdf")

plt.show()

# False positives:

catFracFilter = [[catFrac > thresh for thresh in percentilesCat] \
    for catFrac in finalCatFracList]

distanceFilters = [np.sqrt(np.sum(catData['centres']**2,1)) < 135 \
    for catData in allCatData]

cutFiltersList = [[scaleFilter & (snrLists[k] > snrThresh) & catFilter & \
    distanceFilters[k] for scaleFilter, catFilter in \
    zip(scaleFiltersFinal[k],catFracFilter[k])]\
    for k in range(0,3)]




removedFiltersList = [[scaleFilter & (snrLists[k] > snrThresh) & \
    np.logical_not(catFilter) & distanceFilters[k] \
    for scaleFilter, catFilter in \
    zip(scaleFiltersFinal[k],catFracFilter[k])]\
    for k in range(0,3)]

massBinCentres = plot.binCentres(scaleBins)
counts = np.array([[np.sum(l) for l in cutFiltersList[k]] for k in range(0,3)]).T
plt.semilogx(massBinCentres,counts,label=['5 samples','10 samples','20 samples'])
plt.xlabel('Mass [$M_{\\odot}h^{-1}$]')
plt.ylabel('Number of Voids')
plt.title('Combined Catalogues')
plt.legend(frameon=False)
plt.savefig(figuresFolder + "mass_counts.pdf")
plt.show()

# Need to figure out which voids that were rejected at 20, have a
# chance of appearing at 10:


failedVoids20 = np.zeros(removedFiltersList[2][0].shape,dtype=bool)
for k in range(0,len(removedFiltersList[2])):
    failedVoids20 = failedVoids20 | removedFiltersList[2][k]

catProbabilities = finalCatFracList[2][failedVoids20]
thresholdList = np.zeros(len(filtersFinal[2]))
for filt, thresh in zip(scaleFiltersFinal[2],percentilesCat):
    thresholdList[filt] = thresh

thresholds = thresholdList[failedVoids20]

nSelectCats=10
failureChance = np.zeros(catProbabilities.shape)
for k in range(0,len(catProbabilities)):
    thresh_k = thresholds[k]
    cat_k = catProbabilities[k]
    failureChance[k] = scipy.stats.binom.cdf(nSelectCats*thresh_k,nSelectCats,cat_k)

successChance = 1.0 - failureChance

sampleSums = monteCarloSums(successChance,ntries=1000,seed=1000,\
    returnSamples=True)
[meanSum,varSum] = monteCarloSums(successChance,ntries=1000,seed=1000)


fig, ax = plt.subplots(1,3,figsize=(1.5*textwidth,1.5*0.3*textwidth))

ax[0].hist(sampleSums/len(compareLists[1]),color=seabornColormap[0],alpha=0.5,\
    label='Monte carlo samples',density=False,bins=7)
ax[0].axvline(meanSum/len(compareLists[1]),color='k',linestyle='--',\
    label='Monte-carlo mean \nfalse positive')
ax[0].axvline((len(compareLists[1]) - np.sum(successValues10))/\
    len(compareLists[1]),\
    color='k',linestyle=':',label='Observed false \npositive')

ax[0].set_xlabel('False-positive fraction',fontsize=8)
ax[0].set_ylabel('Monte-carlo samples',fontsize=8)
deviation = (len(compareLists[1]) - np.sum(successValues10) - meanSum)/\
    np.sqrt(varSum)
#plt.legend(prop={"size":8,"family":"serif"},frameon=False)
ax[0].set_title('20 to 10 samples ($' + ("%.2g" % deviation) + "\\sigma$)",\
    fontsize=8)
#plt.savefig(figuresFolder + "false_positive_fraction.pdf")
#plt.show()


nSelectCats=5
failureChance = np.zeros(catProbabilities.shape)
for k in range(0,len(catProbabilities)):
    thresh_k = thresholds[k]
    cat_k = catProbabilities[k]
    failureChance[k] = scipy.stats.binom.cdf(nSelectCats*thresh_k,nSelectCats,cat_k)

successChance = 1.0 - failureChance

sampleSums = monteCarloSums(successChance,ntries=1000,seed=1000,\
    returnSamples=True)
[meanSum,varSum] = monteCarloSums(successChance,ntries=1000,seed=1000)

ax[1].hist(sampleSums/len(compareLists[0]),color=seabornColormap[0],alpha=0.5,\
    label='Monte carlo samples',density=False,bins=10)
ax[1].axvline(meanSum/len(compareLists[0]),color='k',linestyle='--',\
    label='Monte-carlo mean \nfalse positive')
ax[1].axvline((len(compareLists[0]) - np.sum(successValues[1]))/\
    len(compareLists[0]),\
    color='k',linestyle=':',label='Observed false \npositive')

ax[1].set_xlabel('False-positive fraction',fontsize=8)
ax[1].set_ylabel('Monte-carlo samples',fontsize=8)
#plt.legend(prop={"size":8,"family":"serif"},frameon=False)
deviation = (len(compareLists[0]) - np.sum(successValues[1]) - meanSum)/\
    np.sqrt(varSum)
ax[1].set_title('20 to 5 samples ($' + ("%.2g" % deviation) + "\\sigma$)",\
    fontsize=8)
#plt.savefig(figuresFolder + "false_positive_fraction_20-5.pdf")
#plt.show()

# 10-5 case:


failedVoids10 = np.zeros(removedFiltersList[1][0].shape,dtype=bool)
for k in range(0,len(removedFiltersList[1])):
    failedVoids10 = failedVoids10 | removedFiltersList[1][k]

catProbabilities = finalCatFracList[1][failedVoids10]
thresholdList = np.zeros(len(filtersFinal[1]))
for filt, thresh in zip(scaleFiltersFinal[1],percentilesCat):
    thresholdList[filt] = thresh

thresholds = thresholdList[failedVoids10]
nSelectCats=5
failureChance = np.zeros(catProbabilities.shape)
for k in range(0,len(catProbabilities)):
    thresh_k = thresholds[k]
    cat_k = catProbabilities[k]
    failureChance[k] = scipy.stats.binom.cdf(nSelectCats*thresh_k,nSelectCats,cat_k)

successChance = 1.0 - failureChance

sampleSums = monteCarloSums(successChance,ntries=1000,seed=1000,\
    returnSamples=True)
[meanSum,varSum] = monteCarloSums(successChance,ntries=1000,seed=1000)

ax[2].hist(sampleSums/len(compareLists[0]),color=seabornColormap[0],alpha=0.5,\
    label='Monte carlo samples',density=False,bins=10)
ax[2].axvline(meanSum/len(compareLists[0]),color='k',linestyle='--',\
    label='Monte-carlo mean \nfalse positive')
ax[2].axvline((len(compareLists[0]) - np.sum(successValues[0]))/\
    len(compareLists[0]),\
    color='k',linestyle=':',label='Observed false \npositive')

ax[2].set_xlabel('False-positive fraction',fontsize=8)
ax[2].set_ylabel('Monte-carlo samples',fontsize=8)
ax[2].legend(prop={"size":8,"family":"serif"},frameon=False)
deviation = (len(compareLists[0]) - np.sum(successValues[0]) - meanSum)/\
    np.sqrt(varSum)
ax[2].set_title('10 to 5 samples ($' + ("%.2g" % deviation) + "\\sigma$)",\
    fontsize=8)
plt.tight_layout()
plt.savefig(figuresFolder + "false_positive_fraction.pdf")
plt.show()



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

def plotPPTCurves(nc,rBinCentres,posteriorCounts,dataCounts,ax,nonZero,\
        nonZero2MPP,magBinInd,variancesRobust,errorCounts,mcmcCounts,\
        mcmcCountsAll,postCountsErrors,do2MPPerrors=True,errorType="bootstrap",\
        alpha1sigma=0.5,alpha2sigma=0.25,doMCMCerrors=False,\
        labelPosterior="Posterior ($m < 11.5$)",\
        labelData = "2M++ ($m < 11.5$)",color=None):
    if color is None:
        color = seabornColormap[1]
    if len(nonZero) > 1:
        ax.plot(rBinCentres[nonZero],posteriorCounts[nonZero],\
            color=color,label=labelPosterior,\
            linestyle='-')
    if do2MPPerrors and (len(nonZero2MPP) > 1):
        ax.fill_between(rBinCentres[nonZero2MPP],\
            postCountsErrors[0,nonZero2MPP],\
            postCountsErrors[1,nonZero2MPP],alpha=0.5,\
            color=color)
        ax.fill_between(rBinCentres[nonZero2MPP],\
            postCountsErrors[2,nonZero2MPP],\
            postCountsErrors[3,nonZero2MPP],alpha=0.25,\
            color=color)
    else:
        if len(nonZero2MPP) > 1:
            ax.plot(rBinCentres[nonZero2MPP],dataCounts[nonZero2MPP],\
                color=color,label=labelData,\
                linestyle='-')
    if errorType == "poisson":
        bounds = scipy.stats.poisson(posteriorCounts[nonZero]).interval(0.95)
    elif errorType == "quadrature":
        stdRobust = np.sqrt(variancesRobust[:,nc,magBinInd])
        bounds = (posteriorCounts[nonZero] - 2*stdRobust[nonZero],\
            posteriorCounts[nonZero] + 2*stdRobust[nonZero])
    elif errorType == "bootstrap":
        bounds = (errorCounts[nonZero,nc,0,magBinInd],\
            errorCounts[nonZero,nc,1,magBinInd])
    elif errorType == "variance":
        stdDeviation = np.std(mcmcCountsAll,2)[:,nc,magBinInd]/\
            np.sqrt(nsamples)
        bounds = (posteriorCounts[nonZero] - 2*stdDeviation[nonZero],\
            posteriorCounts[nonZero] + 2*stdDeviation[nonZero])
    else:
        raise Exception("Invalid errorType!")
    if doMCMCerrors and (len(nonZero2MPP) > 1):
        ax.fill_between(rBinCentres[nonZero],bounds[0],bounds[1],\
            color=color,alpha=alpha1sigma)

def pptPlotsInBins(ncList,rBins,nCols = 2,fontfamily='serif',\
        fontsize = 8,do2MPPerrors = True,doMCMCerrors = False,\
        logscale = False,xticks = np.array([0,5,10,15,20]),\
        densityPlot = False,truncateTicks = False,xlim = [0,20],\
        powerRange = 2,nAbsBins=8,textwidth=7.1014,\
        alpha1sigma=0.5,alpha2sigma=0.25,widthFactor=1.0,heightFactor=0.7,\
        mLower = 2,swapXY = False,bottom = 0.105,top = 0.92,right = 0.980,\
        left=None,legloc=(0.02,0.3),xlabelOffset=-0.03):
    rBinCentres = plot.binCentres(rBins)
    # Scaling and y limits:
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
    nRows = int(np.ceil(len(range(mLower,nAbsBins))/nCols))
    if swapXY:
        fig, ax = plt.subplots(nCols*len(ncList),nRows,\
            figsize=(widthFactor*textwidth,heightFactor*textwidth))
    else:
        fig, ax = plt.subplots(nRows,nCols*len(ncList),\
            figsize=(widthFactor*textwidth,heightFactor*textwidth))
    for m in range(mLower,nAbsBins):
        for l in range(0,len(ncList)):
            i = int((m-mLower)/nCols)
            j = m - mLower - nCols*i
            magTitle="$" + str(mAbs[m+1]) + " \\leq M_K < " + str(mAbs[m]) + "$"
            # Cluster 1:
            bright = mcmcCounts[:,ncList[l],2*m]/scale
            nz1 = np.where(bright > 0)[0]
            dim = mcmcCounts[:,ncList[l],2*m+1]/scale
            nz2 = np.where(dim > 0)[0]
            bright2Mpp = counts2MPP[:,ncList[l],2*m]/scale
            dim2Mpp = counts2MPP[:,ncList[l],2*m+1]/scale
            dim2MppError = error2MPPAll[:,ncList[l],:,2*m+1].T/scale
            bright2MppError = error2MPPAll[:,ncList[l],:,2*m].T/scale
            nz12Mpp = np.where(bright2Mpp)[0]
            nz22Mpp = np.where(dim2Mpp)[0]
            # Bright catalogue, cluster 1:
            if swapXY:
                axis = ax[j+nCols*l,i]
            else:
                axis = ax[i,j+nCols*l]
            plotPPTCurves(ncList[l],rBinCentres,bright,bright2Mpp,axis,\
                nz1,nz12Mpp,2*m,variancesRobust,errorCounts,mcmcCounts,\
                mcmcCountsAll,bright2MppError,do2MPPerrors=do2MPPerrors,\
                errorType=errorType,alpha1sigma=alpha1sigma,\
                alpha2sigma=alpha2sigma,doMCMCerrors=doMCMCerrors,\
                labelPosterior="Posterior ($m < 11.5$)",\
                labelData = "2M++ ($m < 11.5$)",color=seabornColormap[1])
            # Dim Catalogue, cluster 1:
            plotPPTCurves(ncList[l],rBinCentres,dim,dim2Mpp,axis,\
                nz2,nz22Mpp,2*m,variancesRobust,errorCounts,mcmcCounts,\
                mcmcCountsAll,dim2MppError,do2MPPerrors=do2MPPerrors,\
                errorType=errorType,alpha1sigma=alpha1sigma,\
                alpha2sigma=alpha2sigma,doMCMCerrors=doMCMCerrors,\
                labelPosterior="Posterior ($m > 11.5$)",\
                labelData = "2M++ ($m > 11.5$)",color=seabornColormap[0])
            axis.set_ylim(ylim)
            axis.set_xlim(xlim)
            if swapXY:
                if j+nCols*l == 0:
                    axis.set_title(magTitle,fontfamily=fontfamily,fontsize=7)
            else:
                if logscale:
                    axis.text(0.5*(xlim[1] + xlim[0]),\
                        ylim[0] + 0.5*(ylim[1] - ylim[0]),magTitle,\
                        ha='center',fontfamily=fontfamily,fontsize=7)
                else:
                    axis.text(0.5*(xlim[1] + xlim[0]),\
                        ylim[0] + 0.90*(ylim[1] - ylim[0]),magTitle,\
                        ha='center',fontfamily=fontfamily,fontsize=7)
            if logscale:
                axis.set_yscale('log')
    # Formatting the axis:
    if swapXY:
        iRange = range(0,nCols*len(ncList))
        jRange = range(0,nRows)
        numRows = nCols*len(ncList)
    else:
        iRange = range(0,nRows)
        jRange = range(0,nCols*len(ncList))
        numRows = nRows
    for i in iRange:
        for j in jRange:
            axis = ax[i,j]
            axis.tick_params(axis='both', which='major', labelsize=fontsize)
            axis.tick_params(axis='both', which='minor', labelsize=fontsize)
            if j != 0:
                # Remove the y labels:
                axis.yaxis.set_ticklabels([])
            if i != 0 and j == 0:
                # Change tick label fonts:
                if truncateTicks:
                    axis.set_yticks(yticks[0:-1])
                    ylabels = ["$" + \
                        plot.scientificNotation(tick,powerRange=powerRange) + \
                            "$" for tick in yticks[0:-1]]
                else:
                    axis.set_yticks(yticks)
                    ylabels = ["$" + \
                        plot.scientificNotation(tick,powerRange=powerRange) + \
                        "$" for tick in yticks]
                axis.yaxis.set_ticklabels(ylabels)
            if j == 0 and i == 0:
                axis.set_yticks(yticks)
                ylabels = ["$" + \
                    plot.scientificNotation(tick,powerRange=powerRange) + "$" \
                    for tick in yticks]
                axis.yaxis.set_ticklabels(ylabels)
            if i != numRows - 1:
                # Remove x labels:
                axis.xaxis.set_ticklabels([])
            else:
                # Remove the last tick, from all but the last:
                if j < 2*len(ncList) - 1:
                    axis.set_xticks(xticks[0:-1])
                    xlabels = ["$" + ("%.2g" % tick) + "$" \
                        for tick in xticks[0:-1]]
                    axis.xaxis.set_ticklabels(xlabels)
                else:
                    axis.set_xticks(xticks)
                    xlabels = ["$" + ("%.2g" % tick) + "$" \
                        for tick in xticks]
                    axis.xaxis.set_ticklabels(xlabels)
    legendType = "fake"
    if legendType == "fake":
        # Legend with a single indicator. Colours will be explained in 
        # the caption.
        #fake2MPP = matplotlib.lines.Line2D([0],[0],color='k',\
        #    label='2M++',linestyle='-')
        fakeMCMC = matplotlib.lines.Line2D([0],[0],color='k',\
            label='Mean \nposterior',linestyle='-')
        fakeError1 = matplotlib.patches.Patch(color='k',alpha=alpha1sigma,\
            label='2M++ \n(68% CI)')
        fakeError2 = matplotlib.patches.Patch(color='k',alpha=alpha2sigma,\
            label='2M++ \n(95% CI)')
        ax[0,0].legend(handles = [fakeMCMC,fakeError1,fakeError2],\
            prop={"size":fontsize,"family":fontfamily},frameon=False,\
            loc=legloc)
    else:
        # Default legend
        ax[0,0].legend(prop={"size":fontsize,"family":fontfamily},\
            frameon=False,loc=legloc)
    if left is None:
        if len(ncList) > 1:
            left = 0.095
        else:
            left = 0.15
    plt.subplots_adjust(top=top,bottom=bottom,left=left,right=right,\
        hspace=0.0,wspace=0.0)
    # Common axis labels:
    fig.text((right+left)/2.0, bottom + xlabelOffset,\
        '$r\\,[\\mathrm{Mpc}h^{-1}]$',\
        ha='center',fontsize=fontsize,fontfamily=fontfamily)
    if densityPlot:
        fig.text(left - 0.06,(top+bottom)/2.0,\
            'Galaxy number density [$h^3\\mathrm{Mpc}^{-3}$]',va='center',\
            rotation='vertical',fontsize=fontsize,fontfamily=fontfamily)
    else:
        fig.text(left - 0.06,(top+bottom)/2.0,'Number of galaxies',va='center',\
            rotation='vertical',fontsize=fontsize,fontfamily=fontfamily)
    # Cluster names:
    for l in range(0,len(ncList)):
        start = 1.0/(2*len(ncList))
        spacing = 1.0/(len(ncList))
        if swapXY:
            fig.text(right + 0.025,top + (bottom - top)*(start + l*spacing),\
                clusterNames[ncList[l]][0],\
                fontsize=fontsize,fontfamily=fontfamily,ha='center',\
                rotation='vertical',va='center')
        else:
            fig.text(left + (right - left)*(start + l*spacing),0.97,\
                clusterNames[ncList[l]][0],\
                fontsize=fontsize,fontfamily=fontfamily,ha='center')
    filename = "_vs_".join([clusterNames[ncList[l]][0] \
        for l in range(0,len(ncList))])
    #plt.savefig(figuresFolder + "ppts_compared_" + filename + ".pdf")
    plt.savefig(figuresFolder + "ppts_compared_all.pdf")
    plt.show()

#pptPlotsInBins(ncList,rBins)
pptPlotsInBins([1,3,4,5,6,7,8],rBins,nCols=1,swapXY=True,heightFactor=1.3,\
    bottom = 0.05,top=0.95,right=0.95,left=0.07,legloc=(0.02,0.1),\
    xlabelOffset=-0.04)

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
doVoidProfiles = True
if doVoidProfiles:
    plot.plotConstrainedVsUnconstrainedProfiles(rBinStackCentres,nbarjSepStack,\
        sigmaSepStack,nbarjAllStackedUn,sigmaAllStackedUn,nbar,rMin,mMin,mMax,\
        show = True,fontsize = fontsize,\
        legendFontSize=legendFontsize,labelCon='Constrained',\
        labelRand='Unconstrained \nmean',\
        savename=figuresFolder + "profiles1415.pdf",showTitle=False,\
        meanErrorLabel = 'Unconstrained \nMean',\
        profileErrorLabel = 'Profile \nvariation \n',\
        nbarjUnconstrainedStacks=nbarjSepStackUn,\
        sigmajUnconstrainedStacks = sigmaSepStackUn,showMean=True)


# Get profiles for the constrained voids only:

snapList =  [pynbody.load(samplesFolder + "sample" + str(snapNum) + "/" \
        + "gadget_full_forward_512/snapshot_001") for snapNum in snapNumList]

snapNameList = [samplesFolder + "sample" + str(snapNum) + "/" \
        + "gadget_full_forward_512/snapshot_001" for snapNum in snapNumList]

# These aren't going to be correct though, relative to the simulations.
# Need to re-do this to get the correct centres.
meanCentres = catData['centres'][combinedFilter135]
meanCentresGadgetCoord = snapedit.wrap(np.fliplr(meanCentres) + boxsize/2,boxsize)
meanRadii = catData['radii'][combinedFilter135]
meanMasses = catData['mass'][combinedFilter135]

meanCentres = meanCentre300[filter300]
meanCentresGadgetCoord = snapedit.wrap(np.fliplr(meanCentres) + boxsize/2,boxsize)
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


# Just get the pairs for each void, and sample:

treeList = [scipy.spatial.cKDTree(snap['pos'],boxsize=boxsize) \
    for snap in snapList]
#treeList = [None for snap in snapList]

treeListUncon = [scipy.spatial.cKDTree(snap['pos'],boxsize=boxsize) \
    for snap in snapListUn]
#treeListUncon = [None for snap in snapListUn]


def getAllPairCountsMCMC(meanCentresGadgetCoord,meanRadii,rBinStack,\
        treeList,snapNameList):
    allPairs = []
    allVolumes = []
    for k in range(0,len(snapNameList)):
        [nPairsList,volumesList] = stacking.getPairCounts(meanCentresGadgetCoord,\
                meanRadii,snapNameList[k],rBinStack,tree=treeList[k],\
                method='poisson',vorVolumes=None)
        allPairs.append(nPairsList)
        allVolumes.append(volumesList)
    return [allPairs,allVolumes]

[allPairs,allVolumes] = tools.loadOrRecompute(\
    data_folder + "pair_counts_mcmc_cut.p",\
    getAllPairCountsMCMC,meanCentresGadgetCoord,meanRadii,rBinStack,\
    treeList,snapNameList,_recomputeData=False)



def getAllPairCountsMCMCSamples(allCentres300Gadget,allRadii300,\
        rBinStackCentres,snapNameList,snapList,treeList):
    allPairsSample = np.zeros((len(snapList),len(allRadii300),\
        len(rBinStackCentres)))
    allVolumesSample = np.zeros((len(snapList),len(allRadii300),\
        len(rBinStackCentres)))
    for k in range(0,len(snapNameList)):
        haveAntiHalo = np.isfinite(allRadii300[:,k])
        noAntiHalo = np.logical_not(haveAntiHalo)
        [nPairsList,volumesList] = stacking.getPairCounts(\
            allCentres300Gadget[k,haveAntiHalo,:],\
            allRadii300[haveAntiHalo,k],snapList[k],rBinStack,tree=treeList[k],\
            method='poisson',vorVolumes=None)
        allPairsSample[k,haveAntiHalo,:] = nPairsList
        allPairsSample[k,noAntiHalo,:] = np.nan
        allVolumesSample[k,haveAntiHalo,:] = volumesList
        allVolumesSample[k,noAntiHalo,:] = np.nan
    return [allPairsSample,allVolumesSample]

[allPairsSample,allVolumesSample] = tools.loadOrRecompute(\
    data_folder + "pair_counts_mcmc_cut_samples.p",getAllPairCountsMCMCSamples,\
    allCentres300Gadget,allRadii300,\
    rBinStackCentres,snapNameList,snapList,treeList,_recomputeData=False)

# Mean profiles over all samples:
flattenedPairs = np.array(allPairs)
flattenedVols = np.array(allVolumes)
flattenedDen = flattenedPairs/flattenedVols
meanVols = np.mean(flattenedVols,0)
meanVolsCumulative = np.nanmean(np.cumsum(flattenedVols,2),0)
meanDensity = np.mean(flattenedPairs/flattenedVols,0)
meanDensityCumulative = np.nanmean(np.cumsum(flattenedPairs,2)/\
    np.cumsum(flattenedVols,2),0)
sigmaDensity = np.std(flattenedPairs/flattenedVols,0)/np.sqrt(len(snapNumList))
allDensities = flattenedPairs/flattenedVols

# Average over profiles centred on samples:
flattenedPairs = allPairsSample
flattenedVols = allVolumesSample
flattenedDen = flattenedPairs/flattenedVols
meanVols = np.nanmean(flattenedVols,0)
meanVolsCumulative = np.nanmean(np.cumsum(flattenedVols,2),0)
meanDensity = np.nanmean(flattenedPairs/flattenedVols,0)
meanDensityCumulative = np.nanmean(np.cumsum(flattenedPairs,2)/\
    np.cumsum(flattenedVols,2),0)
sigmaDensity = np.nanstd(flattenedDen,0)/\
    np.sqrt(np.sum(np.isfinite(flattenedDen),0))
sigmaDensityCumulative = np.nanstd(np.cumsum(flattenedPairs,2)/\
    np.cumsum(flattenedVols,2),0)

# Stacked mean profiles:
rhoStacked = (np.sum(meanDensity*meanVols,0) + 1)/np.sum(meanVols,0)
rhoStackedCumulative = (np.sum(meanDensityCumulative*meanVolsCumulative,0)+1)/\
    np.sum(meanVolsCumulative,0)
#allRhoStacked = (np.sum(meanDensity*meanVols,0) + 1)/np.sum(meanVols,0)
varStacked = np.var(meanDensity,0)
weights = meanVols/np.sum(meanVols,0)
varStackedCumulative = np.var(meanDensityCumulative,0)
weightsCumulative = meanVolsCumulative/np.sum(meanVolsCumulative,0)
# Error accounting for profile uncertainty:
sigmaRhoStacked = np.sqrt(np.sum((sigmaDensity**2 + varStacked)*weights**2,0))
sigmaRhoStackedCumulative = np.sqrt(np.sum((sigmaDensityCumulative**2 + \
    varStackedCumulative)*weightsCumulative**2,0))
# Error ignoring the profile uncertainty:
sigmaRhoStacked2 = np.sqrt(np.sum((varStacked)*weights**2,0))

# Profile and errors for each sample:
rhoStackedSep = (np.sum(flattenedDen*flattenedVols,1) + 1)/\
    np.sum(flattenedVols,1)
varStackedSep = np.var(flattenedDen,1)
weightsSep = flattenedVols/(np.sum(flattenedVols,1)[:,None,:])
sigmaRhoStackedSep = np.sqrt(np.sum((varStackedSep[:,None,:])*weightsSep**2,1))

# Now to repeat the analysis for unconstrained simulations, but sampling based
# on the same size distribution:
snapname = "gadget_full_forward_512/snapshot_001"
#snapNameListUncon = [unconstrainedFolderNew + "sample" \
#            + str(snapNum) + "/" + snapname for snapNum in snapNumListUncon]
snapNameListUncon = []
snapSample = snapListUn[0]
#ahPropsUnconstrained = [tools.loadPickle(name + ".AHproperties.p")\
#            for name in snapNameListUncon]
ahPropsUnconstrained = ahPropsUn
ahCentresListUn = [props[5] for props in ahPropsUnconstrained]
antihaloRadiiUn = [props[7] for props in ahPropsUnconstrained]
antihaloMassesUn = [props[3] for props in ahPropsUn]
centralDensityUn = [props[11] for props in ahPropsUnconstrained]
averageDensityUn = [props[12] for props in ahPropsUnconstrained]
ahTreeCentres = [scipy.spatial.cKDTree(centres,boxsize) \
    for centres in ahCentresListUn]

def densityFromSnapshot(snap,centre,radius,tree=None):
    mUnit = snap['mass'][0]*1e10
    boxsize = snap.properties['boxsize'].ratio("Mpc a h**-1")
    Om0 = snap.properties['omegaM0']
    rhoCrit = 2.7754e11
    rhoMean = rhoCrit*Om0
    volSphere = 4*np.pi*radius**3/3
    if tree is None:
        tree = scipy.spatial.cKDTree(snap['pos'],boxsize=boxsize)
    return mUnit*tree.query_ball_point(centre,radius,\
            workers=-1,return_length=True)/(volSphere*rhoMean) - 1.0



# Get a random selection of centres:
def getRandomCentresAndDensities(rSphere,snapListUn,\
        seed=1000,nRandCentres = 10000):
    # Get a random selection of centres:
    np.random.seed(seed)
    # Get random selection of centres and their densities:
    randCentres = np.random.random((nRandCentres,3))*boxsize
    randOverDen = []
    snapSample = snapListUn[0]
    boxsize = snapSample.properties['boxsize'].ratio("Mpc a h**-1")
    for k in range(0,len(snapListUn)):
        snap = snapListUn[k]
        gc.collect() # Clear memory of the previous snapshot
        tree = scipy.spatial.cKDTree(snap['pos'],boxsize=boxsize)
        gc.collect()
        randOverDen.append(densityFromSnapshot(snap,randCentres,rSphere,
            tree = tree))
    return [randCentres,randOverDen]


[randCentres,randOverDen] = tools.loadOrRecompute(\
    data_folder + "random_centres_and_densities.p",\
    getRandomCentresAndDensities,rSphere,snapListUn,_recomputeData=False)


# Get MCMC densities:
#deltaMCMCList = np.array(\
#    [densityFromSnapshot(snap,np.array([boxsize/2]*3),135) \
#    for snap in snapList])

#tools.savePickle(deltaMCMCList,data_folder + "delta_list.p")
deltaMCMCList = tools.loadPickle(data_folder + "delta_list.p")

# Get MAP density:
kde = scipy.stats.gaussian_kde(deltaMCMCList,bw_method="scott")
deltaMAP = scipy.optimize.minimize(lambda x: -kde.evaluate(x),\
    np.mean(deltaMCMCList)).x[0]
kdeSamples = kde.resample(1000,seed=1000)

def getMAPFromSample(sample):
    kde = scipy.stats.gaussian_kde(sample,bw_method="scott")
    return scipy.optimize.minimize(lambda x: -kde.evaluate(x),\
        np.mean(sample)).x[0]

deltaMAPBootstrap = scipy.stats.bootstrap((deltaMCMCList,),\
    getMAPFromSample,confidence_level = 0.68,vectorized=False)

deltaMAPInterval = deltaMAPBootstrap.confidence_interval

# Get comparable density regions:
comparableDensity = [(delta <= deltaListMeanNew + deltaListErrorNew) & \
    (delta > deltaListMeanNew - deltaListErrorNew) for delta in randOverDen]
comparableDensityMAP = [(delta <= deltaMAPInterval[1]) & \
    (delta > deltaMAPInterval[0]) for delta in randOverDen]
#centresToUse = [randCentres[comp] for comp in comparableDensity]
centresToUse = [randCentres[comp] for comp in comparableDensityMAP]

maxRegions = 100


#deltaToUse = [randOverDen[ns][comp] \
#    for ns, comp in zip(range(0,len(snapList)),comparableDensity)]

deltaToUse = [randOverDen[ns][comp] \
    for ns, comp in zip(range(0,len(snapList)),comparableDensityMAP)]



def getDistanceBetweenCentres(centre1,centre2,boxsize):
    if (len(centre1.shape) == 1 and len(centre2.shape) == 1):
        return np.sqrt(np.sum(snapedit.unwrap(centre1 - centre2,boxsize)**2))
    else:
        return np.sqrt(np.sum(snapedit.unwrap(centre1 - centre2,boxsize)**2,1))

# Filter to ensure independence of centres:
centresToUseNonOverlapping = []
deltaToUseNonOverlapping = []
rSep = 2*135
for ns in range(0,len(snapListUn)):
    allCentresNS = centresToUse[ns]
    allDeltasNS = deltaToUse[ns]
    allCentresNonOverlap = []
    allDeltasNonOverlap = []
    for k in range(0,len(allCentresNS)):
        include = True
        for l in range(0,len(allCentresNonOverlap)):
            include = include and (getDistanceBetweenCentres(allCentresNS[k],\
                allCentresNonOverlap[l],boxsize) > rSep)
        if include:
            allCentresNonOverlap.append(allCentresNS[k])
            allDeltasNonOverlap.append(allDeltasNS[k])
    centresToUseNonOverlapping.append(allCentresNonOverlap)
    deltaToUseNonOverlapping.append(allDeltasNonOverlap)

centresToUseNonOverlapping = [np.array(cen) \
    for cen in centresToUseNonOverlapping]
deltaToUseNonOverlappingBySample = [np.array(cen) \
    for cen in deltaToUseNonOverlapping]
deltaToUseNonOverlapping = np.hstack(deltaToUseNonOverlappingBySample)
# Bins to use when building a catalogue similar to the constrained
# catalogue:
rEffBinEdges = np.linspace(10,25,6)
[inRadBinsComb,noInRadBinsComb] = plot.binValues(meanRadii,rEffBinEdges)



def getRandomCataloguePairCounts(centreListToTest,snapListUn,treeListUncon,\
        ahCentresListUn,antihaloRadiiUn,rSphere,radBinEdges,rBinStack,\
        meanRadiiMCMC,boxsize,seed=1000,start=0,end=-1,\
        conditioningQuantityUn=None,conditioningQuantityMCMC=None,\
        conditionBinEdges=None):
    # Get pair counts in similar-density regions:
    allPairsUncon = []
    allVolumesUncon = []
    allSelections = []
    np.random.seed(seed)
    [inRadBinsComb,noInRadBinsComb] = plot.binValues(meanRadiiMCMC,\
        radBinEdges)
    #centreListToTest = centresToUseNonOverlapping
    if end == -1:
        end = len(snapListUn)
    for ns in range(start,end):
        snapLoaded = snapListUn[ns]
        #tree = scipy.spatial.cKDTree(snapLoaded['pos'],boxsize=boxsize)
        tree = treeListUncon[ns]
        for centre, count in zip(centreListToTest[ns],\
            range(0,len(centreListToTest[ns]))):
            # Get anti-halos:
            centralAntihalos = tools.getAntiHalosInSphere(ahCentresListUn[ns],\
                rSphere,origin=centre,boxsize=boxsize)
            # Get radii and randomly select voids with the same
            # radius distribution as the combined catalogue:
            centralRadii = antihaloRadiiUn[ns][centralAntihalos[1]]
            centralCentres = ahCentresListUn[ns][centralAntihalos[1]]
            if conditioningQuantityMCMC is not None:
                centralConditionVariable = conditioningQuantityUn[ns][\
                    centralAntihalos[1]]
            [inRadBins,noInRadBins] = plot.binValues(centralRadii,\
                radBinEdges)
            # Select voids with the same radius distribution as the combined 
            # catalogue:
            selection = []
            for k in range(0,len(radBinEdges)-1):
                if noInRadBinsComb[k] > 0:
                    # If not using a second condition:
                    if conditioningQuantityMCMC is None:
                        selection.append(np.random.choice(inRadBins[k],\
                            np.min([noInRadBinsComb[k],noInRadBins[k]]),\
                            replace=False))
                    else:
                        [inConBinsComb,noInConBinsComb] = plot.binValues(\
                            conditioningQuantityMCMC[inRadBinsComb[k]],\
                            conditionBinEdges)
                        [inConBins,noInConBins] = plot.binValues(\
                            centralConditionVariable[inRadBins[k]],\
                            conditionBinEdges)
                        for l in range(0,len(conditionBinEdges)-1):
                            selection.append(np.random.choice(\
                                inRadBins[k][inConBins[l]],\
                                np.min([noInConBinsComb[l],noInConBins[l]]),\
                                replace=False))
            selectArray = np.hstack(selection)
            # Now get pair counts around these voids:
            [nPairsList,volumesList] = stacking.getPairCounts(\
                centralCentres[selectArray],\
                centralRadii[selectArray],snapLoaded,rBinStack,\
                tree=tree,method='poisson',vorVolumes=None)
            allPairsUncon.append(nPairsList)
            allVolumesUncon.append(volumesList)
            allSelections.append(selectArray)
            print("Done centre " + str(count+1) + " of " + \
                str(len(centreListToTest[ns])))
        tools.savePickle([allPairsUncon,allVolumesUncon],"temp_sample_" + \
            str(ns+1) + ".p")
        print("Done sample " + str(ns + 1) + ".")
        # Delete temporaries to save memory:
        #del snapLoaded, tree
        #gc.collect()
    return [allPairsUncon,allVolumesUncon,allSelections]

[allPairsUncon,allVolumesUncon,allSelectionsUncon] = tools.loadOrRecompute(\
    data_folder + "pair_counts_random_cut.p",getRandomCataloguePairCounts,\
    centresToUse,snapListUn,treeListUncon,ahCentresListUn,\
    antihaloRadiiUn,rSphere,rEffBinEdges,rBinStack,meanRadii,boxsize,\
    _recomputeData=False)

[allPairsUnconNonOverlap,allVolumesUnconNonOverlap,\
    allSelectionsUnconNonOverlap] = tools.loadOrRecompute(\
        data_folder + "pair_counts_random_cut_non_overlapping.p",\
        getRandomCataloguePairCounts,\
        centresToUseNonOverlapping,snapListUn,treeListUncon,ahCentresListUn,\
        antihaloRadiiUn,rSphere,rEffBinEdges,rBinStack,meanRadii,boxsize,\
        _recomputeData=False)

conBinEdges = np.linspace(-1,-0.5,21)
[allPairsUnconNonOverlapOld,allVolumesUnconNonOverlapOld,\
    allSelectionsUnconNonOverlapOld] = tools.loadOrRecompute(\
        data_folder + "pair_counts_density_and_radius_conditioning.p",\
        getRandomCataloguePairCounts,\
        centresToUseNonOverlapping,snapListUn,treeListUncon,ahCentresListUn,\
        antihaloRadiiUn,rSphere,rEffBinEdges,rBinStack,meanRadii,boxsize,\
        _recomputeData=False,\
        conditioningQuantityUn = [props[11] for props in ahPropsUn],\
        conditioningQuantityMCMC = deltaCentralMean[filter300],\
        conditionBinEdges = conBinEdges)

conditioningQuantityUn = [np.vstack([antihaloRadiiUn[ns],\
    ahPropsUn[ns][11],ahPropsUn[ns][12]]).T \
    for ns in range(0,len(snapListUn))]
conditioningQuantityMCMC = np.vstack([meanRadii,\
    deltaCentralMean[filter300],deltaAverageMean[filter300]]).T

[centralConditionVariableAll,centralCentresAll,centralRadiiAll,\
                sampleIndices] = getAllConditionVariables(\
                centreListToTest,ahCentresListUn,\
                antihaloRadiiUn,conditioningQuantityUn)
[_,_,centralMassesAll,_] = getAllConditionVariables(\
                centreListToTest,ahCentresListUn,\
                antihaloMassesUn,conditioningQuantityUn)

[allPairsUnconNonOverlap,allVolumesUnconNonOverlap,\
    allSelectionsUnconNonOverlap] = tools.loadOrRecompute(\
    data_folder + "pair_counts_triple_conditioning.p",\
    getRandomCataloguePairCounts,\
    centresToUseNonOverlapping,snapListUn,treeListUncon,ahCentresListUn,\
    antihaloRadiiUn,rSphere,rEffBinEdges,rBinStack,meanRadii,boxsize,\
    conditioningQuantityUn = conditioningQuantityUn,\
    conditioningQuantityMCMC = conditioningQuantityMCMC,\
    conditionBinEdges = [rEffBinEdges,conBinEdges,conBinEdges],\
    combineRandomRegions=True,_recomputeData=True)


[allPairsUncon,allVolumesUncon] = tools.loadPickle("temp_sample_13.p")


# Central and average densities in similar regions:
# For OLD conditioning:
centralDensityUnconNonOverlap = []
averageDensityUnconNonOverlap = []
massesUnconNonOverlap = []
counter = 0
for ns in range(0,len(snapListUn)):
    centres = centresToUseNonOverlapping[ns]
    numCentres = len(centres)
    for centre, count in zip(centres,range(0,numCentres)):
        centralAHs = tools.getAntiHalosInSphere(ahCentresListUn[ns],\
                rSphere,origin=centre,boxsize=boxsize)
        sphereCentralDensities = centralDensityUn[ns][centralAHs[1]]
        sphereAverageDensities = averageDensityUn[ns][centralAHs[1]]
        sphereMasses = antihaloMassesUn[ns][centralAHs[1]]
        centralDensityUnconNonOverlap.append(\
            sphereCentralDensities[allSelectionsUnconNonOverlap[counter]])
        averageDensityUnconNonOverlap.append(\
            sphereAverageDensities[allSelectionsUnconNonOverlap[counter]])
        massesUnconNonOverlap.append(sphereMasses)
        counter += 1

# For NEW conditioning:
centralDensityUnconNonOverlap = \
    [centralConditionVariableAll[allSelectionsUnconNonOverlap[ns],1] \
    for ns in range(0,len(snapListUn))]
averageDensityUnconNonOverlap = \
    [centralConditionVariableAll[allSelectionsUnconNonOverlap[ns],2] \
    for ns in range(0,len(snapListUn))]
massesUnconNonOverlap = [centralMassesAll[allSelectionsUnconNonOverlap[ns]] \
    for ns in range(0,len(snapListUn))]


#aupcd = tools.loadPickle(data_folder + "all_unconstrained_pair_counts_data.p")

# Stacked profiles in each region:
rhoStackedUnAll = np.array([(np.sum(allPairsUncon[k],0)+1)/\
    np.sum(allVolumesUncon[k],0) for k in range(0,len(allVolumesUncon))])
variancesUnAll = np.array([np.var(allPairsUncon[k]/allVolumesUncon[k],0) \
    for k in range(0,len(allVolumesUncon))])
weightsUnAll = [(allVolumesUncon[k])/np.sum(allVolumesUncon[k],0) \
    for k in range(0,len(allVolumesUncon))]
sumSquareWeights = np.array([np.sum(wi**2,0) for wi in weightsUnAll])

#regionsFilter = range(0,len(allVolumesUnconNonOverlap))
#regionsFilter = np.where( \
#    (deltaToUseNonOverlapping > deltaListMeanNew - deltaListErrorNew/2) & \
#    (deltaToUseNonOverlapping <= deltaListMeanNew + deltaListErrorNew/2) )[0]
#regionsFilter = np.where( \
#    (deltaToUseNonOverlapping > deltaMAPInterval[0]) & \
#    (deltaToUseNonOverlapping <= deltaMAPInterval[1]) )[0]
regionsFilter = range(0,len(snapListUn))

mUnitLowRes = 8*snapList[0]['mass'][0]*1e10
#profileFilterUnconNonOverlap = [np.ones(selection.shape,dtype=bool) \
#    for selection in allSelectionsUnconNonOverlap]
# For use with the old selection:
#profileFilterUnconNonOverlap = [massList[selection] >= mUnitLowRes*100 \
#    for massList, selection in \
#    zip(massesUnconNonOverlap,allSelectionsUnconNonOverlap)]
profileFilterUnconNonOverlap = [massList >= mUnitLowRes*100 \
    for massList in massesUnconNonOverlap]

# Stacked profiles in each region:
rhoStackedUnAllNonOverlap = np.array([\
    (np.sum(allPairsUnconNonOverlap[k][filt],0)+1)/\
    np.sum(allVolumesUnconNonOverlap[k][filt],0) \
    for k, filt in zip(regionsFilter,profileFilterUnconNonOverlap)])
rhoStackedUnAllNonOverlapCumulative = \
    np.array([(np.sum(np.cumsum(allPairsUnconNonOverlap[k][filt],1),0)+1)/\
    np.sum(np.cumsum(allVolumesUnconNonOverlap[k][filt],1),0) \
    for k, filt in zip(regionsFilter,profileFilterUnconNonOverlap)])
variancesUnAllNonOverlap = np.array([np.var(allPairsUnconNonOverlap[k][filt]/\
    allVolumesUnconNonOverlap[k][filt],0) \
    for k, filt in zip(regionsFilter,profileFilterUnconNonOverlap)])
variancesUnAllNonOverlapCumulative = \
    np.array([np.var(np.cumsum(allPairsUnconNonOverlap[k][filt],1)/\
    np.cumsum(allVolumesUnconNonOverlap[k][filt],1),0) \
    for k, filt in zip(regionsFilter,profileFilterUnconNonOverlap)])
weightsUnAllNonOverlap = [(allVolumesUnconNonOverlap[k][filt])/\
    np.sum(allVolumesUnconNonOverlap[k][filt],0) \
    for k, filt in zip(regionsFilter,profileFilterUnconNonOverlap)]
weightsUnAllNonOverlapCumulative = \
    [(np.cumsum(allVolumesUnconNonOverlap[k][filt],1))/\
    np.sum(np.cumsum(allVolumesUnconNonOverlap[k][filt],1),0) \
    for k, filt in zip(regionsFilter,profileFilterUnconNonOverlap)]
sumSquareWeightsNonOverlap = np.array([np.sum(wi**2,0) \
    for wi in weightsUnAllNonOverlap])
sumSquareWeightsNonOverlapCumulative = np.array([np.sum(wi**2,0) \
    for wi in weightsUnAllNonOverlapCumulative])
#rhoRandomToUse = rhoStackedUnAll
#rhoRandomToUse = rhoStackedUnAllNonOverlap
#rhoRandomToUse = rhoStackedUnAllNonOverlapCumulative
#rhoMCMCToUse = rhoStackedCumulative
#sigmaRhoMCMCToUse = sigmaRhoStackedCumulative
rhoRandomToUse = rhoStackedUnAllNonOverlap
rhoMCMCToUse = rhoStacked
sigmaRhoMCMCToUse = sigmaRhoStacked

# Filter to catch broken unconstrained sims during testing:
filterCores = np.where(rhoRandomToUse[:,0] < 0.15)[0]
sigmaStackedUnAll = np.std(rhoRandomToUse[filterCores],0)
meanStackedUnAll = np.mean(rhoRandomToUse[filterCores],0)
intervals = [68,95]
intervalLimits = []
for lim in intervals:
    intervalLimits.append(50 - lim/2)
    intervalLimits.append(50 + lim/2)

credibleIntervals = np.percentile(rhoRandomToUse[filterCores],\
    intervalLimits,axis=0)


# Plot comparison:
fig, ax = plt.subplots(figsize=(textwidth,0.5*textwidth))
ax.fill_between(rBinStackCentres,(meanStackedUnAll - sigmaStackedUnAll)/nbar,\
    (meanStackedUnAll + sigmaStackedUnAll)/nbar,alpha=0.5,color='grey',\
    label='Random catalogue')
ax.errorbar(rBinStackCentres,rhoStacked/nbar,yerr=sigmaRhoStacked/nbar,\
    color='k',linestyle='-',label='Constrained Catalogue')
ax.axvline(1.0,color='grey',linestyle=':')
ax.axhline(1.0,color='grey',linestyle=':')
ax.set_xlabel('$R/R_{\\mathrm{eff}}$',fontsize=8)
ax.set_ylabel('$\\rho/\\bar{\\rho}$',fontsize=8)
ax.legend(prop={"size":fontsize,"family":"serif"},frameon=False)
ax.set_xlim([0,10])
ax.set_ylim([0,1.2])
plt.tight_layout()
plt.savefig(figuresFolder + "profiles_vs_random_sample.pdf")
plt.show()

# With shaded regions only:
fig, ax = plt.subplots(figsize=(textwidth,0.5*textwidth))
ax.fill_between(rBinStackCentres,credibleIntervals[0]/nbar,\
    credibleIntervals[1]/nbar,alpha=0.75,color='grey',\
    label='Random catalogue (68%)')
ax.fill_between(rBinStackCentres,credibleIntervals[2]/nbar,\
    credibleIntervals[3]/nbar,alpha=0.5,color='grey',\
    label='Random catalogue (95%)')
#ax.errorbar(rBinStackCentres,rhoStacked/nbar,yerr=sigmaRhoStacked/nbar,\
#    color='k',linestyle='-',label='Constrained Catalogue')
ax.fill_between(rBinStackCentres,(rhoMCMCToUse - sigmaRhoMCMCToUse)/nbar,\
    (rhoMCMCToUse + sigmaRhoMCMCToUse)/nbar,alpha=0.5,color=seabornColormap[0],\
    label='MCMC catalogue')
ax.axvline(1.0,color='grey',linestyle=':')
ax.axhline(1.0,color='grey',linestyle=':')
ax.fill_between(rBinStackCentres,1 + deltaListMeanNew - deltaListErrorNew,\
    1 + deltaListMeanNew + deltaListErrorNew,alpha=0.5,\
    color=seabornColormap[1],label='Local super-volume density')
ax.set_xlabel('$R/R_{\\mathrm{eff}}$',fontsize=8)
ax.set_ylabel('$\\rho/\\bar{\\rho}$',fontsize=8)
ax.legend(prop={"size":fontsize,"family":"serif"},frameon=False)
ax.set_xlim([0,10])
ax.set_ylim([0,1.2])
plt.tight_layout()
plt.savefig(figuresFolder + "profiles_mcmc_vs_random.pdf")
plt.show()


# Plot comparison:
fig, ax = plt.subplots(figsize=(textwidth,0.5*textwidth))
#ax.fill_between(rBinStackCentres,(meanStackedUnAll - sigmaStackedUnAll)/nbar,\
#    (meanStackedUnAll + sigmaStackedUnAll)/nbar,alpha=0.5,color='grey',\
#    label='Random catalogue')
ax.plot(rBinStackCentres,rhoRandomToUse.T/nbar,color='grey',linestyle=':',\
    zorder=1)
ax.fill_between(rBinStackCentres,(rhoStacked - sigmaRhoStacked)/nbar,\
    (rhoStacked + sigmaRhoStacked)/nbar,alpha=0.5,color='k',\
    label='MCMC catalogue',zorder=3)
ax.axvline(1.0,color='grey',linestyle=':')
ax.axhline(1.0,color='grey',linestyle=':')
ax.fill_between(rBinStackCentres,1 + deltaListMeanNew - deltaListErrorNew,\
    1 + deltaListMeanNew + deltaListErrorNew,alpha=0.5,\
    color=seabornColormap[1],label='Local super-volume density')
ax.set_xlabel('$R/R_{\\mathrm{eff}}$',fontsize=8)
ax.set_ylabel('$\\rho/\\bar{\\rho}$',fontsize=8)
ax.legend(prop={"size":fontsize,"family":"serif"},frameon=False)
ax.set_xlim([0,10])
ax.set_ylim([0,1.2])
plt.tight_layout()
plt.savefig(figuresFolder + "profiles_vs_random_all.pdf")
plt.show()


# Distribution of densities:
deltaMCMCMean = np.mean(deltaMCMCList)
deltaMCMCSigma = np.std(deltaMCMCList)
deltaMCMCStdErr = deltaMCMCSigma/np.sqrt(len(deltaMCMCList))

plt.clf()
plt.hist(deltaMCMCList,bins=np.linspace(-0.055,-0.035,5),\
    color=seabornColormap[0],alpha=0.5,density=True,label='MCMC samples')
plt.axvline(deltaMCMCMean-deltaMCMCStdErr,linestyle='--',color='k',\
    label='Standard Error of the Mean')
plt.axvline(deltaMCMCMean+deltaMCMCStdErr,linestyle='--',color='k')
plt.axvline(deltaMCMCMean-deltaMCMCSigma,linestyle=':',color='k',\
    label='Standard Deviation')
plt.axvline(deltaMCMCMean+deltaMCMCSigma,linestyle=':',color='k')
plt.hist(np.hstack(randOverDen),bins=np.arange(-0.1,0.1,0.005),\
    color=seabornColormap[1],alpha=0.5,\
    label='Random spheres \n(unconditioned)',density=True)
deltaPoints = np.linspace(-0.055,-0.035,51)
plt.plot(deltaPoints,kde.evaluate(deltaPoints),linestyle='-',color='k',\
    label='KDE')
plt.xlabel('Density contrast in $135\\,\\mathrm{Mpc}h^{-1}$')
plt.ylabel('Probability Density')
plt.legend()
plt.savefig(figuresFolder + "mcmc_density_distribution_plot.pdf")
plt.show()


[nbarjStack,sigmaStack] = stacking.computeMeanStacks(centresList,radiiList,\
    massesList,conditionList,\
        pairsList,volumesList,\
        snapList,nbar,rBinStack,rMin,rMax,mMin,mMax,\
        method="poisson",errorType = "Weighted",toInclude = "all")


# Diagnostic plots:

# Randoms distribution at specific points:
titles = [["$R/R_{\\mathrm{eff}} = 0$","$R/R_{\\mathrm{eff}} = 1$"],\
    ["$R/R_{\\mathrm{eff}} = 3$","$R/R_{\\mathrm{eff}} = 10$"]]
indicesij = [[0,9],[29,99]]


plt.clf()
fig, ax = plt.subplots(2,2,figsize=(textwidth,textwidth))
for i in range(0,2):
    for j in range(0,2):
        ind = indicesij[i][j]
        minmax = tools.minmax(rhoRandomToUse[filterCores][:,ind]/nbar)
        ax[i,j].hist(rhoRandomToUse[filterCores][:,ind]/nbar,\
            bins=np.linspace(minmax[0],minmax[1],11),\
            alpha=0.5,color=seabornColormap[0],density=True)
        mean = np.mean(rhoRandomToUse[filterCores][:,ind]/nbar)
        std = np.std(rhoRandomToUse[filterCores][:,ind]/nbar)
        interval68 = np.percentile(\
            rhoRandomToUse[filterCores][:,ind]/nbar,[16,84])
        interval95 = np.percentile(\
            rhoRandomToUse[filterCores][:,ind]/nbar,[2.5,97.5])
        ax[i,j].axvline(mean,color='k',linestyle='-',label='Mean')
        ax[i,j].axvline(mean - std,color='k',linestyle=':',\
            label='standard deviation')
        ax[i,j].axvline(mean + std,color='k',linestyle=':')
        ax[i,j].axvline(interval68[0],color='grey',linestyle='--',\
            label='68% interval')
        ax[i,j].axvline(interval68[1],color='grey',linestyle='-.')
        ax[i,j].axvline(rhoStacked[ind]/nbar,color=seabornColormap[1],\
            linestyle='-',label='MCMC catalogue')
        ax[i,j].set_xlabel('$\\rho/\\bar{\\rho}$')
        ax[i,j].set_ylabel('Probability Density')
        ax[i,j].set_title(titles[i][j])
        #ax[i,j].set_xlim([0,1])

ax[1,1].legend()
plt.tight_layout()
plt.savefig(figuresFolder + "random_profiles_distribution.pdf")
plt.show()


# Analysis:
densityReff1 = rhoRandomToUse[filterCores][:,ind]/nbar
lowPop = np.where(densityReff1 < 0.36)[0]
highPop = np.where(densityReff1 > 0.36)[0]

regionSamples = np.zeros(len(rhoRandomToUse),dtype=int)
regionLists = []
deltaAverageAll = [x[12] for x in ahPropsUn]
deltaCentralAll = [x[11] for x in ahPropsUn]
deltaAverageRandomList = []
deltaCentralRandomList = []
count = 0
for ns in range(0,len(snapList)):
    for centre in centresToUseNonOverlapping[ns]:
        regionSamples[count] = ns
        centralAHs = tools.getAntiHalosInSphere(ahCentresListUn[ns],\
                rSphere,origin=centre,boxsize=boxsize)
        regionLists.append(centralAHs[1])
        selectFilter = allSelectionsUnconNonOverlap[count]
        deltaAverageRandomList.append(deltaAverageAll[ns][selectFilter])
        deltaCentralRandomList.append(deltaCentralAll[ns][selectFilter])
        count += 1


fit = np.polyfit(deltaAverageMean[filter300],deltaCentralMean[filter300],1)
stackedAverageRandom = np.hstack(deltaAverageRandomList)
stackedCentralRandom = np.hstack(deltaCentralRandomList)
fitRand = np.polyfit(stackedAverageRandom,stackedCentralRandom,1)

# Density distribution:
plt.clf()
fig, ax = plt.subplots(1,2,figsize=(textwidth,0.5*textwidth))
histMean = ax[0].hist2d(deltaAverageMean[filter300],\
    deltaCentralMean[filter300],\
    bins=np.array([np.linspace(-0.8,-0.6,21),\
    np.linspace(-1.0,-0.8,21)]),density=True,cmap='Blues')
plt.colorbar(histMean[3],label='Probability density')
xpoints = np.linspace(-0.8,-0.6,21)
ax[0].plot(xpoints,xpoints*fit[0] + fit[1],linestyle='--',color='k',\
    label="\\delta_c = " + ("%.2g" % fit[0]) + " + " + ("%.2g" % fit[1]))
ax[0].set_xlabel('Average density contrast, $\\bar{\\delta}$')
ax[0].set_ylabel('Central density contrast, $\\delta_c$')
ax[0].set_title("Local supervolume Catalogue")

histMean = ax[1].hist2d(stackedAverageRandom,\
    stackedCentralRandom,\
    bins=np.array([np.linspace(-0.8,-0.6,21),\
    np.linspace(-1.0,-0.8,21)]),density=True,cmap='Blues')
plt.colorbar(histMean[3],label='Probability density')
ax[1].plot(xpoints,xpoints*fitRand[0] + fitRand[1],linestyle='--',color='k',\
    label="\\delta_c = " + ("%.2g" % fitRand[0]) + " + " + \
    ("%.2g" % fitRand[1]))
ax[1].set_xlabel('Average density contrast, $\\bar{\\delta}$')
ax[1].set_ylabel('Central density contrast, $\\delta_c$')
ax[1].set_title("144 Random Catalogues")
plt.tight_layout()
plt.savefig(figuresFolder + "density_central_average_distribution.pdf")
plt.show()



# Binning in each random simulation region individually:
numBins = 21
bins=np.linspace(-1,-0.5,numBins)
binCountsAverageUncon = np.zeros(\
    (len(averageDensityUnconNonOverlap),numBins-1),dtype=int)
for k in range(0,len(averageDensityUnconNonOverlap)):
    [_,noInBins] = plot.binValues(averageDensityUnconNonOverlap[k],bins)
    binCountsAverageUncon[k,:] = noInBins

binWidths = bins[1:] - bins[0:-1]
probAverageDensityUncon = (binCountsAverageUncon/\
    np.sum(binCountsAverageUncon,1)[:,None])/binWidths
sigmaAverageDensityUncon = np.std(probAverageDensityUncon,0)

binCountsCentralUncon = np.zeros(\
    (len(centralDensityUnconNonOverlap),numBins-1),dtype=int)
for k in range(0,len(centralDensityUnconNonOverlap)):
    [_,noInBins] = plot.binValues(centralDensityUnconNonOverlap[k],bins)
    binCountsCentralUncon[k,:] = noInBins

binWidths = bins[1:] - bins[0:-1]
probCentralDensityUncon = (binCountsCentralUncon/\
    np.sum(binCountsCentralUncon,1)[:,None])/binWidths
sigmaCentralDensityUncon = np.std(probCentralDensityUncon,0)

from void_analysis.plot import histWithErrors

# Density histograms:
plt.clf()
fig, ax = plt.subplots(1,2,figsize=(textwidth,0.5*textwidth))
ax[0].hist(deltaAverageMean[filter300],
    bins=np.linspace(-1,-0.5,numBins),density=True,\
    color=seabornColormap[1],label='MCMC \ncatalogue',alpha=0.5)
#ax[0].hist(np.hstack(averageDensityUnconNonOverlap),\
#    bins=np.linspace(-1,-0.5,21),density=True,\
#    color=seabornColormap[1],label='Random simulations',alpha=0.5)
#ax[0].hist(averageDensityUnconNonOverlap[0],\
#    bins=np.linspace(-1,-0.5,numBins),density=True,\
#    color=seabornColormap[1],label='Random simulations',alpha=0.5)
histWithErrors(np.mean(probAverageDensityUncon,0),sigmaAverageDensityUncon,\
    bins,ax=ax[0],color=seabornColormap[0],alpha=0.5,\
    label='Random \nsimulations')
ax[0].set_title('Average Density')
ax[0].set_xlim([-1,-0.5])
ax[0].set_xlabel('$\\bar{\\delta}$')
ax[0].set_ylabel('Probability density')
ax[0].set_ylim([0,30])

ax[1].hist(deltaCentralMean[filter300],\
    bins=np.linspace(-1,-0.5,numBins),density=True,\
    color=seabornColormap[1],label='MCMC \ncatalogue',alpha=0.5)
#ax[1].hist(np.hstack(centralDensityUnconNonOverlap),\
#    bins=np.linspace(-1,-0.5,21),density=True,\
#    color=seabornColormap[1],label='Random simulations',alpha=0.5)
#ax[1].hist(centralDensityUnconNonOverlap[0],\
#    bins=np.linspace(-1,-0.5,numBins),density=True,\
#    color=seabornColormap[1],label='Random simulations',alpha=0.5)
histWithErrors(np.mean(probCentralDensityUncon,0),sigmaCentralDensityUncon,\
    bins,ax=ax[1],color=seabornColormap[0],alpha=0.5,\
    label='Random \nsimulations')
ax[1].set_title('Central Density')
ax[1].set_xlim([-1,-0.5])
ax[1].set_xlabel('$\\delta_c$')
ax[1].set_ylabel('Probability density')
ax[1].set_ylim([0,30])

ax[1].legend()
#plt.tight_layout()
plt.subplots_adjust(bottom = 0.15,top=0.90,left = 0.15,right=0.95)
plt.savefig(figuresFolder + "density_histograms.pdf")
plt.show()

# Comparison of means:
stackedCentralRandom = np.hstack(centralDensityUnconNonOverlap)
meansIndividualCentral = np.array([np.nanmean(x) \
    for x in centralDensityUnconNonOverlap])
isNotNanCentral = np.where(np.isfinite(stackedCentralRandom))[0]
meanCentralRandom = np.mean(meansIndividualCentral)
stdErrorCentralRandom = np.nanstd(meansIndividualCentral)

stackedAverageRandom = np.hstack(averageDensityUnconNonOverlap)
meansIndividualAverage = np.array([np.nanmean(x) \
    for x in averageDensityUnconNonOverlap])
isNotNanAverage = np.where(np.isfinite(stackedAverageRandom))[0]
meanAverageRandom = np.nanmean(meansIndividualAverage)
stdErrorAverageRandom = np.nanstd(meansIndividualAverage)


meanCentral = np.mean(deltaCentralMean[filter300])
meanAverage = np.mean(deltaAverageMean[filter300])
stdErrorCentral = np.std(deltaAverageMean[filter300])/\
    np.sqrt(np.sum(filter300))
stdErrorAverage = np.std(deltaAverageMean[filter300])/\
    np.sqrt(np.sum(filter300))

[meanCentralRandom - stdErrorCentralRandom,meanCentralRandom + stdErrorCentralRandom]
[meanAverageRandom - stdErrorAverageRandom,meanAverageRandom + stdErrorAverageRandom]
[meanCentral - stdErrorCentral,meanCentral + stdErrorCentral]
[meanAverage - stdErrorAverage,meanAverage + stdErrorAverage]

averageSigmas = (meanAverage - meanAverageRandom)/stdErrorAverageRandom
centralSigmas = (meanCentral - meanCentralRandom)/stdErrorCentralRandom

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

# Anti-halo centres in Equatorial co-ordinates:
ahCentresEquatorial = [tools.remapAntiHaloCentre(props[5],boxsize,\
    swapXZ=False,reverse=True) for props in allProps]
ahDistancesEquatorial = [np.sqrt(np.sum(pos**2,1)) \
    for pos in ahCentresEquatorial]

for snap in snapList:
    tools.remapBORGSimulation(snap,swapXZ=False,reverse=True)


doClusterMasses = True
if doClusterMasses:
    [meanMasses,meanCentres,sigmaMasses,sigmaCentres,\
            clusterMasses,clusterCentres,clusterCounterparts] = \
                tools.loadOrRecompute(data_folder + "mean_cluster_masses.p",\
                    getBORGClusterMassEstimates,snapNameList,clusterLoc,\
                    equatorialXYZ,_recomputeData=recomputeData)

# Fornax and Virgo:
otherClustersLocRADec = np.array([[187,10,16.5],[9,-35,19.5]])
coordOtherClusters = SkyCoord(ra=otherClustersLocRADec[:,0]*u.deg,\
            dec = otherClustersLocRADec[:,1]*u.deg,\
            distance=otherClustersLocRADec[:,2]*u.Mpc*h,frame='icrs')

otherClustersLocXYZ = np.array([\
    coordOtherClusters.cartesian.x.value,\
    coordOtherClusters.cartesian.y.value,\
    coordOtherClusters.cartesian.z.value]).T

[meanMassesOther,meanCentresOther,sigmaMassesOther,sigmaCentresOther,\
            clusterMassesOther,clusterCentresOther,clusterCounterpartsOther] = \
                tools.loadOrRecompute(\
                    data_folder + "mean_cluster_masses_other.p",\
                    getBORGClusterMassEstimates,snapNameList,\
                    otherClustersLocXYZ,equatorialXYZ,\
                    _recomputeData=recomputeData)

# Displacement:
dispOther = (clusterCentresOther - otherClustersLocXYZ)
distOther = np.sqrt(np.sum(dispOther**2,2))




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
#nVoidsToShow = 10
nVoidsToShow = len(filterToUse)
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
height = 0.55*textwidth
cropPoint = ((scale -1)/2)*np.array([width,height]) + np.array([0,0.09])
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



# All anti-halos:
if doSky:
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
            vmin=1e-2,vmax=1e2,legLoc = 'lower left',bbox_to_anchor = (-0.1,-0.2),\
            snapsort = snapsortList_all[ns],antihaloCentres = None,\
            figOut = figuresFolder + "/ah_match_sample_" + \
            str(ns) + plotFormat,\
            showFig=False,figsize = (scale*textwidth,scale*0.55*textwidth),\
            voidColour = colourListAll[ns],antiHaloLabel=labelListAll[ns],\
            bbox_inches = 'tight',galaxyAngles=equatorialRThetaPhi[:,1:],\
            galaxyDistances = equatorialRThetaPhi[:,0],showGalaxies=False,\
            voidAlpha = 0.6,margins=None)
        plt.show()

# All anti-halos in shells:
distanceShells = [0,50,100,150]
for k in range(1,len(distanceShells)):
    for ns in range(0,len(snapNumList)):
        plt.clf()
        distances = ahDistancesEquatorial[ns][laListAll[ns]]
        inShell = np.where((distances > distanceShells[k-1]) & \
            (distances <= distanceShells[k]))[0]
        plot.plotLocalUniverseMollweide(rCut,snapList[ns],\
            alpha_shapes = [asListAll[ns][x] for x in inShell],\
            largeAntihalos = np.array(laListAll[ns])[inShell],\
            hr=hrList[ns],\
            coordAbell = coordCombinedAbellSphere,\
            abellListLocation = clusterIndMain,\
            nameListLargeClusters = [name[0] for name in clusterNames],\
            ha = ha,va= va, annotationPos = annotationPos,\
            title = 'Antihalos within $' + \
            str(distanceShells[k-1]) + "< r/\\mathrm{\\,Mpc}h^{-1} <= " + \
            str(distanceShells[k]) + "$",\
            vmin=1e-2,vmax=1e2,legLoc = 'lower left',\
            bbox_to_anchor = (-0.1,-0.2),\
            snapsort = snapsortList_all[ns],antihaloCentres = None,\
            figOut = figuresFolder + "/ah_match_sample_" + \
            str(ns) + "_dist_" + str(distanceShells[k-1]) + "-" + \
            str(distanceShells[k]) + plotFormat,\
            showFig=False,figsize = (scale*textwidth,scale*0.55*textwidth),\
            voidColour = [colourListAll[ns][x] for x in inShell],\
            antiHaloLabel=[labelListAll[ns][x] for x in inShell],\
            bbox_inches = 'tight',galaxyAngles=equatorialRThetaPhi[:,1:],\
            galaxyDistances = equatorialRThetaPhi[:,0],showGalaxies=False,\
            voidAlpha = 0.6,margins=None)
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



# Simple plot of the galaxy distribution overlaid:
ns = 0
plot.plotLocalUniverseMollweide(100,snapList[ns],\
            alpha_shapes = None,\
            largeAntihalos = None,hr=None,\
            coordAbell = None,\
            abellListLocation = None,\
            nameListLargeClusters = None,\
            ha = ha,va= va, annotationPos = None,\
            vmin=1e-2,vmax=1e2,legLoc = 'lower left',bbox_to_anchor = (-0.1,-0.2),\
            snapsort = snapsortList_all[ns],antihaloCentres = None,\
            figOut = figuresFolder + "/mollweide_galaxies_" + \
            str(ns) + plotFormat,\
            showFig=False,figsize = (scale*textwidth,scale*0.55*textwidth),\
            voidColour = colourListAll[ns],antiHaloLabel=labelListAll[ns],\
            bbox_inches = bound_box,galaxyAngles=equatorialRThetaPhi[:,1:],\
            galaxyDistances = equatorialRThetaPhi[:,0],showGalaxies=False,\
            voidAlpha = 0.6,labelFontSize=12,legendFontSize=8,title="",dpi=600)

plt.show()




pointsToScatter = plot.filterPolarPointsToAnnulus(equatorialRThetaPhi[:,1:],\
            equatorialRThetaPhi[:,0],135,thickness=135)
healpy.mollview()
ax = plt.gca()
ax.set_autoscale_on(True)
healpy.graticule(color='grey')
plot.mollweideScatter(pointsToScatter,ax=ax)
plot.plotZoA(ax=ax,galacticCentreZOA = [-30,30],\
            nPointsZOA=200,bRangeCentre = [-10,10],bRange = [-5,5],\
            nPointsEdgeZOA = 21,\
            fc='grey',ec=None,alpha=0.5,label='Zone of Avoidance')

plt.show()


#-------------------------------------------------------------------------------
# CLUSTER MASS PLOT

# Mass comparison plot!
if doClusterMasses:
    plotData = massconstraintsplot.showClusterMassConstraints(\
        meanMasses,sigmaMasses,\
        figOut = figuresFolder,catFolder = "./catalogues/",h=h,Om0 = Om0,\
        savename = figuresFolder + "mass_constraints_plot.pdf",\
        savePlotData=True)

#-------------------------------------------------------------------------------
# MASS FUNCTIONS PLOT 135 VS 300

# Get optimal catalogues:
rSphere2 = 300
#muOpt = 0.925
#rSearchOpt = 0.5
muOpt = 0.95
rSearchOpt = 0.15
NWayMatch = False
refineCentres = True
[finalCat300,shortHaloList300,twoWayMatchList300,\
            finalCandidates300,finalRatios300,finalDistances300,\
            allCandidates300,candidateCounts300,allRatios300,\
            finalCombinatoricFrac300,finalCatFrac300,alreadyMatched300] = \
            tools.loadOrRecompute(data_folder + \
                "mcmc_catalogue_optimal_Ncat.p",\
                constructAntihaloCatalogue,\
                snapNumList,snapList=snapList,\
                snapListRev=snapListRev,\
                ahProps=ahProps,hrList=hrList,max_index=None,\
                twoWayOnly=True,blockDuplicates=True,\
                crossMatchThreshold = muOpt,distMax = rSearchOpt,\
                rSphere=rSphere2,massRange = [mLower1,mUpper1],\
                NWayMatch = NWayMatch,rMin=rMin,rMax=rMax,\
                additionalFilters = snrFilter,verbose=False,\
                refineCentres = refineCentres,_recomputeData=True,\
                sortBy = 'radius')

[finalCatRand,shortHaloListRand,twoWayMatchListRand,\
    finalCandidatesRand,finalRatiosRand,finalDistancesRand,\
    allCandidatesRand,candidateCountsRand,allRatiosRand,\
    finalCombinatoricFracRand,finalCatFracRand,alreadyMatchedRand] = \
    tools.loadOrRecompute(data_folder + \
        "random_catalogue_optimal_Ncat.p",\
        constructAntihaloCatalogue,\
        snapNumListUncon,snapList=snapListUn,\
        snapListRev=snapListRevUn,\
        ahProps=ahPropsUn,hrList=hrListUn,max_index=None,\
        twoWayOnly=True,blockDuplicates=True,\
        crossMatchThreshold = muOpt,distMax = rSearchOpt,\
        rSphere=rSphere2,massRange = [mLower1,mUpper1],\
        NWayMatch = NWayMatch,rMin=rMin,rMax=rMax,\
        additionalFilters = None,verbose=False,\
        refineCentres = refineCentres,_recomputeData=True,sortBy = 'radius')

# A few things needed for computing catalogue fractions:
# For MCMC samples:
[mcmcArray,num,N,NCAT,no_bias_params,bias_matrix,mean_field,\
        std_field,hmc_Elh,hmc_Eprior,hades_accept_count,\
        hades_attempt_count] = pickle.load(open(chainFile,"rb"))
snrField = mean_field**2/std_field**2
snrFieldLin = np.reshape(snrField,Nden**3)
grid = snapedit.gridListPermutation(Nden,perm=(2,1,0))
centroids = grid*boxsize/Nden + boxsize/(2*Nden)
positions = snapedit.unwrap(centroids - np.array([boxsize/2]*3),boxsize)
tree = scipy.spatial.cKDTree(snapedit.wrap(positions + boxsize/2,boxsize),\
    boxsize=boxsize)
nearestPointsList = [tree.query_ball_point(\
    snapedit.wrap(antihaloCentres[k] + boxsize/2,boxsize),\
    antihaloRadii[k],workers=-1) \
    for k in range(0,len(antihaloCentres))]
snrAllCatsList = [np.array([np.mean(snrFieldLin[points]) \
    for points in nearestPointsList[k]]) for k in range(0,len(snapNumList))]
snrFilter = [snr > snrThresh for snr in snrAllCatsList]

centralAntihalos = [tools.getAntiHalosInSphere(antihaloCentres[k],rSphere2,\
            filterCondition = (antihaloRadii[k] > rMin) & \
            (antihaloRadii[k] <= rMax) & (antihaloMasses[k] > mMin) & \
            (antihaloMasses[k] <= mMax) & snrFilter[k]) \
            for k in range(0,len(snapNumList))]
centralAntihaloRadii = [\
            antihaloRadii[k][centralAntihalos[k][0]] \
            for k in range(0,len(centralAntihalos))]
centralAntihaloMasses = [\
            antihaloMasses[k][centralAntihalos[k][0]] \
            for k in range(0,len(centralAntihalos))]
#sortedList = [np.flip(np.argsort(centralAntihaloRadii[k])) \
#                    for k in range(0,len(snapNumList))]

sortedList = [np.flip(np.argsort(centralAntihaloRadii[k])) \
                    for k in range(0,len(snapNumList))]


ahCounts = np.array([len(cahs[0]) for cahs in centralAntihalos])
max_index = np.max(ahCounts)
radiiListShort = [np.array([antihaloRadii[l][\
            centralAntihalos[l][0][sortedList[l][k]]] \
            for k in range(0,np.min([ahCounts[l],max_index]))]) \
            for l in range(0,len(snapNumList))]
massListShort = [np.array([antihaloMasses[l][\
            centralAntihalos[l][0][sortedList[l][k]]] \
            for k in range(0,np.min([ahCounts[l],max_index]))]) \
            for l in range(0,len(snapNumList))]
centresListShort = [np.array([antihaloCentres[l][\
            centralAntihalos[l][0][sortedList[l][k]],:] \
            for k in range(0,np.min([ahCounts[l],max_index]))]) \
            for l in range(0,len(snapNumList))]
# For randoms:
centralAntihalosUn = [tools.getAntiHalosInSphere(antihaloCentresUn[k],\
            rSphere2,filterCondition = (antihaloRadiiUn[k] > rMin) & \
            (antihaloRadiiUn[k] <= rMax) & (antihaloMassesUn[k] > mMin) & \
            (antihaloMassesUn[k] <= mMax)) \
            for k in range(0,len(snapNumListUncon))]
centralAntihaloRadiiUn = [\
            antihaloRadiiUn[k][centralAntihalosUn[k][0]] \
            for k in range(0,len(centralAntihalosUn))]
sortedListUn = [np.flip(np.argsort(centralAntihaloRadiiUn[k])) \
                    for k in range(0,len(snapNumListUncon))]
ahCountsUn = np.array([len(cahs[0]) for cahs in centralAntihalosUn])
max_indexUn = np.max(ahCountsUn)
radiiListShortUn = [np.array([antihaloRadiiUn[l][\
    centralAntihalosUn[l][0][sortedListUn[l][k]]] \
    for k in range(0,np.min([ahCountsUn[l],max_indexUn]))]) \
    for l in range(0,len(snapNumListUncon))]
massListShortUn = [np.array([antihaloMassesUn[l][\
    centralAntihalosUn[l][0][sortedListUn[l][k]]] \
    for k in range(0,np.min([ahCountsUn[l],max_indexUn]))]) \
    for l in range(0,len(snapNumListUncon))]

# Compute percentiles:
radiiListOpt = getPropertyFromCat(finalCat300,radiiListShort)
massListOpt = getPropertyFromCat(finalCat300,massListShort)
[radiiMeanOpt, radiiSigmaOpt]  = getMeanProperty(radiiListOpt)
[massMeanOpt, massSigmaOpt]  = getMeanProperty(massListOpt)
scaleFilter = [(radiiMeanOpt > radBins[k]) & \
    (radiiMeanOpt <= radBins[k+1]) \
    for k in range(0,len(radBins) - 1)]
radiiListCombUn = getPropertyFromCat(finalCatRand,radiiListShortUn)
massListCombUn = getPropertyFromCat(finalCatRand,massListShortUn)
[radiiListMeanUn,radiiListSigmaUn] = getMeanProperty(radiiListCombUn)
[massListMeanUn,massListSigmaUn] = getMeanProperty(massListCombUn)
[percentilesCat300, percentilesComb300] = getThresholdsInBins(\
    nBinEdges-1,cutScale,massListMeanUn,radiiListMeanUn,\
    finalCombinatoricFracRand,finalCatFracRand,\
    rLower,rUpper,mLower1,mUpper1,percThresh,massBins=massBins,\
    radBins=radBins)
#percentilesCat300 = [0.0 for k in range(0,7)]
#percentilesComb300 = [0.0 for k in range(0,7)]
finalCentresOptList = np.array([getCentresFromCat(\
    finalCat300,centresListShort,ns) \
    for ns in range(0,len(snapNumList))])
meanCentreOpt = np.nanmean(finalCentresOptList,0)
nearestPoints = tree.query_ball_point(\
    snapedit.wrap(meanCentreOpt + boxsize/2,boxsize),radiiMeanOpt,\
    workers=-1)
snrList = np.array([np.mean(snrFieldLin[points]) \
    for points in nearestPoints])
[combinedFilter300, meanCatFrac300, stdErrCatFrac300, \
    meanCombFrac300, stdErrCombFrac300] = applyCatalogueCuts(\
    finalCatFrac300,finalCombinatoricFrac300,percentilesCat300,\
    percentilesComb300,scaleFilter,snrList,snrThresh,catFracCut,\
    combFracCut,snrCut)

distances  = np.sqrt(np.sum(meanCentreOpt**2,1))
distFilter135 = (distances < 135)

thresholds = getAllThresholds(percentilesCat300,radBins,radiiMeanOpt)
thresholds = 0.0

#leftFilter = combinedFilter300 & distFilter135
#rightFilter = combinedFilter300
leftFilter = (radiiMeanOpt > 10) & (radiiMeanOpt <= 25) & distFilter135 & \
    (finalCatFrac300 > thresholds) & (snrList > snrThresh)
rightFilter = (radiiMeanOpt > 10) & (radiiMeanOpt <= 25) & \
    (finalCatFrac300 > thresholds) & (snrList > snrThresh)


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
    plot.massFunctionComparison(massMeanOpt[leftFilter],\
        massMeanOpt[rightFilter],volSphere135,nBins=nBins,\
        labelLeft = "Combined catalogue \n(well-constrained voids only)",\
        labelRight  ="Combined catalogue \n(well-constrained voids only)",\
        ylabel="Number of antihalos",savename=figuresFolder + \
        "mass_function_combined_300vs135.pdf",massLower=mLower,\
        ylim=[1,1000],Om0 = 0.3111,h=0.6766,sigma8=0.8128,ns=0.9667,\
        fontsize=8,massUpper = mUpper,\
        titleLeft = "Combined catalogue, $<135\\mathrm{Mpc}h^{-1}$",\
        titleRight = "Combined catalogue, $<300\\mathrm{Mpc}h^{-1}$",\
        volSimRight = volSphere,ylimRight=[1,1000],legendLoc="upper right")

# Histogram of catalogue fraction:
plt.clf()
inBinsMCMC = np.where(leftFilter)[0]
#inBinsRand = np.where((radiiListMeanUn > radBins[0]) & \
#    (radiiListMeanUn <= radBins[-1]))[0]
plt.hist(finalCatFrac300[inBinsMCMC],color=seabornColormap[0],alpha=0.5,\
    label='MCMC catalogue',bins=np.linspace(0.025,1+0.025,21))
#plt.hist(finalCatFracRand[inBinsRand],color=seabornColormap[1],alpha=0.5,\
#    label='Random catalogue',bins=np.linspace(0,1,21))
plt.xlabel('Catalogue fraction')
plt.ylabel('Number of voids')
plt.yscale('log')
plt.legend()
plt.savefig(figuresFolder + "catalogue_fraction_histograms_optimal.pdf")
plt.show()


# Histogram of radii:
plt.clf()
inBinsMCMC = np.where(leftFilter)[0]
#inBinsRand = np.where((radiiListMeanUn > radBins[0]) & \
#    (radiiListMeanUn <= radBins[-1]))[0]
plt.hist(radiiMeanOpt[inBinsMCMC],color=seabornColormap[0],alpha=0.5,\
    label='MCMC catalogue',bins=np.linspace(10,25,8))
#plt.hist(radiiListMeanUn[inBinsRand],color=seabornColormap[1],alpha=0.5,\
#    label='Random catalogue',bins=radBins)
plt.xlabel('Radius')
plt.ylabel('Number of voids')
plt.yscale('log')
plt.legend()
plt.savefig(figuresFolder + "radii_histograms_optimal.pdf")
plt.show()

# Scatter plot of catalogue fraction and radius:
plt.clf()
fig, ax = plt.subplots(figsize=(textwidth,textwidth))
inBinsMCMC = np.where(leftFilter)[0]
ax.scatter(finalCatFrac300[inBinsMCMC],radiiMeanOpt[inBinsMCMC])
ax.set_xlabel('Catalogue fraction')
ax.set_ylabel('Radius [$\\mathrm{Mpc}h^{-1}$]')
plt.savefig(figuresFolder + "radius_fcat_scatter.pdf")
plt.show()

#-------------------------------------------------------------------------------
# UNDERDENSE PROFILES PLOT


# Plot for the paper, using the new method:
doProfiles = True
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


#-------------------------------------------------------------------------------
# ANIMATIONS OF VOIDS
import glob
from PIL import Image
from descartes import PolygonPatch
snapList =  [pynbody.load(samplesFolder + "sample" + str(snapNum) + "/" \
        + "gadget_full_forward_512/snapshot_001") for snapNum in snapNumList]
snapListRev =  [pynbody.load(samplesFolder + "sample" + str(snapNum) + "/" \
        + "gadget_full_reverse_512/snapshot_001") for snapNum in snapNumList]

hrList = [snap.halos() for snap in snapListRev]


snapListUn = [pynbody.load("new_chain/unconstrained_samples/sample" + \
    str(num) + "/gadget_full_forward_512/snapshot_001") \
    for num in snapNumListUncon]
snapListRevUn = [pynbody.load(\
    "new_chain/unconstrained_samples/sample" + \
    str(num) + "/gadget_full_reverse_512/snapshot_001") \
    for num in snapNumListUncon]
hrListUn = [snap.halos() for snap in snapListRevUn]
ahPropsUn = [pickle.load(\
            open(snap.filename + ".AHproperties.p","rb")) \
            for snap in snapListUn]
antihaloCentresUn = [tools.remapAntiHaloCentre(props[5],boxsize) \
            for props in ahPropsUn]
antihaloMassesUn = [props[3] for props in ahPropsUn]
antihaloRadiiUn = [props[7] for props in ahPropsUn]

#distArr = np.arange(0,3,0.1)
mLower1 = 1e13
mUpper1 = 1e15
rSphere1 = 135
nBinEdges = 8
rLower = 10
rUpper = 20
percThresh = 99
percThreshList = [50,90,99]
cutScale = "radius"
catFracCut = True
combFracCut = False
snrCut = True
radBins = np.linspace(rLower,rUpper,nBinEdges)
massBins = 10**(np.linspace(np.log10(mLower1),np.log10(mUpper1),nBinEdges))
scaleBins = radBins

def mutualRadiusRatios(radii):
    nsamples = len(radii)
    ratioGrid = np.zeros((nsamples,nsamples))
    for i in range(0,nsamples):
        if (radii[i] > 0) & (np.isfinite(radii[i])):
            for j in range(0,nsamples):
                if (radii[j] > 0) & (np.isfinite(radii[j])):
                    ratioGrid[i,j] = np.min([radii[i],radii[j]])/\
                        np.max([radii[i],radii[j]])
    return ratioGrid

def mutualCentreRatio(radii,centres):
    nsamples = len(radii)
    ratioGrid = np.zeros((nsamples,nsamples))
    for i in range(0,nsamples):
        if (radii[i] > 0) & (np.all(np.isfinite(centres[i]))):
            for j in range(0,nsamples):
                if (radii[j] > 0) & (np.all(np.isfinite(centres[j]))):
                    distance = np.sqrt(np.sum((centres[i] - centres[j])**2))
                    ratioGrid[i,j] = distance/np.sqrt(radii[i]*radii[j])
    return ratioGrid


def getAllThresholds(percentiles,radBins,radii):
    scaleFilter = [(radii > radBins[k]) & \
        (radii <= radBins[k+1]) \
        for k in range(0,len(radBins) - 1)]
    thresholds = np.zeros(radii.shape)
    for filt, perc in zip(scaleFilter,percentiles):
        thresholds[filt] = perc
    return thresholds



def getSNRForVoidRealisations(finalCat,snrAllCatsList,ahNumbers):
    snrCat = np.zeros(finalCat.shape)
    for ns in range(0,len(snrAllCatsList)):
        haveVoids = np.where(finalCat[:,ns] >= 0)[0]
        snrCat[haveVoids,ns] = snrAllCatsList[ns][ahNumbers[ns][\
            finalCat[haveVoids,ns]-1]]
    return snrCat


# Get all voids within a box:
def getAllVoidsWithinBox(boxCentre,boxsize,centres):
    # Get limits:
    if np.isscalar(boxsize):
        xlim = np.array([boxCentre[0] - boxsize/2,boxCentre[0] + boxsize/2])
        ylim = np.array([boxCentre[1] - boxsize/2,boxCentre[1] + boxsize/2])
        zlim = np.array([boxCentre[2] - boxsize/2,boxCentre[2] + boxsize/2])
    elif len(boxsize == 3):
        xlim = np.array([boxCentre[0] - boxsize[0]/2,\
            boxCentre[0] + boxsize[0]/2])
        ylim = np.array([boxCentre[1] - boxsize[1]/2,\
            boxCentre[1] + boxsize[1]/2])
        zlim = np.array([boxCentre[2] - boxsize[2]/2,\
            boxCentre[2] + boxsize[2]/2])
    else:
        raise Exception("Invalid boxsize")
    centreFilter = (centres[:,0] > xlim[0]) & (centres[:,0] <= xlim[1]) & \
        (centres[:,1] > ylim[0]) & (centres[:,1] <= ylim[1]) & \
        (centres[:,2] > zlim[0]) & (centres[:,2] <= zlim[1])
    return centreFilter

#snapSortListRev = [np.argsort(snap['iord']) for snap in snapListRev]
snapSortList = [np.argsort(snap['iord']) for snap in snapList]
ahProps = [tools.loadPickle(name + ".AHproperties.p")\
            for name in snapNameList]
# Get optimal catalogues:
rSphere2 = 300
#muOpt = 0.95
#rSearchOpt = 0.15
muOpt = 0.75
rSearchOpt = 0.5
NWayMatch = False
refineCentres = True
sortBy = "radius"
enforceExclusive = True
#[finalCat300,shortHaloList300,twoWayMatchList300,\
#            finalCandidates300,finalRatios300,finalDistances300,\
#            allCandidates300,candidateCounts300,allRatios300,\
#            finalCombinatoricFrac300,finalCatFrac300,alreadyMatched300,\
#            iteratedCentresList,iteratedRadiiList] = \
#            constructAntihaloCatalogue(\
#                snapNumList,snapList=snapList,\
#                snapListRev=snapListRev,\
#                ahProps=ahProps,hrList=hrList,max_index=None,\
#                twoWayOnly=True,blockDuplicates=True,\
#                crossMatchThreshold = muOpt,distMax = rSearchOpt,\
#                rSphere=rSphere2,massRange = [mMin,mMax],\
#                NWayMatch = NWayMatch,rMin=rMin,rMax=rMax,\
#                additionalFilters = snrFilter,verbose=False,\
#                refineCentres=refineCentres,sortBy=sortBy,\
#                enforceExclusive=enforceExclusive)
snapNameList = [samplesFolder + "sample" + str(k) + "/" + snapnameNew \
    for k in snapNumList]
snapNameListRev = [samplesFolder + "sample" + str(k) + "/" + snapnameNewRev \
    for k in snapNumList]
cat300 = catalogue.combinedCatalogue(snapNameList,snapNameListRev,\
    muOpt,rSearchOpt,rSphere2,\
    ahProps=ahProps,hrList=hrList,max_index=None,\
    twoWayOnly=True,blockDuplicates=True,\
    massRange = [mMin,mMax],\
    NWayMatch = NWayMatch,rMin=rMin,rMax=rMax,\
    additionalFilters = snrFilter,verbose=False,\
    refineCentres=refineCentres,sortBy=sortBy,\
    enforceExclusive=enforceExclusive)
finalCat300 = cat300.constructAntihaloCatalogue()


[centresListShort,centralAntihalos,sortedList,ahCounts,max_index] = \
    [cat300.centresListShort,cat300.centralAntihalos,cat300.sortedList,\
    cat300.ahCounts,cat300.max_index]

snapNameListRand = [snap.filename for snap in snapListUn]
snapNameListRandRev = [snap.filename for snap in snapListRevUn]

cat300Rand = catalogue.combinedCatalogue(snapNameListRand,snapNameListRandRev,\
    muOpt,rSearchOpt,rSphere2,\
    ahProps=ahPropsUn,hrList=hrListUn,max_index=None,\
    twoWayOnly=True,blockDuplicates=True,\
    massRange = [mMin,mMax],\
    NWayMatch = NWayMatch,rMin=rMin,rMax=rMax,\
    additionalFilters = None,verbose=False,\
    refineCentres=refineCentres,sortBy=sortBy,\
    enforceExclusive=enforceExclusive)
finalCat300Rand = cat300Rand.constructAntihaloCatalogue()

ahNumbers = cat300.ahNumbers





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
        muOpt,rSearchOpt,rSphere2,\
        ahProps=[ahProps[k] for k in perm],\
        hrList=[hrList[k] for k in perm],max_index=None,\
        twoWayOnly=True,blockDuplicates=True,\
        massRange = [mMin,mMax],\
        NWayMatch = NWayMatch,rMin=rMin,rMax=rMax,\
        additionalFilters = [snrFilter[k] for k in perm],\
        verbose=False,\
        refineCentres=refineCentres,sortBy=sortBy,\
        enforceExclusive=enforceExclusive)
    finalCatPerm = catPerm.constructAntihaloCatalogue()
    randomlyOrderedCats.append(catPerm)
    [radiiPerm,sigmaRadiiPerm] = catPerm.getMeanProperty("radii")
    meanCentrePerm = catPerm.getMeanCentres()
    distancesPerm = np.sqrt(np.sum(meanCentrePerm**2,1))
    thresholdsPerm = getAllThresholds(percentilesCat300,radBins,radiiPerm)
    filterPerm = (radiiPerm > 10) & (radiiPerm <= 25) & \
        (distancesPerm < 135) & (catPerm.finalCatFrac > thresholdsPerm)
    numVoidsPerm[k] = np.sum(filterPerm)
    goodVoidsPerm.append(filterPerm)
    permutations.append(perm)

goodVoidsPerm300 = []
radiiAllPerms = []
for k in range(0,nPerms):
    catPerm = randomlyOrderedCats[k]
    [radiiPerm,sigmaRadiiPerm] = catPerm.getMeanProperty("radii")
    meanCentrePerm = catPerm.getMeanCentres()
    distancesPerm = np.sqrt(np.sum(meanCentrePerm**2,1))
    thresholdsPerm = getAllThresholds(percentilesCat300,radBins,radiiPerm)
    filterPerm = (radiiPerm > 10) & (radiiPerm <= 25) & \
        (distancesPerm < 300) & (catPerm.finalCatFrac > thresholdsPerm)
    goodVoidsPerm300.append(filterPerm)
    radiiAllPerms.append(radiiPerm)

massFunctionsPerm135 = [cat.getMeanProperty("mass")[0][filt] \
    for cat, filt in zip(randomlyOrderedCats,goodVoidsPerm)]
massFunctionsPerm300 = [cat.getMeanProperty("mass")[0][filt] \
    for cat, filt in zip(randomlyOrderedCats,goodVoidsPerm300)]



splitLists = [catalogue.getSplitList(cat300.finalCat[filter300,:][:,perm],\
    catTest.finalCat[filterTest,:]) \
    for perm, catTest, filterTest \
    in zip(permutations,randomlyOrderedCats,goodVoidsPerm)]

splitListsWide = [catalogue.getSplitList(\
    cat300.finalCat[filter300Wide,:][:,perm],\
    catTest.finalCat[filterTest,:]) \
    for perm, catTest, filterTest \
    in zip(permutations,randomlyOrderedCats,goodVoidsPerm300)]

def getBestCandidate(catRef,catTest,nVRef,allCands):
    if len(allCands) > 0:
        split = np.array([catalogue.getNumVoidsInCommon(\
            catRef[nVRef,:],catTest[test,:]) for test in allCands])
        return allCands[np.argmax(split)]
    else:
        return np.nan


def getBestNumberOfVoidsInCommon(catRef,catTest,nVRef,allCands):
    if len(allCands) > 0:
        split = np.array([catalogue.getNumVoidsInCommon(\
            catRef[nVRef,:],catTest[test,:]) for test in allCands])
        return split[np.argmax(split)]
    else:
        return 0

# Figure out the mapping between the catalogue and the permuted catalogues:
bestCandidatesList = [np.array([\
    getBestCandidate(cat300.finalCat[filter300,:][:,perm],\
    catTest.finalCat[filterTest,:],nV,splitList[nV]) \
    for nV in range(0,np.sum(filter300))]) \
    for perm, catTest, filterTest, splitList in \
    zip(permutations,randomlyOrderedCats,goodVoidsPerm,splitLists)]

# Figure out the mapping between the catalogue and the permuted catalogues:
bestCandidatesListWide = [np.array([\
    getBestCandidate(cat300.finalCat[filter300Wide,:][:,perm],\
    catTest.finalCat[filterTest,:],nV,splitList[nV]) \
    for nV in range(0,np.sum(filter300Wide))]) \
    for perm, catTest, filterTest, splitList in \
    zip(permutations,randomlyOrderedCats,goodVoidsPerm300,splitListsWide)]



# Count the number of voids in common with the counterparts:
inCommonFraction = [np.array([\
    getBestNumberOfVoidsInCommon(cat300.finalCat[filter300,:][:,perm],\
    catTest.finalCat[filterTest,:],nV,splitList[nV]) \
    for nV in range(0,np.sum(filter300))]) \
    for perm, catTest, filterTest, splitList in \
    zip(permutations,randomlyOrderedCats,goodVoidsPerm,splitLists)]
inCommonFractionWide = [np.array([\
    getBestNumberOfVoidsInCommon(cat300.finalCat[filter300Wide,:][:,perm],\
    catTest.finalCat[filterTest,:],nV,splitList[nV]) \
    for nV in range(0,np.sum(filter300))]) \
    for perm, catTest, filterTest, splitList in \
    zip(permutations,randomlyOrderedCats,goodVoidsPerm300,splitListsWide)]

meanInCommon = np.array([np.mean(counts) for counts in inCommonFraction])
meanInCommonExists = np.array([np.mean(counts[counts > 0]) \
    for counts in inCommonFraction])
fractionFound = np.array([np.sum(np.isfinite(x))/np.sum(filter300) \
    for x in bestCandidatesList])
isFound = np.vstack([np.isfinite(best) for best in bestCandidatesList]).T
isFoundWide = np.vstack([np.isfinite(best) \
    for best in bestCandidatesListWide]).T
foundFraction = np.sum(isFound,1)/nPerms
alwaysFound = np.all(isFound,1)
alwaysFoundWide = np.all(isFoundWide,1)
alwaysFoundFraction = np.sum(alwaysFound)/np.sum(filter300)


# Masses for objects which are always found:
massAlwaysFound135 = cat300.getMeanProperty("mass")[0][\
    np.where(filter300)[0][alwaysFound]]
massAlwaysFound300 = cat300.getMeanProperty("mass")[0][\
    np.where(filter300Wide)[0][alwaysFoundWide]]


# Which voids are the problematic ones:
binFiltersPerm = [[(radiiPerm[filt] >= radBins[k]) & \
    (radiiPerm[filt] < radBins[k+1]) \
    for k in range(0,len(radBins)-1)] \
    for radiiPerm, filt in zip(radiiAllPerms,goodVoidsPerm)]
binFilters = [(radiiMean300[filter300] >= radBins[k]) & \
    (radiiMean300[filter300] < radBins[k+1]) for k in range(0,len(radBins)-1)]
countsAlways = np.array([np.sum(alwaysFound & inBin) for inBin in binFilters])
countsNotAlways = np.array([np.sum(np.logical_not(alwaysFound) & inBin) \
    for inBin in binFilters])
countsAll = np.array([np.sum(inBin) for inBin in binFilters])
countsAllPerms = np.array([[np.sum(inBin) for inBin in binFilt] \
    for binFilt in binFiltersPerm],dtype=int)
countsAllMean = np.mean(countsAllPerms,0)

binWidths = (radBins[1:] - radBins[0:-1])
binX = (radBins[1:] + radBins[0:-1])/2

plt.clf()
plt.bar(binX,countsNotAlways,width=binWidths,alpha=0.5,\
    color=seabornColormap[0],label='Not in all catalogues')
plt.bar(binX,countsAlways,width=binWidths,alpha=0.5,\
    color=seabornColormap[1],label='In all catalogues',\
    bottom = countsNotAlways)
plt.xlabel('Radius, $\\mathrm{Mpc}h^{-1}$')
plt.ylabel('Number of voids')
plt.legend()
plt.savefig(figuresFolder + "permutation_consistency.pdf")
plt.show()


# Fraction of mean number of voids in each bin:
plt.clf()
countRange = np.array(scipy.stats.poisson.interval(0.68,countsNotAlways))
countErrors = np.array([countsNotAlways - countRange[0],\
    countRange[1] - countsNotAlways])

plt.errorbar(binX,countsNotAlways/countsAllMean,\
    yerr = countErrors/countsAllMean,\
    color=seabornColormap[0],label='Not in all permutations',marker='x',\
    linestyle='-')
#plt.plot(binX,countsAlways/countsAllMean,\
#    color=seabornColormap[1],label='In all permutations',marker='x',\
#    linestyle='-')
plt.xlabel('Radius, $\\mathrm{Mpc}h^{-1}$')
plt.ylabel('Fraction of mean void count')
plt.legend()
plt.savefig(figuresFolder + "permutation_consistency_fractions.pdf")
plt.show()


# Scatter plot of found fraction vs catalogue fraction:

plt.clf()
fit = np.polyfit(cat300.finalCatFrac[filter300][np.logical_not(alwaysFound)],\
    foundFraction[np.logical_not(alwaysFound)],1)
plt.scatter(cat300.finalCatFrac[filter300][np.logical_not(alwaysFound)],\
    foundFraction[np.logical_not(alwaysFound)])
xpoints = np.linspace(0,1,100)
plt.plot(xpoints,xpoints*fit[0] + fit[1],\
    label="Fit, " + ("%.2g" % fit[0]) + "x + " + ("%.2g" % fit[1]))
plt.legend()
plt.xlabel('Catalogue fraction')
plt.ylabel('Permutation fraction')
plt.savefig(figuresFolder + "permutation_vs_catalogue_fraction.pdf")
plt.show()




snapNameTest = ["data_for_tests/reference_constrained/sample" + str(k) + \
    "/gadget_full_forward/snapshot_001" for k in [2791,3250,5511]]
snapNameRevTest = ["data_for_tests/reference_constrained/sample" + str(k) + \
    "/gadget_full_reverse/snapshot_001" for k in [2791,3250,5511]]
catTest = catalogue.combinedCatalogue(snapNameTest,snapNameRevTest,0.9,0.5,135)
computed = catTest.constructAntihaloCatalogue()
#[finalCat300Rand,shortHaloList300Rand,twoWayMatchList300Rand,\
#            finalCandidates300Rand,finalRatios300Rand,finalDistances300Rand,\
#            allCandidates300Rand,candidateCounts300Rand,allRatios300Rand,\
#            finalCombinatoricFrac300Rand,finalCatFrac300Rand,
#            alreadyMatched300Rand,_,_] = \
#            constructAntihaloCatalogue(\
#                snapNumListUncon,snapList=snapListUn,\
#                snapListRev=snapListRevUn,\
#                ahProps=ahPropsUn,hrList=hrListUn,max_index=None,\
#                twoWayOnly=True,blockDuplicates=True,\
#                crossMatchThreshold = muOpt,distMax = rSearchOpt,\
#                rSphere=rSphere2,massRange = [mMin,mMax],\
#                NWayMatch = NWayMatch,rMin=rMin,rMax=rMax,\
#                additionalFilters = None,verbose=False,\
#                refineCentres=refineCentres,sortBy=sortBy,\
#                enforceExclusive=enforceExclusive)

# Mass functions:

#radiiListShort = getShortenedQuantity(antihaloRadii,centralAntihalos,\
#            centresListShort,sortedList,ahCounts,max_index)
#massListShort = getShortenedQuantity(antihaloMasses,centralAntihalos,\
#            centresListShort,sortedList,ahCounts,max_index)

radiiListShort = cat300.radiusListShort
massListShort = cat300.massListShort
finalCentres300List = cat300.getAllCentres()
meanCentre300 = cat300.getMeanCentres()
#catFractionsOpt = np.array([len(np.where(x > 0)[0])/len(snapNumList) \
#    for x in finalCatOpt])
#catFractionsOpt = finalCombinatoricFracOpt

# Using old settings:


# Construct the final Catalogue using optimal values:
muOpt = 0.9
rSearchOpt = 1
rSphere = 300
NWayMatch = True

diffMap = [np.setdiff1d(np.arange(0,len(snapNumList)),[k]) \
    for k in range(0,len(snapNumList))]

[finalCatOpt,shortHaloListOpt,twoWayMatchListOpt,finalCandidatesOpt,\
    finalRatiosOpt,finalDistancesOpt,allCandidatesOpt,candidateCountsOpt,\
    allRatiosOpt,finalCombinatoricFracOpt,finalCatFracOpt,alreadyMatched,\
    _,_] = \
    constructAntihaloCatalogue(snapNumList,snapList=snapList,\
    snapListRev=snapListRev,ahProps=ahProps,hrList=hrList,max_index=None,\
    twoWayOnly=True,blockDuplicates=True,\
    crossMatchThreshold = muOpt,distMax = rSearchOpt,rSphere=rSphere,\
    massRange = [mMin,mMax],NWayMatch = NWayMatch,rMin=rMin,rMax=rMax,\
    additionalFilters = snrFilter,sortBy = 'radius')

# Random catalogue version:
[finalCatOptRand,shortHaloListOptRand,twoWayMatchListOptRand,\
    finalCandidatesOptRand,finalRatiosOptRand,finalDistancesOptRand,\
    allCandidatesOptRand,candidateCountsOptRand,allRatiosOptRand,\
    finalCombinatoricFracOptRand,finalCatFracOptRand,alreadyMatchedRand,\
    _,_] = \
    constructAntihaloCatalogue(snapNumList,snapList=snapListUn,\
    snapListRev=snapListRevUn,ahProps=ahPropsUn,hrList=hrListUn,max_index=None,\
    twoWayOnly=True,blockDuplicates=True,\
    crossMatchThreshold = muOpt,distMax = rSearchOpt,rSphere=rSphere,\
    massRange = [mMin,mMax],NWayMatch = NWayMatch,rMin=rMin,rMax=rMax,\
    additionalFilters = None,sortBy = 'radius')

[centresListShortUn,centralAntihalosUn,sortedListUn,ahCountsUn,max_indexUn] = \
        computeShortCentresList(snapNumListUncon,antihaloCentresUn,\
            antihaloRadiiUn,antihaloMassesUn,rSphere2,rMin,rMax,\
            massRange=massRange,additionalFilters=None,\
            sortBy = 'radius',max_index=None)

radiiListShortUn = getShortenedQuantity(antihaloRadiiUn,centralAntihalosUn,\
            centresListShortUn,sortedListUn,ahCountsUn,max_indexUn)
massListShortUn = getShortenedQuantity(antihaloMassesUn,centralAntihalosUn,\
            centresListShortUn,sortedListUn,ahCountsUn,max_indexUn)


# Percentiles:
# Compute percentiles:
radiiListOpt = getPropertyFromCat(finalCatOpt,radiiListShort)
massListOpt = getPropertyFromCat(finalCatOpt,massListShort)
[radiiMeanOpt, radiiSigmaOpt]  = getMeanProperty(radiiListOpt)
[massMeanOpt, massSigmaOpt]  = getMeanProperty(massListOpt)
scaleFilter = [(radiiMeanOpt > radBins[k]) & \
    (radiiMeanOpt <= radBins[k+1]) \
    for k in range(0,len(radBins) - 1)]
radiiListCombUn = getPropertyFromCat(finalCatOptRand,radiiListShortUn)
massListCombUn = getPropertyFromCat(finalCatOptRand,massListShortUn)
[radiiListMeanUn,radiiListSigmaUn] = getMeanProperty(radiiListCombUn)
[massListMeanUn,massListSigmaUn] = getMeanProperty(massListCombUn)
[percentilesCatOpt, percentilesCombOpt] = getThresholdsInBins(\
    nBinEdges-1,cutScale,massListMeanUn,radiiListMeanUn,\
    finalCombinatoricFracOptRand,finalCatFracOptRand,\
    rLower,rUpper,mLower1,mUpper1,percThresh,massBins=massBins,\
    radBins=radBins)

radiiListComb300Un = cat300Rand.getAllProperties("radii")
massListComb300Un = cat300Rand.getAllProperties("mass")
[radiiListMean300Un,radiiListSigma300Un] = cat300Rand.getMeanProperty("radii")
[massListMean300Un,massListSigma300Un] = cat300Rand.getMeanProperty("mass")
[percentilesCat300, percentilesComb300] = getThresholdsInBins(\
    nBinEdges-1,cutScale,massListMean300Un,radiiListMean300Un,\
    cat300Rand.finalCombinatoricFrac,cat300Rand.finalCatFrac,\
    rLower,rUpper,mLower1,mUpper1,percThresh,massBins=massBins,\
    radBins=radBins)



radiiList300 = getPropertyFromCat(cat300.finalCat,\
    cat300Rand.radiusListShort)
massList300 = getPropertyFromCat(cat300.finalCat,\
    cat300.massListShort)
[radiiListMean300,radiiListSigma300] = getMeanProperty(radiiList300)
[massListMean300,massListSigma300] = getMeanProperty(massList300)
[percentilesCat300mcmc, percentilesComb300mcmc] = getThresholdsInBins(\
    nBinEdges-1,cutScale,massListMean300,radiiListMean300,\
    cat300.finalCombinatoricFrac,cat300.finalCatFrac,\
    rLower,rUpper,mLower1,mUpper1,percThresh,massBins=massBins,\
    radBins=radBins)

plt.clf()
plt.plot(plot.binCentres(radBins),percentilesCat300,label='Random catalogues')
plt.plot(plot.binCentres(radBins),percentilesCat300mcmc,label='MCMC catalogues')
plt.xlabel('$R [\\mathrm{Mpc}h^{-1}]$')
plt.ylabel('Catalogue fraction')
plt.title('99th percentile catalogue fractions, $\\mu_R = 0.75, \\mu_S = 0.5$')
plt.legend()
plt.savefig(figuresFolder + "percentiles.pdf")


# Get SNR per catalogue:


nV = np.where(finalCat300[:,5] == 2)[0][0]
nV1 = np.where(finalCatOpt[:,5] == 2)[0][0]

centreRat = mutualCentreRatio(radiiList300[nV],finalCentres300List[:,nV,:])
centreRat1 = mutualCentreRatio(radiiListOpt[nV1],finalCentresOptList[:,nV1,:])
radiiRat = mutualRadiusRatios(radiiList300[nV])
radiiRat1 = mutualRadiusRatios(radiiListOpt[nV1])

snrCat300 = getSNRForVoidRealisations(finalCat300,snrAllCatsList,ahNumbers)
snrCatOpt = getSNRForVoidRealisations(finalCatOpt,snrAllCatsList,ahNumbers)
haveVoids300 = np.where(finalCat300 >= 0)
haveVoidsOpt = np.where(finalCatOpt >= 0)

# Mean SNR:

finalCentresOptList = np.array([getCentresFromCat(\
    finalCatOpt,centresListShort,ns) for ns in range(0,len(snapNumList))])

meanCentreOpt = np.nanmean(finalCentresOptList,0)
stdCentreOpt = np.nanstd(finalCentresOptList,0)
dispCentreOpt = finalCentresOptList - meanCentreOpt
distCentreOpt = np.sqrt(np.sum(dispCentreOpt**2,2))

#catFractionsOpt = np.array([len(np.where(x > 0)[0])/len(snapNumList) \
#    for x in finalCatOpt])
catFractionsOpt = finalCatFracOpt
#catFractionsOpt = finalCombinatoricFracOpt


nearestPointsOpt = tree.query_ball_point(\
        snapedit.wrap(meanCentreOpt + boxsize/2,boxsize),\
        radiiMeanOpt,workers=-1)
snrMeanOpt = np.sqrt(np.array([np.mean(snrFieldLin[points]) \
        for points in nearestPointsOpt]))


radiiListOpt = getRadiiFromCat(finalCatOpt,radiiListShort)
massListOpt = getRadiiFromCat(finalCatOpt,massListShort)
[radiiMeanOpt, radiiSigmaOpt]  = getMeanProperty(radiiListOpt)
[massMeanOpt, massSigmaOpt]  = getMeanProperty(massListOpt)

fractionalDistCentre = distCentreOpt/radiiMeanOpt
meanFractionalDist = np.nanmean(fractionalDistCentre,0)

# Get SNR per catalogue:

radiiList300 = getRadiiFromCat(cat300.finalCat,cat300.radiusListShort)
massList300 = getRadiiFromCat(cat300.finalCat,cat300.massListShort)
[radiiMean300, radiiSigma300]  = getMeanProperty(radiiList300)
[massMean300, massSigma300]  = getMeanProperty(massList300)
distances300 = np.sqrt(np.sum(meanCentre300**2,1))
thresholds300 = getAllThresholds(percentilesCat300,radBins,radiiMean300)

# Mass functions:
leftFilter = (radiiMean300 > 10) & (radiiMean300 <= 25) & \
    (distances300 < 135) & (cat300.finalCatFrac > thresholds300)
rightFilter = (radiiMean300 > 10) & (radiiMean300 <= 25) & \
    (distances300 < 300) & (cat300.finalCatFrac > thresholds300)

plt.clf()
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

# Mass functions showing only voids that are always found:
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
    plot.massFunctionComparison(massFunctionsPerm135,\
        massFunctionsPerm300,4*np.pi*135**3/3,nBins=nBins,\
        labelLeft = "Combined catalogue \n(well-constrained voids only)",\
        labelRight  ="Combined catalogue \n(well-constrained voids only)",\
        ylabel="Number of antihalos",savename=figuresFolder + \
        "mass_function_permutations_300vs135_test.pdf",massLower=mLower,\
        ylim=[1,1000],Om0 = 0.3111,h=0.6766,sigma8=0.8128,ns=0.9667,\
        fontsize=8,massUpper = mUpper,\
        titleLeft = "Combined catalogue, $<135\\mathrm{Mpc}h^{-1}$",\
        titleRight = "Combined catalogue, $<300\\mathrm{Mpc}h^{-1}$",\
        volSimRight = 4*np.pi*300**3/3,ylimRight=[1,1000],\
        legendLoc="upper right",massErrors=True)
    plot.massFunctionComparison(massAlwaysFound135,\
        massAlwaysFound300,4*np.pi*135**3/3,nBins=nBins,\
        labelLeft = "Combined catalogue \n(well-constrained voids only)",\
        labelRight  ="Combined catalogue \n(well-constrained voids only)",\
        ylabel="Number of antihalos",savename=figuresFolder + \
        "mass_function_always_found_300vs135_test.pdf",massLower=mLower,\
        ylim=[1,1000],Om0 = 0.3111,h=0.6766,sigma8=0.8128,ns=0.9667,\
        fontsize=8,massUpper = mUpper,\
        titleLeft = "Combined catalogue, $<135\\mathrm{Mpc}h^{-1}$",\
        titleRight = "Combined catalogue, $<300\\mathrm{Mpc}h^{-1}$",\
        volSimRight = 4*np.pi*300**3/3,ylimRight=[1,1000],\
        legendLoc="upper right",massErrors=False)

# Test for void splitting:
nVTest = 0
locator = [np.where((cat300.finalCat[:,k] == finalCatOpt[nVTest][k]) & \
    (cat300.finalCat[:,k] != -1)) \
    for k in range(0,len(snapNumList))]
splitEntries = np.unique(np.hstack(locator))

splitList = []
for nVTest in range(0,len(finalCatOpt)):
    locator = [np.where((cat300.finalCat[:,k] == finalCatOpt[nVTest][k]) & \
        (cat300.finalCat[:,k] != -1)) for k in range(0,len(snapNumList))]
    splitEntries = np.unique(np.hstack(locator))
    splitList.append(splitEntries)

numSplit = np.array([len(x) for x in splitList],dtype=int)

distancesOpt = np.sqrt(np.sum(meanCentreOpt**2,1))
filterOpt = (radiiMeanOpt > 10) & (radiiMeanOpt <= 25) & (distancesOpt < 135)
thresholdsOpt = getAllThresholds(percentilesCatOpt,radBins,radiiMeanOpt)

#filterOptGood = filterOpt & (finalCatFracOpt > thresholdsOpt) & \
#    (snrMeanOpt > 5) & (finalCatFracOpt > 0.7) & (meanFractionalDist < 0.5)
filterOptGood = filterOpt & (finalCatFracOpt > 0.45) & (meanFractionalDist < 0.3)
#filterOptGood = filterOpt & (meanFractionalDist < 0.15)


filter300 = (radiiMean300 > 10) & (radiiMean300 <= 25) & \
    (distances300 < 135) & (cat300.finalCatFrac > thresholds300)
filter300Wide = (radiiMean300 > 10) & (radiiMean300 <= 25) & \
    (distances300 < 300) & (cat300.finalCatFrac > thresholds300)
allInRadius = (radiiMean300 > 10) & (radiiMean300 <= 25) & \
    (distances300 < 135)
allSignificantInRadius = filter300
allInsignificantInRadius = allInRadius & \
    (np.logical_not((cat300.finalCatFrac > thresholds300)))


# Distribution of catalogue fraction:
plt.clf()
plt.hist(cat300.finalCatFrac[allInRadius],bins=np.arange(0.075,1.1,0.05),\
    color=seabornColormap[0],alpha=0.5,density=False,label='MCMC catalogue')
plt.xlabel('Catalogue fraction')
#plt.ylabel('Probability Density')
plt.ylabel('No. of voids')
plt.savefig(figuresFolder + "catalogue_fraction_distribution.pdf")
plt.show()


# Save the high-confidence catalogue:
#tools.savePickle([meanCentreOpt[filterOptGood],filterOptGood,\
#    radiiMeanOpt[filterOptGood],finalCatOpt[filterOptGood],\
#    centresListShort,radiiListShort,massListShort,sortedList],\
#    "high_confidence_voids.p")

[meanCentreOptHC,filterOptGoodHC,radiiMeanOptHC,finalCatOptHC,\
    centresListShort,radiiListShort,massListShort,sortedList] = \
    tools.loadPickle("high_confidence_voids.p")


plt.clf()
plt.hist(numSplit[filterOpt],\
    bins = np.arange(np.min(numSplit)-0.5,np.max(numSplit)+1))
plt.xlabel("Number of voids split between in new catalogue")
plt.ylabel("Number of voids in old catalogue")
plt.savefig(figuresFolder  + "numSplit_histogram.pdf")


numSplitRefined = tools.loadPickle(figuresFolder + "numSplit_refined.p")
numSplitUnrefined = tools.loadPickle(figuresFolder + "numSplit_unrefined.p")

plt.clf()
plt.hist(numSplitRefined[filterOpt],\
    bins = np.arange(np.min(numSplit)-0.5,np.max(numSplit)+1),\
    label="with iterative approach",alpha=0.5,color=seabornColormap[0])
plt.hist(numSplitUnrefined[filterOpt],\
    bins = np.arange(np.min(numSplit)-0.5,np.max(numSplit)+1),\
    label="without iterative approach",alpha=0.5,color=seabornColormap[1])
plt.xlabel("Number of voids split between in new catalogue")
plt.ylabel("Number of voids in old catalogue")
plt.legend()
plt.savefig(figuresFolder  + "numSplit_histogram_compared.pdf")
[matching,notMatching,failed] = analyseSplit(finalCatOpt,finalCat300,0,0)

lowestMatch = np.array([lowestOrNothing(x) for x in splitList],dtype=int)
numberSplitBetween = np.array([len(x) for x in splitList])

commonVoidsList = []
for k in range(0,len(finalCatOpt)):
    numShared = np.array([getNumVoidsInCommon(finalCatOpt[k],finalCat300[l]) \
        for l in splitList[k]])
    commonVoidsList.append(numShared)

goodVoids = np.where(filterOptGood)[0]
splitListGood = [splitList[k] for k in goodVoids]
commonVoidsListGood = [commonVoidsList[k] for k in goodVoids]
bestCandidates = -np.ones(len(commonVoidsListGood),dtype=int)
for k in range(0,len(commonVoidsListGood)):
    if len(commonVoidsListGood[k]):
        bestCandidates[k] = splitListGood[k][np.where(commonVoidsListGood[k] == \
             np.max(commonVoidsListGood[k]))[0][0]]

fractionalDistCentre = distCentreOpt/radiiMeanOpt
meanFractionalDist = np.nanmean(fractionalDistCentre,0)


# Split list, looking only at significant voids:
indFilter300 = np.where(filter300)[0]
splitListGood = getSplitList(finalCatOpt[filterOptGood],cat300.finalCat)
splitListGoodFiltered = getSplitList(finalCatOpt[filterOptGood],\
    cat300.finalCat[filter300])

splitListGoodFilteredInverse = getSplitList(finalCat300[filter300],finalCatOpt[filterOpt])


# Test post-processing:
keepVoid = removeOverlappingVoids(finalCat300,radiiMean300,meanCentre300,\
    0.75,0.5,boxsize)
splitListGoodPostProcessed = getSplitList(finalCatOpt[filterOptGood],\
    finalCat300[keepVoid])
splitListGoodPostProcessedAndFiltered = getSplitList(\
    finalCatOpt[filterOptGood],finalCat300[keepVoid & filter300])

slCountGood = np.array([len(x) for x in splitListGood])
slCountGoodFilt = np.array([len(x) for x in splitListGoodFiltered])
slCountPP = np.array([len(x) for x in splitListGoodPostProcessed])
slCountPPandF = np.array([len(x) \
    for x in splitListGoodPostProcessedAndFiltered])


# Testing void properties:
nVground = 4
nV1 = 5
nV2 = 95
groundTruth = finalCatOpt[filterOptGood][nVground]
void1 = finalCat300[filter300][nV1]
void2 = finalCat300[filter300][nV2]

void1rad = radiiMean300[filter300][nV1]
void2rad = radiiMean300[filter300][nV2]
void1cen = meanCentre300[filter300][nV1]
void2cen = meanCentre300[filter300][nV2]
void1AllCentres = np.array([finalCentres300List[ns][indFilter300[nV1]] \
    for ns in range(0,len(snapList))])
void2AllCentres = np.array([finalCentres300List[ns][indFilter300[nV2]] \
    for ns in range(0,len(snapList))])
void1AllRadii = radiiList300[indFilter300[nV1]]
void2AllRadii = radiiList300[indFilter300[nV2]]



def getRatios(allRadii,allCentres,referenceRadius,referenceCentre):
    numCats = len(allRadii)
    if len(allCentres.shape) == 1:
        distances = np.sqrt(np.sum((allCentres - referenceCentre)**2,1))
    else:
        distances = [np.sqrt(np.sum((centres - referenceCentre)**2,1)) \
            for centres in allCentres]
    # Get the radius ratio:
    radiusRatio = np.zeros(allRadii.shape)
    if len(allCentres.shape) == 1:
        greaterRad = (allRadii >= referenceRadius)
        lesserRad = (allRadii < referenceRadius) & (allRadii > 0)
        radiusRatio[greaterRad] = \
            referenceRadius/allRadii[greaterRad]
        radiusRatio[lesserRad] = \
            allRadii[lesserRad]/referenceRadius
    else:
        greaterRad = [(radii >= referenceRadius) \
            for radii in allRadii]
        lesserRad = [(radii < referenceRadius) & (radii > 0) \
            for radii in allRadii]
        for k in range(0,len(greaterRad)):
            radiusRatio[]
    # Get the distnace ratio:
    distanceRatio = distances/np.sqrt(allRadii*referenceRadius)
    return [radiusRatio,distanceRatio]


[void2RadiusRatio1,void2DistanceRatio1] = getRatios(void2AllRadii,\
    void2AllCentres,void1rad,void1cen)

[void2RadiusRatioOpt4,void2DistanceRatioOpt4] = getRatios(void2AllRadii,\
    void2AllCentres,radiiMeanOpt[filterOptGood][nVground],\
    meanCentreOpt[filterOptGood][nVground])


void2SuccessRadius1 = (void2RadiusRatio1 >= 0.75)
void2SuccessDistance1 = (void2DistanceRatio1 <= 0.5)
void2Success1 = (void2SuccessRadius1) & (void2SuccessDistance1)

void2SuccessRadiusOpt4 = (void2RadiusRatioOpt4 >= 0.75)
void2SuccessDistanceOpt4 = (void2DistanceRatioOpt4 <= 0.5)
void2SuccessOpt4 = (void2SuccessRadiusOpt4) & (void2SuccessDistanceOpt4)

splitAnalysis2 = compareVoids(groundTruth,void2)
splitAnalysis1 = compareVoids(groundTruth,void1)

# Success as a function of iterations:
iteratedCentres = cat300.iteratedCentresList[indFilter300[nV1]]
iteratedRadii = cat300.iteratedRadiiList[indFilter300[nV1]]

radiusSuccess = np.zeros((len(iteratedCentres),len(splitAnalysis2[0])),\
    dtype=bool)
distanceSuccess = np.zeros((len(iteratedCentres),len(splitAnalysis2[0])),\
    dtype=bool)
muR = 0.75
muS = 0.5
for k in range(0,len(iteratedCentres)):
    [radiusRatio,distanceRatio] = getRatios(void2AllRadii,\
        void2AllCentres,iteratedRadii[k],iteratedCentres[k])
    successRad = (radiusRatio >= muR)
    successDist = (distanceRatio <= muS)
    radiusSuccess[k,:] = successRad[splitAnalysis2[0]]
    distanceSuccess[k,:] = successDist[splitAnalysis2[0]]

totalSuccess = (radiusSuccess & distanceSuccess)


[radiusRatioShort,distanceRatioShort] = getRatios(cat300.radiiListShort)

# Histogram of distances:
plt.clf()
plotDensity = True
hist = plt.hist(fractionalDistCentre[:,filterOptGood].flatten(),\
    bins = np.linspace(0,5,51),alpha=0.5,color=seabornColormap[0],\
    density=plotDensity,label='Selected Voids')
hist = plt.hist(fractionalDistCentre.flatten(),\
    bins = np.linspace(0,5,51),alpha=0.5,color=seabornColormap[1],\
    density = plotDensity,label='All voids in old catalogue')
plt.xlabel('(Distance from mean centre)/(mean radius)')
if plotDensity:
    plt.ylabel('Probability density')
else:
    plt.ylabel('Number of voids')

plt.legend()
plt.savefig(figuresFolder + "displacement_distribution.pdf")
plt.show()

radiusRatio = radiiListOpt.T/radiiMeanOpt

# Histogram of radii:
plt.clf()
plotDensity = True
hist = plt.hist(radiusRatio[:,filterOptGood].flatten(),\
    bins = np.linspace(0,2,51),alpha=0.5,color=seabornColormap[0],\
    density=plotDensity,label='Selected Voids')
hist = plt.hist(radiusRatio.flatten(),\
    bins = np.linspace(0,2,51),alpha=0.5,color=seabornColormap[1],\
    density = plotDensity,label='All voids in old catalogue')
plt.xlabel('Radius/(Mean radius)')
if plotDensity:
    plt.ylabel('Probability density')
else:
    plt.ylabel('Number of voids')

plt.legend()
plt.savefig(figuresFolder + "radius_distribution.pdf")
plt.show()



# Scatter of SNR vs splitting:
plt.clf()
plt.scatter(snrMeanOpt,numSplitRefined)
plt.xlabel('Signal/Noise')
plt.ylabel('Number of voids split between')
plt.savefig(figuresFolder + "snr_vs_split.pdf")


# Void Filter:
distance300 = np.sqrt(np.sum(meanCentre300**2,1))
distFilter = (distance300 < 135)
voidFilter300 = np.where((radiiMean300 >= 10) & (radiiMean300 < 25) & \
    (finalCatFrac300 > 0.1))[0]
voidFilter135 = np.where((radiiMean300 >= 10) & (radiiMean300 < 25) & \
    (finalCatFrac300 > 0.1) & distFilter)[0]

catToPlot = finalCat300
#catToPlot = finalCatOpt
radiiToPlot = radiiList300
#radiiToPlot = radiiListOpt
indList = []
nV = 0
#nV = 3858
#title = "Old run, void " + str(nV+1) + \
#    " ($\\mu_R = 0.9,\\mu_S = 1.0$, N-way-match = True)"
title = "New run, void " + str(nV + 1) + \
    " ($\\mu_R = 0.925,\\mu_S = 0.5$, N-way-match = True)"
#nV = np.where(finalCat300[:,5] == 2)[0][0] # Intermediate SNR
#nV = np.where(finalCat300[:,0] == 2)[0][0] # Intermediate SNR
#nV = np.where(finalCat300[:,0] == 785)[0][0] # V high SNR
#nV = np.where(finalCat300[:,0] == 404)[0][0] # High SNR
#nV = np.where(finalCat300[:,0] == 800)[0][0] # Low SNR
#nV = 202
#nV = 33605
#
#nV = 1


densitiesHR = [np.fromfile("new_chain/sample" + str(snap) + \
        "/gadget_full_forward_512/snapshot_001.a_den",\
    dtype=np.float32) for snap in snapNumList]
densities256 = [np.reshape(density,(256,256,256),order='C') \
    for density in densitiesHR]
densities256F = [np.reshape(density,(256,256,256),order='F') \
    for density in densitiesHR]

meanDensity256 = np.mean(densities256,0)

def plotVoidAnimation(nV,centralAntihalos,sortedList,catToPlot,radiiToPlot,\
        snapNumList,centresListShort,radiiListShort,densities,snapList,\
        hrList,snapSortList,\
        Lbox = 100,axis=2,Om0=0.3111,nPix=32,N=256,alphaVal=0.2,vmin=1/1000,\
        vmax=1000,thickness=None,cmap='PuOr_r',showOtherCentres = True,\
        minMuR = 0.75,figuresFolder='./',title=None,namePrefix="",\
        nameSuffix="",framePrefix=None,frameSuffix=None,rMin=10,rMax=25):
    indList = []
    numCats = len(snapList)
    if framePrefix is None:
        framePrefix = namePrefix
    if frameSuffix is None:
        frameSuffix = nameSuffix
    for l in range(0,numCats):
        if catToPlot[nV][l] > -1:
            indList.append(centralAntihalos[l][0][sortedList[l][catToPlot[nV][l]-1]])
        else:
            indList.append(-1)
    centresArray = []
    haveCentreCount = 0
    for l in range(0,numCats):
        if catToPlot[nV][l] > -1:
            centresArray.append(centresListShort[l][catToPlot[nV,l] - 1])
    #meanCentre = np.flip(np.mean(centresArray,0)) + boxsize/2
    meanCentre = np.mean(centresArray,0)
    stdCentre = np.std(centresArray,0)
    rhoBar = Om0*2.7754e11
    binEdgesX = np.linspace(meanCentre[0] - Lbox/2,meanCentre[0] + Lbox/2,nPix)
    binEdgesY = np.linspace(meanCentre[1] - Lbox/2,meanCentre[1] + Lbox/2,nPix)
    cuboidVol = Lbox**3/(nPix**2)
    mUnit = snapList[0]['mass'][0]*1e10
    zSlice = meanCentre[axis]
    N = 256
    alphaVal = 0.2
    vmin = 1/1000
    vmax = 1000
    boxsize = snapList[0].properties['boxsize'].ratio("Mpc a h**-1")
    if thickness is None:
        thickness = Lbox
    indLow = int((zSlice + boxsize/2)*N/boxsize)\
         - int((thickness/2)*N/(boxsize))
    indUpp = int((zSlice + boxsize/2)*N/boxsize)\
         + int((thickness/2)*N/(boxsize))
    indLowX = int((meanCentre[0] + boxsize/2)*N/boxsize)\
         - int((thickness/2)*N/(boxsize))
    indUppX = int((meanCentre[0] + boxsize/2)*N/boxsize)\
         + int((thickness/2)*N/(boxsize))
    indLowY = int((meanCentre[1] + boxsize/2)*N/boxsize)\
         - int((thickness/2)*N/(boxsize))
    indUppY = int((meanCentre[1] + boxsize/2)*N/boxsize)\
         + int((thickness/2)*N/(boxsize))
    sm = cm.ScalarMappable(colors.LogNorm(vmin=vmin,vmax=vmax),\
            cmap=cmap)
    phi = np.linspace(0,2*np.pi,1000)
    Xcirc = np.cos(phi)
    Ycirc = np.sin(phi)
    ahNumbers = [np.array(centralAntihalos[l][0],dtype=int)[sortedList[l]] \
        for l in range(0,numCats)]
    plt.clf()
    if title is None:
        title = "Void " + str(nV + 1)
    for ns in range(0,numCats):
        plt.clf()
        denToPlot = np.mean(densities[ns][indLowX:indUppX,indLowY:indUppY,\
            indLow:indUpp],axis)
        plt.imshow(denToPlot,norm=colors.LogNorm(vmin=vmin,vmax=vmax),\
                cmap=cmap,extent=(meanCentre[0] - Lbox/2,\
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
            sampleCentre = centresListShort[ns][catToPlot[nV,ns] - 1]
            plt.scatter(sampleCentre[0],sampleCentre[1],marker='x',color='b',\
                label='Sample Centre')
            plt.plot(sampleCentre[0] + radiiToPlot[nV,ns]*Xcirc,\
                sampleCentre[1] + radiiToPlot[nV,ns]*Ycirc,\
                linestyle='--',color='b',label='Effective radius\n$' + \
                ("%.2g" % radiiToPlot[nV,ns]) + "\\mathrm{Mpc}^{-1}$")
        # Scatter plot of all voids in the region:
        if showOtherCentres:
            # Get all voids in this box:
            centreFilter = getAllVoidsWithinBox(meanCentre,Lbox,\
                centresListShort[ns])
            centreNumbers = ahNumbers[ns][centreFilter]
            # Remove the sample void so we don't double plot it:
            centreFilter[centreFilter] = \
                (centreNumbers != (catToPlot[nV,ns] - 1))
            # Filter based on radius:
            radiusFilter = (radiiListShort[ns] >= \
                np.max([rMin,minMuR*radiiToPlot[nV,ns]])) & \
                (radiiListShort[ns] <= np.min([rMax,radiiToPlot[nV,ns]/minMuR]))
            centreFilter = centreFilter & radiusFilter
            centreNumbers = ahNumbers[ns][centreFilter]
            # Now plot these:
            plt.scatter(centresListShort[ns][centreFilter,0],\
                centresListShort[ns][centreFilter,1],marker='x',color='k')
            for k in range(0,len(centreNumbers)):
                plt.annotate(str(centreNumbers[k]),\
                    (centresListShort[ns][centreFilter,0][k],\
                    centresListShort[ns][centreFilter,1][k]))
        # Add in labelling:
        plt.scatter(meanCentre[0],meanCentre[1],marker='x',color='r',\
            label='Mean Centre')
        plt.legend(frameon=False,loc="lower right")
        plt.xlabel('x ($\\mathrm{Mpc}h^{-1}$)')
        plt.ylabel('y ($\\mathrm{Mpc}h^{-1}$)')
        plt.xlim([meanCentre[0] - Lbox/2,meanCentre[0] + Lbox/2])
        plt.ylim([meanCentre[1] - Lbox/2,meanCentre[1] + Lbox/2])
        plt.title(title + '\nSample ' + str(snapNumList[ns]))
        cbar = plt.colorbar(sm, orientation="vertical")
        plt.savefig(figuresFolder + framePrefix + "frame_" + \
            str(ns) + frameSuffix + ".png")
        #plt.show()
    frames = []
    #imgs = glob.glob(figuresFolder + "frame_*.png")
    imgs = [figuresFolder + framePrefix + "frame_" + str(k) + frameSuffix + \
        ".png" for k in range(0,numCats)]
    for i in imgs:
        new_frame = Image.open(i)
        frames.append(new_frame)
    # Save into a GIF file that loops forever
    frames[0].save(figuresFolder + namePrefix + 'voids' + nameSuffix + \
        '.gif', format='GIF',append_images=frames[1:],save_all=True,
        duration=1000, loop=0)
    plt.clf()


def plotShowVoidConvergence(iteratedCentres,iteratedRadii,density,\
        boxsize,Lbox = 100,axis=2,thickness=None,vmin=1/1000,\
        vmax=1000,cmap='PuOr_r',figuresFolder='./',\
        filename = 'iterations.gif',colorList = ['b','g','r','c','m','y','k'],\
        groundTruthCentre=None,groundTruthRadius=None,width=7.1014):
    if type(iteratedCentres) == np.ndarray:
        numVoids = 1
        numIterations = np.array([len(iteratedCentres)],dtype=int)
        meanCentre = iteratedCentres[-1]
    elif type(iteratedCentres) == list:
        numVoids = len(iteratedCentres)
        numIterations = np.array([len(x) for x in iteratedCentres],dtype=int)
        meanCentre = np.mean(np.array([centre[-1] \
            for centre in iteratedCentres]),0)
    else:
        raise Exception('Variable iteratedCentres has invalid format.')
    if thickness is None:
        thickness = Lbox
    zSlice = meanCentre[axis]
    indLow = int((zSlice + boxsize/2)*N/boxsize)\
         - int((thickness/2)*N/(boxsize))
    indUpp = int((zSlice + boxsize/2)*N/boxsize)\
         + int((thickness/2)*N/(boxsize))
    indLowX = int((meanCentre[0] + boxsize/2)*N/boxsize)\
         - int((Lbox/2)*N/(boxsize))
    indUppX = int((meanCentre[0] + boxsize/2)*N/boxsize)\
         + int((Lbox/2)*N/(boxsize))
    indLowY = int((meanCentre[1] + boxsize/2)*N/boxsize)\
         - int((Lbox/2)*N/(boxsize))
    indUppY = int((meanCentre[1] + boxsize/2)*N/boxsize)\
         + int((Lbox/2)*N/(boxsize))
    denToPlot = np.mean(density[indLowX:indUppX,indLowY:indUppY,\
            indLow:indUpp],axis)
    sm = cm.ScalarMappable(colors.LogNorm(vmin=vmin,vmax=vmax),\
            cmap=cmap)
    phi = np.linspace(0,2*np.pi,1000)
    Xcirc = np.cos(phi)
    Ycirc = np.sin(phi)
    plt.clf()
    numFrames = np.max(numIterations)
    for k in range(0,numFrames):
        plt.clf()
        fig, ax = plt.subplots(figsize=(width,width))
        plt.imshow(denToPlot,norm=colors.LogNorm(vmin=vmin,vmax=vmax),\
                cmap=cmap,extent=(meanCentre[0] - Lbox/2,\
                meanCentre[0] + Lbox/2,\
                meanCentre[1] - Lbox/2,meanCentre[1] + Lbox/2))
        if (groundTruthCentre is not None) and (groundTruthRadius is not None):
            plt.plot(groundTruthCentre[0] + groundTruthRadius*Xcirc,\
                groundTruthCentre[1] + groundTruthRadius*Ycirc,color='k',\
                linestyle='--',label='Ground truth\n$R = ' + \
                ("%.2g" % groundTruthRadius) + "\\mathrm{Mpc}^{-1}$")
        for l in range(0,numVoids):
            if numVoids == 1:
                centre = iteratedCentres[l][k]
                radius = iteratedRadii[l][k]
            else:
                if k >= numIterations[l]:
                    # Pause if we run out of iterations before the end:
                    centre = iteratedCentres[l][numIterations[l]-1]
                    radius = iteratedRadii[l][numIterations[l]-1]
                else:
                    centre = iteratedCentres[l][k]
                    radius = iteratedRadii[l][k]
            plt.scatter(centre[0],centre[1],marker='x',\
                color=colorList[np.mod(l,len(colorList))],label='centre')
            plt.plot(centre[0] + radius*Xcirc,centre[1] + radius*Ycirc,\
                linestyle='--',color=colorList[np.mod(l,len(colorList))],\
                label='Void ' + str(l+1) + ' radius\n$' + \
                ("%.2g" % radius) + "\\mathrm{Mpc}^{-1}$")
        plt.legend(frameon=False,loc="lower right")
        plt.xlabel('x ($\\mathrm{Mpc}h^{-1}$)')
        plt.ylabel('y ($\\mathrm{Mpc}h^{-1}$)')
        plt.xlim([meanCentre[0] - Lbox/2,meanCentre[0] + Lbox/2])
        plt.ylim([meanCentre[1] - Lbox/2,meanCentre[1] + Lbox/2])
        plt.title('Iteration ' + str(k+1))
        cbar = plt.colorbar(sm, orientation="vertical")
        plt.savefig(figuresFolder + "iteration_frame_" + str(k) + ".png")
    frames = []
    #imgs = glob.glob(figuresFolder + "frame_*.png")
    imgs = [figuresFolder + "iteration_frame_" + str(k) + \
        ".png" for k in range(0,numFrames)]
    for i in imgs:
        new_frame = Image.open(i)
        frames.append(new_frame)
    # Save into a GIF file that loops forever
    frames[0].save(figuresFolder + filename,\
        format='GIF',append_images=frames[1:],save_all=True,
        duration=1000, loop=0)
    plt.clf()

plotShowVoidConvergence(iteratedCentresList[1],iteratedRadiiList[1],\
    meanDensity256,boxsize,figuresFolder=figuresFolder)


# Iterate over the good voids:

iteratedCentres = [iteratedCentresList[filter300ind[x]] \
    for x in splitListGoodFiltered[5]]
iteratedRadii = [iteratedRadiiList[filter300ind[x]] \
    for x in splitListGoodFiltered[5]]



plotShowVoidConvergence(iteratedCentres,iteratedRadii,\
    meanDensity256,boxsize,figuresFolder=figuresFolder)

# Loop over all the good voids:
filter300ind = np.where(filter300)[0]
for k in range(0,len(splitListGoodFiltered)):
    iteratedCentres = [cat300.iteratedCentresList[filter300ind[x]] \
        for x in splitListGoodFiltered[k]]
    iteratedRadii = [cat300.iteratedRadiiList[filter300ind[x]] \
        for x in splitListGoodFiltered[k]]
    if len(iteratedCentres) > 0:
        plotShowVoidConvergence(iteratedCentres,iteratedRadii,\
            meanDensity256,boxsize,figuresFolder=figuresFolder,\
            filename = "curated_iterations_" + str(k) + ".gif",\
            groundTruthCentre = meanCentreOptHC[k],\
            groundTruthRadius = radiiMeanOptHC[k],thickness=20)

#nVold = 113 # snr ~ 12, r = 10.2
#nVold = 116 # snr ~ 2.9, r = 10.9
#nVold = 0 # snr ~ 4.3, r  = 20.9
#nVold = 4 # snr ~ 3.9, r = 17.4
nVold= 61 # snr ~ 12.5, r = 14.1
plotVoidAnimation(nVold,centralAntihalos,sortedList,finalCatOpt,radiiListOpt,\
        snapNumList,centresListShort,radiiListShort,densities256,snapList,\
        hrList,snapSortList,\
        figuresFolder=figuresFolder,nameSuffix = "_nVold_" + str(nVold),\
        minMuR=0.9)


for nV in splitList[nVold]:
    plotVoidAnimation(nV,centralAntihalos,sortedList,finalCat300,radiiList300,\
        snapNumList,centresListShort,radiiListShort,densities256,snapList,\
        hrList,snapSortList,\
        figuresFolder=figuresFolder,nameSuffix = "_nV_" + str(nV) + \
        "_nVold_" + str(nVold),minMuR=0.925)

# Good voids sample:


for nV in np.where(filterOptGood)[0]:
    plotVoidAnimation(nV,centralAntihalos,sortedList,finalCatOpt,radiiListOpt,\
        snapNumList,centresListShort,radiiListShort,densities256,snapList,\
        hrList,snapSortList,\
        figuresFolder=figuresFolder,nameSuffix = "_good_voids_nV_" + str(nV),\
        minMuR=0.9,frameSuffix="",framePrefix="",namePrefix="good_voids/")


# Central and average densities:
deltaCentral300 = [props[11] for props in ahProps]
deltaAverage300 = [props[12] for props in ahProps]
shortedendDeltaCentral = cat300.getShortenedQuantity(deltaCentral300,\
    cat300.centralAntihalos)
shortedendDeltaAverage = cat300.getShortenedQuantity(deltaAverage300,\
    cat300.centralAntihalos)
deltaCentralList = getPropertyFromCat(cat300.finalCat,shortedendDeltaCentral)
deltaAverageList = getPropertyFromCat(cat300.finalCat,shortedendDeltaAverage)
[deltaCentralMean,deltaCentralSigma] = getMeanProperty(deltaCentralList,\
    lowerLimit=-1)
[deltaAverageMean,deltaAverageSigma] = getMeanProperty(deltaAverageList,\
    lowerLimit=-1)


# For curated voids:
deltaCentralOpt = [props[11] for props in ahProps]
deltaAverageOpt = [props[12] for props in ahProps]
shortedendDeltaCentralOpt = getShortenedQuantity(deltaCentralOpt,\
    cat300.centralAntihalos,cat300.centresListShort,cat300.sortedList,\
    cat300.ahCounts,cat300.max_index)
shortedendDeltaAverageOpt = getShortenedQuantity(deltaAverageOpt,\
    cat300.centralAntihalos,cat300.centresListShort,cat300.sortedList,\
    cat300.ahCounts,cat300.max_index)
deltaCentralListOpt = getPropertyFromCat(finalCatOpt,shortedendDeltaCentralOpt)
deltaAverageListOpt = getPropertyFromCat(finalCatOpt,shortedendDeltaAverageOpt)
[deltaCentralMeanOpt,deltaCentralSigmaOpt] = \
    getMeanProperty(deltaCentralListOpt,lowerLimit=-1)
[deltaAverageMeanOpt,deltaAverageSigmaOpt] = \
    getMeanProperty(deltaAverageListOpt,lowerLimit=-1)


# Convergence of iterated centres:
nV = 4
iteratedCentres = [cat300.iteratedCentresList[filter300ind[x]] \
    for x in splitListGoodFiltered[nV]]
iteratedRadii = [cat300.iteratedRadiiList[filter300ind[x]] \
    for x in splitListGoodFiltered[nV]]

finalCentres = [x[-1] for x in iteratedCentres]
finalRadii = [x[-1] for x in iteratedRadii]
distances = [np.sqrt(np.sum((x - final)**2,1)) \
    for x, final in zip(iteratedCentres,finalCentres)]
radiiDistance = [x - final for x, final in zip(iteratedRadii,finalRadii)]

plt.clf()
colorList = ['b','g','r','c','m','y','k']
for k in range(0,len(iteratedCentres)):
    plt.plot(range(0,len(iteratedCentres[k])),distances[k],\
        color=colorList[np.mod(k,len(colorList))],label="Void " + str(k+1))
    plt.xlabel('Iteration')
    plt.ylabel('Distance from final centres [$\\mathrm{Mpc}h^{-1}$]')

plt.legend()
plt.savefig(figuresFolder + "centre_convergence.svg")


plt.clf()
colorList = ['b','g','r','c','m','y','k']
for k in range(0,len(iteratedRadii)):
    plt.plot(range(0,len(iteratedRadii[k])),radiiDistance[k],\
        color=colorList[np.mod(k,len(colorList))],label="Void " + str(k+1))
    plt.xlabel('Iteration')
    plt.ylabel('$R - R_{\\mathrm{final}}$ [$\\mathrm{Mpc}h^{-1}$]')

plt.legend()
plt.savefig(figuresFolder + "radius_convergence.svg")



plt.clf()
colorList = ['b','g','r','c','m','y','k']
for nV in range(0,30):
    iteratedCentres = [cat300.iteratedCentresList[filter300ind[x]] \
        for x in splitListGoodFiltered[nV]]
    iteratedRadii = [cat300.iteratedRadiiList[filter300ind[x]] \
        for x in splitListGoodFiltered[nV]]
    finalCentres = [x[-1] for x in iteratedCentres]
    finalRadii = [x[-1] for x in iteratedRadii]
    distances = [np.sqrt(np.sum((x - meanCentreOpt[filterOptGood][nV])**2,1)) \
        for x in iteratedCentres]
    radiiDistance = [x - final for x, final in zip(iteratedRadii,finalRadii)]
    for k in range(0,1):
        plt.plot(range(0,len(iteratedCentres[k])),distances[k]/\
            radiiMeanOpt[filterOptGood][k],\
            color=colorList[np.mod(k,len(colorList))],label="Void " + str(k+1))
        plt.xlabel('Iteration')
        plt.ylabel('(Distance from Ground Truth Centre)/$R_{\\mathrm{Ground Truth}}$')
        plt.axhline(0.05,color='grey',linestyle=':')

#plt.legend()
plt.savefig(figuresFolder + "centre_convergence_all.svg")



plt.clf()
colorList = ['b','g','r','c','m','y','k']
for nV in range(0,30):
    iteratedCentres = [cat300.iteratedCentresList[filter300ind[x]] \
        for x in splitListGoodFiltered[nV]]
    iteratedRadii = [cat300.iteratedRadiiList[filter300ind[x]] \
        for x in splitListGoodFiltered[nV]]
    finalCentres = [x[-1] for x in iteratedCentres]
    finalRadii = [x[-1] for x in iteratedRadii]
    distances = [np.sqrt(np.sum((x - final)**2,1)) \
        for x, final in zip(iteratedCentres,finalCentres)]
    radiiDistance = [x - radiiMeanOpt[filterOptGood][nV] \
        for x in iteratedRadii]
    for k in range(0,1):
        plt.plot(range(0,len(iteratedRadii[k])),radiiDistance[k]/\
        radiiMeanOpt[filterOptGood][k],\
        color=colorList[np.mod(k,len(colorList))],label="Void " + str(k+1))
    plt.xlabel('Iteration')
    plt.ylabel('$(R - R_{\\mathrm{Ground Truth}})/R_{\\mathrm{Ground Truth}}$')
    plt.axhline(0.01,color='grey',linestyle=':')
    plt.axhline(-0.01,color='grey',linestyle=':')

#plt.legend()
plt.savefig(figuresFolder + "radius_convergence_all.svg")








