#-------------------------------------------------------------------------------
# CONFIGURATION
from void_analysis import plot, tools, snapedit
from void_analysis.paper_plots_borg_antihalos_generate_data import *
from void_analysis.real_clusters import getClusterSkyPositions
from matplotlib import transforms
import pickle
import numpy as np
import seaborn as sns
seabornColormap = sns.color_palette("colorblind",as_cmap=True)
import pynbody
import astropy.units as u
from astropy.coordinates import SkyCoord
import scipy
import os

figuresFolder = "borg-antihalos_paper_figures/all_samples"

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

fontsize = 10
legendFontsize = 8

#-------------------------------------------------------------------------------
# DATA FOR PLOTS



# HMF plots data:

# Snapshots to use:
snapNumListOld = [7422,7500,8000,8500,9000,9500]
#snapNumList = [7000,7200,7400,7600,8000]
#snapNumList = [7000,7200,7400,7600,7800,8000]
#snapNumList = np.arange(7000,10300 +1,300)
#snapNumList = [8800,9100,9400,9700,10000]
snapNumList = [7000,7300,7600,7900,8200,8500,8800,9100,9400,9700,10000,\
    10300,10600,10900,11200,11500,11800,12100,12400,12700,13000]
# Batch5-1:
#snapNumList = [7000,7300,7600,7900,8200]
# Batch5-2:
#snapNumList = [8500,8800,9100,9400,9700]
# Batch5-3:
#snapNumList = [10000,10300,10600,10900,11200]
# Batch5-4:
#snapNumList = [11500,11800,12100,12400,12700]
# Batch10-1 
#snapNumList = [7000,7300,7600,7900,8200,8500,8800,9100,9400,9700]
# Batch10-2
#snapNumList = [10000,10300,10600,10900,11200,11500,11800,12100,12400,12700]


#snapNumListUncon = [1,2,3,4,5]
snapNumListUncon = [1,2,3,4,5,6,7,8,9,10]
#snapNumListUncon = [2,4,5,6,7,8,9,10]
snapNumListUnconOld = [1,2,3,5,6]
boxsize = 677.7

# PPT plots data:
if runTests:
    testComputation(testDataFolder + "ppt_pipeline_test.p",getPPTPlotData,\
        snapNumList = [7000, 7200, 7400],samplesFolder = 'new_chain/',\
        recomputeData = True)

nBinsPPT = 30

[galaxyNumberCountExp,galaxyNumberCountsRobust] = tools.loadOrRecompute(\
    figuresFolder + "ppt_plots_data.p",getPPTPlotData,\
    _recomputeData = recomputeData,nBins = nBinsPPT,nClust=9,nMagBins = 16,\
        N=256,restartFile = 'new_chain_restart/merged_restart.h5',\
        snapNumList = snapNumList,samplesFolder = 'new_chain/',\
        surveyMaskPath = "./2mpp_data/",\
        Om0 = 0.3111,Ode0 = 0.6889,boxsize = 677.7,h=0.6766,Mstarh = -23.28,\
        mmin = 0.0,mmax = 12.5,recomputeData = recomputeData,rBinMin = 0.1,\
        rBinMax = 20,abell_nums = [426,2147,1656,3627,3571,548,2197,2063,1367],\
        nside = 4,nRadialSlices=10,rmax=600,tmppFile = "2mpp_data/2MPP.txt",\
        reductions = 4,iterations = 20,verbose=True,hpIndices=None,\
        centreMethod="density")


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
                recomputeData = [False,False,False,False],\
                unconstrainedFolderNew = unconstrainedFolderNew,\
                unconstrainedFolderOld = unconstrainedFolderOld,\
                snapnameNew = snapnameNew,snapnameNewRev=snapnameNewRev,\
                snapnameOld = snapnameOld,snapnameOldRev = snapnameOldRev,\
                samplesFolder = samplesFolder,samplesFolderOld=samplesFolderOld)

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
        mLower = "auto",mUpper = 2e15,nBins = 8,muOpt = 0.9,rSearchOpt = 1,\
        rSphere = 300,rSphereInner = 135,NWayMatch = True,rMin=5,rMax=30,\
        mMin=1e11,mMax = 1e16,percThresh=99,chainFile="chain_properties.p",\
        Nden=256,recomputeUnconstrained = True,data_folder=data_folder,\
        _recomputeData = recomputeData)

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
massFunctionComparison(massListMean[combinedFilter135],\
    massListMean[combinedFilter],volSphere135,nBins=nBins,\
    labelLeft = "Combined catalogue",\
    labelRight  ="Combined catalogue",\
    ylabel="Number of antihalos",savename=figuresFolder + \
    "mass_function_combined_300vs135.pdf",massLower=mLower,\
    ylim=[1,1000],Om0 = 0.3111,h=0.6766,sigma8=0.8128,ns=0.9667,\
    fontsize=10,massUpper = mUpper,\
    titleLeft = "Combined catalogue, $<135\\mathrm{Mpc}h^{-1}$",\
    titleRight = "Combined catalogue, $<300\\mathrm{Mpc}h^{-1}$",\
    volSimRight = volSphere,ylimRight=[1,1000],legendLoc="upper right")

# Density Profiles:
plt.clf()
textwidth=7.1014
fontsize = 12
legendFontsize=10
fig, ax = plt.subplots(1,2,figsize=(textwidth,0.5*textwidth))
plot.plotConstrainedVsUnconstrainedProfiles(rBinStackCentresCombined,\
    nbarjSepStackCombined,sigmaSepStackCombined,\
    nbarjAllStackedUnCombined,sigmaAllStackedUnCombined,nbar,rMin2,\
    mMin2,mMax2,fontsize = fontsize,legendFontSize=legendFontsize,\
    labelCon='Constrained',\
    labelRand='Unconstrained \nmean',\
    savename=figuresFolder + "profiles1415.pdf",\
    showTitle=True,ax = ax[0],title="Combined Catalogue",\
    meanErrorLabel = 'Unconstrained \nMean',\
    profileErrorLabel = 'Profile \nvariation \n',\
    nbarjUnconstrainedStacks=nbarjSepStackUn,legendLoc='lower right',\
    sigmajUnconstrainedStacks = sigmaSepStackUn,showMean=True,show=False,\
    xlim=[0,10])
plot.plotConstrainedVsUnconstrainedProfiles(rBinStackCentres,nbarjSepStack,\
    sigmaSepStack,nbarjAllStackedUn,sigmaAllStackedUn,nbar,rMin2,\
    mMin2,mMax2,fontsize = fontsize,legendFontSize=legendFontsize,\
    labelCon='Constrained',\
    labelRand='Unconstrained \nmean',\
    savename=figuresFolder + "profiles1415.pdf",\
    showTitle=True,ax = ax[1],title="All Catalogues",\
    meanErrorLabel = 'Unconstrained \nMean',\
    profileErrorLabel = 'Profile \nvariation \n',\
    nbarjUnconstrainedStacks=nbarjSepStackUn,legendLoc='lower right',\
    sigmajUnconstrainedStacks = sigmaSepStackUn,showMean=True,show=False,\
    xlim=[0,10])
ax[0].axhline(0.95,color='grey',linestyle=':')
ax[1].axhline(0.95,color='grey',linestyle=':')
plt.tight_layout()
plt.savefig(figuresFolder + "profiles_plot_vs_underdense.pdf")
plt.show()

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

# Antihalo sky plot data:
if runTests:
    testComputation(testDataFolder + "skyplot_pipeline_test.p",\
        getAntihaloSkyPlotData,\
        [7000, 7200, 7400],samplesFolder=samplesFolder,recomputeData = True)

[alpha_shape_list,largeAntihalos,\
        snapsortList] = tools.loadOrRecompute(figuresFolder + "skyplot_data.p",
            getAntihaloSkyPlotData,snapNumList,samplesFolder=samplesFolder,\
            _recomputeData = recomputeData,recomputeData = recomputeData)

# Can't really un-pickle the halo catalogues without loading the snapshots, so
# we will have to load these, unfortunately. We should change this, as it 
# probably isn't very scalable...
samplesFolder = "new_chain/"
snapListRev =  [pynbody.load(samplesFolder + "sample" + str(snapNum) + "/" \
        + "gadget_full_reverse_512/snapshot_001") for snapNum in snapNumList]
antihaloCatalogueList = [snap.halos() for snap in snapListRev]

#-------------------------------------------------------------------------------
# Reproduce the plots for the borg-antihalos paper


#-------------------------------------------------------------------------------
# PPTs PLOT:
# Names of the clusters of interest for PPT plots:
clusterNames = np.array([['Perseus-Pisces (A426)'],
       ['Hercules B (A2147)'],
       ['Coma (A1656)'],
       ['Norma (A3627)'],
       ['Shapley (A3571)'],
       ['A548'],
       ['Hercules A (A2199)'],
       ['Hercules C (A2063)'],
       ['Leo (A1367)']], dtype='<U21')

# PPTs:
# Options:
suffix = ''
#rBins = np.logspace(np.log10(0.1),np.log10(20),31)
rBins = np.linspace(0.1,20,nBinsPPT+1)


plot.plotPPTProfiles(np.sum(galaxyNumberCountExp,2),\
    np.sum(galaxyNumberCountsRobust,2),\
    savename=figuresFolder + "ppt_Ngal_robust" + suffix + ".pdf",\
    title="Posterior predicted galaxy counts " + \
    " vs 2M++ galaxy counts",ylim=[1,1000],\
    show=True,rBins=rBins,clusterNames=clusterNames,rescale=False,\
    density=False,legLoc = [0.3,0.1],hspace=0.3,\
    ylabel='Number of galaxies $ < r$')

#-------------------------------------------------------------------------------
# HMF/AMF PLOT:

savename = figuresFolder + "hmf_amf_old_vs_new_forward_model.pdf"

plot.plotHMFAMFComparison(\
        constrainedHaloMasses512Old,deltaListMeanOld,deltaListErrorOld,\
        comparableHaloMassesOld,constrainedAntihaloMasses512Old,\
        comparableAntihaloMassesOld,\
        constrainedHaloMasses512New,deltaListMeanNew,deltaListErrorNew,\
        comparableHaloMassesNew,constrainedAntihaloMasses512New,\
        comparableAntihaloMassesNew,\
        referenceSnap,referenceSnapOld,\
        savename = figuresFolder + "test_hmf_amf.pdf",\
        ylabelStartOld = 'Old reconstruction',\
        ylabelStartNew = 'New reconstruction',\
        fontsize=fontsize,legendFontsize=legendFontsize,density=True,\
        xlim = (mLimLower,3e15),nMassBins=11,mLower=1e14,mUpper=3e15)


#-------------------------------------------------------------------------------
# HMF/AMF PLOT, UNDERDENSE:

plot.plotHMFAMFUnderdenseComparison(\
        constrainedHaloMasses512New,deltaListMeanNew,deltaListErrorNew,\
        comparableHaloMassesNew,constrainedAntihaloMasses512New,\
        comparableAntihaloMassesNew,centralHalosNew,centralAntihalosNew,\
        centralHaloMassesNew,centralAntihaloMassesNew,\
        savename = figuresFolder + "hmf_amf_underdense_comparison.pdf",\
        xlim = (mLimLower,3e15),nMassBins=11,mLower=1e14,mUpper=3e15)

#-------------------------------------------------------------------------------
# RADIAL VOID PROFILES

plot.plotConstrainedVsUnconstrainedProfiles(rBinStackCentres,nbarjSepStack,\
    sigmaSepStack,nbarjAllStackedUn,sigmaAllStackedUn,nbar,rMin,mMin,mMax,\
    showImmediately = True,fontsize = fontsize,legendFontSize=legendFontsize,\
    labelCon='Constrained',\
    labelRand='Unconstrained \nmean',\
    savename=figuresFolder + "profiles1415.pdf",\
    showTitle=False,\
    meanErrorLabel = 'Unconstrained \nMean',\
    profileErrorLabel = 'Profile \nvariation \n',\
    nbarjUnconstrainedStacks=nbarjSepStackUn,\
    sigmajUnconstrainedStacks = sigmaSepStackUn,showMean=True)

#-------------------------------------------------------------------------------
# ANTIHALO SKY PLOT:

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

plot.plotLocalUniverseMollweide(rCut,snapToShow,\
    alpha_shapes = alpha_shape_list[ns][1],
    largeAntihalos = largeAntihalos[ns],hr=antihaloCatalogueList[ns],
    coordAbell = coordCombinedAbellSphere,abellListLocation = clusterIndMain,\
    nameListLargeClusters = [name[0] for name in clusterNames],\
    ha = ha,va= va, annotationPos = annotationPos,\
    title = 'Local super-volume: large voids (antihalos) within $' + \
    str(rCut) + "\\mathrm{\\,Mpc}h^{-1}$",
    vmin=1e-2,vmax=1e2,legLoc = 'lower left',bbox_to_anchor = (-0.1,-0.2),
    snapsort = snapsortList[ns],antihaloCentres = None,
    figOut = figuresFolder + "/antihalos_sky_plot.pdf",
    showFig=True,figsize = (scale*textwidth,scale*0.55*textwidth),
    voidColour = seabornColormap[0],antiHaloLabel='inPlot',
    bbox_inches = bound_box,galaxyAngles=equatorialRThetaPhi[:,1:],\
    galaxyDistances = equatorialRThetaPhi[:,0],showGalaxies=False)

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
plot.plotMassTypeComparison(np.array(massList200c)[:,:,clusterFilter],\
    np.array(massListFull200c)[:,clusterFilter],\
    np.array(massList100m)[:,:,clusterFilter],\
    np.array(massListFull100m)[:,clusterFilter],\
    stepsListGADGET,stepsList,logstepsList,stepsList1024,\
    stepsListEPS_0p662,resStepsList,clusterNames[clusterFilter,:],\
    name1 = "$M_{200\\mathrm{c}}$",name2 = "$M_{100\\mathrm{m}}$",\
    show=True,save = True,colorLinear = seabornColormap[0],\
    colorLog=seabornColormap[1],colorGadget='k',colorAdaptive='grey',\
    showGadgetAdaptive = True,showResMasses = False,\
    savename = figuresFolder + "mass_convergence_comparison.pdf",\
    massName = "M",\
    extraMasses = None,extraMassLabel = 'Extra mass scale',\
    xlabel='Number of Steps',nsamples = len(sampleList),\
    returnHandles=False,showLegend=True,nCols=3,resList=resList)


