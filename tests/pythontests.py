import unittest

from void_analysis.tools import mcmcFileToWhiteNoise
from void_analysis import context, tools
import numpy as np
import pynbody
import multiprocessing as mp
import scipy
from void_analysis.postprocessing import process_snapshot
from void_analysis.paper_plots_data.paper_plots_borg_antihalos_generate_data import *
from void_analysis.simulation_tools import ngPerLBin, ngBias
from void_analysis.simulation_tools import biasNew, biasOld
from void_analysis.survey import radialCompleteness, surveyMask
import pickle
from void_analysis import snapedit, real_clusters
import astropy
import h5py

# Tests for functions required in IC generation:
class test_ICgen(unittest.TestCase):
    # Test the extraction and conversion of mcmc file white noise to a
    # format that genetIC understands
    def test_wn(self):
        tools.mcmcFileToWhiteNoise(dataFolder + "mcmc_7000.h5",\
            "test_wn_extraction/wn.npy",\
            normalise = True,fromInverseFourier=False,flip=False,reverse=True)
        reference = np.load(dataFolder + "wn_test.npy")
        testCase = np.load("test_wn_extraction/wn.npy")
        self.assertTrue(np.allclose(testCase,reference,rtol=1e-5,atol=1e-8))

# Tests for functions required for computing anti-halo/halo properties
class test_ahproperties(unittest.TestCase):
    # Test conversion of volumes to the correct units:
    def test_volumes(self):
        dataFolder = "data_for_tests/"
        standard = dataFolder + \
            "reference_constrained/sample7000/forward/snapshot_001"
        snapn = pynbody.load(standard)
        referenceVols = np.load(dataFolder + \
            "reference_constrained/volumes_ref.npy")
        volumes = tools.zobovVolumesToPhysical(standard + ".vols",snapn,\
            dtype=np.double,offset=4)
        self.assertTrue(np.allclose(volumes,referenceVols,rtol=1e-5,atol=1e-8))
    # Test computation of anti-halo centres:
    def test_centres(self):
        dataFolder = "data_for_tests/"
        standard = dataFolder + \
            "reference_constrained/sample7000/forward/snapshot_001"
        reverse = dataFolder + \
            "reference_constrained/sample7000/reverse/snapshot_001"
        snapn = pynbody.load(standard)
        hn = snapn.halos()
        snapr = pynbody.load(reverse)
        hr = snapr.halos()
        # Check whether the snapshots need re-ordering:
        sortedn = np.arange(0,len(snapn))
        sortedr = np.arange(0,len(snapr))
        orderedn = np.all(sortedn == snapn['iord'])
        orderedr = np.all(sortedr == snapr['iord'])
        if not orderedn:
            sortedn = np.argsort(snapn['iord'])
        if not orderedr:
            sortedr = np.argsort(snapr['iord'])
        volumes = np.load(dataFolder + "reference_constrained/volumes_ref.npy")
        boxsize = snapn.properties['boxsize'].ratio("Mpc a h**-1")
        periodicity = [boxsize]*3
        # Test the centre computation for the first anti-halo:
        refCentres = np.load(dataFolder + \
            "reference_constrained/ref_centres.npy")
        refVols = 
        # Test computation of the first few anti-halo centres:
        centreList = np.zeros((10,3))
        for k in range(0,10):
            centreList[k,:] = context.computePeriodicCentreWeighted(\
                snapn['pos'][sortedn[hr[k + 1]['iord']],:],\
                volumes[hr[k + 1]['iord']],periodicity)
        self.assertTrue(np.allclose(centreList,refCentres,rtol=1e-5,atol=1e-8))
    # Test pair counts calculation:
    def test_pair_counts(self):
        thread_count = mp.cpu_count()
        dataFolder = "data_for_tests/"
        refCentres = np.load(dataFolder + \
            "reference_constrained/ref_centres.npy")
        refRadii = np.load(dataFolder + \
            "reference_constrained/ref_radii.npy")
        volumes = np.load(dataFolder + "reference_constrained/volumes_ref.npy")
        standard = dataFolder + \
            "reference_constrained/sample7000/forward/snapshot_001"
        snapn = pynbody.load(standard)
        boxsize = snapn.properties['boxsize'].ratio("Mpc a h**-1")
        tree = scipy.spatial.cKDTree(snapn['pos'],boxsize=boxsize)
        nBins = 31
        rBinStack = np.linspace(0,3.0,nBins)
        pairCountsRef = np.load(dataFolder + \
            "reference_constrained/pair_counts_ref.npy")
        volumesListRef = np.load(dataFolder + \
            "reference_constrained/volumes_list_ref.npy")
        [pairCounts,volumesList] = stacking.getPairCounts(refCentres,refRadii,\
            snapn,rBinStack,nThreads=thread_count,tree=tree,\
            method="poisson",vorVolumes=volumes)
        self.assertTrue(\
            np.allclose(volumesList,volumesListRef,rtol=1e-5,atol=1e-8))
        self.assertTrue(\
            np.allclose(pairCounts,pairCountsRef,rtol=1e-5,atol=1e-8))
    # Test central density calculation:
    def test_central_density(self):
        thread_count = mp.cpu_count()
        dataFolder = "data_for_tests/"
        standard = dataFolder + \
            "reference_constrained/sample7000/forward/snapshot_001"
        snapn = pynbody.load(standard)
        boxsize = snapn.properties['boxsize'].ratio("Mpc a h**-1")
        tree = scipy.spatial.cKDTree(snapn['pos'],boxsize=boxsize)
        refCentres = np.load(dataFolder + \
            "reference_constrained/ref_centres.npy")
        refRadii = np.load(dataFolder + \
            "reference_constrained/ref_radii.npy")
        volumes = np.load(dataFolder + "reference_constrained/volumes_ref.npy")
        densityList = np.zeros((10,3))
        rhoBar = np.sum(snapn['mass'])/(boxsize**3)
        for k in range(0,10):
            densityList[k] = stacking.centralDensity(\
                refCentres[k,:],refRadii[k],snapn['pos'],volumes,\
                snapn['mass'],tree=tree,centralRatio = 2,\
                nThreads=thread_count)/rhoBar - 1.0
    # Test the pipeline end to end:
    def test_properties_pipeline(self):
        dataFolder = "data_for_tests/"
        standard = dataFolder + \
            "reference_constrained/sample7000/forward/snapshot_001"
        reverse = dataFolder + \
            "reference_constrained/reverse/snapshot_001"
        # Run the pipeline:
        process_snapshot.processSnapshot(standard,reverse,31,offset=4,\
            output="test_ahproperties_pipeline/ahproperties.p")
        # Compare the results:
        ahProps = pickle.load("test_ahproperties_pipeline/ahproperties.p")
        ahPropsRef = pickle.load(standard + ".AHproperties.p")
        for k in range(0,len(ahPropsRef)):
            self.assertTrue(\
                np.allclose(ahProps[k],ahPropsRef[k],rtol=1e-5,atol=1e-8))

# Testing the PPT code:
class test_ppts(unittest.TestCase):
    # PPT pipeline:
    def test_ppt_pipeline(self):
        dataFolder = "data_for_tests/"
        hpIndices = pickle.load(open(dataFolder + "hpIndices_ref.p","rb"))
        [galaxyNumberCountExp,galaxyNumberCountsRobust] = getPPTPlotData(\
            hpIndices=hpIndices)
        [galaxyNumberCountExpRef,galaxyNumberCountsRobustRef] = pickle.load(\
            open(dataFolder + "ppt_test_data.p","rb"))
        self.assertTrue(\
            np.allclose(galaxyNumberCountExp,galaxyNumberCountExpRef,\
            rtol=1e-5,atol=1e-8))
        self.assertTrue(\
            np.allclose(galaxyNumberCountsRobust,galaxyNumberCountsRobustRef,\
            rtol=1e-5,atol=1e-8))
    # Survey Mask calculation:
    def test_survey_mask(self):
        dataFolder = "data_for_tests/"
        surveyMask11 = healpy.read_map(dataFolder + "completeness_11_5.fits")
        surveyMask12 = healpy.read_map(dataFolder + "completeness_12_5.fits")
        # Computation of the survey mask:
        N = 256
        Om0 = 0.3111
        Ode0 = 0.6889
        boxsize = 677.7
        h=0.6766
        mmin = 0.0
        mmax = 12.5
        grid = gridListPermutation.gridListPermutation(N,perm=(2,1,0))
        centroids = grid*boxsize/N + boxsize/(2*N)
        positions = snapedit.unwrap(centroids - np.array([boxsize/2]*3),boxsize)
        cosmo = astropy.cosmology.LambdaCDM(100*h,Om0,Ode0)
        computedMaskList = surveyMask(\
                positions,surveyMask11,surveyMask12,cosmo,-0.94,\
                -23.28,keCorr = keCorr,mmin=mmin,numericalIntegration=True,\
                mmax=mmax,splitApparent=True,splitAbsolute=True,\
                returnComponents=True)
        # Reference data:
        referenceMaskList = pickle.load(open(\
            dataFolder + "survey_mask_ref.p","rb"))
        for k in range(0,len(referenceMaskList)):
            self.assertTrue(np.allclose(computedMaskList[k],\
                referenceMaskList[k],rtol=1e-5,atol=1e-8))
    # test of the galaxy count calculation:
    def test_galaxy_counts(self):
        dataFolder = "data_for_tests/"
        nMagBins = 16
        biasData = h5py.File(dataFolder + "mcmc_7000.h5",'r')
        biasParam = np.array([[biasData['scalars']['galaxy_bias_' + \
            str(k)][()] for k in range(0,nMagBins)]])
        [mask,angularMask,radialMas,mask12,mask11] = pickle.load(open(\
            dataFolder + "survey_mask_ref.p","rb"))
        mcmcDenLin_r = np.reshape(np.reshape(np.reshape(\
            1.0 + biasData['scalars']['BORG_final_density'][()],N**3),\
            (N,N,N),order='F'),N**3)
        ngCounts = ngPerLBin(\
            biasParam,return_samples=True,mask=mask,\
            accelerate=True,\
            delta = [mcmcDenLin_r],contrast=False,sampleList=[0],\
            beta=biasParam[:,:,1],rhog = biasParam[:,:,3],\
            epsg=biasParam[:,:,2],\
            nmean=biasParam[:,:,0],biasModel = biasNew)
        ngCountsRef = pickle.load(open(dataFolder + "ngCounts_ref.p","rb"))
        self.assertTrue(np.allclose(ngCounts,ngCountsRef,rtol=1e-5,atol=1e-8))
    # Test mapping between galaxy counts in voxels and healpix patches:
    def test_healpix_mapping(self):
        dataFolder = "data_for_tests/"
        ngMCMC = pickle.load(open(dataFolder + "ngCounts_ref.p","rb"))
        hpIndices = pickle.load(open(dataFolder + "hpIndices_ref.p","rb"))
        ngHPRef = pickle.load(open(dataFolder + "hpCounts_ref.p","rb"))
        ngHP = tools.getCountsInHealpixSlices(ngMCMC,hpIndices,4)
        self.assertTrue(np.allclose(ngHP,ngHPRef,rtol=1e-5,atol=1e-8))
    # Test getting cluster centres:
    def test_cluster_centre_calculation(self):
        dataFolder = "data_for_tests/"
        # Load cluster data:
        [combinedAbellN,combinedAbellPos,abell_nums] = \
            real_clusters.getCombinedAbellCatalogue(catFolder=dataFolder)
        clusterInd = [np.where(combinedAbellN == n)[0] for n in abell_nums]
        clusterLoc = np.zeros((len(clusterInd),3))
        for k in range(0,len(clusterInd)):
            if len(clusterInd[k]) == 1:
                clusterLoc[k,:] = combinedAbellPos[clusterInd[k][0],:]
            else:
                # Average positions:
                clusterLoc[k,:] = np.mean(combinedAbellPos[clusterInd[k],:],0)
        # Snapshot path:
        standard = dataFolder + \
            "reference_constrained/sample7000/forward/snapshot_001"
        snap = pynbody.load(standard)
        boxsize = snap.properties['boxsize'].ratio("Mpc a h**-1")
        # Density field data:
        biasData = h5py.File(dataFolder + "mcmc_7000.h5",'r')
        mcmcDenLin_r = np.reshape(np.reshape(np.reshape(\
            1.0 + biasData['scalars']['BORG_final_density'][()],N**3),\
            (N,N,N),order='F'),N**3)
        # Density field position grid:
        grid = snapedit.gridListPermutation(N,perm=(2,1,0))
        centroids = grid*boxsize/N + boxsize/(2*N)
        positions = snapedit.unwrap(centroids - np.array([boxsize/2]*3),\
            boxsize)
        tree = scipy.spatial.cKDTree(snapedit.wrap(positions + boxsize/2,\
            boxsize),boxsize=boxsize)
        # Compute the centres:
        centreList = simulation_tools.getClusterCentres(clusterLoc,\
            snapPath = standard,fileSuffix = "clusters1",\
            recompute=True,density=np.reshape(mcmcDenLin_r,N**3),\
            boxsize=boxsize,positions=positions,positionTree=tree,\
            method="density",reductions=4,\
            iterations=20)
        refCentreList = pickle.load(open(dataFolder + \
            "cluster_centre_ref.p","rb"))
        # Test
        self.assertTrue(np.allclose(centreList,refCentreList,\
            rtol=1e-5,atol=1e-8))

# Test HMF code:
class test_hmfs(unittest.TestCase):
    # Test computation of the average HMF data:
    def test_constrained_hmfs(self):
        snapNumList = [7000,7200]
        dataFolder = "data_for_tests/"
        snapname = "forward/snapshot_001"
        snapnameRev = "reverse/snapshot_001"
        samplesFolder = dataFolder + "reference_constrained/"
        computedHMFsData = getHMFAMFDataFromSnapshots(\
            snapNumList,snapname,snapnameRev,samplesFolder,\
            recomputeData=True,reCentreSnap=True)
        referenceHMFsData = pickle.load(samplesFolder + \
            "constrained_hmfs_ref.p")
        for k in range(0,len(referenceHMFsData)):
            self.assertTrue(np.allclose(computedHMFsData[k],\
                referenceHMFsData[k],rtol=1e-5,atol=1e-8))
    # Test computation of the average unconstrained HMF data:
    def test_unconstrained_hmfs(self):
        snapNumList = [1,3]
        snapname = "forward/snapshot_001"
        snapnameRev = "reverse/snapshot_001"
        dataFolder = "data_for_tests/"
        samplesFolder = dataFolder + "reference_unconstrained/"
        [constrainedHaloMasses512,constrainedAntihaloMasses512,\
        deltaListMean,deltaListError] = pickle.load(samplesFolder + \
            "constrained_hmfs_ref.p")
        computedHMFsData = getUnconstrainedHMFAMFData(\
            snapNumList,snapname,snapnameRev,samplesFolder,\
            deltaListMean,deltaListError,\
            recomputeData=True,reCentreSnap=True,randomSeed=1000)
        referenceHMFsData = pickle.load(samplesFolder + \
            "unconstrained_hmfs_ref.p")
        for k in range(0,len(referenceHMFsData)):
            self.assertTrue(np.allclose(computedHMFsData[k],\
                referenceHMFsData[k],rtol=1e-5,atol=1e-8))

# Test Void Profiles Code:
class test_void_profiles(unittest.TestCase):
    # Test the procedure for stacking anti-halos from different simulations:
    def test_void_stacking(self):
        snapNumList = [1,3]
        rMin=5
        rMax=25
        dataFolder = "data_for_tests/"
        snapname = "forward/snapshot_001"
        snapnameRev = "reverse/snapshot_001"
        samplesFolder = dataFolder + "reference_unconstrained/"
        # Load snapshots:
        snapList =  [pynbody.load(samplesFolder + "sample" + \
            str(snapNum) + "/" + snapname) for snapNum in snapNumList]
        snapListRev =  [pynbody.load(samplesFolder + "sample" + \
            str(snapNum) + "/" + snapnameRev) for snapNum in snapNumList]
        # Load reference antihalo data:
        ahProps = [pickle.load(\
            open(snap.filename + ".AHproperties.p","rb")) \
            for snap in snapList]
        ahCentresList = [props[5] for props in ahProps]
        ahCentresListRemap = [tools.remapAntiHaloCentre(props[5],boxsize) \
            for props in ahProps]
        antihaloRadii = [props[7] for props in ahProps]
        deltaCentralList = [props[11] for props in ahProps]
        antihaloMassesList = [props[3] for props in ahProps]
        pairCountsList = [props[9] for props in ahProps]
        volumesList = [props[10] for props in ahProps]
        centralAntihalos = [[tools.getAntiHalosInSphere(hcentres,135,\
            origin=centre) for centre in np.array([[0,0,0]])] \
            for hcentres in ahCentresListRemap]
        conditionList = [[(deltaCentralList[ns] < 0) & \
            (centralAHs[1]) for centralAHs in centralAntihalos[ns]] \
            for ns in range(0,len(snapList))]
        stackedRadii = np.hstack(antihaloRadii)
        stackedMasses = np.hstack(antihaloMassesList)
        stackedConditions = np.hstack(conditionList)
        rBins = ahProps[0][8]
        rBinStackCentres = plot.binCentres(rBins)
        # Compute stacks:
        [nbarjAllStacked,sigmaAllStacked] = stacking.stackVoidsWithFilter(\
            np.vstack(ahCentresList),stackedRadii,\
            np.where((stackedRadii > rMin) & (stackedRadii < rMax) & \
            stackedConditions & (stackedMasses > mMin) & \
            (stackedMasses <= mMax))[0],snapList[0],rBins,\
            nPairsList = np.vstack(pairCountsList),\
            volumesList = np.vstack(volumesList),\
            method="poisson",errorType="Weighted")
        # Reference stacks:
        [nbarjAllStackedRef,sigmaAllStackedRef] = pickle.load(open(\
            dataFolder + "stacks_reference.p","rb"))
        # Compare to reference:
        self.assertTrue(np.allclose(nbarjAllStacked,nbarjAllStackedRef,\
            rtol=1e-5,atol=1e-8))
        self.assertTrue(np.allclose(sigmaAllStacked,sigmaAllStackedRef,\
            rtol=1e-5,atol=1e-8))
    # Test the procedure for averaging different stacks:
    def test_stack_averaging(self):
        snapNumList = [7000,7200]
        rMin=5
        rMax=25
        dataFolder = "data_for_tests/"
        snapname = "forward/snapshot_001"
        snapnameRev = "reverse/snapshot_001"
        samplesFolder = dataFolder + "reference_constrained/"
        # Load snapshots:
        snapList =  [pynbody.load(samplesFolder + "sample" + \
            str(snapNum) + "/" + snapname) for snapNum in snapNumList]
        snapListRev =  [pynbody.load(samplesFolder + "sample" + \
            str(snapNum) + "/" + snapnameRev) for snapNum in snapNumList]
        N = 512
        boxsize = snapList[0].properties['boxsize'].ratio("Mpc a h**-1")
        nbar = (N/boxsize)**3
        # Load reference antihalo data:
        ahProps = [pickle.load(\
            open(snap.filename + ".AHproperties.p","rb")) \
            for snap in snapList]
        ahCentresList = [props[5] for props in ahProps]
        ahCentresListRemap = [tools.remapAntiHaloCentre(props[5],boxsize) \
            for props in ahProps]
        antihaloRadii = [props[7] for props in ahProps]
        deltaCentralList = [props[11] for props in ahProps]
        antihaloMassesList = [props[3] for props in ahProps]
        pairCountsList = [props[9] for props in ahProps]
        volumesList = [props[10] for props in ahProps]
        centralAntihalos = [[tools.getAntiHalosInSphere(hcentres,135,\
            origin=centre) for centre in np.array([[0,0,0]])] \
            for hcentres in ahCentresListRemap]
        conditionList = [[(deltaCentralList[ns] < 0) & \
            (centralAHs[1]) for centralAHs in centralAntihalos[ns]] \
            for ns in range(0,len(snapList))]
        stackedRadii = np.hstack(antihaloRadii)
        stackedMasses = np.hstack(antihaloMassesList)
        stackedConditions = np.hstack(conditionList)
        rBins = ahProps[0][8]
        rBinStackCentres = plot.binCentres(rBins)
        # Test average stacking:
        [nbarjSepStack,sigmaSepStack] = stacking.computeMeanStacks(\
            ahCentresList,antihaloRadii,antihaloMassesList,conditionList,\
            pairCountsList,volumesList,snapList,nbar,rBins,rMin,rMax,mMin,mMax)
        # Load reference:
        [nbarjSepStackRef,sigmaSepStackRef] = pickle.load(open(\
            dataFolder + "stack_average_reference.p","rb"))
        # Compare to reference:
        self.assertTrue(np.allclose(nbarjSepStack,nbarjSepStackRef,\
            rtol=1e-5,atol=1e-8))
        self.assertTrue(np.allclose(sigmaSepStack,sigmaSepStackRef,\
            rtol=1e-5,atol=1e-8))
    # End-to-end test of the void profiles pipeline:
    def test_whole_stacking_pipeline(self):
        dataFolder = "data_for_tests/"
        snapname = "forward/snapshot_001"
        snapnameRev = "reverse/snapshot_001"
        computedProfiles = getVoidProfilesData(\
            [7000, 7200],[1,3],\
            unconstrainedFolder = dataFolder + "reference_unconstrained/",\
            samplesFolder = dataFolder + "reference_constrained/",\
            snapname=snapname,snapnameRev = snapnameRev)
        referenceProfiles = pickle.load(open(dataFolder + \
            "void_profiles_pipeline_ref.p","rb"))
        for k in range(0,len(referenceProfiles)):
            self.assertTrue(np.allclose(computedProfiles[k],\
                referenceProfiles[k],rtol=1e-5,atol=1e-8))


# Tests for the tools package:
class test_tools(unittest.TestCase):
    def __init__(self,dataFolder= "data_for_tests/",\
            test_subfolder="function_tests_tools/",rtol=1e-5,atol=1e-8,\
            generateTestData=False):
        unittest.TestCase.__init__()
        self.dataFolder=dataFolder
        self.test_subfolder = test_subfolder
        self.rtol=1e-5
        self.atol=1e-8
        self.generateTestData = generateTestData
    def test_getAntiHalosInSphere():
        # Fixed centres to test:
        np.random.seed(1000)
        centreList = np.random.rand((100,3))*200
        computed = tools.getAntiHalosInSphere(centreList,100,\
            origin=([50,50,50]))
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "getAntiHalosInSphere_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "getAntiHalosInSphere_ref.p","rb"))
        self.assertTrue(np.all(computed[0] == reference[0]))
        self.assertTrue(np.all(computed[1] == reference[1]))
    def test_getCentredDensityConstrast(self):
        snapname = "reference_unconstrained/sample1/forward/snapshot_001"
        snap = pynbody.load(snapname)
        boxsize = snap.properties['boxsize'].ratio("Mpc a h**-1")
        np.random.seed(1000)
        centreList = np.random.rand((100,3))*boxsize
        radius = 135
        computed = tools.getCentredDensityConstrast(snap,centreList,radius)
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "getCentredDensityConstrast_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "getCentredDensityConstrast_ref.p","rb"))
        self.assertTrue(np.allclose(computed,reference,\
            rtol=self.rtol,atol=self.atol))
    def test_getHaloAndAntihaloCountsInDensityRange(self):
        snapname = "reference_unconstrained/sample1/forward/snapshot_001"
        snap = pynbody.load(snapname)
        boxsize = snap.properties['boxsize'].ratio("Mpc a h**-1")
        np.random.seed(1000)
        centreList = np.random.rand((100,3))*boxsize
        radius = 135
        # Load the pre-computed (correct) density list:
        deltaList = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "getCentredDensityConstrast_ref.p","rb"))
        # Other parameters:
        mThresh = 1e13
        ahProps = pickle.load(open(snap.filename + ".AHproperties.p","rb"))
        hncentres = ahProps[0]
        hrcentres = ahProps[2]
        hnmasses = ahProps[1]
        hrmasses = ahProps[3]
        deltaCentral = ahProps[11]
        computed = tools.getHaloAndAntihaloCountsInDensityRange(\
            radius,snap,centres,deltaList,mThresh,hncentres,hrcentres,\
            hnmasses,deltaLow=-0.07,deltaHigh=-0.06,n_jobs=-1)
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "getHaloAndAntihaloCountsInDensityRange_ref.p","wb")
        reference = pickle.load(open(self.dataFolder self.test_subfolder + \
            "getHaloAndAntihaloCountsInDensityRange_ref.p","rb"))
        for k in range(0,len(reference)):
            self.assertTrue(np.all(computed[k] == reference[k]))
    def test_getEquivalents(self):
        samplesFolder = self.dataFolder + "reference_constrained/"
        snapNumList = [7000,7200]
        snapList =  [pynbody.load(samplesFolder + "sample" + \
            str(snapNum) + "/" + snapname) for snapNum in snapNumList]
        halos = [snap.halos() for snap in snapList]
        ahPropsList = [pickle.load(open(snap.filename + ".AHproperties.p",\
            "rb")) for snap in snaplist]
        centresList = [ahProps[2] for ahProps in ahPropsList]
        boxsize = snapList[0].properties['boxsize'].ratio("Mpc a h**-1")
        rSearch = 20
        computed = tools.getEquivalents(halos[0],halos[1],centresList[0],\
            centresList[1],boxsize,rSearch)
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "getEquivalents_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "getEquivalents_ref.p","rb"))
        self.assert(np.all(computed == reference))
    def test_loadAbellCatalogue():
        computed = tools.loadAbellCatalogue(self.dataFolder)
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "loadAbellCatalogue_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "loadAbellCatalogue_ref.p","rb"))
        for k in range(0,len(reference)):
            if k == 2:
                self.assert(np.all(computed[k] == reference[k]))
            else:
                self.assert(np.allclose(computed[k],reference[k],\
                    rtol=self.rtol,atol=self.atol))
    def test_getPoissonAndErrors(self):
        np.random.seed(1000)
        bins = np.linspace(0,10,21)
        binCentres = (bins[1:] + bins[0:-1])/2
        counts = 100/binCentres + np.random.randint(len(binCentres),\
            size=binCentres.shape)
        computed = tools.getPoissonAndErrors(bins,counts,alpha=0.32)
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "getPoissonAndErrors_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "getPoissonAndErrors_ref.p","rb"))
        for k in range(0,len(reference)):
            self.assert(np.allclose(computed[k],reference[k],\
                    rtol=self.rtol,atol=self.atol))
    def test_remapBORGSimulation(self):
        snapname = "reference_unconstrained/sample1/forward/snapshot_001"
        snap = pynbody.load(snapname)
        tools.remapBORGSimulation(snap,swapXZ=False,reverse=True)
        computed = snap['pos']
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "remapBORGSimulation_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "remapBORGSimulation_ref.p","rb"))
        self.assertTrue(np.allclose(computed,reference))
    def test_remapAntiHaloCentre(self):
        snapname = "reference_unconstrained/sample1/forward/snapshot_001"
        ahProps = pickle.load(open(snapname + ".AHproperties.p","rb"))
        boxsize = 677.7
        hrcentres = ahProps[5]
        computed = remapAntiHaloCentre(hrcentres,boxsize)
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "remapAntiHaloCentre_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "remapAntiHaloCentre_ref.p","rb"))
    def test_zobovVolumesToPhysical(self):
        standard = self.dataFolder + \
            "reference_constrained/sample7000/forward/snapshot_001"
        snapn = pynbody.load(standard)
        referenceVols = np.load(self.dataFolder + \
            "reference_constrained/volumes_ref.npy")
        volumes = tools.zobovVolumesToPhysical(standard + ".vols",snapn,\
            dtype=np.double,offset=4)
        self.assertTrue(np.allclose(volumes,referenceVols,\
            rtol=self.rtol,atol=self.atol))
    def test_getHaloCentresAndMassesFromCatalogue(self):
        snapname = "reference_unconstrained/sample1/forward/snapshot_001"
        ahProps = pickle.load(open(snapname + ".AHproperties.p","rb"))
        snap = pynbody.load(snapname)
        hcentresRef = ahProps[0]
        hmassesRef = ahProps[1]
        halos = snap.halos()
        [hcentres,hmasses] = tools.getHaloCentresAndMassesFromCatalogue(halos,\
            inMpcs=True)
        self.assertTrue(np.allclose(hcentres,hcentresRef,\
            rtol=self.rtol,atol=self.atol))
        self.assertTrue(np.allclose(hmasses,hmassesRef,\
            rtol=self.rtol,atol=self.atol))
    def test_getHaloMassesAndVirials(self):
        snapname = "reference_unconstrained/sample1/forward/snapshot_001"
        ahProps = pickle.load(open(snapname + ".AHproperties.p","rb"))
        snap = pynbody.load(snapname)
        hcentres = ahProps[0][0:10,:]
        computed = tools.getHaloMassesAndVirials(snap,hcentres,overden=200,\
            rho_def="critical",massUnit="Msol h**-1",distanceUnit="Mpc a h**-1")
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "getHaloMassesAndVirials_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "getHaloMassesAndVirials_ref.p","rb"))
        for k in range(0,len(reference)):
            self.assert(np.allclose(computed[k],reference[k],\
                rtol=self.rtol,atol=self.atol))
    def test_mcmcFileToWhiteNoise(self):
        tools.mcmcFileToWhiteNoise(self.dataFolder + "mcmc_7000.h5",\
            "test_wn_extraction/wn.npy",\
            normalise = True,fromInverseFourier=False,flip=False,reverse=True)
        testCase = np.load("test_wn_extraction/wn.npy")
        if self.generateTestData:
            np.save(self.dataFolder + "wn.npy",testCase)
        reference = np.load(self.dataFolder + "wn.npy")
        self.assertTrue(np.allclose(testCase,reference,\
            rtol=self.rtol,atol=self.atol))
    def test_minmax(self):
        testArray = np.array([1.0,3.0,2.5])
        minmax = tools.minmax(testArray)
        minmaxRef = np.array([1.0,3.0])
        self.assertTrue(np.allclose(minmax,minmaxRef,\
            rtol=self.rtol,atol=self.atol))
    def test_reorderLinearScalar(self):
        vols = np.load(self.dataFolder + \
            "reference_constrained/volumes_ref.npy")
        computed = tools.reorderLinearScalar(vols,512)
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "reorderLinearScalar_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "reorderLinearScalar_ref.p"))
        self.assert(np.allclose(computed,reference,\
            rtol=self.rtol,atol=self.atol))
    def test_getKDTree(self):
        snap = pynbody.load(self.dataFolder + "kdtree_test/test_ic.gadget2")
        boxsize = snap.properties['boxsize'].ratio("Mpc a h**-1")
        tree = tools.getKDtree(snap,reconstructTree=True)
        # Can't really compare the trees themselves so easily, so test instead
        # that we get the same thing with a query:
        np.random.seed(1000)
        randCentres = np.random.rand((100,3))*boxsize
        radius = 20
        indices = np.array(\
            tree.query_ball_point(centres,radius,return_length=True))
        if self.generateTestData:
            pickle.dump(tree,open(self.dataFolder + self.test_subfolder + \
                "kdtree_test/tree.p","wb")
        refTree = pickle.load(open(self.dataFolder + "kdtree_test/tree.p"))
        indicesRef = np.array(\
            refTree.query_ball_point(centres,radius,return_length=True))
        self.assertTrue(np.all(indices == indeicesRef))
    def test_getCountsInHealpixSlices(self):
        ngMCMC = pickle.load(open(self.dataFolder + "ngCounts_ref.p","rb"))
        hpIndices = pickle.load(open(self.dataFolder + "hpIndices_ref.p","rb"))
        ngHP = tools.getCountsInHealpixSlices(ngMCMC,hpIndices,4)
        if self.generateTestData:
            pickle.dump(ngHP,open(self.dataFolder + self.test_subfolder + \
                "hpIndices_ref.p","wb")
        ngHPRef = pickle.load(open(self.dataFolder + "hpCounts_ref.p","rb"))
        self.assertTrue(np.allclose(ngHP,ngHPRef,rtol=self.rtol,atol=self.atol))

# Tests for cosmology functions:
class test_cosmology(unittest.TestCase):
    def __init__(self,dataFolder= "data_for_tests/",\
            test_subfolder="function_tests_cosmology/",rtol=1e-5,atol=1e-8,\
            generateTestData = False):
        unittest.TestCase.__init__()
        self.dataFolder=dataFolder
        self.test_subfolder = test_subfolder
        self.rtol=1e-5
        self.atol=1e-8
        self.generateTestData = generateTestData
    def test_fsigma(self):
        sigmaArr = np.array([0.1,0.5,0.8,1.0,1.2])
        computed = cosmology.fsigma(sigmaArr,1,1,1,1)
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "fsigma_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "fsigma_ref.p","rb"))
        self.assertTrue(np.allclose(computed,reference,\
            rtol=self.rtol,atol=self.atol))
    def test_fLinear(self):
        zList = np.array([0.2,0])
        computed = cosmology.fLinear(zList,0.3,0.7)
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "fLinear_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "fLinear_ref.p","rb"))
        self.assertTrue(np.allclose(computed,reference,\
            rtol=self.rtol,atol=self.atol))
    def test_interpolareTransfer(self):
        [k,ki,Tki] = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "interpolateTransfer_data.p","rb"))
        computed = cosmology.interpolateTransfer(k,ki,Tki)
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "interpolateTransfer_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "interpolateTransfer_ref.p","rb"))
        self.assertTrue(np.allclose(computed,reference,\
            rtol=self.rtol,atol=self.atol))
    def test_Pkinterp(self):
        [k,ki,Tki] = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "interpolateTransfer_data.p","rb"))
        computed = cosmology.Pkinterp(k,Tki,ki,0.9665,1.0)
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "Pkinterp_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "Pkinterp_ref.p","rb"))
        self.assertTrue(np.allclose(computed,reference,\
            rtol=self.rtol,atol=self.atol))
    def test_What(self):
        krange = 10**(np.linspace(-3,1,101))
        computed = tools.What(krange,1.0)
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "What_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "What_ref.p","rb"))
        self.assertTrue(np.allclose(computed,reference,\
            rtol=self.rtol,atol=self.atol))
    def test_Whatp(self):
        krange = 10**(np.linspace(-3,1,101))
        computed = tools.Whatp(krange,1.0)
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "Whatp_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "Whatp_ref.p","rb"))
        self.assertTrue(np.allclose(computed,reference,\
            rtol=self.rtol,atol=self.atol))
    def test_computeSigma80(self):
        [k,ki,Tki] = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "interpolateTransfer_data.p","rb"))
        computed = cosmology.computeSigma80(1.0,0.9665,Tki,ki)
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "computeSigma80_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "computeSigma80_ref.p","rb"))
        self.assertTrue(np.allclose(computed,reference,\
            rtol=self.rtol,atol=self.atol))
    def test_computePKAmplitude(self):
        [k,ki,Tki] = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "interpolateTransfer_data.p","rb"))
        computed = cosmology.computePkAmplitude(np.linspace(0.7,1.3,101),0,\
            ki,Tki,0.9665)
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "computePKAmplitude_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "computePKAmplitude_ref.p","rb"))
        self.assertTrue(np.allclose(computed,reference,rtol=self.rtol,\
            atol=self.atol))
    def test_sigmaIntegrand(self):
        [k,ki,Tki] = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "interpolateTransfer_data.p","rb"))
        computed = cosmology.sigmaIntegrand(k,1.0,ki,Tki,0.9665,1.0)
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "sigmaIntegrand_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "sigmaIntegrand_ref.p","rb"))
        self.assertTrue(np.allclose(computed,reference,rtol=self.rtol,\
            atol=self.atol))
    def test_computeSigma(self):
        [k,ki,Tki] = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "interpolateTransfer_data.p","rb"))
        rhoB = 0.3*2.7754e11
        computed = cosmology.computeSigma(0.0,1e14,rhoB,Tki,ki,0.9665,1.0)
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "computeSigma_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "computeSigma_ref.p","rb"))
        self.assertTrue(np.allclose(computed,reference,rtol=self.rtol,\
            atol=self.atol))
    def test_sigmapIntegrand(self):
        [k,ki,Tki] = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "interpolateTransfer_data.p","rb"))
        computed = cosmology.sigmapIntegrand(k,1.0,1.0,ki,Tki,0.9665,1.0)
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "sigmapIntegrand_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "sigmapIntegrand_ref.p","rb"))
        self.assertTrue(np.allclose(computed,reference,rtol=self.rtol,\
            atol=self.atol))
    def test_computeDSigmaDM(self):
        [k,ki,Tki] = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "interpolateTransfer_data.p","rb"))
        rhoB = 0.3*2.7754e11
        computed = cosmology.computeDSigmaDM(0.0,1e14,Tki,ki,0.9665,1.0)
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "computeDSigmaDM_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "computeDSigmaDM_ref.p","rb"))
        self.assertTrue(np.allclose(computed,reference,rtol=self.rtol,\
            atol=self.atol))
    def test_TMF(self):
        [k,ki,Tki] = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "interpolateTransfer_data.p","rb"))
        Mrange = 10**(np.linspace(13,15,101))
        rhoB = 0.3*2.7754e11
        computed = cosmology.TMF(Mrange,1.0,1.0,1.0,1.0,0.0,\
            rhoB,Tki,ki,0.9665,0.8)
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "TMF_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "TMF_ref.p","rb"))
        self.assertTrue(np.allclose(computed,reference,rtol=self.rtol,\
            atol=self.atol))
    def test_TMF_from_hmf(self):
        computed = cosmology.TMF_from_hmf(1e13,1e15)
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "TMF_from_hmf_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "TMF_from_hmf_ref.p","rb"))
        for k in range(0,len(reference)):
            self.assertTrue(np.allclose(computed[k],reference[k],\
                rtol=self.rtol,atol=self.atol))
    def test_PSMF(self):
        computed = cosmology.PSMF(1e13,1e15)
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "PSMF_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "PSMF_ref.p","rb"))
        for k in range(0,len(reference)):
            self.assertTrue(np.allclose(computed[k],reference[k],\
                rtol=self.rtol,atol=self.atol))
    def test_dndm_to_n(self):
        [dndm,m] = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "TMF_from_hmf_ref.p","rb"))
        computed = cosmology.dndm_to_n(dndm,m,10**(np.linspace(13,15,31)))
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "dndm_to_n_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "dndm_to_n_ref.p","rb"))
        self.assertTrue(np.allclose(computed,reference,rtol=self.rtol,\
            atol=self.atol))
    def test_vol(self):
        computed = cosmology.vol(0.0,0.05,1.0,0.3)
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "vol_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "vol_ref.p","rb"))
        self.assertTrue(computed == reference)
    def test_Lvol(self):
        computed = cosmology.Lvol(0.0,0.05,1.0,0.3)
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "Lvol_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "Lvol_ref.p","rb"))
        self.assertTrue(np.allclose(computed,reference,\
            rtol=self.rtol,atol=self.rtol))
    def test_E(self):
        computed = cosmology.E(0.5,0.3,0.0,0.0)
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "E_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "E_ref.p","rb"))
        self.assertTrue(np.allclose(computed,reference,\
            rtol=self.rtol,atol=self.rtol))
    def test_drCo(self):
        computed = cosmology.drCo(0.0,0.3)
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "E_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "drCo_ref.p","rb"))
        self.assertTrue(np.allclose(computed,reference,\
            rtol=self.rtol,atol=self.rtol))
    def test_fTinker(self):
        computed = cosmology.fTinker(0.0,0.3)
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "fTinker_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "fTinker_ref.p","rb"))
        self.assertTrue(np.allclose(computed,reference,\
            rtol=self.rtol,atol=self.rtol))
    def test_tmf(self):
        [k,ki,Tki] = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "interpolateTransfer_data.p","rb"))
        Mrange = 10**(np.linspace(13,15,101))
        rhoB = 0.3*2.7754e11
        Pki = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "Pkinterp_ref.p","rb"))
        computed = cosmology.tmf(Mrange,ki,Pki,rhoB,0.3)
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "tmf_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "tmf_ref.p","rb"))
        self.assertTrue(np.allclose(computed,reference,\
            rtol=self.rtol,atol=self.rtol))
    def test_linearGrowthD(self):
        computed = cosmology.linearGrowthD(0.0,0.3)
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "linearGrowthD_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "linearGrowthD_ref.p","rb"))
        self.assertTrue(np.allclose(computed,reference,\
            rtol=self.rtol,atol=self.rtol))
    def test_linearGrowthf(self):
        computed = cosmology.linearGrowthf(0.0,0.3)
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "linearGrowthf_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "linearGrowthf_ref.p","rb"))
        self.assertTrue(np.allclose(computed,reference,\
            rtol=self.rtol,atol=self.rtol))
    def test_windowRtoM(self):
        rhoB = 0.3*2.7754e11
        computed = cosmology.windowRtoM(1.0,rhoB)
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "windowRtoM_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "windowRtoM_ref.p","rb"))
        self.assertTrue(np.allclose(computed,reference,\
            rtol=self.rtol,atol=self.rtol))
    def test_windowMtoR(self):
        rhoB = 0.3*2.7754e11
        computed = cosmology.windowMtoR(1.0,rhoB)
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "windowMtoR_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "windowMtoR_ref.p","rb"))
        self.assertTrue(np.allclose(computed,reference,\
            rtol=self.rtol,atol=self.rtol))
    def test_W_spatialTH(self):
        computed = cosmology.W_spatialTH(1.0)
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "W_spatialTH_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "W_spatialTH_ref.p","rb"))
        self.assertTrue(np.allclose(computed,reference,\
            rtol=self.rtol,atol=self.rtol))
    def test_W_spatialTHp(self):
        computed = cosmology.W_spatialTHp(1.0)
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "W_spatialTHp_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "W_spatialTHp_ref.p","rb"))
        self.assertTrue(np.allclose(computed,reference,\
            rtol=self.rtol,atol=self.rtol))
    def test_sigma2Mtophat(self):
        Pki = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "Pkinterp_ref.p","rb"))
        rhoB = 0.3*2.7754e11
        computed = cosmology.sigma2Mtophat(1e14,Pki,rhoB)
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "sigma2Mtophat_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "sigma2Mtophat_ref.p","rb"))
        self.assertTrue(np.allclose(computed,reference,\
            rtol=self.rtol,atol=self.rtol))
    def test_sigmaTinker(self):
        Pki = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "Pkinterp_ref.p","rb"))
        rhoB = 0.3*2.7754e11
        computed = cosmology.sigmaTinker(1.0,Pki,rhoB,1e-3,10)
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "sigmaTinker_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "sigmaTinker_ref.p","rb"))
        self.assertTrue(np.allclose(computed,reference,\
            rtol=self.rtol,atol=self.rtol))
    def test_jacobian_tinker(self):
        Pki = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "Pkinterp_ref.p","rb"))
        rhoB = 0.3*2.7754e11
        computed = cosmology.jacobian_tinker(0.8,1e14,rhoB,Pki,1e-3,10)
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "jacobian_tinker_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "jacobian_tinker_ref.p","rb"))
        self.assertTrue(np.allclose(computed,reference,\
            rtol=self.rtol,atol=self.rtol))
    def test_sigmaRspatialTH(self):
        Pki = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "Pkinterp_ref.p","rb"))
        [k,ki,Tki] = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "interpolateTransfer_data.p","rb"))
        computed = cosmology.sigmaRspatialTH(1.0,ki,Pki)
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "sigmaRspatialTH_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "sigmaRspatialTH_ref.p","rb"))
        self.assertTrue(np.allclose(computed,reference,\
            rtol=self.rtol,atol=self.rtol))
    def test_f_SVdW(self):
        computed = cosmology.f_SVdW(0.8,-1.686,1.686)
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "f_SVdW_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "f_SVdW_ref.p","rb"))
        self.assertTrue(np.allclose(computed,reference,\
            rtol=self.rtol,atol=self.rtol))
    def test_jacobian_SVdW(self):
        Pki = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "Pkinterp_ref.p","rb"))
        rhoB = 0.3*2.7754e11
        computed = cosmology.jacobian_SVdW(0.8,1e14,rhoB,Pki)
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "jacobian_SVdW_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "jacobian_SVdW_ref.p","rb"))
        self.assertTrue(np.allclose(computed,reference,\
            rtol=self.rtol,atol=self.rtol))
    def test_jacobian_spatialTH(self):
        Pki = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "Pkinterp_ref.p","rb"))
        [k,ki,Tki] = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "interpolateTransfer_data.p","rb"))
        rhoB = 0.3*2.7754e11
        computed = cosmology.jacobian_spatialTH(0.8,1e14,rhoB,ki,Pki)
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "jacobian_spatialTH_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "jacobian_spatialTH_ref.p","rb"))
        self.assertTrue(np.allclose(computed,reference,\
            rtol=self.rtol,atol=self.rtol))
    def test_rhoSI(self):
        computed = cosmology.rhoSI(0.3)
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "rhoSI_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "rhoSI_ref.p","rb"))
        self.assertTrue(np.allclose(computed,reference,\
            rtol=self.rtol,atol=self.rtol))
    def test_rhoCos(self):
        computed = cosmology.rhoCos(0.3)
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "rhoCos_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "rhoCos_ref.p","rb"))
        self.assertTrue(np.allclose(computed,reference,\
            rtol=self.rtol,atol=self.rtol))
    def test_powerSpectrum(self):
        computed = cosmology.powerSpectrum()
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "powerSpectrum_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "powerSpectrum_ref.p","rb"))
        for k in range(0,len(reference)):
            self.assertTrue(np.allclose(computed[k],reference[k],\
                rtol=self.rtol,atol=self.rtol))
    def test_pkToxi(self):
        Pki = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "Pkinterp_ref.p","rb"))
        [k,ki,Tki] = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "interpolateTransfer_data.p","rb"))
        x = np.linspace(0,1.0,101)
        computed = cosmology.pkToxi(x,ki,Pki)
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "pkToxi_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "pkToxi_ref.p","rb"))
        self.assertTrue(np.allclose(computed,reference,\
            rtol=self.rtol,atol=self.rtol))
    def test_vLinear(self):
        computed = cosmology.vLinear(1.0,200,0.3,0.7)
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "vLinear_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "vLinear_ref.p","rb"))
        self.assertTrue(np.allclose(computed,reference,\
            rtol=self.rtol,atol=self.rtol))
    def test_deltaCumulative(self):
        pairCountsRef = np.load(dataFolder + \
            "reference_constrained/pair_counts_ref.npy")
        volumesListRef = np.load(dataFolder + \
            "reference_constrained/volumes_list_ref.npy")
        nbar = (512/677.7)**3
        computed = cosmology.deltaCumulative(pairCountsRef,volumesListRef,nbar)
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "deltaCumulative_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "deltaCumulative_ref.p","rb"))
        self.assertTrue(np.allclose(computed,reference,\
            rtol=self.rtol,atol=self.rtol))
    def test_luminosityToComoving(self):
        cosmo = astropy.cosmology.FlatLambdaCDM(H0 = 70,Om0 = 0.3,Ob0 = 0.045)
        lumDist = np.linspace(0.1,100,101)
        computed = cosmology.luminosityToComoving(lumDist,cosmo)
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "luminosityToComoving_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "luminosityToComoving_ref.p","rb"))
        self.assertTrue(np.allclose(computed,reference,\
            rtol=self.rtol,atol=self.rtol))
    def test_comovingToLuminosity(self):
        comovingDistance = np.linspace(0.1,100,101)
        cosmo = astropy.cosmology.FlatLambdaCDM(H0 = 70,Om0 = 0.3,Ob0 = 0.045)
        computed = cosmology.comovingToLuminosity(comovingDistance,cosmo)
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "comovingToLuminosity_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "comovingToLuminosity_ref.p","rb"))
        self.assertTrue(np.allclose(computed,reference,\
            rtol=self.rtol,atol=self.rtol))
    def test_computePowerSpectrum(self):
        standard = dataFolder + \
            "reference_constrained/sample7000/forward/snapshot_001"
        snap = pynbody.load(standard)
        computed = cosmology.computePowerSpectrum(snap,\
            directory=self.dataFolder + "test_temp/")
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "computePowerSpectrum_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "computePowerSpectrum_ref.p","rb"))
        for k in range(0,len(reference)):
            self.assertTrue(np.allclose(computed[k],reference[k],\
                rtol=self.rtol,atol=self.rtol))
    def test_sigmaPk(self):
        Pki = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "Pkinterp_ref.p","rb"))
        Nk = 100
        Np = 100
        computed = cosmology.sigmaPk(Pki,Nk,Np)
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "sigmaPk_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "sigmaPk_ref.p","rb"))
        self.assertTrue(np.allclose(computed,reference,\
            rtol=self.rtol,atol=self.rtol))
    def test_overdensity(self):
        standard = dataFolder + \
            "reference_constrained/sample7000/forward/snapshot_001"
        snap = pynbody.load(standard)
        computed = cosmology.overdensity(snap,135)
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "overdensity_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "overdensity_ref.p","rb"))
        self.assertTrue(np.allclose(computed,reference,\
            rtol=self.rtol,atol=self.rtol))
    def test_lowMemoryOverdensity(self):
        standard = dataFolder + \
            "reference_constrained/sample7000/forward/snapshot_001"
        snap = pynbody.load(standard)
        computed = cosmology.lowMemoryOverdensity(snap,135)
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "lowMemoryOverdensity_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "lowMemoryOverdensity_ref.p","rb"))
        self.assertTrue(np.allclose(computed,reference,\
            rtol=self.rtol,atol=self.rtol))
    def test_sampleOverdensities(self):
        standard = dataFolder + \
            "reference_constrained/sample7000/forward/snapshot_001"
        snap = pynbody.load(standard)
        computed = cosmology.sampleOverdensities(snap,135,seed=1000)
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "sampleOverdensities_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "sampleOverdensities_ref.p","rb"))
        for k in range(0,len(reference)):
            self.assertTrue(np.allclose(computed[k],reference[k],\
                rtol=self.rtol,atol=self.rtol))
    def test_computeLocalUnderdensity(self):
        standard = dataFolder + \
            "reference_constrained/sample7000/forward/snapshot_001"
        snap = pynbody.load(standard)
        rBins = np.linspace(10,100,101)
        computed = cosmology.computeLocalUnderdensity(snap,rBins)
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "computeLocalUnderdensity_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "computeLocalUnderdensity_ref.p","rb"))
        for k in range(0,len(reference)):
            self.assertTrue(np.allclose(computed[k],reference[k],\
                rtol=self.rtol,atol=self.rtol))
    def test_getMassProfile(self):
        standard = dataFolder + \
            "reference_constrained/sample7000/forward/snapshot_001"
        snap = pynbody.load(standard)
        radii = np.linspace(0.1,10,21)
        computed = cosmology.getMassProfile(radii,np.array([0.0]*3),snap)
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "getMassProfile_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "getMassProfile_ref.p","rb"))
        self.assertTrue(np.allclose(computed,reference,\
            rtol=self.rtol,atol=self.rtol))
    def test_cMBhattacharya(self):
        M = 10**(np.linspace(13,15,101))
        computed = cosmology.cMBhattacharya(M)
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "cMBhattacharya_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "cMBhattacharya_ref.p","rb"))
        self.assertTrue(np.allclose(computed,reference,\
            rtol=self.rtol,atol=self.rtol))
    def test_mNFW(self):
        computed = cosmology.mNFW(0.1)
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "mNFW_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "mNFW_ref.p","rb"))
        self.assertTrue(np.allclose(computed,reference,\
            rtol=self.rtol,atol=self.rtol))
    def test_criticalFunction(self):
        computed = cosmology.criticalFunction(0.1,1e14,200,500)
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "criticalFunction_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "criticalFunction_ref.p","rb"))
        self.assertTrue(np.allclose(computed,reference,\
            rtol=self.rtol,atol=self.rtol))
    def test_convertDeltaToCritical(self):
        computed = []
        computed.append(cosmology.convertDeltaToCritical(200,"mean"))
        computed.append(cosmology.convertDeltaToCritical(200,"critical"))
        computed.append(cosmology.convertDeltaToCritical(200,"virial"))
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "convertDeltaToCritical_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "convertDeltaToCritical_ref.p","rb"))
        for k in range(0,len(reference)):
            self.assertTrue(np.allclose(computed[k],reference[k],\
                rtol=self.rtol,atol=self.rtol))
    def test_getBoundedInterval(self):
        computed = cosmology.getBoundedInterval(1e14,200,500)
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "getBoundedInterval_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "getBoundedInterval_ref.p","rb"))
        self.assertTrue(np.allclose(computed,reference,\
            rtol=self.rtol,atol=self.rtol))
    def test_convertCriticalMass(self):
        computed = cosmology.convertCriticalMass(1e14,500)
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "convertCriticalMass_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "convertCriticalMass_ref.p","rb"))
        self.assertTrue(np.allclose(computed,reference,\
            rtol=self.rtol,atol=self.rtol))
    def test_eulerToZ(self):
        standard = dataFolder + \
            "reference_constrained/sample7000/forward/snapshot_001"
        snap = pynbody.load(standard)
        Om0 = snap.properties['omegaM0']
        h = snap.properties['h']
        cosmo = astropy.cosmology.FlatLambdaCDM(H0 = 100*h,\
            Om0 = Om0,Ob0 = 0.045)
        boxsize = snap.properties['boxsize'].ratio("Mpc a h**-1")
        computed = self.eulerToZ(snap['pos'][0:100,:],snap['vel'][0:100,:],\
            cosmo,boxsize,h)
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "eulerToZ_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "eulerToZ_ref.p","rb"))
        self.assertTrue(np.allclose(computed,reference,\
            rtol=self.rtol,atol=self.rtol))

# antihalos functions:
class test_cosmology(unittest.TestCase):
    def __init__(self,dataFolder= "data_for_tests/",\
            test_subfolder="function_tests_antihalos/",rtol=1e-5,atol=1e-8,\
            generateTestData = False):
        unittest.TestCase.__init__()
        self.dataFolder=dataFolder
        self.test_subfolder = test_subfolder
        self.rtol=1e-5
        self.atol=1e-8
        self.generateTestData = generateTestData
    def test_voidsRadiiFromAntiHalos(self):
        standard = dataFolder + \
            "reference_constrained/sample7000/forward/snapshot_001"
        reverse = dataFolder + \
            "reference_constrained/sample7000/reverse/snapshot_001"
        snapn = pynbody.load(standard)
        snapr = pynbody.load(reverse)
        hn = snapn.halos()
        hr = snapr.halos()
        volumes = np.load(dataFolder + \
            "reference_constrained/volumes_ref.npy")
        computed = antihalos.voidsRadiiFromAntiHalos(snapn,snapr,hn,hr,volumes)
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "voidsRadiiFromAntiHalos_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "voidsRadiiFromAntiHalos_ref.p","rb"))
        self.assertTrue(np.allclose(computed,reference,\
            rtol=self.rtol,atol=self.rtol))
    def test_computeAntiHaloCentres(self):
        standard = dataFolder + \
            "reference_constrained/sample7000/forward/snapshot_001"
        reverse = dataFolder + \
            "reference_constrained/sample7000/reverse/snapshot_001"
        snapn = pynbody.load(standard)
        snapr = pynbody.load(reverse)
        hr = snapr.halos()
        volumes = np.load(dataFolder + \
            "reference_constrained/volumes_ref.npy")
        computed = antihalos.computeAntiHaloCentres(hr,snapn,volumes)
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "computeAntiHaloCentres_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "computeAntiHaloCentres_ref.p","rb"))
        self.assertTrue(np.allclose(computed,reference,\
            rtol=self.rtol,atol=self.rtol))
    def test_getAntiHaloMasses(self):
        reverse = dataFolder + \
            "reference_constrained/sample7000/reverse/snapshot_001"
        snapr = pynbody.load(reverse)
        hr = snapr.halos()
        computed = []
        computed.append(antihalos.getAntiHaloMasses(hr))
        computed.append(antihalos.getAntiHaloMasses(hr,fixedMass=True))
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "getAntiHaloMasses_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "getAntiHaloMasses_ref.p","rb"))
        for k in range(0,len(reference)):
            self.assertTrue(np.allclose(computed[k],reference[k],\
                rtol=self.rtol,atol=self.rtol))
    def test_getAntiHaloDensities(self):
        standard = dataFolder + \
            "reference_constrained/sample7000/forward/snapshot_001"
        reverse = dataFolder + \
            "reference_constrained/sample7000/reverse/snapshot_001"
        snapn = pynbody.load(standard)
        snapr = pynbody.load(reverse)
        hr = snapr.halos()
        volumes = np.load(dataFolder + \
            "reference_constrained/volumes_ref.npy")
        computed = antihalos.getAntiHaloDensities(hr,snapn,volumes=volumes)
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "getAntiHaloDensities_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "getAntiHaloDensities_ref.p","rb"))
        self.assertTrue(np.allclose(computed,reference,\
            rtol=self.rtol,atol=self.rtol))
    def test_fitMassAndRadii(self):
        standard = dataFolder + \
            "reference_constrained/sample7000/forward/snapshot_001"
        snap = pynbody.load(standard)
        ahProps = pickle.load(\
            open(snap.filename + ".AHproperties.p","rb"))
        antihaloRadii = ahProps[7]
        antihaloMasses = ahProps[3]
        computed = antihalos.fitMassAndRadii(antihaloMasses,antihaloRadii)
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "fitMassAndRadii_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "fitMassAndRadii_ref.p","rb"))
        self.assertTrue(np.allclose(computed,reference,\
            rtol=self.rtol,atol=self.rtol))
    def test_MtoR(self):
        computed = antihalos.MtoR(0.1,0.2,0.3)
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "MtoR_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "MtoR_ref.p","rb"))
        self.assertTrue(np.allclose(computed,reference,\
            rtol=self.rtol,atol=self.rtol))
    def test_RtoM(self):
        computed = antihalos.RtoM(0.1,0.2,0.3)
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "RtoM_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "RtoM_ref.p","rb"))
        self.assertTrue(np.allclose(computed,reference,\
            rtol=self.rtol,atol=self.rtol))
    def test_computePeriodicCentreWeighted(self):
        standard = dataFolder + \
            "reference_constrained/sample7000/forward/snapshot_001"
        snap = pynbody.load(standard)
        snapsort = np.argsort(snap['iord'])
        volumes = np.load(dataFolder + \
            "reference_constrained/volumes_ref.npy")
        halos = snap.halos()
        periodicity = snap.properties['boxsize'].ratio("Mpc a h**-1")
        computed = antihalos.computePeriodicCentreWeighted(halos[1]['pos'],\
            volumes[snapsort][halos[1]['iord']],periodicity)
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "computePeriodicCentreWeighted_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "computePeriodicCentreWeighted_ref.p","rb"))
        self.assertTrue(np.allclose(computed,reference,\
            rtol=self.rtol,atol=self.rtol))
    def test_getCoincidingVoids(self):
        standard = dataFolder + \
            "reference_constrained/sample7000/forward/snapshot_001"
        snap = pynbody.load(standard)
        ahProps = pickle.load(\
            open(snap.filename + ".AHproperties.p","rb"))
        ahCentresList = ahProps[5]
        computed = antihalos.getCoincidingVoids(ahCentresList[0,:],40,\
            ahCentresList)
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "getCoincidingVoids_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "getCoincidingVoids_ref.p","rb"))
        self.assertTrue(np.all(computed[0]==reference[0])
    def test_getCoincidingVoidsInRadiusRange(self):
        standard = dataFolder + \
            "reference_constrained/sample7000/forward/snapshot_001"
        snap = pynbody.load(standard)
        ahProps = pickle.load(\
            open(snap.filename + ".AHproperties.p","rb"))
        ahCentresList = ahProps[5]
        antihaloRadii = ahProps[7]
        computed = antihalos.getCoincidingVoidsInRadiusRange(\
            ahCentresList[0,:],40,ahCentresList,antihaloRadii,5,20)
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "getCoincidingVoidsInRadiusRange_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "getCoincidingVoidsInRadiusRange_ref.p","rb"))
        self.assertTrue(np.all(computed==reference)
    def test_getAntihaloOverlapWithVoid(self):
        volumes = np.load(dataFolder + "reference_constrained/volumes_ref.npy")
        reverse1 = dataFolder + \
            "reference_constrained/sample7000/reverse/snapshot_001"
        reverse2 = dataFolder + \
            "reference_constrained/sample7200/reverse/snapshot_001"
        snapr1 = pynbody.load(reverse1)
        snapr2 = pynbody.load(reverse2)
        hr1 = snapr1.halos()
        hr2 = snapr2.halos()
        computed = antihalos.getAntihaloOverlapWithVoid(hr1[1]['iord'],\
            hr2[1]['iord'],volumes)
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "getAntihaloOverlapWithVoid_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "getAntihaloOverlapWithVoid_ref.p","rb"))
        for k in range(0,len(reference)):
            self.assertTrue(np.allclose(computed[k],reference[k],\
                rtol=self.rtol,atol=self.rtol))
    def test_getOverlapFractions(self):
        print("WARNING - TEST OF getOverlapFractions NOT YET IMPLEMENTED")
        self.assertTrue(True)
    def test_getVoidOverlapFractionsWithAntihalos(self):
        volumes = np.load(dataFolder + "reference_constrained/volumes_ref.npy")
        reverse = dataFolder + \
            "reference_constrained/sample7000/reverse/snapshot_001"
        snapr = pynbody.load(reverse)
        standard = dataFolder + \
            "reference_constrained/sample7000/forward/snapshot_001"
        snap = pynbody.load(standard)
        hr = snapr.halos()
        ahProps = pickle.load(\
            open(snapn.filename + ".AHproperties.p","rb"))
        computed = antihalos.getVoidOverlapFractionsWithAntihalos(\
            hr[1]['iord'],hr,np.arange(len(hr)))
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "getVoidOverlapFractionsWithAntihalos_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "getVoidOverlapFractionsWithAntihalos_ref.p","rb"))
        self.assertTrue(np.allclose(computed,reference,\
            rtol=self.rtol,atol=self.rtol))
    def test_removeSubvoids(self):
        print("WARNING - TEST OF removeSubvoids NOT YET IMPLEMENTED")
        self.assertTrue(True)
    def test_getAntiHaloVoidCandidates(self):
        print("WARNING - TEST OF getAntiHaloVoidCandidates NOT YET IMPLEMENTED")
        self.assertTrue(True)
    def test_computeZoneCentres(self):
        print("WARNING - TEST OF computeZoneCentres NOT YET IMPLEMENTED")
        self.assertTrue(True)
    def test_getCorrespondingZoneCandidates(self):
        print("WARNING - TEST OF getCorrespondingZoneCandidates" + \
            " NOT YET IMPLEMENTED")
        self.assertTrue(True)
    def test_getCorrespondingSubVoidCandidates(self):
        print("WARNING - TEST OF getCorrespondingSubVoidCandidates" + \
            " NOT YET IMPLEMENTED")
        self.assertTrue(True)
    def test_runGenPk(self):
        print("WARNING - TEST OF getCorrespondingSubVoidCandidates" + \
            " NOT YET IMPLEMENTED")
        self.assertTrue(True)
    def test_simulationCorrelation(self):
        rBins = np.linspace(0.1,50,21)
        boxsize = 677.7
        standard = dataFolder + \
            "reference_constrained/sample7000/forward/snapshot_001"
        snap = pynbody.load(standard)
        computed = antihalos.simulationCorrelation(rBins,boxsize,snap['pos'])
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "simulationCorrelation_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "simulationCorrelation_ref.p","rb"))
        self.assertTrue(np.allclose(computed,reference,\
            rtol=self.rtol,atol=self.rtol))
    def test_getCrossCorrelations(self):
        standard1 = dataFolder + \
            "reference_constrained/sample7000/standard/snapshot_001"
        standard2 = dataFolder + \
            "reference_constrained/sample7200/standard/snapshot_001"
        ahProps1 = pickle.load(open(standard1 + ".AHproperties.p","rb"))
        ahProps2 = pickle.load(open(standard2 + ".AHproperties.p","rb"))
        centres1 = ahProps1[5]
        centres2 = ahProps2[5]
        radii1 = ahProps1[7]
        radii2 = ahProps2[7]
        computed = antihalos.getCrossCorrelations(centres1,centres2,\
            radii1,radii2,boxsize=677.7)
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "getCrossCorrelations_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "getCrossCorrelations_ref.p","rb"))
        for k in range(0,len(reference)):
            self.assertTrue(np.allclose(computed[k],reference[k],\
                rtol=self.rtol,atol=self.rtol))
    def test_getAutoCorrelations(self):
        standard1 = dataFolder + \
            "reference_constrained/sample7000/standard/snapshot_001"
        standard2 = dataFolder + \
            "reference_constrained/sample7200/standard/snapshot_001"
        ahProps1 = pickle.load(open(standard1 + ".AHproperties.p","rb"))
        ahProps2 = pickle.load(open(standard2 + ".AHproperties.p","rb"))
        centres1 = ahProps1[5]
        centres2 = ahProps2[5]
        radii1 = ahProps1[7]
        radii2 = ahProps2[7]
        computed = antihalos.getAutoCorrelations(centres1,centres2,\
            radii1,radii2,boxsize=677.7)
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "getAutoCorrelations_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "getAutoCorrelations_ref.p","rb"))
        for k in range(0,len(reference)):
            self.assertTrue(np.allclose(computed[k],reference[k],\
                rtol=self.rtol,atol=self.rtol))
    def test_getStacks(self):
        print("WARNING - TEST OF getStacks" + \
            " NOT YET IMPLEMENTED")
        self.assertTrue(True)

class test_context(unittest.TestCase):
    def __init__(self,dataFolder= "data_for_tests/",\
            test_subfolder="function_tests_context/",rtol=1e-5,atol=1e-8,\
            generateTestData = False):
        unittest.TestCase.__init__()
        self.dataFolder=dataFolder
        self.test_subfolder = test_subfolder
        self.rtol=1e-5
        self.atol=1e-8
        self.generateTestData = generateTestData
    def getIsFloatArray(x):
        return (x.dtype == np.float32 or x.dtype == np.float64)
    def getIsIntArray(x):
        return (x.dtype == int or x.dtype == np.int64)
    def arrayComparison(reference,computed):
        if self.getIsFloatArray(reference[k]):
            # Make sure the float arrays are close:
            self.assertTrue(np.allclose(computed[k],\
                reference[k],rtol=self.rtol,atol=self.atol))
        elif self.getIsIntArray(reference[k]):  
            # Compare as ints: (exact equality):
            self.assertTrue(np.all(computed[k] == reference[k]))
        else:
            # Hope that whatever is in this array con be
            # meaningfully compared:
            self.assertTrue(np.all(computed[k] == reference[k]))
    def comparison(computed,name,mode="array",refData = None):
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                name + "_ref.p","wb")
        if refFile is None:
            # Use default filename:
            refData = self.dataFolder + self.test_subfolder + \
                name + "_ref.p"
            reference = pickle.load(open(refData,"rb"))
        elif type(refData) == str:
            # Load the provided file:
            reference = pickle.load(open(refData,"rb"))
        else:
            # Assume we passes the actual data for the comparison:
            reference = refData
        # Now do the comparison:
        if mode == "array":
            self.assertTrue(np.allclose(computed,reference,\
                rtol=self.rtol,atol=self.atol))
        if mode == "ints":
            self.assertTrue(np.all(computed == reference))
        elif mode == "list_arrays":
            for k in range(0,len(reference)):
                self.assertTrue(np.allclose(computed[k],reference[k],\
                    rtol=self.rtol,atol=self.atol))
        elif mode == "list_ints":
            for k in range(0,len(reference)):
                self.assertTrue(np.all(computed[k]==reference[k]))
        elif mode == "auto":
            if type(reference) == list:
                for k in range(0,len(reference)):
                    if type(reference[k]) == np.ndarray:
                        
                
                            
    def test_get_nearest_halos(self):
        standard = dataFolder + \
            "reference_constrained/sample7000/standard/snapshot_001"
        snap = pynbody.load(standard)
        hn = snap.halos()
        ahProps = pickle.load(open(standard + ".AHproperties.p","rb"))
        centres = ahProps[5]
        computed = context.get_nearest_halos(np.array([0.0]*3),hn)
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "get_nearest_halos_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "get_nearest_halos_ref.p","rb"))
        self.assertTrue(np.all(computed == reference))
    def test_halo_centres_and_mass(self):
        standard = dataFolder + \
            "reference_constrained/sample7000/standard/snapshot_001"
        snap = pynbody.load(standard)
        hn = snap.halos()
        computed = context.halo_centres_and_mass(hn)
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "halo_centres_and_mass_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "halo_centres_and_mass_ref.p","rb"))
        for k in range(0,len(reference)):
            self.assertTrue(np.allclose(computed[k],reference[k],\
                rtol=self.rtol,atol=self.rtol))
    def test_void_centres(self):
        standard = dataFolder + \
            "reference_constrained/sample7000/standard/snapshot_001"
        reverse = dataFolder + \
            "reference_constrained/sample7000/reverse/snapshot_001"
        snapn = pynbody.load(standard)
        snapr = pynbody.load(reverse)
        hr = snapr.halos()
        computed = context.void_centres(snapn,snapr,hr)
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "void_centres_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "void_centres_ref.p","rb"))
        self.assertTrue(np.allclose(computed,reference,\
            rtol=self.rtol,atol=self.rtol))
    def test_computePeriodicCentreWeighted(self):
        standard = dataFolder + \
            "reference_constrained/sample7000/standard/snapshot_001"
        reverse = dataFolder + \
            "reference_constrained/sample7000/reverse/snapshot_001"
        snapn = pynbody.load(standard)
        snapr = pynbody.load(reverse)
        volumes = tools.zobovVolumesToPhysical(standard + ".vols",snapn,\
            dtype=np.double,offset=4)
        snapsort = np.argsort(snapr['iord'])
        hr = snapr.halos()
        boxsize = snapn.properties['boxsize'].ratio("Mpc a h**-1")
        computed = context.computePeriodicCentreWeighted(\
            snapn['pos'][snapsort[hr[1]['iord']],:],\
            volumes[snapsort[hr[1]['iord']]],\
            boxsize)
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "computePeriodicCentreWeighted_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "computePeriodicCentreWeighted_ref.p","rb"))
        self.assertTrue(np.allclose(computed,reference,\
            rtol=self.rtol,atol=self.rtol))
    def test_periodicCentreWeighted(self):
        standard = dataFolder + \
            "reference_constrained/sample7000/standard/snapshot_001"
        reverse = dataFolder + \
            "reference_constrained/sample7000/reverse/snapshot_001"
        snapn = pynbody.load(standard)
        snapr = pynbody.load(reverse)
        hr = snapr.halos()
        volumes = tools.zobovVolumesToPhysical(standard + ".vols",snapn,\
            dtype=np.double,offset=4)
        boxsize = snapn.properties['boxsize'].ratio("Mpc a h**-1")
        snapsort = np.argsort(snapr['iord'])
        computed = context.periodicCentreWeighted(hr[1],\
            volumes[snapsort][hr[1]['iord']],boxsize)
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "periodicCentreWeighted_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "periodicCentreWeighted_ref.p","rb"))
        self.assertTrue(np.allclose(computed,reference,\
            rtol=self.rtol,atol=self.rtol))
    def test_periodicCentre(self):
        standard = dataFolder + \
            "reference_constrained/sample7000/standard/snapshot_001"
        reverse = dataFolder + \
            "reference_constrained/sample7000/reverse/snapshot_001"
        snapn = pynbody.load(standard)
        snapr = pynbody.load(reverse)
        hr = snapr.halos()
        boxsize = snapn.properties['boxsize'].ratio("Mpc a h**-1")
        computed = context.periodicCentre(hr[1],boxsize)
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "periodicCentre_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "periodicCentre_ref.p","rb"))
        self.assertTrue(np.allclose(computed,reference,\
            rtol=self.rtol,atol=self.rtol))
    def test_halo_filter(self):
        standard = dataFolder + \
            "reference_constrained/sample7000/standard/snapshot_001"
        snapn = pynbody.load(standard)
        hn = snapn.halos()
        boxsize = snapn.properties['boxsize'].ratio("Mpc a h**-1")
        filt = pynbody.filt.Sphere(20,np.array([boxsize/2]*3))
        computed = context.halo_filter(snapn,hr,filt)
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "halo_filter_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "halo_filter_ref.p","rb"))
        self.assertTrue(np.all(reference == computed))
    def test_halos_in_sphere(self):
        standard = dataFolder + \
            "reference_constrained/sample7000/standard/snapshot_001"
        snapn = pynbody.load(standard)
        boxsize = snapn.properties['boxsize'].ratio("Mpc a h**-1")
        hn = snapn.halos()
        computed = context.halo_filter(hn,135,np.array([boxsize/2]*3))
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "halos_in_sphere_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "halos_in_sphere_ref.p","rb"))
        self.assertTrue(np.all(reference == computed))
    def test_voids_in_sphere(self):
        standard = dataFolder + \
            "reference_constrained/sample7000/standard/snapshot_001"
        reverse = dataFolder + \
            "reference_constrained/sample7000/reverse/snapshot_001"
        snapn = pynbody.load(standard)
        snapr = pynbody.load(reverse)
        hr = snapr.halos()
        boxsize = snapn.properties['boxsize'].ratio("Mpc a h**-1")
        computed = context.voids_in_sphere(hr,135,np.array([boxsize/2]*3),\
            snapn,snapr)
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "voids_in_sphere_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "voids_in_sphere_ref.p","rb"))
        self.assertTrue(np.all(reference == computed))
    def test_rotation_between(self):
        np.random.seed(1000)
        a = np.randon.random(3)
        b = np.randon.random(3)
        computed = context.rotation_between(a,b)
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "rotation_between_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "rotation_between_ref.p","rb"))
        self.assertTrue(np.allclose(computed,reference,\
            rtol=self.rtol,atol=self.rtol))
    def test_orthogonal_part(self):
        np.random.seed(1000)
        a = np.randon.random(3)
        b = np.randon.random(3)
        computed = context.orthogonal_part(a,b)
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "orthogonal_part_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "orthogonal_part_ref.p","rb"))
        self.assertTrue(np.allclose(computed,reference,\
            rtol=self.rtol,atol=self.rtol))
    def test_cluster_fit(self):
        np.random.seed(1000)
        r1 = np.random(3)
        r2 = np.random(3)
        r3 = np.random(3)
        R1 = r1 + 0.1*np.random(3)
        R2 = r2 + 0.1*np.random(3)
        R3 = r3 + 0.1*np.random(3)
        cluster = np.random(3)
        computed = context.cluster_fit(cluster,r1,r2,r3,R1,R2,R3)
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "cluster_fit_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "cluster_fit_ref.p","rb"))
        self.assertTrue(np.allclose(computed,reference,\
            rtol=self.rtol,atol=self.rtol))
    def test_position3d(self):
        np.random.seed(1000)
        ra = 360*np.random.random(100)
        dec = 180*np.random.random(100) - 90
        r = np.random.random(100)
        equatorial = np.vstack((ra,dec,r)).transpose()
        computed = context.position3d(equatorial)
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "position3d_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "position3d_ref.p","rb"))
        self.assertTrue(np.allclose(computed,reference,\
            rtol=self.rtol,atol=self.rtol))
    def test_galactic_to_equatorial(self):
        np.random.seed(1000)
        l = 360*np.random.random(100)
        b = 180*np.random.random(100) - 90
        galactic = np.vstack((l,b)).transpose()
        computed = context.galactic_to_equatorial(galactic)
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "galactic_to_equatorial_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "galactic_to_equatorial_ref.p","rb"))
        self.assertTrue(np.allclose(computed,reference,\
            rtol=self.rtol,atol=self.rtol))
    def test_equatorial_to_galactic(self):
        np.random.seed(1000)
        ra = 360*np.random.random(100)
        dec = 180*np.random.random(100) - 90
        equatorial = np.vstack((ra,dec)).transpose()
        computed = context.equatorial_to_galactic(equatorial)
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "galactic_to_equatorial_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "galactic_to_equatorial_ref.p","rb"))
        self.assertTrue(np.allclose(computed,reference,\
            rtol=self.rtol,atol=self.rtol))
    def test_sgl_gal_matrix(self):
        computed = context.sgl_gal_matrix(\
            137.37*np.pi/180.0,47.37*np.pi/180.0,0.0,bz = 6.32*np.pi/180.0)
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "sgl_gal_matrix_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "sgl_gal_matrix_ref.p","rb"))
        self.assertTrue(np.allclose(computed,reference,\
            rtol=self.rtol,atol=self.rtol))
    def test_galactic_to_supergalactic(self):
        np.random.seed(1000)
        l = 360*np.random.random(100)
        b = 180*np.random.random(100) - 90
        galactic = np.vstack((l,b)).transpose()
        computed = context.galactic_to_supergalactic(galactic)
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "galactic_to_supergalactic_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "galactic_to_supergalactic_ref.p","rb"))
        self.assertTrue(np.allclose(computed,reference,\
            rtol=self.rtol,atol=self.rtol))
    def test_supergalactic_to_galactic(self):
        np.random.seed(1000)
        l = 360*np.random.random(100)
        b = 180*np.random.random(100) - 90
        galactic = np.vstack((l,b)).transpose()
        computed = context.supergalactic_to_galactic(galactic)
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "supergalactic_to_galactic_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "supergalactic_to_galactic_ref.p","rb"))
        self.assertTrue(np.allclose(computed,reference,\
            rtol=self.rtol,atol=self.rtol))
    def test_gal2SG(self):
        np.random.seed(1000)
        l = 360*np.random.random(100)
        b = 180*np.random.random(100) - 90
        galactic = np.vstack((l,b)).transpose()
        computed = context.gal2SG(galactic)
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "gal2SG_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "gal2SG_ref.p","rb"))
        self.assertTrue(np.allclose(computed,reference,\
            rtol=self.rtol,atol=self.rtol))
    def test_sg2gal(self):
        np.random.seed(1000)
        l = 360*np.random.random(100)
        b = 180*np.random.random(100) - 90
        galactic = np.vstack((l,b)).transpose()
        computed = context.sg2gal(galactic)
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "sg2gal_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "sg2gal_ref.p","rb"))
        self.assertTrue(np.allclose(computed,reference,\
            rtol=self.rtol,atol=self.rtol))
    def test_local_group_z_correction(self):
        np.random.seed(1000)
        l = 287
        b = 56
        zh = 0.045
        computed = context.local_group_z_correction(zh,b,l)
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "local_group_z_correction_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "local_group_z_correction_ref.p","rb"))
        self.assertTrue(np.allclose(computed,reference,\
            rtol=self.rtol,atol=self.rtol))
    def test_supergalactic_ang_to_pos(self):
        np.random.seed(1000)
        l = 360*np.random.random(100)
        b = 180*np.random.random(100) - 90
        r = 100*np.random.random(100)
        ang = np.hstack((r,l,b)).transpose()
        computed = context.supergalactic_ang_to_pos(ang)
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "supergalactic_ang_to_pos_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "supergalactic_ang_to_pos_ref.p","rb"))
        self.assertTrue(np.allclose(computed,reference,\
            rtol=self.rtol,atol=self.rtol))
    def test_row2col(self):
        np.random.seed(1000)
        row = np.random.random(4)
        computed = context.row2col(row)
        self.assertTrue(computed.shape == (1,4))
    def test_mean_distance(self):
        standard = dataFolder + \
            "reference_constrained/sample7000/standard/snapshot_001"
        snapn = pynbody.load(standard)
        boxsize = snapn.properties['boxsize'].ratio("Mpc a h**-1")
        hn = snapn.halos()
        computed = context.mean_distance(hn[1])
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "mean_distance_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "mean_distance_ref.p","rb"))
        self.assertTrue(np.allclose(computed,reference,\
            rtol=self.rtol,atol=self.rtol))
    def test_distance(self):
        np.random.seed(1000)
        pos = np.random.random((100,3))
        computed = context.distance(pos)
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "distance_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "distance_ref.p","rb"))
        self.assertTrue(np.allclose(computed,reference,\
            rtol=self.rtol,atol=self.rtol))
    def test_halo_distances(self):
        standard = dataFolder + \
            "reference_constrained/sample7000/standard/snapshot_001"
        snapn = pynbody.load(standard)
        hn = snapn.halos()
        computed = context.halo_distances(hn)
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "halo_distances_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "halo_distances_ref.p","rb"))
        self.assertTrue(np.allclose(computed,reference,\
            rtol=self.rtol,atol=self.rtol))
    def test_void_distances(self):
        standard = dataFolder + \
            "reference_constrained/sample7000/standard/snapshot_001"
        snapn = pynbody.load(standard)
        reverse = dataFolder + \
            "reference_constrained/sample7000/reverse/snapshot_001"
        snapr = pynbody.load(reverse)
        hr = snapr.halos()
        b = pybody.bridge.Bridge(snapn,snapr)
        computed = context.void_distances(hr,b)
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "void_distances_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "void_distances_ref.p","rb"))
        self.assertTrue(np.allclose(computed,reference,\
            rtol=self.rtol,atol=self.rtol))
    def test_snapunion_positions(self):
        standard = dataFolder + \
            "reference_constrained/sample7000/standard/snapshot_001"
        snapn = pynbody.load(standard)
        hn = snapn.halos()
        to_use = [1,2,3,4]
        computed = context.snapunion_positions(hn,to_use)
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "snapunion_positions_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "snapunion_positions_ref.p","rb"))
        self.assertTrue(np.allclose(computed,reference,\
            rtol=self.rtol,atol=self.rtol))
    def test_select_sphere(self):
        standard = dataFolder + \
            "reference_constrained/sample7000/standard/snapshot_001"
        snapn = pynbody.load(standard)
        boxsize = snapn.properties['boxsize'].ratio("Mpc a h**-1")
        radius = 30
        offset = np.array([boxsize/2]*3)
        distance = 100
        direction = np.array([1.0,1.0,1.0])
        computed = context.select_sphere(snapn,radius,distance,direction,\
            offset=offset)['pos']
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "select_sphere_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "select_sphere_ref.p","rb"))
        self.assertTrue(np.allclose(computed,reference,\
            rtol=self.rtol,atol=self.rtol))
    def test_get_containing_halos(self):
        standard = dataFolder + \
            "reference_constrained/sample7000/standard/snapshot_001"
        snap = pynbody.load(standard)
        halos = snap.halos()
        computed = context.get_containing_halos(snap,halos)
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "get_containing_halos_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "get_containing_halos_ref.p","rb"))
        for k in range(0,len(reference)):
            self.assertTrue(np.all(computed[k] == reference[k]))
    def test_combineHalos(self):
        standard = dataFolder + \
            "reference_constrained/sample7000/standard/snapshot_001"
        snap = pynbody.load(standard)
        halos = snap.halos()
        computed = context.combineHalos(snap,halos)['pos']
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "combineHalos_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "combineHalos_ref.p","rb"))
        self.assertTrue(np.allclose(computed,reference,\
            rtol=self.rtol,atol=self.rtol))
    def test_localGroupTest(self):
        print("WARNING - TEST NOT IMPLEMENTED YET FOR localGroupTest")
        self.assertTrue(True)
    def test_spheresMonteCarlo(self):
        np.random.seed(1000)
        centres = np.random.random(1000,3)*100
        radii = np.random.random(1000)*10
        computed = context.spheresMonteCarlo(centres,radii,[100,100,100],\
            seed=1000)
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "spheresMonteCarlo_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "spheresMonteCarlo_ref.p","rb"))
        self.assertTrue(np.allclose(computed,reference,\
            rtol=self.rtol,atol=self.rtol))
    def test_mapEquatorialSnapshotToGalactic(self):
        standard = dataFolder + \
            "reference_constrained/sample7000/standard/snapshot_001"
        snap = pynbody.load(standard)
        context.mapEquatorialSnapshotToGalactic(snap)
        computed = snap['pos']
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "mapEquatorialSnapshotToGalactic_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "mapEquatorialSnapshotToGalactic_ref.p","rb"))
        self.assertTrue(np.allclose(computed,reference,\
            rtol=self.rtol,atol=self.rtol))
    def test_mapGalacticSnapshotToEquatorial(self):
        standard = dataFolder + \
            "reference_constrained/sample7000/standard/snapshot_001"
        snap = pynbody.load(standard)
        context.mapGalacticSnapshotToEquatorial(snap)
        computed = snap['pos']
        if self.generateTestData:
            pickle.dump(computed,open(self.dataFolder + self.test_subfolder + \
                "mapGalacticSnapshotToEquatorial_ref.p","wb")
        reference = pickle.load(open(self.dataFolder + self.test_subfolder + \
            "mapGalacticSnapshotToEquatorial_ref.p","rb"))
        self.assertTrue(np.allclose(computed,reference,\
            rtol=self.rtol,atol=self.rtol))
    def test_mapEquatorialToGalactic(self):
        np.random.seed(1000)
        points = np.random.random((100,3))
        computed = 









