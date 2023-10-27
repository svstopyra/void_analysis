import unittest

from void_analysis.tools import mcmcFileToWhiteNoise
from void_analysis import antihalos, context, cosmology, halos, plot_utilities
from void_analysis import real_clusters, simulation_tools, snapedit, stacking
from void_analysis import survey, tools, catalogue
from void_analysis import paper_plots_borg_antihalos_generate_data as generator
import numpy as np
import pynbody
import multiprocessing as mp
thread_count = mp.cpu_count()
import scipy
#from void_analysis.postprocessing import process_snapshot
from void_analysis.paper_plots_borg_antihalos_generate_data import *
from void_analysis.simulation_tools import ngPerLBin
from void_analysis.simulation_tools import biasNew, biasOld
from void_analysis.survey import radialCompleteness, surveyMask, keCorr
import pickle
import astropy
import h5py
import astropy
import argparse
import configparser
import os

# If true, the code will generate reference data if missing.
#generateMode = True
# Change this if the test data is kept in a non-standard place:
#dataFolder = "data_for_tests/"

# Process configuration options:
config = configparser.ConfigParser()
if os.path.isfile(os.path.dirname(__file__) + "/config.ini"):
    config.read(os.path.dirname(__file__) + "/config.ini")
    print("Using user-defined config options.")
elif os.path.isfile(os.path.dirname(__file__) + "/config_default.ini"):
    config.read(os.path.dirname(__file__) + "/config_default.ini")
    print("Using default config options...")
else:
    raise Exception("config_default.ini is missing.")

dataFolder = config['TESTING']['DataFolder']
generateMode = config['TESTING']['GenerateMode'] == 'True'

# Base test class:
class test_base(unittest.TestCase):
    def getReference(self,referenceFile,computed,mode="pickle"):
        if self.generateTestData and not os.path.isfile(referenceFile):
            if mode == "pickle":
                with open(referenceFile,"wb") as outfile:
                    pickle.dump(computed,outfile)
            elif mode == "numpy":
                np.save(referenceFile,computed)
            else:
                raise Exception("Mode not recognised.")
        if mode == "pickle":
            with open(referenceFile,"rb") as infile:
                    reference = pickle.load(infile)
        elif mode == "numpy":
            reference = np.load(referenceFile)
        return reference
    def getIsIntType(self,reference):
        return type(reference) == int or type(reference) == np.int64
    def getIsFloatType(self,reference):
        return type(reference) == float or type(reference) == np.float32 or \
            type(reference) == np.float64
    def getIsExactType(self,reference):
        return self.getIsIntType(reference) or type(reference) == bool
    def getIsExactDType(self,dtype):
        return dtype == int or dtype == np.int64 or dtype == bool
    def getIsFloatDType(self,dtype):
        return dtype == float or dtype == np.float64 or dtype == np.float32
    def getIsArrayType(self,reference):
        return type(reference) == np.ndarray or \
            type(reference) == pynbody.array.SimArray
    def getIsListType(self,reference):
        return type(reference) == list or type(reference) == tuple
    def compareArrayTypes(self,computed,reference,filterNan = False):
        self.assertTrue(reference.dtype == computed.dtype)
        if filterNan:
            isFiniteRef = np.isfinite(reference)
            isFiniteCom = np.isfinite(computed)
            self.assertTrue(len(isFiniteRef) == len(isFiniteCom))
            self.assertTrue(np.all(isFiniteRef == isFiniteCom))
            ref = reference[isFiniteRef]
            com = computed[isFiniteCom]
        else:
            ref = reference
            com = computed
        if self.getIsExactDType(reference.dtype):
            self.assertTrue(np.all(ref == com))
        elif self.getIsFloatDType(reference.dtype):
            self.assertTrue(np.allclose(com,ref,\
                rtol=self.rtol,atol=self.atol))
    def compareToReference(self,computed,reference,filterNan = False):
        self.assertTrue(type(reference) == type(computed))
        if self.getIsListType(reference):
            nrefs = len(reference)
            self.assertTrue(len(reference) == len(computed))
            for k in range(0,nrefs):
                self.compareToReference(computed[k],reference[k],\
                    filterNan=filterNan)
        elif self.getIsArrayType(reference):
            self.compareArrayTypes(computed,reference,filterNan = filterNan)
        elif self.getIsExactType(reference):
            self.assertTrue(reference == computed)
        elif self.getIsFloatType(reference):
            self.assertTrue(np.allclose(computed,\
                reference,rtol=self.rtol,atol=self.atol))
        elif computed is None:
            self.assertTrue(reference is None)
        elif (type(computed) == bool):
            self.assertTrue(type(reference) == bool)
            self.assertTrue(computed == reference)
        else:
            print("Warning! Comparison for type " + str(type(reference)) + \
                " is not yet defined. Assuming test passes.")

# Tests for functions required in IC generation:
@unittest.skip("Tests in development")
class test_ICgen(test_base):
    def setUp(self):
        self.dataFolder=dataFolder
        self.test_subfolder = "function_tests_ICgen/"
        self.rtol=1e-5
        self.atol=1e-8
        self.generateTestData = generateMode
    # Test the extraction and conversion of mcmc file white noise to a
    # format that genetIC understands
    def test_wn(self):
        print("Running white noise extraction test.")
        tools.mcmcFileToWhiteNoise(self.dataFolder + "mcmc_2791.h5",\
            "wn.npy",\
            normalise = True,fromInverseFourier=False,flip=False,reverse=True)
        computed = np.load("wn.npy")
        referenceFile = self.dataFolder + "wn_test.npy"
        reference = self.getReference(referenceFile,computed,mode="numpy")
        self.assertTrue(np.allclose(computed,reference,\
                rtol=self.rtol,atol=self.atol))

# Tests for functions required for computing anti-halo/halo properties
@unittest.skip("Tests in development")
class test_ahproperties(test_base):
    def setUp(self):
        self.dataFolder=dataFolder
        self.test_subfolder = "function_tests_ahproperties/"
        self.rtol=1e-5
        self.atol=1e-8
        self.generateTestData = generateMode
    # Test conversion of volumes to the correct units:
    def test_volumes(self):
        print("Running volumes test.")
        standard = self.dataFolder + \
            "reference_constrained/sample2791/gadget_full_forward/snapshot_001"
        snapn = pynbody.load(standard)
        computed = tools.zobovVolumesToPhysical(standard + ".vols",snapn,\
            dtype=np.double,offset=4)
        referenceFile = self.dataFolder + \
            "reference_constrained/volumes_ref.npy"
        reference = self.getReference(referenceFile,computed,mode="numpy")
        self.compareToReference(computed,reference)
    # Test computation of anti-halo centres:
    def test_centres_and_radii(self):
        print("Running centres and radii test.")
        standard = self.dataFolder + \
            "reference_constrained/sample2791/gadget_full_forward/snapshot_001"
        reverse = self.dataFolder + \
            "reference_constrained/sample2791/gadget_full_reverse/snapshot_001"
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
        volumes = np.load(self.dataFolder + \
            "reference_constrained/volumes_ref.npy")
        boxsize = snapn.properties['boxsize'].ratio("Mpc a h**-1")
        periodicity = [boxsize]*3
        antiHaloCentres = np.zeros((10,3))
        antiHaloVolumes = np.zeros((10))
        for k in range(0,10):
            antiHaloCentres[k,:] = context.computePeriodicCentreWeighted(\
                snapn['pos'][sortedn[hr[k+1]['iord']],:],\
                volumes[sortedn[hr[k+1]['iord']]],periodicity,accelerate=True)
            antiHaloVolumes[k] = np.sum(volumes[sortedn[hr[k+1]['iord']]])
        antiHaloRadii = np.cbrt(3*antiHaloVolumes/(4*np.pi))
        # Test the centre computation for the first anti-halo:
        referenceFileCentres = self.dataFolder + \
            "reference_constrained/ref_centres.npy"
        referenceFileRadii = self.dataFolder + \
            "reference_constrained/ref_radii.npy"
        if self.generateTestData:
            if not os.path.isfile(referenceFileCentres):
                np.save(referenceFileCentres,antiHaloCentres)
            if not os.path.isfile(referenceFileRadii):
                np.save(referenceFileRadii,antiHaloRadii)
        refCentres = np.load(referenceFileCentres)
        refRadii = np.load(referenceFileRadii)
        computed = [antiHaloCentres,antiHaloRadii]
        reference = [refCentres,refRadii]
        # Test computation of the first few anti-halo centres:
        self.compareToReference(computed,reference)
    # Test pair counts calculation:
    def test_pair_counts(self):
        print("Running pair counts test.")
        thread_count = mp.cpu_count()
        refCentres = np.load(self.dataFolder + \
            "reference_constrained/ref_centres.npy")
        refRadii = np.load(self.dataFolder + \
            "reference_constrained/ref_radii.npy")
        volumes = np.load(self.dataFolder + \
            "reference_constrained/volumes_ref.npy")
        standard = self.dataFolder + \
            "reference_constrained/sample2791/gadget_full_forward/snapshot_001"
        snapn = pynbody.load(standard)
        boxsize = snapn.properties['boxsize'].ratio("Mpc a h**-1")
        tree = scipy.spatial.cKDTree(snapn['pos'],boxsize=boxsize)
        nBins = 31
        rBinStack = np.linspace(0,3.0,nBins)
        [pairCounts,volumesList] = stacking.getPairCounts(refCentres,refRadii,\
            snapn,rBinStack,nThreads=thread_count,tree=tree,\
            method="poisson",vorVolumes=volumes)
        referenceFilePairCounts = self.dataFolder + \
            "reference_constrained/pair_counts_ref.npy"
        referenceFileVolumesList = self.dataFolder + \
            "reference_constrained/volumes_list_ref.npy"
        if self.generateTestData:
            if (not os.path.isfile(referenceFilePairCounts)):
                np.save(referenceFilePairCounts,pairCounts)
            if (not os.path.isfile(referenceFileVolumesList)):
                np.save(referenceFileVolumesList,volumesList)
        pairCountsRef = np.load(referenceFilePairCounts)
        volumesListRef = np.load(referenceFileVolumesList)
        self.assertTrue(\
            np.allclose(volumesList,volumesListRef,\
                rtol=self.rtol,atol=self.atol))
        self.assertTrue(\
            np.allclose(pairCounts,pairCountsRef,\
                rtol=self.rtol,atol=self.atol))
    # Test central density calculation:
    def test_central_density(self):
        print("Running central density test.")
        thread_count = mp.cpu_count()
        standard = self.dataFolder + \
            "reference_constrained/sample2791/gadget_full_forward/snapshot_001"
        snapn = pynbody.load(standard)
        boxsize = snapn.properties['boxsize'].ratio("Mpc a h**-1")
        tree = scipy.spatial.cKDTree(snapn['pos'],boxsize=boxsize)
        refCentres = np.load(self.dataFolder + \
            "reference_constrained/ref_centres.npy")
        refRadii = np.load(self.dataFolder + \
            "reference_constrained/ref_radii.npy")
        volumes = np.load(self.dataFolder + \
            "reference_constrained/volumes_ref.npy")
        computed = np.zeros((10,3))
        rhoBar = np.sum(snapn['mass'])/(boxsize**3)
        for k in range(0,10):
            computed[k] = stacking.centralDensity(\
                refCentres[k,:],refRadii[k],snapn['pos'],volumes,\
                snapn['mass'],tree=tree,centralRatio = 2,\
                nThreads=thread_count)/rhoBar - 1.0
        referenceFile = self.dataFolder + \
            "reference_constrained/ref_central_density.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    # Test the pipeline end to end:
    def test_properties_pipeline(self):
        print("Running properties pipeline test.")
        standard = self.dataFolder + \
            "reference_constrained/sample2791/gadget_full_forward/snapshot_001"
        reverse = self.dataFolder + \
            "reference_constrained/sample2791/gadget_full_reverse/snapshot_001"
        # Run the pipeline:
        simulation_tools.processSnapshot(standard,reverse,31,offset=4,\
            output=self.dataFolder + \
            "test_ahproperties_pipeline/ahproperties.p")
        # Compare the results:
        with open(self.dataFolder + \
            "test_ahproperties_pipeline/ahproperties.p","rb") as infile:
            computed = pickle.load(infile)
        referenceFile = self.dataFolder + \
            "test_ahproperties_pipeline/ahpropertiesRef.p"
        reference = tools.loadPickle(standard + ".AHproperties.p")
        for k in range(0,len(reference)):
            if np.any(np.isnan(reference[k])):
                finite = np.isfinite(reference[k])
                self.compareToReference(\
                    computed[k][finite],reference[k][finite])
            else:
                self.compareToReference(computed[k],reference[k])

@unittest.skip("Tests in development")
# Testing the PPT code:
class test_ppts(test_base):
    def setUp(self):
        self.dataFolder=dataFolder
        self.test_subfolder = "function_tests_ppts/"
        self.rtol=1e-5
        self.atol=1e-8
        self.generateTestData = generateMode
    # PPT pipeline:
    def test_ppt_pipeline(self):
        print("Running ppt pipeline test (ignore resolution warnings " + \
            "- test done at 64^3 for speed)")
        with open(self.dataFolder + "hpIndices_ref.p","rb") as infile:
            hpIndices = pickle.load(infile)
        [galaxyNumberCountExp,galaxyNumberCountsRobust] = getPPTPlotData(\
            hpIndices=hpIndices,snapNumList = [2791,3250],N=64,\
            samplesFolder = self.dataFolder + "reference_constrained/",\
            tmppFile=self.dataFolder + "2mpp_data/2MPP.txt",\
            catFolder=self.dataFolder,surveyMaskPath = self.dataFolder + \
            "2mpp_data/",recomputeData = True,\
            snapname = "/gadget_full_forward/snapshot_001",verbose=False)
        computed = [galaxyNumberCountExp,galaxyNumberCountsRobust]
        referenceFile = self.dataFolder + "ppt_test_data.p"
        [galaxyNumberCountExpRef,galaxyNumberCountsRobustRef] = \
            self.getReference(referenceFile,computed)
        self.assertTrue(\
            np.allclose(galaxyNumberCountExp,galaxyNumberCountExpRef,\
            rtol=self.rtol,atol=self.atol))
        self.assertTrue(\
            np.allclose(galaxyNumberCountsRobust,galaxyNumberCountsRobustRef,\
            rtol=self.rtol,atol=self.atol))
    # Survey Mask calculation:
    def test_survey_mask(self):
        print("Running survey mask test...")
        surveyMask11 = healpy.read_map(self.dataFolder + \
            "2mpp_data/completeness_11_5.fits")
        surveyMask12 = healpy.read_map(self.dataFolder + \
            "2mpp_data/completeness_12_5.fits")
        # Computation of the survey mask:
        N = 64 # Use a smaller one to reduce the storage space.
        Om0 = 0.3111
        Ode0 = 0.6889
        boxsize = 677.7
        h=0.6766
        mmin = 0.0
        mmax = 12.5
        grid = snapedit.gridListPermutation(N,perm=(2,1,0))
        centroids = grid*boxsize/N + boxsize/(2*N)
        positions = snapedit.unwrap(centroids - np.array([boxsize/2]*3),\
            boxsize)
        cosmo = astropy.cosmology.LambdaCDM(100*h,Om0,Ode0)
        computedMaskList = surveyMask(\
                positions,surveyMask11,surveyMask12,cosmo,-0.94,\
                -23.28,keCorr = keCorr,mmin=mmin,numericalIntegration=True,\
                mmax=mmax,splitApparent=True,splitAbsolute=True,\
                returnComponents=True)
        # Reference data:
        referenceFile = self.dataFolder + "survey_mask_ref.p"
        referenceMaskList = self.getReference(referenceFile,computedMaskList)
        self.compareToReference(computedMaskList,referenceMaskList)
    # test of the galaxy count calculation:
    def test_galaxy_counts(self):
        print("Running galaxy counts test...")
        nMagBins = 16
        N = 64
        biasData = h5py.File(self.dataFolder + "mcmc_2791.h5",'r')
        biasParam = np.array([[biasData['scalars']['galaxy_bias_' + \
            str(k)][()] for k in range(0,nMagBins)]])
        with open(self.dataFolder + "survey_mask_ref.p","rb") as infile:
            [mask,angularMask,radialMas,mask12,mask11] = pickle.load(infile)
        mcmcDenLin_r = np.reshape(np.reshape(\
            1.0 + biasData['scalars']['BORG_final_density'][()],256**3),\
            (256,256,256),order='F')
        # degrade this, so that we can make the test smaller:
        mcmcDenLin_r64 = np.reshape(tools.downsample(mcmcDenLin_r,int(256/N)),\
            (N,N,N))
        computed = simulation_tools.ngPerLBin(\
            biasParam,return_samples=True,mask=mask,\
            accelerate=True,N=N,\
            delta = [mcmcDenLin_r64],contrast=False,sampleList=[0],\
            beta=biasParam[:,:,1],rhog = biasParam[:,:,3],\
            epsg=biasParam[:,:,2],\
            nmean=biasParam[:,:,0],biasModel = biasNew)
        referenceFile = self.dataFolder + "ngCounts_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    # Test mapping between galaxy counts in voxels and healpix patches:
    def test_healpix_mapping(self):
        print("Running healpix mapping test...")
        nMagBins = 16
        N = 64
        with open(self.dataFolder + "ngCounts_ref.p","rb") as infile:
            ngMCMC = np.reshape(pickle.load(infile),(nMagBins,N**3))
        with open(self.dataFolder + "hpIndices_ref.p","rb") as infile:
            hpIndices = pickle.load(infile)
        computed = tools.getCountsInHealpixSlices(ngMCMC,hpIndices,4,nres=N)
        referenceFile = self.dataFolder + "hpCounts_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    # Test getting cluster centres:
    def test_cluster_centre_calculation(self):
        print("Running cluster centre calculation test...")
        # Load cluster data:
        [combinedAbellN,combinedAbellPos,abell_nums] = \
            real_clusters.getCombinedAbellCatalogue(catFolder=self.dataFolder)
        clusterInd = [np.where(combinedAbellN == n)[0] for n in abell_nums]
        clusterLoc = np.zeros((len(clusterInd),3))
        for k in range(0,len(clusterInd)):
            if len(clusterInd[k]) == 1:
                clusterLoc[k,:] = combinedAbellPos[clusterInd[k][0],:]
            else:
                # Average positions:
                clusterLoc[k,:] = np.mean(combinedAbellPos[clusterInd[k],:],0)
        # Snapshot path:
        standard = self.dataFolder + \
            "reference_constrained/sample2791/gadget_full_forward/snapshot_001"
        snap = pynbody.load(standard)
        boxsize = snap.properties['boxsize'].ratio("Mpc a h**-1")
        # Density field data:
        biasData = h5py.File(self.dataFolder + "mcmc_2791.h5",'r')
        N = 256
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
        computed = simulation_tools.getClusterCentres(clusterLoc,\
            snapPath = standard,fileSuffix = "clusters1",\
            recompute=True,density=np.reshape(mcmcDenLin_r,N**3),\
            boxsize=boxsize,positions=positions,positionTree=tree,\
            method="density",reductions=4,\
            iterations=20)
        referenceFile = self.dataFolder + "cluster_centre_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_getAllNgsToHealpix(self):
        # Get data:
        hpIndices = tools.loadPickle(self.dataFolder + "hpIndices_ref.p")
        ngList = tools.loadPickle(self.dataFolder + \
            "function_tests_simulation_tools/" + "ngPerLBin_ref.p")
        computed = generator.getAllNgsToHealpix(ngList,hpIndices,[2791],\
            self.dataFolder + "reference_constrained/",4,recomputeData=False,\
            nres=64)
        referenceFile = self.dataFolder + "getAllNgsToHealpix_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_bootstrapGalaxyCounts(self):
        np.random.seed(1000)
        voxels = np.random.randint(100,size=100)
        counts = np.random.randint(100,size=(16,100))
        computed = generator.bootstrapGalaxyCounts(counts,voxels,1000,\
            randomSeed=1000)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "bootstrapGalaxyCounts_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_getAalpha(self):
        # Get data:
        [mask,angularMask,radialMas,mask12,mask11] = tools.loadPickle(\
            self.dataFolder + "function_tests_survey/" + "surveyMask_ref.p")
        hpIndices = tools.loadPickle(self.dataFolder + "hpIndices_ref.p")
        ngHPMCMC = tools.loadPickle(self.dataFolder + \
            "getAllNgsToHealpix_ref.p")
        nside = 4
        npixels = 12*(nside**2)
        # Perform test:
        computed = generator.getAalpha(mask,hpIndices,ngHPMCMC,1,nMagBins,\
            npixels,10)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "getAalpha_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_getVoxelSums(self):
        # Get data:
        N = 64
        hpIndices = tools.loadPickle(self.dataFolder + "hpIndices_ref.p")
        hpIndicesLinear = hpIndices.reshape(N**3)
        with open(self.dataFolder + "ngCounts_ref.p","rb") as infile:
            ngMCMC = np.reshape(pickle.load(infile),(nMagBins,N**3))
        ngHP = tools.loadPickle(self.dataFolder + "hpCounts_ref.p")
        [Aalpha,inverseLambdaTot] = tools.loadPickle(self.dataFolder + \
            self.test_subfolder + "getAalpha_ref.p")
        # Random selection of test voxels:
        voxels = np.arange(85899 - 10,85899 + 10)
        computed = generator.getVoxelSums(\
            voxels,hpIndicesLinear,Aalpha,ngMCMC,\
            inverseLambdaTot,ngHP,"bootstrap",1000,1,\
            16,bootstrapInterval=[2.5,97.5])
        referenceFile = self.dataFolder + self.test_subfolder + \
            "getVoxelSums_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_getPPTForPoints(self):
        points = np.array([[0,0,0],[50,50,50]])
        with open(self.dataFolder + "hpIndices_ref.p","rb") as infile:
            hpIndices = pickle.load(infile)
        computed = generator.getPPTForPoints(points,hpIndices=hpIndices,\
            snapNumList = [2791,3250],N=64,\
            samplesFolder = self.dataFolder + "reference_constrained/",\
            tmppFile=self.dataFolder + "2mpp_data/2MPP.txt",\
            catFolder=self.dataFolder,surveyMaskPath = self.dataFolder + \
            "2mpp_data/",recomputeData = True,\
            snapname = "/gadget_full_forward/snapshot_001",verbose=False)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "getPPTForPoints_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)

# Test HMF code:
@unittest.skip("Tests in development")
class test_hmfs(test_base):
    # Test computation of the average HMF data:
    def setUp(self):
        self.dataFolder=dataFolder
        self.test_subfolder = "function_tests_hmfs/"
        self.rtol=1e-5
        self.atol=1e-8
        self.generateTestData = generateMode
    def test_constrained_hmfs(self):
        print("Running constrained hmfs test...")
        snapNumList = [2791,3250]
        snapname = "gadget_full_forward/snapshot_001"
        snapnameRev = "gadget_full_reverse/snapshot_001"
        samplesFolder = self.dataFolder + "reference_constrained/"
        computedHMFsData = generator.getHMFAMFDataFromSnapshots(\
            snapNumList,snapname,snapnameRev,samplesFolder,\
            recomputeData=True,reCentreSnap=True,verbose=False)
        referenceFile = samplesFolder + "constrained_hmfs_ref.p"
        referenceHMFsData = self.getReference(referenceFile,computedHMFsData)
        self.compareToReference(computedHMFsData,referenceHMFsData)
    # Test computation of the average unconstrained HMF data:
    def test_unconstrained_hmfs(self):
        print("Running unconstrained hmfs test...")
        snapNumList = [2791,3250]
        snapname = "gadget_full_forward/snapshot_001"
        snapnameRev = "gadget_full_reverse/snapshot_001"
        samplesFolder = self.dataFolder + "reference_constrained/"
        with open(samplesFolder + "constrained_hmfs_ref.p","rb") as infile:
            [constrainedHaloMasses512,constrainedAntihaloMasses512,\
            deltaListMean,deltaListError] = pickle.load(infile)
        computedHMFsData = generator.getUnconstrainedHMFAMFData(\
            snapNumList,snapname,snapnameRev,samplesFolder,\
            deltaListMean,deltaListError,\
            recomputeData=True,reCentreSnaps=True,randomSeed=1000,\
            verbose=False)
        referenceFile = samplesFolder + "unconstrained_hmfs_ref.p"
        referenceHMFsData = self.getReference(referenceFile,computedHMFsData)
        self.compareToReference(computedHMFsData,referenceHMFsData)

@unittest.skip("Tests in development")
class test_void_profiles(test_base):
    # Test the procedure for stacking anti-halos from different simulations:
    def setUp(self):
        self.dataFolder=dataFolder
        self.test_subfolder = "function_tests_void_profiles/"
        self.rtol=1e-5
        self.atol=1e-8
        self.generateTestData = generateMode
        # Some things shared by the catalogue tests:
        self.snapNumList = [2791,3250,5511]
        self.samplesFolder = dataFolder + "reference_constrained/"
        self.snapname = "gadget_full_forward/snapshot_001"
        self.snapList =  [pynbody.load(self.samplesFolder + "sample" + \
            str(snapNum) + "/" + self.snapname) for snapNum in self.snapNumList]
        self.ahPropsConstrained = [tools.loadPickle(snap.filename + \
            ".AHproperties.p") \
            for snap in self.snapList]
        self.antihaloRadii = [props[7] for props in self.ahPropsConstrained]
        self.antihaloMassesList = [props[3] \
            for props in self.ahPropsConstrained]
        self.ahCentresList = [props[5] \
            for props in self.ahPropsConstrained]
        self.vorVols = [props[4] for props in self.ahPropsConstrained]
    def test_void_stacking(self):
        print("Running stacking test.")
        snapNumList = [2791,3250]
        rMin=5
        rMax=25
        mMin = 1e14
        mMax = 1e15
        snapname = "gadget_full_forward/snapshot_001"
        snapnameRev = "gadget_full_reverse/snapshot_001"
        samplesFolder = self.dataFolder + "reference_constrained/"
        # Load snapshots:
        snapList =  [pynbody.load(samplesFolder + "sample" + \
            str(snapNum) + "/" + snapname) for snapNum in snapNumList]
        snapListRev =  [pynbody.load(samplesFolder + "sample" + \
            str(snapNum) + "/" + snapnameRev) for snapNum in snapNumList]
        boxsize = snapList[0].properties['boxsize'].ratio("Mpc a h**-1")
        # Load reference antihalo data:
        ahProps = [tools.loadPickle(snap.filename + ".AHproperties.p") \
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
        computed = stacking.stackVoidsWithFilter(\
            np.vstack(ahCentresList),stackedRadii,\
            np.where((stackedRadii > rMin) & (stackedRadii < rMax) & \
            stackedConditions & (stackedMasses > mMin) & \
            (stackedMasses <= mMax))[0],snapList[0],rBins,\
            nPairsList = np.vstack(pairCountsList),\
            volumesList = np.vstack(volumesList),\
            method="poisson",errorType="Weighted")
        # Reference stacks:
        referenceFile = self.dataFolder + "stacks_reference.p"
        reference = self.getReference(referenceFile,computed)
        # Compare to reference:
        self.compareToReference(computed,reference)
    # Test the procedure for averaging different stacks:
    def test_stack_averaging(self):
        print("Running averaging stacks test.")
        snapNumList = [2791,3250]
        rMin=5
        rMax=25
        mMin = 1e14
        mMax = 1e15
        snapname = "gadget_full_forward/snapshot_001"
        snapnameRev = "gadget_full_reverse/snapshot_001"
        samplesFolder = self.dataFolder + "reference_constrained/"
        # Load snapshots:
        snapList =  [pynbody.load(samplesFolder + "sample" + \
            str(snapNum) + "/" + snapname) for snapNum in snapNumList]
        snapListRev =  [pynbody.load(samplesFolder + "sample" + \
            str(snapNum) + "/" + snapnameRev) for snapNum in snapNumList]
        N = 512
        boxsize = snapList[0].properties['boxsize'].ratio("Mpc a h**-1")
        nbar = (N/boxsize)**3
        # Load reference antihalo data:
        ahProps = [tools.loadPickle(snap.filename + ".AHproperties.p") \
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
        computed = stacking.computeMeanStacks(\
            ahCentresList,antihaloRadii,antihaloMassesList,conditionList,\
            pairCountsList,volumesList,snapList,nbar,rBins,rMin,rMax,mMin,mMax)
        # Load reference:
        referenceFile = self.dataFolder + "stack_average_reference.p"
        reference = self.getReference(referenceFile,computed)
        # Compare to reference:
        self.compareToReference(computed,reference)
    # End-to-end test of the void profiles pipeline:
    def test_whole_stacking_pipeline(self):
        print("Running stacking pipeline end-to-end test.")
        snapname = "gadget_full_forward/snapshot_001"
        snapnameRev = "gadget_full_reverse/snapshot_001"
        computed = generator.getVoidProfilesData(\
            [2791, 3250],[2791,3250],\
            unconstrainedFolder = self.dataFolder + "reference_constrained/",\
            samplesFolder = self.dataFolder + "reference_constrained/",\
            snapname=snapname,snapnameRev = snapnameRev)
        referenceFile = self.dataFolder + "void_profiles_pipeline_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_getPartialPairCountsAndVols(self):
        snapNameList = [self.samplesFolder + self.snapname + "sample" + \
            str(snapNum) for snapNum in self.snapNumList]
        filterListToApply = [np.range(0,20) for snapNum in self.snapNumList]
        rEffMax=3.0
        rEffMin=0.0
        nBins=31
        rBins = np.linspace(rEffMin,rEffMax,nBins)
        computed = generator.getPartialPairCountsAndVols(self.snapNameList,\
            self.antihaloRadii,self.antihaloMassesList,\
            self.ahCentresList,self.vorVols,rBins,"poisson",\
            5,25,1e14,1e15,677.7,filterListToApply=filterListToApply)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "getPartialPairCountsAndVols_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_getCentreListUnconstrained(self):
        [computed,tree] = generator.getCentreListUnconstrained(self.snapList,
            randomSeed = 1000,numDenSamples = 1000,rSphere = 135,\
            densityRange = [-0.051,-0.049])
        referenceFile = self.dataFolder + self.test_subfolder + \
            "getCentreListUnconstrained_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_getGridPositionsAndTreeForSims(self):
        computed = getGridPositionsAndTreeForSims(677.7,Nden=256,perm=(2,1,0))
        referenceFile = self.dataFolder + self.test_subfolder + \
            "getGridPositionsAndTreeForSims_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)


# Tests for the tools package:
@unittest.skip("Tests in development")
class test_tools(test_base):
    def setUp(self):
        self.dataFolder=dataFolder
        self.test_subfolder = "function_tests_tools/"
        self.rtol=1e-5
        self.atol=1e-8
        self.generateTestData = generateMode
    def test_getAntiHalosInSphere(self):
        print("Running tools.getAntiHalosInSphere test...")
        # Fixed centres to test:
        np.random.seed(1000)
        centreList = np.random.random((100,3))*200
        computed = tools.getAntiHalosInSphere(centreList,100,\
            origin=np.array([50,50,50]))
        referenceFile = self.dataFolder + self.test_subfolder + \
                "getAntiHalosInSphere_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_getCentredDensityConstrast(self):
        print("Running tools.getCentredDensityConstrast test...")
        snapname = self.dataFolder + \
            "reference_constrained/sample2791/gadget_full_forward/snapshot_001"
        snap = pynbody.load(snapname)
        boxsize = snap.properties['boxsize'].ratio("Mpc a h**-1")
        np.random.seed(1000)
        centreList = np.random.random((100,3))*boxsize
        radius = 135
        computed = tools.getCentredDensityConstrast(snap,centreList,radius)
        referenceFile = self.dataFolder + self.test_subfolder + \
                "getCentredDensityConstrast_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_getHaloAndAntihaloCountsInDensityRange(self):
        print("Running tools.getHaloAndAntihaloCountsInDensityRange test...")
        snapname = self.dataFolder + \
            "reference_constrained/sample2791/gadget_full_forward/snapshot_001"
        snap = pynbody.load(snapname)
        boxsize = snap.properties['boxsize'].ratio("Mpc a h**-1")
        np.random.seed(1000)
        centreList = np.random.random((100,3))*boxsize
        radius = 135
        # Load the pre-computed (correct) density list:
        deltaList = tools.loadPickle(self.dataFolder + self.test_subfolder + \
            "getCentredDensityConstrast_ref.p")
        # Other parameters:
        mThresh = 1e13
        ahProps = tools.loadPickle(snap.filename + ".AHproperties.p")
        hncentres = ahProps[0]
        hrcentres = ahProps[2]
        hnmasses = ahProps[1]
        hrmasses = ahProps[3]
        deltaCentral = ahProps[11]
        computed = tools.getHaloAndAntihaloCountsInDensityRange(\
            radius,snap,centreList,deltaList,mThresh,hncentres,hrcentres,\
            hnmasses,hrmasses,deltaCentral,deltaLow=-0.07,deltaHigh=-0.06,\
            workers=-1)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "getHaloAndAntihaloCountsInDensityRange_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_getEquivalents(self):
        print("Running tools.getEquivalents test...")
        samplesFolder = self.dataFolder + "reference_constrained/"
        snapNumList = [2791,3250]
        snapname = "gadget_full_forward/snapshot_001"
        snapList =  [pynbody.load(samplesFolder + "sample" + \
            str(snapNum) + "/" + snapname) for snapNum in snapNumList]
        halos = [snap.halos() for snap in snapList]
        ahPropsList = [tools.loadPickle(snap.filename + ".AHproperties.p") \
            for snap in snapList]
        centresList = [ahProps[0] for ahProps in ahPropsList]
        boxsize = snapList[0].properties['boxsize'].ratio("Mpc a h**-1")
        rSearch = 20
        computed = tools.getEquivalents(halos[0],halos[1],centresList[0],\
            centresList[1],boxsize,rSearch)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "getEquivalents_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_loadAbellCatalogue(self):
        print("Running tools.loadAbellCatalogue test...")
        computed = tools.loadAbellCatalogue(self.dataFolder + "VII_110A/")
        referenceFile = self.dataFolder + self.test_subfolder + \
            "loadAbellCatalogue_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_getPoissonAndErrors(self):
        print("Running tools.getPoissonAndErrors test...")
        np.random.seed(1000)
        bins = np.linspace(0,10,21)
        binCentres = (bins[1:] + bins[0:-1])/2
        counts = 100/binCentres + np.random.randint(len(binCentres),\
            size=binCentres.shape)
        computed = tools.getPoissonAndErrors(bins,counts,alpha=0.32)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "getPoissonAndErrors_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_remapBORGSimulation(self):
        print("Running tools.remapBORGSimulation test...")
        snapname = self.dataFolder + \
            "reference_constrained/sample2791/gadget_full_forward/snapshot_001"
        snap = pynbody.load(snapname)
        tools.remapBORGSimulation(snap,swapXZ=False,reverse=True)
        computed = snap['pos']
        referenceFile = self.dataFolder + self.test_subfolder + \
            "remapBORGSimulation_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_remapAntiHaloCentre(self):
        print("Running tools.remapAntiHaloCentre test...")
        snapname = self.dataFolder + \
            "reference_constrained/sample2791/gadget_full_forward/snapshot_001"
        ahProps = tools.loadPickle(snapname + ".AHproperties.p")
        boxsize = 677.7
        hrcentres = ahProps[5]
        computed = tools.remapAntiHaloCentre(hrcentres,boxsize)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "remapAntiHaloCentre_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_zobovVolumesToPhysical(self):
        print("Running tools.zobovVolumesToPhysical test...")
        standard = self.dataFolder + \
            "reference_constrained/sample2791/gadget_full_forward/snapshot_001"
        snapn = pynbody.load(standard)
        computed = tools.zobovVolumesToPhysical(standard + ".vols",snapn,\
            dtype=np.double,offset=4)
        referenceFile = self.dataFolder + \
            "reference_constrained/volumes_ref.npy"
        reference = self.getReference(referenceFile,computed,mode="numpy")
        self.compareToReference(computed,reference)
    def test_getHaloCentresAndMassesFromCatalogue(self):
        print("Running tools.getHaloCentresAndMassesFromCatalogue test...")
        snapname = self.dataFolder + \
            "reference_constrained/sample2791/gadget_full_forward/snapshot_001"
        ahProps = tools.loadPickle(snapname + ".AHproperties.p")
        snap = pynbody.load(snapname)
        hcentresRef = ahProps[0]
        hmassesRef = ahProps[1]
        halos = snap.halos()
        computed = tools.getHaloCentresAndMassesFromCatalogue(halos,\
            inMPcs=True)
        reference = [hcentresRef,hmassesRef]
        self.compareToReference(computed,reference)
    def test_getHaloMassesAndVirials(self):
        print("Running tools.getHaloMassesAndVirials test...")
        snapname = self.dataFolder + \
            "reference_constrained/sample2791/gadget_full_forward/snapshot_001"
        ahProps = tools.loadPickle(snapname + ".AHproperties.p")
        snap = pynbody.load(snapname)
        hcentres = ahProps[0][0:10,:]
        computed = tools.getHaloMassesAndVirials(snap,hcentres,overden=200,\
            rho_def="critical",massUnit="Msol h**-1",distanceUnit="Mpc a h**-1")
        referenceFile = self.dataFolder + self.test_subfolder + \
            "getHaloMassesAndVirials_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_mcmcFileToWhiteNoise(self):
        print("Running tools.mcmcFileToWhiteNoise test...")
        tools.mcmcFileToWhiteNoise(self.dataFolder + "mcmc_2791.h5",\
            "wn.npy",\
            normalise = True,fromInverseFourier=False,flip=False,reverse=True)
        computed = np.load("wn.npy")
        referenceFile = self.dataFolder + "wn.npy"
        reference = self.getReference(referenceFile,computed,mode="numpy")
        self.compareToReference(computed,reference)
    def test_minmax(self):
        print("Running tools.minmax test...")
        testArray = np.array([1.0,3.0,2.5])
        minmax = tools.minmax(testArray)
        minmaxRef = np.array([1.0,3.0])
        self.assertTrue(np.allclose(minmax,minmaxRef,\
            rtol=self.rtol,atol=self.atol))
    def test_reorderLinearScalar(self):
        print("Running tools.reorderLinearScalar test...")
        vols = np.load(self.dataFolder + \
            "reference_constrained/volumes_ref.npy")
        computed = tools.reorderLinearScalar(vols,256)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "reorderLinearScalar_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_getKDTree(self):
        print("Running tools.getKDTree test...")
        snap = pynbody.load(self.dataFolder + "kdtree_test/test_ic.gadget2")
        boxsize = snap.properties['boxsize'].ratio("Mpc a h**-1")
        tree = tools.getKDTree(snap,reconstructTree=True)
        # Can't really compare the trees themselves so easily, so test instead
        # that we get the same thing with a query:
        np.random.seed(1000)
        randCentres = np.random.random((100,3))*boxsize
        radius = 20
        computed = np.array(\
            tree.query_ball_point(randCentres,radius,return_length=True))
        referenceFile = self.dataFolder + self.test_subfolder + \
                "kdtree_test/tree.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_getCountsInHealpixSlices(self):
        print("Running tools.getCountsInHealpixSlices test...")
        nMagBins = 16
        N = 64
        ngMCMC = np.reshape(\
            tools.loadPickle(self.dataFolder + "ngCounts_ref.p"),\
            (nMagBins,N**3))
        hpIndices = tools.loadPickle(self.dataFolder + "hpIndices_ref.p")
        computed = tools.getCountsInHealpixSlices(ngMCMC,hpIndices,4,nres=64)
        referenceFile = self.dataFolder + "hpCounts_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_loadPickle(self):
        print("Running tools.loadPickle test...")
        np.random.seed(1000)
        result = np.random.random(1000)
        dataFile = self.dataFolder + "loadPickle_computed.p"
        with open(dataFile,"wb") as outfile: 
            pickle.dump(result,outfile)
        computed = tools.loadPickle(dataFile)
        referenceFile = self.dataFolder + "loadPickle_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_savePickle(self):
        print("Running tools.savePickle test...")
        np.random.seed(1000)
        result = np.random.random(1000)
        dataFile = self.dataFolder + "savePickle_computed.p"
        tools.savePickle(result,dataFile)
        with open(dataFile,"rb") as infile:
            computed = pickle.load(infile)
        referenceFile = self.dataFolder + "savePickle_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_positionsToHealpix(self):
        print("Running tools.positionsToHealpix test...")
        np.random.seed(1000)
        positions = np.random.random((3,1000))*1000 - 1000
        computed = tools.positionsToHealpix(positions)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "positionsToHealpix_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_downsample(self):
        print("Running tools.downsample test...")
        np.random.seed(1000)
        arr = np.random.random((64,64,64))
        computed = tools.downsample(arr,2)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "downsample_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)

@unittest.skip("Tests in development")
# Tests for cosmology functions:
class test_cosmology(test_base):
    def setUp(self):
        self.dataFolder=dataFolder
        self.test_subfolder = "function_tests_cosmology/"
        self.rtol=1e-5
        self.atol=1e-8
        self.generateTestData = generateMode
    def test_fsigma(self):
        print("Running cosmology.fsigma test...")
        sigmaArr = np.array([0.1,0.5,0.8,1.0,1.2])
        computed = cosmology.fsigma(sigmaArr,1,1,1,1)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "fsigma_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_fLinear(self):
        print("Running cosmology.fLinear test...")
        zList = np.array([0.2,0])
        computed = cosmology.fLinear(zList,0.3,0.7)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "fLinear_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_interpolateTransfer(self):
        print("Running cosmology.interpolateTransfer test...")
        k = np.linspace(2e-3,9,1000)
        [ki,Tki] = tools.loadPickle(self.dataFolder + self.test_subfolder + \
            "interpolateTransfer_data.p")
        computed = cosmology.interpolateTransfer(k,ki,Tki)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "interpolateTransfer_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_Pkinterp(self):
        print("Running cosmology.Pkinterp test...")
        [ki,Tki] = tools.loadPickle(self.dataFolder + self.test_subfolder + \
            "interpolateTransfer_data.p")
        computed = cosmology.Pkinterp(ki,Tki,ki,0.9665,1.0)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "Pkinterp_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_What(self):
        print("Running cosmology.What test...")
        krange = 10**(np.linspace(-3,1,101))
        computed = cosmology.What(krange,1.0)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "What_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_Whatp(self):
        print("Running cosmology.Whatp test...")
        krange = 10**(np.linspace(-3,1,101))
        computed = cosmology.Whatp(krange,1.0)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "Whatp_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_computeSigma80(self):
        print("Running cosmology.computeSigma80 test...")
        [ki,Tki] = tools.loadPickle(self.dataFolder + self.test_subfolder + \
            "interpolateTransfer_data.p")
        computed = cosmology.computeSigma80(1.0,0.9665,Tki,ki)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "computeSigma80_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_computePKAmplitude(self):
        print("Running tools.computePkAmplitude test...")
        [ki,Tki] = tools.loadPickle(self.dataFolder + self.test_subfolder + \
            "interpolateTransfer_data.p")
        computed = cosmology.computePkAmplitude(np.linspace(0.7,1.3,101),0,\
            ki,Tki,0.9665)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "computePKAmplitude_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_sigmaIntegrand(self):
        print("Running cosmology.sigmaIntegrand test...")
        [ki,Tki] = tools.loadPickle(self.dataFolder + self.test_subfolder + \
            "interpolateTransfer_data.p")
        k = np.linspace(2e-3,9,1000)
        computed = cosmology.sigmaIntegrand(k,1.0,ki,Tki,0.9665,1.0)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "sigmaIntegrand_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_computeSigma(self):
        print("Running cosmology.computeSigma test...")
        [ki,Tki] = tools.loadPickle(self.dataFolder + self.test_subfolder + \
            "interpolateTransfer_data.p")
        rhoB = 0.3*2.7754e11
        computed = cosmology.computeSigma(0.0,1e14,rhoB,Tki,ki,0.9665,1.0)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "computeSigma_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_sigmapIntegrand(self):
        print("Running cosmology.sigmapIntegrand test...")
        k = np.linspace(2e-3,9,1000)
        [ki,Tki] = tools.loadPickle(self.dataFolder + self.test_subfolder + \
            "interpolateTransfer_data.p")
        computed = cosmology.sigmapIntegrand(k,1.0,1.0,ki,Tki,0.9665,1.0)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "sigmapIntegrand_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_computeDSigmaDM(self):
        print("Running cosmology.computeDSigmaDM test...")
        [ki,Tki] = tools.loadPickle(self.dataFolder + self.test_subfolder + \
            "interpolateTransfer_data.p")
        rhoB = 0.3*2.7754e11
        computed = cosmology.computeDSigmaDM(0.0,1e14,rhoB,Tki,ki,0.9665,1.0)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "computeDSigmaDM_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_TMF(self):
        print("Running cosmology.TMF test...")
        [ki,Tki] = tools.loadPickle(self.dataFolder + self.test_subfolder + \
            "interpolateTransfer_data.p")
        Mrange = 10**(np.linspace(13,15,101))
        rhoB = 0.3*2.7754e11
        computed = cosmology.TMF(Mrange,1.0,1.0,1.0,1.0,0.0,\
            rhoB,Tki,ki,0.9665,0.8)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "TMF_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_TMF_from_hmf(self):
        print("Running cosmology.TMF_from_hmf test...")
        computed = cosmology.TMF_from_hmf(1e13,1e15)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "TMF_from_hmf_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_PSMF(self):
        print("Running cosmology.PSMF test...")
        computed = cosmology.PSMF(1e13,1e15)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "PSMF_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_dndm_to_n(self):
        print("Running cosmology.dndm_to_n test...")
        [dndm,m] = tools.loadPickle(self.dataFolder + self.test_subfolder + \
            "TMF_from_hmf_ref.p")
        computed = cosmology.dndm_to_n(dndm,m,10**(np.linspace(13,15,31)))
        referenceFile = self.dataFolder + self.test_subfolder + \
            "dndm_to_n_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_vol(self):
        print("Running cosmology.vol test...")
        computed = cosmology.vol(0.0,0.05,1.0,0.3)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "vol_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_Lvol(self):
        print("Running cosmology.Lvol test...")
        computed = cosmology.Lvol(0.0,0.05,1.0,0.3)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "Lvol_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_E(self):
        print("Running cosmology.E test...")
        computed = cosmology.E(0.5,0.3,0.0,0.0)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "E_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_drCo(self):
        print("Running cosmology.drCo test...")
        computed = cosmology.drCo(0.0,0.3)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "drCo_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_fTinker(self):
        print("Running cosmology.fTinker test...")
        computed = cosmology.fTinker(0.8,0.186,1.47,2.57,1.19)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "fTinker_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_tmf(self):
        print("Running cosmology.tmf test...")
        [ki,Tki] = tools.loadPickle(self.dataFolder + self.test_subfolder + \
            "interpolateTransfer_data.p")
        Mrange = 10**(np.linspace(13,15,101))
        rhoB = 0.3*2.7754e11
        Pki = tools.loadPickle(self.dataFolder + self.test_subfolder + \
            "Pkinterp_ref.p")
        computed = cosmology.tmf(Mrange,ki,Pki,rhoB,0.3)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "tmf_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_linearGrowthD(self):
        print("Running cosmology.linearGrowthD test...")
        computed = cosmology.linearGrowthD(0.0,0.3)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "linearGrowthD_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_linearGrowthf(self):
        print("Running cosmology.linearGrowthf test...")
        computed = cosmology.linearGrowthf(0.0,0.3)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "linearGrowthf_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_windowRtoM(self):
        print("Running cosmology.windowRtoM test...")
        rhoB = 0.3*2.7754e11
        computed = cosmology.windowRtoM(1.0,rhoB)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "windowRtoM_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_windowMtoR(self):
        print("Running cosmology.windowMtoR test...")
        rhoB = 0.3*2.7754e11
        computed = cosmology.windowMtoR(1.0,rhoB)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "windowMtoR_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_W_spatialTH(self):
        print("Running cosmology.W_spatialTH test...")
        computed = cosmology.W_spatialTH(1.0)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "W_spatialTH_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_W_spatialTHp(self):
        print("Running cosmology.W_spatialTHp test...")
        computed = cosmology.W_spatialTHp(1.0)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "W_spatialTHp_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_sigma2Mtophat(self):
        print("Running cosmology.sigma2Mtophat test...")
        [ki,Tki] = tools.loadPickle(self.dataFolder + self.test_subfolder + \
            "interpolateTransfer_data.p")
        Pki = tools.loadPickle(self.dataFolder + self.test_subfolder + \
            "Pkinterp_ref.p")
        Pk = scipy.interpolate.interp1d(ki,Pki,kind="cubic")
        rhoB = 0.3*2.7754e11
        computed = cosmology.sigma2Mtophat(1e14,Pk,rhoB,ki)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "sigma2Mtophat_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_sigmaTinker(self):
        print("Running cosmology.sigmaTinker test...")
        [ki,Tki] = tools.loadPickle(self.dataFolder + self.test_subfolder + \
            "interpolateTransfer_data.p")
        Pki = tools.loadPickle(self.dataFolder + self.test_subfolder + \
            "Pkinterp_ref.p")
        Pk = scipy.interpolate.interp1d(ki,Pki,kind="cubic")
        rhoB = 0.3*2.7754e11
        computed = cosmology.sigmaTinker(1.0,Pk,rhoB,1e-3,10)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "sigmaTinker_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_jacobian_tinker(self):
        print("Running cosmology.jacobian_tinker test...")
        [ki,Tki] = tools.loadPickle(self.dataFolder + self.test_subfolder + \
            "interpolateTransfer_data.p")
        Pki = tools.loadPickle(self.dataFolder + self.test_subfolder + \
            "Pkinterp_ref.p")
        Pk = scipy.interpolate.interp1d(ki,Pki,kind="cubic")
        rhoB = 0.3*2.7754e11
        computed = cosmology.jacobian_tinker(0.8,1e14,rhoB,Pk,1e-3,10)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "jacobian_tinker_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_sigmaRspatialTH(self):
        print("Running cosmology.sigmaRspatialTH test...")
        Pki = tools.loadPickle(self.dataFolder + self.test_subfolder + \
            "Pkinterp_ref.p")
        [ki,Tki] = tools.loadPickle(self.dataFolder + self.test_subfolder + \
            "interpolateTransfer_data.p")
        computed = cosmology.sigmaRspatialTH(1.0,ki,Pki)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "sigmaRspatialTH_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_f_SVdW(self):
        print("Running cosmology.f_SVdW test...")
        computed = cosmology.f_SVdW(0.8,-1.686,1.686)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "f_SVdW_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_jacobian_SVdW(self):
        print("Running cosmology.jacobian_SVdW test...")
        [ki,Tki] = tools.loadPickle(self.dataFolder + self.test_subfolder + \
            "interpolateTransfer_data.p")
        Pki = tools.loadPickle(self.dataFolder + self.test_subfolder + \
            "Pkinterp_ref.p")
        Pk = scipy.interpolate.interp1d(ki,Pki,kind="cubic")
        rhoB = 0.3*2.7754e11
        computed = cosmology.jacobian_SVdW(0.8,1e14,rhoB,Pk)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "jacobian_SVdW_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_jacobian_spatialTH(self):
        print("Running cosmology.jacobian_spatialTH test...")
        Pki = tools.loadPickle(self.dataFolder + self.test_subfolder + \
            "Pkinterp_ref.p")
        [ki,Tki] = tools.loadPickle(self.dataFolder + self.test_subfolder + \
            "interpolateTransfer_data.p")
        rhoB = 0.3*2.7754e11
        computed = cosmology.jacobian_spatialTH(0.8,1e14,rhoB,ki,Pki)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "jacobian_spatialTH_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_rhoSI(self):
        print("Running cosmology.rhoSI test...")
        computed = cosmology.rhoSI(0.3)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "rhoSI_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_rhoCos(self):
        print("Running cosmology.rhoCos test...")
        computed = cosmology.rhoCos(0.3)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "rhoCos_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    @unittest.skip("Seems to have a system-dependence in external dependency.")
    def test_powerSpectrum(self):
        print("Running cosmology.powerSpectrum test...")
        computed = cosmology.powerSpectrum()
        referenceFile = self.dataFolder + self.test_subfolder + \
            "powerSpectrum_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_pkToxi(self):
        print("Running cosmology.pkToxi test...")
        Pki = tools.loadPickle(self.dataFolder + self.test_subfolder + \
            "Pkinterp_ref.p")
        [ki,Tki] = tools.loadPickle(self.dataFolder + self.test_subfolder + \
            "interpolateTransfer_data.p")
        x = np.linspace(0,1.0,101)
        computed = cosmology.pkToxi(x,ki,Pki)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "pkToxi_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_vLinear(self):
        print("Running cosmology.vLinear test...")
        computed = cosmology.vLinear(1.0,200,0.3,0.7)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "vLinear_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_deltaCumulative(self):
        print("Running cosmology.deltaCumulative test...")
        pairCountsRef = np.load(dataFolder + \
            "reference_constrained/pair_counts_ref.npy")
        volumesListRef = np.load(dataFolder + \
            "reference_constrained/volumes_list_ref.npy")
        nbar = (512/677.7)**3
        computed = cosmology.deltaCumulative(pairCountsRef,volumesListRef,nbar)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "deltaCumulative_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_luminosityToComoving(self):
        print("Running cosmology.luminosityToComoving test...")
        cosmo = astropy.cosmology.FlatLambdaCDM(H0 = 70,Om0 = 0.3,Ob0 = 0.045)
        lumDist = 20
        computed = cosmology.luminosityToComoving(lumDist,cosmo)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "luminosityToComoving_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_comovingToLuminosity(self):
        print("Running cosmology.comovingToLuminosity test...")
        comovingDistance = 2
        cosmo = astropy.cosmology.FlatLambdaCDM(H0 = 70,Om0 = 0.3,Ob0 = 0.045)
        computed = cosmology.comovingToLuminosity(comovingDistance,cosmo)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "comovingToLuminosity_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_sigmaPk(self):
        print("Running cosmology.sigmaPk test...")
        Pki = tools.loadPickle(self.dataFolder + self.test_subfolder + \
            "Pkinterp_ref.p")
        Nk = 100
        Np = 100
        computed = cosmology.sigmaPk(Pki,Nk,Np)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "sigmaPk_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_overdensity(self):
        print("Running cosmology.overdensity test...")
        standard = dataFolder + \
            "reference_constrained/sample2791/gadget_full_forward/snapshot_001"
        snap = pynbody.load(standard)
        computed = cosmology.overdensity(snap,135)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "overdensity_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_lowMemoryOverdensity(self):
        print("Running cosmology.lowMemoryOverdensity test...")
        standard = dataFolder + \
            "reference_constrained/sample2791/gadget_full_forward/snapshot_001"
        snap = pynbody.load(standard)
        computed = cosmology.lowMemoryOverdensity(snap,135)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "lowMemoryOverdensity_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_sampleOverdensities(self):
        print("Running cosmology.sampleOverdensities test...")
        standard = dataFolder + \
            "reference_constrained/sample2791/gadget_full_forward/snapshot_001"
        snap = pynbody.load(standard)
        computed = cosmology.sampleOverdensities(snap,135,1000,seed=1000)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "sampleOverdensities_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_computeLocalUnderdensity(self):
        print("Running cosmology.computeLocalUnderdensity test...")
        standard = dataFolder + \
            "reference_constrained/sample2791/gadget_full_forward/snapshot_001"
        snap = pynbody.load(standard)
        rBins = np.linspace(10,100,101)
        computed = cosmology.computeLocalUnderdensity(snap,rBins)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "computeLocalUnderdensity_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_getMassProfile(self):
        print("Running cosmology.getMassProfile test...")
        standard = dataFolder + \
            "reference_constrained/sample2791/gadget_full_forward/snapshot_001"
        snap = pynbody.load(standard)
        radii = np.linspace(0.1,10,21)
        computed = cosmology.getMassProfile(radii,np.array([0.0]*3),snap)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "getMassProfile_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_cMBhattacharya(self):
        print("Running cosmology.cMBhattacharya test...")
        M = 10**(np.linspace(13,15,101))
        computed = cosmology.cMBhattacharya(M)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "cMBhattacharya_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_mNFW(self):
        print("Running cosmology.mNFW test...")
        computed = cosmology.mNFW(0.1)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "mNFW_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_criticalFunction(self):
        print("Running cosmology.criticalFunction test...")
        computed = cosmology.criticalFunction(0.1,1e14,200,500)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "criticalFunction_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_convertDeltaToCritical(self):
        print("Running cosmology.convertDeltaToCritical test...")
        computed = []
        computed.append(cosmology.convertDeltaToCritical(200,"mean"))
        computed.append(cosmology.convertDeltaToCritical(200,"critical"))
        computed.append(cosmology.convertDeltaToCritical(200,"virial"))
        referenceFile = self.dataFolder + self.test_subfolder + \
            "convertDeltaToCritical_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_getBoundedInterval(self):
        print("Running cosmology.getBoundedInterval test...")
        computed = cosmology.getBoundedInterval(1e14,200,500)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "getBoundedInterval_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_convertCriticalMass(self):
        print("Running cosmology.convertCriticalMass test...")
        computed = cosmology.convertCriticalMass(1e14,500)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "convertCriticalMass_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_eulerToZ(self):
        print("Running cosmology.eulerToZ test...")
        standard = dataFolder + \
            "reference_constrained/sample2791/gadget_full_forward/snapshot_001"
        snap = pynbody.load(standard)
        Om0 = snap.properties['omegaM0']
        h = snap.properties['h']
        cosmo = astropy.cosmology.FlatLambdaCDM(H0 = 100*h,\
            Om0 = Om0,Ob0 = 0.045)
        boxsize = snap.properties['boxsize'].ratio("Mpc a h**-1")
        computed = cosmology.eulerToZ(snap['pos'][0:100,:],\
            snap['vel'][0:100,:],cosmo,boxsize,h)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "eulerToZ_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)


# antihalos functions:
@unittest.skip("Tests in development")
class test_antihalos(test_base):
    def setUp(self):
        self.dataFolder=dataFolder
        self.test_subfolder = "function_tests_antihalos/"
        self.rtol=1e-5
        self.atol=1e-8
        self.generateTestData = generateMode
    def test_voidsRadiiFromAntiHalos(self):
        print("Running antihalos.voidsRadiiFromAntiHalos test...")
        standard = dataFolder + \
            "reference_constrained/sample2791/gadget_full_forward/snapshot_001"
        reverse = dataFolder + \
            "reference_constrained/sample2791/gadget_full_reverse/snapshot_001"
        snapn = pynbody.load(standard)
        snapr = pynbody.load(reverse)
        hn = snapn.halos()
        hr = snapr.halos()
        volumes = np.load(dataFolder + \
            "reference_constrained/volumes_ref.npy")
        computed = antihalos.voidsRadiiFromAntiHalos(snapn,snapr,hn,hr,volumes)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "voidsRadiiFromAntiHalos_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_computeAntiHaloCentres(self):
        print("Running antihalos.computeAntiHaloCentres test...")
        standard = dataFolder + \
            "reference_constrained/sample2791/gadget_full_forward/snapshot_001"
        reverse = dataFolder + \
            "reference_constrained/sample2791/gadget_full_reverse/snapshot_001"
        snapn = pynbody.load(standard)
        snapr = pynbody.load(reverse)
        hr = snapr.halos()
        volumes = np.load(dataFolder + \
            "reference_constrained/volumes_ref.npy")
        computed = antihalos.computeAntiHaloCentres(hr,snapn,volumes)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "computeAntiHaloCentres_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_getAntiHaloMasses(self):
        print("Running antihalos.getAntiHaloMasses test...")
        reverse = dataFolder + \
            "reference_constrained/sample2791/gadget_full_reverse/snapshot_001"
        snapr = pynbody.load(reverse)
        hr = snapr.halos()
        computed = []
        computed.append(antihalos.getAntiHaloMasses(hr))
        computed.append(antihalos.getAntiHaloMasses(hr,fixedMass=True))
        referenceFile = self.dataFolder + self.test_subfolder + \
            "getAntiHaloMasses_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_getAntiHaloDensities(self):
        print("Running antihalos.getAntiHaloDensities test...")
        standard = dataFolder + \
            "reference_constrained/sample2791/gadget_full_forward/snapshot_001"
        reverse = dataFolder + \
            "reference_constrained/sample2791/gadget_full_reverse/snapshot_001"
        snapn = pynbody.load(standard)
        snapr = pynbody.load(reverse)
        hr = snapr.halos()
        volumes = np.load(dataFolder + \
            "reference_constrained/volumes_ref.npy")
        computed = antihalos.getAntiHaloDensities(hr,snapn,volumes=volumes)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "getAntiHaloDensities_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_fitMassAndRadii(self):
        print("Running antihalos.fitMassAndRadii test...")
        standard = dataFolder + \
            "reference_constrained/sample2791/gadget_full_forward/snapshot_001"
        snap = pynbody.load(standard)
        ahProps = tools.loadPickle(snap.filename + ".AHproperties.p")
        antihaloRadii = ahProps[7]
        antihaloMasses = ahProps[3]
        computed = antihalos.fitMassAndRadii(antihaloMasses,antihaloRadii)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "fitMassAndRadii_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_MtoR(self):
        print("Running antihalos.antihalos test...")
        computed = antihalos.MtoR(0.1,0.2,0.3)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "MtoR_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_RtoM(self):
        print("Running antihalos.RtoM test...")
        computed = antihalos.RtoM(0.1,0.2,0.3)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "RtoM_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_computePeriodicCentreWeighted(self):
        print("Running antihalos.computePeriodicCentreWeighted test...")
        standard = dataFolder + \
            "reference_constrained/sample2791/gadget_full_forward/snapshot_001"
        snap = pynbody.load(standard)
        snapsort = np.argsort(snap['iord'])
        volumes = np.load(dataFolder + \
            "reference_constrained/volumes_ref.npy")
        halos = snap.halos()
        periodicity = snap.properties['boxsize'].ratio("Mpc a h**-1")
        computed = antihalos.computePeriodicCentreWeighted(halos[1]['pos'],\
            volumes[snapsort][halos[1]['iord']],periodicity)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "computePeriodicCentreWeighted_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_getCoincidingVoids(self):
        print("Running antihalos.getCoincidingVoids test...")
        standard = dataFolder + \
            "reference_constrained/sample2791/gadget_full_forward/snapshot_001"
        snap = pynbody.load(standard)
        ahProps = tools.loadPickle(snap.filename + ".AHproperties.p")
        ahCentresList = ahProps[5]
        computed = antihalos.getCoincidingVoids(ahCentresList[0,:],40,\
            ahCentresList)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "getCoincidingVoids_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_getCoincidingVoidsInRadiusRange(self):
        print("Running antihalos.getCoincidingVoidsInRadiusRange test...")
        standard = dataFolder + \
            "reference_constrained/sample2791/gadget_full_forward/snapshot_001"
        snap = pynbody.load(standard)
        ahProps = tools.loadPickle(snap.filename + ".AHproperties.p")
        ahCentresList = ahProps[5]
        antihaloRadii = ahProps[7]
        computed = antihalos.getCoincidingVoidsInRadiusRange(\
            ahCentresList[0,:],40,ahCentresList,antihaloRadii,5,20)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "getCoincidingVoidsInRadiusRange_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_getAntihaloOverlapWithVoid(self):
        print("Running antihalos.getAntihaloOverlapWithVoid test...")
        volumes = np.load(dataFolder + "reference_constrained/volumes_ref.npy")
        reverse1 = dataFolder + \
            "reference_constrained/sample2791/gadget_full_reverse/snapshot_001"
        reverse2 = dataFolder + \
            "reference_constrained/sample3250/gadget_full_reverse/snapshot_001"
        snapr1 = pynbody.load(reverse1)
        snapr2 = pynbody.load(reverse2)
        hr1 = snapr1.halos()
        hr2 = snapr2.halos()
        computed = antihalos.getAntihaloOverlapWithVoid(hr1[1]['iord'],\
            hr2[1]['iord'],volumes)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "getAntihaloOverlapWithVoid_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_getOverlapFractions(self):
        print("WARNING - TEST OF getOverlapFractions NOT YET IMPLEMENTED")
        self.assertTrue(True)
    def test_getVoidOverlapFractionsWithAntihalos(self):
        print("Running antihalos.getVoidOverlapFractionsWithAntihalos test...")
        volumes = np.load(dataFolder + "reference_constrained/volumes_ref.npy")
        reverse = dataFolder + \
            "reference_constrained/sample2791/gadget_full_reverse/snapshot_001"
        snapr = pynbody.load(reverse)
        standard = dataFolder + \
            "reference_constrained/sample3250/gadget_full_forward/snapshot_001"
        snapn = pynbody.load(standard)
        hr = snapr.halos()
        ahProps = tools.loadPickle(snapn.filename + ".AHproperties.p")
        computed = antihalos.getVoidOverlapFractionsWithAntihalos(\
            hr[1]['iord'],hr,np.arange(len(hr)),volumes)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "getVoidOverlapFractionsWithAntihalos_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
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
    @unittest.skip("Broken: seems to be a bug in Corrfunc with rBins???")
    def test_getAutoCorrelations(self):
        print("Running antihalos.getAutoCorrelations test...")
        standard1 = dataFolder + \
            "reference_constrained/sample2791/gadget_full_forward/snapshot_001"
        standard2 = dataFolder + \
            "reference_constrained/sample3250/gadget_full_forward/snapshot_001"
        ahProps1 = tools.loadPickle(standard1 + ".AHproperties.p")
        ahProps2 = tools.loadPickle(standard2 + ".AHproperties.p")
        centres1 = ahProps1[5]
        centres2 = ahProps2[5]
        radii1 = ahProps1[7]
        radii2 = ahProps2[7]
        computed = antihalos.getAutoCorrelations(centres1,centres2,\
            radii1,radii2,boxsize=677.7)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "getAutoCorrelations_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    @unittest.skip("Broken: seems to be a bug in Corrfunc with rBins???")
    def test_simulationCorrelation(self):
        print("Running antihalos.simulationCorrelation test...")
        rRange = np.linspace(0.1,10,101)
        boxsize = 200.0
        np.random.seed(1000)
        pos = np.random.random((1000,3))*200
        computed = antihalos.simulationCorrelation(rRange,boxsize,pos,\
            nThreads=nThreads)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "simulationCorrelation_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_getStacks(self):
        print("WARNING - TEST OF getStacks" + \
            " NOT YET IMPLEMENTED")
        self.assertTrue(True)


@unittest.skip("Tests in development")
class test_context(test_base):
    def setUp(self):
        self.dataFolder=dataFolder
        self.test_subfolder = "function_tests_context/"
        self.rtol=1e-5
        self.atol=1e-8
        self.generateTestData = generateMode
    def test_get_nearest_halos(self):
        print("Running context.get_nearest_halos test...")
        standard = self.dataFolder + \
            "reference_constrained/sample2791/gadget_full_forward/snapshot_001"
        snap = pynbody.load(standard)
        hn = snap.halos()
        ahProps = tools.loadPickle(standard + ".AHproperties.p")
        centres = ahProps[5]
        computed = context.get_nearest_halos(np.array([0.0]*3),hn,coms=centres)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "get_nearest_halos_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_halo_centres_and_mass(self):
        print("Running context.halo_centres_and_mass test...")
        standard = dataFolder + \
            "reference_constrained/sample2791/gadget_full_forward/snapshot_001"
        snap = pynbody.load(standard)
        hn = snap.halos()
        haloFilter = np.arange(0,100)
        computed = context.halo_centres_and_mass(hn,haloFilter=haloFilter)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "halo_centres_and_mass_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_computePeriodicCentreWeighted(self):
        print("Running context.computePeriodicCentreWeighted test...")
        standard = dataFolder + \
            "reference_constrained/sample2791/gadget_full_forward/snapshot_001"
        reverse = dataFolder + \
            "reference_constrained/sample2791/gadget_full_reverse/snapshot_001"
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
        referenceFile = self.dataFolder + self.test_subfolder + \
            "computePeriodicCentreWeighted_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_periodicCentreWeighted(self):
        print("Running context.periodicCentreWeighted test...")
        standard = dataFolder + \
            "reference_constrained/sample2791/gadget_full_forward/snapshot_001"
        reverse = dataFolder + \
            "reference_constrained/sample2791/gadget_full_reverse/snapshot_001"
        snapn = pynbody.load(standard)
        snapr = pynbody.load(reverse)
        hr = snapr.halos()
        volumes = tools.zobovVolumesToPhysical(standard + ".vols",snapn,\
            dtype=np.double,offset=4)
        boxsize = snapn.properties['boxsize'].ratio("Mpc a h**-1")
        snapsort = np.argsort(snapr['iord'])
        computed = context.periodicCentreWeighted(hr[1],\
            volumes[snapsort][hr[1]['iord']],boxsize)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "periodicCentreWeighted_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_periodicCentre(self):
        print("Running context.periodicCentre test...")
        standard = dataFolder + \
            "reference_constrained/sample2791/gadget_full_forward/snapshot_001"
        reverse = dataFolder + \
            "reference_constrained/sample2791/gadget_full_reverse/snapshot_001"
        snapn = pynbody.load(standard)
        snapr = pynbody.load(reverse)
        hr = snapr.halos()
        boxsize = snapn.properties['boxsize'].ratio("Mpc a h**-1")
        computed = context.periodicCentre(hr[1],boxsize)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "periodicCentre_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_halo_filter(self):
        print("Running context.halo_filter test...")
        standard = dataFolder + \
            "reference_constrained/sample2791/gadget_full_forward/snapshot_001"
        reverse = dataFolder + \
            "reference_constrained/sample2791/gadget_full_reverse/snapshot_001"
        snapn = pynbody.load(standard)
        hn = snapn.halos()
        snapr = pynbody.load(reverse)
        hr = snapr.halos()
        boxsize = snapn.properties['boxsize'].ratio("Mpc a h**-1")
        filt = pynbody.filt.Sphere(20,np.array([boxsize/2]*3))
        computed = context.halo_filter(snapn,hr,filt)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "halo_filter_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_rotation_between(self):
        print("Running context.rotation_between test...")
        np.random.seed(1000)
        a = np.random.random(3)
        b = np.random.random(3)
        computed = context.rotation_between(a,b)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "rotation_between_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_orthogonal_part(self):
        print("Running context.orthogonal_part test...")
        np.random.seed(1000)
        a = np.random.random(3)
        b = np.random.random(3)
        computed = context.orthogonal_part(a,b)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "orthogonal_part_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_cluster_fit(self):
        print("Running context.cluster_fit test...")
        np.random.seed(1000)
        r1 = np.random.random(3)
        r2 = np.random.random(3)
        r3 = np.random.random(3)
        R1 = r1 + 0.1*np.random.random(3)
        R2 = r2 + 0.1*np.random.random(3)
        R3 = r3 + 0.1*np.random.random(3)
        cluster = np.random.random(3)
        computed = context.cluster_fit(cluster,r1,r2,r3,R1,R2,R3)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "cluster_fit_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_position3d(self):
        print("Running context.position3d test...")
        np.random.seed(1000)
        ra = 360*np.random.random(100)
        dec = 180*np.random.random(100) - 90
        r = np.random.random(100)
        equatorial = np.vstack((ra,dec,r)).transpose()
        computed = context.position3d(equatorial)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "position3d_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_galactic_to_equatorial(self):
        print("Running context.galactic_to_equatorial test...")
        np.random.seed(1000)
        l = 360*np.random.random(100)
        b = 180*np.random.random(100) - 90
        galactic = np.vstack((l,b)).transpose()
        computed = context.galactic_to_equatorial(galactic)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "galactic_to_equatorial_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_equatorial_to_galactic(self):
        print("Running context.equatorial_to_galactic test...")
        np.random.seed(1000)
        ra = 360*np.random.random(100)
        dec = 180*np.random.random(100) - 90
        equatorial = np.vstack((ra,dec)).transpose()
        computed = context.equatorial_to_galactic(equatorial)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "equatorial_to_galactic_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_sgl_gal_matrix(self):
        print("Running context.sgl_gal_matrix test...")
        computed = context.sgl_gal_matrix(\
            137.37*np.pi/180.0,47.37*np.pi/180.0,0.0,bz = 6.32*np.pi/180.0)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "sgl_gal_matrix_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_galactic_to_supergalactic(self):
        print("Running context.galactic_to_supergalactic test...")
        np.random.seed(1000)
        galactic = np.random.random((1000,3))
        computed = context.galactic_to_supergalactic(galactic)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "galactic_to_supergalactic_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_supergalactic_to_galactic(self):
        print("Running context.supergalactic_to_galactic test...")
        np.random.seed(1000)
        galactic = np.random.random((1000,3))
        computed = context.supergalactic_to_galactic(galactic)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "supergalactic_to_galactic_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_gal2SG(self):
        print("Running context.gal2SG test...")
        np.random.seed(1000)
        l = 360*np.random.random(100)
        b = 180*np.random.random(100) - 90
        galactic = np.vstack((l,b)).transpose()
        computed = context.gal2SG(galactic)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "gal2SG_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_sg2gal(self):
        print("Running context.sg2gal test...")
        np.random.seed(1000)
        l = 360*np.random.random(100)
        b = 180*np.random.random(100) - 90
        galactic = np.vstack((l,b)).transpose()
        computed = context.sg2gal(galactic)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "sg2gal_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_local_group_z_correction(self):
        print("Running context.local_group_z_correction test...")
        np.random.seed(1000)
        l = 287
        b = 56
        zh = 0.045
        computed = context.local_group_z_correction(zh,b,l)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "local_group_z_correction_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_supergalactic_ang_to_pos(self):
        print("Running context.supergalactic_ang_to_pos test...")
        np.random.seed(1000)
        l = 360*np.random.random(100)
        b = 180*np.random.random(100) - 90
        r = 100*np.random.random(100)
        ang = np.vstack((r,l,b)).transpose()
        computed = context.supergalactic_ang_to_pos(ang)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "supergalactic_ang_to_pos_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_row2col(self):
        print("Running context.row2col test...")
        np.random.seed(1000)
        row = np.random.random(4)
        computed = context.row2col(row)
        self.assertTrue(computed.shape == (4,1))
    def test_mean_distance(self):
        print("Running context.mean_distance test...")
        standard = dataFolder + \
            "reference_constrained/sample2791/gadget_full_forward/snapshot_001"
        snapn = pynbody.load(standard)
        boxsize = snapn.properties['boxsize'].ratio("Mpc a h**-1")
        hn = snapn.halos()
        computed = context.mean_distance(hn[1])
        referenceFile = self.dataFolder + self.test_subfolder + \
            "mean_distance_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_distance(self):
        print("Running context.distance test...")
        np.random.seed(1000)
        pos = np.random.random((100,3))
        computed = context.distance(pos)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "distance_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_halo_distances(self):
        print("Running context.halo_distances test...")
        standard = dataFolder + \
            "reference_constrained/sample2791/gadget_full_forward/snapshot_001"
        snapn = pynbody.load(standard)
        hn = snapn.halos()
        computed = context.halo_distances(hn)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "halo_distances_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_snapunion_positions(self):
        print("Running context.snapunion_positions test...")
        standard = dataFolder + \
            "reference_constrained/sample2791/gadget_full_forward/snapshot_001"
        snapn = pynbody.load(standard)
        hn = snapn.halos()
        to_use = [1,2,3,4]
        computed = context.snapunion_positions(hn,to_use)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "snapunion_positions_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_select_sphere(self):
        print("Running context.select_sphere test...")
        standard = dataFolder + \
            "reference_constrained/sample2791/gadget_full_forward/snapshot_001"
        snapn = pynbody.load(standard)
        boxsize = snapn.properties['boxsize'].ratio("Mpc a h**-1")
        radius = 30
        offset = np.array([boxsize/2]*3)
        distance = 100
        direction = np.array([1.0,1.0,1.0])
        computed = np.array(context.select_sphere(snapn,radius,distance,\
            direction,offset=offset)['pos'])
        referenceFile = self.dataFolder + self.test_subfolder + \
            "select_sphere_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_get_containing_halos(self):
        print("Running context.get_containing_halos test...")
        standard = dataFolder + \
            "reference_constrained/sample2791/gadget_full_forward/snapshot_001"
        snap = pynbody.load(standard)
        halos = snap.halos()
        # Just run this with one halo as the snap, otherwise it will take
        # a long time to run:
        computed = context.get_containing_halos(halos[1],halos)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "get_containing_halos_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_combineHalos(self):
        print("Running context.combineHalos test...")
        standard = dataFolder + \
            "reference_constrained/sample2791/gadget_full_forward/snapshot_001"
        snap = pynbody.load(standard)
        halos = snap.halos()
        computed = np.array(context.combineHalos(snap,halos)['pos'])
        referenceFile = self.dataFolder + self.test_subfolder + \
            "combineHalos_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_localGroupTest(self):
        print("WARNING - TEST NOT IMPLEMENTED YET FOR localGroupTest")
        self.assertTrue(True)
    def test_spheresMonteCarlo(self):
        print("Running context.spheresMonteCarlo test...")
        np.random.seed(1000)
        centres = np.random.random((1000,3))*100
        radii = np.random.random(1000)*10
        computed = context.spheresMonteCarlo(centres,radii,[100,100,100],\
            seed=1000)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "spheresMonteCarlo_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_mapEquatorialSnapshotToGalactic(self):
        print("Running context.mapEquatorialSnapshotToGalactic test...")
        standard = dataFolder + \
            "reference_constrained/sample2791/gadget_full_forward/snapshot_001"
        snap = pynbody.load(standard)
        context.mapEquatorialSnapshotToGalactic(snap)
        computed = snap['pos']
        referenceFile = self.dataFolder + self.test_subfolder + \
            "mapEquatorialSnapshotToGalactic_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_mapGalacticSnapshotToEquatorial(self):
        print("Running context.mapGalacticSnapshotToEquatorial test...")
        standard = dataFolder + \
            "reference_constrained/sample2791/gadget_full_forward/snapshot_001"
        snap = pynbody.load(standard)
        context.mapGalacticSnapshotToEquatorial(snap)
        computed = snap['pos']
        referenceFile = self.dataFolder + self.test_subfolder + \
            "mapGalacticSnapshotToEquatorial_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_mapEquatorialToGalactic(self):
        print("Running context.mapEquatorialToGalactic test...")
        np.random.seed(1000)
        points = np.random.random((100,3))
        computed = context.mapEquatorialToGalactic(points)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "mapEquatorialToGalactic_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_mapGalacticToEquatorial(self):
        print("Running context.mapGalacticToEquatorial test...")
        np.random.seed(1000)
        points = np.random.random((100,3))
        computed = context.mapGalacticToEquatorial(points)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "mapGalacticToEquatorial_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_equatorialXYZToSkyCoord(self):
        print("Running context.equatorialXYZToSkyCoord test...")
        np.random.seed(1000)
        points = np.random.random((100,3))
        coord = context.equatorialXYZToSkyCoord(points)
        computed = np.array([np.array(coord.icrs.ra),\
            np.array(coord.icrs.dec),np.array(coord.icrs.distance)])
        referenceFile = self.dataFolder + self.test_subfolder + \
            "equatorialXYZToSkyCoord_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)


@unittest.skip("Tests in development")
class test_plot_utilities(test_base):
    def setUp(self):
        self.dataFolder=dataFolder
        self.test_subfolder = "function_tests_plot_utilities/"
        self.rtol=1e-5
        self.atol=1e-8
        self.generateTestData = generateMode
    def test_binCentres(self):
        print("Running plot_utilities.binValues test...")
        np.random.seed(1000)
        values = np.random.poisson(10,1000)
        bins = np.linspace(-0.5,20.5,22)
        computed = plot_utilities.binValues(values,bins = bins)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "binValues_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_binValues2d(self):
        print("Running plot_utilities.binValues2d test...")
        np.random.seed(1000)
        values = np.random.poisson(10,1000)
        bins = np.linspace(-0.5,20.5,22)
        computed = plot_utilities.binValues2d(values,bins = bins)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "binValues2d_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_intersectsSliceWithWrapping(self):
        print("Running plot_utilities.intersectsSliceWithWrapping test...")
        np.random.seed(1000)
        values = np.random.random((10000,2))*1000
        boxsize = 1000
        thickness = 20
        zslice = 300
        computed = plot_utilities.intersectsSliceWithWrapping(values,\
            zslice,thickness,boxsize)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "intersectsSliceWithWrapping_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_pointsInRangeWithWrap(self):
        print("Running plot_utilities.pointsInRangeWithWrap test...")
        np.random.seed(1000)
        values = np.random.random((10000,3))*1000
        boxsize = 1000
        lim = [380,420]
        computed = plot_utilities.pointsInRangeWithWrap(values,lim,\
            axis=2,boxsize=1000)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "pointsInRangeWithWrap_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_pointsInBoundedPlaneWithWrap(self):
        print("Running plot_utilities.pointsInBoundedPlaneWithWrap test...")
        np.random.seed(1000)
        values = np.random.random((10000,3))*1000
        boxsize = 1000
        xlim = [200,400]
        ylim = [200,400]
        computed = plot_utilities.pointsInBoundedPlaneWithWrap(values,\
            xlim,ylim,boxsize=1000)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "pointsInBoundedPlaneWithWrap.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_getAntihaloExtent(self):
        print("Running plot_utilities.getAntihaloExtent test...")
        standard = self.dataFolder + \
            "reference_constrained/sample2791/gadget_full_forward/snapshot_001"
        reverse = self.dataFolder + \
            "reference_constrained/sample2791/gadget_full_reverse/snapshot_001"
        snapn = pynbody.load(standard)
        hn = snapn.halos()
        snapr = pynbody.load(reverse)
        hr = snapr.halos()
        computed = plot_utilities.getAntihaloExtent(snapn,hr[1])
        referenceFile = self.dataFolder + self.test_subfolder + \
            "getAntihaloExtent_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_getAntiHaloParticlesIntersectingSlice(self):
        print("Running " + \
            "plot_utilities.getAntiHaloParticlesIntersectingSlice test...")
        standard = self.dataFolder + \
            "reference_constrained/sample2791/gadget_full_forward/snapshot_001"
        reverse = self.dataFolder + \
            "reference_constrained/sample2791/gadget_full_reverse/snapshot_001"
        snapn = pynbody.load(standard)
        hn = snapn.halos()
        snapr = pynbody.load(reverse)
        hr = snapr.halos()
        snapsort = np.argsort(snapn['iord'])
        antihalo = hr[1]
        positions = snapn['pos'][snapsort[antihalo['iord']]]
        weights = snapn['mass'][snapsort[antihalo['iord']]]
        boxsize = snapn.properties['boxsize'].ratio("Mpc a h**-1")
        centre = context.computePeriodicCentreWeighted(positions,weights,
            boxsize)*pynbody.units.Unit("Mpc a h**-1")
        computed = plot_utilities.getAntiHaloParticlesIntersectingSlice(\
            snapn,antihalo,centre[2],antihaloCentre=centre,thickness=15,\
            snapsort=snapsort)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "test_getAntiHaloParticlesIntersectingSlice_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_float_formatter(self):
        print("Running plot_utilities.float_formatter test...")
        reference = '100000.0'
        computed = plot_utilities.float_formatter(1e5,d=2)
        self.assertTrue(computed == reference)
    def test_floatsToStrings(self):
        print("Running plot_utilities.floatsToStrings test...")
        np.random.seed(1000)
        floatArray = np.random.random(5)
        reference = ['0.65', '0.12', '0.95', '0.48', '0.87']
        computed = plot_utilities.floatsToStrings(floatArray,precision=2)
        self.assertTrue(reference == computed)
    def test_sphericalSlice(self):
        print("Running plot_utilities.sphericalSlice test...")
        standard = self.dataFolder + \
            "reference_constrained/sample2791/gadget_full_forward/snapshot_001"
        snapn = pynbody.load(standard)
        radius = 135
        computed = plot_utilities.sphericalSlice(snapn,radius)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "sphericalSlice_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_filterPolarPointsToAnnulus(self):
        print("Running plot_utilities.sphericalSlice test...")
        np.random.seed(1000)
        lonlat = np.array([np.random.random(1000)*180,\
            np.random.random(1000)*180 - 90]).transpose()
        r = np.random.random(1000)*300
        radius = 135
        thickness = 15
        computed = plot_utilities.filterPolarPointsToAnnulus(lonlat,r,radius,\
            thickness = thickness)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "filterPolarPointsToAnnulus_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_computeMollweidePositions(self):
        print("Running plot_utilities.computeMollweidePositions test...")
        np.random.seed(1000)
        positions = np.random.random((10000,3))*1000
        computed = plot_utilities.computeMollweidePositions(positions)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "computeMollweidePositions_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)

@unittest.skip("Tests in development")
class test_survey(test_base):
    def setUp(self):
        self.dataFolder=dataFolder
        self.test_subfolder = "function_tests_survey/"
        self.rtol=1e-5
        self.atol=1e-8
        self.generateTestData = generateMode
    def test_incompleteGamma(self):
        print("Running survey.incompleteGamma test...")
        alist = np.linspace(1,5,10)
        xlist = np.linspace(0,20,100)
        computed = np.zeros((len(alist),len(xlist)))
        for k in range(0,len(alist)):
            computed[k,:] = survey.incompleteGamma(alist[k],xlist)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "incompleteGamma_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_gammafunc(self):
        print("Running survey.gammafunc test...")
        np.random.seed(1000)
        m = 11.5
        r10 = np.random.random(100)*20
        Mstar = -23.28
        alpha=0.5
        Mmin = -25
        Mmax = -21
        computed = survey.gammafunc(m,r10,Mstar,alpha,Mmin,Mmax)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "gammafunc_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_getCompletenessInSelection(self):
        print("Running survey.getCompletenessInSelection test...")
        np.random.seed(1000)
        points = np.random.random((10000,3))*1000
        centre = np.array([500,500,500])
        rl = np.sqrt(np.sum((points - centre)**2,1))
        mlow = 0.0
        mupp = 12.5
        Mlow = -25
        Mupp = -21
        cosmo = astropy.cosmology.LambdaCDM(70,0.3,0.7)
        nstar=1.14e-2
        numericalIntegration = False
        interpolateMask = False
        Ninterp = 1000
        alpha=0.5
        Mstar = -23.28
        computed = survey.getCompletenessInSelection(rl,mlow,mupp,Mlow,Mupp,\
            cosmo,nstar,None,numericalIntegration,interpolateMask,Ninterp,\
            alpha,Mstar)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "getCompletenessInSelection_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_surveyMask(self):
        print("Running survey.surveyMask test...")
        np.random.seed(1000)
        points = np.random.random((10000,3))*1000
        alpha=0.5
        Mstar = -23.28
        cosmo = astropy.cosmology.LambdaCDM(70,0.3,0.7)
        surveyMaskPath = self.dataFolder + "/2mpp_data/"
        surveyMask11 = healpy.read_map(surveyMaskPath + \
            "completeness_11_5.fits")
        surveyMask12 = healpy.read_map(surveyMaskPath + \
            "completeness_12_5.fits")
        computed = survey.surveyMask(points,surveyMask11,surveyMask12,cosmo,\
            alpha,Mstar)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "surveyMask_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_pointsToEquatorial(self):
        print("Running survey.pointsToEquatorial test...")
        np.random.seed(1000)
        points = np.random.random((10000,3))*1000
        computed = survey.pointsToEquatorial(points,1000)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "pointsToEquatorial_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_radialCompleteness(self):
        print("Running survey.radialCompleteness test...")
        np.random.seed(1000)
        points = np.random.random((10000,3))*1000
        centre = np.array([500,500,500])
        rl = np.sqrt(np.sum((points - centre)**2,1))
        Mstar = -23.28
        alpha=0.5
        Mmin = -25
        Mmax = -21
        mmax = 12.5
        cosmo = astropy.cosmology.LambdaCDM(70,0.3,0.7)
        nstar=1.14e-2
        computed = survey.radialCompleteness(\
            rl,alpha,Mstar,Mmin,Mmax,mmax,cosmo)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "radialCompleteness_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_griddedGalCountFromCatalogue(self):
        print("Running survey.griddedGalCountFromCatalogue test...")
        cosmo = astropy.cosmology.LambdaCDM(70,0.3,0.7)
        computed = survey.griddedGalCountFromCatalogue(cosmo,\
            tmppFile = self.dataFolder + "2mpp_data/2MPP.txt",N=64)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "griddedGalCountFromCatalogue_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)

@unittest.skip("Tests in development")
class test_real_clusters(test_base):
    def setUp(self):
        self.dataFolder=dataFolder
        self.test_subfolder = "function_tests_real_clusters/"
        self.rtol=1e-5
        self.atol=1e-8
        self.generateTestData = generateMode
    def test_getClusterSkyPositions(self):
        print("Running real_clusters.getClusterSkyPositions test...")
        computedList = real_clusters.getClusterSkyPositions(\
            fileroot = self.dataFolder)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "getClusterSkyPositions_ref.p"
        refList = tools.loadPickle(referenceFile)
        # Only compare the objects that are arrays. Directly comparing 
        # astropy objects is difficult as they can change between versions:
        reference = [refList[k] for k in range(0,6)]
        computed = [computedList[k] for k in range(0,6)]
        self.compareToReference(computed,reference)
    def test_getAntiHalosInSphere(self):
        print("Running real_clusters.getAntiHalosInSphere test...")
        np.random.seed(1000)
        centreList = np.random.random((100,3))*200
        computed = real_clusters.getAntiHalosInSphere(centreList,100,\
            origin=np.array([50,50,50]))
        referenceFile = self.dataFolder + self.test_subfolder + \
            "getAntiHalosInSphere_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    # Not happy with this function - is it actually needed?
    def test_getClusterCounterpartPositions(self):
        print("Running real_clusters.getClusterSkyPositions test...")
        abell_nums = [426,2147,1656,3627,3571,548,2197,2063,1367]
        [abell_l,abell_b,abell_n,abell_z,abell_d,p_abell,coordAbell] = \
            real_clusters.getClusterSkyPositions(self.dataFolder)
        snapname = self.dataFolder + \
            "reference_constrained/sample2791/gadget_full_forward/snapshot_001"
        snap = pynbody.load(snapname)
        ahProps = tools.loadPickle(snap.filename + ".AHproperties.p")
        boxsize = snap.properties['boxsize'].ratio("Mpc a h**-1")
        hncentres = ahProps[0]
        hnmasses = ahProps[1]
        computed = real_clusters.getClusterCounterpartPositions(abell_nums,\
            abell_n,-np.fliplr(p_abell),snap,hncentres,hnmasses,\
            boxsize = boxsize,mThresh=1e13)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "getClusterCounterpartPositions_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_getGriddedDensity(self):
        print("Running real_clusters.getGriddedDensity test...")
        snapname = self.dataFolder + \
            "reference_constrained/sample2791/gadget_full_forward/snapshot_001"
        snap = pynbody.load(snapname)
        N = 256
        computed = real_clusters.getGriddedDensity(snap,N)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "getGriddedDensity_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_getCombinedAbellCatalogue(self):
        print("Running real_clusters.getCombinedAbellCatalogue test...")
        computed = real_clusters.getCombinedAbellCatalogue(\
            catFolder = self.dataFolder)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "getCombinedAbellCatalogue_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)

@unittest.skip("Tests in development")
class test_stacking(test_base):
    def setUp(self):
        self.dataFolder=dataFolder
        self.test_subfolder = "function_tests_stacking/"
        self.rtol=1e-5
        self.atol=1e-8
        self.generateTestData = generateMode
    def test_weightedMean(self):
        print("Running stacking.weightedMean test...")
        np.random.seed(1000)
        xi = np.random.random(1000)*100
        wi = np.random.random(1000)
        computed = stacking.weightedMean(xi,wi)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "weightedMean_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_weightedVariance(self):
        print("Running stacking.weightedVariance test...")
        np.random.seed(1000)
        xi = np.random.random(1000)*100
        wi = np.random.random(1000)
        computed = stacking.weightedVariance(xi,wi)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "weightedVariance_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    @unittest.skip("Broken: seems to be a bug in Corrfunc with rBins???")
    def test_simulationCorrelation(self):
        print("Running stacking.simulationCorrelation test...")
        np.random.seed(1000)
        data1 = np.random.random((10000,3))*1000
        data2 = data1 + np.random.random((10000,3))*20 - 10
        boxsize = 1000
        rBins = np.linspace(0,100,21)
        computed = stacking.simulationCorrelation(rBins,boxsize,data1=data1,\
            data2=data2)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "simulationCorrelation_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    @unittest.skip("Broken: seems to be a bug in Corrfunc with rBins???")
    def test_getCrossCorrelations(self):
        print("Running stacking.simulationCorrelation test...")
        np.random.seed(1000)
        data1 = np.random.random((10000,3))*1000
        data2 = data1 + np.random.random((10000,3))*20 - 10
        radii1 = np.random.random(10000)*25
        radii2 = np.random.random(10000)*25
        computed = stacking.getCrossCorrelations(data1,data2,radii1,radii2)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "getCrossCorrelations_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    @unittest.skip("Broken: seems to be a bug in Corrfunc with rBins???")
    def test_getAutoCorrelations(self):
        print("Running stacking.getAutoCorrelations test...")
        np.random.seed(1000)
        data1 = np.random.random((10000,3))*1000
        data2 = data1 + np.random.random((10000,3))*20 - 10
        radii1 = np.random.random(10000)*25
        radii2 = np.random.random(10000)*25
        computed = stacking.getAutoCorrelations(data1,data2,radii1,radii2)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "getAutoCorrelations_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def snaptest_getPairCounts(self,ahCentresList,antihaloRadii,snapn):
        print("Running stacking.getPairCounts test...")
        rBins = np.linspace(10,25,21)
        computed = stacking.getPairCounts(ahCentresList,antihaloRadii,snapn,\
            rBins)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "getCombinedAbellCatalogue_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def snaptest_getRadialVelocityAverages(self,ahCentresList,\
            antihaloRadii,snapn):
        print("Running stacking.getRadialVelocityAverages test...")
        rBins = np.linspace(10,25,21)
        computed = stacking.getRadialVelocityAverages(ahCentresList,\
            antihaloRadii,snapn,rBins)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "getRadialVelocityAverages_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def snaptest_stackScaledVoids(self,ahCentresList,\
            antihaloRadii,snapn):
        print("Running stacking.stackScaledVoids test...")
        rBins = np.linspace(10,25,21)
        computed = stacking.stackScaledVoids(ahCentresList,\
            antihaloRadii,snapn,rBins)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "stackScaledVoids_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def snaptest_stackScaledVoidsVelocities(self,ahCentresList,\
            antihaloRadii,snapn,vorVolumes):
        print("Running stacking.stackScaledVoidsVelocities test...")
        rBins = np.linspace(10,25,21)
        computed = stacking.stackScaledVoidsVelocities(ahCentresList,\
            antihaloRadii,snapn,rBins,vorVolumes=vorVolumes)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "stackScaledVoidsVelocities_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def snaptest_stackVoidsWithFilter(self,ahCentresList,\
            antihaloRadii,snapn):
        print("Running stacking.stackVoidsWithFilter test...")
        filterToApply = np.where((antihaloRadii > 15) & (antihaloRadii < 35))[0]
        computed = stacking.stackVoidsWithFilter(ahCentresList,\
            antihaloRadii,filterToApply,snapn)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "stackVoidsWithFilter_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def snaptest_stackVoidVelocitiesWithFilter(self,ahCentresList,\
            antihaloRadii,snapn,vorVolumes):
        print("Running stacking.stackVoidVelocitiesWithFilter test...")
        filterToApply = np.where((antihaloRadii > 15) & (antihaloRadii < 35))[0]
        computed = stacking.stackVoidVelocitiesWithFilter(ahCentresList,\
            antihaloRadii,filterToApply,snapn,vorVolumes=vorVolumes)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "stackVoidVelocitiesWithFilter_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def snaptest_meanDensityContrast(self,voidParticles,volumes,nbar):
        print("Running stacking.meanDensityContrast test...")
        computed = stacking.meanDensityContrast(voidParticles,volumes,nbar)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "meanDensityContrast_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def snaptest_lambdaVoid(self,voidParticles,volumes,nbar,antihaloRadii):
        print("Running stacking.lambdaVoid test...")
        computed = stacking.lambdaVoid(voidParticles,volumes,nbar,\
            antihaloRadii[0])
        referenceFile = self.dataFolder + self.test_subfolder + \
            "lambdaVoid_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def snaptest_centralDensity(self,ahCentresList,antihaloRadii,\
            positions,volumesVoid,massesVoid,boxsize):
        print("Running stacking.centralDensity test...")
        computed = stacking.centralDensity(ahCentresList[0],antihaloRadii[0],\
            positions,volumesVoid,massesVoid,boxsize=boxsize)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "centralDensity_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def snaptest_centralDensityNN(self,ahCentresList,positions,massesVoid,\
            volumesVoid,boxsize):
        print("Running stacking.centralDensityNN test...")
        computed = stacking.centralDensityNN(ahCentresList[0],\
            positions,massesVoid,volumesVoid,boxsize=boxsize)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "centralDensityNN_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_profileParamsNadathur(self):
        print("Running stacking.profileParamsNadathur test...")
        computed = stacking.profileParamsNadathur(1.0,[0,5,10],1.0)
        reference = [1.57,5.72,0.81,-0.69]
        self.assertTrue(computed == reference)
    def test_profileModel(self):
        print("Running stacking.profileModel test...")
        modelArgs = [1.57,5.72,0.81,-0.69]
        r = 1.0
        computed = [stacking.profileModel(r,"Hamaus",modelArgs),\
            stacking.profileModel(r,"Nadathur",modelArgs)]
        referenceFile = self.dataFolder + self.test_subfolder + \
            "profileModel_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_profileModelHamaus(self):
        print("Running stacking.profileModelHamaus test...")
        rs = 0.81
        deltac = -0.69
        r = np.linspace(1,10,21)
        computed = stacking.profileModelHamaus(r,rs,deltac)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "profileModelHamaus_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_profileParamsHamaus(self):
        print("Running stacking.profileParamsHamaus test...")
        [rBins,rBinCentres,nbarj,sigmabarj,nbar] = tools.loadPickle(\
            self.dataFolder + "profile_examples.p")
        computed = stacking.profileParamsHamaus(nbarj,sigmabarj,rBins,nbar)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "profileParamsHamaus_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_profileModelNadathur(self):
        print("Running stacking.profileModelNadathur test...")
        modelArgs = [1.57,5.72,0.81,-0.69]
        r = np.linspace(1,10,21)
        computed = stacking.profileModelNadathur(r,modelArgs[0],\
            modelArgs[1],modelArgs[2],modelArgs[3])
        referenceFile = self.dataFolder + self.test_subfolder + \
            "profileModelNadathur_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_matchDistributionFilter(self):
        print("Running stacking.matchDistributionFilter test...")
        np.random.seed(1000)
        lambdas1 = np.random.random(1000)
        lambdas2 = np.random.random(1000)
        computed = stacking.matchDistributionFilter(lambdas1,lambdas2,\
            randomSeed=1000)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "matchDistributionFilter_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_profileIntregral(self):
        print("Running stacking.profileIntegral test...")
        [rBins,rBinCentres,nbarj,sigmabarj,nbar] = tools.loadPickle(\
            self.dataFolder + "profile_examples.p")
        computed = stacking.profileIntegral(rBins,nbarj)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "profileIntegral_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def snaptest_getProfileIntegralAndDeltaAverage(self,snapn,rBins,\
            antihaloRadii,ahCentresList,pairCounts,volumesList,deltaBar,\
            nbar,condition):
        print("Running stacking.getProfileIntegralAndDeltaAverage test...")
        computed = stacking.getProfileIntegralAndDeltaAverage(snapn,rBins,\
            antihaloRadii,ahCentresList,pairCounts,volumesList,deltaBar,\
            nbar,condition)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "getProfileIntegralAndDeltaAverage_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_deltaBarEffAverageAndError(self):
        print("Running stacking.deltaBarEffAverageAndError test...")
        np.random.seed(1000)
        radii = np.random.random(1000)*30
        deltaBarEff = np.random.random(1000)*1.2 - 1.0
        condition = np.ones(1000,dtype=bool)
        computed = stacking.deltaBarEffAverageAndError(radii,\
            deltaBarEff,condition)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "deltaBarEffAverageAndError_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_deltaVError(self):
        print("Running stacking.deltaVError test...")
        deltaVList = np.array([1.0,0.2,0.8,1.2])
        computed = stacking.deltaVError(deltaVList)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "deltaVError_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_deltaVRatio(self):
        print("Running stacking.deltaVRatio test...")
        deltaVList = np.array([1.0,0.2,0.8,1.2])
        computed = stacking.deltaVRatio(deltaVList)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "deltaVRatio_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_minmax(self):
        print("Running stacking.minmax test...")
        computed = stacking.minmax([1.0,2.0,3.0])
        reference = np.array([1.0,3.0])
        self.compareToReference(computed,reference)
    def test_getRanges(self):
        print("Running stacking.getRanges test...")
        np.random.seed(1000)
        rhoAH = np.random.random((1000,30))*1.2
        rhoZV = np.random.random((1000,30))*1.2
        rBins = np.linspace(0,3,31)
        rBinCentres = (rBins[0:-1] + rBins[1:])/2
        computed = stacking.getRanges(rhoAH,rhoZV,rBinCentres)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "getRanges_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    @unittest.skip("Seems to have a system-dependence in external dependency.")
    def test_testGammaAcrossBins(self):
        print("Running stacking.testGammaAcrossBins test...")
        np.random.seed(1000)
        rhoAH = np.random.random((1000,30))*1.2
        computed = stacking.testGammaAcrossBins(rhoAH)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "testGammaAcrossBins_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_testNormAcrossBins(self):
        print("Running stacking.testNormAcrossBins test...")
        np.random.seed(1000)
        rhoAH = np.random.random((1000,30))*1.2
        computed = stacking.testNormAcrossBins(rhoAH)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "testNormAcrossBins_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_computeMeanStacks(self):
        print("Running stacking.computeMeanStacks test...")
        snapNumList = [2791,3250]
        rMin=5
        rMax=25
        mMin = 1e14
        mMax = 1e15
        snapname = "gadget_full_forward/snapshot_001"
        snapnameRev = "gadget_full_reverse/snapshot_001"
        samplesFolder = self.dataFolder + "reference_constrained/"
        # Load snapshots:
        snapList =  [pynbody.load(samplesFolder + "sample" + \
            str(snapNum) + "/" + snapname) for snapNum in snapNumList]
        snapListRev =  [pynbody.load(samplesFolder + "sample" + \
            str(snapNum) + "/" + snapnameRev) for snapNum in snapNumList]
        boxsize = snapList[0].properties['boxsize'].ratio("Mpc a h**-1")
        # Load reference antihalo data:
        ahProps = [tools.loadPickle(snap.filename + ".AHproperties.p") \
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
        nbar = len(snapList[0])/boxsize**3
        rBins = np.linspace(0,3,31)
        computed = stacking.computeMeanStacks(ahCentresList,antihaloRadii,\
            antihaloMassesList,conditionList,pairCountsList,volumesList,\
            snapList,nbar,rBins,rMin,rMax,mMin,mMax)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "computeMeanStacks_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_run_snaptests(self):
        print("Running stacking snaptests...")
        # This does all the tests involving a single snapshot in a single go.
        # The reason for this is that loading the test snapshots can be quite
        # slow and expensive. What we want to do then is load them once, and 
        # share them between all the test functions.
        standard = self.dataFolder + \
            "reference_constrained/sample2791/gadget_full_forward/snapshot_001"
        reverse = self.dataFolder + \
            "reference_constrained/sample2791/gadget_full_reverse/snapshot_001"
        snapn = pynbody.load(standard)
        snapr = pynbody.load(reverse)
        hr = snapr.halos()
        hn = snapn.halos()
        voidParticles = hr[1]['iord']
        volumes = tools.zobovVolumesToPhysical(standard + ".vols",snapn,\
            dtype=np.double,offset=4)
        sortedInd = np.argsort(snapn['iord'])
        massesVoid = snapn['mass'][sortedInd[voidParticles]]
        positions = snapn['pos'][sortedInd[voidParticles],:]
        volumesVoid = volumes[sortedInd[voidParticles]]
        boxsize = snapn.properties['boxsize'].ratio("Mpc a h**-1")
        nbar = len(snapn)/boxsize**3
        ahProps = tools.loadPickle(snapn.filename + ".AHproperties.p")
        antihaloRadii = ahProps[7][0:10]
        ahCentresList = ahProps[5][0:10,:]
        pairCounts = ahProps[9][0:10]
        volumesList = ahProps[10][0:10]
        deltaBar = ahProps[12][0:10]
        rBins = np.linspace(0,3,31)
        condition = np.ones(10,dtype=bool)
        # Run all the snaptests:
        self.snaptest_getProfileIntegralAndDeltaAverage(snapn,rBins,\
            antihaloRadii,ahCentresList,pairCounts,volumesList,deltaBar,\
            nbar,condition)
        self.snaptest_centralDensityNN(ahCentresList,positions,massesVoid,\
            volumesVoid,boxsize)
        self.snaptest_centralDensity(ahCentresList,antihaloRadii,\
            positions,volumesVoid,massesVoid,boxsize)
        self.snaptest_lambdaVoid(sortedInd[voidParticles],volumes,nbar,\
            antihaloRadii)
        self.snaptest_meanDensityContrast(sortedInd[voidParticles],\
            volumes,nbar)
        # This test not working yet - find out why!
        #self.snaptest_stackVoidVelocitiesWithFilter(ahCentresList,\
        #    antihaloRadii,snapn,volumes)
        self.snaptest_stackVoidsWithFilter(ahCentresList,\
            antihaloRadii,snapn)
        # This test not working yet - find out why!
        #self.snaptest_stackScaledVoidsVelocities(ahCentresList,\
        #    antihaloRadii,snapn,volumes)
        self.snaptest_stackScaledVoids(ahCentresList,\
            antihaloRadii,snapn)
        #self.snaptest_getRadialVelocityAverages(ahCentresList,\
        #    antihaloRadii,snapn)
        self.snaptest_getPairCounts(ahCentresList,antihaloRadii,snapn)


@unittest.skip("Tests in development")
class test_halos(test_base):
    def setUp(self):
        self.dataFolder=dataFolder
        self.test_subfolder = "function_tests_halos/"
        self.rtol=1e-5
        self.atol=1e-8
        self.generateTestData = generateMode
    def snaptest_getOverdensity(self,radius,centre,snapn,tree):
        print("Running halos.getOverdensity test...")
        computed = halos.getOverdensity(radius,centre,snapn,tree)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "getOverdensity_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def snaptest_findVirialRadius(self,centre,snapn,tree):
        print("Running halos.findVirialRadius test...")
        computed = halos.findVirialRadius(centre,snapn,tree)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "findVirialRadius_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def snaptest_getCorrespondingCentres(self,snapn,snapr,largeHalos,\
            hn,boxsize):
        print("Running halos.getCorrespondingCentres test...")
        computed = halos.getCorrespondingCentres(snapn,snapr,largeHalos,\
            hn=hn,boxsize=boxsize)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "getCorrespondingCentres_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_massCentreAboutPoint(self):
        print("Running halos.getCorrespondingCentres test...")
        np.random.seed(1000)
        positions = np.random.random((10000,3))*1000
        point = np.array([500,500,500])
        boxsize = 1000
        computed = halos.massCentreAboutPoint(point,positions,boxsize)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "massCentreAboutPoint_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def snaptest_getHaloMass(self,posAll,haloPos,nbar,Om,partMasses,boxsize):
        print("Running halos.getHaloMass test...")
        computed = halos.getHaloMass(posAll,haloPos,nbar,Om,partMasses,\
            boxsize=boxsize)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "getHaloMass_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_run_snaptests(self):
        print("Running halos snaptests...")
        # This does all the tests involving a single snapshot in a single go.
        # The reason for this is that loading the test snapshots can be quite
        # slow and expensive. What we want to do then is load them once, and 
        # share them between all the test functions.
        standard = self.dataFolder + \
            "reference_constrained/sample2791/gadget_full_forward/snapshot_001"
        reverse = self.dataFolder + \
            "reference_constrained/sample2791/gadget_full_reverse/snapshot_001"
        snapn = pynbody.load(standard)
        snapr = pynbody.load(reverse)
        hr = snapr.halos()
        hn = snapn.halos()
        voidParticles = hr[1]['iord']
        volumes = tools.zobovVolumesToPhysical(standard + ".vols",snapn,\
            dtype=np.double,offset=4)
        sortedInd = np.argsort(snapn['iord'])
        massesVoid = snapn['mass'][sortedInd[voidParticles]]
        postions = snapn['pos'][sortedInd[voidParticles],:]
        volumesVoid = volumes[sortedInd[voidParticles]]
        boxsize = snapn.properties['boxsize'].ratio("Mpc a h**-1")
        nbar = len(snapn)/boxsize**3
        ahProps = tools.loadPickle(snapn.filename + ".AHproperties.p")
        antihaloRadii = ahProps[7][0:1000]
        ahCentresList = ahProps[5][0:1000,:]
        haloCentresList = ahProps[0][0:1000,:]
        pairCounts = ahProps[9][0:1000]
        volumesList = ahProps[10][0:1000]
        deltaBar = ahProps[12][0:1000]
        rBins = np.linspace(0,3,31)
        condition = np.ones(1000,dtype=bool)
        tree = tools.getKDTree(snapn)
        largeHalos = np.arange(0,20)
        # Run all the snaptests:
        self.snaptest_getOverdensity(10,haloCentresList[0],snapn,tree)
        self.snaptest_findVirialRadius(haloCentresList[0],snapn,tree)
        self.snaptest_getCorrespondingCentres(snapn,snapr,largeHalos,hn,\
            boxsize)
        self.snaptest_getHaloMass(snapn['pos'],haloCentresList[0],nbar,0.3111,\
            snapn['mass'],boxsize)


@unittest.skip("Tests in development")
class test_simulation_tools(test_base):
    def setUp(self):
        self.dataFolder=dataFolder
        self.test_subfolder = "function_tests_simulation_tools/"
        self.rtol=1e-5
        self.atol=1e-8
        self.generateTestData = generateMode
    def test_eulerToZ(self):
        print("Running simulation_tools.eulerToZ test...")
        np.random.seed(1000)
        positions = np.random.random((10000,3))*1000 - 1000
        vel = np.random.random((10000,3))*400 - 400
        cosmo = astropy.cosmology.LambdaCDM(70,0.3111,1.0 - 0.3111)
        boxsize = 1000
        h = 0.7
        computed = simulation_tools.eulerToZ(positions,vel,cosmo,boxsize,h)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "eulerToZ_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def snaptest_getGriddedDensity(self,snapn):
        print("Running simulation_tools.getGriddedDensity test...")
        N = 64
        computed = simulation_tools.getGriddedDensity(snapn,N,\
            redshiftSpace=True)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "getGriddedDensity_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_pointsInRangeWithWrap(self):
        print("Running simulation_tools.pointsInRangeWithWrap test...")
        np.random.seed(1000)
        positions = np.random.random((10000,3))*1000 - 1000
        boxsize = 1000.0
        computed = simulation_tools.pointsInRangeWithWrap(positions,\
            [-100,100],boxsize=boxsize)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "pointsInRangeWithWrap_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_pointsInBoundedPlaneWithWrap(self):
        print("Running simulation_tools.pointsInBoundedPlaneWithWrap test...")
        np.random.seed(1000)
        positions = np.random.random((10000,3))*1000 - 1000
        boxsize = 1000.0
        computed = simulation_tools.pointsInBoundedPlaneWithWrap(positions,\
            [-100,100],[-100,100],boxsize=boxsize)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "pointsInBoundedPlaneWithWrap_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_getGriddedGalCount(self):
        print("Running simulation_tools.getGriddedGalCount test...")
        np.random.seed(1000)
        positions = np.random.random((10000,3))*1000 - 1000
        N = 64
        boxsize = 1000.0
        computed = simulation_tools.getGriddedGalCount(positions,N,boxsize)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "getGriddedGalCount_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_biasOld(self):
        print("Running simulation_tools.biasOld test...")
        rhoArray = np.linspace(0,1.2,101)
        params = [1.0,0.5,0.5,0.5]
        computed = simulation_tools.biasOld(rhoArray,params)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "biasOld_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_biasNew(self):
        print("Running simulation_tools.biasNew test...")
        biasParam = tools.loadPickle(self.dataFolder + "bias_param_example.p")
        rhoArray = np.linspace(0,1.2,101)
        params = biasParam[0][0,:]
        computed = simulation_tools.biasNew(rhoArray,params)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "biasNew_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_ngPerLBin(self):
        print("Running simulation_tools.ngPerLBin test...")
        # Not really ideal, since this is a massive data file. Can we find a
        # smaller test example to make this more lightweight?
        referenceMaskList = tools.loadPickle(\
            self.dataFolder + "survey_mask_ref.p")
        np.random.seed(1000)
        maskRandom = np.random.random((16,256**3))
        biasParam = tools.loadPickle(self.dataFolder + "bias_param_example.p")
        biasData = h5py.File(self.dataFolder + \
            "reference_constrained/sample2791/mcmc_2791.h5",'r')
        biasParam = np.array([[biasData['scalars']['galaxy_bias_' + \
            str(k)][()] for k in range(0,16)]])
        mcmcDen = 1.0 + biasData['scalars']['BORG_final_density'][()]
        N = 64
        mcmcDen = tools.downsample(mcmcDen,int(256/N))
        mcmcDenLin = np.reshape(mcmcDen,N**3)
        mcmcDen_r = np.reshape(mcmcDenLin,(N,N,N),order='F')
        mcmcDenLin_r = np.reshape(mcmcDen_r,N**3)
        computed = simulation_tools.ngPerLBin(\
                biasParam,return_samples=True,mask=referenceMaskList[0],\
                accelerate=True,N=N,\
                delta = [mcmcDenLin_r],contrast=False,sampleList=[0],\
                beta=biasParam[:,:,1],rhog = biasParam[:,:,3],\
                epsg=biasParam[:,:,2],\
                nmean=biasParam[:,:,0],biasModel = simulation_tools.biasNew)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "ngPerLBin_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def snaptest_matchClustersAndHalos(self,ahProps,hn,boxsize):
        print("Running simulation_tools.matchClustersAndHalos test...")
        haloCentresList = tools.remapAntiHaloCentre(ahProps[0][0:1000,:],\
            boxsize)
        haloMasseslist = ahProps[1][0:1000]
        [combinedAbellN,combinedAbellPos,abell_nums] = \
            real_clusters.getCombinedAbellCatalogue(\
            catFolder = self.dataFolder)
        tmppFile=self.dataFolder + "2mpp_data/2MPP.txt"
        cosmo = astropy.cosmology.LambdaCDM(70,0.3,0.7)
        tmpp = np.loadtxt(tmppFile)
        # Comoving distance in Mpc/h
        h = cosmo.h
        d = cosmo.comoving_distance(tmpp[:,3]).value*h
        dL = cosmology.comovingToLuminosity(d[np.where(d > 0)],cosmo)
        posD = np.where(d > 0)[0]
        # Angular co-ordinates:
        theta = tmpp[:,2]
        phi = tmpp[:,1]
        # Cartesian positions:
        Z = d*np.sin(theta)
        X = d*np.cos(theta)*np.cos(phi)
        Y = d*np.cos(theta)*np.sin(phi)
        pos2mpp = np.vstack((X,Y,Z)).T[posD]
        computed = simulation_tools.matchClustersAndHalos(combinedAbellPos,\
            haloCentresList,haloMasseslist,boxsize,pos2mpp)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "matchClustersAndHalos_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def snaptest_getHaloCentresAndMassesFromCatalogue(self,hn,boxsize):
        print("Running simulation_tools.matchClustersAndHalos test...")
        computed = simulation_tools.getHaloCentresAndMassesFromCatalogue(\
            hn,boxsize=boxsize)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "getHaloCentresAndMassesFromCatalogue_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def snaptest_getHaloCentresAndMassesRecomputed(self,hn,boxsize):
        print("Running simulation_tools.matchClustersAndHalos test...")
        computed = simulation_tools.getHaloCentresAndMassesRecomputed(\
            hn,boxsize=boxsize)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "getHaloCentresAndMassesRecomputed_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_getAllHaloCentresAndMasses(self):
        print("Running simulation_tools.getAllHaloCentresAndMasses test...")
        snapNumList = [2791,3250]
        rMin=5
        rMax=25
        snapname = "gadget_full_forward/snapshot_001"
        snapnameRev = "gadget_full_reverse/snapshot_001"
        samplesFolder = self.dataFolder + "reference_constrained/"
        # Load snapshots:
        snapList =  [pynbody.load(samplesFolder + "sample" + \
            str(snapNum) + "/" + snapname) for snapNum in snapNumList]
        boxsize = snapList[0].properties['boxsize'].ratio("Mpc a h**-1")
        computed = simulation_tools.getAllHaloCentresAndMasses(snapList,\
            boxsize,\
            function = simulation_tools.getHaloCentresAndMassesFromCatalogue)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "getAllHaloCentresAndMasses_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def snaptest_getClusterCentres(self,centre,snapn,snapPath):
        print("Running simulation_tools.getClusterCentres test...")
        computed = simulation_tools.getClusterCentres(centre,snap=snapn,\
            snapPath=snapPath)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "getClusterCentres_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_snap_tests(self):
        print("Running simulation_tools snap test...")
        standard = self.dataFolder + \
            "reference_constrained/sample2791/gadget_full_forward/snapshot_001"
        snapn = pynbody.load(standard)
        hn = snapn.halos()
        boxsize = snapn.properties['boxsize'].ratio("Mpc a h**-1")
        ahProps = tools.loadPickle(snapn.filename + ".AHproperties.p")
        [combinedAbellN,combinedAbellPos,abell_nums] = \
            real_clusters.getCombinedAbellCatalogue(\
            catFolder = self.dataFolder)
        self.snaptest_getHaloCentresAndMassesFromCatalogue(hn,boxsize)
        self.snaptest_matchClustersAndHalos(ahProps,hn,boxsize)
        self.snaptest_getGriddedDensity(snapn)
        self.snaptest_getHaloCentresAndMassesRecomputed(hn,boxsize)
        self.snaptest_getClusterCentres(combinedAbellPos[0],snapn,standard)
    def test_get_random_centres_and_densities(self):
        snapNumList = [2791,3250]
        snapname = "gadget_full_forward/snapshot_001"
        samplesFolder = self.dataFolder + "reference_constrained/"
        snapList =  [pynbody.load(samplesFolder + "sample" + \
            str(snapNum) + "/" + snapname) for snapNum in snapNumList]
        computed = simulation_tools.get_random_centres_and_densities(
            135,snapList,seed=1000,nRandCentres = 100)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "get_random_centres_and_densities_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_get_mcmc_supervolume_densities(self):
        snapNumList = [2791,3250]
        snapname = "gadget_full_forward/snapshot_001"
        samplesFolder = self.dataFolder + "reference_constrained/"
        snapList =  [pynbody.load(samplesFolder + "sample" + \
            str(snapNum) + "/" + snapname) for snapNum in snapNumList]
        computed = simulation_tools.get_mcmc_supervolume_densities(
            snapList,r_sphere=135)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "get_mcmc_supervolume_densities_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_get_map_from_sample(self):
        deltaMCMCList = tools.loadPickle(self.dataFolder 
                                         + self.test_subfolder 
                                         + "delta_list.p")
        computed = simulation_tools.get_map_from_sample(deltaMCMCList)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "get_map_from_sample_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_getNonOverlappingCentres(self):
        [randCentres,randOverDen] = tools.loadPickle(
            self.dataFolder + self.test_subfolder + \
            "get_random_centres_and_densities_ref.p")
        snapNumList = [2791,3250]
        snapname = "gadget_full_forward/snapshot_001"
        samplesFolder = self.dataFolder + "reference_constrained/"
        snapList =  [pynbody.load(samplesFolder + "sample" + \
            str(snapNum) + "/" + snapname) for snapNum in snapNumList]
        rSep = 2*135
        boxsize = snapList[0].properties['boxsize'].ratio("Mpc a h**-1")
        centresListAll = [randCentres for ns in range(0,len(snapList))]
        computed = simulation_tools.getNonOverlappingCentres(
            centresListAll,rSep,boxsize,returnIndices=True)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "getNonOverlappingCentres_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_getDistanceBetweenCentres(self):
        [randCentres,randOverDen] = tools.loadPickle(
            self.dataFolder + self.test_subfolder + \
            "get_random_centres_and_densities_ref.p")
        boxsize = 677.7
        computed = simulation_tools.getDistanceBetweenCentres(randCentres[0],
                                                              randCentres[1],
                                                              boxsize)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "getDistanceBetweenCentres_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)

@unittest.skip("Tests in development")
class test_snapedit(test_base):
    def setUp(self):
        self.dataFolder=dataFolder
        self.test_subfolder = "function_tests_snapedit/"
        self.rtol=1e-5
        self.atol=1e-8
        self.generateTestData = generateMode
    def test_gridTest(self):
        print("Running snapedit.gridTest test...")
        ics = pynbody.load(self.dataFolder + \
            "/initial_conditions_examples/gadget_ic_lowres_for.gadget2")
        grid = snapedit.centroid(ics,256)
        computed = snapedit.gridTest(ics,grid)
        reference = (0,1,2)
        self.assertTrue(computed == reference)
    def test_wrap(self):
        print("Running snapedit.wrap test...")
        np.random.seed(1000)
        positions = np.random.random((3,1000))*1000 - 1000
        computed = snapedit.wrap(positions,1000)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "wrap_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_unwrap(self):
        print("Running snapedit.unwrap test...")
        np.random.seed(1000)
        positions = np.random.random((1000,3))*1000
        computed = snapedit.unwrap(positions,1000)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "unwrap_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_index(self):
        print("Running snapedit.index test...")
        np.random.seed(1000)
        positions = np.random.random((1000,3))*1000
        computed = snapedit.index(positions,1000,256)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "index_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_grid_offset(self):
        print("Running snapedit.grid_offset test...")
        np.random.seed(1000)
        positions = np.random.random((1000,3))*1000
        computed = snapedit.grid_offset(positions,1000,256)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "grid_offset_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_zeldovich_factor(self):
        print("Running snapedit.zeldovich_factor test...")
        ics = pynbody.load(self.dataFolder + \
            "/initial_conditions_examples/gadget_ic_lowres_for.gadget2")
        computed = snapedit.zeldovich_factor(ics)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "zeldovich_factor_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_getGridFromZeldovich(self):
        print("Running snapedit.getGridFromZeldovich test...")
        ics = pynbody.load(self.dataFolder + \
            "/initial_conditions_examples/gadget_ic_lowres_for.gadget2")
        computed = snapedit.getGridFromZeldovich(ics,1.0)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "getGridFromZeldovich_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_zeldovich_test(self):
        print("Running snapedit.zeldovich_test test...")
        ics = pynbody.load(self.dataFolder + \
            "/initial_conditions_examples/gadget_ic_lowres_for.gadget2")
        computed = snapedit.zeldovich_test(ics)
        reference = True
        self.assertTrue(computed == reference)
    def test_f1(self):
        print("Running snapedit.f1 test...")
        zrange = np.linspace(0,200,101)
        computed = snapedit.f1(0.3,0.7,zrange)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "f1_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_lin2coord(self):
        print("Running snapedit.lin2coord test...")
        np.random.seed(1000)
        N = 256
        intsList = np.random.randint(N**3,size=100)
        computed = snapedit.lin2coord(intsList,N)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "lin2coord_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_coord2lin(self):
        print("Running snapedit.coord2lin test...")
        np.random.seed(1000)
        N = 256
        reference = np.random.randint(N**3,size=100)
        coord = snapedit.lin2coord(reference,N)
        computed = snapedit.coord2lin(coord,N)
        self.compareToReference(computed,reference)
    def test_gridList(self):
        print("Running snapedit.gridList test...")
        N = 256
        computed = snapedit.gridList(N)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "gridList_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_gridListInverted(self):
        print("Running snapedit.gridListInverted test...")
        N = 256
        computed = snapedit.gridListInverted(N)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "gridListInverted_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_gridListPermutation(self):
        print("Running snapedit.gridListPermutation test...")
        N = 256
        computed = snapedit.gridListPermutation(N)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "gridListPermutation_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_centroid(self):
        print("Running snapedit.centroid test...")
        ics = pynbody.load(self.dataFolder + \
            "/initial_conditions_examples/gadget_ic_lowres_for.gadget2")
        N = 256
        computed = snapedit.centroid(ics,N)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "centroid_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_reverseICs(self):
        print("Running snapedit.reverseICs test...")
        ics = pynbody.load(self.dataFolder + \
            "/initial_conditions_examples/gadget_ic_lowres_for.gadget2")
        computed = np.array(snapedit.reverseICs(ics)['pos'])
        referenceFile = self.dataFolder + self.test_subfolder + \
            "reverseICs_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_zones2Particles(self):
        print("WARNING - TEST test_zones2Particles not implemented yet.")
        self.assertTrue(True)
    def test_voids2Zones(self):
        print("WARNING - TEST test_voids2Zones not implemented yet.")
        self.assertTrue(True)
    def test_importVoidList(self):
        print("WARNING - TEST importVoidList not implemented yet.")
        self.assertTrue(True)
    def test_getIntersectingVoids(self):
        print("WARNING - TEST getIntersectingVoids not implemented yet.")
        self.assertTrue(True)
    def test_getParticlesInHalos(self):
        print("Running snapedit.getParticlesInHalos test...")
        standard = self.dataFolder + \
            "reference_constrained/sample2791/gadget_full_forward/snapshot_001"
        snapn = pynbody.load(standard)
        hn = snapn.halos()
        computed = np.array(snapedit.getParticlesInHalos(snapn,hn)['pos'])
        referenceFile = self.dataFolder + self.test_subfolder + \
            "getParticlesInHalos_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_reorderSnap(self):
        print("Running snapedit.reorderSnap test...")
        standard = self.dataFolder + \
            "reference_constrained/sample2791/gadget_full_forward/snapshot_001"
        snapn = pynbody.load(standard)
        snew = snapedit.reorderSnap(standard)
        computed = np.array(snew['pos'])
        referenceFile = self.dataFolder + self.test_subfolder + \
            "reorderSnap_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_newSnap(self):
        print("Running snapedit.newSnap test...")
        standard = self.dataFolder + \
            "reference_constrained/sample2791/gadget_full_forward/snapshot_001"
        snapn = pynbody.load(standard)
        snew = snapedit.newSnap(None,snapn['pos'],snapn['vel'])
        computed = [snew['pos'],snew['vel']]
        referenceFile = self.dataFolder + self.test_subfolder + \
            "newSnap_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_periodicPadding(self):
        print("Running snapedit.periodicPadding test...")
        standard = self.dataFolder + \
            "reference_constrained/sample2791/gadget_full_forward/snapshot_001"
        snapn = pynbody.load(standard)
        computed = snapedit.periodicPadding(snapn,20,677.7)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "periodicPadding_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_getDisplacements(self):
        print("Running snapedit.getDisplacements test...")
        ics = pynbody.load(self.dataFolder + \
            "/initial_conditions_examples/gadget_ic_lowres_for.gadget2")
        computed = snapedit.getDisplacements(ics)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "getDisplacements_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_extrapolateZeldovich(self):
        print("Running snapedit.extrapolateZeldovich test...")
        ics = pynbody.load(self.dataFolder + \
            "/initial_conditions_examples/gadget_ic_lowres_for.gadget2")
        computed = snapedit.extrapolateZeldovich(ics)['pos']
        referenceFile = self.dataFolder + self.test_subfolder + \
            "extrapolateZeldovich_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)


#@unittest.skip("Tests in development")
# Unit tests for paper plot scripts:
class test_catalogue_code(test_base):
    def setUp(self):
        self.dataFolder=dataFolder
        self.test_subfolder = "function_tests_catalogue/"
        self.rtol=1e-5
        self.atol=1e-8
        self.generateTestData = generateMode
        # Some things shared by the catalogue tests:
        self.snapNumList = [2791,3250,5511]
        self.samplesFolder = dataFolder + "reference_constrained/"
        self.snapname = "gadget_full_forward/snapshot_001"
        self.snapnameRev = "gadget_full_reverse/snapshot_001"
        self.snapNameList = [self.samplesFolder + "sample" + \
            str(snapNum) + "/" + self.snapname for snapNum in self.snapNumList]
        self.snapNameListRev = [self.samplesFolder + "sample" + \
            str(snapNum) + "/" + self.snapnameRev \
            for snapNum in self.snapNumList]
        self.snapList =  [pynbody.load(self.samplesFolder + "sample" + \
            str(snapNum) + "/" + self.snapname) for snapNum in self.snapNumList]
        self.ahPropsConstrained = [tools.loadPickle(snap.filename + \
            ".AHproperties.p") \
            for snap in self.snapList]
        self.antihaloRadii = [props[7] for props in self.ahPropsConstrained]
        self.antihaloMasses = [props[3] \
            for props in self.ahPropsConstrained]
        self.ahCentresList = [props[5] \
            for props in self.ahPropsConstrained]
        self.vorVols = [props[4] for props in self.ahPropsConstrained]
        self.numCats = 3
    def test_constructAntihaloCatalogue(self):
        # Test without N-way matching:
        referenceFile = self.dataFolder + self.test_subfolder + \
            "constructAntihaloCatalogue_ref.p"
        computed = generator.constructAntihaloCatalogue(\
            [2791,3250,5511],\
            samplesFolder="data_for_tests/reference_constrained/",\
            verbose=False,rSphere=135,max_index=None,thresh=0.5,\
            snapname = "gadget_full_forward/snapshot_001",\
            snapnameRev = "gadget_full_reverse/snapshot_001",\
            fileSuffix= '',matchType='distance',crossMatchQuantity='radius',\
            crossMatchThreshold = 0.9,distMax=0.5,sortMethod='ratio',\
            blockDuplicates=True,twoWayOnly = True,\
            snapList=None,snapListRev=None,ahProps=None,hrList=None,\
            rMin = 5,rMax = 30,mode="fractional",massRange = None,\
            snapSortList = None,overlapList = None,NWayMatch = False,\
            additionalFilters = None,refineCentres=False)
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_constructAntihaloCatalogueNway(self):
        # Test with N-way matching:
        referenceFile = self.dataFolder + self.test_subfolder + \
            "constructAntihaloCatalogueNway_ref.p"
        computed = generator.constructAntihaloCatalogue(\
            [2791,3250,5511],\
            samplesFolder="data_for_tests/reference_constrained/",\
            verbose=False,rSphere=135,max_index=None,thresh=0.5,\
            snapname = "gadget_full_forward/snapshot_001",\
            snapnameRev = "gadget_full_reverse/snapshot_001",\
            fileSuffix= '',matchType='distance',crossMatchQuantity='radius',\
            crossMatchThreshold = 0.9,distMax=0.5,sortMethod='ratio',\
            blockDuplicates=True,twoWayOnly = True,\
            snapList=None,snapListRev=None,ahProps=None,hrList=None,\
            rMin = 5,rMax = 30,mode="fractional",massRange = None,\
            snapSortList = None,overlapList = None,NWayMatch = True,\
            additionalFilters = None,refineCentres=False)
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_getSNRFilterFromChainFile(self):
        chainFile = self.dataFolder + "chain_properties.p"
        computed = generator.getSNRFilterFromChainFile(chainFile,10,\
            self.snapNameList,677.7,Nden = 256,allProps=None)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "getSNRFilterFromChainFile_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_getFinalCatalogueAlphaShapes(self):
        # Not really a good way to test alpha-shapes yet...
        print("WARNING - TEST NOT IMPLEMENTED YET FOR getFinalCatalogueAlphaShapes")
        self.assertTrue(True)
    def test_getAntihaloSkyPlotData(self):
        # Not really a good way to test alpha-shapes yet...
        print("WARNING - TEST NOT IMPLEMENTED YET FOR getAntihaloSkyPlotData")
        self.assertTrue(True)
    @unittest.skip("Under development")
    def test_getMatchPynbody(self):
        [snap1,snap2] = [self.snapList[0],self.snapList[1]]
        [cat1,cat2] = [snap.halos() for snap in [snap1,snap2]]
        [quantity1,quantity2] = [self.antihaloRadii[k] for k in range(0,2)]
        computed = generator.getMatchPynbody(snap1,snap2,cat1,cat2,\
            quantity1,quantity2,max_index = 200,threshold = 0.5,\
            quantityThresh=0.5,fractionType='normal')
        referenceFile = self.dataFolder + self.test_subfolder + \
            "getMatchPynbody_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_getMatchDistance(self):
        [snap1,snap2] = [self.snapList[0],self.snapList[1]]
        [centres1,centres2] = [self.ahCentresList[k] for k in range(0,2)]
        [quantity1,quantity2] = [self.antihaloRadii[k] for k in range(0,2)]
        computed = generator.getMatchDistance(snap1,snap2,centres1,centres2,\
            quantity1,quantity2,tree1=None,tree2=None,distMax = 20.0,\
            max_index=200,quantityThresh=0.5,sortMethod='distance',\
            mode="fractional",sortQuantity = 0,cat1=None,cat2=None,\
            volumes1=None,volumes2=None,overlap = None)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "getMatchDistance_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_isInZoA(self):
        computed = generator.isInZoA(np.array([0,0,0]),inUnits="equatorial",\
            galacticCentreZOA = [-30,30],bRangeCentre = [-10,10],\
            bRange = [-5,5])
        referenceFile = self.dataFolder + self.test_subfolder + \
            "isInZoA_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    @unittest.skip("Under development")
    def test_overlapMap(self):
        [snap1,snap2] = [self.snapList[0],self.snapList[1]]
        [cat1,cat2] = [snap.halos() for snap in [snap1,snap2]]
        [volumes1,volumes2] = [self.vorVols[k] for k in range(0,2)]
        computed = generator.overlapMap(cat1,cat2,volumes1,volumes2,
            checkFirst = False,verbose=False)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "overlapMap_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_checkOverlap(self):
        list1 = [1,2,3]
        list2 = [3,4,5]
        list3 = [4,5,6]
        self.assertTrue(checkOverlap(list1,list2))
        self.assertFalse(checkOverlap(list1,list3))
    def test_linearFromIJ(self):
        self.assertTrue(linearFromIJ(2,6,20) == 40)
    def test_getPoissonSamples(self):
        computed = generator.getPoissonSamples(100,100,seed = 1000)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "getPoissonSamples_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_estimatePoissonRatioErrorbarMonteCarlo(self):
        computed = generator.estimatePoissonRatioErrorbarMonteCarlo(100,100,\
            errorVal = 0.67,seed = 1000,nSamples=1000,returnSamples=False)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "estimatePoissonRatioErrorbarMonteCarlo_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_getMeanCentresFromCombinedCatalogue(self):
        # Get Data;
        rMin = 5
        rMax = 30
        boxsize = 677.7
        combinedCat = tools.loadPickle(self.dataFolder + self.test_subfolder + \
            "constructAntihaloCatalogue_ref.p")
        antihaloCentres = [tools.remapAntiHaloCentre(props[5],boxsize) \
            for props in self.ahPropsConstrained]
        [centresListShort,centralAntihalos,sortedList,ahCounts,max_index] = \
            tools.loadPickle(self.dataFolder + \
            self.test_subfolder + "computeShortCentresList_ref.p")
        # Test:
        computed = generator.getMeanCentresFromCombinedCatalogue(\
            combinedCat[0],centresListShort,returnError=False,boxsize=boxsize)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "getMeanCentresFromCombinedCatalogue_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_computeShortCentresList(self):
        antihaloCentres = [tools.remapAntiHaloCentre(props[5],677.7) \
            for props in self.ahPropsConstrained]
        rMin = 5
        rMax = 30
        computed = generator.computeShortCentresList(self.snapNumList,\
            antihaloCentres,self.antihaloRadii,self.antihaloMasses,135,\
            rMin,rMax,massRange=None,additionalFilters=None,\
            sortBy = "mass",max_index=None)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "computeShortCentresList_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_getShortenedQuantity(self):
        [centresListShort,centralAntihalos,sortedList,ahCounts,max_index] = \
            tools.loadPickle(self.dataFolder + \
            self.test_subfolder + "computeShortCentresList_ref.p")
        computed = generator.getShortenedQuantity(self.antihaloRadii,\
            centralAntihalos,centresListShort,sortedList,ahCounts,max_index)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "getShortenedQuantity_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_loadCatalogueData(self):
        [snapList,snapListRev,boxsize,ahProps,antihaloCentres,\
            antihaloMasses,antihaloRadii,snapSortList,volumesList,hrList] = \
            generator.loadCatalogueData(self.snapNameList,self.snapNameListRev,
                None,"mass",None,None,verbose=False)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "loadCatalogueData_ref.p"
        computed = [boxsize,ahProps,antihaloCentres,antihaloMasses,\
            antihaloRadii,snapSortList,volumesList]
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference,filterNan=True)
    def test_constructShortenedCatalogues(self):
        [snapList,snapListRev,boxsize,ahProps,antihaloCentres,\
            antihaloMasses,antihaloRadii,snapSortList,volumesList,hrList] = \
            generator.loadCatalogueData(self.snapNameList,self.snapNameListRev,
                None,"mass",None,None,verbose=False)
        [centresListShort,centralAntihalos,sortedList,ahCounts,max_index] = \
            tools.loadPickle(self.dataFolder + \
            self.test_subfolder + "computeShortCentresList_ref.p")
        computed = generator.constructShortenedCatalogues(3,"distance",\
            "mass",hrList,centralAntihalos,sortedList)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "constructShortenedCatalogues_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_getOneWayMatchesAllCatalogues(self):
        [snapList,snapListRev,boxsize,ahProps,antihaloCentres,\
            antihaloMasses,antihaloRadii,snapSortList,volumesList,hrList] = \
            generator.loadCatalogueData(self.snapNameList,self.snapNameListRev,
                None,"mass",None,None,verbose=False)
        [centresListShort,centralAntihalos,sortedList,ahCounts,max_index] = \
            tools.loadPickle(self.dataFolder + \
            self.test_subfolder + "computeShortCentresList_ref.p")
        treeList = [scipy.spatial.cKDTree(\
            snapedit.wrap(centres,boxsize),boxsize=boxsize) \
            for centres in centresListShort]
        hrListCentral = tools.loadPickle(self.dataFolder + \
            self.test_subfolder + "constructShortenedCatalogues_ref.p")
        quantityListRad =  generator.getShortenedQuantity(antihaloRadii,\
            centralAntihalos,centresListShort,sortedList,ahCounts,max_index)
        quantityListMass = generator.getShortenedQuantity(antihaloMasses,\
            centralAntihalos,centresListShort,sortedList,ahCounts,max_index)
        computed = generator.getOneWayMatchesAllCatalogues(3,"distance",\
            snapListRev,hrListCentral,centresListShort,quantityListRad,\
            max_index,0.5,0.9,ahCounts,quantityListRad,quantityListMass,\
            'radius',treeList,20.0,"ratio","fractional",volumesList)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "getOneWayMatchesAllCatalogues_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_getTwoWayMatches(self):
        [oneWayMatchesAllCatalogues,matchArrayList,allCandidates,\
            allRatios,allDistances] = tools.loadPickle(self.dataFolder + \
                self.test_subfolder + "getOneWayMatchesAllCatalogues_ref.p")
        computed = catalogue.getTwoWayMatches(0,0,np.array([1,2]),3,\
            oneWayMatchesAllCatalogues,oneWayMatchesOther=None)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "getTwoWayMatches_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_getMatchCandidatesTwoCatalogues(self):
        [snapList,snapListRev,boxsize,ahProps,antihaloCentres,\
            antihaloMasses,antihaloRadii,snapSortList,volumesList,hrList] = \
            generator.loadCatalogueData(self.snapNameList,self.snapNameListRev,
                None,"mass",None,None,verbose=False)
        hrListCentral = tools.loadPickle(self.dataFolder + \
            self.test_subfolder + "constructShortenedCatalogues_ref.p")
        [centresListShort,centralAntihalos,sortedList,ahCounts,max_index] = \
            tools.loadPickle(self.dataFolder + \
            self.test_subfolder + "computeShortCentresList_ref.p")
        treeList = [scipy.spatial.cKDTree(\
            snapedit.wrap(centres,boxsize),boxsize=boxsize) \
            for centres in centresListShort]
        quantityListRad =  generator.getShortenedQuantity(antihaloRadii,\
            centralAntihalos,centresListShort,sortedList,ahCounts,max_index)
        quantityListMass = generator.getShortenedQuantity(antihaloMasses,\
            centralAntihalos,centresListShort,sortedList,ahCounts,max_index)
        computed = generator.getMatchCandidatesTwoCatalogues(0,1,"distance",\
            snapListRev,hrListCentral,centresListShort,quantityListRad,\
            max_index,0.5,0.9,quantityListRad,quantityListMass,'radius',\
            treeList,20.0,"ratio","fractional",volumesList)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "getMatchCandidatesTwoCatalogues_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    @unittest.skip("Under development")
    def test_getOverlapList(self):
        [snapList,snapListRev,boxsize,ahProps,antihaloCentres,\
            antihaloMasses,antihaloRadii,snapSortList,volumesList,hrList] = \
            generator.loadCatalogueData(self.snapNameList,self.snapNameListRev,
                None,"mass",None,None,verbose=False)
        hrListCentral = tools.loadPickle(self.dataFolder + \
            self.test_subfolder + "constructShortenedCatalogues_ref.p")
        computed = generator.getOverlapList(3,hrListCentral,volumesList)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "getOverlapList_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_matchVoidToOtherCatalogues(self):
        [oneWayMatchesAllCatalogues,matchArrayList,allCandidates,\
            allRatios,allDistances] = tools.loadPickle(self.dataFolder + \
                self.test_subfolder + "getOneWayMatchesAllCatalogues_ref.p")
        [centresListShort,centralAntihalos,sortedList,ahCounts,max_index] = \
            tools.loadPickle(self.dataFolder + \
            self.test_subfolder + "computeShortCentresList_ref.p")
        oneWayMatches = oneWayMatchesAllCatalogues[0]
        otherColumns = np.array([1,2])
        oneWayMatchesOther = oneWayMatches[:,otherColumns]
        twoWayMatch = tools.loadPickle(self.dataFolder + self.test_subfolder + \
            "getTwoWayMatches_ref.p")
        alreadyMatched = np.zeros((self.numCats,max_index),dtype=bool)
        candidateCounts = [np.zeros((self.numCats,ahCounts[l]),dtype=int) \
            for l in range(0,self.numCats)]
        for m in range(0,self.numCats):
            candidateCounts[0][m,0] = len(allCandidates[0][m][0])
        diffMap = [np.setdiff1d(np.arange(0,3),[k]) \
            for k in range(0,3)]
        finalCandidates = []
        finalCat = []
        finalRatios = []
        finalCombinatoricFrac = []
        finalDistances = []
        finalCatFrac = []
        computed = generator.matchVoidToOtherCatalogues(0,0,3,\
            otherColumns,oneWayMatchesOther,oneWayMatchesAllCatalogues,\
            twoWayMatch,allCandidates,alreadyMatched,candidateCounts,\
            False,allRatios,allDistances,diffMap,finalCandidates,\
            finalCat,finalRatios,finalDistances,finalCombinatoricFrac,\
            finalCatFrac,False,None,None,677.7,0,"distance",\
            0.9,20.0,"fractional")
        referenceFile = self.dataFolder + self.test_subfolder + \
            "matchVoidToOtherCatalogues_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_applyNWayMatching(self):
        [oneWayMatchesAllCatalogues,matchArrayList,allCandidates,\
            allRatios,allDistances] = tools.loadPickle(self.dataFolder + \
                self.test_subfolder + "getOneWayMatchesAllCatalogues_ref.p")
        [centresListShort,centralAntihalos,sortedList,ahCounts,max_index] = \
            tools.loadPickle(self.dataFolder + \
            self.test_subfolder + "computeShortCentresList_ref.p")
        oneWayMatches = oneWayMatchesAllCatalogues[0]
        otherColumns = np.array([1,2])
        oneWayMatchesOther = oneWayMatches[:,otherColumns]
        twoWayMatch = tools.loadPickle(self.dataFolder + self.test_subfolder + \
            "getTwoWayMatches_ref.p")
        alreadyMatched = np.zeros((self.numCats,max_index),dtype=bool)
        candidateCounts = [np.zeros((self.numCats,ahCounts[l]),dtype=int) \
            for l in range(0,self.numCats)]
        for m in range(0,self.numCats):
            candidateCounts[0][m,0] = len(allCandidates[0][m][0])
        diffMap = [np.setdiff1d(np.arange(0,3),[k]) \
            for k in range(0,3)]
        computed = generator.applyNWayMatching(0,0,3,\
            oneWayMatches,alreadyMatched,diffMap,allCandidates,\
            allRatios,allDistances)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "applyNWayMatching_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_followAllMatchChains(self):
        [oneWayMatchesAllCatalogues,matchArrayList,allCandidates,\
            allRatios,allDistances] = tools.loadPickle(self.dataFolder + \
                self.test_subfolder + "getOneWayMatchesAllCatalogues_ref.p")
        oneWayMatches = oneWayMatchesAllCatalogues[0]
        otherColumns = np.array([1,2])
        oneWayMatchesOther = oneWayMatches[:,otherColumns]
        diffMap = [np.setdiff1d(np.arange(0,3),[k]) \
            for k in range(0,3)]
        alreadyMatched = np.zeros((3,max_index),dtype=bool)
        computed = generator.followAllMatchChains(0,0,3,oneWayMatches,\
            alreadyMatched,diffMap,allCandidates)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "followAllMatchChains_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_computeQuantityForCandidates(self):
        [oneWayMatchesAllCatalogues,matchArrayList,allCandidates,\
            allRatios,allDistances] = tools.loadPickle(self.dataFolder + \
                self.test_subfolder + "getOneWayMatchesAllCatalogues_ref.p")
        allCands = tools.loadPickle(self.dataFolder + self.test_subfolder + \
            "followAllMatchChains_ref.p")
        diffMap = [np.setdiff1d(np.arange(0,3),[k]) \
            for k in range(0,3)]
        computed = generator.computeQuantityForCandidates(allRatios,3,\
            allCands,diffMap)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "computeQuantityForCandidates_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_getTotalNumberOfTwoWayMatches(self):
        [oneWayMatchesAllCatalogues,matchArrayList,allCandidates,\
            allRatios,allDistances] = tools.loadPickle(self.dataFolder + \
                self.test_subfolder + "getOneWayMatchesAllCatalogues_ref.p")
        [centresListShort,centralAntihalos,sortedList,ahCounts,max_index] = \
            tools.loadPickle(self.dataFolder + \
            self.test_subfolder + "computeShortCentresList_ref.p")
        oneWayMatches = oneWayMatchesAllCatalogues[0]
        otherColumns = np.array([1,2])
        oneWayMatchesOther = oneWayMatches[:,otherColumns]
        twoWayMatch = tools.loadPickle(self.dataFolder + self.test_subfolder + \
            "getTwoWayMatches_ref.p")
        alreadyMatched = np.zeros((3,max_index),dtype=bool)
        candidateCounts = [np.zeros((self.numCats,ahCounts[l]),dtype=int) \
            for l in range(0,self.numCats)]
        for m in range(0,self.numCats):
            candidateCounts[0][m,0] = len(allCandidates[0][m][0])
        diffMap = [np.setdiff1d(np.arange(0,3),[k]) \
            for k in range(0,3)]
        computed = generator.getTotalNumberOfTwoWayMatches(\
            3,diffMap,allCandidates,oneWayMatches[0])
        referenceFile = self.dataFolder + self.test_subfolder + \
            "getTotalNumberOfTwoWayMatches_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_getNumberOfTwoWayMatchesNway(self):
        allCands = tools.loadPickle(self.dataFolder + self.test_subfolder + \
            "followAllMatchChains_ref.p")
        [oneWayMatchesAllCatalogues,matchArrayList,allCandidates,\
            allRatios,allDistances] = tools.loadPickle(self.dataFolder + \
                self.test_subfolder + "getOneWayMatchesAllCatalogues_ref.p")
        diffMap = [np.setdiff1d(np.arange(0,3),[k]) \
            for k in range(0,3)]
        computed = generator.getNumberOfTwoWayMatchesNway(self.numCats,\
            allCands,allCandidates,diffMap)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "getNumberOfTwoWayMatchesNway_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_followAllMatchChains(self):
        [oneWayMatchesAllCatalogues,matchArrayList,allCandidates,\
            allRatios,allDistances] = tools.loadPickle(self.dataFolder + \
                self.test_subfolder + "getOneWayMatchesAllCatalogues_ref.p")
        [centresListShort,centralAntihalos,sortedList,ahCounts,max_index] = \
            tools.loadPickle(self.dataFolder + \
            self.test_subfolder + "computeShortCentresList_ref.p")
        oneWayMatches = oneWayMatchesAllCatalogues[0]
        otherColumns = np.array([1,2])
        oneWayMatchesOther = oneWayMatches[:,otherColumns]
        twoWayMatch = tools.loadPickle(self.dataFolder + self.test_subfolder + \
            "getTwoWayMatches_ref.p")
        alreadyMatched = np.zeros((self.numCats,max_index),dtype=bool)
        candidateCounts = [np.zeros((self.numCats,ahCounts[l]),dtype=int) \
            for l in range(0,self.numCats)]
        for m in range(0,self.numCats):
            candidateCounts[0][m,0] = len(allCandidates[0][m][0])
        diffMap = [np.setdiff1d(np.arange(0,3),[k]) \
            for k in range(0,3)]
        computed = generator.followAllMatchChains(0,0,3,\
            oneWayMatches,alreadyMatched,diffMap,allCandidates)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "followAllMatchChains_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_gatherCandidatesRatiosAndDistances(self):
        [oneWayMatchesAllCatalogues,matchArrayList,allCandidates,\
            allRatios,allDistances] = tools.loadPickle(self.dataFolder + \
                self.test_subfolder + "getOneWayMatchesAllCatalogues_ref.p")
        [centresListShort,centralAntihalos,sortedList,ahCounts,max_index] = \
            tools.loadPickle(self.dataFolder + \
            self.test_subfolder + "computeShortCentresList_ref.p")
        oneWayMatches = oneWayMatchesAllCatalogues[0]
        otherColumns = np.array([1,2])
        oneWayMatchesOther = oneWayMatches[:,otherColumns]
        twoWayMatch = tools.loadPickle(self.dataFolder + self.test_subfolder + \
            "getTwoWayMatches_ref.p")
        alreadyMatched = np.zeros((3,max_index),dtype=bool)
        candidateCounts = [np.zeros((self.numCats,ahCounts[l]),dtype=int) \
            for l in range(0,self.numCats)]
        for m in range(0,self.numCats):
            candidateCounts[0][m,0] = len(allCandidates[0][m][0])
        diffMap = [np.setdiff1d(np.arange(0,3),[k]) \
            for k in range(0,3)]
        computed = gatherCandidatesRatiosAndDistances(\
            3,0,0,allCandidates,allRatios,allDistances)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "gatherCandidatesRatiosAndDistances_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_checkIfVoidIsNeeded(self):
        [oneWayMatchesAllCatalogues,matchArrayList,allCandidates,\
            allRatios,allDistances] = tools.loadPickle(self.dataFolder + \
                self.test_subfolder + "getOneWayMatchesAllCatalogues_ref.p")
        [centresListShort,centralAntihalos,sortedList,ahCounts,max_index] = \
            tools.loadPickle(self.dataFolder + \
            self.test_subfolder + "computeShortCentresList_ref.p")
        oneWayMatches = oneWayMatchesAllCatalogues[0]
        otherColumns = np.array([1,2])
        oneWayMatchesOther = oneWayMatches[:,otherColumns]
        twoWayMatch = tools.loadPickle(self.dataFolder + self.test_subfolder + \
            "getTwoWayMatches_ref.p")
        alreadyMatched = np.zeros((3,max_index),dtype=bool)
        candidateCounts = [np.zeros((self.numCats,ahCounts[l]),dtype=int) \
            for l in range(0,self.numCats)]
        for m in range(0,self.numCats):
            candidateCounts[0][m,0] = len(allCandidates[0][m][0])
        diffMap = [np.setdiff1d(np.arange(0,3),[k]) \
            for k in range(0,3)]
        computed = generator.checkIfVoidIsNeeded(0,0,alreadyMatched,\
            twoWayMatch,otherColumns,candidateCounts,oneWayMatches,\
            twoWayOnly=True,blockDuplicates=True)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "checkIfVoidIsNeeded_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_getNewMatches(self):
        [oneWayMatchesAllCatalogues,matchArrayList,allCandidates,\
            allRatios,allDistances] = tools.loadPickle(self.dataFolder + \
                self.test_subfolder + "getOneWayMatchesAllCatalogues_ref.p")
        [centresListShort,centralAntihalos,sortedList,ahCounts,max_index] = \
            tools.loadPickle(self.dataFolder + \
            self.test_subfolder + "computeShortCentresList_ref.p")
        oneWayMatches = oneWayMatchesAllCatalogues[0]
        otherColumns = np.array([1,2])
        oneWayMatchesOther = oneWayMatches[:,otherColumns]
        twoWayMatch = tools.loadPickle(self.dataFolder + self.test_subfolder + \
            "getTwoWayMatches_ref.p")
        alreadyMatched = np.zeros((3,max_index),dtype=bool)
        candidateCounts = [np.zeros((self.numCats,ahCounts[l]),dtype=int) \
            for l in range(0,self.numCats)]
        for m in range(0,self.numCats):
            candidateCounts[0][m,0] = len(allCandidates[0][m][0])
        diffMap = [np.setdiff1d(np.arange(0,3),[k]) \
            for k in range(0,3)]
        computed = generator.getNewMatches(0,0,oneWayMatches,\
            alreadyMatched,blockDuplicates=True)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "getNewMatches_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_getNumberOfConditions(self):
        computed = generator.getNumberOfConditions(conditionList)


# Testing of the ProfileStack class and its methods:
class TestProfileStack(test_base):
    def setUp(self):
        # Some things shared by the catalogue tests:
        self.snapNumList = [2791,3250]
        self.rtol=1e-5
        self.atol=1e-8
        self.generateTestData = generateMode
        self.samplesFolder = dataFolder + "reference_constrained/"
        self.dataFolder = dataFolder
        self.test_subfolder = "function_tests_catalogue/"
        self.snapname = "gadget_full_forward/snapshot_001"
        self.snapnameRev = "gadget_full_reverse/snapshot_001"
        self.snapNameList = [self.samplesFolder + "sample" + \
            str(snapNum) + "/" + self.snapname for snapNum in self.snapNumList]
        self.snapNameListRev = [self.samplesFolder + "sample" + \
            str(snapNum) + "/" + self.snapnameRev \
            for snapNum in self.snapNumList]
        self.snapList =  [pynbody.load(self.samplesFolder + "sample" + \
            str(snapNum) + "/" + self.snapname) for snapNum in self.snapNumList]
        self.ahPropsConstrained = [tools.loadPickle(snap.filename + \
            ".AHproperties.p") \
            for snap in self.snapList]
        self.antihaloRadii = [props[7] for props in self.ahPropsConstrained]
        self.antihaloMasses = [props[3] \
            for props in self.ahPropsConstrained]
        self.ahCentresList = [props[5] \
            for props in self.ahPropsConstrained]
        self.vorVols = [props[4] for props in self.ahPropsConstrained]
        self.numCats = 3
        # Get the test centres:
        [randCentres,randOverDen] = tools.loadPickle(
            self.dataFolder + "function_tests_simulation_tools/" + \
            "get_random_centres_and_densities_ref.p")
        centre_list = [randCentres for ns in range(0,len(self.snapList))]
        r_eff_bin_edges = np.linspace(0,10,101)
        self.conBinEdges = np.linspace(-1,-0.5,21)
        self.conditioningQuantity = [np.vstack([self.antihaloRadii[ns],\
            self.ahPropsConstrained[ns][11],
            self.ahPropsConstrained[ns][12]]).T \
            for ns in range(0,len(self.snapList))]
        self.voidRadiusBinEdges = np.linspace(10,25,6)
        [meanRadii,deltaCentral,deltaAverage] = tools.loadPickle(
            self.dataFolder + self.test_subfolder + "condition_data.p")
        self.conditioningQuantityMCMC = np.vstack([meanRadii,deltaCentral,
                                              deltaAverage]).T
        self.allPairCountsList = [props[9] 
                                  for props in self.ahPropsConstrained]
        self.uncombined_stack = catalogue.ProfileStack(
            centre_list,self.snapList,self.ahPropsConstrained,135,
            r_eff_bin_edges,tree_list=[None,None],seed=1000,start=0,end=-1,
            conditioning_quantity=self.conditioningQuantity,
            conditioning_quantity_to_match=self.conditioningQuantityMCMC,
            condition_bin_edges=[self.voidRadiusBinEdges,self.conBinEdges,
                                 self.conBinEdges],
            combine_random_regions=False,replace=False,
            r_min = self.voidRadiusBinEdges[0],
            r_max = self.voidRadiusBinEdges[-1],
            compute_pair_counts=True,max_sampling = 1,
            pair_counts = self.allPairCountsList)
        self.combined_stack = catalogue.ProfileStack(
            centre_list,self.snapList,self.ahPropsConstrained,135,
            r_eff_bin_edges,tree_list=[None,None],seed=1000,start=0,end=-1,
            conditioning_quantity=self.conditioningQuantity,
            conditioning_quantity_to_match=self.conditioningQuantityMCMC,
            condition_bin_edges=[self.voidRadiusBinEdges,self.conBinEdges,
                                 self.conBinEdges],
            combine_random_regions=True,replace=False,
            r_min = self.voidRadiusBinEdges[0],
            r_max = self.voidRadiusBinEdges[-1],
            compute_pair_counts=True,max_sampling = 1,
            pair_counts = self.allPairCountsList)
    def test_get_number_of_radial_bins(self):
        computed = self.uncombined_stack.get_number_of_radial_bins()
        referenceFile = self.dataFolder + self.test_subfolder + \
            "get_number_of_radial_bins_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_get_all_condition_variables(self):
        self.combined_stack.get_all_condition_variables()
        computed = [self.combined_stack.central_condition_variable_all,
                    self.combined_stack.central_centres_all,
                    self.combined_stack.central_radii_all,
                    self.combined_stack.sample_indices,
                    self.combined_stack.void_indices]
        referenceFile = self.dataFolder + self.test_subfolder + \
            "get_all_condition_variables_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference,filterNaN=True)
    def test_get_all_condition_variables_in_range(self):
        self.combined_stack.get_all_condition_variables()
        void_radii = self.combined_stack.central_radii_all
        condition_variable = self.combined_stack.central_condition_variable_all
        computed = self.combined_stack.get_all_condition_variables_in_range(
            condition_variable,void_radii)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "get_all_condition_variables_in_range_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_get_sampling_ratio(self):
        computed = self.uncombined_stack.get_sampling_ratio()
        referenceFile = self.dataFolder + self.test_subfolder + \
            "get_sampling_ratio_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_select_conditioned_random_voids(self):
        centre = self.uncombined_stack.centre_list[0]
        central_antihalos = tools.getAntiHalosInSphere(\
            self.uncombined_stack.ah_centres_list[ns],
            self.uncombined_stack.r_sphere,origin=centre,\
            boxsize=self.uncombined_stack.boxsize)[1]
        central_indices = np.where(central_antihalos)[0]
        num_cond_variables = len(np.where(central_antihalos)[0])
        central_condition_variable = \
            self.uncombined_stack.conditioning_quantity[0]\
            [central_antihalos].reshape(
                num_cond_variables,1)
        central_radii = \
            self.uncombined_stack.antihalo_radii_list[0][central_antihalos]
        computed = self.uncombined_stack.select_conditioned_random_voids(
            central_condition_variable,central_radii)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "select_conditioned_random_voids_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_get_pooled_variable(self):
        computed = self.combined_stack.get_pooled_variable(
            self.combined_stack.ah_centres_list)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "get_pooled_variable_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_get_volumes_of_radial_bins(self):
        centre = self.uncombined_stack.centre_list[0]
        central_antihalos = tools.getAntiHalosInSphere(\
            self.uncombined_stack.ah_centres_list[ns],
            self.uncombined_stack.r_sphere,origin=centre,\
            boxsize=self.uncombined_stack.boxsize)[1]
        central_radii = \
            self.uncombined_stack.antihalo_radii_list[0][central_antihalos]
        select_array = tools.loadPickle(
            self.dataFolder + self.test_subfolder + \
            "select_conditioned_random_voids_ref.p")
        computed = self.uncombined_stack.get_volumes_of_radial_bins(
            central_radii,select_array)
        referenceFile = self.dataFolder + self.test_subfolder + \
            "get_volumes_of_radial_bins_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)
    def test_get_random_catalogue_pair_counts(self):
        dictionary = self.uncombined_stack.get_random_catalogue_pair_counts()
        {'pairs':self.all_pairs,'volumes':self.all_volumes,
            'selections':self.all_selections,'conditions':self.all_conditions,
            'selected_conditions':self.all_selected_conditions,
            'radii':self.all_radii,'indices':self.all_indices,
            'centres':self.all_centres}
        keys = ['pairs','volumes','selections','conditions',
                'selected_conditions','radii','indices','centres']
        computed = [dictionary[key] for key in keys]
        referenceFile = self.dataFolder + self.test_subfolder + \
            "get_random_catalogue_pair_counts_ref.p"
        reference = self.getReference(referenceFile,computed)
        self.compareToReference(computed,reference)

if __name__ == "__main__":
    unittest.main()
