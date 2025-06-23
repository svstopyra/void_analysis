# test_simulation_tools.py

import numpy as np
import pytest
import os
from void_analysis.simulation_tools import *
from void_analysis import tools
import astropy

SNAPSHOT_DIR = os.path.join(os.path.dirname(__file__), "snapshots")

@pytest.fixture
def snapshot_group():
    root_dir = os.path.join(os.path.dirname(__file__), "../test_snaps/")
    snap_list = [
        os.path.join(root_dir,"sample1/forward/snapshot_001"),
        os.path.join(root_dir,"sample2/forward/snapshot_001"),
        os.path.join(root_dir,"sample3/forward/snapshot_001")
    ]
    snap_list_reverse = [
        os.path.join(root_dir,"sample1/reverse/snapshot_001"),
        os.path.join(root_dir,"sample2/reverse/snapshot_001"),
        os.path.join(root_dir,"sample3/reverse/snapshot_001")
    ]
    group = SnapshotGroup(
        snap_list, snap_list_reverse, low_memory_mode=True,
        swapXZ=False, reverse=True, remap_centres=True
    )
    return group


# ---------------------- UNIT TESTS: -------------------------------------------


# ---------------------- REGRESSION TESTS --------------------------------------

def wrapper_get_borg_density_estimate(*args,**kwargs):
    den, CI = get_borg_density_estimate(*args,**kwargs)
    return den.bootstrap_distribution,CI.low,CI.high

def test_get_borg_density_estimate(snapshot_group):
    snaps = snapshot_group
    tools.run_basic_regression_test(
        wrapper_get_borg_density_estimate,
        os.path.join(SNAPSHOT_DIR,"get_borg_density_estimate_ref.npz"),
        snaps, densities_file=None, dist_max=20,seed=1000, interval=0.68,
        nboot = 10
    )

def test_get_random_centres_and_densities(snapshot_group):
    tools.run_basic_regression_test(
        get_random_centres_and_densities,
        os.path.join(SNAPSHOT_DIR,"get_random_centres_and_densities_ref.npz"),
        15,snapshot_group["snaps"],seed=1000,nRandCentres=100
    )


def test_filter_regions_by_density(snapshot_group):
    delta_interval = [-0.5,0.1]
    rand_centres, rand_densities = tools.load_npz_arrays(
        os.path.join(SNAPSHOT_DIR,"get_random_centres_and_densities_ref.npz")
    )
    tools.run_basic_regression_test(
        filter_regions_by_density,
        os.path.join(SNAPSHOT_DIR,"filter_regions_by_density_ref.p"),
        rand_centres, rand_densities, delta_interval
    )

def test_getNonOverlappingCentres(snapshot_group):
    boxsize = snapshot_group.boxsize
    dist_max = 15
    region_masks, centres_to_use = tools.loadPickle(
        os.path.join(SNAPSHOT_DIR,"filter_regions_by_density_ref.p")
    )
    tools.run_basic_regression_test(
        getNonOverlappingCentres,
        os.path.join(SNAPSHOT_DIR,"getNonOverlappingCentres_ref.p"),
        centres_to_use, 2 * dist_max, boxsize, returnIndices=True
    )

def test_compute_void_distances(snapshot_group):
    snaps = snapshot_group
    boxsize = snapshot_group.boxsize
    nonoverlapping_indices = tools.loadPickle(
        os.path.join(SNAPSHOT_DIR,"getNonOverlappingCentres_ref.p")
    )
    region_masks, centres_to_use = tools.loadPickle(
        os.path.join(SNAPSHOT_DIR,"filter_regions_by_density_ref.p")
    )
    selected_region_centres = [
        centres[idx] for centres, idx in zip(
            centres_to_use, nonoverlapping_indices
        )
    ]
    tools.run_basic_regression_test(
        compute_void_distances,
        os.path.join(SNAPSHOT_DIR,"compute_void_distances_ref.p"),
        snaps["void_centres"], selected_region_centres, boxsize
    )

def test_filter_voids_by_distance_and_radius(snapshot_group):
    snaps = snapshot_group
    dist_max = 15
    radii_range = [5,10]
    region_void_dists = tools.loadPickle(
        os.path.join(SNAPSHOT_DIR,"compute_void_distances_ref.p")
    )
    tools.run_basic_regression_test(
        filter_voids_by_distance_and_radius,
        os.path.join(
            SNAPSHOT_DIR,"filter_voids_by_distance_and_radius_ref.p"
        ),
        region_void_dists, snaps["void_radii"], dist_max, radii_range
    )


def test_eulerToZ():
    np.random.seed(1000)
    pos = np.random.randn(100,3)*50
    vel = np.random.randn(100,3)*30
    boxsize = 50.0
    h = 0.7
    centre = np.array([boxsize/2]*3)
    cosmo = astropy.cosmology.LambdaCDM(70,0.3,0.7)
    tools.run_basic_regression_test(
        eulerToZ,
        os.path.join(SNAPSHOT_DIR,"eulerToZ_ref.npy"),
        pos,vel,cosmo,boxsize,h,centre = None,Ninterp=1000,\
        l = 268,b = 38,vl=540,localCorrection = True,velFudge = 1
    )

def test_getGriddedDensity(snapshot_group):
    N = 32
    snapn = snapshot_group.snaps[0]
    tools.run_basic_regression_test(
        getGriddedDensity,
        os.path.join(SNAPSHOT_DIR,"getGriddedDensity_ref.npy"),
        snapn,N,redshiftSpace=True
    )

def test_pointsInRangeWithWrap():
    np.random.seed(1000)
    positions = np.random.random((10000,3))*1000 - 1000
    boxsize = 1000.0
    tools.run_basic_regression_test(
        pointsInRangeWithWrap,
        os.path.join(SNAPSHOT_DIR,"pointsInRangeWithWrap_ref.npy"),
        positions,[-100,100],boxsize=boxsize
    )

def test_pointsInBoundedPlaneWithWrap():
    np.random.seed(1000)
    positions = np.random.random((10000,3))*1000 - 1000
    boxsize = 1000.0
    tools.run_basic_regression_test(
        pointsInBoundedPlaneWithWrap,
        os.path.join(SNAPSHOT_DIR,"run_basic_regression_test_ref.npy"),
        positions,[-100,100],[-100,100],boxsize=boxsize
    )

def test_getGriddedGalCount():
    np.random.seed(1000)
    positions = np.random.random((10000,3))*1000 - 1000
    N = 64
    boxsize = 1000.0
    tools.run_basic_regression_test(
        getGriddedGalCount,
        os.path.join(SNAPSHOT_DIR,"getGriddedGalCount_ref.npy"),
        positions,N,boxsize
    )

def test_biasOld():
    rhoArray = np.linspace(0,1.2,101)
    params = [1.0,0.5,0.5,0.5]
    tools.run_basic_regression_test(
        biasOld,
        os.path.join(SNAPSHOT_DIR,"biasOld_ref.npy"),
        rhoArray,params
    )

def test_biasNew(self):
    biasParam = tools.loadPickle(self.dataFolder + "bias_param_example.p")
    rhoArray = np.linspace(0,1.2,101)
    params = biasParam[0][0,:]
    tools.run_basic_regression_test(
        biasNew,
        os.path.join(SNAPSHOT_DIR,"biasNew_ref.npy"),
        rhoArray,params
    )

def test_ngPerLBin(self):
    N = 16
    referenceMaskList = np.load(
        os.path.joint(SNAPSHOT_DIR,"surveyMask_ref.npy")
    )
    np.random.seed(1000)
    maskRandom = np.random.random((16,16**3))
    biasParam = tools.loadPickle(
        os.path.join(SNAPSHOT_DIR,"bias_param_example.p")
    )
    rng = np.random.default_rng(1000)
    mcmcDen = scipy.stats.lognorm(
        s = 1, scale=np.exp(1),random_state=rng,
    ).rvs((N,N,N))
    mcmcDenLin = np.reshape(mcmcDen,N**3)
    mcmcDen_r = np.reshape(mcmcDenLin,(N,N,N),order='F')
    mcmcDenLin_r = np.reshape(mcmcDen_r,N**3)
    tools.run_basic_regression_test(
        ngPerLBin,
        os.path.join(SNAPSHOT_DIR,"ngPerLBin_ref.npy"),
        biasParam,return_samples=True,mask=referenceMaskList[0],\
        accelerate=True,N=N,\
        delta = [mcmcDenLin_r],contrast=False,sampleList=[0],\
        beta=biasParam[:,:,1],rhog = biasParam[:,:,3],\
        epsg=biasParam[:,:,2],\
        nmean=biasParam[:,:,0],biasModel = biasNew
    )

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



