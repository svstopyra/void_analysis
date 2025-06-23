# generate_simulation_tools_snapshots.py

import numpy as np
from void_analysis.simulation_tools import *
from void_analysis import tools
import astropy

GENERATED_SNAPSHOTS = [
    "get_borg_density_estimate_ref.npz",
    "get_random_centres_and_densities_ref.npz"
    "getNonOverlappingCentres.p",
    "filter_regions_by_density_ref.p",
    "compute_void_distances_ref.p",
    "getNonOverlappingCentres_ref.p",
    "filter_voids_by_distance_and_radius_ref.p",
    "eulerToZ_ref.npy",
    "getGriddedDensity_ref.npy",
    "pointsInRangeWithWrap_ref.npy",
    "run_basic_regression_test_ref.npy",
    "getGriddedGalCount_ref.npy",
    "biasOld_ref.npy",
    "biasNew_ref.npy",
    "ngPerLBin_ref.npy",
    "matchClustersAndHalos_ref.npy"
]

def wrapper_get_borg_density_estimate(*args,**kwargs):
    den, CI = get_borg_density_estimate(*args,**kwargs)
    return den.bootstrap_distribution,CI.low,CI.high


def generate_snapshots():
    np.random.seed(42)
    root_dir = os.path.join(os.path.dirname(__file__),"../../test_snaps/")
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
    snapshot_group = SnapshotGroup(
        snap_list, snap_list_reverse, low_memory_mode=True,
        swapXZ=False, reverse=True, remap_centres=True
    )
    
    tools.generate_regression_test_data(
        wrapper_get_borg_density_estimate,
        "get_borg_density_estimate_ref.npz",
        snapshot_group, densities_file=None, dist_max=20,seed=1000, 
        interval=0.68,nboot = 10
    )
    # get_random_centres_and_densities
    dist_max = 15
    tools.generate_regression_test_data(
        get_random_centres_and_densities,
        "get_random_centres_and_densities_ref.npz",
        dist_max,snapshot_group["snaps"],seed=1000,nRandCentres=100
    )
    # filter_regions_by_density
    delta_interval = [-0.5,0.1]
    rand_centres, rand_densities = tools.load_npz_arrays(
        "get_random_centres_and_densities_ref.npz"
    )
    tools.generate_regression_test_data(
        filter_regions_by_density,
        "filter_regions_by_density_ref.p",
        rand_centres, rand_densities, delta_interval
    )
    # getNonOverlappingCentres
    boxsize = snapshot_group.boxsize
    region_masks, centres_to_use = tools.loadPickle(
        "filter_regions_by_density_ref.p"
    )
    tools.generate_regression_test_data(
        getNonOverlappingCentres,
        "getNonOverlappingCentres_ref.p",
        centres_to_use, 2 * dist_max, boxsize, returnIndices=True
    )
    # compute_void_distances
    nonoverlapping_indices = tools.loadPickle("getNonOverlappingCentres_ref.p")
    selected_region_centres = [
        centres[idx] for centres, idx in zip(
            centres_to_use, nonoverlapping_indices
        )
    ]
    tools.generate_regression_test_data(
        compute_void_distances,
        "compute_void_distances_ref.p",
        snapshot_group["void_centres"], selected_region_centres, boxsize
    )
    # filter_voids_by_distance_and_radius
    radii_range = [5,10]
    region_void_dists = tools.loadPickle("compute_void_distances_ref.p")
    tools.generate_regression_test_data(
        filter_voids_by_distance_and_radius,
        "filter_voids_by_distance_and_radius_ref.p",
        region_void_dists, snapshot_group["void_radii"], dist_max, radii_range
    )
    
    # eulerToZ
    np.random.seed(1000)
    pos = np.random.randn(100,3)*50
    vel = np.random.randn(100,3)*30
    boxsize = 50.0
    h = 0.7
    centre = np.array([boxsize/2]*3)
    cosmo = astropy.cosmology.LambdaCDM(70,0.3,0.7)
    tools.generate_regression_test_data(
        eulerToZ,
        "eulerToZ_ref.npy",
        pos,vel,cosmo,boxsize,h,centre = None,Ninterp=1000,\
        l = 268,b = 38,vl=540,localCorrection = True,velFudge = 1
    )
    # getGriddedDensity
    N = 32
    snapn = snapshot_group.snaps[0]
    tools.generate_regression_test_data(
        getGriddedDensity,
        "getGriddedDensity_ref.npy",
        snapn,N,redshiftSpace=True
    )
    # pointsInRangeWithWrap
    np.random.seed(1000)
    positions = np.random.random((10000,3))*1000 - 1000
    boxsize = 1000.0
    tools.generate_regression_test_data(
        pointsInRangeWithWrap,
        "pointsInRangeWithWrap_ref.npy",
        positions,[-100,100],boxsize=boxsize
    )
    # pointsInBoundedPlaneWithWrap
    np.random.seed(1000)
    positions = np.random.random((10000,3))*1000 - 1000
    boxsize = 1000.0
    tools.generate_regression_test_data(
        pointsInBoundedPlaneWithWrap,
        "run_basic_regression_test_ref.npy",
        positions,[-100,100],[-100,100],boxsize=boxsize
    )
    # getGriddedGalCount
    np.random.seed(1000)
    positions = np.random.random((10000,3))*1000 - 1000
    N = 64
    boxsize = 1000.0
    tools.generate_regression_test_data(
        getGriddedGalCount,
        "getGriddedGalCount_ref.npy",
        positions,N,boxsize
    )
    # biasOld
    rhoArray = np.linspace(0,1.2,101)
    params = [1.0,0.5,0.5,0.5]
    tools.generate_regression_test_data(
        biasOld,
        "biasOld_ref.npy",
        rhoArray,params
    )
    # biasNew
    biasParam = tools.loadPickle(
        os.path.join(SNAPSHOT_DIR,"bias_param_example.p")
    )
    rhoArray = np.linspace(0,1.2,101)
    params = biasParam[0][0,:]
    tools.generate_regression_test_data(
        biasNew,
        "biasNew_ref.npy",
        rhoArray,params
    )
    # ngPerLBin
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
    tools.generate_regression_test_data(
        ngPerLBin,
        "ngPerLBin_ref.npy",
        biasParam,return_samples=True,mask=referenceMaskList[0],\
        accelerate=True,N=N,\
        delta = [mcmcDenLin_r],contrast=False,sampleList=[0],\
        beta=biasParam[:,:,1],rhog = biasParam[:,:,3],\
        epsg=biasParam[:,:,2],\
        nmean=biasParam[:,:,0],biasModel = biasNew
    )
    # matchClustersAndHalos
        ahProps = snapshot_group.all_property_lists[0]
    boxsize = snapshot_group.boxsize
    nVoids = ahProps[0].shape[0]
    haloCentresList = tools.remapAntiHaloCentre(
        ahProps[0][0:nVoids,:],boxsize
    )
    haloMasseslist = ahProps[1][0:nVoids]
    [combinedAbellN,combinedAbellPos,abell_nums] = \
        real_clusters.getCombinedAbellCatalogue(\
        catFolder = SNAPSHOT_DIR + "/")
    tmppFile=os.path.join(SNAPSHOT_DIR,"2mpp_data/2MPP.txt")
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
    tools.generate_regression_test_data(
        matchClustersAndHalos,
        "matchClustersAndHalos_ref.npy",
        combinedAbellPos,haloCentresList,haloMasseslist,boxsize,pos2mpp
    )
    
    print("✅ Tools snapshots saved!")

if __name__ == "__main__":
    generate_snapshots()

