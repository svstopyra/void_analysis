# generate_simulation_tools_snapshots.py

import numpy as np
from void_analysis.simulation_tools import *
from void_analysis import tools

GENERATED_SNAPSHOTS = [
    "get_borg_density_estimate_ref.npz",
    "get_random_centres_and_densities_ref.npz"
    "getNonOverlappingCentres.p",
    "filter_regions_by_density_ref.p",
    "compute_void_distances_ref.p",
    "getNonOverlappingCentres_ref.p",
    "filter_voids_by_distance_and_radius_ref.p"
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
    print("✅ Tools snapshots saved!")

if __name__ == "__main__":
    generate_snapshots()

