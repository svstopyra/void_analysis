# test_simulation_tools.py

import numpy as np
import pytest
import os
from void_analysis.simulation_tools import *
from void_analysis import tools

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
    tools.generate_regression_test_data(
        filter_voids_by_distance_and_radius,
        os.path.join(
            SNAPSHOT_DIR,"filter_voids_by_distance_and_radius_ref.p"
        ),
        region_void_dists, snaps["void_radii"], dist_max, radii_range
    )







