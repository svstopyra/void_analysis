# test_stacking_functions.py

import numpy as np
import scipy
import os
import pytest
from void_analysis.cosmology_inference import (
    get_2d_void_stack_from_los_pos,
    get_2d_field_from_stacked_voids,
    get_2d_fields_per_void,
    get_field_from_los_data,
    trim_los_list,
    get_trimmed_los_list_per_void,
    combine_los_lists,
    get_weights_for_stack,
    get_void_weights,
    get_weights,
    get_additional_weights_borg,
    get_halo_indices,
    get_stacked_void_density_field,
    get_1d_real_space_field,
    get_los_velocities_for_void
)

from void_analysis.simulation_tools import (
    DummySnapshot, 
    generate_synthetic_void_snap
)


SNAPSHOT_DIR = os.path.join(os.path.dirname(__file__), "snapshots")

# ---------------------- UNIT TESTS: get_2d_void_stack_from_los_pos ------------

@pytest.fixture
def synthetic_los_data():
    np.random.seed(0)
    los_pos = [[np.random.rand(10, 2) * 3.0 for _ in range(5)]]
    spar_bins = np.linspace(0, 3, 6)
    sperp_bins = np.linspace(0, 3, 6)
    radii = [np.random.uniform(1.0, 2.0, size=5)]
    return los_pos, spar_bins, sperp_bins, radii



@pytest.fixture
def synthetic_void_snap():
    return generate_synthetic_void_snap(
        N=32,rmax=50,A=0.85,sigma=10,seed=0,H0=70
    )

class DummySnap:
    def __init__(self,synthetic_los_data):
        los_pos, spar_bins, sperp_bins, radii = synthetic_los_data
        self.boxsize = 300
        self.snaps = [np.zeros((100, 3))]
        self.snaps_reverse = [np.zeros((100, 3))]
        self.N = 1
        self.pair_counts = [np.ones(len(spar_bins)-1)]
        self.bin_volumes = [np.ones(len(spar_bins)-1)]
        self.radius_bins = [spar_bins]
        self.cell_volumes = [np.ones(100)]
        self.void_centres = [np.zeros((5, 3))]
        self.void_radii = [np.ones(5)]
        self.property_list = {
            "snaps":self.snaps,
            "snaps_reverse":self.snaps_reverse,
            "pair_counts":self.pair_counts,
            "bin_volumes":self.bin_volumes,
            "radius_bins":self.radius_bins,
            "cell_volumes":self.cell_volumes,
            "void_centres":self.void_centres,
            "void_radii":self.void_radii
        }
    def __getitem__(self, property_name):
        return self.property_list[property_name]

def test_get_2d_void_stack_from_los_pos_shape(synthetic_los_data):
    los_pos, spar_bins, sperp_bins, radii = synthetic_los_data
    stacked_particles = get_2d_void_stack_from_los_pos(
        los_pos, spar_bins, sperp_bins, radii, stacked=True
    )
    assert stacked_particles.shape[1] == 2



# ---------------------- UNIT TESTS: combine_los_lists -------------------------

def test_combine_los_lists_basic():
    los_list1 = [np.random.rand(5, 2), np.random.rand(5, 2)]
    los_list2 = [np.random.rand(5, 2), np.random.rand(5, 2)]
    combined = combine_los_lists([los_list1, los_list2])
    assert isinstance(combined, list)
    assert all(isinstance(item, np.ndarray) for item in combined)

# ---------------------- UNIT TESTS: get_weights_for_stack ---------------------

def test_get_weights_for_stack_basic():
    los_list = [[np.random.rand(5, 2)]]
    radii = [np.random.uniform(1.0, 2.0, size=5)]
    weights = get_weights_for_stack(los_list, radii)
    assert isinstance(weights, np.ndarray)
    assert np.all(weights > 0)

# ---------------------- UNIT TESTS: get_void_weights --------------------------

def test_get_void_weights_basic():
    los_list = [np.random.rand(5, 2) for _ in range(5)]
    voids_used = [np.array([True, False, True, False, True])]
    radii = [np.random.uniform(1.0, 2.0, size=5)]
    weights = get_void_weights(los_list, voids_used, radii)
    assert isinstance(weights, list)

# ---------------------- UNIT TESTS: get_weights -------------------------------

def test_get_weights_basic():
    los_list = [[np.random.rand(5, 2) for _ in range(5)]]
    radii = [np.random.uniform(1.0, 2.0, size=5)]
    weights = get_weights(los_list, radii)
    assert isinstance(weights, np.ndarray)

# ---------------------- UNIT TESTS: get_additional_weights_borg ---------------

def test_get_additional_weights_borg_basic():
    class DummyCat:
        def __init__(self):
            self.numCats = 1
            self.finalCatFrac = np.random.rand(5,1)
        def property_with_filter(self, x, void_filter=True):
            return np.random.rand(5)
        def getAllProperties(self, prop, void_filter=True):
            return np.random.rand(5, 1)

    cat = DummyCat()
    weights = get_additional_weights_borg(cat)
    assert isinstance(weights, list)

# ---------------------- UNIT TESTS: get_halo_indices --------------------------

def test_get_halo_indices_basic():
    class DummyCatalogue:
        def __init__(self):
            self.numCats = 2
            self.indexListShort = [np.arange(5), np.arange(5)]
        def get_final_catalogue(self, void_filter=True):
            return np.random.randint(0, 2, (5, 2))

    catalogue = DummyCatalogue()
    halo_indices = get_halo_indices(catalogue)
    assert isinstance(halo_indices, list)
    assert all(isinstance(arr, np.ndarray) for arr in halo_indices)

# ---------------------- UNIT TESTS: trim_los_list -----------------------------

def test_trim_los_list_basic(synthetic_los_data):
    los_pos, spar_bins, sperp_bins, radii = synthetic_los_data
    los_trimmed, voids_used = trim_los_list(
        los_pos, spar_bins, sperp_bins, radii
    )
    assert isinstance(los_trimmed, list)
    assert isinstance(voids_used, list)

# ---------------------- UNIT TESTS: get_trimmed_los_list_per_void -------------

def test_get_trimmed_los_list_per_void_basic(synthetic_los_data):
    los_pos, spar_bins, sperp_bins, radii = synthetic_los_data
    trimmed = get_trimmed_los_list_per_void(
        los_pos, spar_bins, sperp_bins, radii
    )
    assert isinstance(trimmed, list)

# ---------------------- UNIT TESTS: get_field_from_los_data -------------------

def test_get_field_from_los_data_basic(synthetic_los_data):
    los_pos, spar_bins, sperp_bins, radii = synthetic_los_data
    stacked = get_2d_void_stack_from_los_pos(
        los_pos, spar_bins, sperp_bins, radii
    )
    void_count = 5
    v_weight = np.ones(len(stacked))
    field = get_field_from_los_data(
        stacked, spar_bins, sperp_bins, v_weight, void_count
    )
    assert field.shape == (len(spar_bins)-1, len(sperp_bins)-1)

# ---------------------- UNIT TESTS: get_2d_field_from_stacked_voids -----------

def test_get_2d_field_from_stacked_voids_basic(synthetic_los_data):
    los_pos, spar_bins, sperp_bins, radii = synthetic_los_data
    los_list = get_2d_void_stack_from_los_pos(
        los_pos, spar_bins, sperp_bins, radii, stacked=False
    )
    los_per_void = sum(los_list, [])
    field = get_2d_field_from_stacked_voids(
        los_per_void, sperp_bins, spar_bins, radii[0]
    )
    assert field.shape == (len(spar_bins) - 1, len(sperp_bins) - 1)
    assert np.all(np.isfinite(field))

# ---------------------- UNIT TESTS: field utilities ---------------------------

def test_get_field_from_los_data_shape(synthetic_los_data):
    los_pos, spar_bins, sperp_bins, radii = synthetic_los_data
    stacked_particles = get_2d_void_stack_from_los_pos(
        los_pos, spar_bins, sperp_bins, radii, stacked=True
    )
    v_weight = np.ones(len(stacked_particles))
    void_count = 5
    field = get_field_from_los_data(
        stacked_particles, spar_bins, sperp_bins, v_weight, void_count
    )
    assert field.shape == (len(spar_bins) - 1, len(sperp_bins) - 1)
    assert np.all(np.isfinite(field))

def test_trim_los_list_shapes(synthetic_los_data):
    los_pos, spar_bins, sperp_bins, radii = synthetic_los_data
    los_list_trimmed, voids_used = trim_los_list(
        los_pos, spar_bins, sperp_bins, radii
    )
    assert isinstance(los_list_trimmed, list)
    assert isinstance(voids_used, list)

def test_get_trimmed_los_list_per_void_shape(synthetic_los_data):
    los_pos, spar_bins, sperp_bins, radii = synthetic_los_data
    los_list_per_void = get_trimmed_los_list_per_void(
        los_pos, spar_bins, sperp_bins, radii
    )
    assert isinstance(los_list_per_void, list)

# ---------------------- REGRESSION TESTS --------------------------------------

def test_get_2d_fields_per_void_regression(synthetic_los_data):
    los_pos, spar_bins, sperp_bins, radii = synthetic_los_data
    los_list = get_2d_void_stack_from_los_pos(
        los_pos, spar_bins, sperp_bins, radii, stacked=False
    )
    los_per_void = sum(los_list, [])
    fields = get_2d_fields_per_void(
        los_per_void, sperp_bins, spar_bins, radii[0]
    )

    ref = np.load(os.path.join(SNAPSHOT_DIR, "get_2d_fields_per_void_ref.npy"))
    np.testing.assert_allclose(fields, ref, rtol=1e-6)

def test_get_2d_void_stack_from_los_pos_regression(synthetic_los_data):
    los_pos, spar_bins, sperp_bins, radii = synthetic_los_data
    stacked_particles = get_2d_void_stack_from_los_pos(
        los_pos, spar_bins, sperp_bins, radii, stacked=True
    )

    ref = np.load(
        os.path.join(SNAPSHOT_DIR, "get_2d_void_stack_from_los_pos_ref.npy")
    )
    np.testing.assert_allclose(stacked_particles, ref, rtol=1e-6)

def test_get_2d_field_from_stacked_voids_regression(synthetic_los_data):
    los_pos, spar_bins, sperp_bins, radii = synthetic_los_data
    los_list = get_2d_void_stack_from_los_pos(
        los_pos, spar_bins, sperp_bins, radii, stacked=False
    )
    los_per_void = sum(los_list, [])
    field = get_2d_field_from_stacked_voids(
        los_per_void, sperp_bins, spar_bins, radii[0]
    )

    ref = np.load(
        os.path.join(SNAPSHOT_DIR, "get_2d_field_from_stacked_voids_ref.npy")
    )
    np.testing.assert_allclose(field, ref, rtol=1e-6)

def test_get_field_from_los_data_regression(synthetic_los_data):
    los_pos, spar_bins, sperp_bins, radii = synthetic_los_data
    stacked_particles = get_2d_void_stack_from_los_pos(
        los_pos, spar_bins, sperp_bins, radii, stacked=True
    )
    v_weight = np.ones(len(stacked_particles))
    void_count = 5
    field = get_field_from_los_data(
        stacked_particles, spar_bins, sperp_bins, v_weight, void_count
    )

    ref = np.load(os.path.join(SNAPSHOT_DIR, "get_field_from_los_data_ref.npy"))
    np.testing.assert_allclose(field, ref, rtol=1e-6)

def test_trim_los_list_regression(synthetic_los_data):
    los_pos, spar_bins, sperp_bins, radii = synthetic_los_data
    los_list_trimmed, voids_used = trim_los_list(
        los_pos, spar_bins, sperp_bins, radii
    )

    ref_trimmed = np.load(
        os.path.join(SNAPSHOT_DIR, "trim_los_list_ref.npy"), allow_pickle=True
    )
    ref_voids_used = np.load(
        os.path.join(SNAPSHOT_DIR, "voids_used_ref.npy"), allow_pickle=True
    )

    np.testing.assert_equal(los_list_trimmed, ref_trimmed.tolist())
    np.testing.assert_equal(voids_used, ref_voids_used.tolist())

def test_get_trimmed_los_list_per_void_regression(synthetic_los_data):
    los_pos, spar_bins, sperp_bins, radii = synthetic_los_data
    los_list_per_void = get_trimmed_los_list_per_void(
        los_pos, spar_bins, sperp_bins, radii
    )

    ref = np.load(
        os.path.join(SNAPSHOT_DIR, "get_trimmed_los_list_per_void_ref.npy"),
        allow_pickle=True
    )
    np.testing.assert_equal(los_list_per_void, ref.tolist())

def test_get_stacked_void_density_field_regression(synthetic_los_data):
    los_pos, spar_bins, sperp_bins, radii = synthetic_los_data
    boxsize = 300.0

    snaps = DummySnap(synthetic_los_data)
    stacked_field = get_stacked_void_density_field(
        snaps, radii, los_pos, spar_bins, sperp_bins, halo_indices=None,
        filter_list=None, additional_weights=None, dist_max=3, rmin=10, rmax=20,
        recompute=False, zspace=True, recompute_zspace=False,
        suffix=".lospos_all_zspace2.p",los_pos=los_pos
    )
    ref = np.load(os.path.join(SNAPSHOT_DIR, "stacked_void_density_ref.npy"))
    np.testing.assert_allclose(stacked_field, ref, rtol=1e-5)

def test_get_1d_real_space_field_regression(synthetic_los_data):
    los_pos, spar_bins, sperp_bins, radii = synthetic_los_data
    boxsize = 300.0

    snaps = DummySnap(synthetic_los_data)
    rho_mean, rho_std = get_1d_real_space_field(snaps, rbins=spar_bins)
    ref_mean = np.load(
        os.path.join(SNAPSHOT_DIR, "real_space_profile_mean_ref.npy")
    )
    ref_std = np.load(
        os.path.join(SNAPSHOT_DIR, "real_space_profile_std_ref.npy")
    )
    np.testing.assert_allclose(rho_mean, ref_mean, rtol=1e-5)
    np.testing.assert_allclose(rho_std, ref_std, rtol=1e-5)

def test_get_los_velocities_for_void(synthetic_void_snap):
    snap = synthetic_void_snap
    computed_arrays = get_los_velocities_for_void(
        np.array([0]*3),10,snap,np.linspace(0,30,101)
    )
    ref_file = np.load(
        os.path.join(SNAPSHOT_DIR, "get_los_velocities_for_void_test.npz")
    )
    reference_arrays = [ref_file[key] for key in ref_file]
    for arr, arr_ref in zip(computed_arrays,reference_arrays):
        np.testing.assert_allclose(arr, arr_ref, rtol=1e-5)






















