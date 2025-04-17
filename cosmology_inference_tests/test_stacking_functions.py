# test_stacking_functions.py

import numpy as np
import os
import pytest
from void_analysis.cosmology_inference import (
    get_2d_void_stack_from_los_pos,
    get_2d_field_from_stacked_voids,
    get_2d_fields_per_void,
    get_field_from_los_data,
    trim_los_list,
    get_trimmed_los_list_per_void
)

SNAPSHOT_DIR = os.path.join(os.path.dirname(__file__), "snapshots")

# ---------------------- UNIT TESTS: get_2d_void_stack_from_los_pos ----------------------

@pytest.fixture
def synthetic_los_data():
    np.random.seed(0)
    los_pos = [[np.random.rand(10, 2) * 3.0 for _ in range(5)]]
    spar_bins = np.linspace(0, 3, 6)
    sperp_bins = np.linspace(0, 3, 6)
    radii = [np.random.uniform(1.0, 2.0, size=5)]
    return los_pos, spar_bins, sperp_bins, radii

def test_get_2d_void_stack_from_los_pos_shape(synthetic_los_data):
    los_pos, spar_bins, sperp_bins, radii = synthetic_los_data
    stacked_particles = get_2d_void_stack_from_los_pos(los_pos, spar_bins, sperp_bins, radii, stacked=True)
    assert stacked_particles.shape[1] == 2

# ---------------------- UNIT TESTS: get_2d_field_from_stacked_voids ----------------------

def test_get_2d_field_from_stacked_voids_basic(synthetic_los_data):
    los_pos, spar_bins, sperp_bins, radii = synthetic_los_data
    los_list = get_2d_void_stack_from_los_pos(los_pos, spar_bins, sperp_bins, radii, stacked=False)
    los_per_void = sum(los_list, [])
    field = get_2d_field_from_stacked_voids(los_per_void, sperp_bins, spar_bins, radii[0])
    assert field.shape == (len(spar_bins) - 1, len(sperp_bins) - 1)
    assert np.all(np.isfinite(field))

# ---------------------- UNIT TESTS: field utilities ----------------------

def test_get_field_from_los_data_shape(synthetic_los_data):
    los_pos, spar_bins, sperp_bins, radii = synthetic_los_data
    stacked_particles = get_2d_void_stack_from_los_pos(los_pos, spar_bins, sperp_bins, radii, stacked=True)
    v_weight = np.ones(len(stacked_particles))
    void_count = 5
    field = get_field_from_los_data(stacked_particles, spar_bins, sperp_bins, v_weight, void_count)
    assert field.shape == (len(spar_bins) - 1, len(sperp_bins) - 1)
    assert np.all(np.isfinite(field))

def test_trim_los_list_shapes(synthetic_los_data):
    los_pos, spar_bins, sperp_bins, radii = synthetic_los_data
    los_list_trimmed, voids_used = trim_los_list(los_pos, spar_bins, sperp_bins, radii)
    assert isinstance(los_list_trimmed, list)
    assert isinstance(voids_used, list)

def test_get_trimmed_los_list_per_void_shape(synthetic_los_data):
    los_pos, spar_bins, sperp_bins, radii = synthetic_los_data
    los_list_per_void = get_trimmed_los_list_per_void(los_pos, spar_bins, sperp_bins, radii)
    assert isinstance(los_list_per_void, list)

# ---------------------- REGRESSION TESTS ----------------------

def test_get_2d_fields_per_void_regression(synthetic_los_data):
    los_pos, spar_bins, sperp_bins, radii = synthetic_los_data
    los_list = get_2d_void_stack_from_los_pos(los_pos, spar_bins, sperp_bins, radii, stacked=False)
    los_per_void = sum(los_list, [])
    fields = get_2d_fields_per_void(los_per_void, sperp_bins, spar_bins, radii[0])

    ref = np.load(os.path.join(SNAPSHOT_DIR, "get_2d_fields_per_void_ref.npy"))
    np.testing.assert_allclose(fields, ref, rtol=1e-6)

def test_get_2d_void_stack_from_los_pos_regression(synthetic_los_data):
    los_pos, spar_bins, sperp_bins, radii = synthetic_los_data
    stacked_particles = get_2d_void_stack_from_los_pos(los_pos, spar_bins, sperp_bins, radii, stacked=True)

    ref = np.load(os.path.join(SNAPSHOT_DIR, "get_2d_void_stack_from_los_pos_ref.npy"))
    np.testing.assert_allclose(stacked_particles, ref, rtol=1e-6)

def test_get_2d_field_from_stacked_voids_regression(synthetic_los_data):
    los_pos, spar_bins, sperp_bins, radii = synthetic_los_data
    los_list = get_2d_void_stack_from_los_pos(los_pos, spar_bins, sperp_bins, radii, stacked=False)
    los_per_void = sum(los_list, [])
    field = get_2d_field_from_stacked_voids(los_per_void, sperp_bins, spar_bins, radii[0])

    ref = np.load(os.path.join(SNAPSHOT_DIR, "get_2d_field_from_stacked_voids_ref.npy"))
    np.testing.assert_allclose(field, ref, rtol=1e-6)

def test_get_field_from_los_data_regression(synthetic_los_data):
    los_pos, spar_bins, sperp_bins, radii = synthetic_los_data
    stacked_particles = get_2d_void_stack_from_los_pos(los_pos, spar_bins, sperp_bins, radii, stacked=True)
    v_weight = np.ones(len(stacked_particles))
    void_count = 5
    field = get_field_from_los_data(stacked_particles, spar_bins, sperp_bins, v_weight, void_count)

    ref = np.load(os.path.join(SNAPSHOT_DIR, "get_field_from_los_data_ref.npy"))
    np.testing.assert_allclose(field, ref, rtol=1e-6)

def test_trim_los_list_regression(synthetic_los_data):
    los_pos, spar_bins, sperp_bins, radii = synthetic_los_data
    los_list_trimmed, voids_used = trim_los_list(los_pos, spar_bins, sperp_bins, radii)

    ref_trimmed = np.load(os.path.join(SNAPSHOT_DIR, "trim_los_list_ref.npy"), allow_pickle=True)
    ref_voids_used = np.load(os.path.join(SNAPSHOT_DIR, "voids_used_ref.npy"), allow_pickle=True)

    np.testing.assert_equal(los_list_trimmed, ref_trimmed.tolist())
    np.testing.assert_equal(voids_used, ref_voids_used.tolist())

def test_get_trimmed_los_list_per_void_regression(synthetic_los_data):
    los_pos, spar_bins, sperp_bins, radii = synthetic_los_data
    los_list_per_void = get_trimmed_los_list_per_void(los_pos, spar_bins, sperp_bins, radii)

    ref = np.load(os.path.join(SNAPSHOT_DIR, "get_trimmed_los_list_per_void_ref.npy"), allow_pickle=True)
    np.testing.assert_equal(los_list_per_void, ref.tolist())

