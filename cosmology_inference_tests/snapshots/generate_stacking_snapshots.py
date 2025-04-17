# generate_stacking_snapshots.py

import numpy as np
from void_analysis.cosmology_inference import (
    get_2d_void_stack_from_los_pos,
    get_2d_field_from_stacked_voids,
    get_2d_fields_per_void,
    get_field_from_los_data,
    trim_los_list,
    get_trimmed_los_list_per_void
)

GENERATED_SNAPSHOTS = [
    "get_2d_void_stack_from_los_pos_ref.npy",
    "get_2d_field_from_stacked_voids_ref.npy",
    "get_2d_fields_per_void_ref.npy",
    "get_field_from_los_data_ref.npy",
    "trim_los_list_ref.npy",
    "voids_used_ref.npy",
    "get_trimmed_los_list_per_void_ref.npy"
]

def generate_snapshots():
    np.random.seed(0)
    los_pos = [[np.random.rand(10, 2) * 3.0 for _ in range(5)]]
    spar_bins = np.linspace(0, 3, 6)
    sperp_bins = np.linspace(0, 3, 6)
    radii = [np.random.uniform(1.0, 2.0, size=5)]

    # Generate and save stacked particles
    stacked_particles = get_2d_void_stack_from_los_pos(los_pos, spar_bins, sperp_bins, radii, stacked=True)
    np.save("get_2d_void_stack_from_los_pos_ref.npy", stacked_particles)

    # Generate and save 2D field from stacked voids
    los_list = get_2d_void_stack_from_los_pos(los_pos, spar_bins, sperp_bins, radii, stacked=False)
    los_per_void = sum(los_list, [])
    field_stacked = get_2d_field_from_stacked_voids(los_per_void, sperp_bins, spar_bins, radii[0])
    np.save("get_2d_field_from_stacked_voids_ref.npy", field_stacked)

    # Generate and save 2D fields per void
    fields_per_void = get_2d_fields_per_void(los_per_void, sperp_bins, spar_bins, radii[0])
    np.save("get_2d_fields_per_void_ref.npy", fields_per_void)

    # Generate and save field from los data
    v_weight = np.ones(len(stacked_particles))
    void_count = 5
    field_from_los = get_field_from_los_data(stacked_particles, spar_bins, sperp_bins, v_weight, void_count)
    np.save("get_field_from_los_data_ref.npy", field_from_los)

    # Generate and save trimmed lists
    los_list_trimmed, voids_used = trim_los_list(los_pos, spar_bins, sperp_bins, radii)
    np.save("trim_los_list_ref.npy", np.array(los_list_trimmed, dtype=object))
    np.save("voids_used_ref.npy", np.array(voids_used, dtype=object))

    los_list_per_void = get_trimmed_los_list_per_void(los_pos, spar_bins, sperp_bins, radii)
    np.save("get_trimmed_los_list_per_void_ref.npy", np.array(los_list_per_void, dtype=object))

    print("✅ Stacking snapshots generated successfully.")

if __name__ == "__main__":
    generate_snapshots()

