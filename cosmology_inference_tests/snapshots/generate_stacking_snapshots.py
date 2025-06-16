# generate_stacking_snapshots.py

import numpy as np
from void_analysis.cosmology_inference import (
    get_stacked_void_density_field,
    get_1d_real_space_field,
    get_2d_void_stack_from_los_pos,
    get_2d_field_from_stacked_voids,
    get_2d_fields_per_void,
    get_field_from_los_data,
    trim_los_list,
    get_trimmed_los_list_per_void,
    get_los_velocities_for_void,
    combine_los_lists,
    get_weights_for_stack,
    get_weights,
    get_ur_profile_for_void,
    get_all_ur_profiles,
    get_void_weights
)

from void_analysis.simulation_tools import (
    DummySnapshot, 
    generate_synthetic_void_snap
)

from void_analysis import tools

GENERATED_SNAPSHOTS = [
    "get_2d_void_stack_from_los_pos_ref.npy",
    "get_2d_field_from_stacked_voids_ref.npy",
    "get_2d_fields_per_void_ref.npy",
    "get_field_from_los_data_ref.npy",
    "trim_los_list_ref.npy",
    "voids_used_ref.npy",
    "get_trimmed_los_list_per_void_ref.npy",
    "stacked_void_density_ref.npy",
    "real_space_profile_mean_ref.npy",
    "real_space_profile_std_ref.npy",
    "get_los_velocities_for_void_test.npz",
    "combine_los_lists_ref.npy",
    "get_weights_for_stack_ref.npy",
    "get_weights_ref.npy",
    "get_ur_profile_for_void_ref.npy",
    "get_void_weights_ref.npy",
    "get_all_ur_profiles_ref.npy"
]

def generate_snapshots():
    np.random.seed(0)
    los_pos = [[np.random.rand(10, 2) * 3.0 for _ in range(5)]]
    spar_bins = np.linspace(0, 3, 6)
    sperp_bins = np.linspace(0, 3, 6)
    radii = [np.random.uniform(1.0, 2.0, size=5)]
    boxsize = 300
    
    class DummySnap:
        def __init__(self):
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
    snaps = DummySnap()

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

    # 2D stacked void field
    stacked_field = get_stacked_void_density_field(
        snaps, radii, los_pos, spar_bins, sperp_bins, halo_indices=None,
        filter_list=None, additional_weights=None, dist_max=3, rmin=10, rmax=20,
        recompute=False, zspace=True, recompute_zspace=False, suffix=".lospos_all_zspace2.p",
        los_pos=los_pos
    )
    np.save("stacked_void_density_ref.npy", stacked_field)

    # 1D real-space density profile
    rho_mean, rho_std = get_1d_real_space_field(snaps, rbins=spar_bins)
    np.save("real_space_profile_mean_ref.npy", rho_mean)
    np.save("real_space_profile_std_ref.npy", rho_std)

    # Void velocity LOS test:
    synthetic_void_snap = generate_synthetic_void_snap(
        N=32,rmax=50,A=0.85,sigma=10,seed=0,H0=70
    )
    r_par, u_par, disp, u = get_los_velocities_for_void(
        np.array([0]*3),10,synthetic_void_snap,np.linspace(0,30,101)
    )
    np.savez(
        "get_los_velocities_for_void_test.npz",r_par=r_par, u_par=u_par,
        disp=disp, u=u
    )
    
    # Consistent regression test framework:
    # combine_los_lists:
    np.random.seed(0)
    los_list1 = [np.random.rand(5, 2), np.random.rand(5, 2)]
    los_list2 = [np.random.rand(5, 2), np.random.rand(5, 2)]
    tools.generate_regression_test_data(
        combine_los_lists,
        "combine_los_lists_ref.npy",[los_list1,los_list2]
    )
    # get_weights_for_stack
    np.random.seed(0)
    los_list = [[np.random.rand(5, 2)]]
    radii = [np.random.uniform(1.0, 2.0, size=5)]
    tools.generate_regression_test_data(
        get_weights_for_stack,
        "get_weights_for_stack_ref.npy",
        los_list, radii
    )
    
    # get_weights 
    np.random.seed(0)
    los_list = [[np.random.rand(5, 2) for _ in range(5)]]
    radii = [np.random.uniform(1.0, 2.0, size=5)]
    tools.generate_regression_test_data(
        get_weights,
        "get_weights_ref.npy",
        los_list, radii
    )
    # get_ur_profile_for_void
    void_centre = np.array([0]*3)
    void_radius = 10
    rbins = np.linspace(0,30,31)
    snap = generate_synthetic_void_snap(
        N=32,rmax=50,A=0.85,sigma=10,seed=0,H0=70
    )
    tools.generate_regression_test_data(
        get_ur_profile_for_void,
        "get_ur_profile_for_void_ref.npy",
        void_centre,void_radius,rbins,snap,relative_velocity=True
    )
    # get_all_ur_profiles
    boxsize = 100.0
    centres = np.array([[0,0,0],
                        [boxsize/4,0,0],
                        [-boxsize/4,-boxsize/4,-boxsize/4],
                        [boxsize/4,0,-boxsize/4]
                       ])
    radii = np.array([10,5,5,7])
    snap = synthetic_void_snap
    rbins = np.linspace(0,30,31)
    tools.generate_regression_test_data(
        get_all_ur_profiles,
        "get_all_ur_profiles_ref.npy",
        centres, radii,rbins,snap,relative_velocity=True
    )
    # get_void_weights
    np.random.seed(0)
    los_list = [[np.random.rand(5, 2)] for _ in range(0,3)]
    radii = [np.random.uniform(1.0, 2.0, size=5) for _ in range(0,3)]
    voids_used = [np.random.rand(5) > 0.5 for _ in range(0,3)]
    additional_weights = [[np.random.rand(5)*2.0 - 1.0] for _ in range(0,3)]
    tools.generate_regression_test_data(
        get_void_weights,
        "get_void_weights_ref.npy",
        los_list,voids_used,radii,additional_weights=additional_weights
    )

    print("✅ Stacking snapshots generated successfully.")

if __name__ == "__main__":
    generate_snapshots()

