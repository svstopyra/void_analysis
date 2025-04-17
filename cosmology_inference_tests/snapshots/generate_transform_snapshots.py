# generate_transform_snapshots.py
import numpy as np
from void_analysis.cosmology_inference import to_z_space

GENERATED_SNAPSHOTS = [
    "zspace_transform_ref.npy"
]

def generate_snapshots():
    z = 0.1
    Om = 0.3
    f = 0.8
    r_par = np.linspace(0.1, 3.0, 100)
    r_perp = np.ones_like(r_par)

    Delta = lambda r: 0.2 * np.exp(-r**2)

    s_par, s_perp = to_z_space(r_par, r_perp, z, Om, Delta=Delta, f=f)
    result = np.vstack([s_par, s_perp])
    np.save("zspace_transform_ref.npy", result)
    print("Saved z-space transform snapshot.")

if __name__ == "__main__":
    generate_snapshots()
