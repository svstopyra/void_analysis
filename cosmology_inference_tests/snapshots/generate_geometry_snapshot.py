# generate_geometry_snapshot.py
import numpy as np
from void_analysis.cosmology_inference import geometry_correction

def generate_snapshot():
    s_par = np.linspace(-2, 2, 100)
    s_perp = np.linspace(0.5, 1.5, 100)
    epsilon = 1.08
    s_par_new, s_perp_new = geometry_correction(s_par, s_perp, epsilon)
    result = np.vstack([s_par_new, s_perp_new])
    np.save("geometry_correction_ref.npy", result)
    print("Geometry correction snapshot saved.")

if __name__ == "__main__":
    generate_snapshot()

