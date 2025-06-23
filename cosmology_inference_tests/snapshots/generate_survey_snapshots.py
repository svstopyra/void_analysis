# generate_likelihood_and_posterior_snapshots.py

import numpy as np
from void_analysis.survey import *

from void_analysis import tools
from void_analysis.simulation_tools import gaussian_delta, gaussian_Delta
from void_analysis import snapedit

GENERATED_SNAPSHOTS = [
    "surveyMask_ref.npy",
    "keCorr_ref.npy"
]

def generate_snapshots():
    # surveyMask
    N = 16 # Use a smaller one to reduce the storage space.
    Nmask = 12*N**2
    np.random.seed(1000)
    # Fake survey masks
    surveyMask11 = np.array(np.random.rand(Nmask) > 0.3,dtype=float)
    surveyMask12 = np.array(np.random.rand(Nmask) > 0.3,dtype=float)
    Om0 = 0.3111
    Ode0 = 0.6889
    boxsize = 677.7
    h=0.6766
    mmin = 0.0
    mmax = 12.5
    grid = snapedit.gridListPermutation(N,perm=(2,1,0))
    centroids = grid*boxsize/N + boxsize/(2*N)
    positions = snapedit.unwrap(centroids - np.array([boxsize/2]*3),\
        boxsize)
    cosmo = astropy.cosmology.LambdaCDM(100*h,Om0,Ode0)
    tools.generate_regression_test_data(
        surveyMask,
        "surveyMask_ref.npy",
        positions,surveyMask11,surveyMask12,cosmo,-0.94,
        -23.28,keCorr = keCorr,mmin=mmin,numericalIntegration=True,
        mmax=mmax,splitApparent=True,splitAbsolute=True,
        returnComponents=True
    )
    # keCorr
    tools.generate_regression_test_data(
        keCorr,
        "keCorr_ref.npy",
        0.1,fit = [-1.456552772320231,-0.7687913554110967]
    )
    

    print("✅ Likelihood and posterior snapshots saved!")

if __name__ == "__main__":
    generate_snapshots()

