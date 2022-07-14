import sys
import os
import pynbody
import numpy as np
from void_analysis import context, stacking
from void_analysis.tools import zobovVolumesToPhysical
from void_analyis.tools import getHaloCentresAndMassesFromCatalogue
import multiprocessing as mp
import scipy
thread_count = mp.cpu_count()
import pickle
import argparse

# Master function to process snapshots. Note, this assumes AHF or another 
# halo finder has already been run, as well as ZOBOV, with the relevant 
# volumes data file moved to be in the same place.
def processSnapshot(standard,reverse,nBins,offset=4,output=None):
    if output is None:
        output = standard + ".AHproperties.p"
    # Load snapshots and halo catalogues.
    snapn = pynbody.load(standard)
    hn = snapn.halos()
    snapr = pynbody.load(reverse)
    hr = snapr.halos()
    # Check whether the snapshots need re-ordering:
    sortedn = np.arange(0,len(snapn))
    sortedr = np.arange(0,len(snapr))
    orderedn = np.all(sortedn == snapn['iord'])
    orderedr = np.all(sortedr == snapr['iord'])
    if not orderedn:
        sortedn = np.argsort(snapn['iord'])		
    if not orderedr:
        sortedr = np.argsort(snapr['iord'])

    # Get halo masses and centres from the halo catalogues:
    [hncentres,hnmasses] = getHaloCentresAndMassesFromCatalogue(hn,inMpcs=True)
    [hrcentres,hrmasses] = getHaloCentresAndMassesFromCatalogue(hr,inMpcs=True)
    # Import the ZOBOV Voronoi information:
    haveVoronoi = os.path.isfile(standard + ".vols")
    if haveVoronoi:
        volumes = zobovVolumesToPhysical(standard + ".vols",snapn,\
        dtype=np.double,offset=offset)
    else:
        volumes = snapn['mass']/snapn['rho'] # Use an sph estimate of the 
        # volume weights. Note that these do not necessarily tesselate, 
        # so can't directly obtain the void volumes from them.
    # While we have the halo centres, what we actually need is the
    # anti-halo centres:
    antiHaloCentres = np.zeros((len(hr),3))
    antiHaloVolumes = np.zeros(len(hr))
    boxsize = snapn.properties['boxsize'].ratio("Mpc a h**-1")
    periodicity = [boxsize]*3
    for k in range(0,len(hr)):
        antiHaloCentres[k,:] = context.computePeriodicCentreWeighted(\
            snapn['pos'][sortedn[hr[k+1]['iord']],:],volumes[hr[k+1]['iord']],\
            periodicity)
        antiHaloVolumes[k] = np.sum(volumes[hr[k+1]['iord']])
    antiHaloRadii = np.cbrt(3*antiHaloVolumes/(4*np.pi))
    # Perform pair counting (speeds up computing density profiles, but needs 
    # to be recomputed if we want different bins):
    rBinStack = np.linspace(0,3.0,nBins)
    tree = scipy.spatial.cKDTree(snapn['pos'],boxsize=boxsize)
    [pairCounts,volumesList] = stacking.getPairCounts(\
        antiHaloCentres,antiHaloRadii,snapn,rBinStack,\
        nThreads=thread_count,tree=tree,method="poisson",vorVolumes=volumes)
    # Central and average densities of the anti-halos:
    deltaCentral = np.zeros(len(hr))
    deltaAverage = np.zeros(len(hr))
    rhoBar = np.sum(snapn['mass'])/(boxsize**3) # Cosmological average density
    for k in range(0,len(hr)):
        deltaCentral[k] = stacking.centralDensity(antiHaloCentres[k,:],\
            antiHaloRadii[k],snapn['pos'],volumes,snapn['mass'],tree=tree,\
            centralRatio = 2,nThreads=thread_count)/rhoBar - 1.0
        deltaAverage[k] = np.sum(hr[k+1]['mass'])/\
            np.sum(volumes[hr[k+1]['iord']])/rhoBar - 1.0
    pickle.dump([hncentres,hnmasses,hrcentres,hrmasses,volumes,\
        antiHaloCentres,antiHaloVolumes,antiHaloRadii,rBinStack,pairCounts,\
        volumesList,deltaCentral,deltaAverage],\
        open(output,"wb"))


# Main function, accepts the name of the snapshot to be
#  processed as an argument:
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = \
        "Process a pair and forward " + \
        +"and reverse snapshot files, giving the anti-halo catalogues and " + \
        "various properties of it.")
    parser.add_argument('forward',\
        help = "Name of the snapshot file to process.")
    parser.add_argument('reverse',\
        help = "Name of the reverse snapshot to process.")
    parser.add_argument('--nBins',\
        nargs = 1,help='Number of bins to use',type=int,default=31)
    parser.add_argument('--offset',\
        nargs = 1,help='Skip this number of bytes in the volumes file',\
        type=int,default=4)
    parser.add_argument('--output',nargs = 1,\
        help='Output filename (default: "<snap_filename>.AHproperties.p")',\
        type=str,default = None)
    args = parser.parse_args()
    if args.output is None:
        args.output = args.forward + ".AHproperties.p"
    processSnapshot(args.standard,args.reverse,args.nBins,offset=args.offset,\
        args.output)
















