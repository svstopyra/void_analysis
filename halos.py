import pynbody
from scipy.optimize import brentq
import scipy
from . import tools, context, snapedit
import numpy as np
import gc


# Overdensity within a specific radius, about a given point. Requires a kdtree
# for the particle distribution of interest.
def getOverdensity(radius,centre,snap,tree,mode = "mean"):
    if mode == "mean":
        rhoRef = pynbody.analysis.cosmology.rho_M(snap,unit = "Msol h**2 Mpc**-3")
    else:
        rhoRef = pynbody.analysis.cosmology.rho_crit(snap,unit = "Msol h**2 Mpc**-3")
    if not snap['mass'].units == pynbody.units.NoUnit():
        mUnit = (snap['mass'][0]*snap['mass'].units).in_units("Msol h**-1")
    else:
        mUnit = snap['mass'][0]*1e10
    boxsize = snap.properties['boxsize'].ratio("Mpc a h**-1")
    count = tree.query_ball_point(centre,radius,workers=-1,return_length=True)
    return count*mUnit/(rhoRef*(4*np.pi*radius**3/3))

# Virial radius of a cluster about a particular centre. Requires a kdTree.
def findVirialRadius(centre,snap,tree,overden = 200,mode = "mean",\
        rmax = 10,rmin = 0.1,searchMax = 20):
    minDen = getOverdensity(rmin,centre,snap,tree,mode=mode) - overden
    maxDen = getOverdensity(rmax,centre,snap,tree,mode=mode) - overden
    if minDen*maxDen > 0:
        # Search for somewhere with a higher density than the specified threshold:
        densities = np.zeros(searchMax)
        radii = np.linspace(rmin,rmax,searchMax)
        for k in range(0,searchMax):
            densities[k] = getOverdensity(radii[k],centre,snap,tree,mode=mode)
        # If found, reset rmin:
        if np.any(densities > overden):
            rmin = radii[np.where(densities > overden)[0][0]]
            minDen = getOverdensity(rmin,centre,snap,tree,mode=mode) - overden
        if minDen*maxDen > 0:
            print("Warning: no zero found within initial bounds.")
            return -1
    return brentq(\
            lambda x: getOverdensity(x,centre,snap,tree,mode=mode) - overden,\
                rmin,rmax)

# Gets the centres of the particles in snap2 corresponding to a set of halos in snap1:
def getCorrespondingCentres(snap1,snap2,largeHalos,hn = None,boxsize=None,\
        wrapped=False,useBridge = True,sort=True,sortedList = None):
    if type(snap2) == type(""):
        snap2 = pynbody.load(snap2)
        tools.remapBORGSimulation(snap2)
    if useBridge:
        bridge = pynbody.bridge.OrderBridge(snap1,snap2,monotonic=False)
    if hn is None:
        hn = snap1.halos()
    if boxsize is None:
        boxsize = snap1.properties['boxsize'].ratio("Mpc a h**-1")
    centres2 = np.zeros((len(largeHalos),3))
    for k in range(0,len(largeHalos)):
        halo = hn[largeHalos[k]+1]
        if useBridge:
            counterpartHalo = bridge(halo)
            centres2[k,:] = context.computePeriodicCentreWeighted(\
                counterpartHalo['pos'],counterpartHalo['mass'],boxsize)
        else:
            if sort:
                if sortedList is None:
                    sortedList = np.argsort(snap2['iord'])
                centres2[k,:] = context.computePeriodicCentreWeighted(\
                    snap2['pos'][sortedList[halo['iord']],:],\
                    snap2['mass'][sortedList[halo['iord']]],boxsize)
            else:
                centres2[k,:] = context.computePeriodicCentreWeighted(\
                    snap2['pos'][halo['iord'],:],\
                    snap2['mass'][halo['iord']],boxsize)
    del snap2
    gc.collect() # Make sure we unload the halo data 
    if wrapped:
        return snapedit.unwrap(centres2,boxsize=boxsize)
    else:
        return centres2

# Compute centre of mass about a given point in a set of points, using 
# an iterative method with a shrinking sphere.
def massCentreAboutPoint(point,positions,boxsize,tree=None,reductions=3,\
        iterations = 10,rstart = 20,weights=None,wrap = True,\
        accelerate = True):
    if tree is None:
        tree = scipy.spatial.cKDTree(snapedit.wrap(positions,boxsize),\
            boxsize=boxsize)
    if weights is None:
        weights = np.ones(len(positions))
    if len(point.shape) == 1:
        rsearch = rstart
    else:
        rsearch = rstart*np.ones(point.shape[0])
    if wrap:
        searchPoint = snapedit.wrap(point,boxsize)
    else:
        searchPoint = point
    centre = np.zeros(point.shape)
    for k in range(0,reductions):
        for l in range(0,iterations):
            parts = tree.query_ball_point(searchPoint,rsearch,workers=-1)
            if len(point.shape) == 1:
                if len(parts) > 0:
                    centre = context.computePeriodicCentreWeighted(\
                        positions[parts,:],weights[parts],boxsize,\
                        accelerate=accelerate)
                else:
                    break
            else:
                for m in range(0,len(point)):
                    if len(parts[m]) > 0:
                        centre[m,:] = context.computePeriodicCentreWeighted(\
                            positions[parts[m],:],weights[parts[m]],boxsize,\
                            accelerate=accelerate)
                # Break if we run out of particles in all regions:
                if np.all([len(p) for p in parts] == 0):
                    break
        rsearch /= 2
        # Break the loop early if we run out of particles to iterate on:
        if len(point.shape) == 1:
            if len(parts) == 0:
                break
        else:
            if np.all([len(p) for p in parts] == 0):
                break
    if wrap:
        centre = snapedit.unwrap(centre,boxsize)
    return centre


def getHaloMass(posAll,haloPos,nbar,Om,partMasses,delta=200,massDef = 'critical',boxsize=None,\
        treeToUse = None,nInterp = 200,rMin = 0,rMax = 20,workers =-1):
    if boxsize is not None:
        posToUse = snapedit.wrap(posAll,boxsize)
        haloPosToUse = snapedit.wrap(haloPos,boxsize)
    else:
        posToUse = posAll
        haloPosToUse = haloPos
    if treeToUse is None:
        treeToUse = scipy.spatial.cKDTree(posToUse,boxsize=boxsize)
    if massDef == 'critical':
        nThresh = nbar/Om
    else:
        nThresh = nbar
    # Get interpolating function:
    rBins = np.linspace(rMin,rMax,nInterp + 1)
    if len(haloPosToUse.shape) < 2:
        haloCount = 1
    else:
        haloCount = haloPosToUse.shape[0]
    rho = np.zeros((haloCount,nInterp))
    if np.isscalar(partMasses):
        for k in range(0,nInterp):
            rho[:,k] = treeToUse.query_ball_point(haloPosToUse,rBins[k+1],\
                workers = workers,return_length=True)*partMasses/\
                ((4*np.pi*rBins[k+1]**3/3)*nThresh)
    else:
        for k in range(0,nInterp):
            indices = treeToUse.query_ball_point(haloPosToUse,rBins[k+1],\
                workers = workers)
            if len(haloPosToUse.shape) < 2:
                rho[:,k] = np.sum(partMasses[indices])/\
                    ((4*np.pi*rBins[k+1]**3/3)*nThresh)
            else:
                for l in range(0,rho.shape[0]):
                    rho[:,k] = np.sum(partMasses[indices[l]])/\
                        ((4*np.pi*rBins[k+1]**3/3)*nThresh)
    # Use interpolating function to find the mass:
    masses = np.zeros(haloCount)
    rDeltas = np.zeros(haloCount)
    for k in range(0,haloCount):
        func = lambda r: np.interp(r,rBins[1:],rho[k,:]) - delta
        if func(rBins[1])*func(rBins[-1]) > 0:
            masses[k] = -1
            rDeltas[k] = -1
        else:
            rDeltas[k] = scipy.optimize.brentq(func,rBins[1],rBins[-1])
            if haloCount == 1:
                thisHaloPos = haloPosToUse
            else:
                thisHaloPos = haloPosToUse[k,:]
            if np.isscalar(partMasses):
                    masses[k] = treeToUse.query_ball_point(thisHaloPos,\
                            rDeltas[k],return_length=True)*partMasses
            else:
                indices = treeToUse.query_ball_point(thisHaloPos,rDeltas[k])
                masses[k] = np.sum(partMasses[indices])
    return (masses,rDeltas)


