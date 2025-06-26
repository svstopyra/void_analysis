"""
Contains functions that compute various properties of antihalos.
"""


import numpy as np
import pynbody
from . import context, stacking
import os
import multiprocessing as mp
thread_count = mp.cpu_count()
import Corrfunc

# For backwards compatibility (this function used to be here):
from void_analysis.context import computePeriodicCentreWeighted

def voidsRadiiFromAntiHalos(snapn,snapr,hn,hr,volumes):
    """
    Get the radii of anti-halo voids:
    
    Parameters:
        snapn (pynbody snapshot): Forward simulation.
        snapr (pynbody snapshot): Reverse simulation
        hn (pynbody halo catalogue): Forward simulation halos
        hr (pynbody halo catalogue): Reverse simulation halos
        volumes (array): Voronoi volumes of particles in the forward simulation
    
    Returns:
        array (length of hn): Radii of all voids in the catalogue.
    """
    b = pynbody.bridge.Bridge(snapn,snapr)
    voidRadii = np.zeros(len(hr))
    for k in range(0,len(hr)):
        voidRadii[k] = np.cbrt(3.0*np.sum(volumes[hr[k+1]['iord']])/(4.0*np.pi))
    return voidRadii

def computeAntiHaloCentres(hr,snap,volumes):
    """
    Compute anti-halo centres
    
    Parameters:
        hr (pynbody halo catalogue): Halo catalogue of the reverse simulation
        snap (pynbody snapshot): Forward simulation snapshot
        volumes (array): Voronoi volumes of particles in the forward sim.
    
    Returns:
        array (Nx3): Centres of all voids in the anti-halo-catalogue. 
                     N = len(hr)
    """
    centres = np.zeros((len(hr),3))
    periodicity = [snap.properties['boxsize'].ratio("Mpc a h**-1")]*3
    for k in range(0,len(hr)):
        centres[k,:] = context.computePeriodicCentreWeighted(\
            snap['pos'][hr[k+1]['iord']],volumes[hr[k+1]['iord']],periodicity)
    return centres


def getAntiHaloMasses(hr,fixedMass=False):
    """
    Get vector of anti-halo masses
    
    Parameters:
        hr (pynbody halo catalogue): Halo catalogue of the reverse simulation
        fixedMass (bool): If True, assume all particles have the same mass, 
                          allowing for optimisation.
    
    Returns:
        array (length of hr): Masses of all anti-halos.
    """
    antiHaloMasses = np.zeros(len(hr))
    mUnit = hr[1]['mass'][0]*1e10
    for k in range(0,len(hr)):
        if fixedMass:
            antiHaloMasses[k] = mUnit*len(hr[k+1])
        else:
            antiHaloMasses[k] = np.sum(hr[k+1]['mass'])
    return antiHaloMasses

def getAntiHaloDensities(hr,snap,volumes=None):
    """
    Get the volume averaged densities of all the anti-halos.
    
    Parameters:
        hr (pynbody halo catalogue): Halo catalogue of the reverse simulation
        snap (pynbody snapshot): Forward simulation snapshot
        volumes (array): Voronoi volumes of particles in the forward sim.
    
    Returns:
        array (len(hr)): Average density (Mass/Volume) of all voids.
    """
    if volumes is None:
        # Use the nearest neighbour distance:
        volumes = snap['smooth']
        rho = snap['rho']
    else:
        rho = np.array(snap['mass'].in_units("Msol h**-1"))/volumes
    antiHaloDensities = np.zeros(len(hr))
    for k in range(0,len(hr)):
        antiHaloDensities[k] = np.sum(rho[hr[k+1]['iord']]*\
            volumes[hr[k+1]['iord']])/np.sum(volumes[hr[k+1]['iord']])
    return antiHaloDensities

def fitMassAndRadii(antiHaloMasses,antiHaloRadii,logThresh=14):
    """
    Polynomial fit of the anti-halo massses and their radii above a given mass 
    threshold
    
    Parameters:
        antiHaloMasses (array): Masses of all anti-halos.
        antiHaloRadii (array): Radii of all anti-halos. Must be same length
                               as antiHaloMasses
        logThresh (float): log(10) of the lowest mass included in the fit.
    
    Returns:
        Numpy polyfit for the relation log(Mass) = fit[0]*log(Radii) + fit[1]
    """
    logMass = np.log10(antiHaloMasses)
    logRad = np.log10(antiHaloRadii)
    aboveThresh = np.where(logMass > logThresh)
    fit = np.polyfit(logMass[aboveThresh],logRad[aboveThresh],1)# = (b,a)
    return fit

def MtoR(x,a,b):
    """
    Conversion between mass and radius, according to the fit provided by 
    fitMassAndRadius
    
    Parameters:
        x (array): Masses to convert
        a (float): Prefactor exponent. fit[1] from output of fitMassAndRadius
        b (float): Exponent of log(radii), fit[0] from output of 
                   fitMassAndRadius
    
    Returns:
        array (length of x): Radii of the void of mass x
    """
    return (10**a)*(x**b)

def RtoM(y,a,b):
    """
    Conversion between radius and mass, according to the fit provided by 
    fitMassAndRadius
    
    Parameters:
        x (array): Radii to convert
        a (float): Prefactor exponent. fit[1] from output of fitMassAndRadius
        b (float): Exponent of log(radii), fit[0] from output of 
                   fitMassAndRadius
    """
    return (y/(10**a))**(1/b)

def computeVolumeWeightedBarycentre(positions,volumes):
    """
    Volume weighted barycentres of a set of particles:
    
    Parameters:
        positions (Nx3 array): Positions of the tracers of which to compute
                               the barycentre.
        volumes (Nx1-component array): Voronoi volumes of the tracers.
    
    Returns:
        1x3 array: Volume-weighted barycentre of the tracers.
    """
    if volumes.shape != (1,len(positions)):
        volumes2 = np.reshape(volumes,(1,len(positions)))
    else:
        volumes2 = volumes
    weightedPos = (volumes2.T)*positions
    return np.sum(weightedPos,0)/np.sum(volumes)



def getCoincidingVoids(centre,radius,voidCentres):
    """
    Return the voids which lie within an effective radius of a given halo.
    
    Parameters:
        centre (1x3 array): Centre of the halo or point about which to find
                            voids
        radius (float): Distance out to which to search for voids
        voidCentres (Nx3 array): Centres of the voids
    
    Returns:
        Indices of the halos which lie within radius of the halo centre.
    """
    dist = np.sqrt(np.sum((voidCentres - centre)**2,1))
    return np.where(dist <= radius)

def getCoincidingVoidsInRadiusRange(
        centre,radius,voidCentres,voidRadii,rMin,rMax
    ):
    """
    Return all the coincident voids with a given radius range.
    
    Parameters:
        centre, radius, voidCentres: as in getCoincidingVoids
        voidRadii (Nx1 array): radii of the voids
        rMin, rMax (floats): Lower and upper radii to filter voids by.       
        
    Returns:
        Voids close to a given centre which are in a particular radius range.
    """
    coinciding = getCoincidingVoids(centre,radius,voidCentres)
    inRange = np.where((voidRadii[coinciding] >= rMin) & \
        (voidRadii[coinciding] <= rMax))
    return coinciding[0][inRange]

def getAntihaloOverlapWithVoid(antiHaloParticles,voidParticles,volumes):
    """
    Given a set of anti-halo particles, and a set of void particles from some
    other void definition (or a different anti-halo), compute the fraction of 
    volume the two voids have in common.
    
    Parameters:
        antiHaloParticles (Nx1 array of ints): IDs of the particles in the 
                                               antihalos
        voidParticles (Nx1 array of ints): IDs of the particles in the void
        volumes (Nx1-component array): Voronoi volumes of the tracers.
    
    Returns:
        2 component list. First element is the fraction of anti-halo volume
        in the intersection, Second is the fraction of the void volume.
    """
    intersection = np.intersect1d(antiHaloParticles,voidParticles)
    return [
        np.sum(volumes[intersection])/np.sum(volumes[antiHaloParticles]),
        np.sum(volumes[intersection])/np.sum(volumes[voidParticles])
    ]

def getOverlapFractions(antiHaloParticles,cat,voidList,volumes,mode = 0):
    """
    Given a catalogue of ZOBOV voids, compute the overlap with a given set
    of anti-halo particles.
    
    Parameters:
        antiHaloParticles: as in getAntihaloOverlapWithVoid
        cat: ZOBOV void catalogue
        voidList (list): List of voids we wish to process from the ZOBOV 
                         catalogue.
        volumes (Nx1-component array): Voronoi volumes of the tracers.
        mode (int or string): Component to return (fraction of anti-halo volume 
                    if mode = 0, or of the void volume if mode = 1. If set to 
                    "both", just returns both fractions as an array.
    
    Returns:
        array (Mx1 or Mx2): Volume fractions shared by the voids.
    """
    fraction = np.zeros((len(voidList),2))
    for k in range(0,len(fraction)):
        overlap = getAntihaloOverlapWithVoid(antiHaloParticles,cat.void2Parts(\
            voidList[k]),volumes)
        fraction[k,:] = overlap
    if mode == "both":
        return fraction
    else:
        return fraction[:,mode]

def getVoidOverlapFractionsWithAntihalos(
        voidParticles,hr,antiHaloList,volumes,mode = 0
    ):
    """
    Return the overlap fractions between a list of ZOBOV voids, and a list of
    antihalos.
    
    Parameters:
        voidParticles: ZOBOV void catalogue
        hr (pynbody halo catalogue): Anti-halo catalogue
        antiHaloList (List): Anti-halos to include from hr.
        volumes (Nx1-component array): Voronoi volumes of the tracers.
        mode (int or string): as in getAntihaloOverlapWithVoid
    
    Returns:
        Array with overlap fractions.
    """
    fraction = np.zeros((len(antiHaloList),2))
    for k in range(0,len(fraction)):
        overlap = getAntihaloOverlapWithVoid(hr[antiHaloList[k]+1]['iord'],\
            voidParticles,volumes)
        fraction[k,:] = overlap
    if mode == "both":
        return fraction
    else:
        return fraction[:,mode]

def removeSubvoids(cat,voidSet):
    """
    Remove any voids from the set that are actually subvoids of another in 
    the set.
    
    Parameters:
        cat (ZOBOV void catalogue)
        voidSet: Subset of voids to consider
    """
    # Get ID list:
    voidIDs = cat.voidID[voidSet]
    parentIDs = cat.parentID[voidSet]
    subvoids = np.in1d(parentIDs,voidIDs)
    return voidSet[np.logical_not(subvoids)]


def getAntiHaloVoidCandidates(
        antiHalo,centre,radius,cat,volumes,rMin=None,rMax=None,threshold = 0.5,
        removeSubvoids = False,rank = True
    ):
    """
    Function to figure out if an anti-halo has a corresponding ZOBOV void.
    
    Parameters:
        antiHalo (pynbody halo snapshot): Halo to consider
        centre (1x3 array): Centre of the antihalo
        radius (float): Distance out to which to search for ZOBOV voids.
        volumes (Nx1-component array): Voronoi volumes of the tracers.
        rMin, rMax (floats): Lower and upper radius limits.
        threshold (float): Lower volume fraction for the intersection to 
                           consider something a candidate.
        removeSubvoids (bool): If True, remove any subvoids from the ZOBOV 
                               catalogue.
        rank (bool): If True, sort in descending order of overlap with the 
                     antihalo.
    
    Returns:
        List of ZOBOV voids which might correspond to this antihalo.
    """
    voidCentres = cat.voidCentres
    voidRadii = cat.radius
    if rMin is None:
        rMin = 0.75*radius
    if rMax is None:
        rMax = 1.2*radius
    coinciding = getCoincidingVoidsInRadiusRange(centre,radius,voidCentres,voidRadii,rMin=rMin,rMax=rMax)
    fractions = np.zeros((len(coinciding),2))
    for k in range(0,len(coinciding)):
        fractions[k,:] = getAntihaloOverlapWithVoid(antiHalo['iord'],cat.void2Parts(coinciding[k]),volumes)
    candidates = np.where((fractions[:,0] >= threshold) & (fractions[:,1] >= threshold))
    coincidingCandidates = coinciding[candidates]
    if rank:
        # Sort in descending order of overlap with the anti-halo
        coincidingCandidates = coincidingCandidates[np.argsort(-fractions[candidates,0])]

    if not removeSubvoids:
        return coinciding[candidates]
    else:
        return removeSubvoids(cat,coinciding[candidates])

def computeZoneCentres(snap,cat,volumes):
    """
    Compute the volume weighted barycentres of each zone.
    
    Parameters:
        snap (pynbody snapshot): Simulation snapshot where voids are found.
        cat: ZOBOV void catalogue
        volumes (Nx1-component array): Voronoi volumes of the tracers.
    
    Returns:
        Array: Volume-weighted barycentres of all zones.
    """
    zoneCentres = np.zeros((cat.numZonesTot,3))
    boxsize = snap.properties['boxsize'].ratio("Mpc a h**-1")
    periodicity = [boxsize]*3
    for k in range(0,cat.numZonesTot):
        zoneParts = cat.zones2Parts(k)
        zoneCentres[k,:] = computePeriodicCentreWeighted(snap[zoneParts]['pos'],volumes[zoneParts],periodicity)
    return zoneCentres

def getCorrespondingZoneCandidates(
        antiHalo,centre,radius,volumes,catalog,zoneCentres,threshold = 0.5
    ):
    """
    Find the ZOBOV zones that could correspond to a particular halo
    
    Parameters
        antiHalo (pynbody halo snapshot): Halo to consider
        centre (1x3 array): Centre of the antihalo
        radius (float): Distance out to which to search for ZOBOV voids.
        volumes (Nx1-component array): Voronoi volumes of the tracers.
        catalog: ZOBOV catalogue
        zoneCentres (array): Centres of the ZOBOV zones
        threshold (float): Lower volume fraction for the intersection to 
                           consider something a candidate.
    
    Returns:
        2 component list: 1st component: IDs of candidate zones
                          2nd component: Volume fractions of candidates.
    """
    # Find which zones are within the search radius:
    zoneDistances = np.sqrt(np.sum((zoneCentres - centre)**2,1))
    inRadius = np.where(zoneDistances <= radius)
    # For these zones, compute the fraction of volume they share with the anti-halo:
    volumeShared = np.zeros(inRadius[0].shape)
    for k in range(0,len(volumeShared)):
        volumeShared[k] = getAntihaloOverlapWithVoid(antiHalo['iord'],catalog.zones2Parts(inRadius[0][k]),volumes)[1]
    highlyCorrelated = np.where(volumeShared >= threshold)
    return [inRadius[0][highlyCorrelated],volumeShared[highlyCorrelated]]

def getCorrespondingSubVoidCandidates(
        antiHalo,centre,radius,volumes,catalog,voidCentres,threshold = 0.5,
        subVoidsOnly = False
    ):
    """
    Get possible subvoids that could correspond to a particular anti-halos
    
    Parameters
        antiHalo (pynbody halo snapshot): Halo to consider
        centre (1x3 array): Centre of the antihalo
        radius (float): Distance out to which to search for ZOBOV voids.
        volumes (Nx1-component array): Voronoi volumes of the tracers.
        catalog: ZOBOV catalogue
        voidCentres (array): Centres of the ZOBOV voids
        threshold (float): Lower volume fraction for the intersection to 
                           consider something a candidate.
        subVoidsOnly (bool): If True, return only voids which are subvoids.
    
    Returns:
        2 component list: 1st component: IDs of candidate voids
                          2nd component: Volume fractions of candidates.
    """
    # Find which zones are within the search radius:
    voidDistances = np.sqrt(np.sum((voidCentres - centre)**2,1))
    inRadius = np.where(voidDistances <= radius)
    # For these zones, compute the fraction of volume they share with the anti-halo:
    volumeShared = np.zeros(inRadius[0].shape)
    voidVolumes = np.zeros(inRadius[0].shape)
    voidFullVolumes = np.zeros(voidVolumes.shape)
    antiHaloVolume = np.sum(volumes[antiHalo['iord']])
    for k in range(0,len(volumeShared)):	
        voidParticles = catalog.void2Parts(inRadius[0][k])
        intersection = np.intersect1d(antiHalo['iord'],voidParticles)
        volIntersection = np.sum(volumes[intersection])
        voidVolumes[k] = volIntersection
        voidFullVolumes[k] = np.sum(volumes[voidParticles])
        volumeShared[k] = volIntersection/np.sum(volumes[voidParticles])
    highlyCorrelated = np.where(volumeShared >= threshold) 
    if not subVoidsOnly:
        volumeFraction = np.sum(voidVolumes[highlyCorrelated])/antiHaloVolume
        voidVolumeFraction =  np.sum(voidVolumes[highlyCorrelated])/np.sum(voidFullVolumes[highlyCorrelated])
        return [inRadius[0][highlyCorrelated],volumeShared[highlyCorrelated],volumeFraction,voidVolumeFraction]
    else:
        voidIDs = catalog.voidID[inRadius[0][highlyCorrelated]]
        parentIDs = catalog.parentID[inRadius[0][highlyCorrelated]]
        subvoids = np.in1d(parentIDs,voidIDs)
        parentVoids = np.logical_not(subvoids)
        volumeFraction = np.sum(voidVolumes[highlyCorrelated][parentVoids])/antiHaloVolume
        voidVolumeFraction = np.sum(voidVolumes[highlyCorrelated][parentVoids])/np.sum(voidFullVolumes[highlyCorrelated][parentVoids])
        return [inRadius[0][highlyCorrelated][parentVoids],volumeShared[highlyCorrelated][parentVoids],volumeFraction,voidVolumeFraction]

def runGenPk(
        centresAH,centresZV,massesAH,massesZV,rFilterAH = None,rFilterZV=None
    ):
    """
    Wrapper code that calls GenPK to compute power spectra, if it exists on the
    path.
    
    Parameters:
        centresAH (Nx3 array): Anti-halo centres
        centresZV (Mx3 array): Void centres
        massesAH (Nx1 array): Antohalo masses
        massesZV (Nx1 array): ZOBOV void masses
        rFilterAH (array): Filter for anti-halos
        rFilterZV (array): Filter for ZOBOV voids.
    
    Returns:
        4 component list: 1st component: Antihalo power spectrum
                          2nd component: ZOBOV void power spectrum
                          3rd component: Cross-spectrum of antihalos and voids.
                          4th component: DM power spectrum.
    """
    if rFilterAH is None:
        rFilterAH = slice(len(centresAH[:,0]))
    if rFilterZV is None:
        rFilterZV = slice(len(centresZV[:,0]))
    ahPos = pynbody.array.SimArray(centresAH[rFilterAH,:],"Mpc a h**-1")
    ahVel = pynbody.array.SimArray(np.zeros(ahPos.shape),"km a**1/2 s**-1")
    ahMass = pynbody.array.SimArray(np.ones(len(massesAH[rFilterAH])),"Msol h**-1")
    vdPos = pynbody.array.SimArray(centresZV[rFilterZV,:],"Mpc a h**-1")
    vdVel = pynbody.array.SimArray(np.zeros(catToUse.voidCentres[rFilterZV,:].shape),"km a**1/2 s**-1")
    vdMass = pynbody.array.SimArray(massesZV[rFilter2],"Msol h**-1")
    sAHs = snapedit.newSnap("antihalos.gadget2",ahPos,ahVel,dmMass=ahMass,gasPos = vdPos ,gasVel = vdVel ,gasMass = vdMass,properties=snap.properties,fmt=type(snap))
    # Run Gen-Pk:
    if not os.path.isdir("genpk_out"):
        os.system("mkdir genpk_out")
    os.system("gen-pk -i antihalos.gadget2 -o genpk_out")
    os.system("gen-pk -i antihalos.gadget2 -o genpk_out -c 0")
    # Import results:
    psAHs = np.loadtxt("./genpk_out/PK-DM-antihalos.gadget2")
    psVoids = np.loadtxt("./genpk_out/PK-by-antihalos.gadget2")
    psCross = np.loadtxt("./genpk_out/PK-DMxby-antihalos.gadget2")
    psMatter = np.loadtxt("./genpk_out/PK-DM-snapshot_011")
    return [psAHs,psVoids,psCross,psMatter]

# 

def simulationCorrelation(
        rBins,boxsize,data1,data2=None,nThreads = 1,weights1=None,
        weights2 = None
    ):
    """
    Estimate correlation function of discrete data. If data2 is specified, it 
    computes the cross correlation of the two data sets
    
    Parameters:
        rBins (array): Edges of the radial bins.
        boxsize (float): Periodic box size.
        data1 (N1x3 array): First quantity to compute correlation function of.
        data2 (N2x3 array or None): Second quantity. If None, autocorrelation
                                    is computed for data1. Otherwise, compute
                                    the cross-correlation.
        nThreads (int): Number of threads to use
        weights1 (float): Weights for data1.
        weights2 (float): Weights for data2.
    
    Returns:
        array: Estimated correlation function of the data in the specified bins.
    """
    X1 = data1[:,0]
    Y1 = data1[:,1]
    Z1 = data1[:,2]
    N1 = len(X1)
    # Randoms for each dataset:
    rand_N1 = 3*N1
    X1rand = np.random.uniform(0,boxsize,rand_N1)
    Y1rand = np.random.uniform(0,boxsize,rand_N1)
    Z1rand = np.random.uniform(0,boxsize,rand_N1)
    if data2 is not None:
        X2 = data2[:,0]
        Y2 = data2[:,1]
        Z2 = data2[:,2]
        N2 = len(X2)
        rand_N2 = 3*N2
        X2rand = np.random.uniform(0,boxsize,rand_N2)
        Y2rand = np.random.uniform(0,boxsize,rand_N2)
        Z2rand = np.random.uniform(0,boxsize,rand_N2)
    if data2 is None:
        # Auto-correlation:
        DD1 = Corrfunc.theory.DD(
            1,nThreads,rBins,X1,Y1,Z1,periodic=True,boxsize=boxsize,
            weights1=weights1
        )
        DR1 = Corrfunc.theory.DD(
            0,nThreads,rBins,X1,Y1,Z1,periodic=True,boxsize=boxsize,
            X2 = X1rand,Y2=Y1rand,Z2=Z1rand,weights1=weights1
        )
        RR1 = Corrfunc.theory.DD(
            1,nThreads,rBins,X1rand,Y1rand,Z1rand,periodic=True,
            boxsize=boxsize,weights1=weights1
        )
        xiEst = Corrfunc.utils.convert_3d_counts_to_cf(
            N1,N1,rand_N1,rand_N1,DD1,DR1,DR1,RR1
        )
    else:
        # Cross correlation:
        D1D2 = Corrfunc.theory.DD(
            0,nThreads,rBins,X1,Y1,Z1,X2=X2,Y2=Y2,Z2=Z2,periodic=True,
            boxsize=boxsize,weights1=weights1,weights2=weights2
        )
        D1R2 = Corrfunc.theory.DD(
            0,nThreads,rBins,X1,Y1,Z1,periodic=True,boxsize=boxsize,
            X2 = X2rand,Y2=Y2rand,Z2=Z2rand,weights1=weights1,
            weights2=weights2
        )
        D2R1 = Corrfunc.theory.DD(
            0,nThreads,rBins,X2,Y2,Z2,periodic=True,boxsize=boxsize,
            X2 = X1rand,Y2=Y1rand,Z2=Z1rand,weights1=weights1,
            weights2=weights2
        )
        R1R2 = Corrfunc.theory.DD(
            0,nThreads,rBins,X1rand,Y1rand,Z1rand,X2 = X2rand,Y2=Y2rand,
            Z2=Z2rand,periodic=True,boxsize=boxsize,weights1=weights1,
            weights2=weights2
        )
        xiEst = Corrfunc.utils.convert_3d_counts_to_cf(
            N1,N2,rand_N1,rand_N2,D1D2,D1R2,D2R1,R1R2
        )
    return xiEst


def getAutoCorrelations(ahCentres,voidCentres,ahRadii,voidRadii,rMin = 0,
        rMax = np.inf,rRange = np.linspace(0.1,10,101),nThreads=thread_count,
        boxsize = 200.0
    ):
    """
    Auto correlations of void and anti-halo centres:#
    
    Parameters:
        ahCentres (Nx3 array): Antihalo centres
        voidCentres (Mx3 array): Void centres
        ahRadii (Nx1 array): Antihalo radii
        voidRadii (Mx1 array): Void radii
        rMin, rMax (floats): Lower and upper radius limits
        rRange (array): Points at which to compute correlation function.
        nThreads (int): Number of threads to use
        boxsize (float): Size of the periodic box
    
    Returns:
        2 component list: 1st component: Antihalos autocorrelation
                          2nd component: Voids autocorrelation
    """
    rFilter1 = np.where((ahRadii > rMin) & (ahRadii < rMax))[0]
    rFilter2 = np.where((voidRadii > rMin) & (voidRadii < rMax))[0]
    ahPos = ahCentres[rFilter1,:]
    vdPos = voidCentres[rFilter2,:]
    xiAA = simulationCorrelation(rRange,boxsize,ahPos,nThreads=nThreads)
    xiVV = simulationCorrelation(rRange,boxsize,vdPos,nThreads=nThreads)
    return [xiAA,xiVV]

# Return specified stacks
def getStacks(
        ahRadius,ahMasses,antiHaloCentres,zvRadius,zvMasses,voidCentres,snap,
        pairCountsAH,pairCountsZV,volumesListAH,volumesListZV,
        conditionAH = None,conditionZV = None,showPlot=True,ax=None,
        rBins = np.linspace(0,3,31),sizeBins = [2,4,10,21],plotAH=True,
        plotZV=True,binType="radius",tree=None,sumType='poisson',
        yUpper = 1.3,valuesAH = None,valuesZV = None,binLabel="",
        errorType="Profile"
    ):
    """
    Stack voids and antihalos in 1D and compare on the same plot.
    
    Parameters:
        ahRadius (Nx1 array): Antihalo radii
        ahMasses (Nx1 array): Anti-halo masses
        antiHaloCentres (Nx3 array): Anti-halo centres
        zvRadius (Mx1 array): ZOBOV void radii
        zvMasses (Mx1 array): ZOBOV void masses
        voidCentre (Mx3 array): ZOBOV void centres
        snap (pynbody snapshot): Forward simulation snapshot
        pairCountsAH (array of ints): Pair counts around anti-halos
        pairCountsZV (array of ints): Pair counts around ZOBOV voids.
        volumesListAH (array): Anti-halo volumes
        volumesListZV (array): ZOBOV void volumes.
        conditionAH (array): Filter for antihalos
        conditionZV (array): Filter for ZOBOV voids
        showPlot (bool): If true, display plot after computing.
        ax (Axis handle of None): Axis on which to display the plot.
        rBins (array): Radius bins to use.
        sizeBins (array): Void radius bins to use.
        plotAH (bool): If True, include anti-halos on plot.
        plotZV (bool): If True, include ZOBOV voids on plot.
        binType (string): Type of binning to do. Options are
                          binning by radius, by mass, and by Radius converted
                          to mass using a fitted relationship.
        tree (scipy.cKDTree): KD tree used to speed up stacking.
        sumType (string): Parameter that defines how the void profile is
                          computed.
        yUpper (float): Upper bound for y axis.
        valuesAH (array): Values to bin for antihalos. 
                          If None, determined by binType.
        valuesZV (array): Values to bin for ZOBOV voids. If None, determined
                          by binType.
        binLabel (string): Label for the x axis.
        errorType (string): Method for computing profile errors. See 
                            stacking.stackVoidsWithFilter for details.
    
    Returns:
        4-component list: 1st component: Anti-halo density profile
                          2nd component: ZOBOV void density profile
                          3rd component: Anti-halo profile errors.
                          4th component: ZOBOV profile errors.
        
    """
    AHfilters = []
    ZVfilters = []
    nbar = len(snap)/(snap.properties['boxsize'].ratio("Mpc a h**-1"))**3
    for k in range(0,len(sizeBins)-1):
        if (valuesAH is None) or (valuesZV is None):
            if binType == "radius":
                valuesAH = ahRadius
                valuesZV = zvRadius
            elif binType == "mass":
                valuesAH = ahMasses
                valuesZV = zvMasses
            elif binType == "RtoM":
                valuesAH = ahMasses
                valuesZV = RtoM(zvMasses,a,b)
            else:
                raise Exception("Unrecognised bin type.")
        filterConditionAH = (
            (valuesAH > sizeBins[k]) & (valuesAH <= sizeBins[k+1]
        )
        filterConditionZV = (
            (valuesZV > sizeBins[k]) & (valuesZV <= sizeBins[k+1])
        )
        if conditionAH is not None:
            filterConditionAH = filterConditionAH & conditionAH
        if conditionZV is not None:
            filterConditionZV = filterConditionZV & conditionZV
        AHfilters.append(np.where(filterConditionAH))
        ZVfilters.append(np.where(filterConditionZV))
    nBarsAH = []
    nBarsZV = []
    sigmaBarsAH = []
    sigmaBarsZV = []
    for k in range(0,len(sizeBins)-1):
        [nbarj_AH,sigma_AH] = stacking.stackVoidsWithFilter(
            antiHaloCentres,ahRadius,AHfilters[k][0],snap,rBins,tree=tree,
            method=sumType,nPairsList=pairCountsAH,volumesList=volumesListAH,
            errorType=errorType
        )
        [nbarj_ZV,sigma_ZV] = stacking.stackVoidsWithFilter(
            voidCentres,zvRadius,ZVfilters[k][0],snap,rBins,tree=tree,
            method=sumType,nPairsList=pairCountsZV,volumesList=volumesListZV,
            errorType=errorType
        )
        nBarsAH.append(nbarj_AH)
        nBarsZV.append(nbarj_ZV)
        sigmaBarsAH.append(sigma_AH)
        sigmaBarsZV.append(sigma_ZV)
    if showPlot:
        plotStacks(
            rBins,nBarsAH,nBarsZV,sigmaBarsAH,sigmaBarsZV,sizeBins,binType,
            nbar,plotAH=plotAH,plotZV=plotZV,yUpper = yUpper,binLabel=binLabel
        )
    return [nBarsAH,nBarsZV,sigmaBarsAH,sigmaBarsZV]

