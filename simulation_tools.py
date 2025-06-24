# Tools to compute properties of simulations.
import pynbody
import numpy as np
import astropy
import numexpr as ne
from .halos import massCentreAboutPoint
import scipy
from . import tools, snapedit, context, stacking, cosmology
import pickle
import os
import multiprocessing as mp
thread_count = mp.cpu_count()
from astropy.coordinates import SkyCoord
import astropy.units as u
import gc
import h5py

def eulerToZ(
        pos,vel,cosmo,boxsize,h,centre = None,Ninterp=1000,\
        l = 268,b = 38,vl=540,localCorrection = True,velFudge = 1
    ):
    """
    Convert eulerian co-ordinate to redshift space co-ordinates:
    
    Parameters:
        pos (Nx3 array): Positions in Eulerian co-ordinates.
        vel (Nx3 array): Velocities
        cosmo (astroy cosmology object): Cosmological parameters from astropy.
        boxsize (float): Size of the periodic box in Mpc/h
        h (float): Dimensionless Hubble rate.
        centre (array or None): Centre with respect to which to compute
                                displacements. If None, assume positions are
                                already relative to the centre.
        Ninterp (int): Number of interpolation points, used for speeding up
                       inversion. Inverses are computed only at these points 
                       and other values are interpolated.
        l (float): Galactic longitude of the local direction used for removing
                   local frame.
        b (float): Galactic latitude of local direction.
        vl (float): Velocity in the direction of (l,b)
        localCorrection (bool): If true, apply local correction.
        velFudge (float): Factor by which to multiply velocities (mostly ussed
                          for testing)
    
    Returns:
        array: Positions in redshift space co-ordinates.
    
    Tests:
        Tested in test_simulation_tools.py
        Regression test: test_eulerToZ    
    """    
    # Compute comoving distance:
    if centre is not None:
        dispXYZ = snapedit.unwrap(pos - centre,boxsize)
    else:
        dispXYZ = pos
    r = np.sqrt(np.sum(dispXYZ**2,1))
    rq = r*astropy.units.Mpc/h
    # Compute relationship between comoving distance and redshift at discrete 
    # points.
    # For speed, we will interpolate any other points as computing it for each 
    # point individually requires an expensive function inversion:
    zqMin = astropy.cosmology.z_at_value(cosmo.comoving_distance,np.min(rq))
    zqMax = astropy.cosmology.z_at_value(cosmo.comoving_distance,np.max(rq))
    # Grid of redshifts vs comoving distances used for interpolation:
    zgrid = np.linspace(zqMin,zqMax,Ninterp)
    Dgrid = cosmo.comoving_distance(zgrid)
    # Interpolate the redshifts for each comoving distance
    zq = np.interp(rq.value,Dgrid.value,zgrid)
    # Now get the angular co-ordinates:
    phi = np.arctan2(dispXYZ[:,1],dispXYZ[:,0])
    theta = np.pi/2 - np.arcsin(dispXYZ[:,2]/r)
    # Construct z-space Euclidean co-ordinates:
    posZ = np.zeros(pos.shape)
    c = 3e5
    rz = c*zq/(100)
    # RSDs:
    # Get local peculiar velocity correction:
    if localCorrection:
        local = SkyCoord(l=l*u.deg,b=b*u.deg,distance=1*u.Mpc,frame='galactic')
        vhat = np.array(
            [local.icrs.cartesian.x.value,local.icrs.cartesian.y.value,\
            local.icrs.cartesian.z.value]
        )
        vd = velFudge*np.sum(pos*(vel - vl*vhat),1)/r
    else:
        vd = np.sum(pos*vel,1)/r
    rsd = rz + vd/100
    posZ[:,0] = rsd*np.sin(theta)*np.cos(phi)
    posZ[:,1] = rsd*np.sin(theta)*np.sin(phi)
    posZ[:,2] = rsd*np.cos(theta)
    # Need to make sure things that are outside the periodic domain are now 
    # wrapped back into it:
    posZ[np.where(posZ <= -boxsize/2)] += boxsize
    posZ[np.where(posZ > boxsize/2)] -= boxsize
    return posZ


def getGriddedDensity(
        snap,N,redshiftSpace= False,velFudge = 1,snapPos = None,snapVel = None,
        snapMass = None
    ):
    """
    Computes the density field of a pynbody snapshot using a naiive 
    cloud-in-cell scheme (counts particles in a give cubic cell). Only works
    for exactly cubic particle counts.
    
    Parameters:
        snap (pynbody snapshot): Pynbody snapshot to get density field for.
        N (int): Cube root of particle count.
        redshiftSpace (bool): If true, convert to redshift space first.
        velFudge (float): Rescale velocity by this amount. Used for testing.
        snapPos (array or None): Positions of particles in the snapshot. If 
                                 None, loaded from the simulation. Use this 
                                 argument to over-ride this, especially if 
                                 different co-ordinates are used.
        snapVel (array or None): Simulation velocities over-ride, as snapPos.
        snapMass (array or None): Simulation masses override, as snapPos.
    
    Tests:
        Tested in test_simulation_tools.py
        Regression test: test_getGriddedDensity
    """
    boxsize = snap.properties['boxsize'].ratio("Mpc a h**-1")
    if snapPos is None:
        snapPos = np.array(snap['pos'].in_units("Mpc a h**-1"))
    if snapVel is None:
        snapVel = snap['vel']
    if snapMass is None:
        snapMass = np.array(snap['mass'].in_units('Msol h**-1'))
    if redshiftSpace:
        cosmo = astropy.cosmology.LambdaCDM(snap.properties['h']*100,\
            snap.properties['omegaM0'],snap.properties['omegaL0'])
        pos = eulerToZ(snapPos,snapVel,\
            cosmo,boxsize,snap.properties['h'],velFudge=velFudge)
    else:
        pos = snapPos
    H, edges = np.histogramdd(pos,bins = N,\
        range = ((-boxsize/2,boxsize/2),(-boxsize/2,boxsize/2),\
            (-boxsize/2,boxsize/2)),\
            weights = snapMass,density=False)
    cellWidth = boxsize/N
    cellVol = cellWidth**3
    meanDensity = np.double(np.sum(snapMass))/(boxsize**3)
    density = H/(cellVol*meanDensity)
    # Deal with an ordering issue:
    density = np.reshape(np.reshape(density,N**3),(N,N,N),order='F')
    return density


def pointsInRangeWithWrap(positions,lim,axis=2,boxsize=None):
    """
    Compute the points within a co-ordinate range, accounting for periodic 
    wrapping.
    
    Parameters:
        positions (array): positions to check
        lim (tuple): Limits to considered
        axis (int): 1, 2, or 3, the co-ordinate axis to check.
        boxsize (float or None): Periodic boxsize. If None, assume not periodic.
    
    Returns:
        array of bools: Positions within the specified range.
    
    Tests:
        Tested in test_simulation_tools.py
        Regression test: test_pointsInRangeWithWrap

    """
    if boxsize is None:
        wrappedPositions = positions
        wrappedLim = np.array(lim)
    else:
        wrappedPositions = snapedit.unwrap(positions,boxsize)
        wrappedLim = snapedit.unwrap(np.array(lim),boxsize)
    return (
        (wrappedPositions[:,axis]>= wrappedLim[0]) & 
        (wrappedPositions[:,axis] <= wrappedLim[1])
    )

# Compute the points that lie within a plane with specified bounds, accounting
# for periodicity.
def pointsInBoundedPlaneWithWrap(positions,xlim,ylim,boxsize=None):
    if boxsize is None:
        wrappedPositions = positions
        wrappedXLim = np.array(xlim)
        wrappedYLim = np.array(ylim)
    else:
        wrappedPositions = snapedit.unwrap(positions,boxsize)
        wrappedXLim = snapedit.unwrap(np.array(xlim),boxsize)
        wrappedYLim = snapedit.unwrap(np.array(ylim),boxsize)
    return (
        (wrappedPositions[:,0] >= wrappedXLim[0]) & 
        (wrappedPositions[:,0] <= wrappedXLim[1]) & 
        (wrappedPositions[:,1] >= wrappedYLim[0]) & 
        (wrappedPositions[:,1] <= wrappedYLim[1])
    )

# Count number of objects within N^3 grid cells
def getGriddedGalCount(pos,N,boxsize):
    H, edges = np.histogramdd(pos,bins = N,\
        range = ((-boxsize/2,boxsize/2),(-boxsize/2,boxsize/2),\
            (-boxsize/2,boxsize/2)),density=False)
    # Deal with an ordering issue:
    H = np.reshape(np.reshape(H,N**3),(N,N,N),order='F')
    return H

# Old Bias Model:
def biasOld(rhoArray,params,accelerate = True):
    nmean = params[0]
    rhog = params[1]
    epsg = params[2]
    beta = params[3]
    if accelerate:
        nmeans = nmean
        mrhogs = -rhog
        mepsgs = -epsg
        betas = beta
        resArray = ne.evaluate(\
            "nmeans*(rhoArray**betas)*exp(mrhogs*(rhoArray**mepsgs))")
    else:
        resArray = nmean[ls,:]*np.power(rhoArray,beta[ls,:])*\
            np.exp(-rhog[ls,:]*np.power(rhoArray,-epsg[ls,:]))
    return resArray

# New Bias model (reparameterised, but otherwise the same):
def biasNew(rhoArray,params,accelerate = True,offset = 1e-6):
    nmean = params[0]
    rhog = params[1]
    epsg = params[2]
    beta = params[3]
    x = rhoArray + offset
    rhoArray2 = x/rhog
    if accelerate:
        nmeans = nmean
        mrhogs = -rhog
        mepsgs = -epsg
        betas = beta
        resArray = ne.evaluate(\
            "nmeans*(x**betas)*exp(-(rhoArray2**mepsgs))")
    else:
        resArray = nmean[ls,:]*np.power(x,beta[ls,:])*\
            np.exp(-np.power(rhoArray2,-epsg[ls,:]))
    return resArray + offset

# Galaxy count per luminosity bin:
def ngPerLBin(bias,lumBins = 16,nReal = 6,N = 256,returnError = False,\
        delta=None,reshapeOrder = 'F',sampleList = None,contrast = True,\
        return_samples=False,mask = None,mask2 = None,accelerate=False,\
        beta=None,rhog = None,epsg = None,nmean=None,biasModel=biasOld,\
        fixNormalisation = False,ngExpected=None):
    if (ngExpected is None) and fixNormalisation:
        raise Exception("Expected counts must be specified to fix" + \
                " normalisation.")
    if beta is None:
        beta = bias['bs'][:,:,0]
    if rhog is None:
        rhog = bias['bs'][:,:,1]
    if epsg is None:
        epsg = bias['bs'][:,:,2]
    if nmean is None:
        nmean = bias['nmeans'][:,:,0]
    if delta is None:
        # Use the density field stored in the bias data (this is the most
        # consistent thing to do)
        delta = bias['density']
        nDelta = nReal
        scalar = False
    else:
        # Check whether we have a list of density fields, or just a single 
        # density field:
        if type(delta) == type([]) or type(delta) == type((1,)):
            nDelta = len(delta)
            scalar = False
        else:
            nDelta = 1
            scalar = True
    if sampleList is None:
        # Allows us to pick out specific samples from the bias data, excluding 
        # some if we don't have the right data to use them.
        if scalar:
            sampleList = np.arange(nReal)
        else:
            sampleList = np.arange(nDelta)
    ng = np.zeros((len(sampleList),lumBins,N**3))
    if not scalar:
        # Use each density field in the list with a corresponding set of bias 
        # parameters to get the ngalaxy count in each luminosity bin, 
        # averaging over the samples.
        for l in range(0,len(sampleList)):
            ls = sampleList[l]
            if len(delta[l].shape) == 3:
                rhoUse = np.reshape(delta[l],N**3)
            else:
                rhoUse = delta[l]
            if contrast:
                rhoUse += 1.0
            rhoArray = np.tile(rhoUse,(lumBins,1)).transpose()
            params = [nmean[ls,:],rhog[ls,:],epsg[ls,:],beta[ls,:]]
            resArray = biasModel(rhoArray,params,accelerate=accelerate)
            ng[l,:] = resArray.transpose()
    else:
        # Use the same density field for each set of bias parameters, 
        # and average over the samples.
        if len(delta.shape) == 3:
            deltaUse = np.reshape(delta,N**3)
        else:
            deltaUse = delta
        if not contrast:
            deltaUse -= 1.0
        deltaArray = np.tile(deltaUse,(lumBins,1)).transpose()
        for l in range(0,len(sampleList)):
            ls = sampleList[l]
            resArray = nmean[ls,:]*np.power(1.0 + deltaArray,beta[ls,:])*\
                np.exp(-rhog[ls,:]*np.power(1.0 + deltaArray,-epsg[ls,:]))
            ng[l,:] = resArray.transpose()
    if mask is not None:
        if mask2 is None:
            if accelerate:
                ng = ne.evaluate("ng*mask")
            else:
                ng *= mask
        else:
            # Apply different survey masks to the different catalogues:
            maskMult = np.tile(np.vstack((mask,mask2)),(8,1))
            if accelerate:
                ng = ne.evaluate("ng*maskMult")
            else:
                ng *= maskMult
    if fixNormalisation:
        # Renormalise the amplitude of each bin to match the excpected counts
        # in each bin:
        numerator = ne.evaluate("sum(ng*ngExpected,axis=2)")
        denominator = ne.evaluate("sum(ng**2,axis=2)")
        nbar = nmean*numerator/denominator
        ng = (nbar.reshape((lumBins,1)))*(ng/(nmean.reshape((lumBins,1))))
    if return_samples:
        return ng
    elif not returnError:
        return np.mean(ng,0)
    else:
        return [np.mean(ng,0),np.std(ng,0)/np.sqrt(len(smapleList))]

def matchClustersAndHalos(clusterPos,haloPos,haloMass,boxsize,catalogPos,\
        gatherRadius = 5,neighbourRadius = 10,massProxy = None):
    treeCat = scipy.spatial.cKDTree(snapedit.wrap(catalogPos,boxsize),\
        boxsize=boxsize)
    treeClusters = scipy.spatial.cKDTree(snapedit.wrap(clusterPos,boxsize),\
        boxsize=boxsize)
    treeHalos = scipy.spatial.cKDTree(snapedit.wrap(haloPos,boxsize),\
        boxsize=boxsize)
    if massProxy is None:
        massProxy = treeCat.query_ball_point(snapedit.wrap(clusterPos,boxsize),\
            gatherRadius,workers=-1,return_length=True)
    counterpartClusters = -np.ones(len(haloPos),dtype=int)
    counterpartHalos = -np.ones(len(clusterPos),dtype=int)
    neighbourHalos = treeHalos.query_ball_point(\
        snapedit.wrap(clusterPos,boxsize),\
        neighbourRadius,workers=-1)
    neighbourClusters = treeClusters.query_ball_point(\
        snapedit.wrap(clusterPos,boxsize),neighbourRadius,workers=-1)
    maxNeighbourCluster = np.array(\
        [clusterList[np.argmax(massProxy[clusterList])] \
        for clusterList in neighbourClusters])
    for k in range(0,len(clusterPos)):
        if maxNeighbourCluster[k] == k:
            sortedNearbyClusters = np.flip(\
                np.argsort(massProxy[neighbourClusters[k]]))
            sortedNearbyHalos = np.flip(\
                np.argsort(haloMass[neighbourHalos[k]]))
            descendingClusters = np.array(\
                neighbourClusters[k],dtype=int)[sortedNearbyClusters]
            descendingHalos = np.array(\
                neighbourHalos[k],dtype=int)[sortedNearbyHalos]
            matchableNum = np.min([len(descendingClusters),\
                len(descendingHalos)])
            counterpartClusters[descendingHalos[0:matchableNum]] = \
                 descendingClusters[0:matchableNum]
            counterpartHalos[descendingClusters[0:matchableNum]] = \
                 descendingHalos[0:matchableNum]
    return [counterpartClusters,counterpartHalos]

def getMassCalcMethod(halo):
    # Figure out what kind of mass estimation method is available:
    try:
        mass = halo.properties['mass']
        return "key"
    except(KeyError):
        print("No mass property found. Trying mass array...")
    try:
        allMasses = halo['mass'].in_units("Msol h**-1")
        return "sum"
    except(KeyError):
        print("No mass array found. Try computing from length.")
        return "length"

# Figure out the mass of a single particle in a simulation, assuming
# it is fixed resolution.
def getMassUnit(snap):
    omegaM0 = snap.properties['omegaM0']
    boxsize = snap.properties['boxsize'].ratio("Mpc a h**-1")
    rhoc = 2.7754e11
    # Account for the fact that we might be dealing with a halo, rather than
    # a full simulation snapshot:
    if type(snap) == pynbody.halo.Halo:
        N = len(snap.base)
    else:
        N = len(snap)
    return omegaM0*rhoc*boxsize**3/N

def getHaloMassByMethod(halo,method):
    if(method == "key"):
        # Read from the halo header.
        return halo.properties['mass']
    elif(method == "sum"):
        # Directly sum the masses. This is slower than just reading it.
        return np.sum(np.array(halo['mass'].in_units("Msol h**-1")))
    elif(method == "length"):
        # Only works if the simulation has fixed resolution (ie, is not a 
        # zoom simulation) but if there is no mass array, we have to assume 
        # this is the case or we won't have enough information to compute
        # the mass:
        mUnit = getMassUnit(halo)
        return mUnit*len(halo)

def getHaloCentresAndMassesFromCatalogue(h,remap=True,\
        inMpcs = True,pynbody_offset=0):
    hcentres = np.zeros((len(h),3))
    hmasses = np.zeros(len(h))
    # Figure out what kind of method should work:
    massCalcMethod = getMassCalcMethod(h[pynbody_offset])
    for k in range(0,len(h)):
        hcentres[k,0] = h[k+pynbody_offset].properties['Xc']
        hcentres[k,1] = h[k+pynbody_offset].properties['Yc']
        hcentres[k,2] = h[k+pynbody_offset].properties['Zc']
        #hmasses[k] = h[k+1].properties['mass']
        hmasses[k] = getHaloMassByMethod(h[k+pynbody_offset],massCalcMethod)
    if inMpcs:
        hcentres /= 1000
    if remap:
        boxsize = h[pynbody_offset].properties['boxsize'].ratio("Mpc a h**-1")
        hcentres = tools.remapAntiHaloCentre(hcentres,boxsize)
    return [hcentres,hmasses]

def getHaloCentresAndMassesRecomputed(
        h,boxsize=677.7,fixedMass=True,pynbody_offset=0
    ):
    hcentres = np.zeros((len(h),3))
    hmasses = np.zeros(len(h))
    mUnit = h[pynbody_offset]['mass'].in_units("Msol h**-1")[0]*1e10
    for k in range(0,len(h)):
        if fixedMass:
            hcentres[k,:] = context.computePeriodicCentreWeighted(\
                h[k+pynbody_offset]['pos'],mUnit*np.ones(
                    len(h[k+pynbody_offset])
                ),\
                boxsize,accelerate=True)
            hmasses[k] = mUnit*len(h[k+pynbody_offset])
        else:
            hcentres[k,:] = context.computePeriodicCentreWeighted(\
                h[k+pynbody_offset]['pos'],h[k+pynbody_offset]['mass'],\
                boxsize,accelerate=True)
            hmasses[k] = np.sum(
                h[k+pynbody_offset]['mass'].in_units("Msol h**-1")
            )
    hcentres = tools.remapAntiHaloCentre(hcentres,boxsize)
    return [hcentres,hmasses]

def getAllHaloCentresAndMasses(snapList,recompute=False,\
        suffix = "halo_mass_and_centres",\
        function=getHaloCentresAndMassesFromCatalogue,
        pynbody_offset=0):
    massesAndCentresList = []
    for k in range(0,len(snapList)):
        print("Masses and centres for sample " + str(k+1) + " of " + \
            str(len(snapList)))
        if os.path.isfile(snapList[k].filename + "." + suffix) \
                and not recompute:
            massesAndCentresList.append(tools.loadPickle(\
                snapList[k].filename + "." + suffix))
        else:
            massesAndCentresList.append(
                tools.loadOrRecompute(\
                    snapList[k].filename + "." + suffix,function,\
                    snapList[k].halos(),
                    _recomputeData=recompute,pynbody_offset=pynbody_offset
                )
            )
    return massesAndCentresList

# Recentre clusters to account for simulation drift of cluster locations:
def getClusterCentres(approxCentre,snap=None,snapPath="snapshot_001",
        positions=None,density=None,recompute=True,fileSuffix='clusters',\
        reductions=3,iterations=10,method=None,\
        haloPos = None,haloMass = None,catalogPos=None,\
        gatherRadius = 5,neighbourRadius = 10,massProxy = None,\
        boxsize = None,cache=True,positionTree=None):
    if os.path.isfile(snapPath + "." + fileSuffix) and (not recompute):
        refinedPos =  pickle.load(open(snapPath + "." + fileSuffix,"rb"))
    else:
        # Select a method if none specified:
        if method is None:
            # Pick the first available method:
            if os.path.isfile(snapPath):
                method = "snapshot"
            elif ((density is not None) and (positions is not None)):
                method = "density"
            elif ((haloPos is not None) and (haloMass is not None) and \
                (catalogPos is not None) and (boxsize is not None)):
                method = "halo_match"
            elif os.path.isfile(snapPath + "." + fileSuffix):
                method = "load"
            else:
                raise Exception("No valid method is available.")
        # Verify that the data needed for each method has been provided:
        if not os.path.isfile(snapPath) and method == "snapshot":
            raise Exception("Snapshot is not available.")
        if ((density is None) or (positions is None) or (boxsize is None)) \
                and (method == "density"):
            raise Exception("Positions and Density field must be supplied.")
        if ((haloPos is None) or (haloMass is None) or (catalogPos is None) \
                or boxsize is None) and (method == "halo_match"):
            raise Exception(\
                "haloPos, haloMass, catalogPos, and boxsize must be supplied.")
        if not os.path.isfile(snapPath + "." + fileSuffix) \
                and method == "load":
            raise Exception("File with cluster locations must be provided.")
        # Compute the cluster centres using the relevant method:
        if method == "snapshot":
            if snap is None:
                snap = pynbody.load(snapPath)
                tools.remapBORGSimulation(snap)
            boxsize = snap.properties['boxsize'].ratio("Mpc a h**-1")
            refinedPos = massCentreAboutPoint(\
                snapedit.wrap(approxCentre,boxsize),snap['pos'],\
                boxsize,reductions=reductions,iterations=iterations,\
                tree = tools.getKDTree(snap))
        elif method == "density":
            if positionTree is None:
                positionTree = scipy.spatial.cKDTree(\
                    snapedit.wrap(positions,boxsize),boxsize=boxsize)
            refinedPos = massCentreAboutPoint(\
                snapedit.wrap(approxCentre,boxsize),positions,\
                boxsize,reductions=reductions,iterations=iterations,\
                tree = positionTree,weights=density)
        elif method == "halo_match":
            [counterpartClusters,counterpartHalos] = \
                matchClustersAndHalos(\
                    approxCentre,haloPos,haloMass,boxsize,catalogPos,\
                    gatherRadius = gatherRadius,\
                    neighbourRadius = neighbourRadius,massProxy = massProxy)
            refinedPos = np.zeros(approxCentre.shape)
            for k in range(0,len(approxCentre)):
                if counterpartHalos[k] < 0:
                    print("Warning - could not find counterpart halo for " + \
                        "cluster " + str(k+1) + ". Using input position.")
                    refinedPos[k,:] = approxCentre[k,:]
                else:
                    refinedPos[k,:] = haloPos[counterpartHalos[k],:]
        elif method == "load":
            with open(snapPath + "." + fileSuffix,"rb") as infile:
                refinedPos = pickle.load(infile)
            if refinedPos.shape != approxCentre.shape:
                raise Exception("Cached cluster locations are not a match " + \
                    "for supplied clusters.")
        else:
            raise Exception("Invalid method requested.")
        if cache:
            with open(snapPath + "." + fileSuffix,"wb") as outfile:
                pickle.dump(refinedPos,outfile)
    return refinedPos



# Master function to process snapshots. Note, this assumes AHF or another 
# halo finder has already been run, as well as ZOBOV, with the relevant 
# volumes data file moved to be in the same place.
def processSnapshot(standard,reverse,nBins,offset=4,output=None,rMax=3.0,\
        rMin=0.0,pynbody_offset = 0):
    """
        Process a pair of simulation snapshots to get data for constructing
        catalogues.
        
        Parameters:
            standard (string): Filename of the forward simulation
            reverse (string): Filename of the reverse simulation
            nBins (int): Number of radial bins to use for the void profile.
            offset (int): Bytes to ignore in the volumes file.
            output (string or None): Filename for the output file. If None,
                                     uses the default ".AHproperties.p"
                                     extension to the forward snapshot.
            rMax (float): Maximum radius (in units of void radius) out to which
                          to compute the void density profile.
            rMin (float): Minimum radius (in units of void radius) out to which
                          to compute the void density profile.
            pynbody_offset (int): Offset for accessing halo catalogues. Older
                                  versions of pynbody used halos[1] as the first
                                  entry, but this seems to have changed in 
                                  more recent versions. This parameter allows
                                  for backwards compatibility. Set to 1
                                  for older versions of pynbody.
    """
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
    [hncentres,hnmasses] = getHaloCentresAndMassesFromCatalogue(
        hn,remap=False,pynbody_offset=pynbody_offset
    )
    [hrcentres,hrmasses] = getHaloCentresAndMassesFromCatalogue(
        hr,remap=False,pynbody_offset=pynbody_offset
    )
    # Import the ZOBOV Voronoi information:
    haveVoronoi = os.path.isfile(standard + ".vols")
    if haveVoronoi:
        volumes = tools.zobovVolumesToPhysical(standard + ".vols",snapn,\
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
            snapn['pos'][sortedn[hr[k+pynbody_offset]['iord']],:],\
            volumes[sortedn[hr[k+pynbody_offset]['iord']]],periodicity,
            accelerate=True
        )
        antiHaloVolumes[k] = np.sum(
            volumes[sortedn[hr[k+pynbody_offset]['iord']]]
        )
    antiHaloRadii = np.cbrt(3*antiHaloVolumes/(4*np.pi))
    # Perform pair counting (speeds up computing density profiles, 
    # but needs to be recomputed if we want different bins):
    rBinStack = np.linspace(rMin,rMax,nBins)
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
            antiHaloRadii[k],snapn['pos'],volumes,snapn['mass'],\
            tree=tree,centralRatio = 4,nThreads=thread_count)/rhoBar - 1.0
        deltaAverage[k] = np.sum(hr[k+pynbody_offset]['mass'])/\
            np.sum(volumes[sortedn[hr[k+pynbody_offset]['iord']]])/rhoBar - 1.0
    pickle.dump([hncentres,hnmasses,hrcentres,hrmasses,volumes,\
      antiHaloCentres,antiHaloVolumes,antiHaloRadii,rBinStack,pairCounts,\
      volumesList,deltaCentral,deltaAverage],\
      open(output,"wb"))


# Get the density from a single snap-shot, about a particular point:
def density_from_snapshot(snap,centre,radius,tree=None):
    m_unit = snap['mass'][0]*1e10
    boxsize = snap.properties['boxsize'].ratio("Mpc a h**-1")
    Om0 = snap.properties['omegaM0']
    rho_crit = 2.7754e11
    rho_mean = rho_crit*Om0
    vol_sphere = 4*np.pi*radius**3/3
    if tree is None:
        tree = scipy.spatial.cKDTree(snap['pos'],boxsize=boxsize)
    return m_unit*tree.query_ball_point(centre,radius,\
            workers=-1,return_length=True)/(vol_sphere*rho_mean) - 1.0

# Get a random selection of centres:
def get_random_centres_and_densities(rSphere,snapListUn,\
        seed=1000,nRandCentres = 10000):
    # Get a random selection of centres:
    np.random.seed(seed)
    # Get random selection of centres and their densities:
    randOverDen = []
    snapSample = snapListUn[0]
    boxsize = snapSample.properties['boxsize'].ratio("Mpc a h**-1")
    randCentres = np.random.random((nRandCentres,3))*boxsize
    for k in range(0,len(snapListUn)):
        snap = snapListUn[k]
        gc.collect() # Clear memory of the previous snapshot
        tree = scipy.spatial.cKDTree(snap['pos'],boxsize=boxsize)
        gc.collect()
        randOverDen.append(density_from_snapshot(snap,randCentres,rSphere,
            tree = tree))
    return [randCentres,randOverDen]

# Get the distance between two points in a simulation, accounting for 
# wrapping:
def getDistanceBetweenCentres(centre1,centre2,boxsize):
    if (len(centre1.shape) == 1 and len(centre2.shape) == 1):
        return np.sqrt(np.sum(snapedit.unwrap(centre1 - centre2,boxsize)**2))
    else:
        return np.sqrt(np.sum(snapedit.unwrap(centre1 - centre2,boxsize)**2,1))

# From a list of centres, compute a sub-set which don't overlap by some radius
# rSep:
def getNonOverlappingCentres(centresList,rSep,boxsize,returnIndices=False):
    centresNonOverlapping = []
    indicesNonOverlapping = []
    for ns in range(0,len(centresList)):
        centresNS = centresList[ns]
        centresNSNonOverlap = []
        indicesNSNonOverlap = []
        for k in range(0,len(centresNS)):
            include = True
            for l in range(0,len(centresNSNonOverlap)):
                include = include and (getDistanceBetweenCentres(centresNS[k],\
                    centresNSNonOverlap[l],boxsize) > rSep)
            if include:
                centresNSNonOverlap.append(centresNS[k])
                indicesNSNonOverlap.append(k)
        centresNonOverlapping.append(np.array(centresNSNonOverlap))
        indicesNonOverlapping.append(np.array(indicesNSNonOverlap))
    if returnIndices:
        return indicesNonOverlapping
    else:
        return centresNonOverlapping


def get_mcmc_supervolume_densities(snap_list,r_sphere=135):
    boxsize = snap_list[0].properties['boxsize'].ratio("Mpc a h**-1")
    if np.isscalar(r_sphere):
        deltaMCMCList = np.array(\
            [density_from_snapshot(snap,np.array([boxsize/2]*3),r_sphere) \
             for snap in snap_list])
    else:
        deltaMCMCList = [[] for snap in snap_list]
        for snap in snap_list:
            tree = scipy.spatial.cKDTree(snap['pos'],boxsize=boxsize)
            deltaMCMCList[k] = [density_from_snapshot(snap,
                np.array([boxsize/2]*3),r_sphere[k],tree=tree) 
                for k in range(0,len(r_sphere))]
        deltaMCMCList = np.array(deltaMCMCList)
    return deltaMCMCList

def get_map_from_sample(sample):
    kde = scipy.stats.gaussian_kde(sample,bw_method="scott")
    return scipy.optimize.minimize(lambda x: -kde.evaluate(x),\
        np.mean(sample)).x[0]



# Get the positions of a snapshot in redshift space:
def redshift_space_positions(snap,centre=None):
    boxsize = snap.properties['boxsize'].ratio("Mpc a h**-1")
    if centre is None:
        centre = np.array([boxsize/2]*3)
    a = snap.properties['a']
    z = 1.0/a - 1.0
    Om = snap.properties['omegaM0']
    Ha = cosmology.Hz(z,Om,h=1) # Hubble rate / h
    r = snapedit.unwrap(snap['pos'] - centre,boxsize)
    r2 = np.sum(r**2,1)
    vr = np.sum(snap['vel']*r,1)
    # Assume gadget units:
    gamma = (np.sqrt(a)/Ha)
    return snapedit.wrap(
        (1.0 + gamma*vr/r2)[:,None]*r + centre,boxsize)

# Convert to los co-ordinates:
def get_los_pos(pos,los,boxsize):
    """
    Get the Line of Sight (LOS) co-ordinates of objects at given positions,
    along the specified line of sight. Accounts for box periodicity.
    
    Parameters:
        pos (N x 3 array): Positions of object in 3D space to 
        los (3x1 array): Vector representing the line of sight
        boxsize (float): Size of the periodic box
    
    Returns:
        N x 2 array: First column gives distance parallel to the LOS, 
                     second column distance perpendicular to the LOS.
    """
    los_unit = los/np.sqrt(np.sum(los**2))
    pos_rel = snapedit.unwrap(pos - los,boxsize)
    s_par = np.dot(pos_rel,los_unit)
    s_perp = np.sqrt(np.sum(pos_rel**2,1) - s_par**2)
    return np.vstack((s_par,s_perp)).T

# Get LOS positions, but only for the filtered voids:
def get_los_pos_with_filter(centres,filt,hr_list,void_indices,positions,
                            sorted_indices,boxsize,dist_max,tree,
                            all_particles=True,
                            pynbody_offset=0):
    los_pos_all = []
    for k in tools.progressbar(range(0,len(centres))):
        if filt[k]:
            if np.isscalar(dist_max):
                max_distance = dist_max
            else:
                max_distance = dist_max[k]
            if not all_particles:
                indices = hr_list[void_indices[k]+pynbody_offset]['iord']
                halo_pos = snapedit.unwrap(positions[sorted_indices[indices],:],
                    boxsize)
                distances = np.sqrt(
                    np.sum(snapedit.unwrap(halo_pos - centres[k],boxsize)**2,1))
                halo_pos = halo_pos[distances < max_distance,:]
            else:
                indices = tree.query_ball_point(
                    snapedit.wrap(centres[k],boxsize),max_distance,workers=-1)
                halo_pos = snapedit.unwrap(positions[indices,:],boxsize)
            los_pos = get_los_pos(halo_pos,centres[k],boxsize)
            los_pos_all.append(los_pos)
        else:
            los_pos_all.append(np.zeros((0,2)))
    return los_pos_all

def flatten_filter_list(filter_list):
    filt_combined = np.zeros(len(filter_list[0]),dtype=bool)
    for filt in filter_list:
        filt_combined = np.logical_or(filt_combined,filt)
    return filt_combined

# Get the line of sight (los) co-ordinates of selected particles in a snapshot:
def get_los_pos_for_snapshot(snapname_forward,snapname_reverse,centres,radii,
        dist_max=3,rmin=10,rmax=20,all_particles=True,filter_list=None,
        void_indices=None,sorted_indices=None,reverse_indices=None,
        positions = None,hr_list=None,tree=None,zspace=False,
        recompute_zspace=False,dist_cut_method="relative",
        flatten_filters=False):
    snap = tools.getPynbodySnap(snapname_forward)
    snap_reverse = tools.getPynbodySnap(snapname_reverse)
    if hr_list is None:
        hr_list = snap_reverse.halos()
    boxsize = snap.properties['boxsize'].ratio("Mpc a h**-1")
    # Sorted indices, to allow correct referencing of particles:
    if sorted_indices is None:
        if os.path.isfile(snap.filename + ".snapsort.p"):
            sorted_indices = tools.loadPickle(snap.filename + ".snapsort.p")
        else:
            sorted_indices = np.argsort(snap['iord'])
    if reverse_indices is None:
        reverse_indices = snap_reverse['iord'] # Force loading of reverse 
    # snapshot indices
    print("Sorting complete")
    # Remap positions into correct equatorial co-ordinates:
    if positions is None:
        if zspace:
            positions = tools.loadOrRecompute(
                snap.filename + ".z_space_pos.p",
                redshift_space_positions,snap,centre=np.array([boxsize/2]*3),
                _recomputeData=recompute_zspace)
        else:
            positions = snap['pos']
        positions = tools.remapAntiHaloCentre(positions,boxsize,
                                              swapXZ  = False,reverse = True)
    print("Positions computed")
    # Get relative positions of particles in each halo, remembering to 
    # account for wrapping:
    if all_particles and (tree is None):
        print("Generating Tree")
        tree = scipy.spatial.cKDTree(snapedit.wrap(positions,boxsize),
            boxsize=boxsize)
    print("Computing ellipticities...")
    if void_indices is None:
        void_indices = np.arange(0,len(hr_list))
    if filter_list is None:
        rad_filter = (radii > rmin) & (radii <= rmax)
    elif type(filter_list) is list:
        rad_filter = [filt & (radii > rmin) & (radii <= rmax) 
            for filt in filter_list]
    else:
        rad_filter = filter_list & (radii > rmin) & (radii <= rmax)
    if dist_cut_method == "relative":
        max_distance = dist_max*radii
    else:
        max_distance = dist_max
    if type(rad_filter) is list and (not flatten_filters):
        los_pos_all = []
        for filt in rad_filter:
            lost_pos_list = get_los_pos_with_filter(centres,filt,hr_list,
                                              void_indices,positions,
                                              sorted_indices,boxsize,
                                              max_distance,tree,
                                              all_particles=all_particles)
            los_pos_all.append(lost_pos_list)
    else:
        if flatten_filters and type(rad_filter) is list:
            combined_filter = flatten_filter_list(rad_filter)
        else:
            combined_filter = rad_filter
        los_pos_all = get_los_pos_with_filter(centres,combined_filter,hr_list,
                                              void_indices,positions,
                                              sorted_indices,boxsize,
                                              max_distance,tree,
                                              all_particles=all_particles)
    return los_pos_all

# The the los positions for selected particles in the group of snapshots
def get_los_positions_for_all_catalogues(snapList,snapListRev,
        antihaloCentres,antihaloRadii,recompute=False,filter_list=None,
        suffix=".lospos.p",void_indices=None,**kwargs):
    los_list = []
    if suffix == "":
        raise Exception("Suffix cannot be empty.")
    if void_indices is None:
        void_indices = [None for ns in range(0,len(snapList))]
    for ns in range(0,len(snapList)):
        # Load snapshot (don't use the snapshot list, because that will force
        # loading of all snapshot positions, using up a lot of memory, when 
        # we only want to load these one at a time):
        print("Doing sample " + str(ns+1) + " of " + str(len(snapList)))
        if filter_list is None:
            los_pos_all = tools.loadOrRecompute(
                snapList[ns].filename + suffix,
                get_los_pos_for_snapshot,snapList[ns].filename,
                snapListRev[ns].filename,antihaloCentres[ns],antihaloRadii[ns],
                _recomputeData=recompute,void_indices=void_indices[ns],**kwargs)
        else:
            los_pos_all = tools.loadOrRecompute(
                snapList[ns].filename + suffix,
                get_los_pos_for_snapshot,snapList[ns].filename,
                snapListRev[ns].filename,antihaloCentres[ns],
                antihaloRadii[ns],filter_list=filter_list[ns],
                _recomputeData=recompute,void_indices=void_indices[ns],**kwargs)
        los_list.append(los_pos_all)
        del los_pos_all
        gc.collect()
    return los_list


def sample_from_distribution(pdf, x_range, n_samples=1, max_pdf_value=None,
                             max_attempts = None):
    """
    Generate random samples from an arbitrary distribution using rejection 
    sampling.
    
    Parameters:
    - pdf: function, the probability density function 
           (doesn't need to be normalized).
    - x_range: tuple, the range (min, max) of x values to sample from.
    - n_samples: int, the number of samples to generate.
    - max_pdf_value: float, the maximum value of the pdf over x_range. 
                     If None, it will be estimated.

    Returns:
    - samples: np.ndarray, an array of sampled values.
    """
    xmin, xmax = x_range
    samples = []
    attempts = 0
    if max_attempts is None:
        max_attempts = n_samples * 10  # To avoid infinite loops
    # Estimate max_pdf_value if not provided
    if max_pdf_value is None:
        x_vals = np.linspace(xmin, xmax, 1000)
        max_pdf_value = max(pdf(x) for x in x_vals)
    while len(samples) < n_samples and attempts < max_attempts:
        x = np.random.uniform(xmin, xmax)
        y = np.random.uniform(0, max_pdf_value)
        if y < pdf(x):
            samples.append(x)
        attempts += 1
    if len(samples) < n_samples:
        raise RuntimeError(
            "Sampling failed. Try increasing max_pdf_value or max_attempts."
        )
    return np.array(samples)

class DummySnapshot:
    def __init__(self,pos,vel,mass,**properties):
        self.keydict = {'pos':pos,'vel':vel,'mass':mass}
        self.properties = properties
    def __getitem__(self, property_name):
        return self.keydict[property_name]
    def __len__(self):
        return len(self.keydict['pos'])


def gaussian_delta(r,A=0.85,sigma=1):
    """
    Gaussian model for the void contrast of a void (mostly used for testing)
    
    Parameters:
        r (float or array): Radial distance from void centre
        A (float): Amplitude, equal to the negative of the Central density 
                   contrast
        sigma (float): Width of the Guassian (roughly the void radius)
    
    Returns:
        float or array: Density contrast at r
    """
    return -A*np.exp(-0.5*(r/sigma)**2)

def gaussian_Delta(r,A=0.85,sigma=1,thresh=1e-5,exact=False,taylor=False):
    """
    Gaussian model for the cumulative void contrast of a void. Integral of
    3/r^3\int(x**2*guassian_Delta(x,A,sigma)dx)_0^{r}
    
    Parameters:
        r (float or array): Radial distance from void centre
        A (float): Amplitude, equal to the negative of the Central density 
                   contrast
        sigma (float): Width of the Guassian (roughly the void radius)
        thresh (float): If (r/sigma)^2 < thresh, switch to a Taylor expansion
                        to avoid problems near r = 0
        exact (bool): If true, return the exact expression even if this would
                      have problems near r = 0
        taylor (bool): If true, force the use of the Taylor expansion around
                       r = 0
    
    Returns:
        float or array: Density contrast at r
    """
    if exact:
        # Exact expression. This is numerically unstable near r = 0
        return - 3*A*np.sqrt(np.pi/2)*(sigma/r)**3*scipy.special.erf(
                   (r/sigma)/np.sqrt(2)
               ) + 3*A*(sigma/r)**2*np.exp(-0.5*(r/sigma)**2)
    if taylor:
        # Taylor expansion around r = 0:
        return -A*(1 - (r/sigma)**2/10 + (r/sigma)**4/56)
    if np.isscalar(r):
        if (r/sigma)**2 > 1e-5:
            return gaussian_Delta(r,A=A,sigma=sigma,exact=True)
        else:
            return gaussian_Delta(r,A=A,sigma=sigma,taylor=True)
    else:
        small = (r/sigma)**2 < thresh
        not_small = np.logical_not(small)
        value = np.zeros(r.shape)
        value[small] = gaussian_Delta(r[small],A=A,sigma=sigma,taylor=True)
        value[not_small] = gaussian_Delta(
            r[not_small],A=A,sigma=sigma,exact=True
        )
        return value

def generate_synthetic_void_snap(N=32,rmax=50,A=0.85,sigma=10,seed=0,H0=70):
    """
        Generate a synthetic snapshot with a void, which has realistic
        velocities. Mostly used for testing purposes.
    """
    np.random.seed(seed)
    # Random distribution on a sphere:
    thetas = np.arcsin(np.random.rand(N**3)*2 - 1)
    phis = np.random.rand(N**3)*2*np.pi
    # Radii randomly distributed according to a specified distribution:
    rho_void = lambda r: 1 + gaussian_delta(r,A=A,sigma=sigma)
    # Cumulative density contrast, obtained by integrating:
    Delta_void = lambda r: gaussian_Delta(r,A=A,sigma=sigma)
    rs = sample_from_distribution(
        lambda r: rho_void(r)*r**2,[0,rmax],n_samples=N**3,max_pdf_value = None
    )
    xs, ys, zs = [
        rs*np.cos(thetas)*np.cos(phis),
        rs*np.cos(thetas)*np.sin(phis),
        rs*np.sin(thetas)]
    pos = np.vstack([xs,ys,zs]).T
    # Generate velocities using a 1LPT approximation, given the known density:
    rvec = pos/rs[:,None] # Unit vectors from centre of void
    vr = -100*0.53*(Delta_void(rs)/3)*rs # Radial velocities
    # Velocities in 3D, with Gaussian noise:
    vel = rvec*vr[:,None] + np.random.randn(N**3,3)*3
    # Fake masses:
    G = 6.67e-11
    Mpc = 3.0857e22
    Msol = 1.989e30
    rho_crit = (3*(1e5/Mpc)**2/(8*np.pi*G))*Mpc**3/Msol
    Munit = (4*np.pi*rmax**3/3)*rho_crit/N**3
    mass = np.full(rs.shape,Munit)
    return DummySnapshot(
        pos,vel,mass,boxsize=2*rmax*pynbody.units.Unit("Mpc a h**-1")
    )


def get_borg_density_estimate(snaps, densities_file=None, dist_max=135,
                              seed=1000, interval=0.68,nboot=9999):
    """
    Estimate the density contrast in a specified subvolume using BORG snapshots.

    If precomputed density samples are provided, loads them from file.
    Otherwise, computes them from raw snapshot data using spherical averaging.

    Returns a bootstrap estimate of the MAP (maximum a posteriori) density 
    contrast and its uncertainty interval.

    Parameters:
        snaps (SnapHandler): Object containing snapshots in `snaps["snaps"]`
        densities_file (str or None): Pickle file containing precomputed delta 
                                      samples
        dist_max (float): Radius (in Mpc/h) for subvolume used to compute 
                          densities
        seed (int): RNG seed for reproducibility of bootstrap
        interval (float): Confidence level for bootstrap interval (e.g., 0.68)

    Returns:
        tuple:
            - deltaMAPBootstrap (BootstrapResult): Bootstrap distribution 
                                                   object
            - deltaMAPInterval (ConfidenceInterval): Confidence interval of 
                                                     MAP estimate
    
    Tests:
        Tested in test_simulation_tools.py
        Regression tests: test_get_borg_density_estimate
    """
    boxsize = snaps.boxsize
    # Determine center of sphere based on particle positions
    if np.min(snaps["snaps"][0]["pos"]) < 0:
        centre = np.array([0, 0, 0])
    else:
        centre = np.array([boxsize / 2] * 3)
    # Load or compute density samples
    if densities_file is not None:
        deltaMCMCList = tools.loadPickle(densities_file)
    else:
        deltaMCMCList = np.array([
            density_from_snapshot(snap, centre, dist_max)
            for snap in snaps["snaps"]
        ])
    # Bootstrap MAP density estimator
    # Setup an explicit random number generator rathert than passing the
    # seed to scipy, as this ensures greater consistency between 
    # architectures:
    rng = np.random.default_rng(seed)
    deltaMAPBootstrap = scipy.stats.bootstrap(
        (deltaMCMCList,),
        get_map_from_sample,
        confidence_level=interval,
        vectorized=False,
        random_state=rng,
        n_resamples=nboot
    )
    return deltaMAPBootstrap, deltaMAPBootstrap.confidence_interval



#-------------------------------------------------------------------------------
# SNAPSHOT GROUP CLASS

def get_antihalo_properties(snap, file_suffix="AHproperties",
                            default=".h5", low_memory_mode=True):
    """
    Load anti-halo property data for a given snapshot.

    Attempts to load from an HDF5 file (.h5), and falls back to a legacy 
    pickle format (.p) if necessary. Supports a low-memory mode that avoids 
    loading the data into memory unless explicitly requested.

    Parameters:
        snap (pynbody.Snapshot): The simulation snapshot to load from.
        file_suffix (str): Suffix for the filename (default: "AHproperties").
        default (str): File extension to look for first (default: ".h5").
        low_memory_mode (bool): If True, return just a file reference 
                                instead of loading full data.

    Returns:
        If low_memory_mode is True and using legacy format:
            str: Filename for later loading.
        Else:
            h5py.File or list: Anti-halo data loaded from file.
    Tests:
        No tests implemented
    """
    if isinstance(snap,str):
        snap = pynbody.load(snap)
    filename = snap.filename + "." + file_suffix + default
    filename_old = snap.filename + "." + file_suffix + ".p"
    if os.path.isfile(filename):
        return h5py.File(filename, "r")
    elif os.path.isfile(filename_old):
        if low_memory_mode:
            return filename_old
        else:
            return tools.loadPickle(filename_old)
    else:
        raise Exception("Anti-halo properties file not found.")

class SnapshotGroup:
    """
    A container for managing forward and reverse simulation snapshots and 
    accessing their associated void/halo properties.

    This class handles:
    - Loading multiple simulation snapshots (and their time-reversed 
        counterparts)
    - Accessing halo and void properties (e.g., positions, masses, radii)
    - Remapping coordinates to Equatorial space via configurable transformations
    - Efficient memory usage through optional lazy loading and caching

    Coordinate System Note:
    The underlying simulation snapshots may not be in Equatorial coordinates.
    To convert to Equatorial space, properties like void centres or halo 
    positions are remapped via `tools.remapAntiHaloCentre`, which applies:
        - A shift to box-centred coordinates
        - Optional axis swapping (swapXZ=True swaps X and Z)
        - Optional axis flipping (reverse=True mirrors positions)

    These transformations are controlled by the `swapXZ` and `reverse` flags. 
    Their default values assume a particular snapshot orientation, but users
    should verify and adjust these flags if working with different conventions.
    
    Tests:
        No tests implemented
    """

    def __init__(self, snap_list, snap_list_reverse, low_memory_mode=True,
                 swapXZ=False, reverse=False, remap_centres=False):
        """
        Initialize a SnapshotGroup from lists of forward and reverse snapshots.

        Parameters:
            snap_list (list): Forward-time simulation snapshots
            snap_list_reverse (list): Corresponding reverse-time snapshots
            low_memory_mode (bool): If True, avoid loading all properties into 
                                    memory
            swapXZ (bool): Whether to swap X and Z axes when remapping 
                           coordinates
            reverse (bool): Whether to flip coordinates around box center
            remap_centres (bool): (Not yet implemented - ignored) Whether to 
                                  remap void/halo centres to Equatorial frame.
        """
        self.snaps = [tools.getPynbodySnap(snap) for snap in snap_list]
        self.snaps_reverse = [tools.getPynbodySnap(snap) 
                              for snap in snap_list_reverse]
        self.N = len(self.snaps)
        self.low_memory_mode = low_memory_mode
        if low_memory_mode:
            self.all_property_lists = [None for snap in snap_list]
        else:
            self.all_property_lists = [
                get_antihalo_properties(snap,low_memory_mode=low_memory_mode) 
                for snap in snap_list
            ]
        self.property_list = [
            "halo_centres", "halo_masses",
            "antihalo_centres", "antihalo_masses",
            "cell_volumes", "void_centres",
            "void_volumes", "void_radii",
            "radius_bins", "pair_counts",
            "bin_volumes", "delta_central",
            "delta_average"
        ]
        self.additional_properties = {
            "halos": None,
            "antihalos": None,
            "snaps": self.snaps,
            "snaps_reverse": self.snaps_reverse,
            "trees": None,
            "trees_reverse": None
        }
        self.property_map = {
                                name: idx for idx, name in enumerate(
                                   self.property_list)
                            }
        self.reverse = reverse
        self.swapXZ = swapXZ
        self.remap_centres = remap_centres
        self.boxsize = self.snaps[0].properties['boxsize'].ratio("Mpc a h**-1")
        self.all_properties = [None for _ in self.property_list]
        self.snap_filenames = [snap.filename for snap in self.snaps]
        self.snap_reverse_filenames = [
            snap.filename for snap in self.snaps_reverse
        ]
    def is_valid_property(self, prop):
        if isinstance(prop, int):
            return prop in range(len(self.property_list))
        elif isinstance(prop, str):
            return prop in self.property_list
        return False
    def get_property_index(self, prop):
        if isinstance(prop, int):
            if prop in range(len(self.property_list)):
                return prop
            raise Exception("Property index is out of range.")
        elif isinstance(prop, str):
            if prop in self.property_list:
                return self.property_map[prop]
            raise Exception("Requested property does not exist.")
        else:
            raise Exception("Invalid property type")
    def get_property_name(self, prop):
        if isinstance(prop, int):
            if prop in range(len(self.property_list)):
                return self.property_list[prop]
            raise Exception("Property index is out of range.")
        elif isinstance(prop, str):
            if prop in self.property_list:
                return prop
            raise Exception("Requested property does not exist.")
        else:
            raise Exception("Invalid property type")
    def get_property(self, snap_index, property_name, recompute=False):
        """
        Access a property for a single snapshot, loading from cache or disk.
        """
        prop_index = self.get_property_index(property_name)
        if self.all_properties[prop_index] is not None and not recompute:
            return self.all_properties[prop_index][snap_index]

        property_list = self.all_property_lists[snap_index]
        if property_list is None:
            property_list = get_antihalo_properties(self.snaps[snap_index])

        if isinstance(property_list, h5py._hl.files.File):
            return property_list[self.get_property_name(property_name)]
        elif isinstance(property_list, list):
            return property_list[self.get_property_index(property_name)]
        elif isinstance(property_list, str):
            props_list = tools.loadPickle(property_list)
            return props_list[self.get_property_index(property_name)]
        else:
            raise Exception("Invalid Property Type")
    def check_remappable(self, property_name):
        """
        Check if a property represents a position needing coordinate remapping.
        """
        index = self.get_property_index(property_name)
        return index in [0, 5]  # halo_centres or void_centres
    def get_all_properties(self, property_name, cache=True, recompute=False):
        """
        Get a list of a given property for all snapshots.
        Handles remapping to Equatorial coordinates if applicable.
        """
        prop_index = self.get_property_index(property_name)
        if self.all_properties[prop_index] is None:
            if self.check_remappable(property_name):
                # Remap positions to Equatorial coordinates
                properties = [
                    tools.remapAntiHaloCentre(
                        self.get_property(
                            i, property_name, recompute=recompute
                        ),
                        boxsize=self.boxsize,
                        swapXZ=self.swapXZ,
                        reverse=self.reverse)
                    for i in range(self.N)
                ]
            else:
                properties = [
                    self.get_property(i, property_name, recompute=recompute)
                    for i in range(self.N)
                ]
            if cache:
                self.all_properties[prop_index] = properties
            return properties
        else:
            return self.all_properties[prop_index]
    def __getitem__(self, property_name):
        """
        Enable bracket-access to named or additional properties.
        Automatically returns all-snapshot versions of properties.
        """
        if self.is_valid_property(property_name):
            return self.get_all_properties(property_name)
        elif (
            isinstance(property_name, str)
            and property_name in self.additional_properties
        ):
            if self.additional_properties[property_name] is not None:
                return self.additional_properties[property_name]
            else:
                # Lazy-load derived properties
                if property_name == "halos":
                    self.additional_properties["halos"] = [
                        snap.halos() for snap in self.snaps
                    ]
                elif property_name == "antihalos":
                    self.additional_properties["antihalos"] = [
                        snap.halos() for snap in self.snaps_reverse
                    ]
                elif property_name == "trees":
                    self.additional_properties["trees"] = [
                        scipy.spatial.cKDTree(snap['pos'],boxsize=self.boxsize)
                        for snap in self.snaps
                    ]
                elif property_name == "trees_reverse":
                    self.additional_properties["trees_reverse"] = [
                        scipy.spatial.cKDTree(snap['pos'],boxsize=self.boxsize)
                        for snap in self.snaps_reverse
                    ]
                else:
                    raise Exception("Invalid property_name")
                return self.additional_properties[property_name]
        else:
            raise Exception("Invalid property_name")


def filter_regions_by_density(rand_centres, rand_densities, delta_interval):
    """
    Select underdense regions within a delta contrast range.

    Parameters:
        rand_centres (array, N x 3): Centres to search through
        rand_densities (array, N): Density constrasts in a sphere around
                                   each centre in rand_centres.
        delta_interval (tuple, 2 components): Density range to filter for.

    Returns:
        - region_masks: Boolean masks for selected centres
        - centres_to_use: Filtered centres as list of arrays per snapshot
    
    Tests:
        Tested in test_simulation_tools.py
        Regression tests: test_filter_regions_by_density
    """
    if isinstance(rand_centres,list):
        centres_list = rand_centres
    elif isinstance(rand_centres,np.ndarray):
        centres_list = [rand_centres for _ in rand_densities]
    else:
        raise Exception("Invalid rand_centres")
    if delta_interval is not None:
        interval = np.sort(delta_interval)
        region_masks = [
            (deltas > interval[0]) & (deltas <= interval[1])
            for deltas in rand_densities
        ]
        centres_to_use = [
            centres[mask,:] for centres, mask in zip(centres_list, region_masks)
        ]
    else:
        region_masks = [
            np.ones_like(deltas, dtype=bool) for deltas in rand_densities
        ]
        centres_to_use = centres_list
    return region_masks, centres_to_use

def compute_void_distances(void_centres, region_centres, boxsize):
    """
    Compute distances from each void to every selected region.
    
    Parameters:
        void_centres (list of array, each N x 3): Void centres to compute 
                                                  distances for, in each
                                                  region of the simulation
                                                  considered.
        region_centres (array, M x 3): Centres of each region. M should match
                                       the length of void_centres
        boxsize (float): Size of the periodic box.

    Returns:
        list of lists: [ [distances for region 1], [region 2], ... ] per 
                        snapshot
    
    Tests:
        Tested in test_simulation_tools.py
        Regression tests: test_compute_void_distances
    """
    return [[
        np.sqrt(np.sum(snapedit.unwrap(voids - region, boxsize)**2, axis=1))
        for region in regions
    ] for voids, regions in zip(void_centres, region_centres)]

def filter_voids_by_distance_and_radius(
        dist_lists, radii_lists, dist_max, radii_range
    ):
    """
    Apply filtering to voids based on spatial and size constraints.
    
    Parameters:
        dist_lists (list): List of arrays, each containing the distances of
                           voids from a common region centre.
        radii_lists (list): List of arrays, each containing the radii of voids
                            in a common region. Should be the same length as
                            the corresponding array in dist_lists.
        dist_max (float): Maximum distance out to which to include voids.
        radii_range (list, 2 components): Range of radii to filter for.

    Returns:
        list of lists of boolean arrays: One list per region per snapshot
    
    Tests:
        Tested in test_simulation_tools.py
        Regression tests: test_filter_voids_by_distance_and_radius
    """
    radii_range = np.sort(radii_range)
    return [[
        (dist < dist_max) & (radii > radii_range[0]) & (radii <= radii_range[1])
        for dist in region_dists
    ] for region_dists, radii in zip(dist_lists, radii_lists)]






















