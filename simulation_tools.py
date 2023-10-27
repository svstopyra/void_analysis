# Tools to compute properties of simulations.
import pynbody
import numpy as np
import astropy
import numexpr as ne
from .halos import massCentreAboutPoint
import scipy
from . import tools, snapedit, context, stacking
import pickle
import os
import multiprocessing as mp
thread_count = mp.cpu_count()
from astropy.coordinates import SkyCoord
import astropy.units as u
import gc

# Convert eulerian co-ordinate to redshift space co-ordinates:
def eulerToZ(pos,vel,cosmo,boxsize,h,centre = None,Ninterp=1000,\
        l = 268,b = 38,vl=540,localCorrection = True,velFudge = 1):
    # Compute comoving distance:
    if centre is not None:
        dispXYZ = snapedit.unwrap(pos - centre,boxsize)
    else:
        dispXYZ = pos
    r = np.sqrt(np.sum(dispXYZ**2,1))
    rq = r*astropy.units.Mpc/h
    # Compute relationship between comoving distance and redshift at discrete points.
    # For speed, we will interpolate any other points as computing it for each point
    # individually requires an expensive function inversion:
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
        vhat = np.array([local.icrs.cartesian.x.value,local.icrs.cartesian.y.value,\
            local.icrs.cartesian.z.value])
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

# Computes the density field of a pynbody snapshot using a naiive cloud-in-cell
# scheme (counts particles in a give cubic cell)
def getGriddedDensity(snap,N,redshiftSpace= False,velFudge = 1,snapPos = None,snapVel = None,
        snapMass = None):
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
            weights = snapMass,normed=False)
    cellWidth = boxsize/N
    cellVol = cellWidth**3
    meanDensity = np.double(np.sum(snapMass))/(boxsize**3)
    density = H/(cellVol*meanDensity)
    # Deal with an ordering issue:
    density = np.reshape(np.reshape(density,N**3),(N,N,N),order='F')
    return density

# Compute the points within a co-ordinate range, accounting for periodic 
# wrapping.
def pointsInRangeWithWrap(positions,lim,axis=2,boxsize=None):
    if boxsize is None:
        wrappedPositions = positions
        wrappedLim = np.array(lim)
    else:
        wrappedPositions = snapedit.unwrap(positions,boxsize)
        wrappedLim = snapedit.unwrap(np.array(lim),boxsize)
    return (wrappedPositions[:,axis]>= wrappedLim[0]) & (wrappedPositions[:,axis] <= wrappedLim[1])

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
    return (wrappedPositions[:,0] >= wrappedXLim[0]) & (wrappedPositions[:,0] <= wrappedXLim[1]) & (wrappedPositions[:,1] >= wrappedYLim[0]) & (wrappedPositions[:,1] <= wrappedYLim[1])

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
        inMpcs = True):
    hcentres = np.zeros((len(h),3))
    hmasses = np.zeros(len(h))
    # Figure out what kind of method should work:
    massCalcMethod = getMassCalcMethod(h[1])
    for k in range(0,len(h)):
        hcentres[k,0] = h[k+1].properties['Xc']
        hcentres[k,1] = h[k+1].properties['Yc']
        hcentres[k,2] = h[k+1].properties['Zc']
        #hmasses[k] = h[k+1].properties['mass']
        hmasses[k] = getHaloMassByMethod(h[k+1],massCalcMethod)
    if inMpcs:
        hcentres /= 1000
    if remap:
        boxsize = h[1].properties['boxsize'].ratio("Mpc a h**-1")
        hcentres = tools.remapAntiHaloCentre(hcentres,boxsize)
    return [hcentres,hmasses]

def getHaloCentresAndMassesRecomputed(h,boxsize=677.7,fixedMass=True):
    hcentres = np.zeros((len(h),3))
    hmasses = np.zeros(len(h))
    mUnit = h[1]['mass'].in_units("Msol h**-1")[0]*1e10
    for k in range(0,len(h)):
        if fixedMass:
            hcentres[k,:] = context.computePeriodicCentreWeighted(\
                h[k+1]['pos'],mUnit*np.ones(len(h[k+1])),\
                boxsize,accelerate=True)
            hmasses[k] = mUnit*len(h[k+1])
        else:
            hcentres[k,:] = context.computePeriodicCentreWeighted(\
                h[k+1]['pos'],h[k+1]['mass'],\
                boxsize,accelerate=True)
            hmasses[k] = np.sum(h[k+1]['mass'].in_units("Msol h**-1"))
    hcentres = tools.remapAntiHaloCentre(hcentres,boxsize)
    return [hcentres,hmasses]

def getAllHaloCentresAndMasses(snapList,boxsize,recompute=False,\
        suffix = "halo_mass_and_centres",\
        function=getHaloCentresAndMassesFromCatalogue):
    massesAndCentresList = []
    for k in range(0,len(snapList)):
        print("Masses and centres for sample " + str(k+1) + " of " + \
            str(len(snapList)))
        if os.path.isfile(snapList[k].filename + "." + suffix) \
                and not recompute:
            massesAndCentresList.append(tools.loadPickle(\
                snapList[k].filename + "." + suffix))
        else:
            massesAndCentresList.append(tools.loadOrRecompute(\
                snapList[k].filename + "." + suffix,function,\
                snapList[k].halos(),boxsize=boxsize,_recomputeData=recompute))
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
        rMin=0.0):
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
    [hncentres,hnmasses] = getHaloCentresAndMassesFromCatalogue(hn,remap=False)
    [hrcentres,hrmasses] = getHaloCentresAndMassesFromCatalogue(hr,remap=False)
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
            snapn['pos'][sortedn[hr[k+1]['iord']],:],\
            volumes[sortedn[hr[k+1]['iord']]],periodicity,accelerate=True)
        antiHaloVolumes[k] = np.sum(volumes[sortedn[hr[k+1]['iord']]])
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
        deltaAverage[k] = np.sum(hr[k+1]['mass'])/\
            np.sum(volumes[sortedn[hr[k+1]['iord']]])/rhoBar - 1.0
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




