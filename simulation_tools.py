# Tools to compute properties of simulations.
import pynbody
import numpy as np
import astropy
import numexpr as ne


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
    density = np.reshape(np.reshape(density,256**3),(256,256,256),order='F')
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
            (-boxsize/2,boxsize/2)),normed=False)
    # Deal with an ordering issue:
    H = np.reshape(np.reshape(H,256**3),(256,256,256),order='F')
    return H

# bias function, per realisation:
def ngBias(nmeans,bs,delta,lumBins = 16,nReal = 6):
    beta = bs[:,:,0]
    rhog = bs[:,:,1]
    epsg = bs[:,:,2]
    ng = np.zeros(delta.shape)
    for l in range(0,nReal):
        if len(delta[l].shape) == 3:
            deltaUse = np.reshape(delta[l],N**3)
        else:
            deltaUse = delta[l]
        deltaArray = np.tile(deltaUse,(lumBins,1)).transpose()
        resArray = bias['nmeans'][l,:,0]*np.power(1.0 + deltaArray,beta[l,:])*\
                np.exp(-rhog[l,:]*np.power(1.0 + deltaArray,-epsg[l,:]))
        ng[l] = np.sum(resArray,1)
    return ng

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



