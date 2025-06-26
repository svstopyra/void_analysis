# A collection of tools used for various aspects of the analysis.
import scipy
import scipy.spatial, scipy.stats, scipy.fft
import numpy as np
try:
    import numexpr as ne
except:
    ne = None
    print("WARNING: numexpr not found. Accelerated calculations disabled.")

import alphashape
import pynbody
from . import snapedit, plot_utilities
import pickle
import os
import traceback, logging
try:
    import h5py
except:
    h5py = None
    print("WARNING: h5py not found. Some functionality is disabled.")

from astropy.coordinates import SkyCoord
import astropy.units as u
import astropy
import healpy
import sys
import time

# Fetch all points that lie within a given radius of a specified centres, accounting for wrapping,
# and filtering on arbitrary conditions:
def getAntiHalosInSphere(centres,radius,origin=np.array([0,0,0]),
        deltaCentral = None,boxsize=None,workers=-1,filterCondition = None):
    if filterCondition is None:
        filterCondition = np.ones(len(centres),dtype=bool)
    if deltaCentral is not None:
        usedIndices = np.where((deltaCentral < 0) & filterCondition)[0]
        centresToUse = centres[usedIndices,:]
    else:
        usedIndices = np.where(filterCondition)[0]
        centresToUse = centres[usedIndices,:]
    if boxsize is not None:
        tree = scipy.spatial.cKDTree(snapedit.wrap(centresToUse,boxsize),boxsize=boxsize)
    else:
        tree = scipy.spatial.cKDTree(centresToUse,boxsize=boxsize)
    inRadius = tree.query_ball_point(origin,radius,workers=workers)
    if len(origin.shape) == 1:
        inRadiusFinal = list(usedIndices[inRadius])
        condition = np.zeros(len(centres),dtype=bool)
        condition[inRadiusFinal] = True
    else:
        inRadiusFinal = np.array([list(usedIndices[k]) for k in inRadius])
        condition = np.zeros((len(centres),len(origin)),dtype=bool)
        for k in range(0,len(origin)):
            condition[inRadiusFinal[k],k] = True
    return [inRadiusFinal,condition]

# Get the displacement of postions from centre, assuming periodicity:
def getPeriodicDisplacement(positions,centre,boxsize=None):
    if boxsize is None:
        return positions - centre
    else:
        return snapedit.unwrap(positions - centre,boxsize)

# Distance in a periodic box:
def getPeriodicDistance(positions,centre,boxsize = None):
    disp = getPeriodicDisplacement(positions,centre,boxsize=boxsize)
    if len(positions.shape) < 2:
        return np.sqrt(np.sum(disp**2))
    else:
        return np.sqrt(np.sum(disp**2,1))


# Density contras in centres in a snapshot:
def getCentredDensityConstrast(snap,centres,radius):
    tree = getKDTree(snap)
    mUnit = snap['mass'][0]*1e10
    volSphere = 4*np.pi*radius**3/3
    rhoMean = 2.7754e11*snap.properties['omegaM0']
    return mUnit*tree.query_ball_point(centres,radius,\
            workers=-1,return_length=True)/(volSphere*rhoMean) - 1.0


def getHaloAndAntihaloCountsInDensityRange(radius,snap,centres,deltaList,
        mThresh,hncentres,hrcentres,hnmasses,
        hrmasses,deltaCentral,deltaLow=-0.07,deltaHigh=-0.06,workers=-1):
    similar = np.where((deltaList > deltaLow) & (deltaList <= deltaHigh))[0]
    haloCount = np.zeros(len(similar),dtype=int)
    antihaloCount = np.zeros(len(similar),dtype=int)
    boxsize = snap.properties['boxsize'].ratio("Mpc a h**-1")
    haloTree = scipy.spatial.cKDTree(hncentres[np.where(hnmasses > mThresh)[0]],
        boxsize)
    antihaloTree = scipy.spatial.cKDTree(hrcentres[np.where(hrmasses > mThresh)[0]],
        boxsize)
    haloCount = haloTree.query_ball_point(centres[similar],radius,
        return_length=True,workers=workers)
    antihaloCount = antihaloTree.query_ball_point(centres[similar],radius,
        return_length=True,workers=workers)
    return [haloCount,antihaloCount]

# Given two halo catalogues, find which halos are likely to correspond to each other. 
# This means finding halos that are within a specified distance.
def getEquivalents(hn1,hn2,centres1,centres2,boxsize,rSearch):
    tree1 = scipy.spatial.cKDTree(centres1,boxsize=boxsize)
    tree2 = scipy.spatial.cKDTree(centres2,boxsize=boxsize)
    indicesNear = tree1.query_ball_tree(tree2,rSearch)
    equivalent = np.zeros(len(hn1),dtype=int)
    for k in range(0,len(hn1)):
        if len(indicesNear[k]) == 0:
            equivalent[k] = -1
        else:
            equivalent[k] = indicesNear[k][0]
    return equivalent

# Load cluster data from the Abell catalogue, filtering for galaxy clusters with known redshift
def loadAbellCatalogue(folder,filterForKnownZ = True):
    file3 = open(folder + "/table3.dat",'r')
    fileLines3 = []
    for line in file3:
        fileLines3.append(line)
    file3.close()
    abell_l3 = np.zeros(len(fileLines3))
    abell_b3 = np.zeros(len(fileLines3))
    abell_n3 = np.zeros(len(fileLines3),dtype=int)
    abell_z3 = np.zeros(len(fileLines3))
    for k in range(0,len(fileLines3)):
        abell_n3[k] = int(fileLines3[k][0:4])
        abell_l3[k] = np.double(fileLines3[k][118:124])
        abell_b3[k] = np.double(fileLines3[k][125:131])
        abell_z3[k] = np.double(fileLines3[k][133:138].replace(' ','0'))
    # Indicate missing redshifts:
    abell_z3[np.where(abell_z3 == 0.0)] = -1
    # file 4:
    file4 = open(folder + "/table4.dat",'r')
    fileLines4 = []
    for line in file4:
        fileLines4.append(line)
    file4.close()
    abell_l4 = np.zeros(len(fileLines4))
    abell_b4 = np.zeros(len(fileLines4))
    abell_n4 = np.zeros(len(fileLines4),dtype=int)
    abell_z4 = np.zeros(len(fileLines4))
    for k in range(0,len(fileLines4)):
        abell_n4[k] = int(fileLines4[k][0:4])
        abell_l4[k] = np.double(fileLines4[k][118:124])
        abell_b4[k] = np.double(fileLines4[k][125:131])
        abell_z4[k] = np.double(fileLines4[k][133:138].replace(' ','0'))
    # Indicate missing redshifts:
    abell_z4[np.where(abell_z4 == 0.0)] = -1
    if filterForKnownZ:
        havez3 = np.where(abell_z3 > 0)
        havez4 = np.where(abell_z4 > 0)
        abell_l = np.hstack((abell_l3[havez3],abell_l4[havez4]))
        abell_b = np.hstack((abell_b3[havez3],abell_b4[havez4]))
        abell_n = np.hstack((abell_n3[havez3],abell_n4[havez4]))
        abell_z = np.hstack((abell_z3[havez3],abell_z4[havez4]))
    else:
        abell_l = np.hstack((abell_l3,abell_l4))
        abell_b = np.hstack((abell_b3,abell_b4))
        abell_n = np.hstack((abell_n3,abell_n4))
        abell_z = np.hstack((abell_z3,abell_z4))
    c = 299792.458 # Speed of light in km/s
    abell_d = c*abell_z/100
    return [abell_l,abell_b,abell_n,abell_z,abell_d]

def getPoissonAndErrors(bins,count,alpha=0.32):
    meanCount = np.mean(count)
    sumCount = np.sum(count)
    errorMeanCount = np.std(count)/np.sqrt(len(count))
    meanPoisson = scipy.stats.poisson.pmf(bins,meanCount)
    # Use the ppf for the chi^2 distribution to estimate confidence interval:
    muLower = scipy.stats.chi2.ppf(alpha/2,2*sumCount)/2
    muUpper = scipy.stats.chi2.ppf(1 - alpha/2,2*sumCount + 2)/2
    lambdaLower = muLower/len(count)
    lambdaUpper = muUpper/len(count)
    bound1 = scipy.stats.poisson.pmf(bins,lambdaUpper)
    bound2 = scipy.stats.poisson.pmf(bins,lambdaLower)
    stackedBound = np.vstack((bound1,bound2))
    upperBound = np.max(stackedBound,0)
    lowerBound = np.min(stackedBound,0)
    errorPoisson = np.vstack((meanPoisson - lowerBound,upperBound - meanPoisson))
    return [meanPoisson,errorPoisson]


# Compute alpha-shapes for large antihalos on a Mollweide plot:
def computeMollweideAlphaShapes(snap,largeAntihalos=None,hr=None,alphaVal = 7,\
        snapsort=None,posList=None):
    ahMWPos = []
    alpha_shapes = []
    if snapsort is None:
        snapsort = np.argsort(snap['iord'])
    boxsize = snap.properties['boxsize'].ratio("Mpc a h**-1")
    if (largeAntihalos is not None) and (hr is not None):
        numVoids = len(largeAntihalos)
    elif type(posList) == list:
        numVoids = len(posList)
    else:
        raise Exception("Must specify either list of anti-halos or a list" + \
            " of positions")
    for k in range(0,numVoids):
        if type(posList) == list:
            posXYZ = posList[k]
        elif (largeAntihalos is not None) and (hr is not None):
            posXYZ = snapedit.unwrap(
                snap['pos'][snapsort[hr[largeAntihalos[k]+1]['iord']],:],\
                boxsize)
        else:
            # Technically this should happen as the previous exception should
            # have caught it...
            raise Exception("Error - could not find void particle positions")
        posMW = plot_utilities.computeMollweidePositions(posXYZ)
        ahMWPos.append(posMW)
        alpha_shapes.append(alphashape.alphashape(
            np.array([posMW[0],posMW[1]]).T,alphaVal))
    return [ahMWPos,alpha_shapes]

# Systematic way of writing constraints on cluster masses.
class MassConstraint:
    def __init__(self,mass,massLow,massHigh,virial,virialLow,virialHigh,method='None'
        ,simMass=None,simMassLow=None,simMassHigh=None):
        self.mass = mass
        self.massLow = massLow
        self.massHigh = massHigh
        self.virial = virial
        self.virialLow = virialLow
        self.virialHigh = virialHigh
        self.hasSimMass = False
        self.method = method
        if simMass is not None:
            self.hasSimMass = False
            if (simMassLow is None):
                simMassLow = 0
                print("Warning: no lower error for simulation mass given.")
            if (simMassHigh is None):
                simMassHigh = 0
                print("Warning: no upper error for simulation mass given.")
            self.setSimMass(simMass,simMassLow,simMassHigh)
    def setSimMass(self,mass,masslow,masshigh,recompute=False):
        if ((self.hasSimMass) and (recompute)) or (not self.hasSimMass):
            self.simMass = mass
            self.simMassLow = masslow
            self.simMassHigh = masshigh
            self.hasSimMass = True

# Function to remap snapshots into the correctly aligned equatorial co-ordinates.
def remapBORGSimulation(snap,swapXZ = True,translate=True,reverse=False):
    boxsize = snap.properties['boxsize'].ratio("Mpc a h**-1")
    if translate:
        pynbody.transformation.translate(\
            snap,np.array([-boxsize/2,-boxsize/2,-boxsize/2]))
    if swapXZ:
        swapXZ = np.array([[0,0,1],[0,1,0],[1,0,0]])
        snap.transform(swapXZ)
    if reverse:
        reverseMap = -np.array([[1,0,0],[0,1,0],[0,0,1]])
        snap.transform(reverseMap)


# Remap a set of points to the correct co-ordinates:
def remapAntiHaloCentre(hrcentres,boxsize,swapXZ=True,reverse=False):
    if swapXZ:
        hrcentresRemap = np.fliplr(hrcentres)
    else:
        hrcentresRemap = hrcentres
    hrcentresRemap = snapedit.unwrap(hrcentresRemap - np.array([boxsize/2]*3),boxsize)
    if reverse:
        hrcentresRemap = -hrcentresRemap
    return hrcentresRemap

def zobovVolumesToPhysical(zobovVolumes,snap,dtype=np.double,offset=4):
    N = np.round(np.cbrt(len(snap))).astype(int)
    boxsize = snap.properties['boxsize'].ratio("Mpc a h**-1")
    if type(zobovVolumes) == type(""):
        vols = np.fromfile(zobovVolumes,dtype=dtype,offset=offset)
    else:
        vols = zobovVolumes
    return vols*(boxsize/N)**3

def getHaloCentresAndMassesFromCatalogue(h,inMPcs=True):
    hcentres = np.zeros((len(h),3))
    hmasses = np.zeros(len(h))
    for k in range(0,len(h)):
        hcentres[k,0] = h[k+1].properties['Xc']
        hcentres[k,1] = h[k+1].properties['Yc']
        hcentres[k,2] = h[k+1].properties['Zc']
        hmasses[k] = h[k+1].properties['mass']
    if inMPcs:
        hcentres /= 1000
    return [hcentres,hmasses]

def getHaloMassesAndVirials(snap,centres,overden=200,rho_def = 'critical',
        massUnit="Msol h**-1",distanceUnit = "Mpc a h**-1"):
    masses = np.zeros(len(centres))
    virials = np.zeros(len(centres))
    for k in range(0,len(masses)):
        rvir = pynbody.analysis.halo.virial_radius(snap,cen=centres[k,:],
            overden=overden,rho_def=rho_def)
        filt = pynbody.filt.Sphere(rvir,centres[k,:])
        masses[k] = np.sum(snap[filt]['mass']).in_units(massUnit)
        virials[k] = rvir.in_units(distanceUnit)
    return [masses,virials]

# Load a pynbody snapshot if we provide a string, otherwise just return the 
# snapshot.
def getPynbodySnap(snap):
    if type(snap) == str:
        return pynbody.load(snap)
    else:
        return snap

# Function to either load data, or recompute it:
def loadOrRecompute(filename,func,*args,_recomputeData = False,_cacheData=True,\
        **kwargs):
    if os.path.isfile(filename) and not _recomputeData:
        with open(filename,"rb") as infile:
            result = pickle.load(infile)
    else:
        result = func(*args,**kwargs)
        if _cacheData:
            with open(filename,"wb") as outfile:
                pickle.dump(result,outfile)
    return result

# Compares two variables and checks if they are the same:
def compareData(result,compare,tolerance = 1e-5,enforceExact = False):
    if type(result) != type(compare):
        print("Result types do not match.")
        return False
    elif type(result == np.ndarray):
        if result.shape != compare.shape:
            print("Result array size " + str(result.shape) + \
                " does not match test data size " + str(compare.shape) + ".")
            return False
        difference = result - compare
        if enforceExact:
            return np.array_equal(result,compare)
        elif np.all(np.abs(difference) <= tolerance):
            print("Arrays match within tolerance of " + str(tolerance) + ".")
            return True
        else:
            print("Arrays do not match within tolerance of " + \
                str(tolerance) + ".")
            print("Original = " + str(compare))
            print("New = " + str(result))
            print("Diff = " + str(difference))
            return False
    elif np.isscalar(result):
        if enforceExact:
            return (result == compare)
        else:
            return (np.abs(result - compare) < tolerance)
    else:
        return (result == compare)

# Tests whether the computed results match the stored test data.
def _testComputation(filename,func,*args,_testTolerance = 1e-5,\
        _enforceExact = False,_result = None,_compare=None,\
        _return_result = False,**kwargs):
    if not os.path.isfile(filename):
        raise Exception("Test data not found in " + filename)
    else:
        if _compare is None:
            _compare = pickle.load(open(filename,"rb"))
        if _result is None:
            _result = func(*args,**kwargs)
        if type(_result) == list:
            Nr = len(_result)
            iterable_r = True
        else:
            Nr = 1
            iterable_r = False
        if type(_compare) == list:
            Nc = len(_compare)
            iterable_c = True
        else:
            Nc = 1
            iterable_c = False
        if (Nc != Nr) or (iterable_c != iterable_r):
            print("TESTS FAILED: result not equal in length to test data.\n" + \
                "Test data: " + str(Nc) + " variables.\n" + \
                "Computed data: " + str(Nr) + " variables.")
            if _return_result:
                return [False,_result]
            else:
                return False
        else:
            checkList = [False for k in range(0,Nc)]
            if iterable_c and iterable_r:
                for k in range(0,Nc):
                    print("Comparing result " + str(k+1) + \
                        " of " + str(Nc) + ": ")
                    try:
                        checkList[k] = compareData(_result[k],_compare[k],\
                            tolerance = _testTolerance,\
                            enforceExact=_enforceExact)
                    except Exception as e:
                        logging.error(traceback.format_exc())
                        print("An exception occurred while testing these " + \
                            "elements.")
                testResult = np.all(checkList)
                if testResult:
                    print("TESTS PASSED!")
                else:
                    print("TESTS FAILED!")
                    print("Test results: " + str(checkList))
                if _return_result:
                    return [testResult,_result]
                else:
                    return testResult
            else:
                print("Comparing result with test data: ")
                testResult = False
                try:
                    testResult = compareData(_result,_compare,\
                                tolerance = _testTolerance,\
                                enforceExact=_enforceExact)
                except Exception as e:
                    logging.error(traceback.format_exc())
                    print("An exception occurred while testing.")
                if testResult:
                    print("TESTS PASSED!")
                else:
                    print("TESTS FAILED!")
                if _return_result:
                    return [testResult,_result]
                else:
                    return testResult

# Runs a function to compute something, and saves test data for later
# comparison
def createTestData(filename,func,*args,_return_result = True,\
        _overwritePrevious=False,_result=None,**kwargs):
    if os.path.isfile(filename) and not _overwritePrevious:
        raise Exception("Previous test data found. Specify " + \
            "overwritePrevious=True to overwrite.")
    if _result is None:
        _result = func(*args,**kwargs)
    pickle.dump(_result,open(filename,"wb"))
    if _return_result:
        return _result

# Convert an MCMC file into a white noise file:
def mcmcFileToWhiteNoise(
        mcmcfile,outputName,normalise = True,\
        fromInverseFourier = False,flip = False,reverse=False
    ):
    if h5py is None:
        raise Exception("h5py not found.")
    f = h5py.File(mcmcfile,'r')
    if fromInverseFourier:
        wn = scipy.fft.irfftn(f['scalars']['s_hat_field'][()])
    else:
        wn = f['scalars']['s_field'][()]
    if flip:
        wn = np.flip(wn) # Reverse sign of Fourier convention
    if reverse:
        wn = -wn
    if normalise:
        wn /= np.std(wn)
    np.save(outputName,wn)

# Function to re-order a N^3 length vector that represents data on an NxNxN
# grid. Re-orders the vector in order to swap the XZ axis to switch between
# Fortran and C style ordering.
def reorderLinearScalar(linear,N):
    iord = np.arange(0,N**3)
    coord = snapedit.lin2coord(iord,N)
    coord2 = np.flipud(coord)
    iord2 = snapedit.coord2lin(coord2,N)
    return linear[iord2]

# Return the minimum and maximum of a vector as a 2-element array.
def minmax(x):
    return np.array([np.min(x),np.max(x)])

# Plot a circle:
def plotCircle(centre,radius,fmt='r--',offset = np.array([0,0])):
    theta = np.linspace(0,2*np.pi,100)
    X = radius*np.cos(theta) + centre[0] + offset[0]
    Y = radius*np.sin(theta) + centre[1] + offset[1]
    plt.plot(X,Y,fmt)


# Construct a KDTree for a snapshot, saving it in the snapshot object:
def getKDTree(snap,cacheTree = True,reconstructTree = False):
    try:
        tree = snap.tree
    except AttributeError:
        hasTree = False
    else:
        hasTree = True
    if (not hasTree) or (reconstructTree):
        # Construct the tree:
        boxsize = snap.properties['boxsize'].ratio("Mpc a h**-1")
        if np.any(snap['pos'] < 0) or np.any(snap['pos'] >=boxsize):
            pos = snapedit.wrap(snap['pos'] + boxsize/2,boxsize)
        else:
            pos = snap['pos']
        tree = scipy.spatial.cKDTree(pos,boxsize = boxsize)
        if cacheTree:
            snap.tree = tree
            hasTree = True
    else:
        tree = snap.tree
    return tree

# Compute the galaxy counts in healpix slices:
def getCountsInHealpixSlices(ng,hpIndices,nside = 4,nMagBins = 16,nslices=10,\
        rmax=600,rmin = 0,nres = 256,count_type=float,old_method=False):
    npixels = 12*(nside**2)
    rLimits = np.linspace(rmin,rmax,nslices+1)
    ngHP = np.zeros((nMagBins,npixels*nslices),dtype=count_type)
    reshapedNG = ng.reshape((nMagBins,nres,nres,nres))
    if old_method:
        for k in range(0,npixels*nslices):
            if ne is not None:
                filterK = ne.evaluate("hpIndices == k")
            else:
                filterK = (hpIndices == k)
            for l in range(0,nMagBins):
                ngHP[l,k] = np.sum(reshapedNG[l][filterK])
    else:
        for k in range(0,nMagBins):
            np.add.at(ngHP[k],np.reshape(hpIndices,nres**3),ng[k])
    return ngHP

# Convert positions to healpix regions:
def positionsToHealpix(positions,nside=4,\
        inputCoord='icrs',outputCoord='galactic',rmax = 600,\
        nslices = 10):
    coords = SkyCoord(x = positions[:,0]*u.Mpc,y=positions[:,1]*u.Mpc,\
        z = positions[:,2]*u.Mpc,frame=inputCoord,\
        representation_type="cartesian")
    spherical = coords.represent_as(\
        astropy.coordinates.representation.SphericalRepresentation)
    if outputCoord == "galactic":
        l = coords.galactic.l
        b = coords.galactic.b
    else:
        l = spherical.lon*180.0/np.pi
        b = spherical.lat*180.0/np.pi
    theta = np.pi/2 - np.pi*np.array(b)/180.0
    phi = np.pi*np.array(l)/180.0
    pix = healpy.ang2pix(nside,theta,phi)
    shellThickness = rmax/nslices
    dist = np.sqrt(np.sum(positions**2,1))
    shells = np.array(np.floor(dist/shellThickness),dtype=int)
    npix = 12*nside**2
    hpIndices = pix + shells*npix
    return hpIndices

def downsample(arr,factor):
    old_shape = list(arr.shape)
    mod = np.array([size % factor for size in old_shape])
    if np.any(mod != 0):
        raise Exception("Must downsample by a factor of the array size" + \
            " along every dimension")
    new_shape = []
    for k in range(0,len(old_shape)):
        new_shape.append(int(old_shape[k]/factor))
        new_shape.append(factor)
    newArr = np.reshape(arr,new_shape)
    indicesAv = np.arange(1,len(new_shape),2)
    downSampled = np.mean(newArr,tuple(indicesAv))
    return downSampled


# Wrappers around pickle, since we use this format a lot. Also ensures the file
# always ends up being closed:
def loadPickle(filename):
    with open(filename,"rb") as infile:
        result = pickle.load(infile)
    return result

def savePickle(result,filename):
    with open(filename,"wb") as outfile:
        pickle.dump(result,outfile)


# Flatten a list of lists into a single list:
def flatten(handles):
    handlesOut = []
    for h in handles:
        if type(h) == list:
            handlesOut += flatten(h)
        else:
            handlesOut.append(h)
    return handlesOut



# Simple progress bar code:
def progressbar(it, prefix="", size=60, out=sys.stdout): # Python3.6+
    count = len(it)
    start = time.time()
    def show(j):
        x = int(size*j/count)
        remaining = ((time.time() - start) / j) * (count - j)
        
        mins, sec = divmod(remaining, 60)
        time_str = f"{int(mins):02}:{sec:05.2f}"
        
        print(f"{prefix}[{u'█'*x}{('.'*(size-x))}] {j}/{count} " + 
            f"Est wait {time_str}", end='\r', file=out, flush=True)
        
    for i, item in enumerate(it):
        yield item
        show(i+1)
    print("\n", flush=True, file=out)

def get_weighted_samples(data,weights=None,n_boot = 10000,axis=None):
    n_data = len(data)
    if weights is None:
        weights = np.ones(n_data)
    bootstrap_samples = np.random.choice(n_data,size=(n_data,n_boot))
    bootstrap_averages = np.array([np.average(
        data[bootstrap_samples[:,k]],
        axis=axis,weights=weights[bootstrap_samples[:,k]]) 
        for k in tools.progressbar(range(0,n_boot))]).T
    return bootstrap_averages

def weighted_boostrap_error(data,weights,n_boot = 10000,axis=None):
    bootstrap_averages = get_weighted_samples(data,weights,n_boot=n_boot,
                                              axis=axis)
    return np.std(bootstrap_averages)

def get_unit_vector(v):
    """
    Compute a normalised unit vector parallel to v
        
    Parameters:
        v (array): vector to normalise.
        
    Returns
        array: same shape as v, normalised
    """
    return v/np.sqrt(np.sum(v**2))

def profile_derivative_test(rvals,delta,Delta,profile,profile_log_derivative,
        rtol=1e-5,atol=1e-5,f1=0.53,lower_lim = -10
    ):
    """
    Test whether the logarithmic derivative of a velocity profile actually 
    integrates back to the original profile. Used for testing purposes.
    
    Parameters:
    rvals (float or array): Values of r at which to test the profiles.
    delta (function): Density contrast, function taking r as an argument.
    Delta (function): Cumulative density contrast.
    profile (function): Velocity profile to test. Should take arguments
                        profile(rvals,Delta,f1)
    profile_log_derivative (function): Derivative of profile with respect to
                                       log(r), which we wish to test. Should
                                       take arguments 
                                       profile_log_deriative(
                                           rvals,Delta,delta,f1
                                       )
    rtol (float): Relative tolerance in the difference between the integral
                  and the profile.
    atol (float): Absolute tolerance
    f1 (float): Linear growth rate, which is always a parameter of the models
                we wish to test.
    lower_lim (float): Logarithm of the lower limit of the integral, to avoid
                       integrations problems near r = 0.
    """
    u = lambda r: profile(r,Delta,f1)
    up = lambda logr: profile_log_derivative(
        np.exp(logr),Delta,delta,f1
    )
    ri = rvals[rvals >= np.exp(lower_lim)]
    integrals = np.array(
        [
            scipy.integrate.quad(
                lambda logr: up(logr),lower_lim,np.log(r)
            )[0]
            for r in ri
        ]
    )
    np.testing.assert_allclose(
        u(ri) - u(0), integrals, rtol=rtol,atol=atol
    )

def compatible(computed,reference):
    if isinstance(computed,type(reference)):
        return True
    try:
        if isinstance(np.array(computed),type(reference)):
            return True
    except:
        return False

def compare_output_to_reference(computed,reference,atol=1e-5,rtol=1e-5):
    """
    Test the output of a computation, whether it is a list, an array, tuple,
    or scalar.
    """
    assert compatible(computed,reference)
    if isinstance(reference,list):
        for x, y in zip(computed, reference):
            compare_output_to_reference(x,y,atol=atol,rtol=rtol)
    elif isinstance(reference,np.ndarray) or np.isscalar(reference):
        np.testing.assert_allclose(computed,reference,atol=atol,rtol=rtol)
    elif isinstance(reference,tuple):
        for x, y in zip(computed, reference):
            compare_output_to_reference(x,y,atol=atol,rtol=rtol)

def run_basic_regression_test(
        _function,filename,*args,rtol=1e-5,atol=1e-5,**kwargs
    ):
    """
    Run a basic regression test, comparing the computed output of some function
    to a snapshot of the expected output.
    
    Parameters:
        function (function): function that produces the output we wish to test.
        filename (string): File where reference is stored.
        args: arguments to be passed to the function being tested.
        rtol (float): Relative tolerance for the comparison.
        atol (float): Absolute tolerance for the comparison.
        kwargs (dictionary): Keyword arguments for the function being tested.
    """
    computed = _function(*args,**kwargs)
    name, ext = os.path.splitext(filename)
    if ext == ".npy":
        reference = np.load(filename)
        compare_output_to_reference(computed,reference,rtol=rtol,atol=atol)
        return None
    elif ext == ".npz":
        all_refs = np.load(filename)
        reference = [all_refs[key] for key in all_refs]
    elif ext == ".p":
        reference = loadPickle(filename)
    else:
        raise Exception("File type not implemented or not recognised.")
    if len(computed) != len(reference):
        raise Exception("Number of references does not match number of " + 
                        "computed.")
    for comp, ref in zip(computed, reference):
        compare_output_to_reference(comp,ref,rtol=rtol,atol=atol)

def generate_regression_test_data(_function,filename,*args,**kwargs):
    """
    Generate reference data used for a regression test.
    
    Parameters:
        function (function): Function which generates the regression test data.
        filename (string): File in which to save the test data.
        *args (tuple): Arguments supplied to function
        **kwargs (dictionary): Keyword arguments supplied to function.
    """
    computed = _function(*args,**kwargs)
    name, ext = os.path.splitext(filename)
    if ext == ".npy":
        np.save(filename,computed)
    elif ext == ".npz":
        np.savez(filename,*computed)
    elif ext == ".p":
        savePickle(list(computed),filename)

def ratio_where_finite(x,y,undefined_value = 0.0):
    """
    Return the ratio of x and y, except where y is zero, in which case
    return the specified undefined value (default 0.0)
    
    Parameters:
        x (float or array): Numerator(s)
        y (float or array): Denominator(s)
        undefined_value (default 0.0): Value to use for ratio when y is zero.
    
    Returns:
        ratio (float or array), same shape as x/y
    """
    if np.isscalar(y):
        if y == 0.0:
            return undefined_value
        else:
            return x/y
    else:
        res = np.full(y.shape,undefined_value)
        nz = (y != 0)
        if np.isscalar(x):
            res[nz] = x/y[nz]
        else:
            res[nz] = x[nz]/y[nz]
        return res

def product_where_finite(x,y,undefined_value = 0.0):
    """
    Return the product of x and y, except where x or y is infinite, in which 
    case return the specified undefined value (default 0.0)
    
    Parameters:
        x, y (float or array): Values to multiply
        undefined_value (default 0.0): Value to use for product when x or y is
            not finite.
    
    Returns:
        ratio (float or array), same shape as x*y
    """
    if np.isscalar(x):
        if np.isscalar(y):
            if np.isfinite(x) and np.isfinite(y):
                return x*y
            else:
                return undefined_value
        else:
            if np.isfinite(x):
                res = np.full(y.shape,undefined_value)
                finite = np.isfinite(y)
                res[finite] = x*y[finite]
                return res
            else:
                return np.full(y.shape,undefined_value)
    else:
        if np.isscalar(y):
            if np.isfinite(y):
                res = np.full(x.shape,undefined_value)
                finite = np.isfinite(x)
                res[finite] = x[finite]*y
                return res
            else:
                return undefined_value
        else:
            finite_x = np.isfinite(x)
            finite_y = np.isfinite(y)
            res = np.full(y.shape,undefined_value)
            finite = finite_x & finite_y
            res[finite] = x[finite]*y[finite]
            return res



def gaussian_log_likelihood_function(x,mu,C,use_cholesky=True,
        pre_inverted=False,include_constants=True
    ):
    """
    Compute the Gaussian log-likelihood for observed data, given the mean and 
    covariance matrix.
    
    Parameters:
        x (array): Data
        mu (N-component array): Mean of the Gaussian log-likelihood
        C (NxN matrix): Covariance of the Gaussian log-likelihood
        use_cholesky (bool): If true, use cholesky decomposition to solve, 
                             rather than inverting.
        pre_inverted (bool): If true, assume C is actually the inverse 
                             covariance. Cannot be used together with 
                             use_cholesky = True.
        include_constants(bool): If true, include constants such as factors of
                                 the determinant which don't change with the 
                                 data.
    
    Returns:
        float: Gaussian log-likelihood for the given data.
    """
    delta_vec = x - mu
    N = len(mu)
    if include_constants:
        detC = scipy.linalg.det(C)
        if pre_inverted:
            detC = 1.0/detC            
        constants = -(N/2)*np.log(2*np.pi) - 0.5*np.log(detC)
    else:
        constants = 0.0
    if use_cholesky:
        if pre_inverted:
            raise Exception("Cannot use pre_inverted = True together with " + 
                            "use_cholesky = True.")
        chol = scipy.linalg.cholesky(C,lower=True)
        alpha = scipy.linalg.solve_triangular(chol, delta_vec, lower=True)
        return -0.5 * np.dot(alpha, alpha) + constants
    else:
        inv_cov = C if pre_inverted else scipy.linalg.inv(C)
        return -0.5 * np.dot(delta_vec, inv_cov @ delta_vec) + constants

def get_finite_range(range_list):
    """
    Return a finite range, given a list of ranges that are possible infinite.
    
    Parameters:
        range_list (list of floats): 2-component list with lower and upper
                                     bounds for a variable, possibly infinite
    
    Returns:
        list of floats: For finite ranges, returns the original range, but
                       ranges including +/- np.inf are replaced with
                       +/- 1
    
    Tests:
        Tested in test_tools.py
        Unit tests: test_get_finite_range_basic
    """
    return [x if np.isfinite(x) else np.sign(x) for x in range_list]

def load_npz_arrays(filename):
    name, ext = os.path.splitext(filename)
    if ext != ".npz":
        raise Exception("Not an npz file.")
    loaded = np.load(filename)
    return [loaded[key] for key in loaded]








