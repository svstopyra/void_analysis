# Stacking functions
import Corrfunc
import scipy.spatial
import numpy as np
from . import snapedit, plot_utilities, tools
import multiprocessing as mp
from scipy.optimize import curve_fit
thread_count = mp.cpu_count()
import scipy.integrate as integrate
from scipy.stats import kstest
import gc
import pynbody

# Weighted mean of a variable:
def weightedMean(xi,wi,biasFactor = 0,axis=None):
    return (np.sum(xi*wi,axis=axis) + biasFactor)/np.sum(wi,axis=axis)

# Weighted variance:
def weightedVariance(xi,wi,biasFactor = 0,axis=None):
    xibar = weightedMean(xi,wi,biasFactor=biasFactor,axis=axis)
    M = np.count_nonzero(wi,axis=axis)
    return np.sum(wi*(xi-xibar)**2,axis=axis)/(((M-1)/M)*np.sum(wi,axis=axis))

# Compute correlation function for a given set of points 
# (cross correlation if a second is specified)
def simulationCorrelation(rBins,boxsize,data1,data2=None,nThreads = 1,\
        weights1=None,weights2 = None):
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
        DD1 = Corrfunc.theory.DD(1,nThreads,rBins,X1,Y1,Z1,periodic=True,\
            boxsize=boxsize,weights1=weights1)
        DR1 = Corrfunc.theory.DD(0,nThreads,rBins,X1,Y1,Z1,periodic=True,\
            boxsize=boxsize,X2 = X1rand,Y2=Y1rand,Z2=Z1rand,weights1=weights1)
        RR1 = Corrfunc.theory.DD(1,nThreads,rBins,X1rand,Y1rand,Z1rand,\
            periodic=True,boxsize=boxsize,weights1=weights1)
        xiEst = Corrfunc.utils.convert_3d_counts_to_cf(N1,N1,rand_N1,rand_N1,\
            DD1,DR1,DR1,RR1)
    else:
        # Cross correlation:
        D1D2 = Corrfunc.theory.DD(0,nThreads,rBins,X1,Y1,Z1,X2=X2,Y2=Y2,Z2=Z2,\
            periodic=True,boxsize=boxsize,weights1=weights1,weights2=weights2)
        D1R2 = Corrfunc.theory.DD(0,nThreads,rBins,X1,Y1,Z1,periodic=True,\
            boxsize=boxsize,X2 = X2rand,Y2=Y2rand,Z2=Z2rand,weights1=weights1,\
            weights2=weights2)
        D2R1 = Corrfunc.theory.DD(0,nThreads,rBins,X2,Y2,Z2,periodic=True,\
            boxsize=boxsize,X2 = X1rand,Y2=Y1rand,Z2=Z1rand,weights1=weights1,\
            weights2=weights2)
        R1R2 = Corrfunc.theory.DD(0,nThreads,rBins,X1rand,Y1rand,Z1rand,\
            X2 = X2rand,Y2=Y2rand,Z2=Z2rand,periodic=True,boxsize=boxsize,\
            weights1=weights1,weights2=weights2)
        xiEst = Corrfunc.utils.convert_3d_counts_to_cf(N1,N2,rand_N1,rand_N2,\
            D1D2,D1R2,D2R1,R1R2)
    return xiEst




# Cross correlation of two sets of points:
def getCrossCorrelations(ahCentres,voidCentres,ahRadii,voidRadii,rMin = 0,\
        rMax = np.inf,rRange = np.linspace(0.1,10,101),nThreads=thread_count,\
        boxsize = 200.0):
    rFilter1 = np.where((ahRadii > rMin) & (ahRadii < rMax))[0]
    rFilter2 = np.where((voidRadii > rMin) & (voidRadii < rMax))[0]
    ahPos = ahCentres[rFilter1,:]
    vdPos = voidCentres[rFilter2,:]
    xiAM = simulationCorrelation(rRange,boxsize,ahPos,data2=snap['pos'],\
        nThreads=nThreads,weights2 = snap['mass'])
    xiVM = simulationCorrelation(rRange,boxsize,vdPos,data2=snap['pos'],\
        nThreads=nThreads,weights2 = snap['mass'])
    xiAV = simulationCorrelation(rRange,boxsize,ahPos,data2=vdPos,\
        nThreads=nThreads)
    return [xiAM,xiVM,xiAV]

# Auto correlation of a set of points:
def getAutoCorrelations(ahCentres,voidCentres,ahRadii,voidRadii,rMin = 0,\
        rMax = np.inf,rRange = np.linspace(0.1,10,101),nThreads=thread_count,\
        boxsize = 200.0):
    rFilter1 = np.where((ahRadii > rMin) & (ahRadii < rMax))[0]
    rFilter2 = np.where((voidRadii > rMin) & (voidRadii < rMax))[0]
    ahPos = ahCentres[rFilter1,:]
    vdPos = voidCentres[rFilter2,:]
    xiAA = simulationCorrelation(rRange,boxsize,ahPos,nThreads=nThreads)
    xiVV = simulationCorrelation(rRange,boxsize,vdPos,nThreads=nThreads)
    return [xiAA,xiVV]



def getPairCounts(voidCentres,voidRadii,snap,rBins,nThreads=-1,\
        tree=None,method="poisson",vorVolumes=None,countMethod="ball"):
    if (vorVolumes is None) and (method == "VTFE"):
            raise Exception("Must provide voronoi volumes for VTFE.")
    # Generate KDTree
    if type(snap) == str:
        snapLoaded = pynbody.load(snap)
    else:
        snapLoaded = snap
    boxsize = snapLoaded.properties['boxsize'].ratio("Mpc a h**-1")
    if tree is None:
        tree = scipy.spatial.cKDTree(snapLoaded['pos'],boxsize=boxsize)
    # Volumes of the rBins
    rBinsUp = rBins[1:]
    rBinsLow = rBins[0:-1]
    volumes = 4*np.pi*(rBinsUp**3 - rBinsLow**3)/3
    # Find particles:
    if method != "volume":
        nPairsList = np.zeros((len(voidCentres),len(rBins[1:])),dtype=np.int64)
    else:
        nPairsList = np.zeros((len(voidCentres),len(rBins[1:])))
    volumesList = np.outer(voidRadii**3,volumes)
    numBins = len(rBins)
    if method == "poisson":
        for k in range(0,len(nPairsList)):
            if countMethod == "tree":
                nPairsList[k,:] = tree.count_neighbors(\
                    scipy.spatial.cKDTree(voidCentres[[k],:],boxsize=boxsize),\
                    rBins*voidRadii[k],cumulative=False)[1:]
            elif countMethod == "ball":
                lengths = tree.query_ball_point(\
                    np.tile(voidCentres[k,:],(numBins,1)),\
                    rBins*voidRadii[k],workers=-1,return_length=True)
                nPairsList[k,:] = lengths[1:] - lengths[0:-1]
            else:
                raise Exception("Unrecognised count method")
    elif method == "volume":
        # Volume weighted density in each shell:
        for k in range(0,len(voidRadii)):
            indicesList = np.array(tree.query_ball_point(\
                voidCentres[k,:],rBins[-1]*voidRadii[k],\
                workers=nThreads),dtype=np.int32)
            disp = snapLoaded['pos'][indicesList,:] - voidCentres[k,:]
            r = np.array(np.sqrt(np.sum(disp**2,1)))
            sort = np.argsort(r)
            indicesSorted = indicesList[sort]
            boundaries = np.searchsorted(r[sort],rBins*voidRadii[k])
            for l in range(0,len(rBins)-1):
                indexListShell =  indicesSorted[boundaries[l]:boundaries[l+1]]
                if len(indexListShell) > 0:
                    volSumShell = np.sum(vorVolumes[indexListShell])
                    nPairsList[k,l] = len(indexListShell)/volSumShell
                    volumesList[k,l] = volSumShell
                else:
                    volumesList[k,l] = 0				
            print("Finished void " + str(k) + " of " + str(len(voidRadii)))
    else:
        #for k in range(0,len(rBins)):
        #    indices = tree.query_ball_point(voidCentres,rBins[k]*voidRadii,\
        #    workers=nThreads)
        #    listLengths = np.array([len(list) for list in indices])
        #    volSum = np.array([np.sum(vorVolumes[list]) for list in indices])
        #    if k == 0:
        #        nPairsList[:,k] = listLengths
        #        volumesList[:,k] = volSum
        #    else:
        #        nPairsList[:,k] = listLengths - nPairsList[:,k-1]
        #        volumesList[:,k] = volSum - volumesList[:,k-1]
        for k in range(0,len(voidRadii)):
            indicesList = np.array(tree.query_ball_point(voidCentres[k,:],\
                rBins[-1]*voidRadii[k],workers=nThreads),dtype=np.int32)
            disp = snapLoaded['pos'][indicesList,:] - voidCentres[k,:]
            r = np.array(np.sqrt(np.sum(disp**2,1)))
            sort = np.argsort(r)
            indicesSorted = indicesList[sort]
            boundaries = np.searchsorted(r[sort],rBins*voidRadii[k])
            print("Doing void " + str(k) + " of " + str(len(voidRadii)))
            nPairsList[k,:] = np.diff(boundaries)
            volumesCumulative = np.cumsum(vorVolumes[indicesSorted])
            volumesList[k,:] = volumesCumulative[boundaries[1:]]
    gc.collect()
    return [nPairsList,volumesList]

# Stakcing done around fixed points and with fixed radius, in each
# sample:
def pairCountsFixedPosition(snapNameList,centres,radii,rBins,\
        method="poisson"):
    pairsListMean = []
    volsListMean = []
    for ns in range(0,len(snapNameList)):
        snap = tools.getPynbodySnap(snapNameList[ns])
        boxsize = snap.properties['boxsize'].ratio("Mpc a h**-1")
        gc.collect()
        tree = scipy.spatial.cKDTree(snap['pos'],boxsize=boxsize)
        gc.collect()
        [pairs,vols] = getPairCounts(centres,\
                radii,snap,rBins,nThreads=-1,tree=tree,method=method)
        pairsListMean.append(pairs)
        volsListMean.append(vols)
    return [pairsListMean,volsListMean]

# Unconstrained stacks with same radius distribution:
def stackUnconstrainedWithConstrainedRadii(snapListUn,rBins,antihaloRadiusBins,\
        binCounts,conditionList,antihaloRadiiUn,ahCentresListUn,\
        allPairCountsUn,allVolumesListsUn,\
        seed=100,method="poisson",errorType="Weighted",selection="all"):
    # Set the seed for consistency:
    np.random.seed(100)
    counter = 0
    if selection == "all":
        selection = range(0,len(snapListUn))
    conditionListLengths = np.array([len(cond) for cond in conditionList])
    nSamples = np.sum(conditionListLengths[selection])
    nbarjUnSameRadii = np.zeros((nSamples,len(rBins)-1))
    sigmaUnSameRadii = np.zeros((nSamples,len(rBins)-1))
    for ns in selection:
        for l in range(0,len(conditionList[ns])):
            condition = conditionList[ns][l]
            [binListUn,noInBinsUn] = plot_utilities.binValues(\
                antihaloRadiiUn[ns][condition],antihaloRadiusBins)
            if np.any(noInBinsUn == 0):
                allRandIndices = []
                for k in range(0,len(binListUn)):
                    if noInBinsUn[k] != 0:
                        allRandIndices.append(np.random.choice(binListUn[k],\
                            binCounts[k]))
                randSelect = np.hstack(allRandIndices)
            else:
                randSelect = np.hstack([np.random.choice(binListUn[k],\
                    binCounts[k]) for k in range(0,len(binListUn))])
            [nbarj,sigma] = stackScaledVoids(
                    ahCentresListUn[ns][condition,:][randSelect],\
                    antihaloRadiiUn[ns][condition][randSelect],\
                    snapListUn[ns],rBins,\
                    nPairsList = allPairCountsUn[ns][l][randSelect],\
                    volumesList=allVolumesListsUn[ns][l][randSelect],\
                    method=method,errorType=errorType)
            nbarjUnSameRadii[counter,:] = nbarj
            sigmaUnSameRadii[counter,:] = sigma
            counter += 1
    return [nbarjUnSameRadii,sigmaUnSameRadii]

def getRadialVelocityAverages(voidCentres,voidRadii,snap,rBins,\
        nThreads=thread_count,tree=None,method="poisson",vorVolumes=None,
        relative_velocity=False):
    if (vorVolumes is None) and (method == "VTFE"):
            raise Exception("Must provide voronoi volumes for VTFE.")
    # Generate KDTree
    boxsize = snap.properties['boxsize'].ratio("Mpc a h**-1")
    if tree is None:
        tree = scipy.spatial.cKDTree(snap['pos'],boxsize=boxsize)
    rBinsUp = rBins[1:]
    rBinsLow = rBins[0:-1]
    volumes = 4*np.pi*(rBinsUp**3 - rBinsLow**3)/3
    vRList = np.zeros((len(voidCentres),len(rBins)-1))
    volumesList = np.outer(voidRadii**3,volumes)
    for k in range(0,len(vRList)):
        indicesList = np.array(tree.query_ball_point(voidCentres[k,:],\
            rBins[-1]*voidRadii[k],workers=nThreads),dtype=np.int64)
        disp = snap['pos'][indicesList,:] - voidCentres[k,:]
        r = np.array(np.sqrt(np.sum(disp**2,1)))
        sort = np.argsort(r)
        indicesSorted = indicesList[sort]
        boundaries = np.searchsorted(r[sort],rBins*voidRadii[k])
        if relative_velocity:
            # get velocity relative to void centre:
            v_centre = np.average(snap['vel'][indicesSorted,:],axis=0,
                                  weights=vorVolumes[indicesSorted])
            velocity = snap['vel'][indicesSorted,:] - v_centre
        else:
            velocity = snap['vel'][indicesSorted,:]
        vR = np.sum(disp*velocity,1)/r
        #print("Doing void " + str(k) + " of " + str(len(vRList)))
        for l in range(0,len(rBins)-1):
            indices = indicesSorted[boundaries[l]:boundaries[l+1]]
            if len(indices) > 0:
                #indicesSphere = sort[boundaries[l]:boundaries[l+1]]
                vRList[k,l] = np.average(
                    vR[boundaries[l]:boundaries[l+1]],
                    weights = vorVolumes[boundaries[l]:boundaries[l+1]]
                )
    return [vRList,volumesList]

# Stack from individual void profiles:
def stackProfilesWithError(rhoi,sigmaRhoi,volumesList,biasTerm=1.0,\
        combinationMethod="quadrature"):
    # Volume weights for the stack:
    weights = volumesList/np.sum(volumesList,0)
    # Variance of the mean profile:
    meanVariance = np.var(rhoi,0)*np.sum(weights**2,0)
    # Variance of individual profiles:
    profileVariance = np.sum(weights**2*sigmaRhoi**2,0)
    # Combine the variances as an approximation of the error:
    if combinationMethod=="quadrature":
        combinedVariance = meanVariance + profileVariance
    else:
        raise Exception("Unrecognised combinationMethod")
    # Compute the stacked void profile:
    nbarMean = (biasTerm + np.sum(rhoi*volumesList,0))/np.sum(volumesList,0)
    return [nbarMean,np.sqrt(combinedVariance)]


# Average individual void profiles to compute profiles with an error:
def computeAveragedIndividualProfiles(catalogue,allPairCounts,allVols,\
        existingOnly=False,additionalFilter=None,errorType="Mean"):
    # "catalogue" should be an Nv x Ns array, with Nv then umber of voids we 
    # wish to compute averaged profiles for, and Ns the number of samples over #
    # which we are averaging. It is assumed that this is offset by 1 
    # (ie, first void is 1, not zero), so that we need to subtract 1 when 
    # referencing the arrays The interpretation of "catalogue" is that is 
    # gives a list of Nv voids, each of which has a representative antihalo
    # in each of the Ns samples. The code allows for the possibility that some
    # samples may simply not have representative (indicated by a negative void
    # number). These are skipped from the average.
    Ns = catalogue.shape[1] # Number of MCMC samples
    # Trim the catalogue to include only a subset of voids:
    if additionalFilter is not None:
        # additionalFilter is a boolean array with length equal to the catalogue
        # which allows us to select a subset of the catalogue:
        inTrimmedCatalogue = np.any((catalogue >= 0) & \
            additionalFilter[:,None],1)
    else:
        inTrimmedCatalogue = np.any((catalogue >= 0),1)
    trimmedCatalogue = catalogue[inTrimmedCatalogue]
    selectedVoids = np.where(inTrimmedCatalogue)[0]
    Nv = len(trimmedCatalogue) # Number of voids in the catalogue
    nBins = len(allPairCounts[0][0]) # Number of radius bins
    nbarCombined = np.zeros((Nv,nBins))
    sigmaCombined = np.zeros((Nv,nBins))
    for k in range(0,Nv):
        # Iterating over all voids
        pairs = [] # Pair counts in spherical shells
        vols = [] # volumes of spherical shells
        if existingOnly:
            # Only average over samples which have an extant representative of 
            # of the void
            for ns in range(0,Ns):
                # Iterating over all samples:
                if catalogue[k,ns] >= 0:
                    pairs.append(allPairCounts[ns][trimmedCatalogue[k,ns]-1])
                    vols.append(allVols[ns][trimmedCatalogue[k,ns]-1])
        else:
            # Average over all samples, regardless of whether there is a void
            # there or not:
            ind = selectedVoids[k]
            for ns in range(0,Ns):
                pairs.append(allPairCounts[ns][ind])
                vols.append(allVols[ns][ind])
        nbarj = np.mean(np.vstack(pairs)/np.vstack(vols),0)
        variance = np.var(np.vstack(pairs)/np.vstack(vols),0)
        if errorType == "Mean":
            variance /= Ns
        sigmabarj = np.sqrt(variance)
        nbarCombined[k,:] = nbarj
        sigmaCombined[k,:] = sigmabarj
    return [nbarCombined,sigmaCombined]

# Direct pair counting in rescaled variables:
def stackScaledVoids(voidCentres,voidRadii,snap,rBins,nThreads=thread_count,\
       tree=None,method="poisson",vorVolumes=None,nPairsList=None,\
        volumesList=None,errorType="Mean",interval=68):
    if (nPairsList is None) or (volumesList is None):
        [nPairsList,volumesList] = getPairCounts(voidCentres,voidRadii,snap,\
            rBins,nThreads=nThreads,tree=tree,method=method,\
            vorVolumes=vorVolumes)
    if method == "poisson":
        nbarj = (np.sum(nPairsList,0) + 1)/(np.sum(volumesList,0))
        if errorType == "Mean":
            sigmabarj = np.sqrt(len(voidCentres)*np.sum(nPairsList,0))/\
                np.sum(volumesList,0)/np.sqrt(len(voidCentres)-1)
        elif errorType == "Profile":
            sigmabarj = np.sqrt(weightedVariance(nPairsList/volumesList,\
                volumesList,axis=0))
        elif errorType == "Percentile":
            sigmabarj = np.percentile(nPairsList/volumesList,\
                [(100-interval)/2,50 + interval/2],axis=0)
            sigmabarj[0,:] = nbarj - sigmabarj[0,:]
            sigmabarj[1,:] = sigmabarj[1,:] - nbarj
        elif errorType == "Weighted":
            weights = volumesList/np.sum(volumesList,0)
            sigmabarj = np.sqrt(np.var(nPairsList/(volumesList),0)*\
                np.sum(weights**2,0))
        else:
            raise Exception("Invalid error type.")
    elif method == "naive":
        nbarj = np.sum(nPairsList/volumesList,0)/len(voidCentres)
        if errorType == "Mean":
            sigmabarj = np.std(nPairsList/volumesList,0)/\
                np.sqrt(len(voidCentres)-1)
        elif errorType == "Profile":
            sigmabarj = np.std(nPairsList/volumesList,0)/\
                np.sqrt(len(voidCentres)-1)
        elif errorType == "Percentile":
            sigmabarj = np.std(nPairsList/volumesList,0)
        else:
            raise Exception("Invalid error type.")
    elif method == "VTFE":
        nbarj = np.sum(nPairsList,0)/(np.sum(volumesList,0))
        if errorType == "Mean":
            sigmabarj = np.sqrt(len(voidCentres)*np.sum(nPairsList,0))/\
                np.sum(volumesList,0)
        elif errorType == "Profile":
            sigmabarj = np.sqrt(weightedVariance(nPairsList/volumesList,\
                volumesList,axis=0))
        else:
            raise Exception("Invalid error type.")
    elif method == "cumulative":
        nPairCum = (np.cumsum(nPairsList,axis=1)/np.cumsum(volumesList,axis=1))
        nbarj = weightedMean(nPairCum,volumesList,axis=0)
        sigmabarj = np.sqrt(weightedVariance(nPairCum,volumesList,axis=0)/\
            (len(nPairCum)-1))
    else:
        raise Exception("Unrecognised stacking method.")
        
    return [nbarj,sigmabarj]

# Stack radial velocities of voids (Incidentally, this is mostly the same as the normal stacking, just with a different variable - would be better to merge them):
def stackScaledVoidsVelocities(voidCentres,voidRadii,snap,rBins,\
        nThreads=thread_count,tree=None,method="poisson",vorVolumes=None,\
        nPairsList=None,volumesList=None,errorType="Mean"):
    if (nPairsList is None) or (volumesList is None):
        [nPairsList,volumesList] = getRadialVelocityAverages(voidCentres,\
            voidRadii,snap,rBins,nThreads=nThreads,tree=tree,method=method,\
            vorVolumes=vorVolumes)
    if method == "poisson":
        nbarj = (np.sum(nPairsList,0) + 1)/(np.sum(volumesList,0))
        if errorType == "Mean":
            sigmabarj = np.sqrt(len(voidCentres)*np.sum(nPairsList,0))/\
                np.sum(volumesList,0)
        elif errorType == "Profile":
            sigmabarj = np.sqrt(weightedVariance(nPairsList/volumesList,\
                volumesList,axis=0))/np.sqrt(len(voidCentres)-1)
        elif errorType == "Standard":
            sigmabarj = np.sqrt(weightedVariance(nPairsList/volumesList,\
                volumesList,axis=0))
        else:
            raise Exception("Invalid error type.")
    elif method == "naive":
        nbarj = np.sum(nPairsList/volumesList,0)/len(voidCentres)
        if errorType == "Mean":
            sigmabarj = np.std(nPairsList/volumesList,0)/np.sqrt(len(voidCentres)-1)
        elif errorType == "Profile":
            sigmabarj = np.std(nPairsList/volumesList,0)
        else:
            raise Exception("Invalid error type.")
    elif method == "VTFE":
        nbarj = np.sum(nPairsList,0)/(np.sum(volumesList,0))
        if errorType == "Mean":
            sigmabarj = np.sqrt(len(voidCentres)*np.sum(nPairsList,0))/\
                np.sum(volumesList,0)
        elif errorType == "Profile":
            sigmabarj = np.sqrt(weightedVariance(nPairsList/volumesList,\
                volumesList,axis=0))
        else:
            raise Exception("Invalid error type.")
    else:
        raise Exception("Unrecognised stacking method.")
    return [nbarj,sigmabarj]


# Apply a filter to a set of voids before stacking them:
def stackVoidsWithFilter(voidCentres,voidRadii,filterToApply,snap,\
        rBins=None,nPairsList=None,volumesList=None,\
        nThreads=thread_count,tree=None,method="poisson",vorVolumes=None,\
        errorType="Mean"):
    if rBins is None:
        rBins = np.linspace(0,3,31)
    if (nPairsList is None) or (volumesList is None):
        [nPairsList,volumesList] = getPairCounts(voidCentres[filterToApply,:],\
            voidRadii[filterToApply],snap,rBins,nThreads=nThreads,tree=tree,\
            method=method,vorVolumes=vorVolumes[filterToApply])
    return stackScaledVoids(voidCentres[filterToApply,:],\
        voidRadii[filterToApply],snap,rBins,nThreads=nThreads,tree=tree,\
            method=method,nPairsList=nPairsList[filterToApply,:],\
            volumesList=volumesList[filterToApply,:],errorType=errorType)

# Apply a filter to a stack of voids before summing their velocities.
def stackVoidVelocitiesWithFilter(voidCentres,voidRadii,filterToApply,snap,\
       rBins=None,nPairsList=None,volumesList=None,nThreads=thread_count,\
        tree=None,method="poisson",vorVolumes=None,errorType="Profile"):
    if rBins is None:
        rBins = np.linspace(0,3,31)
    if (nPairsList is None) or (volumesList is None):
        [nPairsList,volumesList] = getRadialVelocityAverages(voidCentres,\
            voidRadii,snap,rBins,nThreads=nThreads,tree=tree,method=method,\
            vorVolumes=vorVolumes)
    return stackScaledVoidsVelocities(voidCentres[filterToApply,:],\
        voidRadii[filterToApply],snap,rBins,nThreads=nThreads,tree=tree,\
        method=method,nPairsList=nPairsList[filterToApply,:],\
        volumesList=volumesList[filterToApply],errorType=errorType)

def meanDensityContrast(voidParticles,volumes,nbar):
    return (len(voidParticles)/(nbar*np.sum(volumes[voidParticles]))) - 1.0

def lambdaVoid(voidParticles,volumes,nbar,radius):
    return meanDensityContrast(voidParticles,volumes,nbar)*((radius)**(1.2))

# Central Density:
def centralDensity(voidCentre,voidRadius,positions,volumes,masses,\
        boxsize=None,tree=None,centralRatio = 4,nThreads=thread_count):
    if tree is None:
        tree = scipy.spatial.cKDTree(positions,boxsize=boxsize)
    central = tree.query_ball_point(voidCentre,voidRadius/centralRatio,\
        workers=nThreads)
    rhoCentral = np.sum(masses[central])/np.sum(volumes[central])
    return rhoCentral

def centralDensityNN(voidCentre,positions,masses,volumes,boxsize=None,\
        tree = None,nThreads = thread_count,nNeighbours=64,rCut=None):
    if tree is None:
        tree = scipy.spatial.cKDTree(positions,boxsize=boxsize)
    if rCut is not None:
        central = tree.query(voidCentre,k=nNeighbours,workers=nThreads,\
            distance_upper_bound = rCut)
    else:
        central = tree.query(voidCentre,k=nNeighbours,workers=nThreads)
    if (len(central[0]) < 1) and (rCut is not None):
        # Try again, ignoring the cut:
        central = tree.query(voidCentre,k=nNeighbours,workers=nThreads)
    rhoCentral = np.sum(masses[central[1]])/np.sum(volumes[central[1]])
    return rhoCentral

def profileParamsNadathur(nbarj,rBins,nbar):
    return [1.57,5.72,0.81,-0.69]


def profileModel(r,model,modelArgs):
    if model == "Hamaus":
        return profileModelHamaus(r,modelArgs[0],modelArgs[1])
    elif model == "Nadathur":
        return profileModelNadathur(r,modelArgs[0],modelArgs[1],\
            modelArgs[2],modelArgs[3])
    else:
        raise Exception("Unknown profile model.")

# Compute the void profile model, from fitting parameters:
def profileModelHamaus(r,rs,deltac):
    rv = 1.0 # Mean effective radius is 1 in these units, by definition.
    alpha = -2.0*(rs/rv) + 4.0
    if rs/rv < 0.91:
        beta = 17.5*(rs/rv) - 6.5
    else:
        beta = -9.8*(rs/rv) + 18.4
    return deltac*(1 - (r/rs)**(alpha))/(1 + r**beta) + 1

# Void profile parameters:
def profileParamsHamaus(nbarj,sigmabarj,rBins,nbar,returnCovariance = False):
    # Determine parameters:
    rBinCentres = plot_utilities.binCentres(rBins)
    [popt, pcov] = curve_fit(profileModelHamaus,rBinCentres,nbarj/nbar,\
        sigma=sigmabarj,maxfev=10000, xtol=5.e-3,
                         p0=[1.0,-1.0])
    rs = popt[0]
    deltac = popt[1]
    perr = np.sqrt(np.diag(pcov))
    if returnCovariance:
        return [rs,deltac,perr[0],perr[1],pcov]
    else:
        return [rs,deltac,perr[0],perr[1]]


def profileModelNadathur(r,alpha,beta,rs,deltac):
	return deltac*(1 - (r/rs)**(alpha))/(1 + (r/rs)**beta) + 1
	
# Attempts to select voids from set1 such that the distribution 
# of lambda matches that of set 2.
def matchDistributionFilter(lambdas1,lambdas2,lambdaBins=None,randomSeed=None,\
        binsToMatch=None,binValues1=None,binValues2=None):
    if lambdaBins is None:
        lambdaBins = np.linspace(np.min([np.min(lambdas1),np.min(lambdas2)]),\
            np.max([np.max(lambdas1),np.max(lambdas2)]),101)
    if randomSeed is not None:
        np.random.seed(seed = randomSeed)
    if binsToMatch is None:
        inBinValue1 = [np.array(range(0,len(lambdas1)))]
        noInBinValue1 = len(lambdas1)
        inBinValue2 = [np.array(range(0,len(lambdas2)))]
        noInBinValue2 = len(lambdas2)
    else:
        # Allow for matching lambdas only within the 
        # specified bins, not over all voids:
        [inBinValue1,noInBinValue1] = plot_utilities.binValues(binValues1,\
            binsToMatch)
        [inBinValue2,noInBinValue2] = plot_utilities.binValues(binValues2,\
            binsToMatch)
    set1List = np.array([],dtype=np.int64)
    for k in range(0,len(inBinValue1)):
        [binList1,noInBins1] = plot_utilities.binValues(\
            lambdas1[inBinValue1[k]],lambdaBins)
        [binList2,noInBins2] = plot_utilities.binValues(\
            lambdas2[inBinValue2[k]],lambdaBins)
        fracs = noInBins2/len(lambdas2[inBinValue2[k]])
        set1ToDrawMax = np.array(np.floor(fracs*\
            len(lambdas1[inBinValue1[k]])),dtype=np.int64)
        for l in range(0,len(fracs)):
            if noInBins1[l] < set1ToDrawMax[l]:
                set1List = np.hstack((set1List,\
                    np.array(inBinValue1[k][binList1[l]])))
            else:
                set1List = np.hstack((set1List,np.random.choice(\
                    inBinValue1[k][binList1[l]],size=set1ToDrawMax[l],\
                    replace=False)))
    return set1List


# Profile integrals:
def profileIntegral(muBins,rho,mumin=0,mumax=1.0,histIntegral=True):
    mu = plot_utilities.binCentres(muBins)
    if not histIntegral:
        intRange = np.where((mu >= mumin) & (mu <= mumax))
        return integrate.simps(3*mu[intRange]**2*rho[intRange],x=mu[intRange])
    else:
        # Best approximation we have, using the histograms:
        intRange = np.where((muBins >= mumin) & (muBins <= mumax))
        return np.sum((muBins[intRange][1:]**3 - muBins[intRange][0:-1]**3)*\
            rho[intRange][0:-1])

# Compute a few examples, and compare to the averages of the volume
# weigted average densities:
def getProfileIntegralAndDeltaAverage(snap,rBins,radii,centres,\
        pairCounts,volumesList,deltaBar,nbar,condition,method="poisson",\
        errorType="Profile",mumin=0.0,mumax=1.0):
    filterToUse = np.where(condition)[0]
    [nbarj,sigma] = stackVoidsWithFilter(centres,radii,filterToUse,snap,\
        rBins,nPairsList = pairCounts,volumesList=volumesList,method=method,\
        errorType=errorType)
    rBinCentres = plot_utilities.binCentres(rBins)
    profInt = profileIntegral(rBins,nbarj/nbar,mumin=mumin,mumax=mumax)
    profIntUpper = profileIntegral(rBins,(nbarj + sigma)/nbar,\
        mumin=mumin,mumax=mumax)
    profIntLower = profileIntegral(rBins,(nbarj - sigma)/nbar,\
        mumin=mumin,mumax=mumax)
    if method == "poisson":
        deltaAverage = np.sum(radii[filterToUse]**3*\
            (1.0 + deltaBar[filterToUse]))/np.sum(radii[filterToUse]**3)
        deltaAverageError = np.sqrt(weightedVariance((1.0 + deltaBar[filterToUse]),radii[filterToUse]**3))/np.sqrt(len(filterToUse))
    else:
        deltaAverage = np.mean((1.0 + deltaBar[filterToUse]))
        deltaAverageError = np.std((1.0 + deltaBar[filterToUse]))/\
            np.sqrt(len(filterToUse))
    return [profInt,deltaAverage,deltaAverageError,profIntLower,profIntUpper]

# Get average and error of the effective average densities:
def deltaBarEffAverageAndError(radii,deltaBarEff,condition,method="poisson"):
    filterToUse = np.where(condition)[0]
    if method == "poisson":
        deltaAverage = np.sum(radii[filterToUse]**3*\
            (1.0 + deltaBarEff[filterToUse]))/np.sum(radii[filterToUse]**3)
        deltaAverageError = np.sqrt(weightedVariance((1.0 + deltaBarEff[filterToUse]),radii[filterToUse]**3))/np.sqrt(len(filterToUse))
    else:
        deltaAverage = np.mean((1.0 + deltaBarEff[filterToUse]))
        deltaAverageError = np.std((1.0 + deltaBarEff[filterToUse]))/\
            np.sqrt(len(filterToUse))
    return [deltaAverage,deltaAverageError]

def deltaVError(deltaVList):
    return (deltaVList[0] - deltaVList[2],deltaVList[3] - deltaVList[0])

def deltaVRatio(deltaVlist):
    return deltaVlist[1]/deltaVlist[0]

# Studying the distribution of the profiles:
def minmax(x):
    return np.array([np.min(x),np.max(x)])

# Get minimum and maximum ranges of the profle distribution:
def getRanges(rhoAH,rhoZV,rBinStackCentres,filterToUseAH=None,\
        filterToUseZV = None):
    if filterToUseAH is None:
        filterToUseAH = np.arange(0,len(rhoAH))
    if filterToUseZV is None:
        filterToUseZV = np.arange(0,len(rhoZV))
    rangesAH = np.zeros((len(rBinStackCentres),2))
    rangesZV = np.zeros((len(rBinStackCentres),2))
    for k in range(0,len(rBinStackCentres)):
        rangesAH[k,:] = minmax(rhoAH[filterToUseAH,k])
        rangesZV[k,:] = minmax(rhoZV[filterToUseZV,k])
    return [rangesAH,rangesZV]


# Plot kstest fit value for gamma distribution for multiple data sets:
def testGammaAcrossBins(rho,arguments="fit"):
    pvals = np.zeros(rho.shape[1])
    for k in range(0,rho.shape[1]):
        if arguments=="fit":
            args = scipy.stats.gamma.fit(rho[:,k])
        elif arguments == "mean":
            mu = np.mean(rho[:,k])
            sigma = np.sqrt(np.var(rho[:,k]))
            args=(mu**2/sigma**2,0,sigma**2/mu)
        else:
            args=arguments[k]
        test = kstest(rho[:,k],'gamma',args=args)
        pvals[k] = test[1]
    return pvals

# Plot kstest fit value for normal distribution for multiple data sets:
def testNormAcrossBins(rho,arguments="fit"):
    pvals = np.zeros(rho.shape[1])
    for k in range(0,rho.shape[1]):
        if arguments=="fit":
            args = scipy.stats.norm.fit(rho[:,k])
        elif arguments == "mean":
            mu = np.mean(rho[:,k])
            sigma = np.sqrt(np.var(rho[:,k]))
            args=(mu,sigma)
        else:
            args=arguments[k]
        test = kstest(rho[:,k],'norm',args=args)
        pvals[k] = test[1]
    return pvals

# Compute mean stacks for a set of snapshots:
def computeMeanStacks(centresList,radiiList,massesList,conditionList,\
        pairsList,volumesList,\
        snapList,nbar,rBins,rMin,rMax,mMin,mMax,\
        method="poisson",errorType = "Weighted",toInclude = "all"):
    if toInclude == "all":
        toInclude = range(0,len(centresList))
    nToStack = len(toInclude)
    nbarjStack = np.zeros((nToStack,len(rBins) - 1))
    sigmaStack = np.zeros((nToStack,len(rBins) - 1))
    if conditionList is None:
        conditionList = [np.ones(len(centres),dtype=bool) \
            for centres in centresList]
    for k in range(0,nToStack):
        [nbarj,sigma] = stackVoidsWithFilter(
            centresList[toInclude[k]],radiiList[toInclude[k]],
            np.where((radiiList[toInclude[k]] > rMin) & \
            (radiiList[toInclude[k]] < rMax) & \
            conditionList[toInclude[k]] & (massesList[toInclude[k]] > mMin) & \
            (massesList[toInclude[k]] <= mMax))[0],snapList[toInclude[k]],\
            rBins = rBins,\
            nPairsList = pairsList[toInclude[k]],
            volumesList=volumesList[toInclude[k]],
            method=method,errorType=errorType)
        nbarjStack[k,:] = nbarj
        sigmaStack[k,:] = sigma
    return [nbarjStack,sigmaStack]


# Get pair counts from the MCMC samples, using a single fixed centre for
# all samples:
def get_all_pair_counts_MCMC(mean_centres_gadget_coord,mean_radii,r_bin_stack,\
                             tree_list,snap_name_list):
    all_pairs = []
    all_volumes = []
    for k in range(0,len(snap_name_list)):
        [pairs_list,volumes_list] = stacking.getPairCounts(
            mean_centres_gadget_coord,mean_radii,snap_name_list[k],
            r_bin_stack,tree=tree_list[k],method='poisson',vorVolumes=None)
        all_pairs.append(pairs_list)
        all_volumes.append(volumes_list)
    return [all_pairs,all_volumes]

# Get pair counts from the MCMC samples, using the MCMC sample centres for 
# all pair counts:
def get_all_pair_counts_MCMC_samples(all_centres_300_gadget,all_radii_300,
                                     r_bin_stack_centres,snap_list,tree_list,
                                     r_bin_stack):
    all_pairs_sample = np.zeros((len(snap_list),len(all_radii_300),\
                               len(r_bin_stack_centres)))
    all_volumes_sample = np.zeros((len(snap_list),len(all_radii_300),\
                                 len(r_bin_stack_centres)))
    for k in range(0,len(snap_list)):
        have_anti_halo = np.isfinite(all_radii_300[:,k])
        no_anti_halo = np.logical_not(have_anti_halo)
        [pairs_list,volumes_list] = getPairCounts(\
            all_centres_300_gadget[k,have_anti_halo,:],\
            all_radii_300[have_anti_halo,k],snap_list[k],r_bin_stack,
            tree=tree_list[k],method='poisson',vorVolumes=None)
        all_pairs_sample[k,have_anti_halo,:] = pairs_list
        all_pairs_sample[k,no_anti_halo,:] = np.nan
        all_volumes_sample[k,have_anti_halo,:] = volumes_list
        all_volumes_sample[k,no_anti_halo,:] = np.nan
    return [all_pairs_sample,all_volumes_sample]

# Compute the mean mcmc profile and its error:
def get_mean_mcmc_profile(all_pairs,all_volumes,cumulative=False):
    # Average the profiles over all MCMC samples:
    if cumulative:
        mean_density = np.nanmean(
            np.cumsum(all_pairs,2)/np.cumsum(all_volumes,2),0)
        mean_vols = np.nanmean(np.cumsum(all_volumes,2),0)
    else:
        mean_density = np.nanmean(all_pairs/all_volumes,0)
        mean_vols = np.nanmean(all_volumes,0)
    # Stacked profile:
    rho_stacked = (
        (np.sum(mean_density*mean_vols,0) + 1)/np.sum(mean_vols,0))
    # Error in stacked profile:
    # Individual profile error bars:
    var_stacked = np.var(mean_density,0)
    weights = mean_vols/np.sum(mean_vols,0)
    # Standard error in the mean:
    num_samples = np.sum(np.isfinite(mean_density),0)
    sigma_density = np.nanstd(mean_density,0)/np.sqrt(num_samples)
    # Combination in quadrature:
    sigma_rho_stacked = np.sqrt(
        np.sum((sigma_density**2 + var_stacked)*weights**2,0))
    return [rho_stacked, sigma_rho_stacked]

# Get the profiles in different regions from random simulations:
def get_profiles_in_regions(all_pairs,all_volumes,cumulative=False):
    if cumulative:
        rho_stacked = np.array([(np.sum(np.cumsum(all_pairs[k],1),0)+1)/\
            np.sum(np.cumsum(all_volumes[k],1),0) \
            for k in range(0,len(all_volumes))])
    else:
        rho_stacked = np.array([(np.sum(all_pairs[k],0)+1)/\
            np.sum(all_volumes[k],0) \
            for k in range(0,len(all_volumes))])
    return rho_stacked

# Get the credible intervals for a set of void profiles in regions:
def get_profile_interval_in_regions(
        profile_dictionary,intervals = [68,95],
        cumulative = False):
    all_pairs = profile_dictionary['pairs']
    all_volumes = profile_dictionary['volumes']
    rho_stacked_un_all = get_profiles_in_regions(
        all_pairs,all_volumes,cumulative=cumulative)
    interval_limits = []
    for lim in intervals:
        interval_limits.append(50 - lim/2)
        interval_limits.append(50 + lim/2)
    credible_intervals = np.percentile(rho_stacked_un_all,\
        interval_limits,axis=0)
    return credible_intervals

# Get void profiles for all voids, regardless of region:
def get_individual_void_profiles(profile_dictionary):
    all_profiles = []
    for pairs, volumes in zip(profile_dictionary['pairs'],\
            profile_dictionary['volumes']):
        all_profiles.append((pairs + 1)/volumes)
    return np.vstack(all_profiles)



