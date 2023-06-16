# Code related to building a void catalogue


class combinedCatalogue:
    # Class to store and compute a combined catalogue
    def __init__(self,snapList,snapListRev,muR,muS,rSphere,\
            ahProps=None,sortMethod="ratio",snapSortList=None,hrList=None,\
            verbose=False,rMin = 5,rMax = 30,massRange = None,\
            NWayMatch = False,additionalFilters = None,sortBy="mass",\
            refineCentres=False,max_index=None):
        self.muR = muR # Radius ratio threshold
        self.muS = muS # Search distance ratio threshold
        self.rSphere = rSphere # Radius from the centre of the simulation box
            # out to which we build the catalogue
        # Load catalogue data:
        self.rMin = rMin # Minimum radius for voids used to build the catalogue
        self.rMax = rMax # Maximum radius for voids used to build the catalogue
        self.massRange = massRange # Mass range for voids in the catalogue
        self.NWayMatch = NWayMatch # Whether to use N-way matching
        self.additionalFilters = additionalFilters # Additional filters to
            # apply to the catalogues
        self.sortBy = sortBy # Whether to sort by mass or radius
        self.refineCentres = refineCentres # Whether to iteratively refine 
            # centres when applying the matching algorithm
        self.max_index = max_index # Maximum number of voids allowed in 
            # each catalogue. If None, no maximum is used.
        [self.snapList,self.snapListRev,self.boxsize,self.ahProps,\
            self.antihaloCentres,self.antihaloMasses,self.antihaloRadii,\
            self.snapSortList,self.volumesList,self.hrList] = \
            loadCatalogueData(snapList,snapListRev,ahProps,sortMethod,\
                snapSortList,hrList,verbose=verbose)
        self.numCats = len(snapList) # Number of catalogues
        # Setup the centres:
        computeShortCentresList()
    def computeShortCentresList(self):
        [self.centresListShort,self.centralAntihalos,self.sortedList,\
            self.ahCounts,self.max_index]  = computeShortCentresList(\
            self.snapList,self.antihaloCentres,\
            self.antihaloRadii,self.antihaloMasses,self.rSphere,\
            self.rMin,self.rMax,massRange = self.massRange,\
            additionalFilter = self.additionalFilters=None,\
            sortBy = self.sortBy,max_index = self.max_index)
    def getShortenedQuantity(quantity,shortenedList):
        return [np.array([quantity[l][\
                self.centralAntihalos[l][0][self.sortedList[l][k]]] \
                for k in range(0,np.min([self.ahCounts[l],self.max_index]))]) \
                for l in range(0,len(shortenedList))]

# Construct an anti-halo catalogue from reversed snapshots
def constructAntihaloCatalogue(snapNumList,samplesFolder="new_chain/",\
        verbose=True,rSphere=135,max_index=None,thresh=0.5,\
        snapname = "gadget_full_forward_512/snapshot_001",\
        snapnameRev = "gadget_full_reverse_512/snapshot_001",\
        fileSuffix= '',matchType='distance',crossMatchQuantity='radius',\
        crossMatchThreshold = 0.5,distMax=20.0,sortMethod='ratio',\
        blockDuplicates=True,twoWayOnly = True,\
        snapList=None,snapListRev=None,ahProps=None,hrList=None,\
        rMin = 5,rMax = 30,mode="fractional",massRange = None,\
        snapSortList = None,overlapList = None,NWayMatch = False,\
        additionalFilters = None,sortBy="mass",refineCentres=False,\
        sortQuantity = 0):
    # Load snapshots:
    [snapList,snapListRev,boxsize,ahProps,antihaloCentres,\
        antihaloMasses,antihaloRadii,snapSortList,volumesList,hrList] = \
        loadCatalogueData(snapNumList,snapList,snapListRev,samplesFolder,\
            snapname,snapnameRev,ahProps,sortMethod,snapSortList,hrList,\
            verbose=verbose)
    numCats = len(snapNumList)
    if verbose:
        print("Loading snapshots...")
    # Construct filtered anti-halo lists:
    [centresListShort,centralAntihalos,sortedList,ahCounts,max_index] = \
        computeShortCentresList(snapNumList,antihaloCentres,\
            antihaloRadii,antihaloMasses,rSphere,rMin,rMax,massRange=massRange,\
            additionalFilters=additionalFilters,sortBy = sortBy,\
            max_index=max_index)
    # Construct new antihalo catalogues from the filtered list:
    if verbose:
        print("Constructing constrained region catalogues...")
    # For some methods, we need to create shortened anti-halo catalogues first
    # otherwise we end up matching a lot of useless halos and wasting time:
    hrListCentral = constructShortenedCatalogues(numCats,matchType,sortMethod,\
        hrList,centralAntihalos,sortedList)
    # List of filtered void antih-halo numbers (pynbody offset):
    shortHaloList = [np.array(centralAntihalos[l][0])[sortedList[l]] + 1 \
        for l in range(0,numCats)]
    # If we are using volume overlaps to match voids, then we need to 
    # create an overlapMap between all pairs of anti-halos in all catalogues:
    if sortMethod == "volumes":
        if overLapList is None:
            overlapList = getOverlapList(numCats,hrListCentral,volumesList)
        else:
            # Check that the supplied overlap list is the correct size. If not,
            # this probably means that a bad overlap list was given:
            if len(overlapList) != int(numCats*(numCats - 1)/2):
                raise Exception("Invalid overlapList!")
    # Construct matches:
    # Create lists of the quantity to match voids with (mass or radius), chosen
    # to match centresListShort:
    quantityListRad = getShortenedQuantity(antihaloRadii,centralAntihalos,\
            centresListShort,sortedList,ahCounts,max_index)
    quantityListMass = getShortenedQuantity(antihaloMasses,\
        centralAntihalos,centresListShort,sortedList,ahCounts,max_index)
    if crossMatchQuantity == 'radius':
        quantityList = quantityListRad
    elif crossMatchQuantity == 'mass':
        quantityList = quantityListMass
    elif crossMatchQuantity == 'both':
        quantityList = quantityList
    else:
        raise Exception('Unrecognised cross-match quantity.')
    # Build a KD tree with the centres in centresListShort for efficient 
    # matching:
    treeList = [scipy.spatial.cKDTree(\
        snapedit.wrap(centres,boxsize),boxsize=boxsize) \
        for centres in centresListShort]
    if verbose:
        print("Computing matches...")
    # Main loop to compute candidate matches:
    [oneWayMatchesAllCatalogues,matchArrayList,allCandidates,\
        allRatios,allDistances] = getOneWayMatchesAllCatalogues(numCats,\
            matchType,snapListRev,hrListCentral,centresListShort,quantityList,\
            max_index,thresh,crossMatchThreshold,ahCounts,quantityListRad,\
            quantityListMass,crossMatchQuantity,treeList,distMax,\
            sortMethod,mode,volumesList)
    # Combined to a single catalogue:
    if verbose:
        print("Combining to a single catalogue...")
    # Lists storing various properties of the final catalogue. Unfortunately, 
    # we can't pre-allocate these as arrays because we don't know the size of
    # the final catalogue before we do the matching, so these are 
    # continuously updated lists:
    twoWayMatchLists = [[] for k in range(0,numCats)] # Stores a list
        # of which matches are two-way matches
    finalCat = [] # Will contain the final catalogue, a list of voids
        # for which we have candidate anti-halos in each mcmc sample
    finalCandidates = [] # Stores the candidate voids in each mcmc sample for
        # each row in the final cadalogue (not just the best)
    finalRatios = [] # Stores the radius (mass) ratios of all the pairs 
        # in the final cataloge
    finalDistances = [] # Stores the distances for all pairs in the final
        # catalogue
    finalCombinatoricFrac = [] # Stores the combinatoric fraction for each void
        # in the final catalogue
    finalCatFrac = [] # Stores the catalogue fraction for each void in the final
        # catalogue
    candidateCounts = [np.zeros((numCats,ahCounts[l]),dtype=int) \
        for l in range(0,numCats)] # Number of candidates
        # that each void could match to.
    # To avoid adding duplicates, we need to remember which voids we have
    # already added to the catalogue somehow. This is achieved using a 
    # boolean array - every time we find a void, we flag it here so that
    # we can later check if it has already been added:
    alreadyMatched = np.zeros((numCats,max_index),dtype=bool)
    # List of the other catalogues, for each catalogue:
    diffMap = [np.setdiff1d(np.arange(0,numCats),[k]) \
        for k in range(0,numCats)]
    # Loop over all catalogues:
    for k in range(0,numCats):
        # Matches for catalogue k to all other catalogues:
        oneWayMatches = oneWayMatchesAllCatalogues[k]
        # Columns corresponding to the other catalogues:
        otherColumns = diffMap[k]
        # One way matches to other catalogues only:
        oneWayMatchesOther = oneWayMatches[:,otherColumns]
        # Loop over all voids in this catalogue:
        for l in range(0,np.min([ahCounts[k],max_index])):
            twoWayMatch = getTwoWayMatches(l,k,otherColumns,numCats,\
                oneWayMatchesAllCatalogues,\
                oneWayMatchesOther=oneWayMatchesOther)
            twoWayMatchLists[k].append(twoWayMatch)
            # Skip if the void has already beeen included, or just
            # doesn't have any two way matches:
            for m in range(0,numCats):
                candidateCounts[k][m,l] = len(allCandidates[k][m][l])
            if not checkIfVoidIsNeeded(l,k,alreadyMatched,twoWayMatch,\
                    otherColumns,candidateCounts,oneWayMatches,\
                    twoWayOnly=twoWayOnly,blockDuplicates=blockDuplicates):
                continue
            matchVoidToOtherCatalogues(l,k,numCats,otherColumns,\
                oneWayMatchesOther,oneWayMatchesAllCatalogues,twoWayMatch,\
                allCandidates,alreadyMatched,candidateCounts,NWayMatch,\
                allRatios,allDistances,diffMap,finalCandidates,\
                finalCat,finalRatios,finalDistances,finalCombinatoricFrac,\
                finalCatFrac,refineCentres,centresListShort,quantityListRad,\
                boxsize,sortQuantity,sortMethod,crossMatchThreshold,distMax,\
                mode,treeList = treeList)
    return [np.array(finalCat),shortHaloList,twoWayMatchLists,\
        finalCandidates,finalRatios,finalDistances,allCandidates,\
        candidateCounts,allRatios,np.array(finalCombinatoricFrac),\
        np.array(finalCatFrac),alreadyMatched]



# Load simulations and catalogue data so that we can combine them. If these
# are already loaded, this function won't reload them.
def loadCatalogueData(snapList,snapListRev,ahProps,sortMethod,snapSortList,\
        hrList,verbose=False):
    if snapList is None:
        snapList =  [pynbody.load(samplesFolder + "sample" + \
            str(snapNum) + "/" + snapname) for snapNum in snapNumList]
    if snapListRev is None:
        snapListRev =  [pynbody.load(samplesFolder + "sample" + \
            str(snapNum) + "/" + snapnameRev) for snapNum in snapNumList]
    # If snaplists are strings, then load them:
    if type(snapList[0]) == str:
        snapList = [pynbody.load(snap) for snap in snapList]
    if type(snapListRev[0]) == str:
        snapListRev = [pynbody.load(snap) for snap in snapListRev]
    # Load centres so that we can filter to the constrained region:
    boxsize = snapList[0].properties['boxsize'].ratio("Mpc a h**-1")
    if verbose:
        print("Extracting anti-halo properties...")
    if ahProps is None:
        ahProps = [tools.loadPickle(snap.filename + ".AHproperties.p") \
            for snap in snapList]
    antihaloCentres = [tools.remapAntiHaloCentre(props[5],boxsize) \
        for props in ahProps]
    antihaloMasses = [props[3] for props in ahProps]
    antihaloRadii = [props[7] for props in ahProps]
    if sortMethod == "volumes":
        if snapSortList is None:
            snapSortList = [np.argsort(snap['iord']) for snap in snapList]
        volumesList = [ahProps[k][4][snapSortList[k]] \
            for k in range(0,len(ahProps))]
    else:
        volumesList = [None for k in range(0,len(ahProps))]
    # Load anti-halo catalogues:
    if hrList is None:
        hrList = [snap.halos() for snap in snapListRev]
    return [snapList,snapListRev,boxsize,ahProps,antihaloCentres,\
        antihaloMasses,antihaloRadii,snapSortList,volumesList,hrList]


# Select the voids which will be included in catalogue matching, applying
# radius and mass cuts, and other arbitrary cuts (such as signel-to-noise):
def computeShortCentresList(snapNumList,antihaloCentres,antihaloRadii,\
        antihaloMasses,rSphere,rMin,rMax,massRange=None,additionalFilters=None,\
        sortBy = "mass",max_index=None):
    # Build a filter from the specified radius range:
    filterCond = [(antihaloRadii[k] > rMin) & (antihaloRadii[k] <= rMax) \
            for k in range(0,len(snapNumList))]
    # Further filter on mass, if mass limits are specified:
    if massRange is not None:
        if len(massRange) < 2:
            raise Exception("Mass range must have an upper and a lower " + \
                "limit.")
        for k in range(0,len(snapNumList)):
            filterCond[k] = filterCond[k] & \
                (antihaloMasses[k] > massRange[0]) & \
                (antihaloMasses[k] <= massRange[1])
    # Apply any additional filters (such as signal-to-noise):
    if additionalFilters is not None:
        for k in range(0,len(snapNumList)):
            filterCond[k] = filterCond[k] & additionalFilters[k]
    # Select anti-halos within rSphere of the centre of the box, applying
    # this filter:
    centralAntihalos = [tools.getAntiHalosInSphere(antihaloCentres[k],\
        rSphere,filterCondition = filterCond[k]) \
        for k in range(0,len(snapNumList))]
    # Sort the list on either mass or radius:
    if sortBy == "mass":
        centralAntihaloMasses = [\
            antihaloMasses[k][centralAntihalos[k][0]] \
            for k in range(0,len(centralAntihalos))]
        sortedList = [np.flip(np.argsort(centralAntihaloMasses[k])) \
            for k in range(0,len(snapNumList))]
    elif sortBy == "radius":
        centralAntihaloRadii = [\
            antihaloRadii[k][centralAntihalos[k][0]] \
            for k in range(0,len(centralAntihalos))]
        sortedList = [np.flip(np.argsort(centralAntihaloRadii[k])) \
            for k in range(0,len(snapNumList))]
    else:
        raise Exception("sortBy argument not recognised.")
    # Include the option to impose an artificial cut on the length of the
    # void list in each catalogue:
    ahCounts = np.array([len(cahs[0]) for cahs in centralAntihalos])
    if max_index is None:
        max_index = np.max(ahCounts)
    # Construct the list of centres for all voids:
    centresListShort = [np.array([antihaloCentres[l][\
        centralAntihalos[l][0][sortedList[l][k]],:] \
        for k in range(0,np.min([ahCounts[l],max_index]))]) \
        for l in range(0,len(snapNumList))]
    return [centresListShort,centralAntihalos,sortedList,ahCounts,max_index]


