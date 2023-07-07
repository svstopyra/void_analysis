# Code related to building a void catalogue
import pynbody
import numpy as np
import scipy
from void_analysis import tools, snapedit

class combinedCatalogue:
    # Class to store and compute a combined catalogue
    def __init__(self,snapList,snapListRev,muR,muS,rSphere,\
            ahProps=None,sortMethod="ratio",snapSortList=None,hrList=None,\
            verbose=False,rMin = 5,rMax = 30,massRange = None,\
            NWayMatch = False,additionalFilters = None,sortBy="mass",\
            refineCentres=False,max_index=None,enforceExclusive=False,\
            blockDuplicates=True,iterMax=100,matchType='distance',\
            crossMatchQuantity='radius',pynbodyThresh=0.5,twoWayOnly=True,\
            mode="fractional",overlapList = None):
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
        self.pynbodyThresh=pynbodyThresh
        self.matchType = matchType
        self.crossMatchQuantity = crossMatchQuantity
        self.sortMethod = sortMethod
        self.mode = mode
        [self.snapList,self.snapListRev,self.boxsize,self.ahProps,\
            self.antihaloCentres,self.antihaloMasses,self.antihaloRadii,\
            self.snapSortList,self.volumesList,self.hrList] = \
            loadCatalogueData(snapList,snapListRev,ahProps,sortMethod,\
                snapSortList,hrList,verbose=verbose)
        self.numCats = len(snapList) # Number of catalogues
        self.enforceExclusive = enforceExclusive
        self.blockDuplicates = blockDuplicates
        self.verbose = verbose
        # Precompute lists of other columns, so we don't have to do it 
        # every time:
        self.iterMax = iterMax # Maximum number of iterations for refining
        # Setup the centres:
        self.computeShortCentresList()
        self.twoWayOnly = twoWayOnly
        self.overLapList = overlapList
        # Lists storing various properties of the final catalogue. 
        # Unfortunately, we can't pre-allocate these as arrays because we don't 
        # know the size of the final catalogue before we do the matching, 
        # so these are continuously updated lists:
        self.twoWayMatchLists = [[] for k in range(0,self.numCats)] # Stores 
            # a list of which matches are two-way matches
        self.finalCat = [] # Will contain the final catalogue, a list of voids
            # for which we have candidate anti-halos in each mcmc sample
        self.finalCandidates = [] # Stores the candidate voids in each mcmc 
            # sample for each row in the final cadalogue (not just the best)
        self.finalRatios = [] # Stores the radius (mass) ratios of all the 
            # pairs in the final cataloge
        self.finalDistances = [] # Stores the distances for all pairs in the 
            # final catalogue
        self.finalCombinatoricFrac = [] # Stores the combinatoric fraction for 
            # each void in the final catalogue
        self.finalCatFrac = [] # Stores the catalogue fraction for each void in 
            # the final catalogue
        self.candidateCounts = [np.zeros((self.numCats,self.ahCounts[l]),\
            dtype=int) for l in range(0,self.numCats)] # Number of candidates
            # that each void could match to.
        # To avoid adding duplicates, we need to remember which voids we have
        # already added to the catalogue somehow. This is achieved using a 
        # boolean array - every time we find a void, we flag it here so that
        # we can later check if it has already been added:
        self.alreadyMatched = np.zeros((self.numCats,self.max_index),dtype=bool)
        # List of the other catalogues, for each catalogue:
        self.diffMap = [np.setdiff1d(np.arange(0,self.numCats),[k]) \
            for k in range(0,self.numCats)]
        self.iteratedCentresList = []
        self.iteratedRadiiList = []
    # Construct an anti-halo catalogue from reversed snapshots
    def constructAntihaloCatalogue(self):
        # Load snapshots:
        # Construct new antihalo catalogues from the filtered list:
        if self.verbose:
            print("Constructing constrained region catalogues...")
        # For some methods, we need to create shortened anti-halo catalogues 
        # first otherwise we end up matching a lot of useless halos and wasting 
        # time:
        self.hrListCentral = self.constructShortenedCatalogues()
        # If we are using volume overlaps to match voids, then we need to 
        # create an overlapMap between all pairs of anti-halos in all catalogues:
        if self.sortMethod == "volumes":
            if self.overLapList is None:
                self.overlapList = getOverlapList()
            else:
                # Check that the supplied overlap list is the correct size. 
                # If not this probably means that a bad overlap list was given:
                if len(overlapList) != int(self.numCats*(self.numCats - 1)/2):
                    raise Exception("Invalid overlapList!")
        # Construct matches:
        # Create lists of the quantity to match voids with (mass or radius),
        # chosen to match centresListShort:
        self.radiusListShort = self.getShortenedQuantity(self.antihaloRadii,\
            self.centralAntihalos)
        self.massListShort = self.getShortenedQuantity(self.antihaloMasses,\
            self.centralAntihalos)
        if self.crossMatchQuantity == 'radius':
            self.quantityList = self.radiusListShort
        elif self.crossMatchQuantity == 'mass':
            self.quantityList = self.massListShort
        else:
            raise Exception('Unrecognised cross-match quantity.')
        if self.verbose:
            print("Computing matches...")
        # Main loop to compute candidate matches:
        [self.oneWayMatchesAllCatalogues,self.matchArrayList,\
            self.allCandidates,self.allRatios,self.allDistances] = \
            self.getOneWayMatchesAllCatalogues()
        # Combined to a single catalogue:
        if self.verbose:
            print("Combining to a single catalogue...")
        # Loop over all catalogues:
        for k in range(0,self.numCats):
            # Matches for catalogue k to all other catalogues:
            oneWayMatches = self.oneWayMatchesAllCatalogues[k]
            # Loop over all voids in this catalogue:
            for l in range(0,np.min([self.ahCounts[k],self.max_index])):
                twoWayMatch = self.getTwoWayMatches(l,k)
                self.twoWayMatchLists[k].append(twoWayMatch)
                # Skip if the void has already beeen included, or just
                # doesn't have any two way matches:
                for m in range(0,self.numCats):
                    self.candidateCounts[k][m,l] = \
                        len(self.allCandidates[k][m][l])
                if not self.checkIfVoidIsNeeded(l,k,twoWayMatch,oneWayMatches):
                    continue
                voidMatches = self.matchVoidToOtherCatalogues(l,k,twoWayMatch)
        return np.array(self.finalCat)
    def getMatchPynbody(self,n1,n2,fractionType='normal'):
        snap1 = self.snapListRev[n1]
        snap2 = self.snapListRev[n2]
        cat1 = self.hrListCentral[n1]
        cat2 = self.hrListCentral[n2]
        quantity1 = self.quantityList[n1]
        quantity2 = self.quantityList[n2]
        bridge = pynbody.bridge.OrderBridge(snap1,snap2,monotonic=False)
        # Ensure ratio is above the threshold:
        match = [-2]
        candidatesList = []
        # get shared particle fractions:
        catTransfer = bridge.catalog_transfer_matrix(groups_1 = cat1,\
            groups_2 = cat2,max_index=self.max_index)
        cat2Lengths = np.array([len(halo) for halo in cat2])
        for k in range(0,np.min([len(cat1),self.max_index])):
            # get quantity (mass or radius) ratio, defined as lower/highest
            # so that it is always <= 1:
            bigger = np.where(quantity2 > quantity1[k])[0]
            quantRatio = quantity2/quantity1[k]
            quantRatio[bigger] = quantity1[k]/quantity2[bigger]
            if fractionType == 'normal':
                fraction = catTransfer[k]/len(cat1[k+1])
            elif fractionType == 'symmetric':
                fraction = catTransfer[k]/np.sqrt(len(cat1[k+1]*cat2Lengths))
            else:
                raise Exception("Unrecognised fraction type requested.")
            # Find candidates that satisfy the threshold requirements:
            candidates = np.where((quantRatio >= self.muR) & \
                (fraction > self.pynbodyThresh))[0]
            candidatesList.append(candidates)
            if len(candidates) < 1:
                match.append(-1)
            else:
                # Select the match with the highest shared particle fraction
                # as the most probable:
                mostProbable = candidates[np.argmax(fraction[candidates])]
                match.append(mostProbable + 1)
            return [np.array(match,dtype=int),candidatesList]
    def getAllCandidatesFromTrees(self,n1,n2):
        centres1 = self.centresListShort[n1]
        quantity1 = self.quantityList[n1]
        quantity2 = self.quantityList[n2]
        tree1 = self.treeList[n1]
        tree2 = self.treeList[n2]
        searchRadii = self.getSearchRadii(quantity1,quantity2)
        if self.mode == "fractional":
            # Interpret distMax as a fraction of the void radius, not the 
            # distance in Mpc/h.
            # Choose a search radius that is no greater than the void radius 
            # divided by the radius ratio. If the other anti-halo is further 
            # away than this then it wouldn't match to us anyway, so we don't 
            # need to consider it.
            searchOther = tree2.query_ball_point(snapedit.wrap(\
                centres1,self.boxsize),searchRadii,workers=-1)
        else:
            searchOther = tree1.query_ball_tree(tree2,self.muS)
        return [searchRadii,searchOther]
    # Function to match all voids:
    def getMatchDistance(self,n1,n2,overlap = None):
        snap1 = self.snapListRev[n1]
        snap2 = self.snapListRev[n2]
        centres1 = self.centresListShort[n1]
        centres2 = self.centresListShort[n2]
        quantity1 = self.quantityList[n1]
        quantity2 = self.quantityList[n2]
        tree1 = self.treeList[n1]
        tree2 = self.treeList[n2]
        cat1 = self.hrListCentral[n1]
        cat2 = self.hrListCentral[n2]
        volumes1 = self.volumesList[n1]
        volumes2 = self.volumesList[n2]
        # Our procedure here is to get the closest anti-halo that lies within the 
        # threshold:
        match = [-2] # Always include -2, for compatibility with pynbody output
        candidatesList = []
        ratioList = []
        distList = []
        # Fina candidates for all anti-halos:
        [searchRadii,searchOther] = self.getAllCandidatesFromTrees(n1,n2)
        # Build an overlap map, if we are using this method:
        if overlap is None and self.sortMethod == "volumes":
            if cat1 is None or cat2 is None or \
            volumes1 is None or volumes2 is None:
                raise Exception("Anti-halo catalogue required for " + \
                    "volumes based matching.")
            overlap = overlapMap(cat1,cat2,volumes1,volumes2)
        # Process all the candidates, to find which are above the specified 
        # thresholds:
        for k in range(0,np.min([len(centres1),self.max_index])):
            candidates = searchOther[k]
            centre = centres1[k]
            if overlap is None:
                overlapForVoid = None
            else:
                overlapForVoid = overlap[k]
            [selectedMatches,selectCandidates,selectedQuantRatios,\
                selectedDistances] = self.findAndProcessCandidates(\
                    n2,centre,quantity1[k],searchRadii,candidates=candidates,\
                    overlapForVoid=overlapForVoid)
            candidatesList.append(selectCandidates)
            ratioList.append(selectedQuantRatios)
            distList.append(selectedDistances)
            match.append(selectedMatches)
        return [np.array(match,dtype=int),candidatesList,ratioList,distList]
    # Perform matching between two catalogues:
    def getMatchCandidatesTwoCatalogues(self,n1,n2):
        if self.matchType == 'pynbody':
            # Use pynbody's halo catalogue matching to identify likely
            # matches:
            [match, candidatesList] = self.getMatchPynbody(n1,n2)
            ratioList = None
            distList = None
        elif self.matchType == 'distance':
            # This is the conventional approach: matching on distance
            # radius criteria:
            [match, candidatesList,ratioList,distList] = \
                self.getMatchDistance(n1,n2)
        else:
            raise Exception("Unrecognised matching type requested.")
        return [match, candidatesList,ratioList,distList]
    def getOneWayMatchesAllCatalogues(self):
        matchArrayList = [[] for k in range(0,self.numCats)] # List of best 
            # matches in each one way pair
        allCandidates = [] # List of candidates for each one way pair
        allRatios = [] # List of the radius (or mass) ratios of each candidate
        allDistances = [] # List of distances from a void to all it's candidates
        for k in range(0,self.numCats):
            matchArrayListNew = matchArrayList[k]
            allCandidatesNew = []
            allRatiosNew = []
            allDistancesNew = []
            for l in range(0,self.numCats):
                if l != k:
                    [match, candidatesList,ratioList,distList] = \
                        self.getMatchCandidatesTwoCatalogues(k,l)
                    matchArrayListNew.append(match)
                elif l == k:
                    # Placeholder entries when matching a catalogue to itself:
                    matchArrayListNew.append([-2] + \
                        list(np.arange(1,np.min([self.ahCounts[l],\
                        self.max_index])+1)))
                    candidatesList = np.array([[m] \
                        for m in np.arange(1,np.min([self.ahCounts[l],\
                        self.max_index])+1)])
                    ratioList = np.ones(len(candidatesList))
                    distList = np.zeros(len(candidatesList))
                allCandidatesNew.append(candidatesList)
                allRatiosNew.append(ratioList)
                allDistancesNew.append(distList)
            matchArrayList[k] = list(matchArrayListNew)
            allCandidates.append(allCandidatesNew)
            allRatios.append(allRatiosNew)
            allDistances.append(allDistancesNew)
        # Here we arrange the one-way matches of every void in every 
        # catalogue. This is a list of Ncat matrices, one for each catalogue.
        # Each matrix has dimensions Nvoids_i x Ncat (Nvoids_i being the number of
        # voids in catalogue i):
        oneWayMatchesAllCatalogues = \
            [np.array(matchArrayList[k]).transpose()[1:,:] \
            for k in range(0,self.numCats)]
        return [oneWayMatchesAllCatalogues,matchArrayList,allCandidates,\
            allRatios,allDistances]
    def getOverlapList(self):
        overlapList = []
        for k in range(0,self.numCats):
            for l in range(0,self.numCats):
                if k >= l:
                    continue
                overlapList.append(overlapMap(self.hrListCentral[k],\
                    self.hrListCentral[l],self.volumesList[k],\
                    self.volumesList[l],verbose=False))
        return overlapList
    def constructShortenedCatalogues(self):
        if self.matchType == "pynbody" or self.sortMethod == "volumes":
            # These methods need copies of the catalogues, which we will then
            # filter below:
            hrListCentral = [copy.deepcopy(halos) for halos in self.hrList]
        else:
            # Other methods don't need these catalogues, so we store \
            # place-holders:
            hrListCentral = [None for halos in self.hrList]
        # List of the halo numbers (in pynbody convention) for all anti-halos
        # being used for matching:
        for l in range(0,self.numCats):
            if self.matchType == "pynbody" or self.sortMethod == "volumes":
                # Manually edit the pynbody anti-halo catalogues to
                # only include the relevant anti-halos:
                hrListCentral[l]._halos = dict([(k+1,\
                    self.hrList[l][self.centralAntihalos[l][0][\
                    self.sortedList[l][k]]+1]) \
                    for k in range(0,len(self.centralAntihalos[l][0]))])
                hrListCentral[l]._nhalos = len(self.centralAntihalos[l][0])
        return hrListCentral
    def computeShortCentresList(self):
        [self.centresListShort,self.centralAntihalos,self.sortedList,\
            self.ahCounts,self.max_index]  = computeShortCentresList(\
            self.snapList,self.antihaloCentres,\
            self.antihaloRadii,self.antihaloMasses,self.rSphere,\
            self.rMin,self.rMax,massRange = self.massRange,\
            additionalFilters = self.additionalFilters,\
            sortBy = self.sortBy,max_index = self.max_index)
        # Build a KD tree with the centres in centresListShort for efficient 
        # matching:
        self.treeList = [scipy.spatial.cKDTree(\
            snapedit.wrap(centres,self.boxsize),boxsize=self.boxsize) \
            for centres in self.centresListShort]
    def getShortenedQuantity(self,quantity,shortenedList):
        return [np.array([quantity[l][\
                self.centralAntihalos[l][0][self.sortedList[l][k]]] \
                for k in range(0,np.min([self.ahCounts[l],self.max_index]))]) \
                for l in range(0,len(shortenedList))]
    # Get the two way matches of a void in other catalogues:
    def getTwoWayMatches(self,nVoid,nCat):
        oneWayMatches = self.oneWayMatchesAllCatalogues[nCat]
        otherColumns = self.diffMap[nCat]
        oneWayMatchesOther = oneWayMatches[:,otherColumns]
        twoWayMatch = np.zeros(oneWayMatchesOther[nVoid].shape,dtype=bool)
        for m in range(0,self.numCats-1):
            if oneWayMatches[nVoid,otherColumns[m]] < 0:
                # Fails if we don't match to anything
                twoWayMatch[m] = False
            else:
                # 2-way only if the other matches back to this:
                twoWayMatch[m] = (\
                    self.oneWayMatchesAllCatalogues[otherColumns[m]][\
                    oneWayMatches[nVoid,otherColumns[m]] - 1,nCat] == nVoid+1)
                if self.enforceExclusive:
                    twoWayMatch[m] = twoWayMatch[m] and \
                        (not self.alreadyMatched[m,oneWayMatches[nVoid][m]-1])
        return twoWayMatch
    # Check which of a void's matches are new:
    def getNewMatches(self,nVoid,nCat,oneWayMatches):
        isNewMatch = np.zeros(\
            oneWayMatches[nVoid].shape,dtype=bool)
        for m in range(0,len(isNewMatch)):
            if (oneWayMatches[nVoid][m] > 0) and (m != nCat):
                if self.blockDuplicates:
                    isNewMatch[m] = \
                    not self.alreadyMatched[m,oneWayMatches[nVoid][m]-1]
                else:
                    isNewMatch[m] = True
            if (oneWayMatches[nVoid][m] < 0):
                isNewMatch[m] = False
        return isNewMatch
    # Check whether we should include a void or not:
    def checkIfVoidIsNeeded(self,nVoid,nCat,twoWayMatch,oneWayMatches):
        otherColumns = self.diffMap[nCat]
        voidAlreadyFound = self.alreadyMatched[nCat,nVoid]
        atLeastOneTwoWayMatch = np.any(twoWayMatch)
        atLeastOneMatchWithUniqueCandidate = np.any(\
            self.candidateCounts[nCat][otherColumns,nVoid] == 1)
        haveNewMatch = self.getNewMatches(nVoid,nCat,oneWayMatches)
        atLeastOneNewMatch = np.any(haveNewMatch[otherColumns])
        if self.twoWayOnly:
            needed = (not voidAlreadyFound) and atLeastOneTwoWayMatch and \
                atLeastOneMatchWithUniqueCandidate and atLeastOneNewMatch
        else:
            needed = (not voidAlreadyFound) and \
                atLeastOneMatchWithUniqueCandidate and atLeastOneNewMatch
        return needed
    # Remove anything which is missing, or already been found, from 
    # consideration
    def getUniqueEntriesInCatalogueRow(self,catalogueRow):
        isNeededList = np.zeros(self.numCats,dtype=bool)
        for ns in range(0,self.numCats):
            if catalogueRow[ns] > -1:
                # Missing voids automatically fail. For others, they pass
                # only if they haven't been found:
                isNeededList[ns] = (not self.alreadyMatched[ns,\
                    catalogueRow[ns]-1])
        return isNeededList
    def gatherCandidatesRatiosAndDistances(self,nCat,nVoid):
        candm = []
        ratiosm = []
        distancesm = []
        for m in range(0,self.numCats):
            if (m != nCat):
                candm.append(self.allCandidates[nCat][m][nVoid])
                ratiosm.append(self.allRatios[nCat][m][nVoid])
                distancesm.append(self.allDistances[nCat][m][nVoid])
        return [candm,ratiosm,distancesm]
    # Mark the two-way matches of a void as already found, so that we don't 
    # accidentally include them:
    def markCompanionsAsFound(self,nVoid,nCat,voidMatches):
        for m in range(0,self.numCats):
            if (m != nCat) and (voidMatches[m] > 0):
                # Only deem something to be already matched
                # if it maps back to this with a single unique 
                # candidate
                self.alreadyMatched[m][voidMatches[m] - 1] = \
                    (self.oneWayMatchesAllCatalogues[m]\
                    [voidMatches[m] - 1,nCat] == nVoid+1) \
                    and (len(self.allCandidates[m][nCat]\
                    [voidMatches[m] - 1]) == 1)
            if m == nCat:
                self.alreadyMatched[m][nVoid] = True
    # Get all the possible matches of a given void by following chains of 
    # two way matches:
    def followAllMatchChains(self,nVoid,nCat,oneWayMatches):
        # Track down all possible matches that are connected
        # to this one.
        # First, get an initial scan of possible candidates:
        allCands = [[] for m in range(0,self.numCats)] # Candidates that could
            # be connected to this void
        lengthsList = np.zeros(self.numCats,dtype=int) # Number of candidates 
            # in each catalogue
        for m in range(0,self.numCats):
            twoWayCand = oneWayMatches[nVoid][m]
            haveMatch = twoWayCand > -1
            alreadyIncluded = self.alreadyMatched[m][twoWayCand - 1]
            if haveMatch and (not alreadyIncluded):
                allCands[m].append(twoWayCand-1)
        # Number of candidates we have in each catalogue:
        lengthsListNew = np.array(\
            [len(cand) for cand in allCands],dtype=int)
        # Keep iterating until we stop finding new matches:
        while not np.all(lengthsListNew == lengthsList):
            lengthsList = lengthsListNew
            # Loop over all catalogues:
            for n in range(0,self.numCats):
                if len(allCands[n]) > 0:
                    # If we have candidates, follow their 
                    # 2-way matches to the other catalogues.
                    # Loop over all other catalogues, d, and 
                    # the candidate anti-halos in catalogue n:
                    for d in self.diffMap[n]:
                        for m in range(0,len(allCands[n])):
                            # For each antihalo in catalogue n, 
                            # get the candidates in catalogue d
                            # to which it has two-way matches:
                            otherCatCandidates = \
                                self.allCandidates[n][d][\
                                allCands[n][m]]
                            if len(otherCatCandidates) > 0:
                                # The first candidate is the
                                # two way match:
                                bestCand = otherCatCandidates[0]
                                # Add this iff we haven't
                                # already found it, and it
                                # hasn't already been marked
                                # as belonging to another
                                # void:
                                inOtherList = np.isin(\
                                        bestCand,allCands[d])
                                alreadyIncluded = \
                                    self.alreadyMatched[d,bestCand]
                                if (not inOtherList) and \
                                        (not alreadyIncluded):
                                    allCands[d].append(bestCand)
            lengthsListNew = np.array([len(cand) \
                for cand in allCands],dtype=int)
        return allCands
    # Count the number of two way matches in a list with candidates for each
    # mcmc sample, when using N-way matching
    def getNumberOfTwoWayMatchesNway(self,allCands):
        twoWayMatchesAllCands = [[] for m in range(0,self.numCats)]
        for m in range(0,self.numCats):
            for n in range(0,len(allCands[m])):
                nTW = np.sum(np.array(\
                    [len(self.allCandidates[m][d][allCands[m][n]]) \
                    for d in self.diffMap[m]]) > 0)
                twoWayMatchesAllCands[m].append(nTW)
        return twoWayMatchesAllCands
    # Count the number of two way matches.
    # TODO - can we merge this with getNumberOfTwoWayMatchesNway? They are 
    # similar, but doing slightly different things:
    def getTotalNumberOfTwoWayMatches(self,voidMatches):
        twoWayMatchCounts = 0
        for m in range(0,self.numCats):
            for d in self.diffMap[m]:
                allCands = self.allCandidates[m][d][\
                    voidMatches[m]-1]
                if len(allCands) > 0:
                    if allCands[0] == voidMatches[d]-1:
                        twoWayMatchCounts += 1
        return twoWayMatchCounts
    # Compute a quantity (such as radius ratio or distance ratio) which 
    # is defined for all candidates connected to a particular void:
    def computeQuantityForCandidates(self,quantity,allCands):
        quantityAverages = [[] for k in range(0,self.numCats)]
        for m in range(0,self.numCats):
            for n in range(0,len(allCands[m])):
                individualQuantities = np.zeros(self.numCats-1)
                for d in range(0,self.numCats-1):
                    if len(quantity[m][self.diffMap[m][d]][\
                            allCands[m][n]]) > 0:
                        individualQuantities[d] = quantity[m][\
                            self.diffMap[m][d]][allCands[m][n]][0]
                qR = np.mean(individualQuantities)
                quantityAverages[m].append(qR)
        return quantityAverages
    def applyNWayMatching(self,nVoid,nCat,oneWayMatches):
        # Follow all chains of two way matches to get possible void candidates:
        allCands = self.followAllMatchChains(nVoid,nCat,oneWayMatches)
        # Count two way matches:
        twoWayMatchesAllCands = self.getNumberOfTwoWayMatchesNway(allCands)
        # Compute the average radius ratio:
        ratioAverages = self.computeQuantityForCandidates(self.allRatios,\
            allCands)
        # Compute the distances:
        distAverages = self.computeQuantityForCandidates(self.allDistances,\
            allCands)
        # Now figure out the best candidates to include:
        bestCandidates = -np.ones(self.numCats,dtype=int)
        bestRatios = np.zeros(self.numCats)
        bestDistances = np.zeros(self.numCats)
        numberOfLinks = 0
        for m in range(0,self.numCats):
            if len(allCands[m]) == 1:
                bestCandidates[m] = allCands[m][0] + 1
                bestRatios[m] = ratioAverages[m][0]
                bestDistances[m] = distAverages[m][0]
                numberOfLinks += twoWayMatchesAllCands[m][0]
            elif len(allCands[m]) > 1:
                maxTW = np.max(allCands[m])
                haveMaxTW = np.where(\
                    np.array(allCands[m]) == maxTW)[0]
                if len(haveMaxTW) > 1:
                    # Need to use the ratio criteria to choose
                    # instead if we have a tie:
                    maxRat = np.max(ratioAverages[m])
                    haveMaxRat = np.where(\
                        np.array(ratioAverages[m]) == maxRat)[0]
                    bestCandidates[m] = allCands[m][\
                        haveMaxRat[0]] + 1
                    bestRatios[m] = ratioAverages[m][\
                        haveMaxRat[0]]
                    bestDistances[m] = distAverages[m][\
                        haveMaxRat[0]]
                    numberOfLinks += twoWayMatchesAllCands[m][\
                        haveMaxRat[0]]
                else:
                    # Otherwise, we just use the one with the most number
                    # of two way matches:
                    bestCandidates[m] = allCands[m][\
                        haveMaxTW[0]] + 1
                    bestRatios[m] = ratioAverages[m][\
                        haveMaxTW[0]]
                    bestDistances[m] = distAverages[m][\
                        haveMaxTW[0]]
                    numberOfLinks += twoWayMatchesAllCands[m][\
                        haveMaxTW[0]]
            # If no candidates, just leave it as -1
        # Now we mark the other voids as already included. Remember that we
        # only include the best matches as already included, freeing the
        # other candidates to possibly be included elsewhere:
        for m in range(0,self.numCats):
            self.alreadyMatched[m,bestCandidates[m] - 1] = True
        return [bestCandidates,bestRatios,bestDistances,numberOfLinks]
    def getAllQuantityiesFromVoidMatches(self,voidMatches,quantitiesList):
        quantitiesArray = []
        for l in range(0,self.numCats):
            if voidMatches[l] > -1:
                quantitiesArray.append(quantitiesList[l][voidMatches[l] - 1])
        return quantitiesArray
    def getStdCentreFromVoidMatches(self,voidMatches,centresList):
        centresArray = getAllQuantityiesFromVoidMatches(voidMatches,centresList)
        stdCentres = np.std(centresArray,0)
        return stdCentres
    def getMeanCentreFromVoidMatches(self,voidMatches):
        centresArray = self.getAllQuantityiesFromVoidMatches(voidMatches,\
            self.centresListShort)
        meanCentres = np.mean(centresArray,0)
        return meanCentres
    def getMeanRadiusFromVoidMatches(self,voidMatches):
        radiusArray = self.getAllQuantityiesFromVoidMatches(voidMatches,\
            self.radiusListShort)
        meanRadius = np.mean(radiusArray)
        return meanRadius
    # Go through a void list and remove anything that has previously been 
    # included as part of a different void:
    def removeAlreadyIncludedVoids(self,voidMatches):
        voidMatchesFiltered = np.array(voidMatches)
        for m in range(0,self.numCats):
            if self.alreadyMatched[m,voidMatches[m]-1]:
                voidMatchesFiltered[m] = -1
        return voidMatchesFiltered
    # Get the radius/mass ratios of all candidates, without sorting:
    def getUnsortedRatios(self,candidates,searchQuantity,otherQuantities):
        if not np.isscalar(searchQuantity):
            # This happens if we are checking multiple quantities, or possibly
            # multiple thresholds, simultaneously.
            nQuantLen = len(searchQuantity) # Number of quantities being
                # tested
            quantRatio = np.zeros((len(candidates),nQuantLen))
            for l in range(0,nQuantLen):
                bigger = np.where(\
                    otherQuantities[candidates,l] > searchQuantity[l])[0]
                quantRatio[:,l] = otherQuantities[candidates,l]/\
                    searchQuantity[l]
                quantRatio[bigger,l] = searchQuantity[l]/\
                    otherQuantities[candidates,l][bigger]
        else:
            bigger = np.where(\
                otherQuantities[candidates] > searchQuantity)[0]
            quantRatio = otherQuantities[candidates]/searchQuantity
            quantRatio[bigger] = searchQuantity/\
                otherQuantities[candidates][bigger]
        return quantRatio
    def sortCandidatesByDistance(self,candidates,quantRatio,distances):
        indSort = np.argsort(distances)
        sortedCandidates = np.array(candidates,dtype=int)[indSort]
        quantRatio = quantRatio[indSort]
        return [quantRatio,sortedCandidates,indSort]
    def sortQuantRatiosByRatio(self,candidates,quantRatio,distances):
        # sort the quantRatio from biggest to smallest, so we find
        # the most similar anti-halo within the search distance:
        indSort = np.flip(np.argsort(quantRatio))
        quantRatio = quantRatio[indSort]
        sortedCandidates = np.array(candidates,dtype=int)[indSort]
        return [quantRatio,sortedCandidates,indSort]
    def sortCandidatesByVolumes(self,candidates,quantRatio,overlapForVoid):
        volOverlapFrac = overlapForVoid[candidates]
        indSort = np.flip(np.argsort(volOverlapFrac))
        quantRatio = quantRatio[indSort]
        sortedCandidates = np.array(candidates,dtype=int)[indSort]
        return [quantRatio,sortedCandidates,indSort]
    def getSortedQuantRatio(self,candidates,quantRatio,distances,\
            overlapForVoid):
        if self.sortMethod == 'distance':
            # Sort the antihalos by distance. Candidate is the closest
            # halo which satisfies the threshold criterion:
            [quantRatio,sortedCandidates,indSort] = \
                self.sortCandidatesByDistance(candidates,quantRatio,distances)
        elif self.sortMethod == 'ratio':
            # sort the quantRatio from biggest to smallest, so we find
            # the most similar anti-halo within the search distance:
            [quantRatio,sortedCandidates,indSort] = \
                self.sortQuantRatiosByRatio(candidates,quantRatio,distances)
        elif self.sortMethod == "volumes":
            [quantRatio,sortedCandidates,indSort] = \
                self.sortCandidatesByVolumes(candidates,quantRatio,\
                    overlapForVoid)
        else:
            raise Exception("Unrecognised sorting method")
        return [quantRatio,sortedCandidates,indSort]
    def getVoidsAboveThresholds(self,quantRatio,distances,\
            searchQuantity,otherQuantities,sortedCandidates,indSort):
        # Number of search quantities (radius or mass) to process for this void:
        if np.isscalar(searchQuantity):
            nQuantLen = 1
        else:
            nQuantLen = len(searchQuantity)
        if self.mode == "fractional":
            if not np.isscalar(searchQuantity):
                candRadii = otherQuantities[sortedCandidates,0]
                # Geometric mean of radii, to ensure symmetry.
                geometricRadii = np.sqrt(searchQuantity[0]*candRadii)
                condition = (quantRatio[:,0] >= self.muR) & \
                    (distances[indSort] <= geometricRadii*self.muS)
                for l in range(1,nQuantLen):
                    condition = condition & \
                        (quantRatio[:,l] >= self.muR)
                matchingVoids = np.where(condition)[0]
            else:
                candRadii = otherQuantities[sortedCandidates]
                # Geometric mean of radii, to ensure symmetry.
                geometricRadii = np.sqrt(searchQuantity*candRadii)
                matchingVoids = np.where((quantRatio >= self.muR) & \
                    (distances[indSort] <= geometricRadii*self.muS))[0]
        else:
            if not np.isscalar(searchQuantity):
                condition = np.ones(quantRatio.shape[0],dtype=bool)
                for l in range(0,nQuantLen):
                    condition = condition & \
                        (quantRatio[:,l] >= self.muR[l])
                matchingVoids = np.where(condition)[0]
            else:
                matchingVoids = np.where((quantRatio >= self.muR))[0]
        return matchingVoids
    # Function to process candidates for a match to a given void. We compute 
    # their radius ratio and distance ratio, to check whether they are 
    # within the thresholds required:
    def findAndProcessCandidates(self,nCatTest,centre,searchQuantity,\
            searchRadii,candidates=None,overlapForVoid=None):
        otherCentres = self.centresListShort[nCatTest]
        otherQuantities = self.quantityList[nCatTest]
        treeOther = self.treeList[nCatTest]
        # If we don't have candidates already, then we should find them:
        if candidates is None:
            candidates = treeOther.query_ball_point(\
                snapedit.wrap(centre,self.boxsize),searchRadii,workers=-1)
        # Check we have a sensible overlap map:
        if (overlapForVoid is None) and (self.sortMethod == "volumes"):
            raise Exception("overlap map required for volume sort method.")
        if len(candidates) > 0:
            # Sort indices:
            distances = np.sqrt(np.sum((\
                    otherCentres[candidates,:] - centre)**2,1))
            # Unsorted radius (or mass) ratios:
            quantRatio = self.getUnsortedRatios(candidates,searchQuantity,\
                otherQuantities)
            [quantRatio,sortedCandidates,indSort] = self.getSortedQuantRatio(\
                candidates,quantRatio,distances,overlapForVoid)
            # Get voids above the specified thresholds for these candidates:
            matchingVoids = self.getVoidsAboveThresholds(\
                quantRatio,distances,searchQuantity,otherQuantities,\
                sortedCandidates,indSort)
            selectCandidates = np.array(candidates)[matchingVoids]
            selectedQuantRatios = quantRatio[matchingVoids]
            selectedDistances = distances[matchingVoids]
            if len(matchingVoids) > 0:
                # Add the most probable - remembering the +1 offset for 
                # pynbody halo catalogue IDs:
                selectedMatches = sortedCandidates[matchingVoids[0]] + 1
            else:
                selectedMatches = -1
        else:
            selectedMatches = -1
            selectCandidates = np.array([])
            selectedQuantRatios = []
            selectedDistances = []
        return [selectedMatches,selectCandidates,selectedQuantRatios,\
            selectedDistances]
    def getSearchRadii(self,quantity1,quantity2):
        if self.mode == "fractional":
            radii1 = quantity1/self.muR
            radii2 = quantity2/self.muR
            if np.isscalar(radii1):
                searchRadii = radii1
            elif len(radii1.shape) > 1:
                searchRadii = radii1[:,0]
            else:
                searchRadii = radii1
        else:
            searchRadii = self.muS
        return searchRadii
    # Get the candidates in all other catalogues:
    def getCandidatesForVoidInAllCatalogues(self,centre,radius):
        newCatalogueRow = []
        if self.treeList is None:
            self.treeList = [scipy.spatial.cKDTree(\
                    snapedit.wrap(centres,self.boxsize),boxsize=self.boxsize) \
                    for centres in self.centresListShort]
        for ns in range(0,self.numCats):
            searchRadii = self.getSearchRadii(radius,self.quantityList[ns])
            [selectedMatches,selectCandidates,selectedQuantRatios,\
                selectedDistances] = self.findAndProcessCandidates(ns,centre,\
                    radius,searchRadii,candidates=None)
            newCatalogueRow.append(selectedMatches)
        return np.array(newCatalogueRow)
    # Code to iterate on the centres of a given void, so that we are less
    # dependent on matching to a particular void:
    def refineVoidCentres(self,voidMatches,ratiosm,distancesm):
        voidMatchesLast = np.array([-1 for k in range(0,self.numCats)])
        voidMatchesNew = voidMatches
        iterations = 0
        success = True
        # For convergence diagnostics:
        allCentres = []
        allRadii = []
        while not np.all(voidMatchesLast == voidMatchesNew):
            voidMatchesLast = voidMatchesNew
            # First, compute the mean centre of the voids in this set:
            meanCentres = self.getMeanCentreFromVoidMatches(voidMatchesNew)
            meanRadius = self.getMeanRadiusFromVoidMatches(voidMatchesNew)
            allCentres.append(meanCentres)
            allRadii.append(meanRadius)
            # Get all the voids within the thresholds from this centre:
            voidMatchesNew = self.getCandidatesForVoidInAllCatalogues(\
                meanCentres,meanRadius)
            if self.enforceExclusive:
                # Filter any already included voids so that they can't appear
                # in multiple voids:
                voidMatchesNew = self.removeAlreadyIncludedVoids(voidMatchesNew)
            # Check that we didn't run completely out of voids, as this will
            # make our centre meaningless.
            if np.all(voidMatchesNew < 0):
                break
            iterations += 1
            if iterations > self.iterMax:
                success = False
                break
        return [voidMatchesNew,ratiosm,distancesm,success,allCentres,allRadii]
    def checkForDuplicates(self,voidMatches):
        duplicates = np.any((np.array(self.finalCat) == voidMatches) & \
            (voidMatches > 0),1)
        return duplicates
    # Add an entry to the catalogue:
    def matchVoidToOtherCatalogues(self,nVoid,nCat,twoWayMatch):
        otherColumns = self.diffMap[nCat]
        oneWayMatches = self.oneWayMatchesAllCatalogues[nCat]
        oneWayMatchesOther = oneWayMatches[:,otherColumns]
        # Mark companions of this void as already found, to avoid duplication.
        # Additionally, store the candidates (candm), radius ratios (ratiosm) 
        # and distances to candidates (distancesm) of this void for output data:
        [candm,ratiosm,distancesm] = self.gatherCandidatesRatiosAndDistances(\
            nCat,nVoid)
        self.finalCandidates.append(candm)
        if self.NWayMatch:
            # Get the best candidates using the N-way matching code:
            [voidMatches,ratiosm,distancesm,numberOfLinks] = \
                self.applyNWayMatching(nVoid,nCat,oneWayMatches)
            combFrac = float(numberOfLinks/(self.numCats*(self.numCats-1)))
            success = True
        else:
            if self.refineCentres:
                [voidMatches,ratiosm,distancesm,success,allCentres,allRadii] = \
                    self.refineVoidCentres(oneWayMatches[nVoid],ratiosm,\
                        distancesm)
                duplicates = self.checkForDuplicates(voidMatches)
                if np.any(duplicates):
                    print("Found duplicates at " + str(np.where(duplicates)[0])
                    print("This match : " + str(voidMatches))
                    print("Duplicates: ")
                    for d in np.where(duplicates)[0]:
                        print(str(d) + " : " + str(self.finalCat[d]))
                    matchList = []
                    for ns,nv in zip(range(0,self.numCats),voidMatches):
                        if nv < 0:
                            matchList.append(False)
                        else:
                            matchList.append(self.alreadyMatched[ns,nv-1])
                    print("Already-Matched status:" + str(matchList))
                # Check the new entry is still unique:
                if success:
                    success = np.any(self.getUniqueEntriesInCatalogueRow(\
                        voidMatches)) # Skip the void if it's just 
                        # a duplicate of something that already existed, or
                        # a subset.
                if success and (np.sum(voidMatches > 0) > 1):
                    self.iteratedCentresList.append(np.array(allCentres))
                    self.iteratedRadiiList.append(np.array(allRadii))
            else:
                voidMatches = oneWayMatches[nVoid]
                success = True
            if not success:
                print("WARNING: void centre refining did not converge.")
                # Do nothing else - don't add a failed void to the catalogue!
            else:
                # Block the voids we have identified from appearing again:
                self.markCompanionsAsFound(nVoid,nCat,voidMatches)
                # Provided we found at least two voids, then add it to the 
                # catalogue
                # Compute the combinatoric fraction:
                twoWayMatchCounts = self.getTotalNumberOfTwoWayMatches(\
                    voidMatches)
                combFrac = twoWayMatchCounts/(self.numCats*(self.numCats-1))
        if (np.sum(voidMatches > 0) > 1) and success:
            catFrac = float(len(np.where(voidMatches > 0)[0])/self.numCats)
            self.finalCat.append(voidMatches)
            self.finalRatios.append(ratiosm)
            self.finalDistances.append(distancesm)
            self.finalCatFrac.append(catFrac)
            self.finalCombinatoricFrac.append(combFrac)
        return voidMatches




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


def overlapMap(cat1,cat2,volumes1,volumes2,checkFirst = False,verbose=False):
    overlap = np.zeros((len(cat1),len(cat2)))
    vol1 = np.array([np.sum(volumes1[halo['iord']]) for halo in cat1])
    vol2 = np.array([np.sum(volumes2[halo['iord']]) for halo in cat2])
    if checkFirst:
        for k in range(0,len(cat1)):
            for l in range(0,len(cat2)):
                if checkOverlap(cat1[k+1]['iord'],cat2[l+1]['iord']):
                    intersection = np.intersect1d(cat1[k+1]['iord'],\
                        cat2[l+1]['iord'])
                    overlap[k,l] = np.sum(\
                        volumes1[intersection])/np.sqrt(vol1[k]*vol2[l])
    else:
        for k in range(0,len(cat1)):
            for l in range(0,len(cat2)):
                intersection = np.intersect1d(cat1[k+1]['iord'],\
                    cat2[l+1]['iord'])
                if len(intersection) > 0:
                    overlap[k,l] = np.sum(\
                        volumes1[intersection])/np.sqrt(vol1[k]*vol2[l])
            if verbose:
                print(("%.3g" % (100*(k*len(cat1) + l + 1)/\
                    (len(cat1)*len(cat2)))) + "% complete")
    return overlap

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


