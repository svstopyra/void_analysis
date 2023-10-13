# Code related to building a void catalogue
import pynbody
import numpy as np
import scipy
from void_analysis import tools, snapedit, plot_utilities, stacking

class combinedCatalogue:
    # Class to store and compute a combined catalogue
    def __init__(self,snapList,snapListRev,muR,muS,rSphere,
                 ahProps=None,sortMethod="ratio",snapSortList=None,hrList=None,
                 verbose=False,r_min = 5,r_max = 30,massRange = None,
                 NWayMatch = False,additionalFilters = None,sortBy="mass",
                 refineCentres=False,max_index=None,enforceExclusive=False,
                 blockDuplicates=True,iterMax=100,matchType='distance',
                 crossMatchQuantity='radius',pynbodyThresh=0.5,twoWayOnly=True,
                 mode="fractional",overlapList = None):
        self.muR = muR # Radius ratio threshold
        self.muS = muS # Search distance ratio threshold
        self.rSphere = rSphere # Radius from the centre of the simulation box
            # out to which we build the catalogue
        # Load catalogue data:
        self.r_min = r_min # Minimum radius for voids used to build the 
            # catalogue
        self.r_max = r_max # Maximum radius for voids used to build the 
            # catalogue
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
        [_,_,self.boxsize,_,
            self.antihaloCentres,self.antihaloMasses,self.antihaloRadii,
            _,self.volumesList,_,self.deltaCentral,self.deltaAverage] = \
            loadCatalogueData(snapList,snapListRev,ahProps,sortMethod,
                              snapSortList,hrList,verbose=verbose)
        self.numCats = len(snapList) # Number of catalogues
        self.enforceExclusive = enforceExclusive
        self.blockDuplicates = blockDuplicates
        self.verbose = verbose
        if(type(snapList[0]) == str):
            self.snapNameList = snapList
        elif type(snapList[0]) == pynbody.snapshot.gadget.GadgetSnap:
            self.snapNameList = [snap.filename for snap in snapList]
        else:
            raise Exception("snapList must be list of snapshots or strings.")
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
        self.alreadyMatched = np.zeros((self.numCats,self.max_index),
                                       dtype=bool)
        # List of the other catalogues, for each catalogue:
        self.diffMap = [np.setdiff1d(np.arange(0,self.numCats),[k]) \
            for k in range(0,self.numCats)]
        self.iteratedCentresList = []
        self.iteratedRadiiList = []
        # Derived quantities:
        self.radiiList = None
        self.massList = None
        self.deltaCentralList = None
        self.deltaAverageList = None
        self.meanRadii = None
        self.meanMass = None
        self.meanDeltaCentral = None
        self.meanDeltaAverage = None
        self.sigmaRadii = None
        self.sigmaMass = None
        self.sigmaDeltaCentral = None
        self.sigmaDeltaAverage = None
        self.propertyDict = {"radii":self.radiiList,"mass":self.massList,\
            "deltaCentral":self.deltaCentralList,\
            "deltaAverage":self.deltaAverageList}
        self.meanDict = {"radii":self.meanRadii,"mass":self.meanMass,\
            "deltaCentral":self.meanDeltaCentral,\
            "deltaAverage":self.meanDeltaAverage}
        self.sigmaDict = {"radii":self.sigmaRadii,"mass":self.sigmaMass,\
            "deltaCentral":self.sigmaDeltaCentral,\
            "deltaAverage":self.sigmaDeltaAverage}
        # Construct matches:
        # Create lists of the quantity to match voids with (mass or radius),
        # chosen to match centresListShort:
        self.radiusListShort = self.getShortenedQuantity(self.antihaloRadii,\
            self.centralAntihalos)
        self.massListShort = self.getShortenedQuantity(self.antihaloMasses,\
            self.centralAntihalos)
        self.deltaCentralListShort = self.getShortenedQuantity(\
            self.deltaCentral,self.centralAntihalos)
        self.deltaAverageListShort = self.getShortenedQuantity(\
            self.deltaAverage,self.centralAntihalos)
        self.shortListDict = {"radii":self.radiusListShort,\
            "mass":self.massListShort,\
            "deltaCentral":self.deltaCentralListShort,\
            "deltaAverage":self.deltaAverageListShort}
        self.filter = None
        self.thresholds = None
        self.threshold_bins = None
        if self.crossMatchQuantity == 'radius':
            self.quantityList = self.radiusListShort
        elif self.crossMatchQuantity == 'mass':
            self.quantityList = self.massListShort
        else:
            raise Exception('Unrecognised cross-match quantity.')
    # Construct an anti-halo catalogue from reversed snapshots
    def constructAntihaloCatalogue(self):
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
        # Convert to arrays:
        self.finalCat = np.array(self.finalCat)
        self.finalCombinatoricFrac = np.array(self.finalCombinatoricFrac)
        self.finalCatFrac = np.array(self.finalCatFrac)
        return self.finalCat
    def loadCatalogue(catalogue):
        if type(catalogue) == str:
            catalogue = tools.loadPickle(catalogue)
        if type(catalogue) != np.array:
            raise Exception("Invalid catalogue type.")
        # Now check the catalogue is sensible:
        if len(catalogue.shape) != 2:
            raise Exception("Catalogue has invalid shape.")
        if catalogue.shape[1] != self.numCats:
            raise Exception("Number of catalogues is not valid.")
        if catalogue.dtype != int:
            raise Exception("Invalid array type for catalogue.")
        # Setup required variables:
        self.finalCat = catalogue
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
        centres1 = self.centresListShort[n1]
        centres2 = self.centresListShort[n2]
        quantity1 = self.quantityList[n1]
        #quantity2 = self.quantityList[n2]
        # Our procedure here is to get the closest anti-halo that lies within
        # the threshold:
        match = [-2] # Always include -2, for compatibility with pynbody output
        candidatesList = []
        ratioList = []
        distList = []
        # Fina candidates for all anti-halos:
        [searchRadii,searchOther] = self.getAllCandidatesFromTrees(n1,n2)
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
        if self.matchType == 'distance':
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
        allDistances = [] # List of distances from a void to all it's 
            # candidates
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
        # Each matrix has dimensions Nvoids_i x Ncat (Nvoids_i being the 
        # number of voids in catalogue i):
        oneWayMatchesAllCatalogues = \
            [np.array(matchArrayList[k]).transpose()[1:,:] \
            for k in range(0,self.numCats)]
        return [oneWayMatchesAllCatalogues,matchArrayList,allCandidates,\
            allRatios,allDistances]
    def computeShortCentresList(self):
        [self.centresListShort,self.centralAntihalos,self.sortedList,\
            self.ahCounts,self.max_index]  = computeShortCentresList(\
            self.snapNameList,self.antihaloCentres,\
            self.antihaloRadii,self.antihaloMasses,self.rSphere,\
            self.r_min,self.r_max,massRange = self.massRange,\
            additionalFilters = self.additionalFilters,\
            sortBy = self.sortBy,max_index = self.max_index)
        # Build a KD tree with the centres in centresListShort for efficient 
        # matching:
        self.treeList = [scipy.spatial.cKDTree(\
            snapedit.wrap(centres,self.boxsize),boxsize=self.boxsize) \
            for centres in self.centresListShort]
        # List of anti-halo numbers:
        self.ahNumbers = [np.array(self.centralAntihalos[l][0],dtype=int)\
            [self.sortedList[l]] for l in range(0,self.numCats)]
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
        centresArray = getAllQuantityiesFromVoidMatches(voidMatches,
                                                        centresList)
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
        # Number of search quantities (radius or mass) to process for this 
        # void:
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
                voidMatchesNew = self.removeAlreadyIncludedVoids(
                    voidMatchesNew)
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
    def blockAllVoidsFromInclusion(self,voidMatches):
        for ns, nv in zip(range(0,self.numCats),voidMatches):
            if nv > 0:
                self.alreadyMatched[ns,nv-1] = True
    # Add an entry to the catalogue:
    def matchVoidToOtherCatalogues(self,nVoid,nCat,twoWayMatch):
        otherColumns = self.diffMap[nCat]
        oneWayMatches = self.oneWayMatchesAllCatalogues[nCat]
        oneWayMatchesOther = oneWayMatches[:,otherColumns]
        # Mark companions of this void as already found, to avoid duplication.
        # Additionally, store the candidates (candm), radius ratios (ratiosm) 
        # and distances to candidates (distancesm) of this void for output 
        # data:
        [candm,ratiosm,distancesm] = self.gatherCandidatesRatiosAndDistances(
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
                [voidMatches,ratiosm,distancesm,success,allCentres,
                 allRadii] = self.refineVoidCentres(oneWayMatches[nVoid],
                                                    ratiosm,distancesm)
                if len(self.finalCat) > 0:
                    duplicates = self.checkForDuplicates(voidMatches)
                    if np.any(duplicates):
                        print("Found duplicates at " + \
                            str(np.where(duplicates)[0]))
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
                    # Important to prevent these voids being included again:
                    self.blockAllVoidsFromInclusion(voidMatches)
            else:
                voidMatches = oneWayMatches[nVoid]
                success = True
            if not success:
                print("WARNING: void centre refining did not converge.")
                # Do nothing else - don't add a failed void to the catalogue!
            else:
                # Block the voids we have identified from appearing again:
                if not self.refineCentres:
                    # Default method of blocking future inclusion. 
                    # refineCentres methods uses it's own approach, so
                    # no need to do this.
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
    # Get the catalogue fraction thresholds for each void:
    def getAllThresholds(self,percentiles,radBins,void_filter=False):
        if len(percentiles) != len(radBins) - 1:
            raise Exception("Bins not compatible with specified percentiles.")
        if self.meanRadii is None:
            [self.meanRadii,self.sigmaRadii] = self.getMeanProperty('radii')
        scaleFilter = [(self.meanRadii > radBins[k]) & \
            (self.meanRadii <= radBins[k+1]) \
            for k in range(0,len(radBins) - 1)]
        thresholds = np.zeros(self.meanRadii.shape)
        for filt, perc in zip(scaleFilter,percentiles):
            thresholds[filt] = perc
        return self.property_with_filter(thresholds,void_filter)
    # Compute mean properties for all voids, averaged over corresponding
    # anti-halos.
    def getMeanProperty(self,prop,stdError=True,\
            recompute=False,void_filter=False):
        if self.finalCat is None:
            raise Exception("Final catalogue has not yet been computed.")
        if type(prop) == str:
            if prop in self.propertyDict:
                propertyToProcess = self.propertyDict[prop]
                if propertyToProcess is None:
                    propertyToProcess = self.getAllProperties(prop)
            else:
                raise Exception("Property not implemented.")
        elif type(prop) == list:
            # Check it is sensible:
            if len(prop) != self.numCats:
                raise Exception("Property list has invalid length.")
            if (not np.all(np.array([len(x) for x in prop]) == self.ahCounts)):
                raise Exception("Property list lengths do not match " + \
                    "supplied catalogue lengths.")
            propertyToProcess = prop
        # Begin computing the mean property:
        meanProperty = np.zeros(len(self.finalCat))
        sigmaProperty = np.zeros(len(self.finalCat))
        for k in range(0,len(self.finalCat)):
            condition = (np.isfinite(propertyToProcess[k,:]))
            haveProperty = np.where(condition)[0]
            meanProperty[k] = np.mean(propertyToProcess[k,haveProperty])
            sigmaProperty[k] = np.std(propertyToProcess[k,haveProperty])
            if stdError:
                sigmaProperty[k] /= np.sqrt(len(haveProperty))
        self.meanDict[prop] = meanProperty
        self.sigmaDict[prop] = sigmaProperty
        return [self.property_with_filter(meanProperty,void_filter),
                self.property_with_filter(sigmaProperty,void_filter)]
    # Optionally apply a filter to the output catalogue so we only include
    # a specific subset of voids. Usually, this would be those which are
    # above a given significance threshold.
    def property_with_filter(self,prop,void_filter):
        if type(void_filter) is bool:
            if void_filter:
                if self.filter is None:
                    raise Exception("Void filter has not yet " +
                                    "been specified!")
                return prop[self.filter]
            else:
                return prop
        else:
            return prop[void_filter]
    def getAllProperties(self,prop,void_filter=False):
        if prop not in self.propertyDict:
            raise Exception("Property " + str(prop) + " not yet implemented.")
        if self.finalCat is None:
            raise Exception("Final catalogue has not yet been computed.")
        else:
            self.propertyDict[prop] = np.zeros(self.finalCat.shape,dtype=float)
            for k in range(0,len(self.finalCat)):
                for l in range(0,self.numCats):
                    if self.finalCat[k,l] > 0:
                        self.propertyDict[prop][k,l] = \
                            self.shortListDict[prop][l][self.finalCat[k,l]-1]
                    else:
                        self.propertyDict[prop][k,l] = np.nan
            return self.property_with_filter(
                self.propertyDict[prop],void_filter)
    def getSNRForVoidRealisations(self,snrAllCatsList,void_filter=False):
        if self.finalCat is None:
            raise Exception("Final catalogue is not yet computed.")
        snrCat = np.zeros(self.finalCat.shape)
        for ns in range(0,len(snrAllCatsList)):
            haveVoids = np.where(self.finalCat[:,ns] >= 0)[0]
            snrCat[haveVoids,ns] = snrAllCatsList[ns][self.ahNumbers[ns][\
                finalCat[haveVoids,ns]-1]]
        return self.property_with_filter(snrCat,void_filter)
    def getAllCentres(self,void_filter=False):
        all_centres = np.array([self.property_with_filter(
            self.getCentresInSample(ns), void_filter)\
            for ns in range(0,self.numCats)])
        return all_centres
    def getCentresInSample(self,ns,void_filter=False):
        centresListOut = np.zeros((len(self.finalCat),3),dtype=float)
        for k in range(0,len(self.finalCat)):
            if self.finalCat[k,ns] > 0:
                centresListOut[k,:] = self.centresListShort[ns]\
                    [self.finalCat[k,ns]-1]
            else:
                centresListOut[k,:] = np.nan
        return self.property_with_filter(centresListOut,void_filter)
    def getMeanCentres(self,void_filter=False):
        return self.property_with_filter(
            np.nanmean(self.getAllCentres(),0),void_filter)
    # Generate percentile thresholds (mostly used by random catalogues):
    def getThresholdsInBins(self,binEdges,percThresh,binVariable="radius"):
        if self.meanMass is None:
            [self.meanMass,self.sigmaMass] = self.getMeanProperty('mass')
        if self.meanRadii is None:
            [self.meanRadii,self.sigmaRadii] = self.getMeanProperty('radii')
        [inRadBins,noInRadBins] = \
            plot_utilities.binValues(self.meanRadii,binEdges)
        percentilesComb = []
        percentilesCat = []
        if binVariable == "mass":
            [inMassBins,noInMassBins] = \
                plot_utilities.binValues(self.meanMass,binEdges)
            selection = inMassBins[k]
            [selection,_] = \
                plot_utilities.binValues(self.meanMass,binEdges)
        elif binVariable == "radius":
            [selection,_] = \
                plot_utilities.binValues(self.meanRadii,binEdges)
        else:
            raise Exception("Unrecognised 'binVariable' value ")
        for k in range(0,len(binEdges)-1):
            if len(selection[k]) > 0:
                percentilesComb.append(np.percentile(\
                    self.finalCombinatoricFrac[selection[k]],percThresh))
                percentilesCat.append(np.percentile(\
                    self.finalCatFrac[selection[k]],percThresh))
            else:
                percentilesComb.append(0.0)
                percentilesCat.append(0.0)
        return [np.array(percentilesCat),np.array(percentilesComb)]
    # Set a filter to be applied to the catalogue, usually ot focus on 
    # the highest-significance voids.
    def set_filter(self,thresholds,threshold_bins,r_min=None,r_max=None,
                   r_sphere=None):
        if len(thresholds) != len(threshold_bins) - 1:
            raise Exception("Bins not compatible with specified percentiles.")
        self.thresholds = thresholds
        self.threshold_bins = threshold_bins
        all_thresholds = self.getAllThresholds(self.thresholds,
                                               self.threshold_bins)
        if r_min is None:
            r_min = self.r_min
        if r_max is None:
            r_max = self.r_max
        if r_sphere is None:
            r_sphere = self.rSphere
        # Construct a filter with this threshold:
        if self.meanRadii is None:
            [self.meanRadii,self.sigmaRadii] = self.getMeanProperty('radii')
        meanCentres  = self.getMeanCentres()
        distances = np.sqrt(np.sum(meanCentres**2,1))
        self.filter = ( (self.meanRadii > r_min) & (self.meanRadii <= r_max) 
                       & (distances < r_sphere) 
                       & (self.finalCatFrac > all_thresholds) )
    # As set_filter, but computes the thresholds directly from another
    # catalogue (usually a random catalogue)
    def set_filter_from_random_catalogue(self,random_catalogue,threshold_bins,
                                         percentile=99,**kwargs):
        thresholds = random_catalogue.getThresholdsInBins(threshold_bins,
                                                          percentile)
        self.set_filter(thresholds,threshold_bins,**kwargs)
    def get_alpha_shapes(self,snapList,snapListRev,antihaloCatalogueList=None,
                         ahProps = None,snapsortList=None):
        if reCentreSnaps:
            for snap in snapList:
                tools.remapBORGSimulation(snap,swapXZ=False,reverse=True)
                snap.recentred = True
        boxsize = snapList[0].properties['boxsize'].ratio("Mpc a h**-1")
        if antihaloCatalogueList is None:
            antihaloCatalogueList = [snap.halos() for snap in snapListRev]
        if ahProps is None:
            ahProps = [tools.loadPickle(snap.filename + ".AHproperties.p") \
                for snap in snapList]
        if snapsortList is None:
            snapsortList = [np.argsort(snap['iord']) \
                for snap in snapList]
        radSortCentral = [\
            np.flip(np.argsort(
            self.antihaloRadii[k][self.centralAntihalos[k][0]])) \
            for k in range(0,len(self.centralAntihalos))]
        largeAntihalos = [np.array(self.centralAntihalos[ns][0],dtype=int)[\
                radSortCentral[ns]] for ns in range(0,len(snapList))]
        if verbose:
            print("Computing alpha shapes...")
        # From here, we have to combined the positions of ALL voids:
        positionLists = [] # Positions of all particles in all voids
        centralAntihaloRadii = [\
                self.antihaloRadii[k][self.centralAntihalos[k][0]] \
                for k in range(0,len(self.centralAntihalos))]
        sortedList = [np.flip(np.argsort(centralAntihaloRadii[ns])) \
            for ns in range(0,self.numCats)]
        fullListAll = [np.array(self.centralAntihalos[ns][0])[sortedList[ns]] \
            for ns in range(0,self.numCats)]
        alpha_shapes = []
        ahMWPos = []
        for k in range(0,self.finalCat.shape[0]):
            allPosXYZ = np.full((0,3),0)
            for ns in range(0,self.finalCat.shape[1]):
                # Select the correct anti-halo
                fullList = fullListAll[ns]
                listPosition = self.finalCat[k,ns]-1
                if listPosition >= 0:
                    # Only include anti-halos which we have representatives for
                    # in a given catalogue
                    ahNumber = fullList[listPosition]
                    posXYZ = snapedit.unwrap(
                        snapList[ns]['pos'][snapsortList[ns][\
                        antihaloCatalogueList[ns][\
                        largeAntihalos[ns][ahNumber]+1]['iord']],:],boxsize)
                    allPosXYZ = np.vstack((allPosXYZ,posXYZ))
            posMW = plot_utilities.computeMollweidePositions(allPosXYZ)
            ahMWPos.append(posMW)
            alpha_shapes.append(alphashape.alphashape(
                    np.array([posMW[0],posMW[1]]).T,alphaVal))
        return [ahMWPos,alpha_shapes]

def loadSnapshots(snapList):
    if type(snapList[0]) == str:
        snapshotsList = [pynbody.load(snapname) for snapname in snapList]
    elif type(snapList[0]) == pynbody.snapshot.gadget.GadgetSnap:
        snapshotsList = snapList
    else:
        raise Exception("Snapshot type not supported.")
    return snapshotsList

# Load simulations and catalogue data so that we can combine them. If these
# are already loaded, this function won't reload them.
def loadCatalogueData(snapList,snapListRev,ahProps,sortMethod,snapSortList,\
        hrList,verbose=False):
    snapshotsList = loadSnapshots(snapList)
    snapshotsListRev = loadSnapshots(snapListRev)
    # Load centres so that we can filter to the constrained region:
    boxsize = snapshotsList[0].properties['boxsize'].ratio("Mpc a h**-1")
    if verbose:
        print("Extracting anti-halo properties...")
    if ahProps is None:
        ahProps = [tools.loadPickle(snap.filename + ".AHproperties.p") \
            for snap in snapshotsList]
    antihaloCentres = [tools.remapAntiHaloCentre(props[5],boxsize) \
        for props in ahProps]
    antihaloMasses = [props[3] for props in ahProps]
    antihaloRadii = [props[7] for props in ahProps]
    deltaCentral = [props[11] for props in ahProps]
    deltaAverage = [props[12] for props in ahProps]
    if sortMethod == "volumes":
        if snapSortList is None:
            snapSortList = [np.argsort(snap['iord']) for snap in snapshotsList]
        volumesList = [ahProps[k][4][snapSortList[k]] \
            for k in range(0,len(ahProps))]
    else:
        volumesList = [None for k in range(0,len(ahProps))]
    # Load anti-halo catalogues:
    if hrList is None:
        hrList = [snap.halos() for snap in snapshotsListRev]
    return [snapshotsList,snapshotsListRev,boxsize,ahProps,antihaloCentres,\
        antihaloMasses,antihaloRadii,snapSortList,volumesList,hrList,\
        deltaCentral,deltaAverage]


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
def computeShortCentresList(snapNumList,antihaloCentres,antihaloRadii,
        antihaloMasses,rSphere,rMin,rMax,massRange=None,additionalFilters=None,
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


# Other utility functions:

def getSplitVoids(catRef,catTest,nVRef):
    nCats = catRef.shape[1]
    locator = [np.where((catTest[:,k] == catRef[nVRef][k]) & \
        (catTest[:,k] != -1)) for k in range(0,nCats)]
    splitEntries = np.unique(np.hstack(locator))
    return splitEntries

def getSplitList(catRef,catTest):
    splitList = []
    for nVRef in range(0,len(catRef)):
        splitList.append(getSplitVoids(catRef,catTest,nVRef))
    return splitList

# Converts a short catalogue (listing halos only in the central region)
# into a long catalogue (listing all halos in the simulation box)
def shortCatalogueToLongCatalogue(catalogue,centralAntihalos,sortedList):
    if catalogue.shape[1] != len(centralAntihalos):
        raise Exception("Incompatible catalogue and centralAntihalos list.")
    longCatalogue = -np.ones(catalogue.shape,dtype=int)
    nCats = len(centralAntihalos)
    for ns in range(0,nCats):
        haveCandidates = np.where(catalogue[:,ns] >= 0)[0]
        ahNumbersList = np.array(centralAntihalos[ns][0])[sortedList[ns]]
        longCatalogue[haveCandidates,ns] = ahNumbersList\
            [catalogue[haveCandidates,ns] - 1] + 1
    return longCatalogue


# Should be the inverse, if everything worked out:
def longCatalogueToShortCatalogue(longCatalogue,centralAntihalos,sortedList):
    if longCatalogue.shape[1] != len(centralAntihalos):
        raise Exception("Incompatible catalogue and centralAntihalos list.")
    catalogue = -np.ones(longCatalogue.shape,dtype=int)
    nCats = len(centralAntihalos)
    for ns in range(0,nCats):
        haveCandidates = (longCatalogue[:,ns] >= 0)
        ahNumbersList = np.array(centralAntihalos[ns][0])[sortedList[ns]]
        inShortList = np.isin(longCatalogue[:,ns]-1,ahNumbersList)
        canInvert = haveCandidates & inShortList
        search = np.where(np.in1d(ahNumbersList,longCatalogue[:,ns]-1))[0]
        catalogue[canInvert,ns] = search + 1
    return catalogue



def lowestOrNothing(x):
    if len(x) > 0:
        return np.min(x)
    else:
        return np.nan


# Count the number of voids that two final catalogue entries have in common:
def getNumVoidsInCommon(voidRef,voidTest):
    return np.sum(voidRef == voidTest)

# Figure out how similar too voids are:
def compareVoids(voidRef,voidTest):
    matching = np.where((voidTest == voidRef) & (voidRef > -1))[0]
    notMatching = np.where((voidTest != voidRef) & (voidTest > -1))[0]
    failed = np.where((voidTest == -1) & (voidRef > -1))[0]
    return [matching,notMatching,failed]

# Compare voids from two different catalogues:
def analyseSplit(catRef,catTest,nVRef,nVTest):
    return compareVoids(catRef[nVRef],catTest[nVTest])

# Count the number of conditions being applied:
def get_number_of_conditions(conditionList):
    if type(conditionList) is list:
        numCond = len(conditionList)
    elif type(conditionList) is np.ndarray:
        if len(conditionList.shape) == 1:
            numCond = 1
        elif len(conditionList.shape) > 1:
            numCond = conditionList.shape[1]
        else:
            raise Exception("Invalid conditions list.")
    elif conditionList is None:
        numCond = 0
    else:
        raise Exception("Invalid conditions list.")
    return numCond


# Class to store information about a void stack:
class ProfileStack:
    def __init__(self,centre_list,snap_list,props_list,r_sphere,
                 r_eff_bin_edges,tree_list=None,seed=1000,start=0,end=-1,
                 conditioning_quantity=None,
                 conditioning_quantity_to_match=None,condition_bin_edges=None,
                 combine_random_regions=False,replace=False,r_min=None,
                 r_max=None,compute_pair_counts = True,max_sampling=None,
                 pair_counts=None):
        self.centre_list = centre_list
        self.snap_list = snap_list
        if tree_list is None:
            # Generate the tree list from scratch (expensive in both cputime 
            # and memory, which is why we give the option to supply a 
            # pre-generated tree list
            self.tree_list = [scipy.spatial.cKDTree(snap['pos'],\
                boxsize=boxsize) for snap in snap_list]
        else:
            self.tree_list = tree_list
        self.ah_centres_list = [props[5] for props in props_list]
        self.antihalo_radii_list = [props[7] for props in props_list]
        self.boxsize = snap_list[0].properties['boxsize'].ratio("Mpc a h**-1")
        self.seed = seed
        self.r_sphere = r_sphere
        self.r_eff_bin_edges = r_eff_bin_edges
        self.r_min = r_min
        self.r_max = r_max
        self.start=0
        self.max_sampling = max_sampling
        if end == -1:
            self.end = len(snap_list)
        else:
            self.end = end
        self.conditioning_quantity = conditioning_quantity
        self.conditioning_quantity_to_match = conditioning_quantity_to_match
        self.condition_bin_edges = condition_bin_edges
        self.combine_random_regions = combine_random_regions
        self.replace = replace
        self.compute_pair_counts = compute_pair_counts
        self.pair_counts = pair_counts
        self.sampling_MCMC = None
        self.sampling_rand = None
        # Verify number of conditions matches the bins specified:
        self.num_cond = get_number_of_conditions(
            self.conditioning_quantity_to_match)
        if self.condition_bin_edges is not None:
            if len(self.condition_bin_edges) != self.num_cond:
                raise Exception("List of bins must match list of " + 
                                "condition data")
        # Verify the same for the randoms:
        if self.conditioning_quantity is not None:
            self.num_cond_rand = get_number_of_conditions(\
                self.conditioning_quantity[0])
            if self.num_cond_rand != self.num_cond:
                raise Exception("Condition list for randoms does " + 
                                "not match condition list for MCMC.")
    def get_number_of_radial_bins(self):
        if self.pair_counts is not None:
            return self.pair_counts[0].shape[1]
        else:
            return len(self.r_eff_bin_edges) - 1
    def get_all_condition_variables_in_range(self,condition_variables,
                                             void_radii):
        in_all_ranges = np.ones(len(condition_variables),dtype = bool)
        num_cond = get_number_of_conditions(condition_variables)
        for n in range(0,num_cond):
            in_all_ranges = (
                in_all_ranges 
                & (condition_variables[:,n] 
                   >= self.condition_bin_edges[n][0])
                & (condition_variables[:,n]
                   < self.condition_bin_edges[n][-1]))
        if self.r_min is not None:
            in_all_ranges = in_all_ranges & (void_radii > self.r_min)
        if self.r_max is not None:
            in_all_ranges = in_all_ranges & (void_radii < self.r_max)
        return in_all_ranges
    def get_sampling_ratio(self,condition_variable = None,void_radii=None):
        if not self.combine_random_regions:
            min_ratio = 1
        else:
            if condition_variable is None:
                condition_variable = self.central_condition_variable_all
            if void_radii is None:
                void_radii = self.central_radii_all
            in_all_ranges = self.get_all_condition_variables_in_range(
                condition_variable,void_radii)
            # Having verified that the input is sane, now bin everything:
            [self.sampling_MCMC,edges] = np.histogramdd(\
                self.conditioning_quantity_to_match,
                bins = self.condition_bin_edges)
            self.sampling_MCMC_lin = np.array(self.sampling_MCMC.flatten(),
                                              dtype=int)
            [self.sampling_rand,edges] = np.histogramdd(\
                condition_variable[in_all_ranges],
                bins = self.condition_bin_edges)
            self.sampling_rand_lin = np.array(self.sampling_rand.flatten(),
                                              dtype=int)
            # Figure out how many times we can sample the random set, and 
            # retain the same distribution as the MCMC set. This is defined 
            # as the lowest ratio between the counts in MCMC bins and in 
            # random bins.
            nz_MCMC = np.where(self.sampling_MCMC_lin > 0)
            rat = np.zeros(self.sampling_MCMC_lin.shape,dtype=int)
            rat[nz_MCMC] = \
                self.sampling_rand_lin[nz_MCMC]/self.sampling_MCMC_lin[nz_MCMC]
            if len(rat[rat > 0]) > 0:
                min_ratio = np.min(rat[rat > 0])
            else:
                min_ratio = 1
        min_ratio = np.min([min_ratio,self.max_sampling])
        return min_ratio
    # Select random catalogue voids that match the distribution of properties
    # in the MCMC catalogue.
    def select_conditioned_random_voids(self,conditioning_variable,void_radii,
                                        min_ratio = None):
        num_cond = get_number_of_conditions(
            self.conditioning_quantity_to_match)
        if len(self.condition_bin_edges) != num_cond:
            raise Exception("List of bins must match list of " + \
                "condition data")
        # Verify the same for the randoms:
        num_cond_rand = get_number_of_conditions(conditioning_variable)
        if type(conditioning_variable) is list:
            conditioning_variable = np.array(conditioning_variable)
        if type(self.conditioning_quantity_to_match) is list:
            self.conditioning_quantity_to_match = \
                np.array(self.conditioning_quantity_to_match)
        if num_cond_rand != num_cond:
            raise Exception("Condition list for randoms does " + \
                "not match condition list for MCMC.")
        # Having verified that the input is sane, now bin everything:
        [self.sampling_MCMC,edges] = np.histogramdd(\
            self.conditioning_quantity_to_match,\
            bins = self.condition_bin_edges)
        self.sampling_MCMC_lin = np.array(self.sampling_MCMC.flatten(),
                                          dtype=int)
        [self.sampling_rand,edges] = np.histogramdd(conditioning_variable,\
            bins = self.condition_bin_edges)
        self.sampling_rand_lin = np.array(self.sampling_rand.flatten(),
                                          dtype=int)
        # Fina all voids in the correct radius range, which lie within some
        # of the bins. These are the voids we can sample from:
        in_all_ranges = self.get_all_condition_variables_in_range(
            conditioning_variable,void_radii)
        in_all_ranges_ind = np.where(in_all_ranges)[0]
        # Minimum ratio between available random-catalogue samples, and MCMC
        # catalogue samples. Use this to determine how many times we can 
        # sample the voids while keeping the same distribution as the MCMC
        # catalogue:
        if min_ratio is None:
            min_ratio = self.get_sampling_ratio()
        indices_rand = []
        for n in range(0,num_cond):
            indices_rand.append(np.digitize(\
                conditioning_variable[in_all_ranges,n],
                self.condition_bin_edges[n])-1)
        indices_rand = np.array(indices_rand)
        dims = [len(x) - 1 for x in self.condition_bin_edges]
        num_bins_tot = np.prod(dims)
        linearIndices = np.ravel_multi_index(indices_rand,tuple(dims))
        # Now sample these to try and match the MCMC conditions:
        selection = []
        np.random.seed(self.seed)
        self.region_index = np.zeros(0,dtype=int)
        samples_taken = np.zeros(num_bins_tot,dtype=int)
        available_samples = np.zeros(num_bins_tot,dtype=int)
        for k in range(0,num_bins_tot):
            this_index = np.where((linearIndices == k))[0]
            available_samples[k] = len(this_index)
            if self.replace:
                if len(this_index) > 0:
                    num_samples_to_take = min_ratio*self.sampling_MCMC_lin[k]
                else:
                    num_samples_to_take = 0
            else:
                num_samples_to_take = np.min(
                    [min_ratio*self.sampling_MCMC_lin[k],len(this_index)])
            selection.append(
                np.random.choice(this_index,num_samples_to_take,\
                                 replace=self.replace))
            samples_taken[k] = num_samples_to_take
            if self.combine_random_regions:
                # Store which of the fake regions each sample belongs to:
                self.region_index = np.hstack((
                    self.region_index,np.array(
                        np.arange(0,num_samples_to_take,1) 
                        / self.sampling_MCMC_lin[k],
                        dtype=int)))
        select_array = np.hstack(selection)
        return in_all_ranges_ind[select_array]
    # Get arrays which store the variables for each void we wish to sample
    # from:
    def get_all_condition_variables(self):
        self.central_condition_variable_all = np.zeros((0,self.num_cond))
        self.central_centres_all = np.zeros((0,3))
        self.central_radii_all = np.zeros(0)
        self.sample_indices = np.zeros(0,dtype=int)
        self.void_indices = np.zeros(0,dtype=int)
        # Iterate over all simulations:
        for ns in range(self.start,self.end):
            # Iterate over all regions within each simulation:
            for centre, count in zip(self.centre_list[ns],\
                                     range(0,len(self.centre_list[ns]))):
                # Get all voids within this region:
                central_antihalos = tools.getAntiHalosInSphere(\
                    self.ah_centres_list[ns],self.r_sphere,origin=centre,\
                    boxsize=self.boxsize)[1]
                num_central = np.sum(central_antihalos)
                # Void radii:
                central_radii = self.antihalo_radii_list[ns][central_antihalos]
                # Void centres:
                central_centres = self.ah_centres_list[ns][central_antihalos]
                # Void indices:
                central_indices = np.where(central_antihalos)[0]
                # Void condition variables:
                if self.num_cond == 1:
                    central_condition_variable = np.array(\
                        self.conditioning_quantity[ns][central_antihalos])
                    central_condition_variable = \
                        central_condition_variable.reshape(\
                        (num_central,self.num_cond))
                else:
                    central_condition_variable = np.array(\
                        self.conditioning_quantity[ns][central_antihalos,:])
                # Stack all variables into one array for each variable:
                self.central_condition_variable_all = np.vstack(
                    (self.central_condition_variable_all,
                     central_condition_variable))
                self.central_centres_all = np.vstack(
                    (self.central_centres_all,central_centres))
                self.central_radii_all = np.hstack(
                    (self.central_radii_all,central_radii))
                self.sample_indices = np.hstack(
                    (self.sample_indices,
                     ns*np.ones(central_radii.shape,dtype=int)))
                self.void_indices = np.hstack(
                    (self.void_indices,central_indices))
    # Computes a pooled array of variables:
    def get_pooled_variable(self,variable_list):
        num_dims = len(variable_list[0].shape)
        if num_dims == 1:
            pooled_variable = np.zeros(0,dtype=variable_list[0].dtype)
        elif num_dims == 2:
            pooled_variable = np.zeros(
                (0,variable_list[0].shape[1]),dtype=variable_list[0].dtype)
        else:
            raise Exception("Not implemented for num_dims = " + str(num_dims))
        for ns in range(self.start,self.end):
            for centre, count in zip(self.centre_list[ns],\
                                     range(0,len(self.centre_list[ns]))):
                central_antihalos = tools.getAntiHalosInSphere(\
                    self.ah_centres_list[ns],self.r_sphere,origin=centre,\
                    boxsize=self.boxsize)[1]
                num_central = np.sum(central_antihalos)
                central_variables = variable_list[ns][central_antihalos]
                if num_dims == 1:
                    pooled_variable = np.hstack(
                        (pooled_variable,central_variables))
                else:
                    pooled_variable = np.vstack(
                        (pooled_variable,central_variables))
        return pooled_variable
    def get_volumes_of_radial_bins(self,central_radii,select_array):
        volumes = 4*np.pi*(self.r_eff_bin_edges[1:]**3 - \
            self.r_eff_bin_edges[0:-1]**3)/3
        void_radii = central_radii[select_array]
        volumes_list = np.outer(void_radii**3,volumes)
        return volumes_list
    def restore_from_dictionary(self,dictionary):
        self.all_pairs = dictionary['pairs']
        self.all_volumes = dictionary['volumes']
        self.all_selections = dictionary['selections']
        self.all_conditions = dictionary['conditions']
        self.all_selected_conditions = dictionary['selectedConditions']
        if self.combine_random_regions and \
                (self.conditioning_quantity_to_match is not None):
            self.get_all_condition_variables()
            self.num_voids_total = len(self.central_radii_all)
    # Sampling a set of random catalogues while matching to an MCMC catalogue:
    def get_random_catalogue_pair_counts(self):
        # Get pair counts in similar-density regions:
        self.all_pairs = []
        self.all_volumes = []
        self.all_selections = []
        self.all_conditions = []
        self.all_selected_conditions = []
        self.all_radii = []
        self.all_indices = []
        self.all_centres = []
        if self.combine_random_regions:
            if self.conditioning_quantity_to_match is not None:
                self.get_all_condition_variables()
            self.num_voids_total = len(self.central_radii_all)
            self.min_ratio = self.get_sampling_ratio()
            if self.conditioning_quantity_to_match is not None:
                select_array = self.select_conditioned_random_voids(\
                    self.central_condition_variable_all,\
                    self.central_radii_all)
            else:
                select_array = np.range(0,self.num_voids_total)
            self.all_pairs = [
                np.zeros((0,self.get_number_of_radial_bins()),dtype=int) 
                for k in range(0,self.min_ratio)]
            self.all_volumes = [
                np.zeros((0,self.get_number_of_radial_bins()),dtype=int) 
                for k in range(0,self.min_ratio)]
            self.all_selections = [
                np.zeros((0),dtype=int) 
                for k in range(0,self.min_ratio)]
            self.all_selected_conditions = [
                np.zeros((0,self.num_cond),dtype=int) 
                for k in range(0,self.min_ratio)]
            self.region_counts = np.zeros(self.min_ratio,dtype=int)
            for k in range(0,self.min_ratio):
                for ns in range(self.start,self.end):
                    snap_loaded = self.snap_list[ns]
                    tree = self.tree_list[ns]
                    num_selected = len(select_array)
                    k_filter = np.where(self.region_index == k)[0]
                    self.region_counts[k] = len(k_filter)
                    ns_select_array = select_array[k_filter][\
                        self.sample_indices[select_array][k_filter] == ns]
                    if self.compute_pair_counts:
                        if self.pair_counts is None:
                            [n_pairs_list,volumes_list] = \
                                stacking.getPairCounts(
                                    self.central_centres_all[ns_select_array],
                                    self.central_radii_all[ns_select_array],
                                    snap_loaded,self.r_eff_bin_edges,tree=tree,
                                    method='poisson',vorVolumes=None)
                        else:
                            n_pairs_list = self.pair_counts[ns][\
                                    self.void_indices[ns_select_array],:]
                            volumes_list = \
                                self.get_volumes_of_radial_bins(\
                                    self.central_radii_all,ns_select_array)
                        self.all_pairs[k] = np.vstack((self.all_pairs[k],
                                                     n_pairs_list))
                        self.all_volumes[k] = np.vstack((self.all_volumes[k],
                                                     volumes_list))
                    else:
                        volumes = 4*np.pi*(self.r_eff_bin_edges[1:]**3 - \
                            self.r_eff_bin_edges[0:-1]**3)/3
                        void_radii = self.central_radii_all[ns_select_array]
                        volumes_list = np.outer(void_radii**3,volumes)
                        self.all_volumes[k] = np.vstack((self.all_volumes[k],
                                                     volumes_list))
                    self.all_selected_conditions[k] = np.vstack(
                        (self.all_selected_conditions[k],
                         self.central_condition_variable_all[
                         ns_select_array,:]))
                    self.all_selections[k] = np.hstack((self.all_selections[k],
                                                       ns_select_array))
                    self.all_radii.append(
                        self.central_radii_all[ns_select_array])
                    self.all_centres.append(
                        self.central_centres_all[ns_select_array])
            self.all_conditions.append(self.central_condition_variable_all)
        else:
            lengths_array = np.zeros(0,dtype=int)
            for ns in range(self.start,self.end):
                snap_loaded = self.snap_list[ns]
                tree = self.tree_list[ns]
                for centre, count in zip(self.centre_list[ns],\
                    range(0,len(self.centre_list[ns]))):
                    # Get anti-halos:
                    central_antihalos = tools.getAntiHalosInSphere(\
                        self.ah_centres_list[ns],self.r_sphere,origin=centre,\
                        boxsize=self.boxsize)[1]
                    central_indices = np.where(central_antihalos)[0]
                    # Get radii and randomly select voids with the same
                    # radius distribution as the combined catalogue:
                    central_radii = \
                        self.antihalo_radii_list[ns][central_antihalos]
                    central_centres = \
                        self.ah_centres_list[ns][central_antihalos]
                    if self.conditioning_quantity_to_match is not None:
                        if self.num_cond == 1:
                            num_cond_variables = \
                                len(np.where(central_antihalos)[0])
                            central_condition_variable = \
                                self.conditioning_quantity[ns]\
                                [central_antihalos].reshape(
                                    num_cond_variables,1)
                        else:
                            central_condition_variable = \
                                self.conditioning_quantity[ns][\
                                central_antihalos,:]
                        self.all_conditions.append(central_condition_variable)
                        select_array = self.select_conditioned_random_voids(\
                            central_condition_variable,central_radii)
                        self.all_selected_conditions.append(\
                            central_condition_variable[select_array,:])
                    else:
                        condition = np.ones(central_radii.shape,dtype=bool)
                        # Bounding with radius cuts:
                        if self.r_min is not None:
                            condition = (condition 
                                         & (central_radii > self.r_min))
                        if self.r_max is not None:
                            condition = (condition 
                                         & (central_radii < self.r_max))
                        select_array = np.where(condition)[0]
                    self.all_radii.append(central_radii[select_array])
                    self.all_centres.append(central_centres[select_array])
                    self.all_indices.append(central_indices[select_array])
                    # Now get pair counts around these voids:
                    lengths_array = np.hstack(
                        (lengths_array,np.array([len(select_array)])))
                    if self.compute_pair_counts:
                        if self.pair_counts is None:
                            # Regenerate pair counts from scratch:
                            [n_pairs_list,volumes_list] = \
                                stacking.getPairCounts(
                                    central_centres[select_array],
                                    central_radii[select_array],snap_loaded,
                                    self.r_eff_bin_edges,tree=tree,
                                    method='poisson',vorVolumes=None)
                        else:
                            # Use a pre-computed list:
                            n_pairs_list = \
                                self.pair_counts[ns][\
                                    central_indices[select_array],:]
                            volumes_list = \
                                self.get_volumes_of_radial_bins(
                                    central_radii,select_array)
                    else:
                        volumes_list = \
                            self.get_volumes_of_radial_bins(central_radii,
                                    select_array)
                    print("Done centre " + str(count+1) + " of " + \
                        str(len(self.centre_list[ns])))
                    if self.compute_pair_counts:
                        self.all_pairs.append(n_pairs_list)
                    self.all_volumes.append(volumes_list)
                    self.all_selections.append(central_indices[select_array])
                print("Done sample " + str(ns + 1) + ".")
        return {'pairs':self.all_pairs,'volumes':self.all_volumes,
            'selections':self.all_selections,'conditions':self.all_conditions,
            'selected_conditions':self.all_selected_conditions,
            'radii':self.all_radii,'indices':self.all_indices,
            'centres':self.all_centres}






