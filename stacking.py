# Stacking functions
import Corrfunc
import scipy.spatial
import numpy as np
from . import snapedit, plot_utilities
import multiprocessing as mp
from scipy.optimize import curve_fit
thread_count = mp.cpu_count()
import scipy.integrate as integrate
from scipy.stats import kstest

# Weighted mean of a variable:
def weightedMean(xi,wi,biasFactor = 0,axis=None):
	return (np.sum(xi*wi,axis=axis) + biasFactor)/np.sum(wi,axis=axis)

# Weighted variance:
def weightedVariance(xi,wi,biasFactor = 0,axis=None):
	xibar = weightedMean(xi,wi,biasFactor=biasFactor,axis=axis)
	M = np.count_nonzero(wi,axis=axis)
	return np.sum(wi*(xi-xibar)**2,axis=axis)/(((M-1)/M)*np.sum(wi,axis=axis))

# Compute correlation function for a given set of points (cross correlation if a second is specified)
def simulationCorrelation(rBins,boxsize,data1,data2=None,nThreads = 1,weights1=None,weights2 = None):
	
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
		DD1 = Corrfunc.theory.DD(1,nThreads,rBins,X1,Y1,Z1,periodic=True,boxsize=boxsize,weights1=weights1)
		DR1 = Corrfunc.theory.DD(0,nThreads,rBins,X1,Y1,Z1,periodic=True,boxsize=boxsize,X2 = X1rand,Y2=Y1rand,Z2=Z1rand,weights1=weights1)
		RR1 = Corrfunc.theory.DD(1,nThreads,rBins,X1rand,Y1rand,Z1rand,periodic=True,boxsize=boxsize,weights1=weights1)
		xiEst = Corrfunc.utils.convert_3d_counts_to_cf(N1,N1,rand_N1,rand_N1,DD1,DR1,DR1,RR1)
	else:
		# Cross correlation:
		D1D2 = Corrfunc.theory.DD(0,nThreads,rBins,X1,Y1,Z1,X2=X2,Y2=Y2,Z2=Z2,periodic=True,boxsize=boxsize,weights1=weights1,weights2=weights2)
		D1R2 = Corrfunc.theory.DD(0,nThreads,rBins,X1,Y1,Z1,periodic=True,boxsize=boxsize,X2 = X2rand,Y2=Y2rand,Z2=Z2rand,weights1=weights1,weights2=weights2)
		D2R1 = Corrfunc.theory.DD(0,nThreads,rBins,X2,Y2,Z2,periodic=True,boxsize=boxsize,X2 = X1rand,Y2=Y1rand,Z2=Z1rand,weights1=weights1,weights2=weights2)
		R1R2 = Corrfunc.theory.DD(0,nThreads,rBins,X1rand,Y1rand,Z1rand,X2 = X2rand,Y2=Y2rand,Z2=Z2rand,periodic=True,boxsize=boxsize,weights1=weights1,weights2=weights2)
		xiEst = Corrfunc.utils.convert_3d_counts_to_cf(N1,N2,rand_N1,rand_N2,D1D2,D1R2,D2R1,R1R2)
	return xiEst
	



# Cross correlation of two sets of points:
def getCrossCorrelations(ahCentres,voidCentres,ahRadii,voidRadii,rMin = 0,rMax = np.inf,rRange = np.linspace(0.1,10,101),nThreads=thread_count,boxsize = 200.0):
	rFilter1 = np.where((ahRadii > rMin) & (ahRadii < rMax))[0]
	rFilter2 = np.where((voidRadii > rMin) & (voidRadii < rMax))[0]
	ahPos = ahCentres[rFilter1,:]
	vdPos = voidCentres[rFilter2,:]
	xiAM = simulationCorrelation(rRange,boxsize,ahPos,data2=snap['pos'],nThreads=nThreads,weights2 = snap['mass'])
	xiVM = simulationCorrelation(rRange,boxsize,vdPos,data2=snap['pos'],nThreads=nThreads,weights2 = snap['mass'])
	xiAV = simulationCorrelation(rRange,boxsize,ahPos,data2=vdPos,nThreads=nThreads)
	return [xiAM,xiVM,xiAV]

# Auto correlation of a set of points:
def getAutoCorrelations(ahCentres,voidCentres,ahRadii,voidRadii,rMin = 0,rMax = np.inf,rRange = np.linspace(0.1,10,101),nThreads=thread_count,boxsize = 200.0):
	rFilter1 = np.where((ahRadii > rMin) & (ahRadii < rMax))[0]
	rFilter2 = np.where((voidRadii > rMin) & (voidRadii < rMax))[0]
	ahPos = ahCentres[rFilter1,:]
	vdPos = voidCentres[rFilter2,:]
	xiAA = simulationCorrelation(rRange,boxsize,ahPos,nThreads=nThreads)
	xiVV = simulationCorrelation(rRange,boxsize,vdPos,nThreads=nThreads)
	return [xiAA,xiVV]

def getPairCounts(voidCentres,voidRadii,snap,rBins,nThreads=thread_count,tree=None,method="poisson",vorVolumes=None):
	if (vorVolumes is None) and (method == "VTFE"):
			raise Exception("Must provide voronoi volumes for VTFE.")
	# Generate KDTree
	boxsize = snap.properties['boxsize'].ratio("Mpc a h**-1")
	if tree is None:
		tree = scipy.spatial.cKDTree(snap['pos'],boxsize=boxsize)
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
	if method == "poisson":
		for k in range(0,len(nPairsList)):
			nPairsList[k,:] = tree.count_neighbors(scipy.spatial.cKDTree(voidCentres[[k],:],boxsize=boxsize),rBins*voidRadii[k],cumulative=False)[1:]
	elif method == "volume":
		# Volume weighted density in each shell:
		for k in range(0,len(voidRadii)):
			indicesList = np.array(tree.query_ball_point(voidCentres[k,:],rBins[-1]*voidRadii[k],workers=nThreads),dtype=np.int32)
			disp = snap['pos'][indicesList,:] - voidCentres[k,:]
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
		#	indices = tree.query_ball_point(voidCentres,rBins[k]*voidRadii,workers=nThreads)
		#	listLengths = np.array([len(list) for list in indices])
		#	volSum = np.array([np.sum(vorVolumes[list]) for list in indices])
		#	if k == 0:
		#		nPairsList[:,k] = listLengths
		#		volumesList[:,k] = volSum
		#	else:
		#		nPairsList[:,k] = listLengths - nPairsList[:,k-1]
		#		volumesList[:,k] = volSum - volumesList[:,k-1]
		for k in range(0,len(voidRadii)):
			indicesList = np.array(tree.query_ball_point(voidCentres[k,:],rBins[-1]*voidRadii[k],workers=nThreads),dtype=np.int32)
			disp = snap['pos'][indicesList,:] - voidCentres[k,:]
			r = np.array(np.sqrt(np.sum(disp**2,1)))
			sort = np.argsort(r)
			indicesSorted = indicesList[sort]
			boundaries = np.searchsorted(r[sort],rBins*voidRadii[k])
			print("Doing void " + str(k) + " of " + str(len(voidRadii)))
			nPairsList[k,:] = np.diff(boundaries)
			volumesCumulative = np.cumsum(vorVolumes[indicesSorted])
			volumesList[k,:] = volumesCumulative[boundaries[1:]]			
	return [nPairsList,volumesList]

def getRadialVelocityAverages(voidCentres,voidRadii,snap,rBins,nThreads=thread_count,tree=None,method="poisson",vorVolumes=None):
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
	nPartList = np.zeros((len(voidCentres),len(rBins)-1),dtype=np.int64)
	volumesList = np.outer(voidRadii**3,volumes)
	for k in range(0,len(vRList)):
		indicesList = np.array(tree.query_ball_point(voidCentres[k,:],rBins[-1]*voidRadii[k],workers=nThreads),dtype=np.int32)
		disp = snap['pos'][indicesList,:] - voidCentres[k,:]
		r = np.array(np.sqrt(np.sum(disp**2,1)))
		sort = np.argsort(r)
		indicesSorted = indicesList[sort]
		boundaries = np.searchsorted(r[sort],rBins*voidRadii[k])
		vR = np.sum(disp*snap['vel'][indicesSorted,:],1)*\
			vorVolumes[indicesSorted]/r
		#print("Doing void " + str(k) + " of " + str(len(vRList)))
		for l in range(0,len(rBins)-1):
			indices = indicesSorted[boundaries[l]:boundaries[l+1]]
			if len(indices) > 0:
				#indicesSphere = sort[boundaries[l]:boundaries[l+1]]
				vRList[k,l] = np.sum(vR[boundaries[l]:boundaries[l+1]])/\
					np.sum(vorVolumes[boundaries[l]:boundaries[l+1]])
				nPartList[k,l] = len(indices)
	return [vRList,volumesList]

# Direct pair counting in rescaled variables:
def stackScaledVoids(voidCentres,voidRadii,snap,rBins,nThreads=thread_count,tree=None,method="poisson",vorVolumes=None,nPairsList=None,volumesList=None,errorType="Mean",interval=68):
	if (nPairsList is None) or (volumesList is None):
		[nPairsList,volumesList] = getPairCounts(voidCentres,voidRadii,snap,rBins,nThreads=nThreads,tree=tree,method=method,vorVolumes=vorVolumes)
	if method == "poisson":
		nbarj = (np.sum(nPairsList,0) + 1)/(np.sum(volumesList,0))
		if errorType == "Mean":
			sigmabarj = np.sqrt(len(voidCentres)*np.sum(nPairsList,0))/np.sum(volumesList,0)/np.sqrt(len(voidCentres)-1)
		elif errorType == "Profile":
			sigmabarj = np.sqrt(weightedVariance(nPairsList/volumesList,volumesList,axis=0))
		elif errorType == "Percentile":
			sigmabarj = np.percentile(nPairsList/volumesList,[(100-interval)/2,50 + interval/2],axis=0)
			sigmabarj[0,:] = nbarj - sigmabarj[0,:]
			sigmabarj[1,:] = sigmabarj[1,:] - nbarj
		elif errorType == "Weighted":
			weights = volumesList/np.sum(volumesList,0)
			sigmabarj = np.sqrt(np.var(nPairsList/(volumesList),0)*np.sum(weights**2,0))
		else:
			raise Exception("Invalid error type.")
	elif method == "naive":
		nbarj = np.sum(nPairsList/volumesList,0)/len(voidCentres)
		if errorType == "Mean":
			sigmabarj = np.std(nPairsList/volumesList,0)/np.sqrt(len(voidCentres)-1)
		elif errorType == "Profile":
			sigmabarj = np.std(nPairsList/volumesList,0)/np.sqrt(len(voidCentres)-1)
		elif errorType == "Percentile":
			sigmabarj = np.std(nPairsList/volumesList,0)
		else:
			raise Exception("Invalid error type.")
	elif method == "VTFE":
		nbarj = np.sum(nPairsList,0)/(np.sum(volumesList,0))
		if errorType == "Mean":
			sigmabarj = np.sqrt(len(voidCentres)*np.sum(nPairsList,0))/np.sum(volumesList,0)
		elif errorType == "Profile":
			sigmabarj = np.sqrt(weightedVariance(nPairsList/volumesList,volumesList,axis=0))
		else:
			raise Exception("Invalid error type.")
	elif method == "cumulative":
		nPairCum = (np.cumsum(nPairsList,axis=1)/np.cumsum(volumesList,axis=1))
		nbarj = weightedMean(nPairCum,volumesList,axis=0)
		sigmabarj = np.sqrt(weightedVariance(nPairCum,volumesList,axis=0)/(len(nPairCum)-1))
	else:
		raise Exception("Unrecognised stacking method.")
		
	return [nbarj,sigmabarj]

# Stack radial velocities of voids (Incidentally, this is mostly the same as the normal stacking, just with a different variable - would be better to merge them):
def stackScaledVoidsVelocities(voidCentres,voidRadii,snap,rBins,nThreads=thread_count,tree=None,method="poisson",vorVolumes=None,nPairsList=None,volumesList=None,errorType="Mean"):
	if (nPairsList is None) or (volumesList is None):
		[nPairsList,volumesList] = getRadialVelocityAverages(voidCentres,voidRadii,snap,rBins,nThreads=nThreads,tree=tree,method=method,vorVolumes=vorVolumes)
	if method == "poisson":
		nbarj = (np.sum(nPairsList,0) + 1)/(np.sum(volumesList,0))
		if errorType == "Mean":
			sigmabarj = np.sqrt(len(voidCentres)*np.sum(nPairsList,0))/np.sum(volumesList,0)
		elif errorType == "Profile":
			sigmabarj = np.sqrt(weightedVariance(nPairsList/volumesList,volumesList,axis=0))/np.sqrt(len(voidCentres)-1)
		elif errorType == "Standard":
			sigmabarj = np.sqrt(weightedVariance(nPairsList/volumesList,volumesList,axis=0))
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
			sigmabarj = np.sqrt(len(voidCentres)*np.sum(nPairsList,0))/np.sum(volumesList,0)
		elif errorType == "Profile":
			sigmabarj = np.sqrt(weightedVariance(nPairsList/volumesList,volumesList,axis=0))
		else:
			raise Exception("Invalid error type.")
	else:
		raise Exception("Unrecognised stacking method.")
		
	return [nbarj,sigmabarj]


# Apply a filter to a set of voids before stacking them:
def stackVoidsWithFilter(voidCentres,voidRadii,filterToApply,snap,rBins=None,nPairsList=None,volumesList=None,nThreads=thread_count,tree=None,method="poisson",vorVolumes=None,errorType="Mean"):
	if rBins is None:
		rBins = np.linspace(0,3,31)
	if (nPairsList is None) or (volumesList is None):
		[nPairsList,volumesList] = getPairCounts(voidCentres,voidRadii,snap,rBins,nThreads=nThreads,tree=tree,method=method,vorVolumes=vorVolumes)
	return stackScaledVoids(voidCentres[filterToApply,:],voidRadii[filterToApply],snap,rBins,nThreads=nThreads,tree=tree,method=method,nPairsList=nPairsList[filterToApply,:],volumesList=volumesList[filterToApply,:],errorType=errorType)
		
# Apply a filter to a stack of voids before summing their velocities.
def stackVoidVelocitiesWithFilter(voidCentres,voidRadii,filterToApply,snap,rBins=None,nPairsList=None,volumesList=None,nThreads=thread_count,tree=None,method="poisson",vorVolumes=None,errorType="Profile"):
	if rBins is None:
		rBins = np.linspace(0,3,31)
	if (nPairsList is None) or (volumesList is None):
		[nPairsList,volumesList] = getRadialVelocityAverages(voidCentres,voidRadii,snap,rBins,nThreads=nThreads,tree=tree,method=method,vorVolumes=vorVolumes)
	return stackScaledVoidsVelocities(voidCentres[filterToApply,:],voidRadii[filterToApply],snap,rBins,nThreads=nThreads,tree=tree,method=method,nPairsList=nPairsList[filterToApply,:],volumesList=volumesList[filterToApply],errorType=errorType)

def meanDensityContrast(voidParticles,volumes,nbar):
	return (len(voidParticles)/(nbar*np.sum(volumes[voidParticles]))) - 1.0

def lambdaVoid(voidParticles,volumes,nbar,radius):
	return meanDensityContrast(voidParticles,volumes,nbar)*((radius)**(1.2))

# Central Density:
def centralDensity(voidCentre,voidRadius,positions,volumes,masses,boxsize=None,tree=None,centralRatio = 4,nThreads=thread_count):
	if tree is None:
		tree = scipy.spatial.cKDTree(positions,boxsize=boxsize)
	central = tree.query_ball_point(voidCentre,voidRadius/centralRatio,workers=nThreads)
	rhoCentral = np.sum(masses[central])/np.sum(volumes[central])
	return rhoCentral
	
def centralDensityNN(voidCentre,positions,masses,volumes,boxsize=None,tree = None,nThreads = thread_count,nNeighbours=64,rCut=None):
	if tree is None:
		tree = scipy.spatial.cKDTree(positions,boxsize=boxsize)
	if rCut is not None:
		central = tree.query(voidCentre,k=nNeighbours,workers=nThreads,distance_upper_bound = rCut)
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
		return profileModelNadathur(r,modelArgs[0],modelArgs[1],modelArgs[2],modelArgs[3])
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
	[popt, pcov] = curve_fit(profileModelHamaus,rBinCentres,nbarj/nbar,sigma=sigmabarj,maxfev=10000, xtol=5.e-3,
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
	
# Attempts to select voids from set1 such that the distribution of lambda matches that of set 2.
def matchDistributionFilter(lambdas1,lambdas2,lambdaBins=None,randomSeed=None,binsToMatch=None,binValues1=None,binValues2=None):
	if lambdaBins is None:
		lambdaBins = np.linspace(np.min([np.min(lambdas1),np.min(lambdas2)]),np.max([np.max(lambdas1),np.max(lambdas2)]),101)
	if randomSeed is not None:
		np.random.seed(seed = randomSeed)
	if binsToMatch is None:
		inBinValue1 = [np.array(range(0,len(lambdas1)))]
		noInBinValue1 = len(lambdas1)
		inBinValue2 = [np.array(range(0,len(lambdas2)))]
		noInBinValue2 = len(lambdas2)
	else:
		# Allow for matching lambdas only within the specified bins, not over all voids:
		[inBinValue1,noInBinValue1] = plot_utilities.binValues(binValues1,binsToMatch)
		[inBinValue2,noInBinValue2] = plot_utilities.binValues(binValues2,binsToMatch)
	set1List = np.array([],dtype=np.int64)
	for k in range(0,len(inBinValue1)):
		[binList1,noInBins1] = plot_utilities.binValues(lambdas1[inBinValue1[k]],lambdaBins)
		[binList2,noInBins2] = plot_utilities.binValues(lambdas2[inBinValue2[k]],lambdaBins)
		fracs = noInBins2/len(lambdas2[inBinValue2[k]])
		set1ToDrawMax = np.array(np.floor(fracs*len(lambdas1[inBinValue1[k]])),dtype=np.int64)
		for l in range(0,len(fracs)):
			if noInBins1[l] < set1ToDrawMax[l]:
				set1List = np.hstack((set1List,np.array(inBinValue1[k][binList1[l]])))
			else:
				set1List = np.hstack((set1List,np.random.choice(inBinValue1[k][binList1[l]],size=set1ToDrawMax[l],replace=False)))
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
		return np.sum((muBins[intRange][1:]**3 - muBins[intRange][0:-1]**3)*rho[intRange][0:-1])

# Compute a few examples, and compare to the averages of the volume weigted average densities:
def getProfileIntegralAndDeltaAverage(snap,rBins,radii,centres,pairCounts,volumesList,deltaBar,nbar,condition,method="poisson",errorType="Profile",mumin=0.0,mumax=1.0):
	filterToUse = np.where(condition)[0]
	[nbarj,sigma] = stackVoidsWithFilter(centres,radii,filterToUse,snap,rBins,nPairsList = pairCounts,volumesList=volumesList,method=method,errorType=errorType)
	rBinCentres = plot_utilities.binCentres(rBins)
	profInt = profileIntegral(rBins,nbarj/nbar,mumin=mumin,mumax=mumax)
	profIntUpper = profileIntegral(rBins,(nbarj + sigma)/nbar,mumin=mumin,mumax=mumax)
	profIntLower = profileIntegral(rBins,(nbarj - sigma)/nbar,mumin=mumin,mumax=mumax)
	if method == "poisson":
		deltaAverage = np.sum(radii[filterToUse]**3*(1.0 + deltaBar[filterToUse]))/np.sum(radii[filterToUse]**3)
		deltaAverageError = np.sqrt(weightedVariance((1.0 + deltaBar[filterToUse]),radii[filterToUse]**3))/np.sqrt(len(filterToUse))
	else:
		deltaAverage = np.mean((1.0 + deltaBar[filterToUse]))
		deltaAverageError = np.std((1.0 + deltaBar[filterToUse]))/np.sqrt(len(filterToUse))
	return [profInt,deltaAverage,deltaAverageError,profIntLower,profIntUpper]

# Get average and error of the effective average densities:
def deltaBarEffAverageAndError(radii,deltaBarEff,condition,method="poisson"):
	filterToUse = np.where(condition)[0]
	if method == "poisson":
		deltaAverage = np.sum(radii[filterToUse]**3*(1.0 + deltaBarEff[filterToUse]))/np.sum(radii[filterToUse]**3)
		deltaAverageError = np.sqrt(weightedVariance((1.0 + deltaBarEff[filterToUse]),radii[filterToUse]**3))/np.sqrt(len(filterToUse))
	else:
		deltaAverage = np.mean((1.0 + deltaBarEff[filterToUse]))
		deltaAverageError = np.std((1.0 + deltaBarEff[filterToUse]))/np.sqrt(len(filterToUse))
	return [deltaAverage,deltaAverageError]

def deltaVError(deltaVList):
	return (deltaVList[0] - deltaVList[2],deltaVList[3] - deltaVList[0])

def deltaVRatio(deltaVlist):
	return deltaVlist[1]/deltaVlist[0]

# Studying the distribution of the profiles:
def minmax(x):
	return np.array([np.min(x),np.max(x)])

# Get minimum and maximum ranges of the profle distribution:
def getRanges(rhoAH,rhoZV,rBinStackCentres,filterToUseAH=None,filterToUseZV = None):
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
def computeMeanStacks(centresList,radiiList,massesList,conditionList,pairsList,volumesList,
		snapList,nbar,rBins,rMin,rMax,mMin,mMax,
		method="poisson",errorType = "Weighted",toInclude = "all"):
	if toInclude == "all":
		toInclude = range(0,len(centresList))
	nToStack = len(toInclude)
	nbarjStack = np.zeros((nToStack,len(rBins) - 1))
	sigmaStack = np.zeros((nToStack,len(rBins) - 1))
	for k in range(0,nToStack):
		[nbarj,sigma] = stackVoidsWithFilter(
			centresList[toInclude[k]],radiiList[toInclude[k]],
			np.where((radiiList[toInclude[k]] > rMin) & \
			(radiiList[toInclude[k]] < rMax) & \
			conditionList[toInclude[k]] & (massesList[toInclude[k]] > mMin) & \
			(massesList[toInclude[k]] <= mMax))[0],snapList[toInclude[k]],rBins,
			nPairsList = pairsList[toInclude[k]],
			volumesList=volumesList[toInclude[k]],
			method=method,errorType=errorType)
		nbarjStack[k,:] = nbarj
		sigmaStack[k,:] = sigma
	return [nbarjStack,sigmaStack]
		










