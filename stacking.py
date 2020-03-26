# Stacking functions
import Corrfunc
import scipy.spatial
import numpy as np
from . import snapedit
import multiprocessing as mp
from scipy.optimize import curve_fit
from . import plot
thread_count = mp.cpu_count()

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
def getCrossCorrelations(ahCentres,voidCentres,ahRadii,voidRadii,matterSnap,rMin = 0,rMax = np.inf,rRange = np.linspace(0.1,10,101),nThreads=thread_count,boxsize = 200.0):
	rFilter1 = np.where((ahRadii > rMin) & (ahRadii < rMax))[0]
	rFilter2 = np.where((voidRadii > rMin) & (voidRadii < rMax))[0]
	ahPos = ahCentres[rFilter1,:]
	vdPos = voidCentres[rFilter2,:]
	xiAM = simulationCorrelation(rRange,boxsize,ahPos,data2=snap['pos'],nThreads=nThreads,weights2 = snap['mass'])
	xiVM = simulationCorrelation(rRange,boxsize,vdPos,data2=snap['pos'],nThreads=nThreads,weights2 = snap['mass'])
	xiAV = simulationCorrelation(rRange,boxsize,ahPos,data2=vdPos,nThreads=nThreads)
	return [xiAM,xiVM,xiAV]

# Auto correlation of a set of points:
def getAutoCorrelations(ahCentres,voidCentres,ahRadii,voidRadii,matterSnap,rMin = 0,rMax = np.inf,rRange = np.linspace(0.1,10,101),nThreads=thread_count,boxsize = 200.0):
	rFilter1 = np.where((ahRadii > rMin) & (ahRadii < rMax))[0]
	rFilter2 = np.where((voidRadii > rMin) & (voidRadii < rMax))[0]
	ahPos = ahCentres[rFilter1,:]
	vdPos = voidCentres[rFilter2,:]
	xiAA = simulationCorrelation(rRange,boxsize,ahPos,nThreads=nThreads)
	xiVV = simulationCorrelation(rRange,boxsize,vdPos,nThreads=nThreads)
	return [xiAA,xiVV]

# Plot the correlation function:
def plotCrossCorrelations(rBins,xiAM,xiVM,xiAV,ax=None,rMin=0,rMax = np.inf):
	rBinCentres = plot.binCentres(rBins)
	if ax is None:
		fig, ax = plt.subplots()
	ax.plot(rBinCentres,xiAM,label="Antihalo-Matter cross correlation")
	ax.plot(rBinCentres,xiVM,label="ZOBOV void-Matter cross correlation")
	ax.plot(rBinCentres,xiAV,label="ZOBOV void-Antihalo cross correlation")
	ax.set_xlabel('$r [\mathrm{Mpc}/h]$')
	ax.set_ylabel('$\\xi(r)$',fontsize = 15)
	ax.legend()
	ax.tick_params(axis='both',labelsize=15)
	ax.set_ylim([-1,5])
	if rMin > 0:
		if np.isfinite(rMax):
			ax.set_title('$R_{\mathrm{eff}} = $' + scientificFormat(rMin) + '-' + scientificFormat(rMax) + '$\,\mathrm{Mpc}/h ($' + scientificFormat(RtoM(rMin,a,b)) + '-' + scientificFormat(RtoM(rMax,a,b)) + '$M_{\mathrm{sol}}/h)$')
		else:
			ax.set_title('$R_{\mathrm{eff}} > $' + scientificFormat(rMin) + '$\,\mathrm{Mpc}/h ($' + scientificFormat(RtoM(rMin,a,b)) + '$M_{\mathrm{sol}}/h)$')
	else:
		if np.isfinite(rMax):
			ax.set_title('$R_{\mathrm{eff}} < $' + scientificFormat(rMax) + '$\,\mathrm{Mpc}/h ($' + scientificFormat(RtoM(rMax,a,b)) + '$M_{\mathrm{sol}}/h)$')
		else:
			ax.set_title('All voids/anti-halos')


# Stacking:
def stackVoids(voidCentres,snap,rBins,nThreads=thread_count,voidScales=None):
	if voidScales is None:
		voidScales = np.ones(len(voidCentres))
	boxsize = snap.properties['boxsize'].ratio("Mpc a h**-1")
	massCentres = snap['pos'].in_units("Mpc a h**-1")
	massValues = snap['mass'].in_units("Msol h**-1")
	# Get pairs:
	DD = Corrfunc.theory.DD(0,nThreads,rBins,voidCentres[:,0],voidCentres[:,1],voidCentres[:,2],X2=massCentres[:,0],Y2=massCentres[:,1],Z2=massCentres[:,2],weights2=massValues,periodic=True,boxsize=boxsize)
	# Get void volume sum:
	rBinsUp = rBins[1:]
	rBinsLow = rBins[0:-1]
	volumes = 4*np.pi*(rBinsup**3 - rBinsLow**3)/3
	volsum = np.sum(voidScales**3)*volumes

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
	nPairsList = np.zeros((len(voidCentres),len(rBins[1:])),dtype=np.int64)
	volumesList = np.outer(voidRadii**3,volumes)
	if method != "VTFE":
		for k in range(0,len(nPairsList)):
			nPairsList[k,:] = tree.count_neighbors(scipy.spatial.cKDTree(voidCentres[[k],:],boxsize=boxsize),rBins*voidRadii[k],cumulative=False)[1:]
	else:
		for k in range(0,len(rBins)):
			indices = tree.query_ball_point(voidCentres,rBins[k]*voidRadii,n_jobs=nThreads)
			listLengths = np.array([len(list) for list in indices])
			volSum = np.array([np.sum(vorVolumes[list]) for list in indices])
			if k == 0:
				nPairsList[:,k] = listLengths
				volumesList[:,k] = volSum
			else:
				nPairsList[:,k] = listLengths - nPairsList[:,k-1]
				volumesList[:,k] = volSum - volumesList[:,k-1]
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
		indicesList = np.array(tree.query_ball_point(voidCentres[k,:],rBins[-1]*voidRadii[k],n_jobs=nThreads),dtype=np.int32)
		disp = snap['pos'][indicesList,:] - voidCentres[k,:]
		r = np.array(np.sqrt(np.sum(disp**2,1)))
		sort = np.argsort(r)
		indicesSorted = indicesList[sort]
		boundaries = np.searchsorted(r[sort],rBins*voidRadii[k])
		vR = np.sum(disp*snap['vel'][indicesList,:],1)*vorVolumes[indicesList]/r
		print("Doing void " + str(k) + " of " + str(len(vRList)))
		for l in range(0,len(rBins)-1):
			indices = indicesSorted[boundaries[l]:boundaries[l+1]]
			if len(indices) > 0:
				indicesSphere = sort[boundaries[l]:boundaries[l+1]]
				vRList[k,l] = np.sum(vR[indicesSphere])/np.sum(vorVolumes[indices])
				nPartList[k,l] = len(indices)
	return [vRList,volumesList]

# Direct pair counting in rescaled variables:
def stackScaledVoids(voidCentres,voidRadii,snap,rBins,nThreads=thread_count,tree=None,method="poisson",vorVolumes=None,nPairsList=None,volumesList=None,errorType="Mean"):
	if (nPairsList is None) or (volumesList is None):
		[nPairsList,volumesList] = getPairCounts(voidCentres,voidRadii,snap,rBins,nThreads=nThreads,tree=tree,method=method,vorVolumes=vorVolumes)
	if method == "poisson":
		nbarj = (np.sum(nPairsList,0) + 1)/(np.sum(volumesList,0))
		if errorType == "Mean":
			sigmabarj = np.sqrt(len(voidCentres)*np.sum(nPairsList,0))/np.sum(volumesList,0)
		elif errorType == "Profile":
			sigmabarj = np.sqrt(weightedVariance(nPairsList/volumesList,volumesList,axis=0))/np.sqrt(len(voidCentres)-1)
		else:
			raise Exception("Invalid error type.")
	elif method == "naive":
		nbarj = np.sum(nPairsList/volumesList,0)/len(voidCentres)
		if errorType == "Mean":
			sigmabarj = np.std(nPairsList/volumesList,0)/np.sqrt(len(voidCentres)-1)
		elif errorType == "Profile":
			sigmabarj = np.std(nPairsList/volumesList,0)/np.sqrt(len(voidCentres)-1)
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
		else:
			raise Exception("Invalid error type.")
	elif method == "naive":
		nbarj = np.sum(nPairsList/volumesList,0)/len(voidCentres)
		if errorType == "Mean":
			sigmabarj = np.std(nPairsList/volumesList,0)/np.sqrt(len(voidCentres)-1)
		elif errorType == "Profile":
			sigmabarj = np.std(nPairsList/volumesList,0)/np.sqrt(len(voidCentres)-1)
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
def stackVoidsWithFilter(voidCentres,voidRadii,filterToApply,snap,rBins=None,nPairsList=None,volumesList=None,nThreads=thread_count,tree=None,method="poisson",vorVolumes=None,errorType="Profile"):
	if rBins is None:
		rBins = np.linspace(0,3,31)
	if (nPairsList is None) or (volumesList is None):
		[nPairsList,volumesList] = getPairCounts(voidCentres,voidRadii,snap,rBins,nThreads=nThreads,tree=tree,method=method,vorVolumes=vorVolumes)
	return stackScaledVoids(voidCentres[filterToApply,:],voidRadii[filterToApply],snap,rBins,nThreads=nThreads,tree=tree,method=method,nPairsList=nPairsList[filterToApply,:],volumesList=volumesList[filterToApply],errorType=errorType)
		
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
	central = tree.query_ball_point(voidCentre,voidRadius/centralRatio,n_jobs=nThreads)
	rhoCentral = np.sum(masses[central])/np.sum(volumes[central])
	return rhoCentral
	
def centralDensityNN(voidCentre,positions,masses,volumes,boxsize=None,tree = None,nThreads = thread_count,nNeighbours=64,rCut=None):
	if tree is None:
		tree = scipy.spatial.cKDTree(positions,boxsize=boxsize)
	if rCut is not None:
		central = tree.query(voidCentre,k=nNeighbours,n_jobs=nThreads,distance_upper_bound = rCut)
	else:
		central = tree.query(voidCentre,k=nNeighbours,n_jobs=nThreads)
	if (len(central[0]) < 1) and (rCut is not None):
		# Try again, ignoring the cut:
		central = tree.query(voidCentre,k=nNeighbours,n_jobs=nThreads)
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
	rBinCentres = plot.binCentres(rBins)
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
	
#Plot a void profile against its prediction:
def plotProfileVsPrediction(nbarj,sigambarj,rBins,nbar,fontsize=14,ax=None,formatAxis=True,model="Hamaus",color='r',labelSuffix = ""):
	if model == "Hamaus":
		[rs,dc,srs,sdc] = profileParamsHamaus(nbarj,sigambarj,rBins,nbar)
		params = [rs,dc]
	elif model == "Nadathur":
		[alpha,beta,rs,deltac] = profileParamsNadathur(nbarj,rBins,nbar)
		params = [alpha,beta,rs,deltac]
	else:
		raise Exception("Unknown profile model.")
	if ax is None:
		fig, ax = plt.subplots()
	rBinCentres = plot.binCentres(rBins)
	ax.errorbar(rBinCentres,nbarj/nbar,yerr = sigambarj/nbar,fmt='x-',color=color,label='Computed profile' + labelSuffix)
	ax.plot(rBinCentres,profileModel(rBinCentres,model,params),'--',color=color,label='Predicted profile' + labelSuffix)
	if model == "Hamaus":
		paramspp = [rs + srs,dc + sdc]
		paramspm = [rs + srs,dc - sdc]
		paramsmp = [rs - srs,dc + sdc]
		paramsmm = [rs - srs,dc - sdc]
		profpp = profileModel(rBinCentres,model,paramspp)
		profpm = profileModel(rBinCentres,model,paramspm)
		profmp = profileModel(rBinCentres,model,paramsmp)
		profmm = profileModel(rBinCentres,model,paramsmm)
		proffComb = np.vstack((profpp,profpm,profmp,profmm))
		profUpp = np.max(proffComb,0)
		profLow = np.min(proffComb,0)
		ax.fill_between(rBinCentres,profLow,profUpp,facecolor=color,alpha=0.5)
	if formatAxis:
		ax.set_xlabel('$r/r_{\\mathrm{eff}}$',fontsize=fontsize)
		ax.set_ylabel('$\\rho/\\bar{\\rho}$',fontsize=fontsize)
		ax.tick_params(axis='both',labelsize=fontsize)
		ax.legend(prop={"size":fontsize})

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
		[inBinValue1,noInBinValue1] = plot.binValues(binValues1,binsToMatch)
		[inBinValue2,noInBinValue2] = plot.binValues(binValues2,binsToMatch)
	set1List = np.array([],dtype=np.int64)
	for k in range(0,len(inBinValue1)):
		[binList1,noInBins1] = plot.binValues(lambdas1[inBinValue1[k]],lambdaBins)
		[binList2,noInBins2] = plot.binValues(lambdas2[inBinValue2[k]],lambdaBins)
		fracs = noInBins2/len(lambdas2[inBinValue2[k]])
		set1ToDrawMax = np.array(np.floor(fracs*len(lambdas1[inBinValue1[k]])),dtype=np.int64)
		for l in range(0,len(fracs)):
			if noInBins1[l] < set1ToDrawMax[l]:
				set1List = np.hstack((set1List,np.array(inBinValue1[k][binList1[l]])))
			else:
				set1List = np.hstack((set1List,np.random.choice(inBinValue1[k][binList1[l]],size=set1ToDrawMax[l],replace=False)))
	return set1List
	

