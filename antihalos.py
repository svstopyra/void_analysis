import numpy as np
import pynbody
from . import context, anti_halo_plot, stacking
import os
import multiprocessing as mp
thread_count = mp.cpu_count()

# Get the radii of anti-halo voids:
def voidsRadiiFromAntiHalos(snapn,snapr,hn,hr,volumes):
	b = pynbody.bridge.Bridge(snapn,snapr)
	voidRadii = np.zeros(len(hr))
	for k in range(0,len(hr)):
		voidRadii[k] = np.cbrt(3.0*np.sum(volumes[hr[k+1]['iord']])/(4.0*np.pi))
	return voidRadii

# Compute anti-halo centres:
def computeAntiHaloCentres(hr,snap,volumes):
	centres = np.zeros((len(hr),3))
	periodicity = [snap.properties['boxsize'].ratio("Mpc a h**-1")]*3
	for k in range(0,len(hr)):
		centres[k,:] = context.computePeriodicCentreWeighted(snap['pos'][hr[k+1]['iord']],volumes[hr[k+1]['iord']],periodicity)
	return centres

# Get vector of anti-halo masses
def getAntiHaloMasses(hr):
	antiHaloMasses = np.zeros(len(hr))
	for k in range(0,len(hr)):
		antiHaloMasses[k] = np.sum(hr[k+1]['mass'])

# Get the volume averaged densities of all the anti-halos.
def getAntiHaloDensities(hr,snap,volumes=None):
	if volumes is None:
		# Use the nearest neighbour distance:
		volumes = snap['smooth']
		rho = snap['rho']
	else:
		rho = np.array(snap['mass'].in_units("Msol h**-1"))/volumes
	antiHaloDensities = np.zeros(len(hr))
	for k in range(0,len(hr)):
		antiHaloDensities[k] = np.sum(rho[hr[k+1]['iord']]*volumes[hr[k+1]['iord']])/np.sum(volumes[hr[k+1]['iord']])

# Polynomial fit of the anti-halo massses and their radii above a given mass threshold
def fitMassAndRadii(antiHaloMasses,antiHaloRadii,logThresh=14):
	logMass = np.log10(antiHaloMasses)
	logRad = np.log10(antiHaloRadii)
	aboveThresh = np.where(logMass > logThresh)
	fit = np.polyfit(logMass[aboveThresh],logRad[aboveThresh],1)# = (b,a)
	return fit
	
# Conversion between mass and radius, according to the fit provided by fitMassAndRadius
def MtoR(x,a,b):
	return (10**a)*(x**b)

# Conversion between radius and mass, according to the fit provided by fitMassAndRadius
def RtoM(y,a,b):
	return (y/(10**a))**(1/b)
	
# Volume weighted barycentres of a set of particles:
def computeVolumeWeightedBarycentre(positions,volumes):
	if volumes.shape != (1,len(positions)):
		volumes2 = np.reshape(volumes,(1,len(positions)))
	else:
		volumes2 = volumes
	weightedPos = (volumes2.T)*positions
	return np.sum(weightedPos,0)/np.sum(volumes)

# Weighted barycentre of particles, accounting for periodic boundary conditions.
def computePeriodicCentreWeighted(positions,weight,periodicity):
	if np.isscalar(periodicity):
		period = (periodicity,periodicity,periodicity)
	else:
		period = periodicity
	if(len(period) != 3):
		raise Exception("Periodicity must be a length 3 vector or a scalar.")
	# Map everything into angles so that we can properly account for how close particles are:
	theta = np.zeros((len(positions),3))
	theta[:,0] = (positions[:,0])*2.0*np.pi/period[0]
	theta[:,1] = (positions[:,1])*2.0*np.pi/period[1]
	theta[:,2] = (positions[:,2])*2.0*np.pi/period[2]
	M = np.sum(weight)
	xi = np.cos(theta)
	zeta = np.sin(theta)
	# Angular averages:
	xibar = np.sum(weight[:,None]*xi,0)/M
	zetabar = np.sum(weight[:,None]*zeta,0)/M
	# Back to theta:
	thetabar = np.arctan2(-zetabar,-xibar) + np.pi
	return (period*thetabar/(2.0*np.pi))
	
# Return the voids which lie within an effective radius of a given halo:
def getCoincidingVoids(centre,radius,voidCentres):
	dist = np.sqrt(np.sum((voidCentres - centre)**2,1))
	return np.where(dist <= radius)

# Return all the coincident voids with a given radius range	
def getCoincidingVoidsInRadiusRange(centre,radius,voidCentres,voidRadii,rMin,rMax):
	coinciding = getCoincidingVoids(centre,radius,voidCentres)
	inRange = np.where((voidRadii[coinciding] >= rMin) & (voidRadii[coinciding] <= rMax))
	return coinciding[0][inRange]	
def getAntihaloOverlapWithVoid(antiHaloParticles,voidParticles,volumes):
	intersection = np.intersect1d(antiHaloParticles,voidParticles)
	return [np.sum(volumes[intersection])/np.sum(volumes[antiHaloParticles]),np.sum(volumes[intersection])/np.sum(volumes[voidParticles])]

def getOverlapFractions(antiHaloParticles,cat,voidList,volumes,mode = 0):
	fraction = np.zeros((len(voidList),2))
	for k in range(0,len(fraction)):
		overlap = getAntihaloOverlapWithVoid(antiHaloParticles,cat.void2Parts(voidList[k]),volumes)
		fraction[k,:] = overlap
	if mode == "both":
		return fraction
	else:
		return fraction[:,mode]

def getVoidOverlapFractionsWithAntihalos(voidParticles,hr,antiHaloList,volumes,mode = 0):
	fraction = np.zeros((len(antiHaloList),2))
	for k in range(0,len(fraction)):
		overlap = getAntihaloOverlapWithVoid(hr[antiHaloList[k]+1]['iord'],voidParticles,volumes)
		fraction[k,:] = overlap
	if mode == "both":
		return fraction
	else:
		return fraction[:,mode]

# Remove any voids from the set that are actually subvoids of another in the set:
def removeSubvoids(cat,voidSet):
	# Get ID list:
	voidIDs = cat.voidID[voidSet]
	parentIDs = cat.parentID[voidSet]
	subvoids = np.in1d(parentIDs,voidIDs)
	return voidSet[np.logical_not(subvoids)]

# Function to figure out if an anti-halo has a corresponding ZOBOV void:
def getAntiHaloVoidCandidates(antiHalo,centre,radius,cat,volumes,rMin=None,rMax=None,threshold = 0.5,removeSubvoids = False,rank = True,searchRadius = None):
	voidCentres = cat.voidCentres
	voidRadii = cat.radius
	if searchRadius is None:
		searchRadius = radius
	if rMin is None:
		rMin = 0.75*searchRadius
	if rMax is None:
		rMax = 1.2*searchRadius
	coinciding = getCoincidingVoidsInRadiusRange(centre,searchRadius,voidCentres,voidRadii,rMin=rMin,rMax=rMax)
	fractions = np.zeros((len(coinciding),2))
	for k in range(0,len(coinciding)):
		fractions[k,:] = getAntihaloOverlapWithVoid(antiHalo['iord'],cat.void2Parts(coinciding[k]),volumes)
	candidates = np.where((fractions[:,0] >= threshold) & (fractions[:,1] >= threshold))
	coincidingCandidates = coinciding[candidates]
	if rank:
		# Sort in descending order of overlap with the anti-halo
		coincidingCandidates = coincidingCandidates[np.argsort(-fractions[candidates,0])]
	
	if not removeSubvoids:
		return coinciding[candidates]
	else:
		return removeSubvoids(cat,coinciding[candidates])

# Compute the volume weighted barycentres of each zone:
def computeZoneCentres(snap,cat,volumes):
	zoneCentres = np.zeros((cat.numZonesTot,3))
	boxsize = snap.properties['boxsize'].ratio("Mpc a h**-1")
	periodicity = [boxsize]*3
	for k in range(0,cat.numZonesTot):
		zoneParts = cat.zones2Parts(k)
		zoneCentres[k,:] = computePeriodicCentreWeighted(snap[zoneParts]['pos'],volumes[zoneParts],periodicity)
	return zoneCentres

# Find the ZOBOV zones that could correspond to a particular halo
def getCorrespondingZoneCandidates(antiHalo,centre,radius,volumes,catalog,zoneCentres,threshold = 0.5,searchRadius = None):
	# Automatically choose a search radius based on the anti-halo radius, if not specified:
	if searchRadius is None:
		searchRadius = 1.5*radius
	# Find which zones are within the search radius:
	zoneDistances = np.sqrt(np.sum((zoneCentres - centre)**2,1))
	inRadius = np.where(zoneDistances <= searchRadius)
	# For these zones, compute the fraction of volume they share with the anti-halo:
	volumeShared = np.zeros(inRadius[0].shape)
	for k in range(0,len(volumeShared)):
		volumeShared[k] = getAntihaloOverlapWithVoid(antiHalo['iord'],catalog.zones2Parts(inRadius[0][k]),volumes)[1]
	highlyCorrelated = np.where(volumeShared >= threshold)
	return [inRadius[0][highlyCorrelated],volumeShared[highlyCorrelated]]

# Get possible subvoids that could correspond to a particular anti-halos
def getCorrespondingSubVoidCandidates(antiHalo,centre,radius,volumes,catalog,voidCentres,threshold = 0.5,searchRadius = None,subVoidsOnly = False):
	# Automatically choose a search radius based on the anti-halo radius, if not specified:
	if searchRadius is None:
		searchRadius = 1.5*radius
	# Find which zones are within the search radius:
	voidDistances = np.sqrt(np.sum((voidCentres - centre)**2,1))
	inRadius = np.where(voidDistances <= searchRadius)
	# For these zones, compute the fraction of volume they share with the anti-halo:
	volumeShared = np.zeros(inRadius[0].shape)
	voidVolumes = np.zeros(inRadius[0].shape)
	voidFullVolumes = np.zeros(voidVolumes.shape)
	antiHaloVolume = np.sum(volumes[antiHalo['iord']])
	for k in range(0,len(volumeShared)):	
		voidParticles = catalog.void2Parts(inRadius[0][k])
		intersection = np.intersect1d(antiHalo['iord'],voidParticles)
		volIntersection = np.sum(volumes[intersection])
		voidVolumes[k] = volIntersection
		voidFullVolumes[k] = np.sum(volumes[voidParticles])
		volumeShared[k] = volIntersection/np.sum(volumes[voidParticles])
	highlyCorrelated = np.where(volumeShared >= threshold) 
	if not subVoidsOnly:
		volumeFraction = np.sum(voidVolumes[highlyCorrelated])/antiHaloVolume
		voidVolumeFraction =  np.sum(voidVolumes[highlyCorrelated])/np.sum(voidFullVolumes[highlyCorrelated])
		return [inRadius[0][highlyCorrelated],volumeShared[highlyCorrelated],volumeFraction,voidVolumeFraction]
	else:
		voidIDs = catalog.voidID[inRadius[0][highlyCorrelated]]
		parentIDs = catalog.parentID[inRadius[0][highlyCorrelated]]
		subvoids = np.in1d(parentIDs,voidIDs)
		parentVoids = np.logical_not(subvoids)
		volumeFraction = np.sum(voidVolumes[highlyCorrelated][parentVoids])/antiHaloVolume
		voidVolumeFraction = np.sum(voidVolumes[highlyCorrelated][parentVoids])/np.sum(voidFullVolumes[highlyCorrelated][parentVoids])
		return [inRadius[0][highlyCorrelated][parentVoids],volumeShared[highlyCorrelated][parentVoids],volumeFraction,voidVolumeFraction]

def runGenPk(centresAH,centresZV,massesAH,massesZV,rFilterAH = None,rFilterZV=None):
	if rFilterAH is None:
		rFilterAH = slice(len(centresAH[:,0]))
	if rFilterZV is None:
		rFilterZV = slice(len(centresZV[:,0]))
	ahPos = pynbody.array.SimArray(centresAH[rFilterAH,:],"Mpc a h**-1")
	ahVel = pynbody.array.SimArray(np.zeros(ahPos.shape),"km a**1/2 s**-1")
	ahMass = pynbody.array.SimArray(np.ones(len(massesAH[rFilterAH])),"Msol h**-1")
	vdPos = pynbody.array.SimArray(centresZV[rFilterZV,:],"Mpc a h**-1")
	vdVel = pynbody.array.SimArray(np.zeros(catToUse.voidCentres[rFilterZV,:].shape),"km a**1/2 s**-1")
	vdMass = pynbody.array.SimArray(massesZV[rFilter2],"Msol h**-1")
	sAHs = snapedit.newSnap("antihalos.gadget2",ahPos,ahVel,dmMass=ahMass,gasPos = vdPos ,gasVel = vdVel ,gasMass = vdMass,properties=snap.properties,fmt=type(snap))
	# Run Gen-Pk:
	if not os.path.isdir("genpk_out"):
		os.system("mkdir genpk_out")
	os.system("gen-pk -i antihalos.gadget2 -o genpk_out")
	os.system("gen-pk -i antihalos.gadget2 -o genpk_out -c 0")
	# Import results:
	psAHs = np.loadtxt("./genpk_out/PK-DM-antihalos.gadget2")
	psVoids = np.loadtxt("./genpk_out/PK-by-antihalos.gadget2")
	psCross = np.loadtxt("./genpk_out/PK-DMxby-antihalos.gadget2")
	psMatter = np.loadtxt("./genpk_out/PK-DM-snapshot_011")
	return [psAHs,psVoids,psCross,psMatter]

# Estimate correlation function of discrete data. If data2 is specified, it computes the cross correlation of the two data sets
import Corrfunc
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

# Cross correlation of voids and anti-halo centres:
def getCrossCorrelations(ahCentres,voidCentres,ahRadii,voidRadii,matterSnap,rMin = 0,rMax = np.inf,rRange = np.linspace(0.1,10,101),nThreads=thread_count,boxsize = 200.0):
	rFilter1 = np.where((ahRadii > rMin) & (ahRadii < rMax))[0]
	rFilter2 = np.where((voidRadii > rMin) & (voidRadii < rMax))[0]
	ahPos = ahCentres[rFilter1,:]
	vdPos = voidCentres[rFilter2,:]
	xiAM = simulationCorrelation(rRange,boxsize,ahPos,data2=snap['pos'],nThreads=nThreads,weights2 = snap['mass'])
	xiVM = simulationCorrelation(rRange,boxsize,vdPos,data2=snap['pos'],nThreads=nThreads,weights2 = snap['mass'])
	xiAV = simulationCorrelation(rRange,boxsize,ahPos,data2=vdPos,nThreads=nThreads)
	return [xiAM,xiVM,xiAV]

# Auto correlations of void and anti-halo centres:
def getAutoCorrelations(ahCentres,voidCentres,ahRadii,voidRadii,matterSnap,rMin = 0,rMax = np.inf,rRange = np.linspace(0.1,10,101),nThreads=thread_count,boxsize = 200.0):
	rFilter1 = np.where((ahRadii > rMin) & (ahRadii < rMax))[0]
	rFilter2 = np.where((voidRadii > rMin) & (voidRadii < rMax))[0]
	ahPos = ahCentres[rFilter1,:]
	vdPos = voidCentres[rFilter2,:]
	xiAA = simulationCorrelation(rRange,boxsize,ahPos,nThreads=nThreads)
	xiVV = simulationCorrelation(rRange,boxsize,vdPos,nThreads=nThreads)
	return [xiAA,xiVV]

# Return specified stacks
def getStacks(ahRadius,ahMasses,antiHaloCentres,zvRadius,zvMasses,voidCentres,snap,pairCountsAH,pairCountsZV,volumesListAH,volumesListZV,conditionAH = None,conditionZV = None,showPlot=True,ax=None,rBins = np.linspace(0,3,31),sizeBins = [2,4,10,21],plotAH=True,plotZV=True,binType="radius",tree=None,sumType='poisson',yUpper = 1.3,valuesAH = None,valuesZV = None,binLabel="",errorType="Profile"):
	AHfilters = []
	ZVfilters = []
	nbar = len(snap)/(snap.properties['boxsize'].ratio("Mpc a h**-1"))**3
	for k in range(0,len(sizeBins)-1):
		if (valuesAH is None) or (valuesZV is None):
			if binType == "radius":
				valuesAH = ahRadius
				valuesZV = zvRadius
			elif binType == "mass":
				valuesAH = ahMasses
				valuesZV = zvMasses
			elif binType == "RtoM":
				valuesAH = ahMasses
				valuesZV = RtoM(zvMasses,a,b)
			else:
				raise Exception("Unrecognised bin type.")
		filterConditionAH = (valuesAH > sizeBins[k]) & (valuesAH <= sizeBins[k+1])
		filterConditionZV = (valuesZV > sizeBins[k]) & (valuesZV <= sizeBins[k+1])
		if conditionAH is not None:
			filterConditionAH = filterConditionAH & conditionAH
		if conditionZV is not None:
			filterConditionZV = filterConditionZV & conditionZV
		AHfilters.append(np.where(filterConditionAH))
		ZVfilters.append(np.where(filterConditionZV))
	nBarsAH = []
	nBarsZV = []
	sigmaBarsAH = []
	sigmaBarsZV = []
	for k in range(0,len(sizeBins)-1):
		[nbarj_AH,sigma_AH] = stacking.stackVoidsWithFilter(antiHaloCentres,ahRadius,AHfilters[k][0],snap,rBins,tree=tree,method=sumType,nPairsList=pairCountsAH,volumesList=volumesListAH,errorType=errorType)
		[nbarj_ZV,sigma_ZV] = stacking.stackVoidsWithFilter(voidCentres,zvRadius,ZVfilters[k][0],snap,rBins,tree=tree,method=sumType,nPairsList=pairCountsZV,volumesList=volumesListZV,errorType=errorType)
		nBarsAH.append(nbarj_AH)
		nBarsZV.append(nbarj_ZV)
		sigmaBarsAH.append(sigma_AH)
		sigmaBarsZV.append(sigma_ZV)
	if showPlot:
		plotStacks(rBins,nBarsAH,nBarsZV,sigmaBarsAH,sigmaBarsZV,sizeBins,binType,nbar,plotAH=plotAH,plotZV=plotZV,yUpper = yUpper,binLabel=binLabel)
	
	return [nBarsAH,nBarsZV,sigmaBarsAH,sigmaBarsZV]


# Pair counts about void centres:


