import healpy
import pynbody
import numpy as np
from scipy import io
from void_analysis import snapedit, context, stacking
import pynbody.plot.sph as sph
import matplotlib.pylab as plt
import pickle
import os
import astropy
import scipy.special as special
import scipy
import alphashape
from descartes import PolygonPatch
import matplotlib.cm as cm
import matplotlib.colors as colors

from astropy.coordinates import SkyCoord
import astropy.units as u

# Constants:
c = 299792.458


def getClusterSkyPositions(fileroot="/data2/UCL_data_backup/catalogues/Abell/"):
	# Abell clusters catalogue data:
	file3 = open(fileroot + "VII_110A/table3.dat",'r')
	fileLines3 = []
	for line in file3:
		fileLines3.append(line)
	
	file4 = open(fileroot + "VII_110A/table4.dat",'r')
	fileLines4 = []
	for line in file4:
		fileLines4.append(line)
	
	# Extract sky co-ordinates, abell numbers, and redshift:
	abell_l3 = np.zeros(len(fileLines3))
	abell_b3 = np.zeros(len(fileLines3))
	abell_n3 = np.zeros(len(fileLines3),dtype=np.int)
	abell_z3 = np.zeros(len(fileLines3))
	abell_l4 = np.zeros(len(fileLines4))
	abell_b4 = np.zeros(len(fileLines4))
	abell_n4 = np.zeros(len(fileLines4),dtype=np.int)
	abell_z4 = np.zeros(len(fileLines4))
	for k in range(0,len(fileLines3)):
		abell_n3[k] = np.int(fileLines3[k][0:4])
		abell_l3[k] = np.double(fileLines3[k][118:124])
		abell_b3[k] = np.double(fileLines3[k][125:131])
		abell_z3[k] = np.double(fileLines3[k][133:138].replace(' ','0'))
    
	for k in range(0,len(fileLines4)):
		abell_n4[k] = np.int(fileLines4[k][0:4])
		abell_l4[k] = np.double(fileLines4[k][118:124])
		abell_b4[k] = np.double(fileLines4[k][125:131])
		abell_z4[k] = np.double(fileLines4[k][133:138].replace(' ','0'))
    
	# Indicate missing redshifts:
	abell_z3[np.where(abell_z3 == 0.0)] = -1
	abell_z4[np.where(abell_z4 == 0.0)] = -1
	havez3 = np.where(abell_z3 > 0)
	havez4 = np.where(abell_z4 > 0)
    
	# Combine into a single set of arrays:
	abell_l = np.hstack((abell_l3[havez3],abell_l4[havez4]))
	abell_b = np.hstack((abell_b3[havez3],abell_b4[havez4]))
	abell_n = np.hstack((abell_n3[havez3],abell_n4[havez4]))
	abell_z = np.hstack((abell_z3[havez3],abell_z4[havez4]))
	abell_d = c*abell_z/100 # Distance in Mpc/h
    	
	# Convert to Cartesian coordinates:
	coordAbell = SkyCoord(l=abell_l*u.deg,b=abell_b*u.deg,\
		distance=abell_d*u.Mpc,frame='galactic')
	p_abell = np.zeros((len(coordAbell),3))
	p_abell[:,0] = coordAbell.icrs.cartesian.x.value
	p_abell[:,1] = coordAbell.icrs.cartesian.y.value
	p_abell[:,2] = coordAbell.icrs.cartesian.z.value
    	
	return [abell_l,abell_b,abell_n,abell_z,abell_d,p_abell,coordAbell]


def getAntiHalosInSphere(centres,radius,origin=np.array([0,0,0]),deltaCentral = None,boxsize=None,n_jobs=-1,filterCondition = None):
	if filterCondition is None:
		filterCondition = np.ones(len(centres),dtype=np.bool)
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
	inRadius = tree.query_ball_point(origin,radius,n_jobs=n_jobs)
	
	if len(origin.shape) == 1:
		inRadiusFinal = list(usedIndices[inRadius])
		condition = np.zeros(len(centres),dtype=np.bool)
		condition[inRadiusFinal] = True
	else:
		inRadiusFinal = np.array([list(usedIndices[k]) for k in inRadius])
		condition = np.zeros((len(centres),len(origin)),dtype=np.bool)
		for k in range(0,len(origin)):
			condition[inRadiusFinal[k],k] = True
	return [inRadiusFinal,condition]
	
def getClusterCounterpartPositions(abell_nums,abell_n,p_abell,snap,hncentres,hnmasses,\
		rSearch = 20,mThresh=4e14,boxsize = None):
	superClusterCentres = p_abell[np.isin(\
		abell_n,abell_nums),:]
	if boxsize is None:
		boxsize = snap.properties['boxsize'].ratio("Mpc a h**-1")
	
	[haloCounterpartsLarge,condition] =\
		getAntiHalosInSphere(hncentres,rSearch,origin=superClusterCentres,\
			boxsize=boxsize,filterCondition = (hnmasses > mThresh))
	largeHalosList = np.array([haloCounterpartsLarge[k][0]\
		 for k in range(0,len(haloCounterpartsLarge))])
	sort = np.array([np.where(abell_n[np.isin(abell_n,abell_nums)] == abell_nums[k])[0][0] \
		for k in range(0,len(abell_nums))],dtype=int)
	largeHalos = largeHalosList[sort]
	return largeHalos


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

