# This holds code used by plot that doesn't need to integrate with any graphical backends.
# This avoids having to load those backends on systems (such as clusters) that might not support them
import numpy as np
from . import context, snapedit
import healpy
import pynbody
# Returns the centres of the bins, specified from their boundaries. Has one fewer elements.
def binCentres(bins):
	return (bins[1:len(bins)] + bins[0:(len(bins)-1)])/2

# Put the specified values in the specified bins:
# values - values to bin (array)
# bins - boundaries of the bins to use (will be one fewer bins than there are boundaries)
# Returns:
# binList - list of arrays giving the indices of all the elements of values that are in each bin.
# noInBins - number of elements in each bin
def binValues(values,bins):
	binList = []
	noInBins = np.zeros(len(bins)-1,dtype=int)
	for k in range(0,len(bins)-1):
		inThisBin = np.where((values >= bins[k]) & (values < bins[k+1]))[0]
		binList.append(inThisBin)
		noInBins[k] = len(inThisBin)		
	return [binList,noInBins]

def binValues2d(values,bins):
	binList = []
	for k in range(0,len(bins)-1):
		inThisBin = np.where((values >= bins[k]) & (values < bins[k+1]))
		binList.append(inThisBin)		
	return binList

# Figure out if some specified range of z intersects a given z-slice, taking into account any
# wrapping that may apply.
def intersectsSliceWithWrapping(zrange,zslice,thickness,boxsize):
	if(zslice - thickness/2 <= 0):
		wrappedZRange = snapedit.unwrap(zrange,boxsize)
	elif(zslice + thickness/2 >= boxsize):
		wrappedZRange = snapedit.unwrap(zrange,boxsize) + boxsize
	else:
		wrappedZrange = snapedit.wrap(zrange,boxsize)
	condition = (wrappedZRange[:,0] <= zslice + thickness/2) & \
		(wrappedZRange[:,1] >= zslice - thickness/2)
	return condition

# Get points in a specified range, without accounting for wrapping.
def pointsInRangeWithWrap(positions,lim,axis=2,boxsize=None):
	if boxsize is None:
		wrappedPositions = positions
		wrappedLim = np.array(lim)
	else:
		wrappedPositions = snapedit.unwrap(positions,boxsize)
		wrappedLim = snapedit.unwrap(np.array(lim),boxsize)
	return (wrappedPositions[:,axis]>= wrappedLim[0]) & \
		(wrappedPositions[:,axis] <= wrappedLim[1])

# Get the points in a place that lie in the specified rectangle
def pointsInBoundedPlaneWithWrap(positions,xlim,ylim,boxsize=None):
	if boxsize is None:
		wrappedPositions = positions
		wrappedXLim = np.array(xlim)
		wrappedYLim = np.array(ylim)
	else:
		wrappedPositions = snapedit.unwrap(positions,boxsize)
		wrappedXLim = snapedit.unwrap(np.array(xlim),boxsize)
		wrappedYLim = snapedit.unwrap(np.array(ylim),boxsize)
	return (wrappedPositions[:,0] >= wrappedXLim[0]) & \
		(wrappedPositions[:,0] <= wrappedXLim[1]) & \
		(wrappedPositions[:,1] >= wrappedYLim[0]) & \
		(wrappedPositions[:,1] <= wrappedYLim[1])


# Figure out the extent of a set of particles representting an antihalo in one direction.
def getAntihaloExtent(snap,antihalo,centre = None,axis = 2,snapsort = None):
	if snapsort is None:
		snapsort = np.argsort(snap['iord'])
	positions = snap['pos'][snapsort[antihalo['iord']]]
	weights = snap['mass'][snapsort[antihalo['iord']]]
	boxsize = snap.properties['boxsize'].ratio("Mpc a h**-1")
	if centre is None:
		centre = context.computePeriodicCentreWeighted(positions,weights,
			boxsize)*pynbody.units.Unit("Mpc a h**-1")
	positionsRelative = snapedit.unwrap(positions - centre,boxsize)
	return centre[axis] + minmax(positionsRelative[:,axis])

# Code to find particles in an anti-halo that intersect a given slice. 
# Used in constructing projected alpha-shapes.
def getAntiHaloParticlesIntersectingSlice(snap,antihalo,zslice,
	antihaloCentre=None,thickness=15,snapsort = None):
	if snapsort is None:
		snapsort = np.argsort(snap['iord'])
	zminmax = getAntihaloExtent(snap,antihalo,axis=2,snapsort=snapsort,centre=antihaloCentre)
	if (zminmax[1] < zslice - thickness/2) or (zminmax[0] > zslice + thickness/2):
		# No intersection with the slice:
		return np.array([])
	boxsize = snap.properties['boxsize'].ratio("Mpc a h**-1")
	zpos = snapedit.unwrap(snap['pos'][:,2] - zslice,boxsize)
	slicez = np.where((zpos >= - thickness/2) & (zpos <= thickness/2))
	intersecting = np.intersect1d(slicez[0],antihalo['iord'])
	return intersecting

# Format a float converted to a string to have a fixed number of decimal places.
def float_formatter(x,d=2):
	return str(np.around(x,decimals=d))
	
# Convert an array of floats into an array of strings:
def floatsToStrings(floatArray,precision=2):
	return [("%." + str(precision) + "f") % number for number in floatArray]

from healpy import nside2npix, ang2pix, projector
# Healpix map for a spherical slice through a simulation
def sphericalSlice(snap,radius,centre=np.array([0,0,0]),thickness=15,nside=64,fillZeros = 1e-3):
	annulus = pynbody.filt.Annulus(radius-thickness/2,radius+thickness/2,cen=centre)
	posXYZ = snap[annulus]['pos']
	boxsize = snap.properties['boxsize'].ratio("Mpc a h**-1")
	# Wrapped displacements:
	dispXYZ = snapedit.unwrap(posXYZ - centre,boxsize)
	# Angular coordinates
	r = np.sqrt(np.sum(dispXYZ**2,1))
	phi = np.arctan2(dispXYZ[:,1],dispXYZ[:,0])
	theta = np.pi/2 - np.arcsin(dispXYZ[:,2]/r)# Assuming input is a declination.
	# Healpix conversion:
	npix = nside2npix(nside)
	ind = ang2pix(nside,theta,phi)
	# Density:
	hpxMap = np.zeros(npix)
	voxelVolume = (4*np.pi*(radius+thickness/2)**3/3 - \
		4*np.pi*(radius-thickness/2)**3/3)/npix
	#np.add.at(hpxMap,ind,snap[annulus]['rho'])
	np.add.at(hpxMap,ind,snap[annulus]['mass'].in_units("Msol h**-1")/voxelVolume)
	if fillZeros is not None:
		hpxMap[np.where(hpxMap == 0.0)] += fillZeros
	return hpxMap

# Get all points in a spherical shell at a given radius with a given thickness
def filterPolarPointsToAnnulus(lonlat,r,radius,thickness = 15):
	return lonlat[np.where((r >= radius - thickness/2) & (r <= radius + thickness/2))[0],:]

# Compute the XY positions on a Mollwiede plot from the supplied equatorial positions
def computeMollweidePositions(positions,angleUnit="deg",angleCoord="ra_dec",
		centre=np.array([0,0,0]),boxsize=None,h=0.705):
	MW = healpy.projector.MollweideProj()
	if boxsize is not None:
		if centre is None:
			centre = np.array([0,0,0])
		skycoord = context.equatorialXYZToSkyCoord(
			snapedit.unwrap(positions - centre,boxsize),h=h)
	else:
		skycoord = context.equatorialXYZToSkyCoord(positions - centre,h=h)
	angles = np.array([skycoord.icrs.ra.value,skycoord.icrs.dec.value]).T
	if angleUnit == "deg":
		angleFactor = np.pi/180
	elif angleUnit == "rad":
		angleFactor = 1.0
	else:
		raise Exception("Unrecognised angle unit (options = {'deg','rad'}).")
	if angleCoord == "ra_dec":
		if len(positions.shape) == 1:
			posMW = MW.ang2xy(theta = np.pi/2 - angleFactor*angles[1],
				phi=angleFactor*angles[0],lonlat=False)
		else:
			posMW = MW.ang2xy(theta = np.pi/2 - angleFactor*angles[:,1],
				phi=angleFactor*angles[:,0],lonlat=False)
	elif angleCoord == "spherical":
		if len(positions.shape) == 1:
			posMW = MW.ang2xy(theta = angleFactor*angles[1],
				phi=angleFactor*angles[0],lonlat=False)
		else:
			posMW = MW.ang2xy(theta = angleFactor*angles[:,1],
				phi=angleFactor*angles[:,0],lonlat=False)
	return posMW




