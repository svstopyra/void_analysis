#!/usr/bin/python
# Code to analyse anti-halos in a simulation and track their density evolution.

from __future__ import print_function

# Import relevant modules:
import pynbody
import numpy as np
import sys
import scipy.spatial as spatial
import os
import pickle
import multiprocessing as mp
import matplotlib.pylab as plt
import gc
from void_analysis.context import periodicCentre, halo_centres_and_mass, combineHalos, spheresMonteCarlo
from void_analysis.snapedit import wrap, unwrap
from void_analysis.plot import binValues, binValues2d
import scipy.optimize as optimize

"""# Process only a single snapshot
def evaluate(arg1):
	# Import the snapshot supplied as the first argument:
	s = pynbody.load(arg1)

	# Get the relevant halo catalogue:
	print("Retrieving halos...")
	h = s.halos()
	print("complete\n")

	# Check for a stored kd-tree for this snapshop - if it exists, then import it:
	if os.path.isfile('./' + arg1 + '_ckdtree.p'):
		print("Found kd-tree for this snapshot. Importing...")
		tree = pickle.load(open(arg1 + '_ckdtree.p',"rb"))
	else:
		# Create tree from scratch
		print("No kd-tree found. Constructing one from scratch (this may take some time)")
		tree = spatial.cKDTree(s['pos'])
		# Save tree for future use:
		pickle.dump(tree,open(arg1 + '_ckdtree.p',"wb"))

	# Next check for saved halo density:
	if os.path.isfile('./' + arg1 + '_halo_densities.p'):
		print("Found halo densities data. Importing...")
		halo_average_density = pickle.load(open(arg1 + '_halo_densities.p',"rb"))
	else:
		print("Finding nearest neighbours and computing volume weighted densities for all halos:")
		# Initialise array to store average densities:
		halo_average_density = np.zeros(len(h))
		total_particles = 0
		for i in range(1,len(h)):
			total_particle = total_particles + len(h[i])
		
		
		# Compute volume weighted density for all halos:
		done = 0;
		for i in range(1,len(h)):
			halo_average_density[i] = halo_density(h[i],tree,s)
			done = done + len(h[i])
			print("Completed " + str(i) + " of " + str(len(h)) + " halos (" + str(100.0*(done*(1.0/total_particle))) +  "% done).\n")
		
		# Save the data for future use:
		pickle.dump(halo_average_density,open(arg1 + '_halo_densities.p',"wb"))"""
		

# Function to compute volume weighted halo density for halo
def halo_density(near64,mi):	
	# Get nearest neighbours of everything in the halo:
	neighbours = 65	
	# Max of distances to nearest neighbours for each particle in halo:
	hi = np.max(near64[0][:,1:neighbours],1)
	# Masses of the nearest neighbours:
	
	# Average density in the smallest sphere containing the nearest neighbours that is centred on each particle:
	rhoi = np.sum(mi,1)/((4*np.pi*hi**3)/3)
	# Volume-weighted average of these densities for the entire halo:
	densum = np.sum(rhoi*hi**3)
	volsum = np.sum(hi**3)		
	halo_average_density = densum/volsum
	return halo_average_density

# Function to run through all snapshots one by one and obtain the density of particles identified in halos for the last snapshot
def compute_densities_for_all_snapshots(base,snapname,snaps_to_process='all',suffix=''):
	# First find the snapshot files to be read in one by one:
	snapcount = 1
	if os.path.isfile('./' + snapname + "{:0>3d}".format(snapcount) + suffix):
		# Found snapshot files. Start by counting them
		stop = 0
		while(stop == 0):
			if(os.path.isfile('./' + snapname + "{:0>3d}".format(snapcount + 1) + suffix)):
				snapcount = snapcount + 1
			else:
				stop = 1				
		print("Found " + str(snapcount) + " snapshots.")
		if(snaps_to_process != 'all'):
			snapcount = len(snaps_to_process)
		# Get the file that contains the halos we want to track:
		s = pynbody.load(base)
		h = s.halos()
		
		# Create grid to store halo average density data:
		rhoVgrid = np.zeros((snapcount,len(h)))
		# If specified, only compute certain snapshots:
		if snaps_to_process == 'all':
			snaprange = range(1,snapcount+1)
		else:
			snaprange = snaps_to_process
		# One by one, load the snapshots, and compute the densities of these particles:
		if os.path.isfile('./' + snapname + "rhoV_data.p"):
			rhoVgrid = pickle.load(open('./' + snapname + "rhoV_data.p","rb"))
		else:
			counter = 0
			for i in snaprange:
				if(os.path.isfile('./' + snapname + "{:0>3d}".format(i) + "_rhoVi_data.p")):
					rhoVi = pickle.load(open('./' + snapname + "{:0>3d}".format(i) + "_rhoVi_data.p","rb"))
					rhoVgrid[counter,:] = rhoVi
				else:
					print('Processing snapshot ' + str(i))
					rhoVi = compute_halo_densities(snapname + "{:0>3d}".format(i) + suffix,s,h)				
					# Add this row to the combined data:
					rhoVgrid[counter,:] = rhoVi
				counter = counter + 1
					
			# We now have all the data stored, so store it:
			pickle.dump(rhoVgrid,open(snapname + "rhoV_data.p","wb"))						
	else:
		# No snapshots!
		raise Exception("Could not find snapshot files with snapname name: " + snapname)
	return rhoVgrid

def compute_halo_densities(snapname,s,h):
	# Have to compute it. Load the snapshot:
	si = pynbody.load(snapname)
	# Create a bridge to track the particles from the last snapshot:
	b = pynbody.bridge.Bridge(si,s)
	# Have to generate this data from scratch:	
	rhoVi = np.zeros(len(h))	

	# Fastest way to access this is actually to compute local density
	# for all particle (not just the ones in halos), and then pull what we need from there:
	rhos = si['rho']
	hi3s = (2.0*si['smooth'])**3
	for j in range(1,len(h) + 1):
		# Create bridge object (faster than calling b(h[j]) twice)
		bhj = b(h[j])
		# Compute the density of all particles in the halo.
		# This will be slow the first time this is called, as we need to build the kd tree.
		#rhoh = bhj['rho'] 
		# Get the largest distance to a nearest neighbour particle.
		# TODO - Check the factor of two in the code here is correct.
		#hi = 2.0*bhj['smooth']
		# Return the volume weighted average density:
		hi3 = hi3s[bhj['iord']]
		rhoVi[j-1] = np.sum(rhos[bhj['iord']]*(hi3))/np.sum(hi3)
		print("Completed " + str(j) +  " of " + str(len(h)) + " halos.\n")				

	# Save data:
	pickle.dump(rhoVi,open("./" + snapname + "_rhoVi_data.p","wb"))
	# Garbage collection, since sometimes this doesn't happen correctly leading to memory errors:
	del si
	gc.collect()
	return rhoVi

# Function to extract redshift and density information:
def get_z_and_rhoB(snapname,snaps_to_process,suffix=''):
	snapcount = len(snaps_to_process)
	a0 = 1.0
	redshift = np.zeros(snapcount)
	rhoB = np.zeros(snapcount)
	for i in range(0,snapcount):
		si = pynbody.load(snapname + "{:0>3d}".format(snaps_to_process[i]) + suffix)
		redshift[i] = (a0/si.properties['a']) - 1.0
		rhoB[i] = pynbody.analysis.cosmology.rho_M(si,0)
		# Garbage collection:
		del si
		gc.collect()
		print("Extracted data from snap " + str(snaps_to_process[i]))
	return [redshift,rhoB*(1 + redshift)**3]
	
# Function which takes the density data and plots it for different redshifts:
def bin_halo_densities(rhoVgrid,masses,redshift,rhoB,bins=None):
	# Load snapshots (but not data) to get the cosmological properties:
	# Get masses of all the halos and bin them
	# Arbitrary bins:
	if bins is None:
		# Vector should contain the boundaries of the bins in question
		bins = [1e12,1e13,1e14,1e15]
	if len(redshift) != rhoVgrid.shape[0]:
		raise Exception("Supplied redshift list does not match halo densities list.")
	# Average density array:
	rhoVav = np.zeros((len(redshift),len(bins)-1))
	rhoVsd = np.zeros((len(redshift),len(bins)-1))
	binList = []
	for i in range(0,len(bins)-1):
		inBins = np.where( (masses >= bins[i]) & (masses < bins[i+1]))[0]
		binList.append(inBins)
		rhoVav[:,i] = np.mean(rhoVgrid[:,inBins[0]]/rhoB[:,None],axis=1)
		rhoVsd[:,i] = np.sqrt(np.var(rhoVgrid[:,inBins[0]]/rhoB[:,None],axis=1))
	return [rhoVav,rhoVsd,binList,redshift,rhoB]


		

if __name__ == "__main__":
    compute_densities_for_all_snapshots(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5])

# Return the fraction of anti-halos that are below average density, as a function of mass
def getUnderdenseFraction(rhoV,masses,massBins,rhoB,z=1.0):
	logrhoV = np.log(rhoV/rhoB)
	inBins = np.zeros(len(massBins)-1,dtype=int)
	frac = np.zeros(len(massBins)-1)
	sigma = np.zeros(len(massBins)-1)
	for k in range(0,len(massBins)-1):
		inThisBin = np.where((masses >= massBins[k]) & (masses < massBins[k+1]))[0]
		inBins[k] = len(inThisBin)
		voidLike = np.where(logrhoV[inThisBin] < 0.0)[0]
		p = len(voidLike)/inBins[k]
		frac[k] = p
		sigma[k] = z*np.sqrt(p*(1.0 - p)/inBins[k])
	return [frac,sigma,inBins]

# Return the fraction of anti-halos that expanded since the previous time-step
def getExpandingFraction(rhoVfull,redshift,masses,massBins,rhoBfull,z=1.0,mode=1):
	logrhoV = np.log(rhoVfull/rhoBfull[:,None])
	inBins = np.zeros(len(massBins)-1,dtype=int)
	frac = np.zeros(len(massBins)-1)
	sigma = np.zeros(len(massBins)-1)
	noOfZs = logrhoV.shape[0]
	# Get trend of expansion or collapse for all halos:
	expansionDirection = np.polyfit(redshift,logrhoV,1)
	# Only count halos as expanding it their general trend is towards lower density, and the last time step leads to their density decreasing.
	if mode == 1:
		expanding = np.where((expansionDirection[0,:] > 0.0) & (logrhoV[noOfZs-1,:] < logrhoV[noOfZs-2,:]))[0]
	elif mode == 2:
		expanding = np.where(logrhoV[noOfZs-1,:] < logrhoV[noOfZs-2,:])[0]
	else:
		expanding = np.where(logrhoV[noOfZs-1,:] < 0.0)[0]
		
	# Record the halos which are in each mass bin:
	binList = []
	for k in range(0,len(massBins)-1):
		inThisBin = np.where((masses >= massBins[k]) & (masses < massBins[k+1]))[0]
		binList.append(inThisBin)
		inBins[k] = len(inThisBin)
		p = len(np.intersect1d(expanding,inThisBin))/inBins[k]
		frac[k] = p
		sigma[k] = z*np.sqrt(p*(1.0 - p)/inBins[k])
	return [frac,sigma,inBins,expanding,binList]


#[bin13n,bin14n,bin15n] = pickle.load(open("unreversed/density_plot_outfile_halo_bins.p","rb"))
#[rhoV13n,rhoV14n,rhoV15n] = pickle.load(open("unreversed/density_plot_outfile_rho_bins.p","rb"))
#[zn,rhoBn] = pickle.load(open("unreversed/density_plot_outfile_z_rhoB_data.p","rb"))
#[bin13r,bin14r,bin15r] = pickle.load(open("reversed/density_plot_outfile_halo_bins.p","rb"))
#[rhoV13r,rhoV14r,rhoV15r] = pickle.load(open("reversed/density_plot_outfile_rho_bins.p","rb"))
#[zr,rhoBr] = pickle.load(open("reversed/density_plot_outfile_z_rhoB_data.p","rb"))

# Computed the density contrast in a sphere of increasing radius around the centre of a specified point. Radii are assumed to be in units of 'Mpc a h**-1', but should be dimensionless.
def integratedDensityContrast(snap,centre,radii,rhoB = None):
	if rhoB is None:
		redshift = (1.0/snap.properties['a']) - 1.0
		rhoB = pynbody.analysis.cosmology.rho_M(snap,0)*((1 + redshift)**3)
	Delta = np.zeros(len(radii))
	for k in range(0,len(radii)):
		filt = pynbody.filt.Sphere(radii[k],centre)
		radk = radii[k]*pynbody.units.Unit('Mpc a h**-1')
		Delta[k] = (np.sum(snap[filt]['mass'])/((4.0*np.pi/3.0)*radk**3))/rhoB - 1.0
		
# Compute and store the centres of each halo in all the snapshots, so that we can track them:
def haloHistory(base,snapname,snaps_to_process = None,suffix=''):
	# Search for all available snaps if list isn't specified:
	if snaps_to_process is None:
		snapcount = 1
		if os.path.isfile('./' + snapname + "{:0>3d}".format(snapcount) + suffix):
			stop = 0
			while(stop == 0):
				if(os.path.isfile('./' + snapname + "{:0>3d}".format(snapcount + 1) + suffix)):
					snapcount = snapcount + 1
				else:
					stop = 1
			snaps_to_process = range(1,snapcount+1)
		else:
			raise Exception("Could not find snapshot files with snapname name: " + snapname)
	else:
		snapcount = len(snaps_to_process)
	# Get them ony by one:
	s = pynbody.load(base)
	h = s.halos()
	halo_centres = np.zeros((len(h),snapcount,3))
	for j in range(0,snapcount):
		sj = pynbody.load(snapname + "{:0>3d}".format(snaps_to_process[j]) + suffix)
		units = sj['pos'].units
		boxsize = sj.properties['boxsize'].ratio(units)
		b = pynbody.bridge.Bridge(s,sj)
		for i in range(0,len(h)):
			halo_centres[i,j,:] = periodicCentre(h[i+1],boxsize,units=units)
		
		del sj
		gc.collect()
	return halo_centres

# Compute the convex hull of the set of particles, accounting for the periodic boundary conditions by properly unwrapping them relative to the centre of mass.
def getConvexHull(snap):
	units = snap['pos'].units
	boxsize = snap.properties['boxsize'].ratio(units)
	centre = periodicCentre(snap,boxsize,units=units)
	hull = spatial.ConvexHull(unwrap(snap['pos'] - centre,boxsize) + centre)
	return hull
	

# Volume of a set of points, computed using the convex hull
def snapVolume(snap,hull=None):
	if hull is None:
		hull = getConvexHull(snap)
	# Get and return the volume:
	return hull.volume

# Effective radius of a set of points, defined as the radius of a sphere with volume equivalent to that of the convex hull of the points.
def effectiveRadius(snap,hull=None):
	if hull is None:
		hull = getConvexHull(snap)
	volume = snapVolume(snap,hull=hull)
	return np.cbrt(3*volume/(4*np.pi))

# Volume weighted centre of a set of points. Requires the volume weight for each point.
def volumeWeightedCentre(snap,Vi):
	if(len(Vi) == len(snap)):
		# Assume that this is just the volume weights of the snap:
		return np.sum(snap['pos']*Vi[:,None],0)/np.sum(Vi)
	else:
		# Assume we have been given the full list of volumes for all particles, and we have to extract the right volumes:
		return np.sum(snap['pos']*Vi[snap['iord'],None],0)/np.sum(Vi[snap['iord']])

# Compute a void stack, using the anti-halo definition of a void:
def stackAntiHaloVoids(sn,sr,hr,Vi,rbins,ndzbins,Reff=None,vwb = None,densityCutoff=None):
	b = pynbody.bridge.Bridge(sn,sr)
	if Reff is None:
		# Compute effective radii for all anti-halos
		Reff = np.zeros(len(hr))
		for k in range(0,len(hr)):
			Reff[k] = effectiveRadius(b(hr[k+1]))
	if vwb is None:
		# Compute the volume weighted barycentre (vwb):
		vwb = np.zeros((len(hr),3))
		for k in range(0,len(hr)):
			vwb[k,:] = volumeWeightedCentre(b(hr[k+1]),Vi)
	# Figure out which voids belong in each bin:
	[rbinList,noInBins] = binValues(Reff,rbins)
	
	# Cosmological average density (for filtering void candidates):
	redshift = (1.0/sn.properties['a']) - 1.0
	rhoB = pynbody.analysis.cosmology.rho_M(sn,0)*(1 + redshift)**3
	
	# Density for each rbin:
	dzbins = np.zeros((len(rbins)-1,ndzbins))
	ndz2d = np.zeros((len(rbins)-1,ndzbins - 1,ndzbins - 1))
	nr = np.zeros((len(rbins)-1,len(ndzbins) - 1))
	rhodz2d = np.zeros((len(rbins)-1,ndzbins - 1,ndzbins - 1))
	rhor = np.zeros((len(rbins)-1,ndzbins - 1))
	# Boxsize, so we can account for wrapping:
	wrapScale = sn.properties['boxsize'].ratio(sn['pos'].units)
	
	# Compute density profiles:	
	for j in range(0,len(rbins)-1):
		# Fill dzbins up to the relevant 3 times the radius:
		dzbins[j] = np.linspace(0,3.0*rbins[j],ndzbins)
		# Filter for core densities that are too high to count as voids:
		if(densityCutoff is not None):
			tooDense = []
			for k in range(0,len(rbinList[j])):
				radii = np.sqrt(np.sum((b(hr[rbinList[j][k]+1])['pos'] - vwb[rbinList[j][k],:])**2,1))
				inCore = np.where(radii < Reff[rbinList[j][k]]/4)
				rhoCore = np.sum(b(hr[rbinList[j][k]+1])['mass'])/((4*np.pi/3)*(Reff[rbinList[j][k]]/4)**3)
				if(rhoCore > densityCutoff*rhoB):
					tooDense.append(k)
			inRange = np.setdiff(rbinList[j],tooDense)
		else:
			inRange = rbinList[j]
		# Compute the stack density profile:
		[rhodz1,ndz1,rhor1,nr1] = stackAntiHalosInRange(sn,inRange,hr,b,vwb,dzbins[j],wrapScale,rhoB)
		rhodz2d[j,:,:] = rhodz1
		ndz2d[j,:,:] = ndz1
		rhor[j,:] = rhor1
		nr[j,:] = nr1
	return [rhodz2d,ndz2d,rhor,nr,dzbins]
		
		
		
	
# Convert (x,y,z) co-ordinates to line of sight (d = \sqrt{x^2 + y^2},|z|) co-ordinates
def xyz_to_dz(xyz):
	dz = np.zeros((len(xyz),2))
	dz[:,0] = np.sqrt(xyz[:,0]**2 + xyz[:,1]**2)
	dz[:,1] = np.abs(xyz[:,2])
	return dz

# Stack all the anti-halos in the specified range (assumes that the stack has already been filtered for negligible voids and any we wish to skin, etc...)
def stackAntiHalosInRange(sn,inRange,hr,bridge,vwb,dzbins,wrapScale,rhoB):
	nVoids = len(inRange) # No. of voids in stack
	nStack = np.zeros(len(inRange)+1,dtype=int) # Cumulative number of particles surrounding each void
	endRadius = dzbins[len(dzbins)-1]
	ndz = np.zeros((len(dzbins)-1,len(dzbins)-1))
	nr = np.zeros(len(dzbins)-1)
	for k in range(0,nVoids):
		cutout = pynbody.filt.Sphere(endRadius,vwb[k])
		# Get the (d,z) positions of all particles:
		positions = snapedit.unwrap(sn[cutout]['pos'] - vwb[k],wrapScale)
		dzstack = xyz_to_dz(positions)
		ndz = ndz + np.histogram2d(dzstack[:,0],dzstack[:,1],bins=dzbins,density=False)[0]	
		radius = np.sqrt(np.sum(positions**2,1))
		nr = nr + np.histogram(radius,bins=dzbins,density=False)[0]
		print("Done " + str(k+1) + " of " + str(nVoids))
	# Now normalise for volume, using the Jacobian factor in cylindrical co-ordinates:
	dwidth = dzbins[1:len(dzbins)] - dzbins[0:(len(dzbins)-1)]
	d = (dzbins[1:len(dzbins)] + dzbins[0:(len(dzbins)-1)])/2
	zwidth = dzbins[1:len(dzbins)] - dzbins[0:(len(dzbins)-1)]
	rhodz = ndz/d[:,None] # Divide by jacobian factor
	rhodz = rhodz/dwidth[:,None] # Divide by bin width along d direction
	rhodz = rhodz/zwidth[None,:] # Divide by bin width along z direction (gives density/volume)
	rhodz = rhodz/nVoids # Divide by number of voids (gives density/(volume*void))
	rhodz = (rhodz*sn['mass'][0])/(4*np.pi*rhoB) # Divide by angular factors (2*pi), and an additional factor of 2 (accounting for the fact that + and - z are stacked on top of each other), and the cosmological background density to get the density fraction.
	rhor = nr/(4*np.pi*(d**2)) # Divide by shell area
	rhor = rhor/dwidth # Divide by shell thickness to get number density
	rhor = rhor*(sn['mass'][0])/(nVoids*rhoB0) # Divide by number of voids to get 
	return [rhodz,ndz,rhor,nr]


# Ellipticity of a halo stack

def haloStackEllipticity(sn,inRange,endRadius,centre,wrapScale,multiMass=True,tree=None,npar=-1):
	nVoids = len(inRange)
	# Second order moments matrix:
	Iij = np.zeros((3,3))
	# Generate a kd tree if none provided:
	if tree is None:
		tree = spatial.cKDTree(sn['pos'],boxsize=wrapScale)
	for k in range(0,nVoids):
		print("Done " + str(k+1) + " of " + str(nVoids))
		#cutout = pynbody.filt.Sphere(endRadius,centre[k])
		cutout = tree.query_ball_point(centre[k],endRadius,n_jobs=npar)
		positions = unwrap(sn[cutout]['pos'] - centre[k],wrapScale)
		if multiMass:
			Iij = Iij + np.matmul(positions.transpose(),positions*sn[cutout]['mass'][:,None])
		else:
			Iij = Iij + sn['mass'][0]*np.matmul(positions.transpose(),positions)
	return Iij/nVoids
	
		

	
# Compute the distance of each particle in the snapshot to the 64th nearest neighbour.
def neighbourDistance(snap,noNeighbours=64,nblock=1000000,tree=None,returnTree = False):
	N = len(snap)
	# Want to compute the largest distance to one of the 64 nearest neighbours
	hi = np.zeros(N)
	# Have to perform this calculation in blocks, otherwise we risk running our of memory.
	blocks = np.floor(N/nblock).astype(int)
	if tree is None:
		# Generate a kd tree if none exists already
		tree = spatial.cKDTree(snap['pos'])
	for i in range(0,blocks):
		nearest = tree.query(snap['pos'][(i*nblock):((i+1)*nblock)],k=(noNeighbours+1))[0]
		hi[(i*nblock):((i+1)*nblock)] = nearest[:,noNeighbours]
	if N > blocks*nblock:
		nearest = tree.query(snap['pos'][(blocks*nblock):N],k=(noNeighbours+1))[0]
		hi[(blocks*nblock):N] = nearest[:,noNeighbours]
	if returnTree:
		return [hi,tree]
	else:
		return hi

# Compute the volume weighted underdense fraction in each bin:
def volumeWeightedUnderdenseFraction(sn,sr,antiHalos,antiHaloMasses,massBins,rhoB,volumeWeight=None,density=None,return_underdense=False):
	b = pynbody.bridge.Bridge(sn,sr)
	[binList,noInBins] = binValues(antiHaloMasses,massBins)
	combinedVoids = []
	if return_underdense:
		underdense = []
	vwf = np.zeros(len(binList))
	if volumeWeight is None:
		volumeWeight = sn['smooth']**3
	if density is None:
		density = sn['rho']
	for k in range(0,len(binList)):
		combinedVoids.append(combineHalos(sn,antiHalos,binList[k]))
		underdense_frac = np.where(density[combinedVoids[k]['iord']] < rhoB)[0]
		if return_underdense:
			underdense.append(underdense_frac)
		vwf[k] = np.sum(volumeWeight[combinedVoids[k]['iord']][underdense_frac])/np.sum(volumeWeight[combinedVoids[k]['iord']])
	if return_underdense:
		return [vwf,underdense_frac,combinedVoids]
	else:
		return vwf

# Volume averaged density of some subset of particles. Requires the volumeWeights and density of the whole simulation snap to have been computed.
def volumeWeightedDensity(subSnap,volumeWeight,density):
	return np.sum(density[subSnap['iord']]*volumeWeight[subSnap['iord']])/np.sum(volumeWeight[subSnap['iord']])

# Compute the volume-weighted averaged density in the supplied mass bins, by summing over all particles in the bin, rather than individual halos and then averaging them.
def volumeWeightedDensityByBins(sn,antiHalos,antiHaloMasses,massBins,volumeWeight=None,density=None):
	[binList,noInBins] = binValues(antiHaloMasses,massBins)
	vad = np.zeros(len(binList))
	if volumeWeight is None:
		volumeWeight = sn['smooth']**3
	if density is None:
		density = sn['rho']
	for k in range(0,len(binList)):
		combinedVoids = combineHalos(sn,antiHalos,binList[k])
		vad[k] = volumeWeightedDensity(combinedVoids,volumeWeight,density)
	return vad
		
	
		
	

# Class to store analysis data concerning a given snapshot of a simulation
class SnapAnalysis:
	def __init__(self,snapname,reverse_snapname,recompute=False,getHalos=False,getDensity=False,saveTree=False,loadStacking=False):
		self.snap = pynbody.load(snapname)
		self.rev = pynbody.load(reverse_snapname)
		self.snapname = snapname
		self.reversename = reverse_snapname
		self.b = pynbody.bridge.Bridge(self.snap,self.rev)
		if getHalos:
			self.halos = self.snap.halos()
			self.antihalos = self.rev.halos()
			if os.path.isfile('./' + snapname + "_halo_additional_properties.p") and (not recompute):
				[self.halo_centres,self.halo_masses] = pickle.load(open('./' + snapname + "_halo_additional_properties.p","rb"))
			else:
				print("Computing halo centres and mass list")
				[self.halo_centres,self.halo_masses] = halo_centres_and_mass(self.halos,'./' + snapname + "_halo_additional_properties.p")
			if os.path.isfile('./' + reverse_snapname + "_halo_additional_properties.p") and (not recompute):
				[self.antihalo_centres,self.antihalo_masses] = pickle.load(open('./' + reverse_snapname + "_halo_additional_properties.p","rb"))
			else:
				print("Computing halo centres and mass list")
				[self.antihalo_centres,self.antihalo_masses] = halo_centres_and_mass(self.antihalos,'./' + reverse_snapname + "_halo_additional_properties.p")
		if getDensity:
			# Density information:
			if(os.path.isfile('./' + snapname + "_rhoVi_data.p") and (not recompute) and getHalos):
				self.rhoVi = pickle.load(open('./' + snapname + "_rhoVi_data.p","rb"))
			else:
				print("Computing halo density data...")
				self.rhoVi = self.computeDensityData()
			# Local volume information:
			if(os.path.isfile('./' + snapname + "_volumes.p") and (not recompute)):
				self.hi = pickle.load(open('./' + snapname + "_volumes.p","rb"))
			else:
				self.hi = self.nnDistance(saveTree)
				pickle.dump(self.hi,open('./' + snapname + "_volumes.p","wb"))
		if loadStacking:
			if(os.path.isfile('./' + snapname + "_stack_data.p") and (not recompute) and getHalos):
				[self.Reff,self.vwb] = pickle.load(open('./' + snapname + "_stack_data.p","rb"))
			else:
				print("Computing stacking statistics...")
				self.Reff = np.zeros(len(self.antihalos))
				self.vwb = np.zeros((len(self.antihalos),3))
				for k in range(0,len(self.antihalos)):
					self.Reff[k] = effectiveRadius(self.b(self.antihalos[k+1]))
					self.vwb[k,:] = volumeWeightedCentre(self.b(self.antihalos[k+1]),self.hi**3)
	# Compute local density for all the halos in the snapshot:
	def computeDensityData(self):
		if(not hasattr(self,'halos')):
			self.halos = self.snap.halos()
		return compute_halo_densities(self.snapname,self.snap,self.halos)
	# Compute the nearest neighbour distance for each particle:
	def nnDistance(self,saveKDTree):
		print("Computing volumes for snapshot...")
		if(os.path.isfile('./' + self.snapname + "_kdtree.p")):
			tree = pickle.load(open('./' + self.snapname + "_kdtree.p","rb"))		
		else:
			# Build the tree from scratch:
			print("Building kd-tree for snapshot.")
			tree = spatial.cKDTree(self.snap['pos'])
			if(saveKDTree):
				pickle.dump(tree,open('./' + self.snapname + "_kdtree.p","wb"))
		print("Computing max distance to nearest neighbours")
		hi = neighbourDistance(self.snap,tree)
		return hi

# Void volumes from monte-carlo:
def volumesFromMonteCarlo(sn,hr,void_centres,b,radiusList):	
	volList = np.zeros(len(hr))
	wrapLength = sn.properties['boxsize'].ratio(sn['pos'][0].units)
	for k in range(0,len(hr)):
		centresUnshifted = unwrap(b(hr[k+1])['pos'] - void_centres[k],wrapLength)
		centres = centresUnshifted + np.abs(np.min(centresUnshifted))
		boxLength = np.max(centres)
		radii = radiusList[b(hr[k+1])['iord']]
		# Only need percent level accuracy, really:
		volList[k] = spheresMonteCarlo(centres,radii,[boxLength,boxLength,boxLength],tol=1e-2)
		print("Done " + str(k+1) + " of " + str(len(hr)) + " anti-halos.")
	return volList
		
# Ellipse as a function of theta, defined by y^2/a^2 + x^2/b^2 = 1
def ellipseR(theta,a,b):
	return a/np.sqrt(1 + ((a/b)**2-1)*np.cos(theta)**2)

# Fit elliptical model to different density bins:
def ellipseFit(rhodz,X,Y,denBins,returnCovariance=False):
	denBinsList = binValues2d(rhodz,denBins)
	param = np.zeros((len(denBins)-1,2))
	paramError = np.zeros((len(denBins)-1,2))
	if returnCovariance:
		cov = []
	for k in range(0,len(denBins)-1):
		thetaToFit = np.arctan(Y[denBinsList[k]]/X[denBinsList[k]])
		RToFit = np.sqrt(X[denBinsList[k]]**2 + Y[denBinsList[k]]**2)
		fit = optimize.curve_fit(ellipseR,thetaToFit,RToFit)
		param[k,:] = fit[0]
		paramError[k,:] = np.sqrt(np.diag(fit[1]))
		if returnCovariance:
			cov.append(fit[1])
	if returnCovariance:
		return [param,paramError,cov]
	else:
		return [param,paramError]

# Compute eccentricity (or ellipticity) from a and b parameters of ellipse:
def eccentricity(ab):
	a = ab[:,0]
	b = ab[:,1]
	aIsSemiMajor = np.where(a > b)[0]
	bIsSemiMajor = np.setdiff1d(range(0,len(a)),aIsSemiMajor)
	ecc = np.zeros(len(a))
	ecc[aIsSemiMajor] = np.sqrt(1 - b[aIsSemiMajor]**2/a[aIsSemiMajor]**2)
	ecc[bIsSemiMajor] = np.sqrt(1 - a[bIsSemiMajor]**2/b[bIsSemiMajor]**2)
	return ecc

# Get the radius from the binned density profile:
def radiusFromDensity(r,rhor,thresh = 0.5):
	radii = np.zeros(len(rhor))
	for k in range(0,len(radii)):
		print(str(k))
		f = lambda x: np.interp(x,r,rhor[k]) - thresh
		if f(r[0])*f(r[-1]) > 0:
			print("Warning: could not find Reff for k = " + str(k))
		else:
			radii[k] = optimize.brentq(f,r[0],r[-1])
	return radii
			
