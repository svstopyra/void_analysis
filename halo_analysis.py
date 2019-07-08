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
		b = pynbody.bridge.Bridge(s,sj)
		for i in range(0,len(h)):
			halo_centres[i,j,:] = pynbody.analysis.halo.center_of_mass(h[i+1])
		
		del sj
		gc.collect()
	return halo_centres

# Volume of a set of points, computed using the convex hull
def snapVolume(snap,hull=None):
	if hull is None:
		# Compute convex hull if we don't already have it.
		hull = spatial.ConvexHull(snap['pos'])
	# Get and return the volume:
	return hull.volume

# Effective radius of a set of points, defined as the radius of a sphere with volume equivalent to that of the convex hull of the points.
def effectiveRadius(snap,hull=None):
	if hull is None:
		# Compute convex hull if we don't already have it.
		hull = spatial.ConvexHull(snap['pos'])
	volume = snapVolume(snap,hull=hull)
	return np.cbrt(3*volume/(4*np.pi))	

