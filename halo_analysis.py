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
			snaprange = np.flip(range(1,snapcount+1))
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
					rhoVi = compute_halo_densities(snapname + "{:0>3d}".format(i),s,h)				
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
	return rhoVi
	
# Function which takes the density data and plots it for different redshifts:
def halo_density_plot(rhoVgrid,base,snapname,redshift='none',rhoB='none',bin13='none',bin14 = 'none',bin15='none',snaps_to_process='all',suffix=''):
	# Load snapshots (but not data) to get the cosmological properties:
	s = pynbody.load(base)
	h = s.halos()
	si = []
	snapcount = np.shape(rhoVgrid)[0]
	if(snaps_to_process == 'all'):
		snaps_to_process = range(1,snapcount + 1)
	else:
		snapcount = len(snaps_to_process)
		
		
	# Extract redshift and background densities if not provided:
	
	if(redshift == 'none' or rhoB == 'none'):
		a0 = s[snapcount-1].properties['a']
		redshift = np.zeros(snapcount)
		rhoB = np.zeros(snapcount)
		for i in range(0,snapcount):
			si = pynbody.load(snapname + "{:0>3d}".format(i+1) + suffix)
			redshift[i] = (a0/si.properties['a']) - 1.0
			rhoB[i] = pynbody.analysis.cosmology.rho_M(si,0)
			print("Extracted data from " + str(i+1) + " of " + str(snapcount+1) + " snapshots.")
	
	# Get masses of all the halos and bin them
	# TODO - re-write this so that we can specify arbitrary bins:
	compute = np.array([0,0,0])
	bin_lower = np.array([1e14,1e13,1e12])
	bin_list = [[],[],[]]
	if(bin15 == 'none'):
		compute[0] = 1
		bin15 = []
	if(bin14 == 'none'):
		compute[1] = 1
		bin14 = []
	if(bin13 == 'none'):
		compute[2] = 1
		bin13 = []
	mass = np.zeros(len(h))
	if(np.any(compute)):
		for i in range(0,len(h)):
			massi = np.sum(h[i]['mass'])
			if (massi.in_units('Msol h**-1') > 1e14 and compute[0] == 1):
				bin15.append(i)
			elif (massi.in_units('Msol h**-1') > 1e13 and compute[1] == 1) :
				bin14.append(i)
			elif (massi.in_units('Msol h**-1') > 1e12 and compute[2] == 1):
				bin13.append(i)
			print("Binned " + str(i+1) + " of " + str(len(h) + 1) + " halos.")
	# Get average density in each bin:
	rhoV13 = np.sum(rhoVgrid[:,bin13]/rhoB[:,None],1)/len(bin13)
	rhoV14 = np.sum(rhoVgrid[:,bin14]/rhoB[:,None],1)/len(bin14)
	rhoV15 = np.sum(rhoVgrid[:,bin15]/rhoB[:,None],1)/len(bin15)
	
	# Plot the results:
	plt.semilogy(redshift,rhoV13,redshift,rhoV14,redshift,rhoV15)
	plt.show()
	
	
	

if __name__ == "__main__":
    compute_densities_for_all_snapshots(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5])




