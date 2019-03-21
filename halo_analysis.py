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
import pymp

# Process only a single snapshot
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
		pickle.dump(halo_average_density,open(arg1 + '_halo_densities.p',"wb"))
		

# Function to compute volume weighted halo density for halo
def halo_density(halo,tree,s):	
	# Get nearest neighbhours of everything in the halo:
	neighbours = np.min([65,len(halo)])
	near64 = tree.query(halo['pos'],neighbours)
	# Max of distances to nearest neighbours for each particle in halo:
	hi = np.max(near64[0][:,1:neighbours],1)
	# Masses of the nearest neighbours:
	mi = s['mass'][near64[1][:,1:neighbours]]
	# Average density in the smallest sphere containing the nearest neighbours that is centred on each particle:
	rhoi = np.sum(mi,1)/((4*np.pi*hi**3)/3)
	# Volume-weighted average of these densities for the entire halo:
	densum = np.sum(rhoi*hi**3)
	volsum = np.sum(hi**3)		
	halo_average_density = densum/volsum
	return halo_average_density

# Function to run through all snapshots one by one and obtain the density of particles identified in halos for the last snapshot
def compute_densities_for_all_snapshots(base,snaps_to_process='none',suffix=''):
	# First find the snapshot files to be read in one by one:
	snapcount = 1
	if os.path.isfile('./' + base + "{:0>3d}".format(snapcount) + suffix):
		# Found snapshot files. Start by counting them
		stop = 0
		while(stop == 0):
			if(os.path.isfile('./' + base + "{:0>3d}".format(snapcount + 1) + suffix)):
				snapcount = snapcount + 1
			else:
				stop = 1				
		print(snapcount)
		# Assume that the last file is the one which defines the halos we will trace back.
		s = pynbody.load(base + "{:0>3d}".format(snapcount) + suffix)
		h = s.halos()
		total_particles = 0
		for i in range(1,len(h)):
			total_particles = total_particles + len(h[i])
		# Create grid to store halo average density data:
		rhoVgrid = np.zeros((snapcount,len(h)))
		# If specified, only compute certain snapshots:
		if snaps_to_process == 'none':
			snaprange = np.flip(range(1,snapcount+1))
		else:
			snaprange = snaps_to_process
		# One by one, load the snapshots, and compute the densities of these particles:
		for i in snaprange:
			print('Processing snapshot ' + str(i))
			# Just load the data if it already exists:
			if os.path.isfile('./' + base + "{:0>3d}".format(i) + "_halo_densities.p"):
				rhoVi = pickle.load(open('./' + base + "{:0>3d}".format(i) + "_halo_densities.p","rb"))
			else:
				# Have to compute it. Load the snapshot:
				si = pynbody.load(base + "{:0>3d}".format(i))
				# Create a bridge to track the particles from the last snapshot:
				b = pynbody.bridge.OrderBridge(si,s)
				if os.path.isfile('./' + base + "{:0>3d}".format(i) + '_halo_densities.p'):
					print("Found halo densities data. Importing...")
					rhoVi = pickle.load(open(base + "{:0>3d}".format(i) + '_halo_densities.p',"rb"))
				else:
					# Check for a stored kd-tree for this snapshop - if it exists, then import it:
					if os.path.isfile('./' + base + "{:0>3d}".format(i) + '_ckdtree.p'):
						print("Found kd-tree for this snapshot. Importing...")
						tree = pickle.load(open(base + "{:0>3d}".format(i) + '_ckdtree.p',"rb"))
					else:
						# Create tree from scratch
						print("No kd-tree found. Constructing one from scratch (this may take some time)")
						tree = spatial.cKDTree(si['pos'])
						# Save tree for future use (actually, skip this for now as it will be a large file...)
						# pickle.dump(tree,open(base + "{:0>3d}".format(i) + '_ckdtree.p',"wb"))
						print("Finding nearest neighbours and computing volume weighted densities for all halos:")
					# Initialise array to store average densities:
					rhoVi = pymp.shared.array(len(h))	
					
					# Compute volume weighted density for all halos:
					done = 0
					with pymp.Parallel(6) as p:
						for i in p.range(1,len(h)):
							rhoVi[i] = halo_density(b(h[i]),tree,si)
							done = done + len(h[i])
							p.print("Completed " + str(i) + " of " + str(len(h)) + " halos (" + str(100.0*(done*(1.0/total_particles))) +  "% done).\n")
					
					# Save the data for future use:
					pickle.dump(rhoVi,open(base + "{:0>3d}".format(i) + '_halo_densities.p',"wb"))
			# Add this row to the combined data:
			rhoVgrid[i - 1,:] = rhoVi
		# We now have all the data stored, so store it:
		pickle.dump(rhoVgrid,open(base + "rhoV_data.p","wb"))						
	else:
		# No snapshots!
		raise Exception("Could not find snapshot files with base name: " + base)
	return rhoVgrid

	
"""# Function which takes the density data and plots it for different redshifts:
def halo_density_plot(rhoVgrid,base,suffix=''):
	# Load snapshots (but not data) to get the cosmological properties:
	s = []
	snapcount = np.shape(rhoVgrid)[0]
	for i in range(0,snapcount):
		s.append(pynbody.load(base + "{:0>3d}".format(i+1) + suffix))
	# Redshifts and background densities:
	GN = 6.67e-11# Newton's constant
	H0factor = 1e5/(1e6*3.0857e16) # 100km/s/Mpc
	a0 = s[snapcount-1].properties['a']
	# Cosmological average density:
	rhoB0 = s[snapcount-1].properties['OmegaM0']*3*(s[snapcount-1].properties['h']*H0factor)**2/(8*np.pi*GN)
	redshift = np.zeros(snapcount)
	for i in range(0,snapcount)
		redshift[i] = (a0/s[i].properties['a']) - 1.0
	rhoB = rhoB0/(1.0 + redshift) # Density at earlier redshifts:"""
	
	
	
	

if __name__ == "__main__":
    evaluate(sys.argv[1])




