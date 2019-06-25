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
	return [redshift,rhoB]
	
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
		inBins = np.where( (masses >= bins[i]) & (masses < bins[i+1]))
		binList.append(inBins)
		rhoVav[:,i] = np.mean(rhoVgrid[:,inBins[0]]/rhoB[:,None],axis=1)
		rhoVsd[:,i] = np.sqrt(np.var(rhoVgrid[:,inBins[0]]/rhoB[:,None],axis=1))
	return [rhoVav,rhoVsd,binList,redshift,rhoB]
		

if __name__ == "__main__":
    compute_densities_for_all_snapshots(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5])

# Halves the RGB values of a specified color
def half_color(color):
	if(len(color) != 3):
		raise Exception("Colour must be a three element tuple.")
	return (color[0]/2,color[1]/2,color[2]/2)

# Construct a set of ncolors even spaced colours in rgb space.
def construct_color_list(n,ncolors):
	ncbrt = np.ceil(np.cbrt(ncolors))
	if(n > ncolors):
		raise Exception("Requested colour exceeds maximum specified number of colours. Specify more colours.")
	k = np.floor(n/(ncbrt**2))
	j = np.floor((n - k*ncbrt**2)/ncbrt)
	i = (n - k*ncbrt**2 - j*ncbrt)
	return (i/(ncbrt-1),j/(ncbrt-1),k/(ncbrt-1))

# Returns the specified number in scientific notation:
def scientificNotation(x,latex=False,s=2):
	log10x = np.log10(x)
	z = np.floor(log10x).astype(int)
	y = 10.0**(log10x - z)
	resultString = "10^{" + "{0:0d}".format(z) + "}"
	if np.floor(y) != 1.0:
		resultString = ("{0:." + str(s) + "g}").format(y) + "\\times " + resultString
	if latex:
		resultString = "$" + resultString + "$"
	return resultString

# Plot binned halo densities as a function of redshift
def plot_halo_void_densities(z,rhoVnav,rhoVnsd,rhoVrav,rhoVrsd,bins):
	binNo = len(bins) - 1
	legendList = []
	# Want to format bins in scientific notation:
	
	for k in range(0,binNo):
		plt.semilogy(z,rhoVnav[:,k],color=construct_color_list(k+1,2*binNo))
		plt.fill_between(z,rhoVnav[:,k] - rhoVnsd[:,k],rhoVnav[:,k] + rhoVnsd[:,k],color=half_color(construct_color_list(k+1,2*binNo)))
		legendList.append('Halo Density, $' + scientificNotation(bins[k]) + '$ - $' + scientificNotation(bins[k+1]) + ' M_{sol}/h$')
	for k in range(0,binNo):
		plt.semilogy(z,rhoVrav[:,k],color=construct_color_list(binNo + k+1,2*binNo))
		plt.fill_between(z,rhoVrav[:,k] - rhoVrsd[:,k],rhoVrav[:,k] + rhoVrsd[:,k],color=half_color(construct_color_list(binNo + k+1,2*binNo)))
		legendList.append('Anti-halo Density, $' + scientificNotation(bins[k]) + '$ - $' + scientificNotation(bins[k+1]) + ' M_{sol}/h$')
	plt.xlabel('z')
	plt.ylabel('(Local Density)/(Background Density)')
	plt.legend(legendList)
	plt.show()

#[bin13n,bin14n,bin15n] = pickle.load(open("unreversed/density_plot_outfile_halo_bins.p","rb"))
#[rhoV13n,rhoV14n,rhoV15n] = pickle.load(open("unreversed/density_plot_outfile_rho_bins.p","rb"))
#[zn,rhoBn] = pickle.load(open("unreversed/density_plot_outfile_z_rhoB_data.p","rb"))
#[bin13r,bin14r,bin15r] = pickle.load(open("reversed/density_plot_outfile_halo_bins.p","rb"))
#[rhoV13r,rhoV14r,rhoV15r] = pickle.load(open("reversed/density_plot_outfile_rho_bins.p","rb"))
#[zr,rhoBr] = pickle.load(open("reversed/density_plot_outfile_z_rhoB_data.p","rb"))



