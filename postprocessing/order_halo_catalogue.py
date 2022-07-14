import sys
import os
import pynbody
import numpy as np


def orderHalos(fname):
	snap = pynbody.load(fname)
	halos = snap.halos()
	# Catalogue file names:
	appendFname = "_ordered"
	halosFname = fname + ".z0.000.AHF_halos"
	profilesFname = fname + ".z0.000.AHF_profiles"
	particlesFname = fname + ".z0.000.AHF_particles"
	substructureFname = fname + ".z0.000.AHF_substructure"
	# Import the new halo catalogue and re-order it:
	haloData = np.loadtxt(halosFname)
	ids = np.loadtxt(halosFname,dtype=np.int,usecols = (0))
	masses = haloData[:,3]
	sort = np.argsort(masses)[::-1]
	f_halos = open(halosFname,"r")
	f_halos_lines = []
	haveFirst = False
	for line in f_halos:
		if line[0] != '#':
			f_halos_lines.append(line)
		else:
			if not haveFirst:
				firstLine = line
	# output the new halos file:
	f_halos_new = open(halosFname + appendFname,"w")
	f_halos_new.write(firstLine)
	for k in range(0,len(halos)):
		# Replace ID field with the new ID:
		#newString = f_halos_lines[sort[k]].replace(f_halos_lines[sort[k]].split()[0],str(k))
		splitString = f_halos_lines[sort[k]].split()
		splitString[0] = str(k)
		# Replace host halo with correct host id, or -1 (if recorded as 0):
		hostID = np.int(splitString[1])
		if hostID != 0:
			newIDList = np.where(np.isin(ids[sort],hostID))[0]
			if len(newIDList) < 1:
				if hostID > 0:
					print("Warning: cannot find host with ID" + str(hostID) +\
						 " for halo with ID " + \
						 f_halos_lines[sort[k]].split()[0])
				splitString[1] = str(-1)
			else:
				splitString[1] = str(newIDList[0])
		else:
			splitString[1] = str(-1)
		f_halos_new.write("\t".join(splitString) + "\n")
	f_halos.close()
	print("Completed halo properties file.\n")
	# Particle file is mostly unchanged, but we need to update the IDs and re-order things:
	f_part = open(particlesFname,"r")
	f_part_lines = []
	for line in f_part:
		f_part_lines.append(line)
	counter = 0
	halo_counter = 0
	# Get arrays of particles for each halo:
	nparts = np.zeros(len(halos),dtype=np.int)
	partStarts = np.zeros(len(halos),dtype=np.int)
	while counter < len(f_part_lines):
		# Skip lines with only a single entry, as these denote the boundaries of individual cpu files.
		lineAsArray = np.array(f_part_lines[counter].split(),dtype=np.int)
		if len(lineAsArray) == 1:
			counter += 1
			continue
		nparts[halo_counter] = lineAsArray[0]
		counter +=1
		partStarts[halo_counter] = counter
		counter += nparts[halo_counter]
		halo_counter += 1
	# Output particle data, in order:		
	f_particles_new = open(particlesFname + appendFname,"w")
	f_particles_new.write("\t" + str(len(halos)) + "\n")
	for k in range(0,len(sort)):
		f_particles_new.write(str(len(halos[sort[k]+1])) + "\t" + str(k) + "\n")
		linesToSave = f_part_lines[partStarts[sort[k]]:(partStarts[sort[k]] + nparts[sort[k]])]
		for line in linesToSave:
			f_particles_new.write(line)
		#print("Done halo " + str(k+1) + " of " + str(len(halos)))
	print("Completed particle file\n")
	# Re-order the profiles file:
	f_profiles = open(profilesFname,"r")
	f_profiles_new = open(profilesFname + appendFname,"w")
	f_profiles_lines = []
	for line in f_profiles:
		f_profiles_lines.append(line)
	linestarts = np.ones(len(halos),dtype=np.int)
	bins = np.array(haloData[:,36],dtype=np.int)
	linestarts[1:] += np.cumsum(bins)[0:-1]
	f_profiles_new.write(f_profiles_lines[0])
	for k in range(0,len(halos)):
		linesToSave = f_profiles_lines[linestarts[sort[k]]:(linestarts[sort[k]] + bins[sort[k]])]
		for line in linesToSave:
			f_profiles_new.write(line)
	f_profiles.close()
	print("Completed profiles file.\n")
	# Re-order the substructure file:
	f_sub = open(substructureFname,"r")
	f_sub_new = open(substructureFname + appendFname,"w")
	f_sub_lines = []
	for line in f_sub:
		f_sub_lines.append(np.array(line.split(),dtype=np.int))
	numHalosWithSubstructure = np.int(len(f_sub_lines)/2)	
	for k in range(0,numHalosWithSubstructure):
		index = np.where(np.isin(ids[sort],f_sub_lines[2*k][0]))[0]
		if len(index) < 1:
			print("Warning, line " + str(2*k + 1) + " of substructure file refers to non-existent halo (possibly removed due to mpi halo-finding procedure).")
			continue
		f_sub_new.write(str(np.where(np.isin(ids[sort],f_sub_lines[2*k][0]))[0][0]) + "\t" + str(f_sub_lines[2*k][1]) + "\n")
		f_sub_new.write("\t" + "\t".join([str(int) for int in np.where(np.isin(ids[sort],f_sub_lines[2*k+1]))[0]]) + "\n")
	f_sub.close()
	print("Completed substructure file.\n")
	# Close all output files:
	f_halos_new.close()
	f_sub_new.close()
	f_profiles_new.close()
	f_particles_new.close()
	exit(0)



# Main function, accepts the name of the snapshot to be processed as an argument:
if __name__ == "__main__":
	if len(sys.argv) < 2:
		print("Usage: order_halo_catalogue.py snapshot_filename")
		exit(1)
	fname = sys.argv[1]
	orderHalos(fname)
