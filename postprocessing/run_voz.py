import sys
import os
import pynbody
from scipy.io import FortranFile
import numpy as np


# Turn a snapshot into a particle file that we can process with voz1b1:
def generateParticleFile(zobov_pos_file,snap,units = "Mpc a h**-1",dtype=np.double):
	snap['pos'].convert_units(units)
	pos = np.array(snap['pos'],dtype=dtype)
	f = open(zobov_pos_file,"wb")
	np.array(len(snap),dtype=np.int32).tofile(f)
	pos[:,0].tofile(f)
	pos[:,1].tofile(f)
	pos[:,2].tofile(f)
	f.close()

def run_voronoi(mode,simsnap,ndiv,ncores,buf,zobov_path):
	# Create a position file, if one does not already exist:
	zobov_pos_file = simsnap + ".pos"
	snap = pynbody.load(simsnap)
	if not os.path.isfile(zobov_pos_file):
		generateParticleFile(zobov_pos_file,snap)
	# If mode 1, run ZOBOV. Otherwise, just leave it at generating the particle file (in case we want to run with slurm, using different options than the mpirun options used here)
	if mode == 1:
		boxsize = snap.properties['boxsize'].ratio("Mpc a h**-1")
		if ncores < 2:
			# Run vozinit and then run the script for each subvolume 1 by 1:
			# This is only needed if we aren't using mpi, where it creates a script for this whole process.
			os.system(zobov_path + "vozinit " + zobov_pos_file + " " + str(buf) + " " + str(boxsize) + " " + str(ndiv) + " vor " + str(ncores))
			os.system('./scrvor')
		else:
			# Run voz1b1 in mpi mode. No need for vozinit, as this can allocate its own processes.
			os.system("mpirun -np " + str(ncores) + " " + zobov_path + "voz1b1_mpi " + zobov_pos_file + " " + str(buf) + " " + str(boxsize) + " " + " vor " + str(ndiv))
		# Run voztie to combine this information into a single volumes file:
		os.system(zobov_path + "voztie " + str(ndiv) + " vor")
		# Rename so that our processing pipeline can find it:
		os.system("mv vol.vor.dat " + simsnap + ".vols")



if __name__ == "__main__":
	if len(sys.argv) <= 2:
		print("Usage: run_voz.py mode snapshot ndiv ncores buf zobov_path")
		sys.exit()
	mode = int(sys.argv[1])
	simsnap = sys.argv[2]
	if len(sys.argv) <= 3:
		ndiv = 2
	else:
		ndiv = int(sys.argv[3])
	if len(sys.argv) <= 4:
		ncores = 1
	else:
		ncores = int(sys.argv[4])
	if len(sys.argv) <= 5:
		buf = 0.1
	else:
		buf = np.float32(sys.argv[5])
	if len(sys.argv) <= 6:
		zobov_path = ""
	else:
		zobov_path = sys.argv[6]
	run_voronoi(mode,simsnap,ndiv,ncores,buf,zobov_path)
